"""Unit tests for the SpatialIndex class.

This module tests the optimized KD-tree spatial query system including:
- Position change tracking with dirty flags
- Hash-based change detection
- Count-based optimization
- Spatial query methods
- Performance optimizations
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from farm.core.resources import Resource
from farm.core.spatial_index import SpatialIndex


# Create a mock BaseAgent class
class MockBaseAgent:
    def __init__(self, agent_id, position, resource_level, environment, generation=0):
        self.agent_id = agent_id
        self.position = list(position)  # Convert to list for mutable assignment
        self.resource_level = resource_level
        self.environment = environment
        self.generation = generation
        self.alive = True
        self.current_health = 100.0
        self.starting_health = 100.0
        self.starvation_counter = 0
        self.starvation_threshold = 100.0
        self.total_reward = 0.0
        self.birth_time = 0
        self.is_defending = False
        self.genome_id = f"genome_{agent_id}"

    def get_action_weights(self):
        return {
            "move": 0.3,
            "gather": 0.3,
            "share": 0.15,
            "attack": 0.1,
            "reproduce": 0.15,
        }


class TestSpatialIndex(unittest.TestCase):
    """Test cases for SpatialIndex functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create spatial index
        self.spatial_index = SpatialIndex(width=100, height=100)

        # Create mock agents and resources
        self.agents = []
        self.resources = []

        # Add some test resources
        for i in range(5):
            resource = Resource(
                resource_id=i,
                position=(i * 20, i * 20),
                amount=10,
                max_amount=20,
                regeneration_rate=0.1,
            )
            self.resources.append(resource)

        # Set references but don't call update() to keep initial state
        self.spatial_index.set_references(self.agents, self.resources)

    def test_initialization(self):
        """Test that SpatialIndex is properly initialized."""
        # Check that KD-tree attributes exist
        self.assertIsNone(self.spatial_index.agent_kdtree)
        self.assertIsNone(self.spatial_index.resource_kdtree)  # Not built initially
        self.assertIsNone(self.spatial_index.agent_positions)
        self.assertIsNone(self.spatial_index.resource_positions)  # Not built initially

        # Check position tracking attributes (initial state)
        self.assertTrue(self.spatial_index._positions_dirty)
        self.assertIsNone(self.spatial_index._cached_counts)
        self.assertIsNone(self.spatial_index._cached_hash)

    def test_mark_positions_dirty(self):
        """Test that marking positions dirty works correctly."""
        # Initially dirty from setup
        self.assertTrue(self.spatial_index._positions_dirty)

        # Clear dirty flag
        self.spatial_index._positions_dirty = False
        self.assertFalse(self.spatial_index._positions_dirty)

        # Mark as dirty
        self.spatial_index.mark_positions_dirty()
        self.assertTrue(self.spatial_index._positions_dirty)

    def test_counts_changed_no_agents(self):
        """Test count change detection with no agents."""
        # Initial state
        self.assertIsNone(self.spatial_index._cached_counts)

        # First call should return True and set cache
        self.assertTrue(self.spatial_index._counts_changed(0))
        self.assertEqual(self.spatial_index._cached_counts, (0, 5))  # 0 agents, 5 resources

        # Second call with same counts should return False
        self.assertFalse(self.spatial_index._counts_changed(0))

    def test_counts_changed_with_agents(self):
        """Test count change detection with agents."""
        # Add an agent
        agent = MockBaseAgent(
            agent_id="test_agent",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        self.agents.append(agent)

        # Force rebuild to update counts
        self.spatial_index._rebuild_kdtrees()

        # Check that counts are updated
        self.assertTrue(self.spatial_index._counts_changed(1))
        self.assertEqual(self.spatial_index._cached_counts, (1, 5))  # 1 agent, 5 resources

    def test_counts_changed_agent_death(self):
        """Test count change detection when agent dies."""
        # Add an agent
        agent = MockBaseAgent(
            agent_id="test_agent",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        self.agents.append(agent)

        # Force rebuild to update counts
        self.spatial_index._rebuild_kdtrees()
        self.spatial_index._counts_changed(1)  # Cache the counts

        # Kill the agent
        agent.alive = False

        # Check that counts changed
        self.assertTrue(self.spatial_index._counts_changed(0))
        self.assertEqual(self.spatial_index._cached_counts, (0, 5))  # 0 alive agents, 5 resources

    def test_hash_positions_changed_no_positions(self):
        """Test hash change detection with no positions."""
        # Initial state
        self.assertIsNone(self.spatial_index._cached_hash)

        # First call should return True and set cache
        self.assertTrue(self.spatial_index._hash_positions_changed([]))
        self.assertIsNotNone(self.spatial_index._cached_hash)

        # Second call with same positions should return False
        self.assertFalse(self.spatial_index._hash_positions_changed([]))

    def test_hash_positions_changed_with_agents(self):
        """Test hash change detection with agents."""
        # Add an agent
        agent = MockBaseAgent(
            agent_id="test_agent",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        self.agents.append(agent)

        # Force rebuild to update positions
        self.spatial_index._rebuild_kdtrees()

        # Check that hash changed
        self.assertTrue(self.spatial_index._hash_positions_changed([agent]))
        self.assertIsNotNone(self.spatial_index._cached_hash)

    def test_hash_positions_changed_position_update(self):
        """Test hash change detection when agent position changes."""
        # Add an agent
        agent = MockBaseAgent(
            agent_id="test_agent",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        self.agents.append(agent)

        # Force rebuild to update positions
        self.spatial_index._rebuild_kdtrees()
        self.spatial_index._hash_positions_changed([agent])  # Cache the hash

        # Change agent position
        agent.position[0] = 20
        agent.position[1] = 20

        # Check that hash changed
        self.assertTrue(self.spatial_index._hash_positions_changed([agent]))

    def test_rebuild_kdtrees_no_agents(self):
        """Test KD-tree rebuilding with no agents."""
        # Initial state - no KD-trees built yet
        self.assertIsNone(self.spatial_index.agent_kdtree)
        self.assertIsNone(self.spatial_index.resource_kdtree)

        # Rebuild
        self.spatial_index._rebuild_kdtrees()

        # Should still be None for agents, but resource tree should exist
        self.assertIsNone(self.spatial_index.agent_kdtree)
        self.assertIsNotNone(self.spatial_index.resource_kdtree)

    def test_rebuild_kdtrees_with_agents(self):
        """Test KD-tree rebuilding with agents."""
        # Add agents
        agent1 = MockBaseAgent(
            agent_id="agent1",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agent2 = MockBaseAgent(
            agent_id="agent2",
            position=(20, 20),
            resource_level=30,
            environment=None,
            generation=0,
        )
        self.agents.append(agent1)
        self.agents.append(agent2)

        # Rebuild
        self.spatial_index._rebuild_kdtrees()

        # Check that agent KD-tree was built
        self.assertIsNotNone(self.spatial_index.agent_kdtree)
        self.assertIsNotNone(self.spatial_index.agent_positions)
        if self.spatial_index.agent_positions is not None:
            self.assertEqual(len(self.spatial_index.agent_positions), 2)

    def test_rebuild_kdtrees_with_dead_agents(self):
        """Test KD-tree rebuilding excludes dead agents."""
        # Add agents
        agent1 = MockBaseAgent(
            agent_id="agent1",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agent2 = MockBaseAgent(
            agent_id="agent2",
            position=(20, 20),
            resource_level=30,
            environment=None,
            generation=0,
        )
        self.agents.append(agent1)
        self.agents.append(agent2)

        # Kill one agent
        agent2.alive = False

        # Rebuild
        self.spatial_index._rebuild_kdtrees()

        # Check that only alive agent is included
        self.assertIsNotNone(self.spatial_index.agent_kdtree)
        if self.spatial_index.agent_positions is not None:
            self.assertEqual(len(self.spatial_index.agent_positions), 1)
            self.assertEqual(self.spatial_index.agent_positions[0][0], 10)  # agent1's x position

    def test_update_no_changes(self):
        """Test update when no changes occur."""
        # Clear dirty flag
        self.spatial_index._positions_dirty = False

        # Mock the change detection methods
        with patch.object(self.spatial_index, "_counts_changed", return_value=False) as mock_counts, patch.object(
            self.spatial_index, "_hash_positions_changed", return_value=False
        ) as mock_hash, patch.object(self.spatial_index, "_rebuild_kdtrees") as mock_rebuild:
            self.spatial_index.update()

            # Should not call any methods when not dirty
            mock_counts.assert_not_called()
            mock_hash.assert_not_called()
            mock_rebuild.assert_not_called()

            # Dirty flag should remain False
            self.assertFalse(self.spatial_index._positions_dirty)

    def test_update_counts_changed(self):
        """Test update when counts change."""
        # Mock the change detection methods
        with patch.object(self.spatial_index, "_counts_changed", return_value=True) as mock_counts, patch.object(
            self.spatial_index, "_hash_positions_changed"
        ) as mock_hash, patch.object(self.spatial_index, "_rebuild_kdtrees") as mock_rebuild:
            self.spatial_index.update()

            # Should call rebuild and not check hash
            # Note: _counts_changed is called with current_agent_count parameter
            self.assertEqual(mock_counts.call_count, 1)
            mock_hash.assert_not_called()
            mock_rebuild.assert_called_once()

            # Dirty flag should be cleared
            self.assertFalse(self.spatial_index._positions_dirty)

    def test_update_hash_changed(self):
        """Test update when hash changes."""
        # Mock the change detection methods
        with patch.object(self.spatial_index, "_counts_changed", return_value=False) as mock_counts, patch.object(
            self.spatial_index, "_hash_positions_changed", return_value=True
        ) as mock_hash, patch.object(self.spatial_index, "_rebuild_kdtrees") as mock_rebuild:
            self.spatial_index.update()

            # Should call both checks and rebuild
            # Note: Both methods are called with their required parameters
            self.assertEqual(mock_counts.call_count, 1)
            self.assertEqual(mock_hash.call_count, 1)
            mock_rebuild.assert_called_once()

            # Dirty flag should be cleared
            self.assertFalse(self.spatial_index._positions_dirty)

    def test_get_nearby_agents_validation(self):
        """Test input validation for get_nearby_agents."""
        # Test invalid radius
        result = self.spatial_index.get_nearby_agents((10, 10), -1)
        self.assertEqual(result, [])

        result = self.spatial_index.get_nearby_agents((10, 10), 0)
        self.assertEqual(result, [])

        # Test invalid position
        result = self.spatial_index.get_nearby_agents((-1, 10), 5)
        self.assertEqual(result, [])

        result = self.spatial_index.get_nearby_agents((10, 101), 5)  # Outside height
        self.assertEqual(result, [])

    def test_get_nearby_agents_no_kdtree(self):
        """Test get_nearby_agents when KD-tree is None."""
        # Ensure no agents and no KD-tree
        self.agents = []
        self.spatial_index.set_references(self.agents, self.resources)
        self.spatial_index.agent_kdtree = None

        result = self.spatial_index.get_nearby_agents((10, 10), 5)
        self.assertEqual(result, [])

    def test_get_nearby_agents_with_agents(self):
        """Test get_nearby_agents with actual agents."""
        # Add agents
        agent1 = MockBaseAgent(
            agent_id="agent1",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agent2 = MockBaseAgent(
            agent_id="agent2",
            position=(15, 15),
            resource_level=30,
            environment=None,
            generation=0,
        )
        agent3 = MockBaseAgent(
            agent_id="agent3",
            position=(50, 50),
            resource_level=20,
            environment=None,
            generation=0,
        )
        self.agents.append(agent1)
        self.agents.append(agent2)
        self.agents.append(agent3)

        # Force rebuild KD-tree
        self.spatial_index._rebuild_kdtrees()

        # Test query around agent1
        nearby = self.spatial_index.get_nearby_agents((10, 10), 10)
        self.assertEqual(len(nearby), 2)  # agent1 and agent2
        agent_ids = [agent.agent_id for agent in nearby]
        self.assertIn("agent1", agent_ids)
        self.assertIn("agent2", agent_ids)
        self.assertNotIn("agent3", agent_ids)

    def test_get_nearby_agents_excludes_dead_agents(self):
        """Test that get_nearby_agents excludes dead agents."""
        # Add agents
        agent1 = MockBaseAgent(
            agent_id="agent1",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agent2 = MockBaseAgent(
            agent_id="agent2",
            position=(15, 15),
            resource_level=30,
            environment=None,
            generation=0,
        )
        self.agents.append(agent1)
        self.agents.append(agent2)

        # Kill one agent
        agent2.alive = False

        # Force rebuild KD-tree
        self.spatial_index._rebuild_kdtrees()

        # Test query
        nearby = self.spatial_index.get_nearby_agents((10, 10), 10)
        self.assertEqual(len(nearby), 1)
        self.assertEqual(nearby[0].agent_id, "agent1")

    def test_cache_invalidation_on_agent_death(self):
        """Test that cache is properly invalidated when agents die, preventing stale cache issues."""
        # Add agents
        agent1 = MockBaseAgent(
            agent_id="agent1",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agent2 = MockBaseAgent(
            agent_id="agent2",
            position=(15, 15),
            resource_level=30,
            environment=None,
            generation=0,
        )
        self.agents.append(agent1)
        self.agents.append(agent2)

        # Build KD-tree with both agents alive
        self.spatial_index._rebuild_kdtrees()

        # Verify both agents are in cache
        self.assertIsNotNone(self.spatial_index._cached_alive_agents)
        cached_agents = self.spatial_index._cached_alive_agents
        assert cached_agents is not None  # Type assertion for linter
        self.assertEqual(len(cached_agents), 2)

        # Kill agent2 and trigger proper cache invalidation (simulating proper death handling)
        agent2.alive = False
        self.spatial_index.mark_positions_dirty()  # This is what Environment.remove_agent() does
        self.spatial_index.update()  # This rebuilds the cache with only alive agents

        # Query with properly updated cache
        nearby = self.spatial_index.get_nearby_agents((10, 10), 10)

        # Should only return alive agent since cache was properly invalidated
        self.assertEqual(len(nearby), 1)
        self.assertEqual(nearby[0].agent_id, "agent1")
        self.assertNotIn(agent2, nearby)

        # Verify the dead agent is no longer in the updated cache
        updated_cache = self.spatial_index._cached_alive_agents
        assert updated_cache is not None  # Type assertion for linter
        self.assertNotIn(agent2, updated_cache)
        self.assertEqual(len(updated_cache), 1)  # Cache contains only alive agents

    def test_get_nearby_resources_input_validation(self):
        """Test input validation for get_nearby_resources (bug fix)."""
        # Test invalid radius
        result = self.spatial_index.get_nearby_resources((10, 10), -1)
        self.assertEqual(result, [])

        result = self.spatial_index.get_nearby_resources((10, 10), 0)
        self.assertEqual(result, [])

        # Test invalid position
        result = self.spatial_index.get_nearby_resources((-1, 10), 5)
        self.assertEqual(result, [])

        result = self.spatial_index.get_nearby_resources((10, 101), 5)  # Outside height
        self.assertEqual(result, [])

        # Test valid inputs still work
        self.spatial_index._rebuild_kdtrees()
        result = self.spatial_index.get_nearby_resources((10, 10), 5)
        self.assertIsInstance(result, list)

    def test_get_nearest_resource_input_validation(self):
        """Test input validation for get_nearest_resource (bug fix)."""
        # Test invalid position (outside 1% margin)
        result = self.spatial_index.get_nearest_resource((-2, 10))  # Outside 1% margin
        self.assertIsNone(result)

        result = self.spatial_index.get_nearest_resource((10, 102))  # Outside 1% margin
        self.assertIsNone(result)

        # Test valid inputs still work
        self.spatial_index._rebuild_kdtrees()
        result = self.spatial_index.get_nearest_resource((10, 10))
        self.assertIsInstance(result, Resource)

    def test_consistent_input_validation_across_methods(self):
        """Test that all spatial query methods have consistent input validation."""
        # Test boundary conditions (positions outside 1% margin)
        boundary_positions = [
            (-2, 50),  # Outside 1% margin left boundary
            (102, 50),  # Outside 1% margin right boundary
            (50, -2),  # Outside 1% margin top boundary
            (50, 102),  # Outside 1% margin bottom boundary
            (-10, -10),  # Way outside all boundaries
            (110, 110),  # Way outside all boundaries
        ]

        # Test that all methods reject invalid positions consistently
        for pos in boundary_positions:
            # All methods should handle invalid positions gracefully
            agents_result = self.spatial_index.get_nearby_agents(pos, 5)
            resources_result = self.spatial_index.get_nearby_resources(pos, 5)
            nearest_result = self.spatial_index.get_nearest_resource(pos)

            self.assertEqual(agents_result, [])
            self.assertEqual(resources_result, [])
            self.assertIsNone(nearest_result)

        # Test valid boundary positions (should work)
        valid_boundary_positions = [
            (0, 0),  # Top-left corner
            (100, 100),  # Bottom-right corner
            (0, 50),  # Left edge
            (100, 50),  # Right edge
            (50, 0),  # Top edge
            (50, 100),  # Bottom edge
        ]

        self.spatial_index._rebuild_kdtrees()
        for pos in valid_boundary_positions:
            # All methods should accept valid boundary positions
            agents_result = self.spatial_index.get_nearby_agents(pos, 5)
            resources_result = self.spatial_index.get_nearby_resources(pos, 5)
            nearest_result = self.spatial_index.get_nearest_resource(pos)

            self.assertIsInstance(agents_result, list)
            self.assertIsInstance(resources_result, list)
            # nearest_result could be None if no resources, but should not crash

    def test_get_nearby_resources(self):
        """Test get_nearby_resources functionality."""
        # Test with no KD-tree
        self.spatial_index.resource_kdtree = None
        result = self.spatial_index.get_nearby_resources((10, 10), 5)
        self.assertEqual(result, [])

        # Rebuild KD-tree
        self.spatial_index._rebuild_kdtrees()

        # Test query
        nearby = self.spatial_index.get_nearby_resources((10, 10), 20)
        self.assertGreater(len(nearby), 0)
        self.assertLessEqual(len(nearby), len(self.resources))

    def test_get_nearest_resource(self):
        """Test get_nearest_resource functionality."""
        # Test with no KD-tree
        self.spatial_index.resource_kdtree = None
        with patch.object(self.spatial_index, "update") as mock_update:
            result = self.spatial_index.get_nearest_resource((10, 10))
            self.assertIsNone(result)
            mock_update.assert_called_once()

        # Rebuild KD-tree
        self.spatial_index._rebuild_kdtrees()

        # Test query
        nearest = self.spatial_index.get_nearest_resource((10, 10))
        self.assertIsNotNone(nearest)
        self.assertIsInstance(nearest, Resource)

    def test_get_agent_count(self):
        """Test get_agent_count method."""
        # Initially no agents
        self.assertEqual(self.spatial_index.get_agent_count(), 0)

        # Add an agent
        agent = MockBaseAgent(
            agent_id="test_agent",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        self.agents.append(agent)
        self.assertEqual(self.spatial_index.get_agent_count(), 1)

        # Kill the agent
        agent.alive = False
        self.assertEqual(self.spatial_index.get_agent_count(), 0)

    def test_get_resource_count(self):
        """Test get_resource_count method."""
        self.assertEqual(self.spatial_index.get_resource_count(), 5)

    def test_is_dirty(self):
        """Test is_dirty method."""
        # Initially dirty
        self.assertTrue(self.spatial_index.is_dirty())

        # Clear dirty flag
        self.spatial_index._positions_dirty = False
        self.assertFalse(self.spatial_index.is_dirty())

        # Mark as dirty
        self.spatial_index.mark_positions_dirty()
        self.assertTrue(self.spatial_index.is_dirty())

    def test_force_rebuild(self):
        """Test force_rebuild method."""
        # Mark as dirty
        self.spatial_index.mark_positions_dirty()
        self.assertTrue(self.spatial_index.is_dirty())

        # Force rebuild
        self.spatial_index.force_rebuild()
        self.assertFalse(self.spatial_index.is_dirty())

    def test_get_stats(self):
        """Test get_stats method."""
        stats = self.spatial_index.get_stats()

        self.assertIn("agent_count", stats)
        self.assertIn("resource_count", stats)
        self.assertIn("agent_kdtree_exists", stats)
        self.assertIn("resource_kdtree_exists", stats)
        self.assertIn("positions_dirty", stats)
        self.assertIn("cached_counts", stats)
        self.assertIn("cached_hash", stats)

        self.assertEqual(stats["agent_count"], 0)
        self.assertEqual(stats["resource_count"], 5)
        self.assertFalse(stats["agent_kdtree_exists"])
        self.assertFalse(stats["resource_kdtree_exists"])  # Not built initially
        self.assertTrue(stats["positions_dirty"])  # Initially dirty

    def test_performance_optimization_no_rebuild(self):
        """Test that KD-trees are not rebuilt unnecessarily."""
        # Add an agent
        agent = MockBaseAgent(
            agent_id="test_agent",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        self.agents.append(agent)

        # Force initial rebuild and mark as dirty to trigger update logic
        self.spatial_index._rebuild_kdtrees()
        self.spatial_index._positions_dirty = True

        # Mock the change detection methods to return False
        with patch.object(self.spatial_index, "_counts_changed", return_value=False) as mock_counts, patch.object(
            self.spatial_index, "_hash_positions_changed", return_value=False
        ) as mock_hash, patch.object(self.spatial_index, "_rebuild_kdtrees") as mock_rebuild:
            # Call update multiple times, marking dirty each time
            for _ in range(5):
                self.spatial_index._positions_dirty = True
                self.spatial_index.update()

            # Should check for changes but not rebuild
            self.assertEqual(mock_counts.call_count, 5)
            self.assertEqual(mock_hash.call_count, 5)
            mock_rebuild.assert_not_called()

    def test_cached_alive_agents(self):
        """Test that cached alive agents are properly used and invalidated."""
        # Add agents and rebuild
        agent1 = MockBaseAgent(
            agent_id="agent1",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agent2 = MockBaseAgent(
            agent_id="agent2",
            position=(20, 20),
            resource_level=30,
            environment=None,
            generation=0,
        )
        self.agents.append(agent1)
        self.agents.append(agent2)

        # Force rebuild to set cache
        self.spatial_index.force_rebuild()

        # Verify cache is set correctly
        self.assertIsNotNone(self.spatial_index._cached_alive_agents)
        cached_agents = self.spatial_index._cached_alive_agents
        assert cached_agents is not None  # Type assertion for linter
        self.assertEqual(len(cached_agents), 2)
        self.assertIn(agent1, cached_agents)
        self.assertIn(agent2, cached_agents)

        # Kill one agent and verify cache is updated on rebuild
        agent2.alive = False
        self.spatial_index.force_rebuild()
        cached_agents = self.spatial_index._cached_alive_agents
        self.assertIsNotNone(cached_agents)
        assert cached_agents is not None  # Type assertion for linter
        self.assertEqual(len(cached_agents), 1)
        self.assertIn(agent1, cached_agents)
        self.assertNotIn(agent2, cached_agents)

        # Verify queries use cached agents correctly
        nearby = self.spatial_index.get_nearby_agents((10, 10), 5)
        self.assertEqual(len(nearby), 1)
        self.assertEqual(nearby[0].agent_id, "agent1")

    def test_position_duplicates(self):
        """Test handling of multiple agents at the same position."""
        duplicate_pos = (50, 50)
        num_duplicates = 5

        # Add agents at duplicate position
        for i in range(num_duplicates):
            agent = MockBaseAgent(
                agent_id=f"dup{i}",
                position=duplicate_pos,
                resource_level=50,
                environment=None,
                generation=0,
            )
            self.agents.append(agent)

        # Add some other agents
        for i in range(3):
            agent = MockBaseAgent(
                agent_id=f"other{i}",
                position=(i * 10, i * 10),
                resource_level=50,
                environment=None,
                generation=0,
            )
            self.agents.append(agent)

        self.spatial_index.force_rebuild()

        # Query at the duplicate position with small radius
        nearby = self.spatial_index.get_nearby_agents(duplicate_pos, 0.1)
        self.assertEqual(len(nearby), num_duplicates)  # Should return all duplicates

        # Check that all returned agents are at the position
        for agent in nearby:
            self.assertEqual(tuple(agent.position), duplicate_pos)

    def test_floating_point_precision_in_hashes(self):
        """Test hash detection with floating-point precision issues."""
        # Add agents with precise positions
        agent1 = MockBaseAgent(
            agent_id="agent1",
            position=(10.0, 10.0),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agent2 = MockBaseAgent(
            agent_id="agent2",
            position=(20.0, 20.0),
            resource_level=50,
            environment=None,
            generation=0,
        )
        self.agents.append(agent1)
        self.agents.append(agent2)

        self.spatial_index.force_rebuild()
        alive_agents = [agent1, agent2]
        self.spatial_index._hash_positions_changed(alive_agents)  # Cache initial hash

        # Make a tiny floating-point change (e.g., arithmetic precision)
        agent1.position[0] += 1e-10  # Very small change
        self.assertTrue(self.spatial_index._hash_positions_changed(alive_agents))  # Should detect

        # Reset and test identical positions (should not detect change)
        agent1.position[0] -= 1e-10  # Restore exact
        self.spatial_index.force_rebuild()
        self.spatial_index._hash_positions_changed(alive_agents)  # Recache
        self.assertFalse(self.spatial_index._hash_positions_changed(alive_agents))  # No change

        # Test with floating-point equality (e.g., 0.1 + 0.2 != 0.3 but close)
        agent1.position[0] = 0.1 + 0.2
        self.spatial_index.force_rebuild()
        self.spatial_index._hash_positions_changed(alive_agents)
        agent1.position[0] = 0.3  # Mathematically equal but different binary
        self.assertTrue(self.spatial_index._hash_positions_changed(alive_agents))  # Should detect binary difference

    def test_very_large_number_of_agents(self):
        """Test performance and correctness with >10k agents."""
        import time

        num_agents = 15000  # >10k
        for i in range(num_agents):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            agent = MockBaseAgent(
                agent_id=f"agent{i}",
                position=(x, y),
                resource_level=50,
                environment=None,
                generation=0,
            )
            self.agents.append(agent)

        # Measure build time with flexible threshold
        start_time = time.time()
        self.spatial_index.force_rebuild()
        build_time = time.time() - start_time
        # Allow 0.5ms per agent for build time
        max_build_time = max(2.0, num_agents * 0.0005)
        self.assertLess(
            build_time,
            max_build_time,
            f"Build time {build_time:.2f}s exceeded threshold {max_build_time:.2f}s",
        )

        # Measure query time with flexible threshold
        start_time = time.time()
        num_queries = 100
        for _ in range(num_queries):
            nearby = self.spatial_index.get_nearby_agents((50, 50), 10)
            self.assertIsInstance(nearby, list)
        query_time = time.time() - start_time
        # Allow 10ms per query
        max_query_time = max(1.0, num_queries * 0.01)
        self.assertLess(
            query_time,
            max_query_time,
            f"Query time {query_time:.2f}s exceeded threshold {max_query_time:.2f}s",
        )

        # Check count
        self.assertEqual(self.spatial_index.get_agent_count(), num_agents)

    def test_spatial_query_performance(self):
        """Test that spatial queries are efficient."""
        # Add many agents
        agents = []
        for i in range(50):
            agent = MockBaseAgent(
                agent_id=f"agent{i}",
                position=(i * 2, i * 2),
                resource_level=50,
                environment=None,
                generation=0,
            )
            agents.append(agent)
            self.agents.append(agent)

        # Force rebuild KD-tree
        self.spatial_index._rebuild_kdtrees()

        # Test multiple queries with flexible threshold
        import time

        start_time = time.time()

        num_queries = 100
        for _ in range(num_queries):
            self.spatial_index.get_nearby_agents((50, 50), 10)

        end_time = time.time()
        query_time = end_time - start_time

        # Allow 10ms per query with minimum 1 second
        max_query_time = max(1.0, num_queries * 0.01)
        self.assertLess(
            query_time,
            max_query_time,
            f"Query time {query_time:.2f}s exceeded threshold {max_query_time:.2f}s",
        )

    def test_hash_collision_handling(self):
        """Test that hash-based change detection handles edge cases."""
        # Test with no agents and no resources (empty lists)
        self.agents = []
        self.resources = []
        self.spatial_index.set_references(self.agents, self.resources)

        # Should not crash and should return True on first call
        result = self.spatial_index._hash_positions_changed([])
        self.assertTrue(result)  # First call should return True

        # Test with empty lists but cached hash exists
        result = self.spatial_index._hash_positions_changed([])
        self.assertFalse(result)  # Second call with same state should return False

        # Test with None cached hash (edge case)
        self.spatial_index._cached_hash = None
        result = self.spatial_index._hash_positions_changed([])
        self.assertTrue(result)  # Should return True when cached hash is None

        # Test with actual position changes
        agent = MockBaseAgent(
            agent_id="test_agent",
            position=(10, 10),
            resource_level=50,
            environment=None,
            generation=0,
        )
        self.agents.append(agent)

        # Should detect change when agent is added
        result = self.spatial_index._hash_positions_changed([agent])
        self.assertTrue(result)  # Should detect change when agent is added

    def test_boundary_conditions(self):
        """Test SpatialIndex behavior at boundary conditions."""
        # Test with maximum number of agents
        agents = []
        for i in range(1000):
            agent = MockBaseAgent(
                agent_id=f"agent{i}",
                position=(i % 100, i % 100),
                resource_level=50,
                environment=None,
                generation=0,
            )
            agents.append(agent)
            self.agents.append(agent)

        # Should not crash
        self.spatial_index._rebuild_kdtrees()

        # Should be able to query
        nearby = self.spatial_index.get_nearby_agents((50, 50), 10)
        self.assertIsInstance(nearby, list)

        # Test with no resources
        self.resources = []
        self.spatial_index.set_references(self.agents, self.resources)
        self.spatial_index._rebuild_kdtrees()

        # Should handle gracefully
        nearby_resources = self.spatial_index.get_nearby_resources((10, 10), 5)
        self.assertEqual(nearby_resources, [])

        nearest_resource = self.spatial_index.get_nearest_resource((10, 10))
        self.assertIsNone(nearest_resource)

    def test_concurrent_modifications(self):
        """Test behavior under simulated concurrent modifications."""
        import threading
        import time

        # Add initial agents
        for i in range(10):
            agent = MockBaseAgent(
                agent_id=f"agent{i}",
                position=(i * 10, i * 10),
                resource_level=50,
                environment=None,
                generation=0,
            )
            self.agents.append(agent)

        self.spatial_index.force_rebuild()

        # Function to modify positions in another thread
        def modify_positions():
            time.sleep(0.1)  # Wait a bit to overlap with query
            for agent in self.agents[:5]:  # Modify first 5 agents
                agent.position[0] += 1.0
            self.spatial_index.mark_positions_dirty()
            self.spatial_index.update()  # Simulate update in another thread

        # Start modification thread
        mod_thread = threading.Thread(target=modify_positions)
        mod_thread.start()

        # Perform query in main thread (may race)
        nearby = self.spatial_index.get_nearby_agents((0, 0), 50)
        self.assertIsInstance(nearby, list)  # Should not crash

        mod_thread.join()

        # After modification, force rebuild and check
        self.spatial_index.force_rebuild()
        nearby_after = self.spatial_index.get_nearby_agents((0, 0), 50)
        self.assertIsInstance(nearby_after, list)

    def test_is_valid_position_method(self):
        """Test the _is_valid_position method directly."""
        # Test valid positions
        self.assertTrue(self.spatial_index._is_valid_position((0, 0)))
        self.assertTrue(self.spatial_index._is_valid_position((50, 50)))
        self.assertTrue(self.spatial_index._is_valid_position((100, 100)))
        self.assertTrue(self.spatial_index._is_valid_position((25.5, 75.3)))

        # Test invalid positions (outside 1% margin)
        self.assertFalse(self.spatial_index._is_valid_position((-2, 50)))  # Outside 1% margin
        self.assertFalse(self.spatial_index._is_valid_position((102, 50)))  # Outside 1% margin
        self.assertFalse(self.spatial_index._is_valid_position((50, -2)))  # Outside 1% margin
        self.assertFalse(self.spatial_index._is_valid_position((50, 102)))  # Outside 1% margin
        self.assertFalse(self.spatial_index._is_valid_position((-10, -10)))  # Way outside bounds

        # Test positions at exact boundaries (should be valid)
        self.assertTrue(self.spatial_index._is_valid_position((0, 0)))
        self.assertTrue(self.spatial_index._is_valid_position((100, 100)))

        # Test positions within the 1% margin outside bounds (should be valid)
        margin_x = 100 * 0.01  # 1.0
        margin_y = 100 * 0.01  # 1.0
        self.assertTrue(self.spatial_index._is_valid_position((-margin_x + 0.1, 50)))
        self.assertTrue(self.spatial_index._is_valid_position((100 + margin_x - 0.1, 50)))
        self.assertTrue(self.spatial_index._is_valid_position((50, -margin_y + 0.1)))
        self.assertTrue(self.spatial_index._is_valid_position((50, 100 + margin_y - 0.1)))

        # Test positions just outside the margin (should be invalid)
        self.assertFalse(self.spatial_index._is_valid_position((-margin_x - 0.1, 50)))
        self.assertFalse(self.spatial_index._is_valid_position((100 + margin_x + 0.1, 50)))
        self.assertFalse(self.spatial_index._is_valid_position((50, -margin_y - 0.1)))
        self.assertFalse(self.spatial_index._is_valid_position((50, 100 + margin_y + 0.1)))

    def test_extreme_environment_sizes(self):
        """Test SpatialIndex with very small and very large environments."""
        # Test with very small environment
        small_index = SpatialIndex(width=1.0, height=1.0)
        small_index.set_references([], [])

        # Valid positions in small environment
        self.assertTrue(small_index._is_valid_position((0, 0)))
        self.assertTrue(small_index._is_valid_position((1, 1)))
        self.assertTrue(small_index._is_valid_position((0.5, 0.5)))

        # Invalid positions in small environment
        self.assertFalse(small_index._is_valid_position((-1, 0.5)))
        self.assertFalse(small_index._is_valid_position((0.5, -1)))
        self.assertFalse(small_index._is_valid_position((2, 0.5)))

        # Test margin calculation for small environment (1% of 1.0 = 0.01)
        self.assertTrue(small_index._is_valid_position((-0.009, 0.5)))  # Within margin
        self.assertFalse(small_index._is_valid_position((-0.011, 0.5)))  # Outside margin

        # Test with very large environment
        large_index = SpatialIndex(width=1000000, height=1000000)
        large_index.set_references([], [])

        # Valid positions in large environment
        self.assertTrue(large_index._is_valid_position((0, 0)))
        self.assertTrue(large_index._is_valid_position((500000, 500000)))
        self.assertTrue(large_index._is_valid_position((1000000, 1000000)))

        # Test margin calculation for large environment (1% of 1000000 = 10000)
        self.assertTrue(large_index._is_valid_position((-9999, 500000)))  # Within margin
        self.assertFalse(large_index._is_valid_position((-10001, 500000)))  # Outside margin

    def test_boundary_resources(self):
        """Test resources positioned at exact boundaries."""
        # Create resources at boundary positions
        boundary_resources = [
            Resource(0, (0, 0), 10, 20, 0.1),  # Top-left corner
            Resource(1, (100, 0), 10, 20, 0.1),  # Top-right corner
            Resource(2, (0, 100), 10, 20, 0.1),  # Bottom-left corner
            Resource(3, (100, 100), 10, 20, 0.1),  # Bottom-right corner
            Resource(4, (50, 0), 10, 20, 0.1),  # Top edge
            Resource(5, (0, 50), 10, 20, 0.1),  # Left edge
            Resource(6, (100, 50), 10, 20, 0.1),  # Right edge
            Resource(7, (50, 100), 10, 20, 0.1),  # Bottom edge
        ]

        # Test with boundary resources
        boundary_index = SpatialIndex(width=100, height=100)
        boundary_index.set_references([], boundary_resources)
        boundary_index._rebuild_kdtrees()

        # Test queries at boundary positions
        for resource in boundary_resources:
            # Query at exact resource position with small radius
            nearby = boundary_index.get_nearby_resources(resource.position, 0.1)
            self.assertEqual(len(nearby), 1)
            self.assertEqual(nearby[0].resource_id, resource.resource_id)

            # Test nearest resource query
            nearest = boundary_index.get_nearest_resource(resource.position)
            self.assertIsNotNone(nearest)
            self.assertEqual(nearest.resource_id, resource.resource_id)

    def test_complex_agent_lifecycle(self):
        """Test complex scenarios with agent birth, death, and position changes."""
        # Start with empty environment
        lifecycle_index = SpatialIndex(width=100, height=100)
        agents = []
        resources = []
        lifecycle_index.set_references(agents, resources)

        # Phase 1: Add initial agents
        initial_agents = []
        for i in range(10):
            agent = MockBaseAgent(
                agent_id=f"initial_{i}",
                position=(i * 10, i * 10),
                resource_level=50,
                environment=None,
                generation=i,
            )
            initial_agents.append(agent)
            agents.append(agent)

        lifecycle_index.update()
        self.assertEqual(lifecycle_index.get_agent_count(), 10)

        # Phase 2: Some agents die
        dead_count = 0
        for i in range(0, 10, 2):  # Kill every other agent
            initial_agents[i].alive = False
            dead_count += 1

        lifecycle_index.mark_positions_dirty()
        lifecycle_index.update()
        self.assertEqual(lifecycle_index.get_agent_count(), 10 - dead_count)

        # Phase 3: New agents are born
        new_agents = []
        for i in range(5):
            agent = MockBaseAgent(
                agent_id=f"new_{i}",
                position=(i * 20 + 5, i * 20 + 5),
                resource_level=30,
                environment=None,
                generation=10 + i,
            )
            new_agents.append(agent)
            agents.append(agent)

        lifecycle_index.mark_positions_dirty()
        lifecycle_index.update()
        expected_alive = (10 - dead_count) + 5
        self.assertEqual(lifecycle_index.get_agent_count(), expected_alive)

        # Phase 4: Agents move around
        for agent in agents:
            if agent.alive:
                agent.position[0] += np.random.uniform(-2, 2)
                agent.position[1] += np.random.uniform(-2, 2)

        lifecycle_index.mark_positions_dirty()
        lifecycle_index.update()

        # Phase 5: Batch operations (multiple deaths and births)
        # Kill some more agents
        additional_deaths = 0
        for agent in agents[:3]:
            if agent.alive:
                agent.alive = False
                additional_deaths += 1

        # Add more agents
        batch_new_agents = []
        for i in range(3):
            agent = MockBaseAgent(
                agent_id=f"batch_{i}",
                position=(np.random.uniform(0, 100), np.random.uniform(0, 100)),
                resource_level=40,
                environment=None,
                generation=15 + i,
            )
            batch_new_agents.append(agent)
            agents.append(agent)

        lifecycle_index.mark_positions_dirty()
        lifecycle_index.update()
        final_count = lifecycle_index.get_agent_count()
        self.assertEqual(final_count, expected_alive - additional_deaths + 3)

        # Verify spatial queries still work after complex lifecycle
        center_pos = (50, 50)
        nearby = lifecycle_index.get_nearby_agents(center_pos, 20)
        self.assertIsInstance(nearby, list)
        # All nearby agents should be alive (the KD-tree should only contain alive agents)
        for agent in nearby:
            self.assertTrue(agent.alive, f"Agent {agent.agent_id} is not alive but was returned by spatial query")

        # Also verify that get_agent_count matches the number of alive agents
        alive_count = sum(1 for agent in agents if agent.alive)
        self.assertEqual(lifecycle_index.get_agent_count(), alive_count)

    def test_hash_edge_cases(self):
        """Test hash-based change detection with edge cases."""
        hash_index = SpatialIndex(width=100, height=100)
        agents = []
        resources = []
        hash_index.set_references(agents, resources)

        # Test with empty state
        self.assertTrue(hash_index._hash_positions_changed([]))  # First call should return True
        self.assertFalse(hash_index._hash_positions_changed([]))  # Second call should return False

        # Test with very large coordinates
        large_coord_agent = MockBaseAgent(
            agent_id="large_coord",
            position=(1e6, 1e6),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agents.append(large_coord_agent)
        self.assertTrue(hash_index._hash_positions_changed([large_coord_agent]))

        # Test with very small coordinates
        small_coord_agent = MockBaseAgent(
            agent_id="small_coord",
            position=(1e-6, 1e-6),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agents.clear()
        agents.append(small_coord_agent)
        hash_index._rebuild_kdtrees()
        hash_index._hash_positions_changed([small_coord_agent])  # Cache hash

        # Make tiny change
        small_coord_agent.position[0] += 1e-10
        self.assertTrue(hash_index._hash_positions_changed([small_coord_agent]))

        # Test with negative coordinates
        negative_coord_agent = MockBaseAgent(
            agent_id="negative_coord",
            position=(-50, -50),
            resource_level=50,
            environment=None,
            generation=0,
        )
        agents.clear()
        agents.append(negative_coord_agent)
        self.assertTrue(hash_index._hash_positions_changed([negative_coord_agent]))

        # Test hash consistency with identical positions
        agents.clear()
        for i in range(3):
            agent = MockBaseAgent(
                agent_id=f"identical_{i}",
                position=(10.123456789, 20.987654321),
                resource_level=50,
                environment=None,
                generation=i,
            )
            agents.append(agent)

        hash_index._rebuild_kdtrees()
        original_hash = hash_index._cached_hash

        # Force rebuild again - hash should be identical
        hash_index._rebuild_kdtrees()
        self.assertEqual(hash_index._cached_hash, original_hash)

    def test_memory_usage_large_datasets(self):
        """Test memory usage and performance with large datasets."""
        import os

        import psutil

        large_index = SpatialIndex(width=1000, height=1000)
        agents = []
        resources = []

        # Create large number of agents
        num_agents = 50000
        for i in range(num_agents):
            agent = MockBaseAgent(
                agent_id=f"mem_test_{i}",
                position=(np.random.uniform(0, 1000), np.random.uniform(0, 1000)),
                resource_level=50,
                environment=None,
                generation=i % 10,
            )
            agents.append(agent)

        # Create large number of resources
        num_resources = 10000
        for i in range(num_resources):
            resource = Resource(
                resource_id=i,
                position=(np.random.uniform(0, 1000), np.random.uniform(0, 1000)),
                amount=10,
                max_amount=20,
                regeneration_rate=0.1,
            )
            resources.append(resource)

        large_index.set_references(agents, resources)

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Build KD-trees
        start_time = time.time()
        large_index._rebuild_kdtrees()
        build_time = time.time() - start_time

        build_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = build_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for 50k agents + 10k resources)
        self.assertLess(memory_increase, 100, f"Memory increase {memory_increase:.1f}MB exceeds threshold")

        # Build time should be reasonable (less than 5 seconds)
        self.assertLess(build_time, 5.0, f"Build time {build_time:.2f}s exceeds threshold")

        # Test query performance
        query_start = time.time()
        num_queries = 100
        for _ in range(num_queries):
            nearby_agents = large_index.get_nearby_agents((500, 500), 50)
            nearby_resources = large_index.get_nearby_resources((500, 500), 50)

        query_time = time.time() - query_start
        avg_query_time = query_time / num_queries

        # Average query time should be less than 10ms
        self.assertLess(avg_query_time, 0.01, f"Average query time {avg_query_time:.4f}s exceeds threshold")

        # Verify correctness
        self.assertEqual(large_index.get_agent_count(), num_agents)
        self.assertEqual(large_index.get_resource_count(), num_resources)

        # Test that queries return reasonable results
        center_query = large_index.get_nearby_agents((500, 500), 50)
        self.assertIsInstance(center_query, list)
        # Should find some agents in a 50-unit radius around center of 1000x1000 area
        self.assertGreater(len(center_query), 0)

    def test_margin_calculation_edge_cases(self):
        """Test edge cases in margin calculation for position validation."""
        # Test with zero dimensions (edge case)
        zero_index = SpatialIndex(width=0, height=0)
        # With zero dimensions, margin is 0, so only (0,0) is valid
        self.assertTrue(zero_index._is_valid_position((0, 0)))
        self.assertFalse(zero_index._is_valid_position((0.1, 0)))
        self.assertFalse(zero_index._is_valid_position((0, 0.1)))

        # Test with very small dimensions
        tiny_index = SpatialIndex(width=0.1, height=0.1)
        # Margin is 0.001 (1% of 0.1)
        self.assertTrue(tiny_index._is_valid_position((0, 0)))
        self.assertTrue(tiny_index._is_valid_position((0.1, 0.1)))
        self.assertTrue(tiny_index._is_valid_position((-0.0009, 0.05)))  # Within margin
        self.assertFalse(tiny_index._is_valid_position((-0.0011, 0.05)))  # Outside margin

        # Test with fractional dimensions
        fractional_index = SpatialIndex(width=10.5, height=7.25)
        margin_x = 10.5 * 0.01  # 0.105
        margin_y = 7.25 * 0.01  # 0.0725

        # Test boundary positions
        self.assertTrue(fractional_index._is_valid_position((0, 0)))
        self.assertTrue(fractional_index._is_valid_position((10.5, 7.25)))

        # Test positions within margin
        self.assertTrue(fractional_index._is_valid_position((-margin_x + 0.01, 3.625)))
        self.assertTrue(fractional_index._is_valid_position((10.5 + margin_x - 0.01, 3.625)))
        self.assertTrue(fractional_index._is_valid_position((5.25, -margin_y + 0.01)))
        self.assertTrue(fractional_index._is_valid_position((5.25, 7.25 + margin_y - 0.01)))

        # Test positions outside margin
        self.assertFalse(fractional_index._is_valid_position((-margin_x - 0.01, 3.625)))
        self.assertFalse(fractional_index._is_valid_position((10.5 + margin_x + 0.01, 3.625)))
        self.assertFalse(fractional_index._is_valid_position((5.25, -margin_y - 0.01)))
        self.assertFalse(fractional_index._is_valid_position((5.25, 7.25 + margin_y + 0.01)))

class TestSpatialIndexNamedIndices(unittest.TestCase):
    """Additional tests for configurable named indices and generic getters."""

    def test_register_custom_index_filtering_and_queries(self):
        index = SpatialIndex(width=100, height=100)

        # Prepare agents with a custom attribute used for filtering
        agents = []
        predator = MockBaseAgent(
            agent_id="pred1",
            position=(10, 10),
            resource_level=10,
            environment=None,
        )
        setattr(predator, "type", "predator")

        prey = MockBaseAgent(
            agent_id="prey1",
            position=(40, 40),
            resource_level=10,
            environment=None,
        )
        setattr(prey, "type", "prey")

        agents.extend([predator, prey])

        # Minimal resources
        resources = [
            Resource(resource_id=1, position=(0, 0), amount=5, max_amount=10, regeneration_rate=0.1),
            Resource(resource_id=2, position=(80, 80), amount=5, max_amount=10, regeneration_rate=0.1),
        ]

        index.set_references(agents, resources)

        # Register a custom named index of only predators
        index.register_index(
            name="predators",
            data_reference=agents,
            position_getter=lambda a: a.position,
            filter_func=lambda a: getattr(a, "type", None) == "predator",
        )

        # Build structures
        index.force_rebuild()

        # Generic nearby query (search all by default)
        nearby = index.get_nearby((12, 12), 10)
        self.assertIn("agents", nearby)
        self.assertIn("resources", nearby)
        self.assertIn("predators", nearby)

        # Predator should be found near (12,12); prey should not be in the custom list
        self.assertTrue(any(a.agent_id == "pred1" for a in nearby["predators"]))
        self.assertFalse(any(a.agent_id == "prey1" for a in nearby["predators"]))

        # Generic nearest limited to a specific index
        nearest = index.get_nearest((9, 9), index_names=["predators"])  # closest to predator
        self.assertIn("predators", nearest)
        self.assertIsNotNone(nearest["predators"])
        self.assertEqual(nearest["predators"].agent_id, "pred1")

    def test_constructor_supplied_indices(self):
        # Initial lists
        agents = []
        resources = []

        # Provide initial config/data for a custom "all_agents" index
        idx_configs = {
            "all_agents": {
                "position_getter": lambda a: a.position,
                # no filter -> include all
            }
        }
        idx_data = {
            "all_agents": agents,
        }

        custom_index = SpatialIndex(width=100, height=100, index_configs=idx_configs, index_data=idx_data)

        # Populate references
        a1 = MockBaseAgent("a1", (5, 5), 0, None)
        a2 = MockBaseAgent("a2", (60, 60), 0, None)
        agents.extend([a1, a2])
        resources.append(Resource(1, (10, 10), 5, 10, 0.1))

        custom_index.set_references(agents, resources)
        custom_index.force_rebuild()

        # Query only the constructor-supplied index
        nearby = custom_index.get_nearby((6, 6), 5, index_names=["all_agents"])
        self.assertIn("all_agents", nearby)
        self.assertTrue(any(a.agent_id == "a1" for a in nearby["all_agents"]))
        self.assertFalse(any(a.agent_id == "a2" for a in nearby["all_agents"]))

    def test_get_nearest_with_index_names(self):
        index = SpatialIndex(width=100, height=100)
        agents = []
        resources = []
        a1 = MockBaseAgent("x1", (25, 25), 0, None)
        a2 = MockBaseAgent("x2", (75, 75), 0, None)
        agents.extend([a1, a2])
        for i in range(3):
            resources.append(Resource(i, (i * 30, i * 30), 5, 10, 0.1))

        index.set_references(agents, resources)
        index.force_rebuild()

        nearest = index.get_nearest((26, 26), index_names=["agents"])  # should pick a1
        self.assertEqual(nearest.get("agents").agent_id, "x1")

        nearest_all = index.get_nearest((2, 2))  # all indices by default
        self.assertIn("agents", nearest_all)
        self.assertIn("resources", nearest_all)


if __name__ == "__main__":
    unittest.main()
