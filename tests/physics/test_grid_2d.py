"""Unit tests for Grid2DPhysics implementation.

This module tests all IPhysicsEngine methods implemented by Grid2DPhysics,
ensuring correct behavior for 2D grid-based environments.
"""

import math
import random
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest
from gymnasium import spaces

from farm.config.config import EnvironmentConfig, SimulationConfig
from farm.core.physics.grid_2d import Grid2DPhysics
from farm.core.observations import ObservationConfig


class TestGrid2DPhysics(unittest.TestCase):
    """Test Grid2DPhysics implementation of IPhysicsEngine protocol."""

    def setUp(self):
        """Set up test fixtures."""
        self.width = 100
        self.height = 100
        self.physics = Grid2DPhysics(
            width=self.width,
            height=self.height,
            seed=42
        )

    def test_validate_position_valid_positions(self):
        """Test position validation for valid positions."""
        # Test positions within bounds
        valid_positions = [
            (0, 0),           # Corner
            (50, 50),         # Center
            (99, 99),         # Other corner
            (0, 50),          # Edge
            (50, 0),          # Edge
            (25.5, 75.3),     # Float positions
        ]
        
        for position in valid_positions:
            with self.subTest(position=position):
                self.assertTrue(self.physics.validate_position(position))

    def test_validate_position_invalid_positions(self):
        """Test position validation for invalid positions."""
        # Test positions outside bounds
        invalid_positions = [
            (-1, 0),          # Negative x
            (0, -1),          # Negative y
            (100, 0),         # x at boundary (exclusive)
            (0, 100),         # y at boundary (exclusive)
            (100, 100),       # Both at boundary
            (-10, -10),       # Both negative
            (150, 50),        # x too large
            (50, 150),        # y too large
            (150, 150),       # Both too large
        ]
        
        for position in invalid_positions:
            with self.subTest(position=position):
                self.assertFalse(self.physics.validate_position(position))

    def test_validate_position_boundary_cases(self):
        """Test position validation for boundary cases."""
        # Test non-tuple inputs
        invalid_inputs = [
            "not a tuple",
            42,
            [0, 0],
            (0,),             # Too few elements
            (0, 0, 0),        # Too many elements
            None,
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                self.assertFalse(self.physics.validate_position(invalid_input))

    def test_get_nearby_entities_agents(self):
        """Test getting nearby agents."""
        # Create mock agents
        agent1 = Mock()
        agent1.position = (50, 50)
        agent2 = Mock()
        agent2.position = (52, 52)
        agent3 = Mock()
        agent3.position = (100, 100)  # Outside radius
        
        # Set up physics with agents
        self.physics._entities["agents"] = [agent1, agent2, agent3]
        self.physics.spatial_index.set_references([agent1, agent2, agent3], [])
        
        # Test nearby query
        nearby = self.physics.get_nearby_entities((50, 50), 5.0, "agents")
        
        # Should find agent1 and agent2 (within radius ~3.5)
        self.assertEqual(len(nearby), 2)
        self.assertIn(agent1, nearby)
        self.assertIn(agent2, nearby)
        self.assertNotIn(agent3, nearby)

    def test_get_nearby_entities_resources(self):
        """Test getting nearby resources."""
        # Create mock resources
        resource1 = Mock()
        resource1.position = (25, 25)
        resource2 = Mock()
        resource2.position = (30, 30)
        resource3 = Mock()
        resource3.position = (100, 100)  # Outside radius
        
        # Set up physics with resources
        self.physics._entities["resources"] = [resource1, resource2, resource3]
        self.physics.spatial_index.set_references([], [resource1, resource2, resource3])
        
        # Test nearby query
        nearby = self.physics.get_nearby_entities((25, 25), 10.0, "resources")
        
        # Should find resource1 and resource2 (within radius ~7.1)
        self.assertEqual(len(nearby), 2)
        self.assertIn(resource1, nearby)
        self.assertIn(resource2, nearby)
        self.assertNotIn(resource3, nearby)

    def test_get_nearby_entities_empty(self):
        """Test getting nearby entities when none exist."""
        # Test with no entities
        nearby = self.physics.get_nearby_entities((50, 50), 10.0, "agents")
        self.assertEqual(len(nearby), 0)
        
        # Test with invalid position
        nearby = self.physics.get_nearby_entities((-1, -1), 10.0, "agents")
        self.assertEqual(len(nearby), 0)

    def test_compute_distance_euclidean(self):
        """Test Euclidean distance computation."""
        test_cases = [
            ((0, 0), (3, 4), 5.0),      # 3-4-5 triangle
            ((0, 0), (0, 0), 0.0),      # Same point
            ((1, 1), (1, 1), 0.0),      # Same point (non-zero)
            ((0, 0), (1, 0), 1.0),      # Horizontal distance
            ((0, 0), (0, 1), 1.0),      # Vertical distance
            ((0, 0), (1, 1), math.sqrt(2)),  # Diagonal distance
        ]
        
        for pos1, pos2, expected in test_cases:
            with self.subTest(pos1=pos1, pos2=pos2):
                distance = self.physics.compute_distance(pos1, pos2)
                self.assertAlmostEqual(distance, expected, places=10)

    def test_get_state_shape(self):
        """Test state shape retrieval."""
        shape = self.physics.get_state_shape()
        self.assertEqual(shape, (self.width, self.height))

    def test_get_observation_space(self):
        """Test observation space definition."""
        obs_space = self.physics.get_observation_space("test_agent")
        
        # Should be a Box space
        self.assertIsInstance(obs_space, spaces.Box)
        
        # Should have correct shape (NUM_CHANNELS, S, S) where S = 2*R + 1
        expected_R = self.physics.observation_config.R
        expected_S = 2 * expected_R + 1
        expected_shape = (self.physics._observation_space.shape[0], expected_S, expected_S)
        
        self.assertEqual(obs_space.shape, expected_shape)
        self.assertEqual(obs_space.dtype, np.float32)

    def test_sample_position_within_bounds(self):
        """Test that sampled positions are within bounds."""
        # Set seed for deterministic testing
        random.seed(42)
        np.random.seed(42)
        
        for _ in range(100):
            position = self.physics.sample_position()
            self.assertTrue(self.physics.validate_position(position))
            self.assertGreaterEqual(position[0], 0)
            self.assertLess(position[0], self.width)
            self.assertGreaterEqual(position[1], 0)
            self.assertLess(position[1], self.height)

    def test_sample_position_distribution(self):
        """Test that sampled positions have reasonable distribution."""
        # Set seed for deterministic testing
        random.seed(42)
        np.random.seed(42)
        
        positions = [self.physics.sample_position() for _ in range(1000)]
        
        # Check that positions are distributed across the space
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Mean should be roughly in the center
        self.assertAlmostEqual(np.mean(x_coords), self.width / 2, delta=5)
        self.assertAlmostEqual(np.mean(y_coords), self.height / 2, delta=5)
        
        # Should have reasonable spread
        self.assertGreater(np.std(x_coords), 10)
        self.assertGreater(np.std(y_coords), 10)

    def test_update_and_reset(self):
        """Test physics update and reset methods."""
        # Test update (should not raise)
        self.physics.update(dt=1.0)
        
        # Test reset
        self.physics._entities["agents"] = [Mock()]
        self.physics._entities["resources"] = [Mock()]
        
        self.physics.reset()
        
        # Entities should be cleared
        self.assertEqual(len(self.physics._entities["agents"]), 0)
        self.assertEqual(len(self.physics._entities["resources"]), 0)

    def test_get_config(self):
        """Test configuration retrieval."""
        config = self.physics.get_config()
        
        # Check required fields
        self.assertEqual(config["type"], "grid_2d")
        self.assertEqual(config["width"], self.width)
        self.assertEqual(config["height"], self.height)
        self.assertEqual(config["seed"], 42)
        
        # Check observation config
        self.assertIn("observation_config", config)
        self.assertEqual(config["observation_config"]["R"], self.physics.observation_config.R)
        
        # Check spatial index config
        self.assertIn("spatial_index", config)

    def test_get_bounds(self):
        """Test bounds retrieval."""
        bounds = self.physics.get_bounds()
        
        self.assertEqual(len(bounds), 2)
        min_bounds, max_bounds = bounds
        
        self.assertEqual(min_bounds, (0.0, 0.0))
        self.assertEqual(max_bounds, (float(self.width), float(self.height)))

    def test_spatial_index_integration(self):
        """Test integration with spatial index."""
        # Test that spatial index is properly initialized
        self.assertIsNotNone(self.physics.spatial_index)
        self.assertEqual(self.physics.spatial_index.width, self.width)
        self.assertEqual(self.physics.spatial_index.height, self.height)

    def test_quadtree_indices(self):
        """Test quadtree indices functionality."""
        # Test enabling quadtree indices
        self.physics._enable_quadtree_indices()
        
        # Should not raise an error
        self.assertIsNotNone(self.physics.spatial_index)

    def test_spatial_hash_indices(self):
        """Test spatial hash indices functionality."""
        # Test enabling spatial hash indices
        self.physics._enable_spatial_hash_indices(cell_size=10.0)
        
        # Should not raise an error
        self.assertIsNotNone(self.physics.spatial_index)

    def test_set_entity_references(self):
        """Test setting entity references for spatial indexing."""
        agents = [Mock(), Mock()]
        resources = [Mock(), Mock()]
        
        # Set entity references
        self.physics.set_entity_references(agents, resources)
        
        # Check that entities are stored
        self.assertEqual(self.physics._entities["agents"], agents)
        self.assertEqual(self.physics._entities["resources"], resources)

    def test_mark_positions_dirty(self):
        """Test marking positions as dirty."""
        # Should not raise an error
        self.physics.mark_positions_dirty()

    def test_process_batch_spatial_updates(self):
        """Test processing batch spatial updates."""
        # Should not raise an error
        self.physics.process_batch_spatial_updates(force=True)

    def test_get_spatial_performance_stats(self):
        """Test getting spatial performance statistics."""
        stats = self.physics.get_spatial_performance_stats()
        
        # Check required fields
        self.assertIn("width", stats)
        self.assertIn("height", stats)
        self.assertIn("total_entities", stats)
        self.assertIn("agents_count", stats)
        self.assertIn("resources_count", stats)
        
        self.assertEqual(stats["width"], self.width)
        self.assertEqual(stats["height"], self.height)

    def test_get_nearest_resource(self):
        """Test getting nearest resource."""
        # Create mock resources
        resource1 = Mock()
        resource1.position = (10, 10)
        resource2 = Mock()
        resource2.position = (20, 20)
        resource3 = Mock()
        resource3.position = (50, 50)
        
        self.physics._entities["resources"] = [resource1, resource2, resource3]
        
        # Test nearest resource query
        nearest = self.physics.get_nearest_resource((15, 15))
        
        # Should find resource2 (closest to (15, 15))
        self.assertEqual(nearest, resource2)
        
        # Test with no resources
        self.physics._entities["resources"] = []
        nearest = self.physics.get_nearest_resource((15, 15))
        self.assertIsNone(nearest)

    def test_initialization_with_config(self):
        """Test initialization with custom configuration."""
        # Create custom observation config
        obs_config = ObservationConfig()
        obs_config.R = 10
        obs_config.dtype = "float64"
        
        # Create physics with custom config
        physics = Grid2DPhysics(
            width=50,
            height=50,
            observation_config=obs_config,
            seed=123
        )
        
        # Check that config is applied
        self.assertEqual(physics.observation_config.R, 10)
        self.assertEqual(physics.width, 50)
        self.assertEqual(physics.height, 50)
        self.assertEqual(physics.seed, 123)

    def test_initialization_with_spatial_config(self):
        """Test initialization with spatial configuration."""
        # Create spatial config
        spatial_config = Mock()
        spatial_config.enable_batch_updates = True
        spatial_config.region_size = 25.0
        spatial_config.max_batch_size = 50
        spatial_config.dirty_region_batch_size = 5
        spatial_config.enable_quadtree_indices = True
        spatial_config.enable_spatial_hash_indices = True
        spatial_config.spatial_hash_cell_size = 10.0
        
        # Create physics with spatial config
        physics = Grid2DPhysics(
            width=100,
            height=100,
            spatial_config=spatial_config
        )
        
        # Check that spatial index is properly configured
        self.assertIsNotNone(physics.spatial_index)
        self.assertTrue(physics.spatial_index.enable_batch_updates)


class TestGrid2DPhysicsIntegration(unittest.TestCase):
    """Integration tests for Grid2DPhysics with real entities."""

    def setUp(self):
        """Set up test fixtures."""
        self.physics = Grid2DPhysics(width=100, height=100, seed=42)

    def test_full_entity_lifecycle(self):
        """Test complete entity lifecycle with physics engine."""
        # Create mock entities
        agents = []
        resources = []
        
        for i in range(5):
            agent = Mock()
            agent.position = (i * 10, i * 10)
            agents.append(agent)
            
            resource = Mock()
            resource.position = (i * 10 + 5, i * 10 + 5)
            resources.append(resource)
        
        # Set entity references
        self.physics.set_entity_references(agents, resources)
        
        # Test spatial queries
        nearby_agents = self.physics.get_nearby_entities((0, 0), 15.0, "agents")
        self.assertEqual(len(nearby_agents), 2)  # Should find first two agents
        
        nearby_resources = self.physics.get_nearby_entities((0, 0), 15.0, "resources")
        self.assertEqual(len(nearby_resources), 1)  # Should find first resource
        
        # Test nearest resource
        nearest = self.physics.get_nearest_resource((0, 0))
        self.assertEqual(nearest, resources[0])
        
        # Test position validation
        for agent in agents:
            self.assertTrue(self.physics.validate_position(agent.position))
        
        # Test distance computation
        distance = self.physics.compute_distance(agents[0].position, agents[1].position)
        expected_distance = math.sqrt(200)  # sqrt((10-0)^2 + (10-0)^2)
        self.assertAlmostEqual(distance, expected_distance, places=10)


if __name__ == "__main__":
    unittest.main()
