"""Integration tests for physics engine with Environment.

This module tests the integration between physics engines and the Environment class,
ensuring that all spatial operations work correctly through the physics abstraction.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest

from farm.config.config import EnvironmentConfig, PopulationConfig, ResourceConfig, SimulationConfig
from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.physics import create_physics_engine
from farm.core.resources import Resource


class TestPhysicsEnvironmentIntegration(unittest.TestCase):
    """Test integration between physics engines and Environment."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=100, height=100),
            population=PopulationConfig(system_agents=2, independent_agents=1, control_agents=1),
            resources=ResourceConfig(initial_resources=10),
            max_steps=100,
            seed=42,
        )
        
        self.physics = create_physics_engine(self.config, seed=42)
        self.env = Environment(
            physics_engine=self.physics,
            resource_distribution={"type": "random", "amount": 10},
            config=self.config,
            seed=42
        )

    def test_environment_with_grid2d_physics(self):
        """Test basic integration with Grid2D physics."""
        # Check that physics engine is properly set
        self.assertIsNotNone(self.env.physics)
        self.assertEqual(self.env.physics.get_config()["type"], "grid_2d")
        
        # Check that width/height properties work
        self.assertEqual(self.env.width, 100)
        self.assertEqual(self.env.height, 100)
        
        # Check that spatial service is properly configured
        self.assertIsNotNone(self.env.spatial_service)

    def test_agent_movement_with_physics(self):
        """Test that agents can move and physics handles it correctly."""
        # Add an agent
        agent = BaseAgent(
            agent_id="test_agent",
            position=(50, 50),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        
        self.env.add_agent(agent)
        
        # Test position validation
        self.assertTrue(self.env.is_valid_position((50, 50)))
        self.assertFalse(self.env.is_valid_position((-1, -1)))
        self.assertFalse(self.env.is_valid_position((100, 100)))
        
        # Test agent movement
        new_position = (60, 60)
        if self.env.is_valid_position(new_position):
            agent.position = new_position
            self.env.mark_positions_dirty()
            
            # Verify position is updated
            self.assertEqual(agent.position, new_position)

    def test_resource_queries_with_physics(self):
        """Test resource finding through physics engine."""
        # Check that resources were created
        self.assertGreater(len(self.env.resources), 0)
        
        # Test nearby resource queries
        position = (50, 50)
        nearby_resources = self.env.get_nearby_resources(position, 20.0)
        
        # Should find some resources within radius
        self.assertIsInstance(nearby_resources, list)
        
        # Test nearest resource
        nearest = self.env.get_nearest_resource(position)
        if nearest:
            self.assertIsInstance(nearest, Resource)
            self.assertTrue(self.env.is_valid_position(nearest.position))

    def test_combat_with_physics(self):
        """Test spatial combat mechanics through physics."""
        # Add two agents close to each other
        agent1 = BaseAgent(
            agent_id="agent1",
            position=(50, 50),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        
        agent2 = BaseAgent(
            agent_id="agent2",
            position=(52, 52),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        
        self.env.add_agent(agent1)
        self.env.add_agent(agent2)
        
        # Test nearby agent queries
        nearby_agents = self.env.get_nearby_agents((50, 50), 5.0)
        
        # Should find both agents (they're close)
        self.assertEqual(len(nearby_agents), 2)
        self.assertIn(agent1, nearby_agents)
        self.assertIn(agent2, nearby_agents)
        
        # Test distance computation
        distance = self.env.physics.compute_distance(agent1.position, agent2.position)
        expected_distance = np.sqrt(8)  # sqrt((52-50)^2 + (52-50)^2)
        self.assertAlmostEqual(distance, expected_distance, places=10)

    def test_reproduction_with_physics(self):
        """Test spatial reproduction mechanics."""
        # Add an agent
        agent = BaseAgent(
            agent_id="parent_agent",
            position=(50, 50),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        
        self.env.add_agent(agent)
        
        # Test that agent can find nearby resources for reproduction
        nearby_resources = self.env.get_nearby_resources(agent.position, 10.0)
        
        # Should find some resources
        self.assertGreater(len(nearby_resources), 0)
        
        # Test that agent can find nearby agents for reproduction
        nearby_agents = self.env.get_nearby_agents(agent.position, 10.0)
        
        # Should find itself
        self.assertEqual(len(nearby_agents), 1)
        self.assertIn(agent, nearby_agents)

    def test_observation_generation_with_physics(self):
        """Test observation generation through physics engine."""
        # Add an agent
        agent = BaseAgent(
            agent_id="observer_agent",
            position=(50, 50),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        
        self.env.add_agent(agent)
        
        # Test observation space
        obs_space = self.env.observation_space(agent.agent_id)
        self.assertIsNotNone(obs_space)
        
        # Test observation generation
        observation = self.env.observe(agent.agent_id)
        self.assertIsNotNone(observation)
        self.assertIsInstance(observation, np.ndarray)
        
        # Check observation shape matches space
        self.assertEqual(observation.shape, obs_space.shape)

    def test_spatial_index_updates(self):
        """Test that spatial index updates work through physics."""
        # Add an agent
        agent = BaseAgent(
            agent_id="moving_agent",
            position=(10, 10),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        
        self.env.add_agent(agent)
        
        # Move agent
        agent.position = (20, 20)
        self.env.mark_positions_dirty()
        
        # Process batch updates
        self.env.process_batch_spatial_updates(force=True)
        
        # Verify agent is found at new position
        nearby_agents = self.env.get_nearby_agents((20, 20), 5.0)
        self.assertIn(agent, nearby_agents)

    def test_physics_performance_stats(self):
        """Test physics performance statistics."""
        # Add some agents and resources
        for i in range(3):
            agent = BaseAgent(
                agent_id=f"agent_{i}",
                position=(i * 10, i * 10),
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            self.env.add_agent(agent)
        
        # Get performance stats
        stats = self.env.get_spatial_performance_stats()
        
        # Check that stats are returned
        self.assertIsInstance(stats, dict)
        self.assertIn("agents_count", stats)
        self.assertIn("resources_count", stats)
        self.assertIn("width", stats)
        self.assertIn("height", stats)

    def test_physics_engine_configuration(self):
        """Test that physics engine configuration is properly applied."""
        # Check physics config
        physics_config = self.env.physics.get_config()
        
        self.assertEqual(physics_config["type"], "grid_2d")
        self.assertEqual(physics_config["width"], 100)
        self.assertEqual(physics_config["height"], 100)
        self.assertEqual(physics_config["seed"], 42)
        
        # Check bounds
        bounds = self.env.physics.get_bounds()
        self.assertEqual(bounds, ((0.0, 0.0), (100.0, 100.0)))

    def test_environment_reset_with_physics(self):
        """Test environment reset with physics engine."""
        # Add some agents
        for i in range(2):
            agent = BaseAgent(
                agent_id=f"agent_{i}",
                position=(i * 10, i * 10),
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            self.env.add_agent(agent)
        
        # Reset environment
        self.env.reset()
        
        # Check that agents are cleared
        self.assertEqual(len(self.env.agents), 0)
        
        # Check that physics is reset
        self.assertEqual(len(self.env.physics._entities["agents"]), 0)

    def test_resource_manager_integration(self):
        """Test that ResourceManager works with physics engine."""
        # Check that resource manager is properly initialized
        self.assertIsNotNone(self.env.resource_manager)
        self.assertIsNotNone(self.env.resource_manager.physics)
        
        # Check that resources were created
        self.assertGreater(len(self.env.resources), 0)
        
        # Test resource queries through resource manager
        position = (50, 50)
        nearby_resources = self.env.resource_manager.get_nearby_resources(position, 20.0)
        self.assertIsInstance(nearby_resources, list)
        
        nearest_resource = self.env.resource_manager.get_nearest_resource(position)
        if nearest_resource:
            self.assertIsInstance(nearest_resource, Resource)


class TestPhysicsEnvironmentEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=10, height=10),  # Small environment
            population=PopulationConfig(system_agents=1, independent_agents=0, control_agents=0),
            resources=ResourceConfig(initial_resources=1),
            max_steps=10,
            seed=42,
        )
        
        self.physics = create_physics_engine(self.config, seed=42)
        self.env = Environment(
            physics_engine=self.physics,
            resource_distribution={"type": "random", "amount": 1},
            config=self.config,
            seed=42
        )

    def test_edge_case_positions(self):
        """Test edge case positions."""
        # Test boundary positions
        boundary_positions = [
            (0, 0),      # Corner
            (9, 9),      # Other corner
            (0, 5),      # Edge
            (5, 0),      # Edge
            (9, 5),      # Edge
            (5, 9),      # Edge
        ]
        
        for position in boundary_positions:
            with self.subTest(position=position):
                self.assertTrue(self.env.is_valid_position(position))

    def test_invalid_position_handling(self):
        """Test handling of invalid positions."""
        invalid_positions = [
            (-1, 0),
            (0, -1),
            (10, 0),
            (0, 10),
            (10, 10),
            (-1, -1),
        ]
        
        for position in invalid_positions:
            with self.subTest(position=position):
                self.assertFalse(self.env.is_valid_position(position))
                
                # Nearby queries should return empty for invalid positions
                nearby = self.env.get_nearby_agents(position, 1.0)
                self.assertEqual(len(nearby), 0)

    def test_empty_environment_queries(self):
        """Test queries in empty environment."""
        # Remove all agents and resources
        self.env.agents.clear()
        self.env.resources.clear()
        
        # Test queries should return empty results
        nearby_agents = self.env.get_nearby_agents((5, 5), 10.0)
        self.assertEqual(len(nearby_agents), 0)
        
        nearby_resources = self.env.get_nearby_resources((5, 5), 10.0)
        self.assertEqual(len(nearby_resources), 0)
        
        nearest_resource = self.env.get_nearest_resource((5, 5))
        self.assertIsNone(nearest_resource)

    def test_large_radius_queries(self):
        """Test queries with very large radius."""
        # Add an agent
        agent = BaseAgent(
            agent_id="test_agent",
            position=(5, 5),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        self.env.add_agent(agent)
        
        # Test with radius larger than environment
        nearby_agents = self.env.get_nearby_agents((5, 5), 100.0)
        self.assertEqual(len(nearby_agents), 1)
        self.assertIn(agent, nearby_agents)

    def test_zero_radius_queries(self):
        """Test queries with zero radius."""
        # Add an agent
        agent = BaseAgent(
            agent_id="test_agent",
            position=(5, 5),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        self.env.add_agent(agent)
        
        # Test with zero radius
        nearby_agents = self.env.get_nearby_agents((5, 5), 0.0)
        self.assertEqual(len(nearby_agents), 1)  # Should find agent at exact position
        
        # Test with very small radius
        nearby_agents = self.env.get_nearby_agents((5.1, 5.1), 0.1)
        self.assertEqual(len(nearby_agents), 1)  # Should find nearby agent


if __name__ == "__main__":
    unittest.main()
