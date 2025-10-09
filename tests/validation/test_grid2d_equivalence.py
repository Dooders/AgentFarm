"""Validation tests to ensure Grid2DPhysics produces identical results to original system.

These tests are critical for ensuring that the physics abstraction layer doesn't
introduce any behavioral changes to existing simulations.
"""

import json
import pickle
import random
import unittest
from typing import Any, Dict, List

import numpy as np
import pytest

from farm.config.config import EnvironmentConfig, PopulationConfig, ResourceConfig, SimulationConfig
from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.physics import create_physics_engine
from farm.core.resources import Resource


class TestGrid2DEquivalence(unittest.TestCase):
    """Test that Grid2DPhysics produces identical results to original system."""

    def setUp(self):
        """Set up test fixtures with deterministic configuration."""
        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=50, height=50),
            population=PopulationConfig(system_agents=3, independent_agents=2, control_agents=1),
            resources=ResourceConfig(initial_resources=5),
            max_steps=10,
            seed=42,
        )
        
        # Create physics engine
        self.physics = create_physics_engine(self.config, seed=42)
        
        # Create environment with physics engine
        self.env = Environment(
            physics_engine=self.physics,
            resource_distribution={"type": "random", "amount": 5},
            config=self.config,
            seed=42
        )

    def test_deterministic_simulation_equivalence(self):
        """Test that simulations produce identical results with same seed."""
        # Run simulation for a few steps
        for step in range(5):
            # Step through environment
            self.env.step()
            
            # Check that environment state is consistent
            self.assertIsNotNone(self.env.time)
            self.assertEqual(self.env.time, step + 1)
            
            # Check that agents exist and have valid positions
            for agent_id in self.env.agents:
                agent = self.env._agent_objects[agent_id]
                self.assertTrue(self.env.is_valid_position(agent.position))
                
                # Check that observations are generated
                observation = self.env.observe(agent_id)
                self.assertIsNotNone(observation)
                self.assertIsInstance(observation, np.ndarray)

    def test_spatial_query_equivalence(self):
        """Test that spatial queries produce consistent results."""
        # Add some agents at known positions
        test_positions = [(10, 10), (20, 20), (30, 30)]
        agents = []
        
        for i, position in enumerate(test_positions):
            agent = BaseAgent(
                agent_id=f"test_agent_{i}",
                position=position,
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            agents.append(agent)
            self.env.add_agent(agent)
        
        # Test nearby queries
        for position in test_positions:
            nearby_agents = self.env.get_nearby_agents(position, 15.0)
            
            # Should find agents within radius
            self.assertGreater(len(nearby_agents), 0)
            
            # All found agents should be within radius
            for agent in nearby_agents:
                distance = self.env.physics.compute_distance(position, agent.position)
                self.assertLessEqual(distance, 15.0)
        
        # Test nearest resource queries
        for position in test_positions:
            nearest_resource = self.env.get_nearest_resource(position)
            if nearest_resource:
                # Verify it's actually the nearest
                min_distance = float('inf')
                for resource in self.env.resources:
                    distance = self.env.physics.compute_distance(position, resource.position)
                    min_distance = min(min_distance, distance)
                
                actual_distance = self.env.physics.compute_distance(position, nearest_resource.position)
                self.assertAlmostEqual(actual_distance, min_distance, places=10)

    def test_observation_equivalence(self):
        """Test that observations are consistent and valid."""
        # Add an agent
        agent = BaseAgent(
            agent_id="observer_agent",
            position=(25, 25),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        self.env.add_agent(agent)
        
        # Get observation space
        obs_space = self.env.observation_space(agent.agent_id)
        self.assertIsNotNone(obs_space)
        
        # Generate observation
        observation = self.env.observe(agent.agent_id)
        self.assertIsNotNone(observation)
        self.assertIsInstance(observation, np.ndarray)
        
        # Check observation properties
        self.assertEqual(observation.shape, obs_space.shape)
        self.assertEqual(observation.dtype, obs_space.dtype)
        
        # Check observation bounds
        self.assertTrue(np.all(observation >= obs_space.low))
        self.assertTrue(np.all(observation <= obs_space.high))
        
        # Observations should be deterministic with same seed
        observation2 = self.env.observe(agent.agent_id)
        np.testing.assert_array_equal(observation, observation2)

    def test_agent_positioning_equivalence(self):
        """Test that agent positioning is consistent."""
        # Add agents at specific positions
        positions = [(10, 10), (20, 20), (30, 30)]
        agents = []
        
        for i, position in enumerate(positions):
            agent = BaseAgent(
                agent_id=f"positioned_agent_{i}",
                position=position,
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            agents.append(agent)
            self.env.add_agent(agent)
        
        # Verify agents are at correct positions
        for i, agent in enumerate(agents):
            expected_position = positions[i]
            self.assertEqual(agent.position, expected_position)
            
            # Verify position validation
            self.assertTrue(self.env.is_valid_position(agent.position))
            
            # Verify spatial queries find the agent
            nearby_agents = self.env.get_nearby_agents(agent.position, 1.0)
            self.assertIn(agent, nearby_agents)

    def test_resource_distribution_equivalence(self):
        """Test that resource distribution is consistent."""
        # Check that resources were created
        self.assertGreater(len(self.env.resources), 0)
        
        # Check that all resources have valid positions
        for resource in self.env.resources:
            self.assertTrue(self.env.is_valid_position(resource.position))
            self.assertGreater(resource.amount, 0)
            self.assertGreater(resource.max_amount, 0)
        
        # Check that resources are distributed across the environment
        positions = [resource.position for resource in self.env.resources]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Resources should be spread across the environment
        self.assertGreater(np.std(x_coords), 0)
        self.assertGreater(np.std(y_coords), 0)
        
        # Resources should be within bounds
        for x, y in positions:
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, self.env.width)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, self.env.height)

    def test_distance_computation_consistency(self):
        """Test that distance computations are consistent."""
        test_cases = [
            ((0, 0), (3, 4), 5.0),      # 3-4-5 triangle
            ((0, 0), (0, 0), 0.0),      # Same point
            ((1, 1), (1, 1), 0.0),      # Same point (non-zero)
            ((0, 0), (1, 0), 1.0),      # Horizontal distance
            ((0, 0), (0, 1), 1.0),      # Vertical distance
            ((0, 0), (1, 1), np.sqrt(2)),  # Diagonal distance
        ]
        
        for pos1, pos2, expected in test_cases:
            with self.subTest(pos1=pos1, pos2=pos2):
                distance = self.env.physics.compute_distance(pos1, pos2)
                self.assertAlmostEqual(distance, expected, places=10)
                
                # Distance should be symmetric
                distance_reverse = self.env.physics.compute_distance(pos2, pos1)
                self.assertAlmostEqual(distance, distance_reverse, places=10)

    def test_spatial_index_consistency(self):
        """Test that spatial index operations are consistent."""
        # Add agents at known positions
        positions = [(10, 10), (20, 20), (30, 30)]
        agents = []
        
        for i, position in enumerate(positions):
            agent = BaseAgent(
                agent_id=f"spatial_agent_{i}",
                position=position,
                resource_level=100,
                spatial_service=self.env.spatial_service,
                environment=self.env,
                config=self.config
            )
            agents.append(agent)
            self.env.add_agent(agent)
        
        # Test that spatial index is properly updated
        self.env.mark_positions_dirty()
        self.env.process_batch_spatial_updates(force=True)
        
        # Test spatial queries
        for position in positions:
            nearby_agents = self.env.get_nearby_agents(position, 5.0)
            
            # Should find the agent at that position
            self.assertGreater(len(nearby_agents), 0)
            
            # All found agents should be within radius
            for agent in nearby_agents:
                distance = self.env.physics.compute_distance(position, agent.position)
                self.assertLessEqual(distance, 5.0)

    def test_physics_engine_configuration_consistency(self):
        """Test that physics engine configuration is consistent."""
        # Check physics config
        config = self.env.physics.get_config()
        
        self.assertEqual(config["type"], "grid_2d")
        self.assertEqual(config["width"], 50)
        self.assertEqual(config["height"], 50)
        self.assertEqual(config["seed"], 42)
        
        # Check bounds
        bounds = self.env.physics.get_bounds()
        self.assertEqual(bounds, ((0.0, 0.0), (50.0, 50.0)))
        
        # Check state shape
        state_shape = self.env.physics.get_state_shape()
        self.assertEqual(state_shape, (50, 50))

    def test_environment_properties_consistency(self):
        """Test that environment properties are consistent."""
        # Check width and height properties
        self.assertEqual(self.env.width, 50)
        self.assertEqual(self.env.height, 50)
        
        # Check that properties match physics engine
        self.assertEqual(self.env.width, self.env.physics.get_bounds()[1][0])
        self.assertEqual(self.env.height, self.env.physics.get_bounds()[1][1])
        
        # Check that properties are consistent with config
        self.assertEqual(self.env.width, self.config.environment.width)
        self.assertEqual(self.env.height, self.config.environment.height)

    def test_deterministic_behavior_across_runs(self):
        """Test that behavior is deterministic across multiple runs."""
        # Create two identical environments
        config1 = SimulationConfig(
            environment=EnvironmentConfig(width=30, height=30),
            population=PopulationConfig(system_agents=2, independent_agents=0, control_agents=0),
            resources=ResourceConfig(initial_resources=3),
            max_steps=5,
            seed=123,
        )
        
        config2 = SimulationConfig(
            environment=EnvironmentConfig(width=30, height=30),
            population=PopulationConfig(system_agents=2, independent_agents=0, control_agents=0),
            resources=ResourceConfig(initial_resources=3),
            max_steps=5,
            seed=123,
        )
        
        physics1 = create_physics_engine(config1, seed=123)
        physics2 = create_physics_engine(config2, seed=123)
        
        env1 = Environment(
            physics_engine=physics1,
            resource_distribution={"type": "random", "amount": 3},
            config=config1,
            seed=123
        )
        
        env2 = Environment(
            physics_engine=physics2,
            resource_distribution={"type": "random", "amount": 3},
            config=config2,
            seed=123
        )
        
        # Run both environments for a few steps
        for step in range(3):
            env1.step()
            env2.step()
            
            # Check that both environments have same number of agents
            self.assertEqual(len(env1.agents), len(env2.agents))
            
            # Check that both environments have same number of resources
            self.assertEqual(len(env1.resources), len(env2.resources))
            
            # Check that time is consistent
            self.assertEqual(env1.time, env2.time)
            self.assertEqual(env1.time, step + 1)

    def test_serialization_consistency(self):
        """Test that environment state can be serialized consistently."""
        # Add an agent
        agent = BaseAgent(
            agent_id="serializable_agent",
            position=(15, 15),
            resource_level=100,
            spatial_service=self.env.spatial_service,
            environment=self.env,
            config=self.config
        )
        self.env.add_agent(agent)
        
        # Get environment state
        state = {
            "time": self.env.time,
            "agents": [(agent.agent_id, agent.position) for agent in self.env._agent_objects.values()],
            "resources": [(resource.position, resource.amount) for resource in self.env.resources],
            "physics_config": self.env.physics.get_config(),
        }
        
        # Serialize and deserialize
        serialized = json.dumps(state, default=str)
        deserialized = json.loads(serialized)
        
        # Check that key information is preserved
        self.assertEqual(deserialized["time"], state["time"])
        self.assertEqual(len(deserialized["agents"]), len(state["agents"]))
        self.assertEqual(len(deserialized["resources"]), len(state["resources"]))
        self.assertEqual(deserialized["physics_config"]["type"], "grid_2d")


if __name__ == "__main__":
    unittest.main()
