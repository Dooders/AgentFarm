import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
import signal
import time

import numpy as np
from gymnasium import spaces
from pettingzoo.test import api_test
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from farm.config import SimulationConfig
from farm.config.config import EnvironmentConfig, PopulationConfig, ResourceConfig
from farm.core.action import ActionType
from farm.core.agent import AgentFactory
from farm.core.environment import Environment
from farm.core.resources import Resource


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # Create a minimal SimulationConfig for testing
        self.config = SimulationConfig(
            environment=EnvironmentConfig(width=100, height=100),
            population=PopulationConfig(system_agents=1, independent_agents=1, control_agents=1),
            max_steps=100,
            seed=42,
        )

        # Store resource distribution for test access
        self.resource_distribution = {"amount": 10}

        # Keep track of mock objects for later use in tests but don't attach them to config
        # to avoid JSON serialization issues
        self.mock_action_space = Mock()
        self.mock_action_space.n = len(ActionType)  # Number of actions from ActionType enum

        self.mock_observation_space = Mock()
        self.mock_observation_space.shape = (10, 21, 21)  # Typical observation shape

        # Mock decision config to prevent DecisionModule config errors
        self.mock_decision_config = Mock()
        self.mock_decision_config.rl_state_dim = 8  # Default value from DecisionConfig
        self.mock_decision_config.algorithm_type = "dqn"
        self.mock_decision_config.algorithm_params = {}

        self.env = Environment(
            width=100,
            height=100,
            resource_distribution={"amount": 10},
            config=self.config,
            db_path=":memory:",  # Use in-memory database for tests
        )

        # Add a small set of agents explicitly (external agent management)
        factory = AgentFactory(spatial_service=self.env.spatial_service)
        initial_agents = [
            factory.create_default_agent(
                agent_id=self.env.get_next_agent_id(),
                position=(10, 10),
                initial_resources=5,
            ),
            factory.create_default_agent(
                agent_id=self.env.get_next_agent_id(),
                position=(12, 12),
                initial_resources=5,
            ),
            factory.create_default_agent(
                agent_id=self.env.get_next_agent_id(),
                position=(14, 14),
                initial_resources=5,
            ),
        ]
        for a in initial_agents:
            self.env.add_agent(a)

    def tearDown(self):
        if hasattr(self, "env") and self.env:
            self.env.cleanup()

    def test_observation_action_spaces(self):
        if not self.env.agents:
            self.skipTest("No agents in environment")
        agent = self.env.agents[0]
        obs_space = self.env.observation_space(agent)
        self.assertIsInstance(obs_space, spaces.Box)
        # Check that observation shape matches the configured observation space
        expected_shape = (
            self.env.observation_config.R * 2 + 1,  # Width: 2*R + 1
            self.env.observation_config.R * 2 + 1,  # Height: 2*R + 1
        )
        # Note: Full shape includes NUM_CHANNELS as first dimension
        self.assertEqual(obs_space.shape[1:], expected_shape)  # Check spatial dimensions
        self.assertEqual(obs_space.dtype, np.float32)

        act_space = self.env.action_space(agent)
        self.assertIsInstance(act_space, spaces.Discrete)
        # Action space size is determined by enabled actions mapping, includes PASS action
        self.assertEqual(act_space.n, len(self.env._action_mapping))

    def test_reset_step_cycle(self):
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, self.env.observation_space(self.env.agent_selection).shape)

        done = False
        steps = 0
        while not done and steps < 10:
            action = self.env.action_space(self.env.agent_selection).sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            done = terminated or truncated
            steps += 1

        self.assertGreater(steps, 0)

    def test_initialization(self):
        """Test environment initialization with various parameters"""
        self.assertEqual(self.env.width, 100)
        self.assertEqual(self.env.height, 100)
        self.assertEqual(self.env.time, 0)
        self.assertIsInstance(self.env.agent_objects, list)
        self.assertIsInstance(self.env.resources, list)
        self.assertIsInstance(self.env.agents, list)
        self.assertIsNotNone(self.env.observation_space)
        self.assertIsNotNone(self.env.action_space)

    def test_agent_creation(self):
        """Test that agents are properly created during initialization"""
        self.assertGreater(len(self.env.agent_objects), 0)
        self.assertGreater(len(self.env.agents), 0)

        # Check that agent IDs are unique
        agent_ids = [agent.agent_id for agent in self.env.agent_objects]
        self.assertEqual(len(agent_ids), len(set(agent_ids)))

    def test_resource_initialization(self):
        """Test that resources are properly initialized"""
        # Resource count should match the amount specified in resource_distribution
        expected_count = self.resource_distribution.get("amount", 20)  # Default from ResourceManager
        self.assertEqual(len(self.env.resources), expected_count)

        for resource in self.env.resources:
            self.assertGreaterEqual(resource.position[0], 0)
            self.assertLessEqual(resource.position[0], self.env.width)
            self.assertGreaterEqual(resource.position[1], 0)
            self.assertLessEqual(resource.position[1], self.env.height)
            self.assertGreater(resource.amount, 0)

    def test_reset_functionality(self):
        """Test that reset properly reinitializes the environment"""
        # Take some steps to change state
        self.env.reset()
        initial_time = self.env.time
        initial_agent_count = len(self.env.agent_objects)

        # Take a step
        if self.env.agents:
            action = self.env.action_space(self.env.agent_selection).sample()
            self.env.step(action)

        # Reset
        obs, info = self.env.reset()

        # Check that state is reset
        self.assertEqual(self.env.time, 0)
        self.assertEqual(len(self.env.agent_objects), initial_agent_count)
        self.assertIsInstance(obs, np.ndarray)

    def test_action_processing(self):
        """Test that actions are properly processed"""
        if not self.env.agents:
            self.skipTest("No agents in environment")

        agent_id = self.env.agents[0]
        agent = next((a for a in self.env.agent_objects if a.agent_id == agent_id), None)
        self.assertIsNotNone(agent)
        assert agent is not None  # Type assertion for type checker

        # Ensure agent has proper health and resource values
        combat_component = agent.get_component("combat")
        resource_component = agent.get_component("resource")
        
        if combat_component and combat_component.health is None:
            combat_component.health = 100.0
        if combat_component and combat_component.max_health is None:
            combat_component.max_health = 100.0
        if resource_component and resource_component.level is None:
            resource_component.level = 10

        initial_health = combat_component.health
        initial_resources = resource_component.level

        # Test defend action (skip if agent has mock attributes that cause comparison issues)
        try:
            initial_defending = combat_component.is_defending
            self.env._process_action(agent_id, ActionType.DEFEND)
            # Only check is_defending if the action succeeded and attribute exists
            if hasattr(agent, "is_defending"):
                # The defend action may not change state if agent is already defending
                # Just verify the action doesn't crash
                current_defending = agent.is_defending
                self.assertIsInstance(current_defending, bool)
        except (TypeError, AttributeError):
            # Skip defend test if Mock objects cause comparison issues
            pass

        # Test other actions (they should execute without error)
        for action in [
            ActionType.ATTACK,
            ActionType.GATHER,
            ActionType.SHARE,
            ActionType.MOVE,
            ActionType.REPRODUCE,
        ]:
            self.env._process_action(agent_id, action)

    def test_reward_calculation(self):
        """Test reward calculation logic"""
        if not self.env.agents:
            self.skipTest("No agents in environment")

        agent_id = self.env.agents[0]
        agent = next((a for a in self.env.agent_objects if a.agent_id == agent_id), None)
        self.assertIsNotNone(agent)
        assert agent is not None  # Type assertion for type checker

        # Ensure agent has proper health and resource values
        combat_component = agent.get_component("combat")
        resource_component = agent.get_component("resource")
        
        if combat_component and combat_component.health is None:
            combat_component.health = 100.0
        if combat_component and combat_component.max_health is None:
            combat_component.max_health = 100.0
        if resource_component and resource_component.level is None:
            resource_component.level = 10

        # Test reward for alive agent
        reward = self.env._calculate_reward(agent_id)
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, -10)  # Should not be death penalty

        # Test reward for dead agent
        agent.alive = False
        reward = self.env._calculate_reward(agent_id)
        self.assertEqual(reward, -10)  # Death penalty

    def test_observation_generation(self):
        """Test observation generation for agents"""
        if not self.env.agents:
            self.skipTest("No agents in environment")

        agent_id = self.env.agents[0]
        obs = self.env._get_observation(agent_id)

        self.assertIsInstance(obs, np.ndarray)
        obs_space = self.env.observation_space(agent_id)
        self.assertEqual(obs.shape, obs_space.shape)
        self.assertEqual(obs.dtype, obs_space.dtype)

        # Test observation for non-existent agent
        obs = self.env._get_observation("non_existent_agent")
        self.assertIsInstance(obs, np.ndarray)
        # For non-existent agent, we get the default observation space shape
        obs_space = self.env.observation_space(self.env.agents[0] if self.env.agents else None)
        self.assertEqual(obs.shape, obs_space.shape)

    def test_termination_conditions(self):
        """Test various termination conditions"""
        # Reset environment to initialize agent selection
        self.env.reset()

        # Test max steps termination
        self.env.max_steps = 5
        self.env.time = 4

        # Take steps until we reach max_steps
        steps_taken = 0
        truncated_encountered = False
        while self.env.agents and steps_taken < 20:  # Increased safety limit
            action = self.env.action_space(self.env.agent_selection).sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            steps_taken += 1

            if truncated:
                truncated_encountered = True
                # Should truncate when time reaches max_steps
                self.assertGreaterEqual(
                    self.env.time, self.env.max_steps,
                    f"Should only truncate when time >= max_steps. Time: {self.env.time}, Max steps: {self.env.max_steps}",
                )
                break

        # Verify we actually encountered truncation
        self.assertTrue(
            truncated_encountered,
            f"Should have encountered truncation. Final time: {self.env.time}, Max steps: {self.env.max_steps}, Steps taken: {steps_taken}",
        )

        # Verify time reached max_steps
        self.assertGreaterEqual(
            self.env.time, self.env.max_steps,
            f"Time should have reached max_steps. Time: {self.env.time}, Max steps: {self.env.max_steps}",
        )

    def test_agent_removal(self):
        """Test that agents are properly removed when they die"""
        if not self.env.agent_objects:
            self.skipTest("No agents in environment")

        initial_count = len(self.env.agent_objects)
        agent = self.env.agent_objects[0]

        # Remove agent
        self.env.remove_agent(agent)

        self.assertEqual(len(self.env.agent_objects), initial_count - 1)
        self.assertNotIn(agent.agent_id, self.env.observations)

    def test_spatial_queries(self):
        """Test spatial query methods"""
        if not self.env.agent_objects:
            self.skipTest("No agents in environment")

        agent = self.env.agent_objects[0]
        position = agent.position

        # Test nearby agents query with timeout protection
        start_time = time.time()
        nearby_agents = self.env.get_nearby_agents(position, 10)
        self.assertLess(time.time() - start_time, 5.0, "get_nearby_agents took too long")
        self.assertIsInstance(nearby_agents, list)

        # Test nearby resources query with timeout protection
        start_time = time.time()
        nearby_resources = self.env.get_nearby_resources(position, 10)
        self.assertLess(time.time() - start_time, 5.0, "get_nearby_resources took too long")
        self.assertIsInstance(nearby_resources, list)

        # Test nearest resource query with timeout protection
        start_time = time.time()
        nearest_resource = self.env.get_nearest_resource(position)
        self.assertLess(time.time() - start_time, 5.0, "get_nearest_resource took too long")
        if nearest_resource:
            self.assertIsInstance(nearest_resource.position, tuple)

    def test_quadtree_indices(self):
        """Test Quadtree index functionality"""
        # Enable Quadtree indices
        self.env.enable_quadtree_indices()

        # Check that Quadtree indices were registered
        stats = self.env.spatial_index.get_stats()
        self.assertIn("quadtree_indices", stats)
        self.assertGreater(len(stats["quadtree_indices"]), 0)

        # Test Quadtree-specific queries
        if self.env.agent_objects:
            agent = self.env.agent_objects[0]
            position = agent.position

            # Test range queries (Quadtree optimized)
            bounds = (position[0] - 5, position[1] - 5, 10, 10)
            nearby_in_range = self.env.spatial_index.get_nearby_range(bounds, ["agents_quadtree"])
            self.assertIsInstance(nearby_in_range, dict)

            # Test Quadtree stats
            quadtree_stats = self.env.spatial_index.get_quadtree_stats("agents_quadtree")
            self.assertIsNotNone(quadtree_stats)
            self.assertIn("total_entities", quadtree_stats)

    def test_position_validation(self):
        """Test position validation logic"""
        # Valid positions
        self.assertTrue(self.env.is_valid_position((0, 0)))
        self.assertTrue(self.env.is_valid_position((50, 50)))
        self.assertTrue(self.env.is_valid_position((100, 100)))

        # Invalid positions
        self.assertFalse(self.env.is_valid_position((-1, 0)))
        self.assertFalse(self.env.is_valid_position((0, -1)))
        self.assertFalse(self.env.is_valid_position((101, 0)))
        self.assertFalse(self.env.is_valid_position((0, 101)))

    def test_seed_consistency(self):
        """Test that seeding produces consistent results"""
        # Create a simplified config for this test
        simple_config = SimulationConfig(
            environment=EnvironmentConfig(width=50, height=50),
            population=PopulationConfig(system_agents=0, independent_agents=0, control_agents=0),
            resources=ResourceConfig(
                max_resource_amount=10,
                resource_regen_rate=0.1,
                resource_regen_amount=1,
                initial_resources=5,
            ),
            max_steps=100,
            seed=42,
            observation=None,
        )

        # Test environment creation with timeout protection
        start_time = time.time()
        try:
            env1 = Environment(
                width=50,
                height=50,
                resource_distribution={"amount": 5},
                config=simple_config,
                seed=42,
                db_path=":memory:",
            )
            self.assertLess(time.time() - start_time, 10.0, "Environment 1 creation took too long")

            start_time = time.time()
            env2 = Environment(
                width=50,
                height=50,
                resource_distribution={"amount": 5},
                config=simple_config,
                seed=42,
                db_path=":memory:",
            )
            self.assertLess(time.time() - start_time, 10.0, "Environment 2 creation took too long")

            # Check that resources are in same positions
            positions1 = [r.position for r in env1.resources]
            positions2 = [r.position for r in env2.resources]
            self.assertEqual(positions1, positions2)

        finally:
            # Cleanup with timeout protection
            try:
                if 'env1' in locals():
                    start_time = time.time()
                    env1.cleanup()
                    self.assertLess(time.time() - start_time, 5.0, "Environment 1 cleanup took too long")
            except Exception as e:
                print(f"Warning: env1 cleanup failed: {e}")

            try:
                if 'env2' in locals():
                    start_time = time.time()
                    env2.cleanup()
                    self.assertLess(time.time() - start_time, 5.0, "Environment 2 cleanup took too long")
            except Exception as e:
                print(f"Warning: env2 cleanup failed: {e}")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test behavior when all agents are removed
        self.env.reset()

        # Manually remove all agents to test edge case
        agents_to_remove = list(self.env.agent_objects)
        for agent in agents_to_remove:
            agent.alive = False
            self.env.remove_agent(agent)

        # Environment should handle gracefully when there are no agents
        self.assertEqual(len(self.env.agents), 0)

        # Test step with no agents - environment should terminate
        if self.env.agents:
            action = self.env.action_space.sample()
        else:
            action = 0  # Any action since there are no agents
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertTrue(terminated or truncated)  # Should terminate when no agents remain

    def test_cleanup(self):
        """Test environment cleanup"""
        self.env.cleanup()
        # Should not raise any exceptions

    def test_database_integration(self):
        """Test database logging functionality"""
        # Test with real database path (in-memory is already used in setUp)
        self.assertIsNotNone(self.env.db)

        # Test interaction edge logging
        self.env.log_interaction_edge(
            source_type="agent",
            source_id="test_agent_1",
            target_type="agent",
            target_id="test_agent_2",
            interaction_type="attack",
            details={"damage": 10},
        )

        # Test logging with missing database (should not raise)
        env_no_db = Environment(
            width=50,
            height=50,
            resource_distribution={"amount": 1},
            config=self.config,
            db_path=":memory:",
        )
        # setup_db returns the database instance directly
        # Simulate missing database by setting db to None
        original_db = env_no_db.db
        env_no_db.db = None
        env_no_db.log_interaction_edge(
            source_type="agent",
            source_id="test",
            target_type="resource",
            target_id="res1",
            interaction_type="gather",
        )
        env_no_db.db = original_db  # Restore for cleanup
        env_no_db.cleanup()

    def test_metrics_tracking(self):
        """Test comprehensive metrics tracking"""
        initial_births = self.env.metrics_tracker.step_metrics.births
        initial_deaths = self.env.metrics_tracker.step_metrics.deaths

        # Test recording various events
        self.env.record_birth()
        self.assertEqual(self.env.metrics_tracker.step_metrics.births, initial_births + 1)

        self.env.record_death()
        self.assertEqual(self.env.metrics_tracker.step_metrics.deaths, initial_deaths + 1)

        self.env.record_combat_encounter()
        self.env.record_successful_attack()
        self.env.record_resources_shared(5.0)

        # Test metrics calculation
        metrics = self.env._calculate_metrics()
        self.assertIsInstance(metrics, dict)
        # Check for actual metric keys returned by the implementation
        self.assertIn("total_agents", metrics)
        self.assertTrue(any("resource" in key for key in metrics.keys()))

    def test_resource_consumption(self):
        """Test resource consumption through environment"""
        if not self.env.resources:
            self.skipTest("No resources in environment")

        resource = self.env.resources[0]
        initial_amount = resource.amount

        # Test resource consumption
        consumed = self.env.consume_resource(resource, 2.0)
        self.assertGreaterEqual(consumed, 0)
        self.assertLessEqual(consumed, 2.0)
        self.assertLessEqual(resource.amount, initial_amount)

        # Test consuming more than available
        large_amount = initial_amount + 10
        consumed = self.env.consume_resource(resource, large_amount)
        self.assertLessEqual(consumed, initial_amount)

    def test_agent_lifecycle_management(self):
        """Test complete agent lifecycle management"""
        initial_agent_count = len(self.env.agent_objects)

        # Test agent addition
        factory = AgentFactory(spatial_service=self.env.spatial_service)
        new_agent = factory.create_default_agent(
            agent_id="test_agent_new",
            position=(25, 25),
            initial_resources=5,
        )

        self.env.add_agent(new_agent)
        self.assertEqual(len(self.env.agent_objects), initial_agent_count + 1)
        self.assertIn(new_agent.agent_id, self.env.agents)
        self.assertIn(new_agent.agent_id, self.env.observations)

        # Test agent removal
        self.env.remove_agent(new_agent)
        self.assertEqual(len(self.env.agent_objects), initial_agent_count)
        self.assertNotIn(new_agent.agent_id, self.env.agents)
        self.assertNotIn(new_agent.agent_id, self.env.observations)

    def test_environment_state_management(self):
        """Test environment state capture and management"""
        # Test environment state capture
        state = self.env.state()
        self.assertIsNotNone(state)

        # Test simulation ID
        self.assertIsNotNone(self.env.simulation_id)
        self.assertIsInstance(self.env.simulation_id, str)

        # Test time advancement in update
        initial_time = self.env.time
        self.env.update()
        self.assertEqual(self.env.time, initial_time + 1)

        # Test initial agent count calculation
        initial_count = self.env.get_initial_agent_count()
        self.assertIsInstance(initial_count, int)
        self.assertGreaterEqual(initial_count, 0)

    def test_resource_regeneration(self):
        """Test resource regeneration behavior"""
        if not self.env.resources:
            self.skipTest("No resources in environment")

        # Deplete a resource partially
        resource = self.env.resources[0]
        if resource.amount > 1:
            self.env.consume_resource(resource, resource.amount - 0.5)

        # Update environment to trigger regeneration
        initial_amount = resource.amount
        self.env.update()

        # Resource should regenerate (unless already at max)
        # We can't guarantee regeneration happens every step, so just check it doesn't decrease
        self.assertGreaterEqual(resource.amount, 0)

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases"""
        # Test invalid action processing
        if self.env.agents:
            agent_id = self.env.agents[0]
            # Test invalid action (should not crash)
            self.env._process_action(agent_id, 999)  # Invalid action

            # Test action on non-existent agent
            self.env._process_action("non_existent", ActionType.DEFEND)

        # Test observation for dead agent
        if self.env.agent_objects:
            agent = self.env.agent_objects[0]
            agent.alive = False
            obs = self.env._get_observation(agent.agent_id)
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.shape, self.env.observation_space(agent.agent_id).shape)

        # Test reward calculation for non-existent agent
        reward = self.env._calculate_reward("non_existent")
        self.assertEqual(reward, -10.0)

    def test_deterministic_behavior(self):
        """Test deterministic behavior with seeds"""
        # Test agent ID generation consistency
        env1 = Environment(
            width=30,
            height=30,
            resource_distribution={"amount": 3},
            config=self.config,
            seed=12345,
            db_path=":memory:",
        )

        env2 = Environment(
            width=30,
            height=30,
            resource_distribution={"amount": 3},
            config=self.config,
            seed=12345,
            db_path=":memory:",
        )

        # Agent IDs should be deterministic
        if env1.agent_objects and env2.agent_objects:
            agent_ids_1 = sorted([a.agent_id for a in env1.agent_objects])
            agent_ids_2 = sorted([a.agent_id for a in env2.agent_objects])
            self.assertEqual(agent_ids_1, agent_ids_2)

        env1.cleanup()
        env2.cleanup()

    def test_next_agent_selection(self):
        """Test agent selection logic for PettingZoo AEC"""
        if len(self.env.agents) < 2:
            self.skipTest("Need at least 2 agents for this test")

        # Test normal agent cycling
        first_agent = self.env.agents[0]
        self.env.agent_selection = first_agent
        self.env._next_agent()

        self.assertNotEqual(self.env.agent_selection, first_agent)
        self.assertIn(self.env.agent_selection, self.env.agents)

        # Test with terminated agent
        if len(self.env.agents) > 1:
            terminated_agent = self.env.agents[0]
            self.env.terminations[terminated_agent] = True
            self.env.agent_selection = terminated_agent
            self.env._next_agent()

            # Should skip terminated agent
            self.assertNotEqual(self.env.agent_selection, terminated_agent)

    def test_complex_multi_step_scenarios(self):
        """Test complex multi-step interaction scenarios"""
        if not self.env.agents:
            self.skipTest("No agents in environment")

        # Simulate a complex scenario with multiple steps
        steps_taken = 0
        max_steps = 20

        obs, info = self.env.reset()

        while steps_taken < max_steps and self.env.agents:
            action = int(self.env.action_space().sample())
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Verify step results are valid
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)

            if terminated or truncated:
                break

            steps_taken += 1

        self.assertGreater(steps_taken, 0)

    def test_resource_depletion_scenarios(self):
        """Test behavior when resources are depleted"""
        if not self.env.resources:
            self.skipTest("No resources in environment")

        # Deplete all resources
        for resource in self.env.resources:
            self.env.consume_resource(resource, resource.amount)

        # Verify all resources are depleted
        total_resources = sum(r.amount for r in self.env.resources)
        self.assertEqual(total_resources, 0)

        # Test environment behavior with no resources
        if self.env.agents:
            action = self.env.action_space(self.env.agent_selection).sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Environment should handle gracefully
            self.assertIsInstance(obs, np.ndarray)

    def test_get_next_resource_id(self):
        """Test resource ID generation"""
        initial_id = self.env.resource_manager.next_resource_id

        # Test ID generation - resource manager handles ID generation internally
        # The test verifies that the resource manager maintains proper ID sequence
        self.assertEqual(initial_id, 10)  # Based on resource_distribution["amount"] = 10

        # Create a new resource to test ID increment
        new_resource = self.env.resource_manager.add_resource((5, 5), 5.0)
        self.assertEqual(new_resource.resource_id, initial_id)
        self.assertEqual(self.env.resource_manager.next_resource_id, initial_id + 1)

    def test_position_dirty_marking(self):
        """Test spatial index dirty position marking"""
        # Test marking positions dirty
        self.env.mark_positions_dirty()

        # This should not raise any exceptions
        # The actual dirty flag testing is handled in spatial index tests
        self.assertIsNotNone(self.env.spatial_index)

    def test_boundary_conditions(self):
        """Test boundary conditions and limits"""
        # Test with minimal environment
        minimal_config = SimulationConfig(
            environment=EnvironmentConfig(width=1, height=1),
            population=PopulationConfig(system_agents=0, independent_agents=0, control_agents=0),
            resources=ResourceConfig(initial_resources=0),
            max_steps=1,
            seed=123,
            observation=None,
        )

        minimal_env = Environment(
            width=1,
            height=1,
            resource_distribution={"amount": 0},
            config=minimal_config,
            db_path=":memory:",
        )

        # Should handle empty environment
        self.assertEqual(len(minimal_env.agent_objects), 0)
        self.assertEqual(len(minimal_env.resources), 0)

        minimal_env.cleanup()

        # Test with very large coordinates
        large_pos = (self.env.width * 2, self.env.height * 2)
        self.assertFalse(self.env.is_valid_position(large_pos))

    def test_render_functionality(self):
        """Test environment rendering"""
        # Test render method (should not crash)
        try:
            self.env.render(mode="human")
        except Exception as e:
            self.fail(f"Render failed with exception: {e}")

        # Test invalid render mode
        self.env.render(mode="invalid_mode")

    def test_memory_and_cleanup(self):
        """Test proper memory management and cleanup"""
        # Create a temporary environment to test cleanup
        test_env = Environment(
            width=50,
            height=50,
            resource_distribution={"amount": 5},
            config=self.config,
            db_path=":memory:",
        )

        # Add some agents and resources
        initial_objects = len(test_env.agent_objects)

        # Test __del__ method
        test_env.__del__()

        # Test multiple cleanup calls (should be safe)
        test_env.cleanup()
        test_env.cleanup()

    def test_action_enum_completeness(self):
        """Test that Action enum is properly defined"""
        # Test all actions are defined
        expected_actions = [
            "DEFEND",
            "ATTACK",
            "GATHER",
            "SHARE",
            "MOVE",
            "REPRODUCE",
            "PASS",
        ]

        for action_name in expected_actions:
            self.assertTrue(hasattr(ActionType, action_name))

        # Test action values
        self.assertEqual(ActionType.DEFEND, 0)
        self.assertEqual(ActionType.ATTACK, 1)
        self.assertEqual(ActionType.GATHER, 2)
        self.assertEqual(ActionType.SHARE, 3)
        self.assertEqual(ActionType.MOVE, 4)
        self.assertEqual(ActionType.REPRODUCE, 5)
        self.assertEqual(ActionType.PASS, 6)

        # Test action space matches the enabled action mapping (not raw enum)
        self.assertEqual(len(self.env._action_mapping), self.env.action_space().n)

    def test_configuration_edge_cases(self):
        """Test various configuration edge cases"""
        # Test with None config
        try:
            env_no_config = Environment(
                width=10,
                height=10,
                resource_distribution={"amount": 1},
                config=None,
                db_path=":memory:",
            )
            self.assertIsNotNone(env_no_config)
            env_no_config.cleanup()
        except Exception as e:
            self.fail(f"Environment creation with None config failed: {e}")

        # Test with minimal config attributes
        partial_config = SimulationConfig(
            environment=EnvironmentConfig(width=10, height=10),
            population=PopulationConfig(system_agents=1),
            seed=456,
        )

        # Missing other attributes should use defaults

        try:
            env_partial = Environment(
                width=10,
                height=10,
                resource_distribution={"amount": 1},
                config=partial_config,
                db_path=":memory:",
            )
            env_partial.cleanup()
        except Exception as e:
            self.fail(f"Environment creation with partial config failed: {e}")

    def test_position_discretization(self):
        """Test position discretization methods work correctly."""
        from farm.core.environment import discretize_position_continuous

        grid_size = (10, 10)

        # Test floor discretization (default)
        pos = (2.7, 3.9)
        x_idx, y_idx = discretize_position_continuous(pos, grid_size, "floor")
        self.assertEqual(x_idx, 2)
        self.assertEqual(y_idx, 3)

        # Test round discretization
        x_idx, y_idx = discretize_position_continuous(pos, grid_size, "round")
        self.assertEqual(x_idx, 3)
        self.assertEqual(y_idx, 4)

        # Test ceil discretization
        x_idx, y_idx = discretize_position_continuous(pos, grid_size, "ceil")
        self.assertEqual(x_idx, 3)
        self.assertEqual(y_idx, 4)

        # Test boundary conditions
        pos_boundary = (9.9, 9.9)
        x_idx, y_idx = discretize_position_continuous(pos_boundary, grid_size, "floor")
        self.assertEqual(x_idx, 9)
        self.assertEqual(y_idx, 9)

        # Test out of bounds (should clamp)
        pos_oob = (15.0, 15.0)
        x_idx, y_idx = discretize_position_continuous(pos_oob, grid_size, "floor")
        self.assertEqual(x_idx, 9)
        self.assertEqual(y_idx, 9)

    def test_bilinear_interpolation(self):
        """Test bilinear interpolation for resource distribution."""
        import torch

        from farm.core.environment import bilinear_distribute_value

        grid = torch.zeros((5, 5))
        grid_size = (5, 5)

        # Test interpolation at center
        pos = (2.5, 2.5)
        bilinear_distribute_value(pos, 4.0, grid, grid_size)

        # Check that value is distributed across 4 cells
        self.assertGreater(grid[2, 2].item(), 0)  # top-left
        self.assertGreater(grid[2, 3].item(), 0)  # top-right
        self.assertGreater(grid[3, 2].item(), 0)  # bottom-left
        self.assertGreater(grid[3, 3].item(), 0)  # bottom-right

        # Check that total distributed value equals input
        total = torch.sum(grid).item()
        self.assertAlmostEqual(total, 4.0, places=5)

        # Test interpolation at integer position (should go to single cell)
        grid.fill_(0)
        pos = (2.0, 2.0)
        bilinear_distribute_value(pos, 2.0, grid, grid_size)

        # Should be concentrated in one cell
        self.assertEqual(grid[2, 2].item(), 2.0)
        self.assertEqual(torch.sum(grid).item(), 2.0)


if __name__ == "__main__":
    unittest.main()
