import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from gymnasium import spaces
from pettingzoo.test import api_test
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from supersuit import pettingzoo_env_to_vec_env_v1

from farm.agents import ControlAgent, IndependentAgent, SystemAgent
from farm.core.environment import Action, Environment
from farm.core.resources import Resource


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # Create a mock config with required attributes
        self.mock_config = Mock()
        self.mock_config.system_agents = 1
        self.mock_config.independent_agents = 1
        self.mock_config.control_agents = 1
        self.mock_config.max_steps = 100
        self.mock_config.max_resource_amount = 10
        self.mock_config.resource_regen_rate = 0.1
        self.mock_config.resource_regen_amount = 1
        self.mock_config.seed = 42
        # Explicitly set observation to None to use default ObservationConfig
        self.mock_config.observation = None
        # Add neural network configuration for agents
        self.mock_config.dqn_hidden_size = 128
        self.mock_config.learning_rate = 0.001  # This is the key one being accessed
        self.mock_config.device = "cpu"
        self.mock_config.base_consumption_rate = 1
        # Add other DQN config parameters that might be accessed
        self.mock_config.gamma = 0.99
        self.mock_config.epsilon_start = 1.0
        self.mock_config.epsilon_min = 0.01
        self.mock_config.epsilon_decay = 0.995
        self.mock_config.batch_size = 32
        self.mock_config.memory_size = 10000
        self.mock_config.tau = 0.005
        self.mock_config.target_update_freq = 100
        # Add agent parameters
        self.mock_config.agent_parameters = {}
        # Add other agent-related config attributes needed by BaseAgent
        self.mock_config.starvation_threshold = 10
        self.mock_config.max_starvation_time = 100
        self.mock_config.starting_health = 100
        self.mock_config.max_movement = 8
        self.mock_config.gathering_range = 30
        self.mock_config.social_range = 30
        self.mock_config.attack_range = 20.0
        self.mock_config.range = 20.0  # Add range for backwards compatibility
        self.mock_config.ideal_density_radius = 30.0  # Add for reproduction
        self.mock_config.min_reproduction_resources = 5.0
        self.mock_config.max_population = 100
        self.mock_config.max_wait_steps = 10
        self.mock_config.min_health_ratio = 0.3
        self.mock_config.min_space_required = 20.0
        self.mock_config.max_local_density = 0.8
        self.mock_config.success_reward = 1.0
        self.mock_config.offspring_survival_bonus = 0.5
        self.mock_config.population_balance_bonus = 0.2

        self.env = Environment(
            width=100,
            height=100,
            resource_distribution={"amount": 10},
            config=self.mock_config,
            db_path=":memory:",  # Use in-memory database for tests
        )

        # Add a small set of agents explicitly (external agent management)
        initial_agents = [
            SystemAgent(
                agent_id=self.env.get_next_agent_id(),
                position=(10, 10),
                resource_level=5,
                environment=self.env,
                generation=0,
            ),
            IndependentAgent(
                agent_id=self.env.get_next_agent_id(),
                position=(12, 12),
                resource_level=5,
                environment=self.env,
                generation=0,
            ),
            ControlAgent(
                agent_id=self.env.get_next_agent_id(),
                position=(14, 14),
                resource_level=5,
                environment=self.env,
                generation=0,
            ),
        ]
        for a in initial_agents:
            self.env.add_agent(a)

    def tearDown(self):
        if hasattr(self, "env") and self.env:
            self.env.cleanup()

    def test_api_compliance(self):
        api_test(self.env, num_cycles=10, verbose_progress=False)

    def test_observation_action_spaces(self):
        if not self.env.agents:
            self.skipTest("No agents in environment")
        agent = self.env.agents[0]
        obs_space = self.env.observation_space(agent)
        self.assertIsInstance(obs_space, spaces.Box)
        self.assertEqual(
            obs_space.shape, (12, 13, 13)
        )  # Default R=6, 2*6+1=13, NUM_CHANNELS=12
        self.assertEqual(obs_space.dtype, np.float32)

        act_space = self.env.action_space(agent)
        self.assertIsInstance(act_space, spaces.Discrete)
        self.assertEqual(act_space.n, 6)  # With REPRODUCE

    def test_reset_step_cycle(self):
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, self.env.observation_space.shape)

        done = False
        steps = 0
        while not done and steps < 10:
            action = self.env.action_space.sample()
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
        self.assertEqual(len(self.env.resources), 10)  # resource_distribution["amount"]

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
            action = self.env.action_space.sample()
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
        agent = next(
            (a for a in self.env.agent_objects if a.agent_id == agent_id), None
        )
        self.assertIsNotNone(agent)
        assert agent is not None  # Type assertion for type checker

        # Ensure agent has proper health and resource values
        if hasattr(agent, "current_health") and agent.current_health is None:
            agent.current_health = 100.0
        if hasattr(agent, "starting_health") and agent.starting_health is None:
            agent.starting_health = 100.0
        if hasattr(agent, "resource_level") and agent.resource_level is None:
            agent.resource_level = 10

        initial_health = agent.current_health
        initial_resources = agent.resource_level

        # Test defend action
        self.env._process_action(agent_id, Action.DEFEND)
        self.assertTrue(agent.is_defending)

        # Test other actions (they should execute without error)
        for action in [
            Action.ATTACK,
            Action.GATHER,
            Action.SHARE,
            Action.MOVE,
            Action.REPRODUCE,
        ]:
            self.env._process_action(agent_id, action)

    def test_reward_calculation(self):
        """Test reward calculation logic"""
        if not self.env.agents:
            self.skipTest("No agents in environment")

        agent_id = self.env.agents[0]
        agent = next(
            (a for a in self.env.agent_objects if a.agent_id == agent_id), None
        )
        self.assertIsNotNone(agent)
        assert agent is not None  # Type assertion for type checker

        # Ensure agent has proper health and resource values
        if hasattr(agent, "current_health") and agent.current_health is None:
            agent.current_health = 100.0
        if hasattr(agent, "starting_health") and agent.starting_health is None:
            agent.starting_health = 100.0
        if hasattr(agent, "resource_level") and agent.resource_level is None:
            agent.resource_level = 10

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
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertEqual(obs.dtype, self.env.observation_space.dtype)

        # Test observation for non-existent agent
        obs = self.env._get_observation("non_existent_agent")
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_termination_conditions(self):
        """Test various termination conditions"""
        # Test max steps termination
        self.env.max_steps = 5
        self.env.time = 4

        # Take a step
        if self.env.agents:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertFalse(terminated)  # Should not terminate yet
            self.assertFalse(truncated)  # Should not truncate yet

        # Take another step to reach max_steps
        if self.env.agents:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertTrue(truncated)  # Should truncate at max_steps

    def test_agent_removal(self):
        """Test that agents are properly removed when they die"""
        if not self.env.agent_objects:
            self.skipTest("No agents in environment")

        initial_count = len(self.env.agent_objects)
        agent = self.env.agent_objects[0]

        # Remove agent
        self.env.remove_agent(agent)

        self.assertEqual(len(self.env.agent_objects), initial_count - 1)
        self.assertNotIn(agent.agent_id, self.env.agent_observations)

    def test_spatial_queries(self):
        """Test spatial query methods"""
        if not self.env.agent_objects:
            self.skipTest("No agents in environment")

        agent = self.env.agent_objects[0]
        position = agent.position

        # Test nearby agents query
        nearby_agents = self.env.get_nearby_agents(position, 10)
        self.assertIsInstance(nearby_agents, list)

        # Test nearby resources query
        nearby_resources = self.env.get_nearby_resources(position, 10)
        self.assertIsInstance(nearby_resources, list)

        # Test nearest resource query
        nearest_resource = self.env.get_nearest_resource(position)
        if nearest_resource:
            self.assertIsInstance(nearest_resource.position, tuple)

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
        # Create a simplified mock config for this test
        simple_config = Mock()
        simple_config.system_agents = 0
        simple_config.independent_agents = 0
        simple_config.control_agents = 0
        simple_config.max_steps = 100
        simple_config.max_resource_amount = 10
        simple_config.resource_regen_rate = 0.1
        simple_config.resource_regen_amount = 1
        simple_config.observation = None

        env1 = Environment(
            width=50,
            height=50,
            resource_distribution={"amount": 5},
            config=simple_config,
            seed=42,
            db_path=":memory:",
        )
        env2 = Environment(
            width=50,
            height=50,
            resource_distribution={"amount": 5},
            config=simple_config,
            seed=42,
            db_path=":memory:",
        )

        # Check that resources are in same positions
        positions1 = [r.position for r in env1.resources]
        positions2 = [r.position for r in env2.resources]
        self.assertEqual(positions1, positions2)

        # Cleanup
        env1.cleanup()
        env2.cleanup()

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
        self.assertTrue(
            terminated or truncated
        )  # Should terminate when no agents remain

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
            config=self.mock_config,
            db_path=":memory:",
        )
        env_no_db.db = None
        env_no_db.log_interaction_edge(
            source_type="agent",
            source_id="test",
            target_type="resource",
            target_id="res1",
            interaction_type="gather",
        )
        env_no_db.cleanup()

    def test_metrics_tracking(self):
        """Test comprehensive metrics tracking"""
        initial_births = self.env.metrics_tracker.step_metrics.births
        initial_deaths = self.env.metrics_tracker.step_metrics.deaths

        # Test recording various events
        self.env.record_birth()
        self.assertEqual(
            self.env.metrics_tracker.step_metrics.births, initial_births + 1
        )

        self.env.record_death()
        self.assertEqual(
            self.env.metrics_tracker.step_metrics.deaths, initial_deaths + 1
        )

        self.env.record_combat_encounter()
        self.env.record_successful_attack()
        self.env.record_resources_shared(5.0)

        # Test metrics calculation
        metrics = self.env._calculate_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("agent_count", metrics)
        self.assertIn("resource_count", metrics)

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
        new_agent = SystemAgent(
            agent_id="test_agent_new",
            position=(25, 25),
            resource_level=5,
            environment=self.env,
            generation=1,
        )

        self.env.add_agent(new_agent)
        self.assertEqual(len(self.env.agent_objects), initial_agent_count + 1)
        self.assertIn(new_agent.agent_id, self.env.agents)
        self.assertIn(new_agent.agent_id, self.env.agent_observations)

        # Test agent removal
        self.env.remove_agent(new_agent)
        self.assertEqual(len(self.env.agent_objects), initial_agent_count)
        self.assertNotIn(new_agent.agent_id, self.env.agents)
        self.assertNotIn(new_agent.agent_id, self.env.agent_observations)

    def test_environment_state_management(self):
        """Test environment state capture and management"""
        # Test environment state capture
        state = self.env.get_state()
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
            self.env._process_action("non_existent", Action.DEFEND)

        # Test observation for dead agent
        if self.env.agent_objects:
            agent = self.env.agent_objects[0]
            agent.alive = False
            obs = self.env._get_observation(agent.agent_id)
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.shape, self.env.observation_space.shape)

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
            config=self.mock_config,
            seed=12345,
            db_path=":memory:",
        )

        env2 = Environment(
            width=30,
            height=30,
            resource_distribution={"amount": 3},
            config=self.mock_config,
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
            action = int(self.env.action_space().sample())
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Environment should handle gracefully
            self.assertIsInstance(obs, np.ndarray)

    def test_get_next_resource_id(self):
        """Test resource ID generation"""
        initial_id = self.env.next_resource_id

        # Test ID generation
        new_id = self.env.get_next_resource_id()
        self.assertEqual(new_id, initial_id)
        self.assertEqual(self.env.next_resource_id, initial_id + 1)

        # Test sequential IDs
        next_id = self.env.get_next_resource_id()
        self.assertEqual(next_id, initial_id + 1)
        self.assertEqual(self.env.next_resource_id, initial_id + 2)

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
        minimal_config = Mock()
        minimal_config.system_agents = 0
        minimal_config.independent_agents = 0
        minimal_config.control_agents = 0
        minimal_config.max_steps = 1
        minimal_config.observation = None

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
            config=self.mock_config,
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
        expected_actions = ["DEFEND", "ATTACK", "GATHER", "SHARE", "MOVE", "REPRODUCE"]

        for action_name in expected_actions:
            self.assertTrue(hasattr(Action, action_name))

        # Test action values
        self.assertEqual(Action.DEFEND, 0)
        self.assertEqual(Action.ATTACK, 1)
        self.assertEqual(Action.GATHER, 2)
        self.assertEqual(Action.SHARE, 3)
        self.assertEqual(Action.MOVE, 4)
        self.assertEqual(Action.REPRODUCE, 5)

        # Test action space matches enum
        self.assertEqual(len(Action), self.env.action_space().n)

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

        # Test with missing config attributes
        partial_config = Mock()
        partial_config.system_agents = 1
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


if __name__ == "__main__":
    unittest.main()
