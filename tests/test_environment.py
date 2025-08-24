import unittest
import numpy as np
from farm.core.environment import Environment, Action
from pettingzoo.test import api_test
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from supersuit import pettingzoo_env_to_vec_env_v1
from unittest.mock import Mock, patch

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
        
        self.env = Environment(
            width=100,
            height=100,
            resource_distribution={"amount": 10},
            config=self.mock_config,
            db_path=None  # Disable database logging for tests
        )

    def tearDown(self):
        if hasattr(self, 'env') and self.env:
            self.env.cleanup()

    def test_api_compliance(self):
        api_test(self.env, num_cycles=10, verbose_progress=False)

    def test_observation_action_spaces(self):
        if not self.env.agents:
            self.skipTest("No agents in environment")
        agent = self.env.agents[0]
        obs_space = self.env.observation_space(agent)
        self.assertIsInstance(obs_space, spaces.Box)
        self.assertEqual(obs_space.shape, (12, 13, 13))  # Default R=6, 2*6+1=13, NUM_CHANNELS=12
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

    def test_rl_compatibility(self):
        # Wrap PettingZoo env for SB3 compatibility
        vec_env = pettingzoo_env_to_vec_env_v1(self.env)
        vec_env = make_vec_env(lambda: vec_env, n_envs=1)

        model = PPO("MlpPolicy", vec_env, verbose=0)
        model.learn(total_timesteps=10)

        # If no errors, it's compatible
        self.assertTrue(True)

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
        agent = next((a for a in self.env.agent_objects if a.agent_id == agent_id), None)
        self.assertIsNotNone(agent)
        
        initial_health = agent.current_health
        initial_resources = agent.resource_level
        
        # Test defend action
        self.env._process_action(agent_id, Action.DEFEND)
        self.assertTrue(agent.is_defending)
        
        # Test other actions (they should execute without error)
        for action in [Action.ATTACK, Action.GATHER, Action.SHARE, Action.MOVE, Action.REPRODUCE]:
            self.env._process_action(agent_id, action)

    def test_reward_calculation(self):
        """Test reward calculation logic"""
        if not self.env.agents:
            self.skipTest("No agents in environment")
        
        agent_id = self.env.agents[0]
        agent = next((a for a in self.env.agent_objects if a.agent_id == agent_id), None)
        
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
            self.assertFalse(truncated)   # Should not truncate yet
        
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
            width=50, height=50,
            resource_distribution={"amount": 5},
            config=simple_config,
            seed=42,
            db_path=None
        )
        env2 = Environment(
            width=50, height=50,
            resource_distribution={"amount": 5},
            config=simple_config,
            seed=42,
            db_path=None
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
        # Test with no agents
        env = Environment(
            width=50, height=50,
            resource_distribution={"amount": 5},
            config=None,
            db_path=None
        )
        
        # Should handle gracefully
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        
        # Test step with no agents
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertTrue(truncated)
        
        # Cleanup
        env.cleanup()

    def test_cleanup(self):
        """Test environment cleanup"""
        self.env.cleanup()
        # Should not raise any exceptions

if __name__ == '__main__':
    unittest.main()
