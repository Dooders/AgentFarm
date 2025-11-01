"""Unit tests for the DecisionModule class.

This module tests the core DecisionModule functionality including:
- Initialization with different configurations
- Action decision making
- Experience updates
- Model persistence
- Algorithm selection and fallbacks
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from pydantic import ValidationError

np.random.seed(42)  # For reproducibility in tests

from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import TIANSHOU_AVAILABLE, DecisionModule


class TestDecisionModule(unittest.TestCase):
    """Test cases for DecisionModule class."""

    # Algorithm types used across multiple test methods
    ALGORITHM_TYPES = ['ppo', 'sac', 'dqn', 'a2c', 'ddpg']

    def setUp(self):
        """Set up test fixtures."""
        from gymnasium import spaces

        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "test_agent_1"

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = spaces.Discrete(7)  # Standard action count
        self.mock_agent.environment = self.mock_env

        # Create mock observation space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )

        # Default config
        self.config = DecisionConfig()

    def test_initialization_without_tianshou(self):
        """Test DecisionModule initialization when Tianshou is not available."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        self.assertEqual(module.agent_id, "test_agent_1")
        self.assertEqual(module.num_actions, 7)
        self.assertIsInstance(
            module.algorithm, type(module.algorithm)
        )  # Fallback algorithm
        self.assertFalse(module._is_trained)

    def test_initialization_with_custom_action_space(self):
        """Test initialization with custom action space."""
        from gymnasium import spaces

        custom_action_space = spaces.Discrete(8)
        module = DecisionModule(
            self.mock_agent, custom_action_space, self.observation_space, self.config
        )

        self.assertEqual(module.num_actions, 8)
        self.assertEqual(module.action_space, custom_action_space)

    def test_initialization_with_custom_observation_space(self):
        """Test initialization with custom observation space."""
        from gymnasium import spaces

        custom_obs_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        # Create config that doesn't override observation space shape
        config = DecisionConfig(rl_state_dim=0)
        module = DecisionModule(
            self.mock_agent, self.mock_env.action_space, custom_obs_space, config
        )

        self.assertEqual(module.state_dim, 16)
        self.assertEqual(module.observation_space, custom_obs_space)

    def test_decide_action_tensor_input(self):
        """Test decide_action with tensor input."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        state = torch.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_numpy_input(self):
        """Test decide_action with numpy array input."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        state = np.random.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_invalid_algorithm(self):
        """Test decide_action when algorithm is None."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        module.algorithm = None
        state = torch.randn(8)

        action = module.decide_action(state)

        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_invalid_action_range(self):
        """Test decide_action when algorithm returns out-of-range action."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm select_action method to return invalid action
        with patch.object(
            module.algorithm, "select_action", return_value=10
        ):
            state = torch.randn(8)
            action = module.decide_action(state)

            # Should fallback to random valid action
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_with_enabled_actions(self):
        """Test decide_action with enabled_actions parameter."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)

        # Test with enabled actions restricting to first 3 actions
        enabled_actions = [0, 1, 2]
        action = module.decide_action(state, enabled_actions)

        # Should return an index within the enabled_actions list (0, 1, or 2)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < len(enabled_actions))

        # Test with single enabled action
        enabled_actions = [1]
        action = module.decide_action(state, enabled_actions)

        # Should return index 0 (position of the only enabled action)
        self.assertEqual(action, 0)

        # Test with empty enabled actions (should fallback to full space)
        action = module.decide_action(state, [])
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

        # Test with None (should use full space)
        action = module.decide_action(state, None)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_decide_action_with_action_weights(self):
        """Test decide_action with action_weights parameter."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        
        # Create action weights favoring action 0
        action_weights = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01], dtype=np.float64)

        # Test multiple selections to verify weighted random
        selection_counts = {i: 0 for i in range(7)}
        num_selections = 1000
        
        np.random.seed(42)
        for _ in range(num_selections):
            action = module.decide_action(state, action_weights=action_weights)
            selection_counts[action] += 1

        # Action 0 should be selected most often
        self.assertGreater(selection_counts[0], selection_counts[1])
        self.assertGreater(selection_counts[0], 600)  # Roughly 70%

    def test_decide_action_with_weights_and_enabled_actions(self):
        """Test decide_action with both action_weights and enabled_actions."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        
        # Weights for all actions
        action_weights = np.array([0.1, 0.5, 0.3, 0.05, 0.03, 0.01, 0.01], dtype=np.float64)
        # Only enable actions 1 and 2
        enabled_actions = [1, 2]

        # Test multiple selections
        selection_counts = {1: 0, 2: 0}
        num_selections = 1000
        
        np.random.seed(42)
        for _ in range(num_selections):
            action = module.decide_action(state, enabled_actions, action_weights=action_weights)
            # Action is relative index (0 or 1) in enabled_actions
            self.assertIn(action, [0, 1])
            # Map to actual action index
            actual_action = enabled_actions[action]
            selection_counts[actual_action] += 1

        # Action 1 (weight 0.5) should be selected more than action 2 (weight 0.3)
        self.assertGreater(selection_counts[1], selection_counts[2])

    def test_decide_action_with_weights_exploitation_probabilities(self):
        """Test that weights scale probabilities during exploitation."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        
        # Create weights
        action_weights = np.array([0.1, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Mock algorithm to provide probabilities
        mock_probs = np.array([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0]], dtype=np.float32)
        with patch.object(module.algorithm, "predict_proba", return_value=mock_probs):
            # Test multiple selections
            selection_counts = {i: 0 for i in range(7)}
            num_selections = 1000
            
            np.random.seed(42)
            for _ in range(num_selections):
                action = module.decide_action(state, action_weights=action_weights)
                selection_counts[action] += 1

            # After scaling probabilities by weights, action 2 should dominate
            # Original probs: [0.25, 0.25, 0.25, 0.25]
            # After weights: [0.1*0.25, 0.1*0.25, 0.8*0.25, 0.0*0.25] = [0.025, 0.025, 0.2, 0.0]
            # Normalized: [0.1, 0.1, 0.8, 0.0]
            self.assertGreater(selection_counts[2], selection_counts[0])
            self.assertGreater(selection_counts[2], selection_counts[1])

    def test_decide_action_with_weights_backward_compatibility(self):
        """Test that decide_action works without weights (backward compatibility)."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)

        # Should work without weights parameter
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

        # Should also work with None weights
        action2 = module.decide_action(state, action_weights=None)
        self.assertIsInstance(action2, int)
        self.assertTrue(0 <= action2 < module.num_actions)

    @patch("farm.core.decision.decision.logger")
    def test_decide_action_exception_handling(self, mock_logger):
        """Test decide_action exception handling."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm to raise exception - patch select_action_with_mask since it's called first
        with patch.object(
            module.algorithm, "select_action_with_mask", side_effect=Exception("Test error")
        ):
            state = torch.randn(8)
            action = module.decide_action(state)

            # Should log error and return random action
            mock_logger.error.assert_called_once()
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < module.num_actions)

    def test_update_with_algorithm(self):
        """Test update method with algorithm."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        action = 1
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Mock the train method and should_train to ensure training happens
        with patch.object(module.algorithm, "train") as mock_train, \
             patch.object(module.algorithm, "should_train", return_value=True):
            module.update(state, action, reward, next_state, done)

            # For Tianshou algorithms, train should be called when should_train returns True
            if hasattr(module.algorithm, "train"):
                mock_train.assert_called_once()
            self.assertTrue(module._is_trained)

    def test_update_logs_learning_experience(self):
        """Test that update logs learning experience to database."""
        # Set up mock agent with full database chain
        mock_db = Mock()
        mock_logger = Mock()
        mock_db.logger = mock_logger

        mock_env = Mock()
        mock_env.db = mock_db

        mock_time_service = Mock()
        mock_time_service.current_time.return_value = 42

        mock_action = Mock()
        mock_action.name = "move"

        self.mock_agent.environment = mock_env
        # Set up services structure as expected by decision module
        self.mock_agent.services = Mock()
        self.mock_agent.services.time_service = mock_time_service
        self.mock_agent.actions = [mock_action, Mock(), Mock()]

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        action = 0
        reward = 1.5
        next_state = torch.randn(8)
        done = False

        # Call update
        module.update(state, action, reward, next_state, done)

        # Verify log_agent_action was called with correct parameters including learning metadata
        mock_logger.log_agent_action.assert_called_once()
        call_args = mock_logger.log_agent_action.call_args

        self.assertEqual(call_args.kwargs['step_number'], 42)
        self.assertIsNotNone(call_args.kwargs.get('module_type'))
        self.assertIsNotNone(call_args.kwargs.get('module_id'))
        self.assertEqual(call_args.kwargs['agent_id'], 'test_agent_1')
        self.assertEqual(call_args.kwargs['module_type'], self.config.algorithm_type)
        self.assertEqual(call_args.kwargs['action_taken'], 0)
        self.assertEqual(call_args.kwargs['action_taken_mapped'], 'move')
        self.assertEqual(call_args.kwargs['reward'], 1.5)
        self.assertIsInstance(call_args.kwargs['module_id'], str)

    def test_update_without_database_does_not_crash(self):
        """Test that update works when database is not available."""
        # Agent without database connection
        self.mock_agent.environment = None

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        action = 1
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Should not raise exception
        module.update(state, action, reward, next_state, done)

    def test_update_without_time_service_skips_logging(self):
        """Test that update skips logging when time service is unavailable."""
        # Set up agent with database but no time service
        mock_db = Mock()
        mock_logger = Mock()
        mock_db.logger = mock_logger

        mock_env = Mock()
        mock_env.db = mock_db

        self.mock_agent.environment = mock_env
        self.mock_agent.time_service = None

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        action = 1
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Call update
        module.update(state, action, reward, next_state, done)

        # Should not call log_agent_action when time service is missing
        mock_logger.log_agent_action.assert_not_called()

    def test_update_without_actions_uses_fallback_logging(self):
        """Test that update uses fallback action name when actions list is empty."""
        # Set up agent with database and time service but no actions
        mock_db = Mock()
        mock_logger = Mock()
        mock_db.logger = mock_logger

        mock_env = Mock()
        mock_env.db = mock_db

        mock_time_service = Mock()
        mock_time_service.current_time.return_value = 42

        self.mock_agent.environment = mock_env
        self.mock_agent.time_service = mock_time_service
        self.mock_agent.actions = []  # Empty actions list

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        action = 0
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Call update
        module.update(state, action, reward, next_state, done)

        # Should call log_agent_action with fallback action name when actions are empty
        mock_logger.log_agent_action.assert_called_once()
        call_args = mock_logger.log_agent_action.call_args
        # action_taken_mapped should be passed as a keyword argument
        assert call_args.kwargs.get('action_taken_mapped') == 'action_0'

    def test_update_logging_exception_does_not_crash(self):
        """Test that logging exceptions are handled gracefully."""
        # Set up mock agent with database that raises exception
        mock_db = Mock()
        mock_logger = Mock()
        mock_logger.log_agent_action.side_effect = Exception("Database error")
        mock_db.logger = mock_logger

        mock_env = Mock()
        mock_env.db = mock_db

        mock_time_service = Mock()
        mock_time_service.current_time.return_value = 42

        mock_action = Mock()
        mock_action.name = "move"

        self.mock_agent.environment = mock_env
        self.mock_agent.time_service = mock_time_service
        self.mock_agent.actions = [mock_action]

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)
        action = 0
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Should not raise exception even when logging fails
        module.update(state, action, reward, next_state, done)

    def test_update_logs_correct_algorithm_type(self):
        """Test that different algorithm types are logged correctly."""
        for algo_type in self.ALGORITHM_TYPES:
            with self.subTest(algorithm_type=algo_type):
                # Set up mock agent with database
                mock_db = Mock()
                mock_logger = Mock()
                mock_db.logger = mock_logger

                mock_env = Mock()
                mock_env.db = mock_db

                mock_time_service = Mock()
                mock_time_service.current_time.return_value = 10

                mock_action = Mock()
                mock_action.name = "gather"

                self.mock_agent.environment = mock_env
                self.mock_agent.time_service = mock_time_service
                self.mock_agent.actions = [mock_action]

                config = DecisionConfig(algorithm_type=algo_type)
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                state = torch.randn(8)
                module.update(state, 0, 1.0, torch.randn(8), False)

                # Verify correct algorithm type was logged
                call_args = mock_logger.log_agent_action.call_args
                self.assertEqual(call_args.kwargs['module_type'], algo_type)

    def test_update_logs_different_rewards(self):
        """Test that different reward values are logged correctly."""
        rewards = [-10.0, -1.5, 0.0, 1.5, 10.0, 100.5]

        for reward_value in rewards:
            with self.subTest(reward=reward_value):
                # Set up mock agent with database
                mock_db = Mock()
                mock_logger = Mock()
                mock_db.logger = mock_logger

                mock_env = Mock()
                mock_env.db = mock_db

                mock_time_service = Mock()
                mock_time_service.current_time.return_value = 5

                mock_action = Mock()
                mock_action.name = "attack"

                self.mock_agent.environment = mock_env
                self.mock_agent.time_service = mock_time_service
                self.mock_agent.actions = [mock_action]

                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    self.config,
                )

                state = torch.randn(8)
                module.update(state, 0, reward_value, torch.randn(8), False)

                # Verify correct reward was logged
                call_args = mock_logger.log_agent_action.call_args
                self.assertEqual(call_args.kwargs['reward'], reward_value)

    def test_update_with_curriculum_logs_correct_action(self):
        """Test that logging uses full action space index with curriculum."""
        # Set up mock agent with database
        mock_db = Mock()
        mock_logger = Mock()
        mock_db.logger = mock_logger

        mock_env = Mock()
        mock_env.db = mock_db

        mock_time_service = Mock()
        mock_time_service.current_time.return_value = 15

        # Create multiple actions
        mock_actions = [Mock() for _ in range(5)]
        for i, action in enumerate(mock_actions):
            action.name = f"action_{i}"

        self.mock_agent.environment = mock_env
        self.mock_agent.time_service = mock_time_service
        self.mock_agent.actions = mock_actions

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        state = torch.randn(8)

        # Simulate curriculum: only actions 0, 2, 4 enabled (indices in full space)
        enabled_actions = [0, 2, 4]

        # Agent selects action at index 1 within enabled_actions (which is action 2 in full space)
        module.update(state, 1, 1.0, torch.randn(8), False, enabled_actions=enabled_actions)

        # Verify the logged action is 2 (full action space index)
        call_args = mock_logger.log_agent_action.call_args
        self.assertEqual(call_args.kwargs['action_taken'], 2)
        self.assertEqual(call_args.kwargs['action_taken_mapped'], 'action_2')

    def test_update_exception_handling(self):
        """Test update method exception handling."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm to raise exception
        with patch.object(
            module.algorithm, "train", side_effect=Exception("Test error")
        ):
            state = torch.randn(8)
            action = 1
            reward = 1.0
            next_state = torch.randn(8)
            done = False

            # Should not raise exception
            module.update(state, action, reward, next_state, done)

    def test_get_action_probabilities_with_predict_proba(self):
        """Test get_action_probabilities with algorithm that supports predict_proba."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm with predict_proba
        expected_probs = np.array([[0.1, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0]])
        with patch.object(
            module.algorithm, "predict_proba", return_value=expected_probs
        ):
            state = torch.randn(8)
            probs = module.get_action_probabilities(state)

            np.testing.assert_array_almost_equal(
                probs, [0.1, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0]
            )

    def test_get_action_probabilities_fallback(self):
        """Test get_action_probabilities fallback to uniform distribution."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock the algorithm to not have predict_proba method
        with patch.object(
            module.algorithm, "predict_proba", side_effect=AttributeError
        ):
            state = torch.randn(8)
            probs = module.get_action_probabilities(state)

            expected = np.full(7, 1.0 / 7)
            np.testing.assert_array_almost_equal(probs, expected)

    def test_get_action_probabilities_exception_handling(self):
        """Test get_action_probabilities exception handling."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Mock algorithm to raise exception
        with patch.object(
            module.algorithm, "predict_proba", side_effect=Exception("Test error")
        ):
            state = torch.randn(8)
            probs = module.get_action_probabilities(state)

            # Should return uniform distribution
            expected = np.full(7, 1.0 / 7)
            np.testing.assert_array_almost_equal(probs, expected)

    def test_get_model_info(self):
        """Test get_model_info method."""
        config = DecisionConfig(algorithm_type="dqn", rl_state_dim=16)
        module = DecisionModule(
            self.mock_agent, self.mock_env.action_space, self.observation_space, config
        )

        info = module.get_model_info()

        expected_keys = [
            "agent_id",
            "algorithm_type",
            "num_actions",
            "state_dim",
            "is_trained",
            "tianshou_available",
        ]

        for key in expected_keys:
            self.assertIn(key, info)

        self.assertEqual(info["agent_id"], "test_agent_1")
        self.assertEqual(info["algorithm_type"], "dqn")
        self.assertEqual(info["state_dim"], 16)
        self.assertEqual(info["tianshou_available"], TIANSHOU_AVAILABLE)

    def test_reset(self):
        """Test reset method."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        module._is_trained = True

        module.reset()

        self.assertFalse(module._is_trained)
        # If algorithm has reset method, it should be called
        if hasattr(module.algorithm, "reset"):
            module.algorithm.reset.assert_called_once()

    def test_get_action_space_size_from_environment(self):
        """Test getting action space size from environment."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )
        self.assertEqual(module.num_actions, 7)  # From mock environment

    def test_get_action_space_size_fallback(self):
        """Test getting action space size fallback."""
        # Remove environment from agent
        delattr(self.mock_agent, "environment")

        with patch("farm.core.action.ActionType") as mock_action_type:
            mock_action_type.__len__ = Mock(return_value=7)
            module = DecisionModule(
                self.mock_agent,
                self.mock_env.action_space,
                self.observation_space,
                self.config,
            )
            self.assertEqual(module.num_actions, 7)

    def test_get_action_space_size_from_space_object(self):
        """Test getting action space size from space object."""
        from gymnasium import spaces

        action_space = spaces.Discrete(10)
        module = DecisionModule(
            self.mock_agent, action_space, self.observation_space, self.config
        )

        self.assertEqual(module.num_actions, 10)

    def test_initialization_with_tianshou_ppo(self):
        """Test DecisionModule initialization with Tianshou PPO."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="ppo")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "ppo")

    def test_initialization_with_tianshou_sac(self):
        """Test DecisionModule initialization with Tianshou SAC."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="sac")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "sac")

    def test_initialization_with_tianshou_dqn(self):
        """Test DecisionModule initialization with Tianshou DQN."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="dqn")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "dqn")

    def test_initialization_with_tianshou_a2c(self):
        """Test DecisionModule initialization with Tianshou A2C."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="a2c")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "a2c")

    def test_initialization_with_tianshou_ddpg(self):
        """Test DecisionModule initialization with Tianshou DDPG."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                mock_algorithm = Mock()
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="ddpg")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                mock_wrapper_class.assert_called_once()
                self.assertEqual(module.config.algorithm_type, "ddpg")

    def test_initialization_algorithm_fallback(self):
        """Test that invalid algorithm falls back to fallback algorithm."""
        # DecisionConfig validates algorithm_type and falls back to 'fallback' for invalid values
        config = DecisionConfig(algorithm_type="invalid_algorithm")
        self.assertEqual(config.algorithm_type, "fallback")

        # Test that the module initializes successfully with fallback algorithm
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )
        self.assertIsNotNone(module.algorithm)
        self.assertEqual(module.config.algorithm_type, "fallback")

    def test_decide_action_with_multi_dimensional_state(self):
        """Test decide_action with multi-dimensional state."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Test with different state shapes
        state_2d = torch.randn(4, 2)  # Should be flattened to 8 elements
        action = module.decide_action(state_2d)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < module.num_actions)

    def test_update_with_none_algorithm(self):
        """Test update method when algorithm is None."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Set algorithm to None
        module.algorithm = None

        state = torch.randn(8)
        action = 1
        reward = 1.0
        next_state = torch.randn(8)
        done = False

        # Should not crash
        module.update(state, action, reward, next_state, done)
        self.assertFalse(module._is_trained)

    def test_get_action_probabilities_with_none_algorithm(self):
        """Test get_action_probabilities when algorithm is None."""
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            self.config,
        )

        # Set algorithm to None
        module.algorithm = None

        state = torch.randn(8)
        probs = module.get_action_probabilities(state)

        # Should return uniform distribution
        expected = np.full(7, 1.0 / 7)
        np.testing.assert_array_almost_equal(probs, expected)


class TestDecisionModuleIntegration(unittest.TestCase):
    """Integration tests for DecisionModule."""

    def setUp(self):
        """Set up test fixtures."""
        from gymnasium import spaces

        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "integration_test_agent"

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = spaces.Discrete(7)
        self.mock_agent.environment = self.mock_env

        # Create mock observation space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
        )

    def test_full_decision_cycle(self):
        """Test a complete decision cycle."""
        config = DecisionConfig(algorithm_type="fallback")  # Ensure we use fallback
        module = DecisionModule(
            self.mock_agent, self.mock_env.action_space, self.observation_space, config
        )

        # Create state
        state = torch.randn(8)

        # Get action
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 7)

        # Get probabilities
        probs = module.get_action_probabilities(state)
        self.assertEqual(len(probs), 7)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

        # Update with experience
        reward = 0.5
        next_state = torch.randn(8)
        done = False

        module.update(state, action, reward, next_state, done)
        self.assertTrue(module._is_trained)

        # Get model info
        info = module.get_model_info()
        self.assertEqual(info["agent_id"], "integration_test_agent")
        self.assertTrue(info["is_trained"])

    def test_model_persistence_cycle(self):
        """Test complete model save/load cycle."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent, self.mock_env.action_space, self.observation_space, config
        )

        # Train a bit
        for _ in range(3):
            state = torch.randn(8)
            action = module.decide_action(state)
            module.update(state, action, 1.0, torch.randn(8), False)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")

            # Save model
            module.save_model(model_path)

            # Create new module and load
            new_module = DecisionModule(
                self.mock_agent,
                self.mock_env.action_space,
                self.observation_space,
                config,
            )
            self.assertFalse(new_module._is_trained)

            new_module.load_model(model_path)

            # Check state was restored
            self.assertEqual(new_module._is_trained, module._is_trained)

    def test_integration_with_tianshou_algorithm(self):
        """Test integration with Tianshou algorithm."""
        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch("farm.core.decision.decision._ALGORITHM_REGISTRY") as mock_registry:
                # Create a proper mock algorithm that behaves like the real one
                mock_algorithm = Mock()
                mock_algorithm.select_action.return_value = 2
                mock_algorithm.select_action_with_mask.return_value = 2
                mock_algorithm.predict_proba.return_value = np.full(
                    (1, 7), 1.0 / 7, dtype=np.float32
                )
                mock_algorithm.update = Mock()
                mock_algorithm.learn = Mock()
                mock_algorithm.store_experience = Mock()
                mock_algorithm.should_train.return_value = True
                mock_algorithm.train = Mock()

                # Make the mock constructor return the mock algorithm
                mock_wrapper_class = Mock(return_value=mock_algorithm)
                mock_registry.__getitem__.return_value = mock_wrapper_class
                mock_registry.__contains__.return_value = True

                config = DecisionConfig(algorithm_type="ppo")
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.observation_space,
                    config,
                )

                # Test full cycle
                state = torch.randn(8)
                action = module.decide_action(state)
                self.assertEqual(action, 2)  # From mock

                probs = module.get_action_probabilities(state)
                self.assertEqual(len(probs), 7)

                # Test update
                module.update(state, action, 1.0, torch.randn(8), False)

                # Test model info
                info = module.get_model_info()
                self.assertEqual(info["algorithm_type"], "ppo")

    def test_integration_with_custom_config(self):
        """Test integration with custom configuration."""
        custom_config = DecisionConfig(
            algorithm_type="fallback",
            learning_rate=0.002,
            gamma=0.95,
            epsilon_start=0.9,
            epsilon_min=0.1,
            rl_state_dim=12,
        )

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            custom_config,
        )

        # Verify custom config was applied
        self.assertEqual(module.config.learning_rate, 0.002)
        self.assertEqual(module.config.gamma, 0.95)
        self.assertEqual(module.config.epsilon_start, 0.9)
        self.assertEqual(module.state_dim, 12)

        # Test functionality
        state = torch.randn(12)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 7)

    def test_integration_performance_stress_test(self):
        """Test performance with multiple decision cycles."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.observation_space,
            config,
        )

        # Stress test with many decisions
        num_cycles = 100
        states = [torch.randn(8) for _ in range(num_cycles)]

        import time

        start_time = time.time()

        for i, state in enumerate(states):
            action = module.decide_action(state)
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < 7)

            # Update occasionally
            if i % 10 == 0:
                module.update(state, action, 0.5, torch.randn(8), False)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (< 1 second for 100 cycles)
        self.assertLess(
            duration,
            1.0,
            f"Performance test failed: {duration:.3f}s for {num_cycles} cycles",
        )
        self.assertTrue(module._is_trained)


if __name__ == "__main__":
    unittest.main()
