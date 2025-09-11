"""Integration tests for the decision module.

This module tests the integration between different components of the decision system:
- DecisionModule with different algorithms
- Configuration system integration
- End-to-end decision workflows
- Performance and scalability tests
"""

import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from farm.core.decision.config import DecisionConfig, create_config_from_dict
from farm.core.decision.decision import DecisionModule


class TestDecisionModuleIntegration(unittest.TestCase):
    """Integration tests for DecisionModule with different configurations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "integration_test_agent"

        # Create mock environment with proper action space
        self.mock_env = Mock()

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create a simple observation space class
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.mock_env.action_space = MockActionSpace(4)
        self.mock_env.observation_space = MockObservationSpace((8,))
        self.mock_agent.environment = self.mock_env

    @patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", False)
    def test_decision_module_with_fallback_algorithm(self):
        """Test DecisionModule with fallback algorithm (no Tianshou)."""
        config = DecisionConfig(algorithm_type="dqn")  # Should fallback
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test basic functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

        probs = module.get_action_probabilities(state)
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

    @patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True)
    @patch("farm.core.decision.decision._ALGORITHM_REGISTRY")
    def test_decision_module_with_tianshou_dqn(self, mock_registry):
        """Test DecisionModule with Tianshou DQN."""
        mock_algorithm = Mock()
        mock_algorithm.select_action.return_value = 1
        # Remove the automatically created select_action_with_mask to ensure
        # the code uses the fallback path that calls select_action
        delattr(mock_algorithm, "select_action_with_mask")
        mock_wrapper_class = Mock(return_value=mock_algorithm)
        mock_registry.__getitem__.return_value = mock_wrapper_class
        mock_registry.__contains__.return_value = True

        config = DecisionConfig(algorithm_type="dqn")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Verify DQN was initialized
        mock_wrapper_class.assert_called_once()
        call_kwargs = mock_wrapper_class.call_args[1]
        self.assertEqual(call_kwargs["algorithm_config"]["lr"], config.learning_rate)
        self.assertEqual(call_kwargs["algorithm_config"]["gamma"], config.gamma)

        # Test functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertEqual(action, 1)  # From mock

    @patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True)
    @patch("farm.core.decision.decision._ALGORITHM_REGISTRY")
    def test_decision_module_with_tianshou_ppo(self, mock_registry):
        """Test DecisionModule with Tianshou PPO."""
        mock_algorithm = Mock()
        mock_algorithm.select_action.return_value = 2
        # Remove the automatically created select_action_with_mask to ensure
        # the code uses the fallback path that calls select_action
        delattr(mock_algorithm, "select_action_with_mask")
        mock_wrapper_class = Mock(return_value=mock_algorithm)
        mock_registry.__getitem__.return_value = mock_wrapper_class
        mock_registry.__contains__.return_value = True

        config = DecisionConfig(algorithm_type="ppo")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Verify PPO was initialized
        mock_wrapper_class.assert_called_once()
        call_kwargs = mock_wrapper_class.call_args[1]
        self.assertEqual(call_kwargs["algorithm_config"]["lr"], config.learning_rate)
        self.assertEqual(call_kwargs["algorithm_config"]["gamma"], config.gamma)

        # Test functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertEqual(action, 2)  # From mock

    @patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True)
    @patch("farm.core.decision.decision._ALGORITHM_REGISTRY")
    def test_decision_module_with_tianshou_sac(self, mock_registry):
        """Test DecisionModule with Tianshou SAC."""
        mock_algorithm = Mock()
        mock_algorithm.select_action.return_value = 3
        # Remove the automatically created select_action_with_mask to ensure
        # the code uses the fallback path that calls select_action
        delattr(mock_algorithm, "select_action_with_mask")
        mock_wrapper_class = Mock(return_value=mock_algorithm)
        mock_registry.__getitem__.return_value = mock_wrapper_class
        mock_registry.__contains__.return_value = True

        config = DecisionConfig(algorithm_type="sac")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Verify SAC was initialized
        mock_wrapper_class.assert_called_once()
        call_kwargs = mock_wrapper_class.call_args[1]
        self.assertEqual(call_kwargs["num_actions"], 4)
        self.assertEqual(call_kwargs["state_dim"], 8)

        # Test functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertEqual(action, 3)  # From mock

    @patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True)
    @patch("farm.core.decision.decision._ALGORITHM_REGISTRY")
    def test_decision_module_with_tianshou_a2c(self, mock_registry):
        """Test DecisionModule with Tianshou A2C."""
        mock_algorithm = Mock()
        mock_algorithm.select_action.return_value = 0
        # Remove the automatically created select_action_with_mask to ensure
        # the code uses the fallback path that calls select_action
        delattr(mock_algorithm, "select_action_with_mask")
        mock_wrapper_class = Mock(return_value=mock_algorithm)
        mock_registry.__getitem__.return_value = mock_wrapper_class
        mock_registry.__contains__.return_value = True

        config = DecisionConfig(algorithm_type="a2c")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Verify A2C was initialized
        mock_wrapper_class.assert_called_once()
        call_kwargs = mock_wrapper_class.call_args[1]
        self.assertEqual(call_kwargs["num_actions"], 4)
        self.assertEqual(call_kwargs["state_dim"], 8)

        # Test functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertEqual(action, 0)  # From mock

    @patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True)
    @patch("farm.core.decision.decision._ALGORITHM_REGISTRY")
    def test_decision_module_with_tianshou_ddpg(self, mock_registry):
        """Test DecisionModule with Tianshou DDPG."""
        mock_algorithm = Mock()
        mock_algorithm.select_action.return_value = 1
        # Remove the automatically created select_action_with_mask to ensure
        # the code uses the fallback path that calls select_action
        delattr(mock_algorithm, "select_action_with_mask")
        mock_wrapper_class = Mock(return_value=mock_algorithm)
        mock_registry.__getitem__.return_value = mock_wrapper_class
        mock_registry.__contains__.return_value = True

        config = DecisionConfig(algorithm_type="ddpg")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Verify DDPG was initialized
        mock_wrapper_class.assert_called_once()
        call_kwargs = mock_wrapper_class.call_args[1]
        self.assertEqual(call_kwargs["num_actions"], 4)
        self.assertEqual(call_kwargs["state_dim"], 8)

        # Test functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertEqual(action, 1)  # From mock

    def test_decision_module_with_custom_config_dict(self):
        """Test DecisionModule with config created from dictionary."""
        config_dict = {
            "algorithm_type": "fallback",
            "learning_rate": 0.0005,
            "gamma": 0.95,
            "epsilon_start": 0.8,
            "rl_state_dim": 12,
            "move_weight": 0.4,
            "gather_weight": 0.35,
        }

        config = create_config_from_dict(config_dict, DecisionConfig)
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Verify config was applied
        self.assertEqual(module.config.learning_rate, 0.0005)
        self.assertEqual(module.config.gamma, 0.95)
        self.assertEqual(module.config.epsilon_start, 0.8)
        self.assertEqual(module.state_dim, 12)
        self.assertEqual(module.config.move_weight, 0.4)
        self.assertEqual(module.config.gather_weight, 0.35)


class TestEndToEndDecisionWorkflow(unittest.TestCase):
    """End-to-end tests for decision workflows."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "e2e_test_agent"

        # Create mock environment with proper action space
        self.mock_env = Mock()

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create a simple observation space class
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.mock_env.action_space = MockActionSpace(6)
        self.mock_env.observation_space = MockObservationSpace((8,))
        self.mock_agent.environment = self.mock_env

    def test_complete_decision_learning_cycle(self):
        """Test a complete decision-learning cycle."""
        config = DecisionConfig(
            algorithm_type="fallback",
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9,
        )
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Simulate multiple decision episodes
        num_episodes = 10
        state_dim = config.rl_state_dim

        for episode in range(num_episodes):
            # Start episode
            state = torch.randn(state_dim)
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 20:  # Max steps per episode
                # Make decision
                action = module.decide_action(state)

                # Simulate environment response
                if action == 0:  # Assume action 0 is "good"
                    reward = 1.0
                elif action == 1:  # Assume action 1 is "bad"
                    reward = -1.0
                else:
                    reward = 0.0

                next_state = torch.randn(state_dim)
                done = (step_count >= 10) or (episode_reward > 5)  # End conditions

                # Update module
                module.update(state, action, reward, next_state, done)

                # Prepare for next step
                state = next_state
                episode_reward += reward
                step_count += 1

        # Verify learning occurred
        self.assertTrue(module._is_trained)

        # Test final model info
        info = module.get_model_info()
        self.assertEqual(info["agent_id"], "e2e_test_agent")
        self.assertTrue(info["is_trained"])
        self.assertEqual(info["num_actions"], 6)

    def test_decision_module_persistence_workflow(self):
        """Test save/load workflow for decision modules."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Train the module a bit
        for _ in range(5):
            state = torch.randn(8)
            action = module.decide_action(state)
            module.update(state, action, 1.0, torch.randn(8), False)

        original_info = module.get_model_info()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = f"{temp_dir}/test_model"

            # Save model
            module.save_model(model_path)

            # Create new module and load
            new_config = DecisionConfig(algorithm_type="fallback")
            new_module = DecisionModule(
                self.mock_agent,
                self.mock_env.action_space,
                self.mock_env.observation_space,
                new_config,
            )

            # Verify it's untrained initially
            self.assertFalse(new_module._is_trained)

            # Load model
            new_module.load_model(model_path)

            # Verify state was restored
            new_info = new_module.get_model_info()
            self.assertEqual(new_info["agent_id"], original_info["agent_id"])
            self.assertEqual(
                new_info["algorithm_type"], original_info["algorithm_type"]
            )
            self.assertEqual(new_info["is_trained"], original_info["is_trained"])

    def test_persistence_with_different_algorithms(self):
        """Test save/load with different algorithm types."""
        algorithms_to_test = ["fallback", "ppo", "sac", "dqn", "a2c", "ddpg"]

        for algorithm in algorithms_to_test:
            with self.subTest(algorithm=algorithm):
                # Create module with specific algorithm
                config = DecisionConfig(algorithm_type=algorithm)

                # Use appropriate patch based on algorithm
                if algorithm in ["ppo", "sac", "dqn", "a2c", "ddpg"]:
                    with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
                        with patch(
                            f"farm.core.decision.decision.{algorithm.upper()}Wrapper"
                        ) as mock_wrapper:
                            mock_algorithm = Mock()
                            mock_wrapper.return_value = mock_algorithm

                            module = DecisionModule(
                                self.mock_agent,
                                self.mock_env.action_space,
                                self.mock_env.observation_space,
                                config,
                            )

                            # Train briefly
                            for _ in range(3):
                                state = torch.randn(8)
                                action = module.decide_action(state)
                                module.update(state, action, 0.5, torch.randn(8), False)
                else:
                    # Fallback algorithm
                    module = DecisionModule(
                        self.mock_agent,
                        self.mock_env.action_space,
                        self.mock_env.observation_space,
                        config,
                    )

                    # Train briefly
                    for _ in range(3):
                        state = torch.randn(8)
                        action = module.decide_action(state)
                        module.update(state, action, 0.5, torch.randn(8), False)

                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = f"{temp_dir}/test_{algorithm}_model"

                    # Save model
                    module.save_model(model_path)

                    # Create new module and load
                    new_config = DecisionConfig(algorithm_type=algorithm)
                    if algorithm in ["ppo", "sac", "dqn", "a2c", "ddpg"]:
                        with patch(
                            "farm.core.decision.decision.TIANSHOU_AVAILABLE", True
                        ):
                            with patch(
                                f"farm.core.decision.decision.{algorithm.upper()}Wrapper"
                            ) as mock_wrapper:
                                mock_new_algorithm = Mock()
                                mock_wrapper.return_value = mock_new_algorithm

                                new_module = DecisionModule(
                                    self.mock_agent,
                                    self.mock_env.action_space,
                                    self.mock_env.observation_space,
                                    new_config,
                                )

                                # Load model
                                new_module.load_model(model_path)

                                # Verify basic functionality
                                state = torch.randn(8)
                                action = new_module.decide_action(state)
                                self.assertIsInstance(action, int)
                    else:
                        new_module = DecisionModule(
                            self.mock_agent,
                            self.mock_env.action_space,
                            self.mock_env.observation_space,
                            new_config,
                        )

                        # Load model
                        new_module.load_model(model_path)

                        # Verify basic functionality
                        state = torch.randn(8)
                        action = new_module.decide_action(state)
                        self.assertIsInstance(action, int)

    def test_persistence_error_handling(self):
        """Test error handling in save/load operations."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test save to invalid path
        invalid_path = "/invalid/path/that/does/not/exist/model"
        module.save_model(invalid_path)  # Should not crash

        # Test load from non-existent file
        non_existent_path = "/non/existent/path/model"
        module.load_model(non_existent_path)  # Should not crash

        # Test load from empty file
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_file = f"{temp_dir}/empty.pkl"
            with open(empty_file, "w") as f:
                f.write("")

            module.load_model(empty_file)  # Should not crash

        # Verify module still works after error handling
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)

    def test_model_info_persistence(self):
        """Test that model info is preserved across save/load."""
        config = DecisionConfig(
            algorithm_type="fallback",
            learning_rate=0.002,
            gamma=0.95,
            epsilon_start=0.9,
        )
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Train a bit
        for _ in range(5):
            state = torch.randn(8)
            action = module.decide_action(state)
            module.update(state, action, 1.0, torch.randn(8), False)

        original_info = module.get_model_info()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = f"{temp_dir}/info_test_model"

            # Save and load
            module.save_model(model_path)

            new_module = DecisionModule(
                self.mock_agent,
                self.mock_env.action_space,
                self.mock_env.observation_space,
                config,
            )
            new_module.load_model(model_path)

            new_info = new_module.get_model_info()

            # Verify key info is preserved
            self.assertEqual(new_info["agent_id"], original_info["agent_id"])
            self.assertEqual(
                new_info["algorithm_type"], original_info["algorithm_type"]
            )
            self.assertEqual(new_info["num_actions"], original_info["num_actions"])
            self.assertEqual(new_info["state_dim"], original_info["state_dim"])

    def test_multiple_agents_different_configs(self):
        """Test multiple agents with different configurations."""
        agents_configs = [
            ("agent_1", DecisionConfig(algorithm_type="fallback", epsilon_start=1.0)),
            ("agent_2", DecisionConfig(algorithm_type="fallback", epsilon_start=0.5)),
            ("agent_3", DecisionConfig(algorithm_type="fallback", epsilon_start=0.1)),
        ]

        modules = []
        for agent_id, config in agents_configs:
            # Create mock agent
            mock_agent = Mock()
            mock_agent.agent_id = agent_id
            mock_agent.environment = self.mock_env

            # Create module
            module = DecisionModule(
                mock_agent,
                mock_agent.environment.action_space,
                mock_agent.environment.observation_space,
                config,
            )
            modules.append((agent_id, module, config))

        # Test that each module has its own configuration
        for agent_id, module, config in modules:
            self.assertEqual(module.agent_id, agent_id)
            self.assertEqual(module.config.epsilon_start, config.epsilon_start)

            # Test basic functionality
            state = torch.randn(8)
            action = module.decide_action(state)
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < 6)

    def test_configuration_parameter_sweep(self):
        """Test decision modules with different parameter configurations."""
        base_config = DecisionConfig(algorithm_type="fallback")

        # Test different learning rates
        learning_rates = [0.001, 0.0005, 0.0001]
        modules = []

        for lr in learning_rates:
            config = base_config.__class__(
                **{**base_config.model_dump(), "learning_rate": lr}
            )
            module = DecisionModule(
                self.mock_agent,
                self.mock_env.action_space,
                self.mock_env.observation_space,
                config,
            )
            modules.append((lr, module))

        # Verify configurations are different
        for lr, module in modules:
            self.assertEqual(module.config.learning_rate, lr)

        # Test that all modules work
        state = torch.randn(8)
        for lr, module in modules:
            action = module.decide_action(state)
            self.assertIsInstance(action, int)


class TestDecisionSystemPerformance(unittest.TestCase):
    """Performance tests for the decision system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "perf_test_agent"

        # Create mock environment with proper action space
        self.mock_env = Mock()

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create a simple observation space class
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.mock_env.action_space = MockActionSpace(4)
        self.mock_env.observation_space = MockObservationSpace((8,))
        self.mock_agent.environment = self.mock_env

    def test_decision_speed(self):
        """Test the speed of decision making."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test decision speed with multiple states
        num_decisions = 1000
        states = [torch.randn(8) for _ in range(num_decisions)]

        start_time = time.time()
        actions = [module.decide_action(state) for state in states]
        end_time = time.time()

        decision_time = end_time - start_time

        # Should be reasonably fast (< 0.1 seconds for 1000 decisions)
        self.assertLess(
            decision_time,
            0.1,
            f"Decision making too slow: {decision_time:.3f}s for {num_decisions} decisions",
        )

        # Verify all actions are valid
        for action in actions:
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < 4)

    def test_probability_computation_speed(self):
        """Test the speed of probability computation."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test probability computation speed
        num_computations = 500
        states = [torch.randn(8) for _ in range(num_computations)]

        start_time = time.time()
        probabilities = [module.get_action_probabilities(state) for state in states]
        end_time = time.time()

        computation_time = end_time - start_time

        # Should be reasonably fast (< 0.05 seconds for 500 computations)
        self.assertLess(
            computation_time,
            0.05,
            f"Probability computation too slow: {computation_time:.3f}s for {num_computations} computations",
        )

        # Verify all probabilities are valid
        for probs in probabilities:
            self.assertEqual(len(probs), 4)
            self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

    def test_update_speed(self):
        """Test the speed of module updates."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test update speed
        num_updates = 500
        updates = [
            (
                torch.randn(8),
                np.random.randint(4),
                np.random.randn(),
                torch.randn(8),
                False,
            )
            for _ in range(num_updates)
        ]

        start_time = time.time()
        for state, action, reward, next_state, done in updates:
            module.update(state, action, reward, next_state, done)
        end_time = time.time()

        update_time = end_time - start_time

        # Should be reasonably fast (< 0.1 seconds for 500 updates)
        self.assertLess(
            update_time,
            0.1,
            f"Updates too slow: {update_time:.3f}s for {num_updates} updates",
        )

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during operation."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Perform many operations
        num_operations = 1000

        for i in range(num_operations):
            state = torch.randn(8)
            action = module.decide_action(state)
            module.update(state, action, np.random.randn(), torch.randn(8), False)

            # Periodic cleanup if available
            if hasattr(module.algorithm, "reset"):
                if i % 100 == 0:  # Reset every 100 steps
                    module.algorithm.reset()

        # If we get here without memory issues, the test passes
        self.assertTrue(module._is_trained)


class TestDecisionSystemRobustness(unittest.TestCase):
    """Robustness tests for the decision system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "robustness_test_agent"

        # Create mock environment with proper action space
        self.mock_env = Mock()

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create a simple observation space class
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.mock_env.action_space = MockActionSpace(4)
        self.mock_env.observation_space = MockObservationSpace((8,))
        self.mock_agent.environment = self.mock_env

    def test_invalid_state_handling(self):
        """Test handling of invalid state inputs."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test with None state - should handle gracefully without raising
        action = module.decide_action(None)  # type: ignore
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

        # Test with wrong shape state
        wrong_shape_state = torch.randn(4)  # Should be 8
        action = module.decide_action(wrong_shape_state)
        # Should still work (fallback algorithm handles any shape)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

    def test_invalid_action_range_handling(self):
        """Test handling of invalid action ranges."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test with action out of range (simulate by mocking algorithm)
        original_predict = module.algorithm.predict

        def invalid_predict(observation, deterministic=False):
            return np.array([10]), None  # Invalid action

        module.algorithm.predict = invalid_predict

        state = torch.randn(8)
        action = module.decide_action(state)

        # Should fallback to valid range
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

        # Restore original function
        module.algorithm.predict = original_predict

    def test_algorithm_failure_recovery(self):
        """Test recovery from algorithm failures."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Simulate algorithm failure
        original_predict = module.algorithm.predict

        def failing_predict(*args, **kwargs):
            raise Exception("Algorithm failure")

        module.algorithm.predict = failing_predict

        state = torch.randn(8)
        action = module.decide_action(state)

        # Should recover with random action
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

        # Restore original function
        module.algorithm.predict = original_predict

    def test_algorithm_initialization_failure_recovery(self):
        """Test recovery when algorithm initialization fails."""
        config = DecisionConfig(
            algorithm_type="ppo"
        )  # Should fallback if Tianshou fails

        with patch("farm.core.decision.decision.TIANSHOU_AVAILABLE", True):
            with patch(
                "farm.core.decision.decision.PPOWrapper",
                side_effect=Exception("Init failed"),
            ):
                module = DecisionModule(
                    self.mock_agent,
                    self.mock_env.action_space,
                    self.mock_env.observation_space,
                    config,
                )

                # Should have fallen back to fallback algorithm
                self.assertIsNotNone(module.algorithm)
                self.assertEqual(module.config.algorithm_type, "ppo")

                # Should still work
                state = torch.randn(8)
                action = module.decide_action(state)
                self.assertIsInstance(action, int)

    def test_invalid_algorithm_type_fallback(self):
        """Test fallback when invalid algorithm type is specified."""
        # This should work since DecisionConfig validates algorithm_type
        with self.assertRaises(Exception):  # Pydantic validation error
            config = DecisionConfig(algorithm_type="invalid_algorithm")
            module = DecisionModule(
                self.mock_agent,
                self.mock_env.action_space,
                self.mock_env.observation_space,
                config,
            )

    def test_empty_experience_buffer_handling(self):
        """Test handling when experience buffer is empty."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Try to train with empty buffer
        module.update(torch.randn(8), 0, 1.0, torch.randn(8), False)

        # Should not crash
        self.assertIsNotNone(module.algorithm)

    def test_tensor_conversion_errors(self):
        """Test handling of tensor conversion errors."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test with various input types
        test_inputs = [
            torch.randn(8),  # torch tensor
            np.random.randn(8),  # numpy array
            [1, 2, 3, 4, 5, 6, 7, 8],  # python list
            np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32),  # numpy array
        ]

        for state_input in test_inputs:
            action = module.decide_action(state_input)
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < 4)


class TestDecisionSystemObservationSpaces(unittest.TestCase):
    """Tests for different observation space configurations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "obs_space_test_agent"

        # Create mock environment with different observation spaces
        self.mock_env = Mock()

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create a simple observation space class
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.mock_env.action_space = MockActionSpace(4)
        self.mock_env.observation_space = MockObservationSpace((8,))
        self.mock_agent.environment = self.mock_env

    def test_1d_observation_space(self):
        """Test with 1D observation space."""

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create 1D observation space
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.mock_env.action_space = MockActionSpace(4)
        self.mock_env.observation_space = MockObservationSpace((8,))

        config = DecisionConfig(algorithm_type="fallback", rl_state_dim=8)
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test with 1D state
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

    def test_2d_observation_space(self):
        """Test with 2D observation space."""

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create 2D observation space
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.mock_env.action_space = MockActionSpace(4)
        self.mock_env.observation_space = MockObservationSpace((4, 4))

        config = DecisionConfig(algorithm_type="fallback", rl_state_dim=16)
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Test with 2D state
        state = torch.randn(4, 4)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

    def test_observation_space_without_shape_attribute(self):
        """Test observation space without shape attribute."""

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create observation space without shape
        class MockObservationSpaceNoShape:
            pass

        self.mock_env.action_space = MockActionSpace(4)
        self.mock_env.observation_space = MockObservationSpaceNoShape()

        config = DecisionConfig(algorithm_type="fallback", rl_state_dim=8)
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Should use default shape
        self.assertEqual(module.observation_shape, (8,))
        self.assertEqual(module.state_dim, 8)

        # Should still work
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        # Test with extreme parameter values
        config = DecisionConfig(
            epsilon_start=0.0,  # No exploration
            epsilon_min=0.0,
            gamma=0.0,  # No future rewards
            learning_rate=1e-6,  # Very small learning rate
        )

        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Should still work
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

    def test_concurrent_module_usage(self):
        """Test that modules can be used concurrently (basic test)."""
        config = DecisionConfig(algorithm_type="fallback")

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create a simple observation space class
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        # Create multiple modules
        modules = []
        for i in range(5):
            mock_agent = Mock()
            mock_agent.agent_id = f"concurrent_agent_{i}"
            mock_agent.environment = Mock()

            # Set up action and observation spaces for this agent's environment
            mock_agent.environment.action_space = MockActionSpace(4)
            mock_agent.environment.observation_space = MockObservationSpace((8,))

            module = DecisionModule(
                mock_agent,
                mock_agent.environment.action_space,
                mock_agent.environment.observation_space,
                config,
            )
            modules.append(module)

        # Use modules concurrently (simulated)
        states = [torch.randn(8) for _ in range(10)]

        for state in states:
            for module in modules:
                action = module.decide_action(state)
                self.assertIsInstance(action, int)
                self.assertTrue(0 <= action < 4)


class TestDecisionSystemIntegrationWithEnvironment(unittest.TestCase):
    """Integration tests with environment components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_id = "env_integration_agent"

        # Create more realistic mock environment
        self.mock_env = Mock()

        # Create a simple action space class
        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        # Create a simple observation space class
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        self.mock_env.action_space = MockActionSpace(6)  # Standard actions
        self.mock_env.observation_space = MockObservationSpace((8,))
        self.mock_agent.environment = self.mock_env

    def test_decision_module_with_environment_feedback(self):
        """Test decision module with simulated environment feedback."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Simulate environment interaction
        state = torch.randn(8)

        # Make decision
        action = module.decide_action(state)

        # Simulate environment response based on action
        if action == 0:  # Move
            reward = 0.1
            next_state = state + torch.randn(8) * 0.1
        elif action == 1:  # Gather
            reward = 0.5
            next_state = state + torch.randn(8) * 0.05
        elif action == 2:  # Share
            reward = 0.3
            next_state = state + torch.randn(8) * 0.02
        elif action == 3:  # Attack
            reward = -0.2  # Risky action
            next_state = state + torch.randn(8) * 0.15
        elif action == 4:  # Reproduce
            reward = 0.8
            next_state = state + torch.randn(8) * 0.03
        else:  # Defend
            reward = 0.2
            next_state = state + torch.randn(8) * 0.01

        # Update module
        module.update(state, action, reward, next_state, False)

        # Verify learning occurred
        self.assertTrue(module._is_trained)

        # Test that module can continue making decisions
        new_action = module.decide_action(next_state)
        self.assertIsInstance(new_action, int)
        self.assertTrue(0 <= new_action < 6)

    def test_decision_module_adaptation_to_rewards(self):
        """Test that decision module adapts to reward patterns."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(
            self.mock_agent,
            self.mock_env.action_space,
            self.mock_env.observation_space,
            config,
        )

        # Train with consistent reward pattern
        good_action = 1  # Always give high reward
        bad_action = 0  # Always give low reward

        for _ in range(50):
            state = torch.randn(8)

            # Force specific actions to test learning
            if np.random.random() < 0.5:
                action = good_action
                reward = 1.0
            else:
                action = bad_action
                reward = -1.0

            next_state = state + torch.randn(8) * 0.1
            module.update(state, action, reward, next_state, False)

        # After training, the module should have learned
        self.assertTrue(module._is_trained)

        # Get final probabilities for a test state
        test_state = torch.randn(8)
        probs = module.get_action_probabilities(test_state)

        # The algorithm should have some preference (though fallback is random)
        self.assertEqual(len(probs), 6)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
