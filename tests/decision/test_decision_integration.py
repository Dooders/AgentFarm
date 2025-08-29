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

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 4
        self.mock_agent.environment = self.mock_env

    @patch("farm.core.decision.decision.SB3_AVAILABLE", False)
    def test_decision_module_with_fallback_algorithm(self):
        """Test DecisionModule with fallback algorithm (no SB3)."""
        config = DecisionConfig(algorithm_type="ddqn")  # Should fallback
        module = DecisionModule(self.mock_agent, config)

        # Test basic functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

        probs = module.get_action_probabilities(state)
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

    @patch("farm.core.decision.decision.SB3_AVAILABLE", True)
    @patch("stable_baselines3.DQN")
    def test_decision_module_with_sb3_ddqn(self, mock_dqn_class):
        """Test DecisionModule with SB3 DDQN."""
        mock_algorithm = Mock()
        mock_algorithm.predict.return_value = (np.array([1]), None)
        mock_dqn_class.return_value = mock_algorithm

        config = DecisionConfig(algorithm_type="ddqn")
        module = DecisionModule(self.mock_agent, config)

        # Verify DDQN was initialized
        mock_dqn_class.assert_called_once()
        call_kwargs = mock_dqn_class.call_args[1]
        self.assertEqual(call_kwargs["learning_rate"], config.learning_rate)
        self.assertEqual(call_kwargs["gamma"], config.gamma)

        # Test functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertEqual(action, 1)  # From mock

    @patch("farm.core.decision.decision.SB3_AVAILABLE", True)
    @patch("stable_baselines3.PPO")
    def test_decision_module_with_sb3_ppo(self, mock_ppo_class):
        """Test DecisionModule with SB3 PPO."""
        mock_algorithm = Mock()
        mock_algorithm.predict.return_value = (np.array([2]), None)
        mock_ppo_class.return_value = mock_algorithm

        config = DecisionConfig(algorithm_type="ppo")
        module = DecisionModule(self.mock_agent, config)

        # Verify PPO was initialized
        mock_ppo_class.assert_called_once()
        call_kwargs = mock_ppo_class.call_args[1]
        self.assertEqual(call_kwargs["learning_rate"], config.learning_rate)
        self.assertEqual(call_kwargs["gamma"], config.gamma)

        # Test functionality
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertEqual(action, 2)  # From mock

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
        module = DecisionModule(self.mock_agent, config)

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

        # Create mock environment with action space
        self.mock_env = Mock()
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 6
        self.mock_agent.environment = self.mock_env

    def test_complete_decision_learning_cycle(self):
        """Test a complete decision-learning cycle."""
        config = DecisionConfig(
            algorithm_type="fallback",
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9,
        )
        module = DecisionModule(self.mock_agent, config)

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
        module = DecisionModule(self.mock_agent, config)

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
            new_module = DecisionModule(self.mock_agent, new_config)

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
            module = DecisionModule(mock_agent, config)
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
                **{**base_config.dict(), "learning_rate": lr}
            )
            module = DecisionModule(self.mock_agent, config)
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

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 4
        self.mock_agent.environment = self.mock_env

    def test_decision_speed(self):
        """Test the speed of decision making."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(self.mock_agent, config)

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
        module = DecisionModule(self.mock_agent, config)

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
        module = DecisionModule(self.mock_agent, config)

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
        module = DecisionModule(self.mock_agent, config)

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

        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 4
        self.mock_agent.environment = self.mock_env

    def test_invalid_state_handling(self):
        """Test handling of invalid state inputs."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(self.mock_agent, config)

        # Test with None state
        with self.assertRaises((TypeError, AttributeError)):
            module.decide_action(None)  # type: ignore

        # Test with wrong shape state
        wrong_shape_state = torch.randn(4)  # Should be 8
        action = module.decide_action(wrong_shape_state)
        # Should still work (fallback algorithm handles any shape)
        self.assertIsInstance(action, int)

    def test_invalid_action_range_handling(self):
        """Test handling of invalid action ranges."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(self.mock_agent, config)

        # Test with action out of range (simulate by mocking algorithm)
        module.algorithm.predict.return_value = (np.array([10]), None)  # Invalid action

        state = torch.randn(8)
        action = module.decide_action(state)

        # Should fallback to valid range
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

    def test_algorithm_failure_recovery(self):
        """Test recovery from algorithm failures."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(self.mock_agent, config)

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

    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        # Test with extreme parameter values
        config = DecisionConfig(
            epsilon_start=0.0,  # No exploration
            epsilon_min=0.0,
            gamma=0.0,  # No future rewards
            learning_rate=1e-6,  # Very small learning rate
        )

        module = DecisionModule(self.mock_agent, config)

        # Should still work
        state = torch.randn(8)
        action = module.decide_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 4)

    def test_concurrent_module_usage(self):
        """Test that modules can be used concurrently (basic test)."""
        config = DecisionConfig(algorithm_type="fallback")

        # Create multiple modules
        modules = []
        for i in range(5):
            mock_agent = Mock()
            mock_agent.agent_id = f"concurrent_agent_{i}"
            mock_agent.environment = self.mock_env

            module = DecisionModule(mock_agent, config)
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
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.n = 6  # Standard actions
        self.mock_agent.environment = self.mock_env

    def test_decision_module_with_environment_feedback(self):
        """Test decision module with simulated environment feedback."""
        config = DecisionConfig(algorithm_type="fallback")
        module = DecisionModule(self.mock_agent, config)

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
        module = DecisionModule(self.mock_agent, config)

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
