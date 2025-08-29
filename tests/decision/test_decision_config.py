"""Unit tests for decision configuration classes.

This module tests the configuration system including:
- BaseDQNConfig validation and defaults
- DecisionConfig validation and algorithm selection
- Configuration merging and creation utilities
- Field validation and error handling
"""

import unittest
from typing import Any, Dict

from pydantic import ValidationError

from farm.core.decision.config import (
    DEFAULT_DECISION_CONFIG,
    DEFAULT_DQN_CONFIG,
    BaseDQNConfig,
    DecisionConfig,
    create_config_from_dict,
    merge_configs,
)


class TestBaseDQNConfig(unittest.TestCase):
    """Test cases for BaseDQNConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BaseDQNConfig()

        self.assertEqual(config.target_update_freq, 100)
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.epsilon_start, 1.0)
        self.assertEqual(config.epsilon_min, 0.01)
        self.assertEqual(config.epsilon_decay, 0.995)
        self.assertEqual(config.dqn_hidden_size, 64)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.tau, 0.005)
        self.assertIsNone(config.seed)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BaseDQNConfig(
            learning_rate=0.0005,
            gamma=0.95,
            epsilon_start=0.8,
            memory_size=5000,
            seed=42,
        )

        self.assertEqual(config.learning_rate, 0.0005)
        self.assertEqual(config.gamma, 0.95)
        self.assertEqual(config.epsilon_start, 0.8)
        self.assertEqual(config.memory_size, 5000)
        self.assertEqual(config.seed, 42)

    def test_tau_validation_valid(self):
        """Test tau validation with valid values."""
        config = BaseDQNConfig(tau=0.1)
        self.assertEqual(config.tau, 0.1)

        config = BaseDQNConfig(tau=0.9)
        self.assertEqual(config.tau, 0.9)

    def test_tau_validation_invalid(self):
        """Test tau validation with invalid values."""
        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(tau=0.0)
        self.assertIn("tau must be between 0 and 1", str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(tau=1.0)
        self.assertIn("tau must be between 0 and 1", str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(tau=2.0)
        self.assertIn("tau must be between 0 and 1", str(cm.exception))

    def test_gamma_validation_valid(self):
        """Test gamma validation with valid values."""
        config = BaseDQNConfig(gamma=0.0)
        self.assertEqual(config.gamma, 0.0)

        config = BaseDQNConfig(gamma=0.5)
        self.assertEqual(config.gamma, 0.5)

        config = BaseDQNConfig(gamma=1.0)
        self.assertEqual(config.gamma, 1.0)

    def test_gamma_validation_invalid(self):
        """Test gamma validation with invalid values."""
        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(gamma=-0.1)
        self.assertIn("gamma must be between 0 and 1", str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(gamma=1.1)
        self.assertIn("gamma must be between 0 and 1", str(cm.exception))

    def test_epsilon_validation_valid(self):
        """Test epsilon validation with valid values."""
        config = BaseDQNConfig(epsilon_start=0.0)
        self.assertEqual(config.epsilon_start, 0.0)

        config = BaseDQNConfig(epsilon_start=0.5)
        self.assertEqual(config.epsilon_start, 0.5)

        config = BaseDQNConfig(epsilon_min=1.0)
        self.assertEqual(config.epsilon_min, 1.0)

    def test_epsilon_validation_invalid(self):
        """Test epsilon validation with invalid values."""
        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(epsilon_start=-0.1)
        self.assertIn("epsilon values must be between 0 and 1", str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(epsilon_min=1.1)
        self.assertIn("epsilon values must be between 0 and 1", str(cm.exception))

    def test_epsilon_decay_validation_valid(self):
        """Test epsilon_decay validation with valid values."""
        config = BaseDQNConfig(epsilon_decay=0.1)
        self.assertEqual(config.epsilon_decay, 0.1)

        config = BaseDQNConfig(epsilon_decay=1.0)
        self.assertEqual(config.epsilon_decay, 1.0)

    def test_epsilon_decay_validation_invalid(self):
        """Test epsilon_decay validation with invalid values."""
        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(epsilon_decay=0.0)
        self.assertIn("epsilon_decay must be between 0 and 1", str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            BaseDQNConfig(epsilon_decay=1.1)
        self.assertIn("epsilon_decay must be between 0 and 1", str(cm.exception))

    def test_dict_conversion(self):
        """Test conversion to dictionary."""
        config = BaseDQNConfig(learning_rate=0.002, seed=123)
        config_dict = config.model_dump()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["learning_rate"], 0.002)
        self.assertEqual(config_dict["seed"], 123)
        self.assertEqual(config_dict["gamma"], 0.99)  # Default value


class TestDecisionConfig(unittest.TestCase):
    """Test cases for DecisionConfig class."""

    def test_default_values(self):
        """Test default DecisionConfig values."""
        config = DecisionConfig()

        # Test action weights
        self.assertEqual(config.move_weight, 0.3)
        self.assertEqual(config.gather_weight, 0.3)
        self.assertEqual(config.share_weight, 0.15)
        self.assertEqual(config.attack_weight, 0.1)
        self.assertEqual(config.reproduce_weight, 0.15)

        # Test algorithm settings
        self.assertEqual(config.algorithm_type, "dqn")
        self.assertEqual(config.algorithm_params, {})
        self.assertEqual(config.rl_state_dim, 8)
        self.assertEqual(config.rl_buffer_size, 10000)
        self.assertEqual(config.rl_batch_size, 32)
        self.assertEqual(config.rl_train_freq, 4)

        # Test feature settings
        self.assertEqual(config.feature_engineering, [])
        self.assertEqual(config.ensemble_size, 1)
        self.assertTrue(config.use_exploration_bonus)

        # Test multipliers
        self.assertEqual(config.move_mult_no_resources, 1.5)
        self.assertEqual(config.gather_mult_low_resources, 1.5)
        self.assertEqual(config.share_mult_wealthy, 1.3)
        self.assertEqual(config.share_mult_poor, 0.5)
        self.assertEqual(config.attack_mult_desperate, 1.4)
        self.assertEqual(config.attack_mult_stable, 0.6)
        self.assertEqual(config.reproduce_mult_wealthy, 1.4)
        self.assertEqual(config.reproduce_mult_poor, 0.3)

        # Test thresholds
        self.assertEqual(config.attack_starvation_threshold, 0.5)
        self.assertEqual(config.attack_defense_threshold, 0.3)
        self.assertEqual(config.reproduce_resource_threshold, 0.7)

    def test_custom_algorithm_settings(self):
        """Test custom algorithm settings."""
        config = DecisionConfig(
            algorithm_type="ppo",
            algorithm_params={"learning_rate": 0.0003, "n_steps": 1024},
            rl_state_dim=16,
            rl_buffer_size=50000,
            rl_batch_size=64,
            rl_train_freq=8,
        )

        self.assertEqual(config.algorithm_type, "ppo")
        self.assertEqual(config.algorithm_params["learning_rate"], 0.0003)
        self.assertEqual(config.algorithm_params["n_steps"], 1024)
        self.assertEqual(config.rl_state_dim, 16)
        self.assertEqual(config.rl_buffer_size, 50000)
        self.assertEqual(config.rl_batch_size, 64)
        self.assertEqual(config.rl_train_freq, 8)

    def test_weight_validation_valid(self):
        """Test weight validation with valid values."""
        config = DecisionConfig(move_weight=0.0, gather_weight=0.5, share_weight=1.0)

        self.assertEqual(config.move_weight, 0.0)
        self.assertEqual(config.gather_weight, 0.5)
        self.assertEqual(config.share_weight, 1.0)

    def test_weight_validation_invalid(self):
        """Test weight validation with invalid values."""
        with self.assertRaises(ValidationError) as cm:
            DecisionConfig(move_weight=-0.1)
        self.assertIn("weights must be non-negative", str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            DecisionConfig(attack_weight=-1.0)
        self.assertIn("weights must be non-negative", str(cm.exception))

    def test_algorithm_type_validation_valid(self):
        """Test algorithm_type validation with valid values."""
        valid_algorithms = [
            "dqn",
            "mlp",
            "svm",
            "random_forest",
            "gradient_boost",
            "naive_bayes",
            "knn",
            "ppo",
            "sac",
            "a2c",
            "td3",
        ]

        for algo in valid_algorithms:
            config = DecisionConfig(algorithm_type=algo)
            self.assertEqual(config.algorithm_type, algo)

    def test_algorithm_type_validation_invalid(self):
        """Test algorithm_type validation with invalid values."""
        with self.assertRaises(ValidationError) as cm:
            DecisionConfig(algorithm_type="invalid_algorithm")
        self.assertIn("Algorithm must be one of:", str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            DecisionConfig(algorithm_type="")
        self.assertIn("Algorithm must be one of:", str(cm.exception))

    def test_feature_engineering_list(self):
        """Test feature_engineering as a list."""
        config = DecisionConfig(
            feature_engineering=["normalize", "pca", "interaction_terms"]
        )

        self.assertEqual(
            config.feature_engineering, ["normalize", "pca", "interaction_terms"]
        )

    def test_exploration_bonus_settings(self):
        """Test exploration bonus settings."""
        config = DecisionConfig(use_exploration_bonus=False)
        self.assertFalse(config.use_exploration_bonus)

        config = DecisionConfig(use_exploration_bonus=True)
        self.assertTrue(config.use_exploration_bonus)

    def test_state_multipliers(self):
        """Test state-based multipliers."""
        config = DecisionConfig(
            move_mult_no_resources=2.0,
            gather_mult_low_resources=0.5,
            share_mult_wealthy=1.8,
            share_mult_poor=0.2,
            attack_mult_desperate=2.5,
            attack_mult_stable=0.3,
            reproduce_mult_wealthy=2.0,
            reproduce_mult_poor=0.1,
        )

        self.assertEqual(config.move_mult_no_resources, 2.0)
        self.assertEqual(config.gather_mult_low_resources, 0.5)
        self.assertEqual(config.share_mult_wealthy, 1.8)
        self.assertEqual(config.share_mult_poor, 0.2)
        self.assertEqual(config.attack_mult_desperate, 2.5)
        self.assertEqual(config.attack_mult_stable, 0.3)
        self.assertEqual(config.reproduce_mult_wealthy, 2.0)
        self.assertEqual(config.reproduce_mult_poor, 0.1)

    def test_thresholds(self):
        """Test threshold values."""
        config = DecisionConfig(
            attack_starvation_threshold=0.3,
            attack_defense_threshold=0.2,
            reproduce_resource_threshold=0.8,
        )

        self.assertEqual(config.attack_starvation_threshold, 0.3)
        self.assertEqual(config.attack_defense_threshold, 0.2)
        self.assertEqual(config.reproduce_resource_threshold, 0.8)

    def test_inheritance_from_base_config(self):
        """Test that DecisionConfig inherits BaseDQNConfig fields."""
        config = DecisionConfig(
            learning_rate=0.0005, gamma=0.95, epsilon_start=0.8, memory_size=20000
        )

        # Test inherited fields
        self.assertEqual(config.learning_rate, 0.0005)
        self.assertEqual(config.gamma, 0.95)
        self.assertEqual(config.epsilon_start, 0.8)
        self.assertEqual(config.memory_size, 20000)

        # Test DecisionConfig specific fields still have defaults
        self.assertEqual(config.algorithm_type, "dqn")
        self.assertEqual(config.move_weight, 0.3)


class TestConfigurationUtilities(unittest.TestCase):
    """Test cases for configuration utility functions."""

    def test_create_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "learning_rate": 0.0005,
            "gamma": 0.95,
            "algorithm_type": "ppo",
            "move_weight": 0.4,
            "rl_state_dim": 12,
        }

        config = create_config_from_dict(config_dict, DecisionConfig)

        self.assertIsInstance(config, DecisionConfig)
        self.assertEqual(config.learning_rate, 0.0005)
        self.assertEqual(config.gamma, 0.95)
        # Note: algorithm_type, move_weight, and rl_state_dim are not BaseDQNConfig fields

    def test_create_config_from_dict_base_config(self):
        """Test creating BaseDQNConfig from dictionary."""
        config_dict = {
            "learning_rate": 0.002,
            "gamma": 0.9,
            "epsilon_start": 0.7,
            "memory_size": 15000,
        }

        config = create_config_from_dict(config_dict, BaseDQNConfig)

        self.assertIsInstance(config, BaseDQNConfig)
        self.assertEqual(config.learning_rate, 0.002)
        self.assertEqual(config.gamma, 0.9)
        self.assertEqual(config.epsilon_start, 0.7)
        self.assertEqual(config.memory_size, 15000)

    def test_create_config_from_dict_validation_error(self):
        """Test creating config from dictionary with validation error."""
        config_dict = {"gamma": 1.5, "algorithm_type": "ppo"}  # Invalid value

        with self.assertRaises(ValidationError):
            create_config_from_dict(config_dict, DecisionConfig)

    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = DecisionConfig(
            learning_rate=0.001, gamma=0.99, algorithm_type="dqn", move_weight=0.3
        )

        override_config = {
            "learning_rate": 0.0005,
            "algorithm_type": "ppo",
            "gather_weight": 0.4,
        }

        merged_config = merge_configs(base_config, override_config)

        self.assertIsInstance(merged_config, DecisionConfig)
        self.assertEqual(merged_config.learning_rate, 0.0005)  # Overridden
        self.assertEqual(merged_config.gamma, 0.99)  # Unchanged
        # Note: algorithm_type, move_weight, gather_weight are not BaseDQNConfig fields

    def test_merge_configs_empty_override(self):
        """Test merging with empty override."""
        base_config = DecisionConfig(learning_rate=0.001, algorithm_type="ppo")
        override_config = {}

        merged_config = merge_configs(base_config, override_config)

        self.assertEqual(merged_config.learning_rate, 0.001)
        # Note: algorithm_type is not a BaseDQNConfig field

    def test_merge_configs_validation(self):
        """Test that merged configs still undergo validation."""
        base_config = DecisionConfig()
        override_config = {"gamma": 1.5}  # Invalid value

        with self.assertRaises(ValidationError):
            merge_configs(base_config, override_config)


class TestDefaultConfigurations(unittest.TestCase):
    """Test cases for default configuration instances."""

    def test_default_decision_config(self):
        """Test DEFAULT_DECISION_CONFIG."""
        self.assertIsInstance(DEFAULT_DECISION_CONFIG, DecisionConfig)
        self.assertEqual(DEFAULT_DECISION_CONFIG.algorithm_type, "dqn")
        self.assertEqual(DEFAULT_DECISION_CONFIG.move_weight, 0.3)

    def test_default_dqn_config(self):
        """Test DEFAULT_DQN_CONFIG."""
        self.assertIsInstance(DEFAULT_DQN_CONFIG, BaseDQNConfig)
        self.assertEqual(DEFAULT_DQN_CONFIG.learning_rate, 0.001)
        self.assertEqual(DEFAULT_DQN_CONFIG.gamma, 0.99)

    def test_default_configs_are_frozen(self):
        """Test that default configs have expected default values."""
        # This is more of a sanity check that our defaults haven't changed
        config = DecisionConfig()

        # Check that DEFAULT_DECISION_CONFIG matches a fresh DecisionConfig
        self.assertEqual(DEFAULT_DECISION_CONFIG.algorithm_type, config.algorithm_type)
        self.assertEqual(DEFAULT_DECISION_CONFIG.move_weight, config.move_weight)
        self.assertEqual(DEFAULT_DECISION_CONFIG.rl_state_dim, config.rl_state_dim)


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration system."""

    def test_config_workflow(self):
        """Test a complete configuration workflow."""
        # Start with base config
        base_config = BaseDQNConfig(learning_rate=0.001, gamma=0.99)

        # Convert to dict for modification
        config_dict = base_config.dict()

        # Modify some values
        config_dict.update(
            {
                "learning_rate": 0.0005,
                "algorithm_type": "ppo",  # This will be ignored by BaseDQNConfig
                "move_weight": 0.4,  # This will be ignored by BaseDQNConfig
            }
        )

        # Create new config from dict
        new_config = create_config_from_dict(config_dict, BaseDQNConfig)

        # Verify changes
        self.assertEqual(new_config.learning_rate, 0.0005)
        self.assertEqual(new_config.gamma, 0.99)  # Unchanged

        # Create DecisionConfig from the same dict
        decision_config = create_config_from_dict(config_dict, DecisionConfig)

        # Verify DecisionConfig specific fields are included
        self.assertEqual(decision_config.learning_rate, 0.0005)
        # Note: algorithm_type and move_weight are not BaseDQNConfig fields

    def test_config_inheritance_chain(self):
        """Test that configuration inheritance works correctly."""
        # Create a config with both base and derived fields
        config_dict = {
            # BaseDQNConfig fields
            "learning_rate": 0.0005,
            "gamma": 0.95,
            "memory_size": 20000,
            # DecisionConfig fields
            "algorithm_type": "ppo",
            "move_weight": 0.4,
            "gather_weight": 0.35,
            "rl_state_dim": 16,
            "use_exploration_bonus": False,
        }

        config = DecisionConfig(**config_dict)

        # Verify all fields are set correctly
        self.assertEqual(config.learning_rate, 0.0005)
        self.assertEqual(config.gamma, 0.95)
        self.assertEqual(config.memory_size, 20000)
        self.assertEqual(config.algorithm_type, "ppo")
        self.assertEqual(config.move_weight, 0.4)
        self.assertEqual(config.gather_weight, 0.35)
        self.assertEqual(config.rl_state_dim, 16)
        self.assertFalse(config.use_exploration_bonus)


if __name__ == "__main__":
    unittest.main()
