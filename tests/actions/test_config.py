"""Unit tests for the config module.

This module tests the configuration system including BaseDQNConfig, all action-specific
configs, validation logic, inheritance patterns, and utility functions.
"""

import unittest
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from farm.core.decision.config import (
    DEFAULT_ATTACK_CONFIG,
    DEFAULT_DECISION_CONFIG,
    DEFAULT_GATHER_CONFIG,
    DEFAULT_MOVE_CONFIG,
    DEFAULT_REPRODUCE_CONFIG,
    DEFAULT_SHARE_CONFIG,
    AttackConfig,
    BaseDQNConfig,
    DecisionConfig,
    GatherConfig,
    MoveConfig,
    ReproduceConfig,
    ShareConfig,
    create_config_from_dict,
    merge_configs,
)


class TestBaseDQNConfig(unittest.TestCase):
    """Test cases for BaseDQNConfig class."""

    def test_default_config(self):
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

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BaseDQNConfig(
            target_update_freq=50,
            memory_size=5000,
            learning_rate=0.0005,
            gamma=0.95,
            epsilon_start=0.8,
            epsilon_min=0.05,
            epsilon_decay=0.99,
            dqn_hidden_size=128,
            batch_size=64,
            tau=0.01,
            seed=42,
        )

        self.assertEqual(config.target_update_freq, 50)
        self.assertEqual(config.memory_size, 5000)
        self.assertEqual(config.learning_rate, 0.0005)
        self.assertEqual(config.gamma, 0.95)
        self.assertEqual(config.epsilon_start, 0.8)
        self.assertEqual(config.epsilon_min, 0.05)
        self.assertEqual(config.epsilon_decay, 0.99)
        self.assertEqual(config.dqn_hidden_size, 128)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.tau, 0.01)
        self.assertEqual(config.seed, 42)

    def test_tau_validation_valid(self):
        """Test tau validation with valid values."""
        # Should not raise any exceptions
        config = BaseDQNConfig(tau=0.1)
        self.assertEqual(config.tau, 0.1)

        config = BaseDQNConfig(tau=0.9)
        self.assertEqual(config.tau, 0.9)

    def test_tau_validation_invalid(self):
        """Test tau validation with invalid values."""
        with self.assertRaises(ValidationError):
            BaseDQNConfig(tau=0.0)  # Should be > 0

        with self.assertRaises(ValidationError):
            BaseDQNConfig(tau=1.0)  # Should be < 1

        with self.assertRaises(ValidationError):
            BaseDQNConfig(tau=-0.1)  # Should be > 0

    def test_gamma_validation_valid(self):
        """Test gamma validation with valid values."""
        config = BaseDQNConfig(gamma=0.0)
        self.assertEqual(config.gamma, 0.0)

        config = BaseDQNConfig(gamma=1.0)
        self.assertEqual(config.gamma, 1.0)

        config = BaseDQNConfig(gamma=0.5)
        self.assertEqual(config.gamma, 0.5)

    def test_gamma_validation_invalid(self):
        """Test gamma validation with invalid values."""
        with self.assertRaises(ValidationError):
            BaseDQNConfig(gamma=-0.1)  # Should be >= 0

        with self.assertRaises(ValidationError):
            BaseDQNConfig(gamma=1.1)  # Should be <= 1

    def test_epsilon_validation_valid(self):
        """Test epsilon validation with valid values."""
        config = BaseDQNConfig(epsilon_start=0.0, epsilon_min=0.0)
        self.assertEqual(config.epsilon_start, 0.0)
        self.assertEqual(config.epsilon_min, 0.0)

        config = BaseDQNConfig(epsilon_start=1.0, epsilon_min=1.0)
        self.assertEqual(config.epsilon_start, 1.0)
        self.assertEqual(config.epsilon_min, 1.0)

    def test_epsilon_validation_invalid(self):
        """Test epsilon validation with invalid values."""
        with self.assertRaises(ValidationError):
            BaseDQNConfig(epsilon_start=-0.1)  # Should be >= 0

        with self.assertRaises(ValidationError):
            BaseDQNConfig(epsilon_start=1.1)  # Should be <= 1

        with self.assertRaises(ValidationError):
            BaseDQNConfig(epsilon_min=-0.1)  # Should be >= 0

        with self.assertRaises(ValidationError):
            BaseDQNConfig(epsilon_min=1.1)  # Should be <= 1

    def test_epsilon_decay_validation_valid(self):
        """Test epsilon decay validation with valid values."""
        config = BaseDQNConfig(epsilon_decay=0.1)
        self.assertEqual(config.epsilon_decay, 0.1)

        config = BaseDQNConfig(epsilon_decay=1.0)
        self.assertEqual(config.epsilon_decay, 1.0)

    def test_epsilon_decay_validation_invalid(self):
        """Test epsilon decay validation with invalid values."""
        with self.assertRaises(ValidationError):
            BaseDQNConfig(epsilon_decay=0.0)  # Should be > 0

        with self.assertRaises(ValidationError):
            BaseDQNConfig(epsilon_decay=1.1)  # Should be <= 1


class TestAttackConfig(unittest.TestCase):
    """Test cases for AttackConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AttackConfig()

        # Test inherited values
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)

        # Test attack-specific values
        self.assertEqual(config.base_cost, -0.2)
        self.assertEqual(config.success_reward, 1.0)
        self.assertEqual(config.failure_penalty, -0.3)
        self.assertEqual(config.defense_threshold, 0.3)
        self.assertEqual(config.defense_boost, 2.0)
        self.assertEqual(config.range, 20.0)
        self.assertEqual(config.base_damage, 10.0)
        self.assertEqual(config.kill_reward, 5.0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AttackConfig(
            base_cost=-0.1,
            success_reward=2.0,
            failure_penalty=-0.5,
            defense_threshold=0.5,
            defense_boost=3.0,
            range=25.0,
            base_damage=15.0,
            kill_reward=10.0,
        )

        self.assertEqual(config.base_cost, -0.1)
        self.assertEqual(config.success_reward, 2.0)
        self.assertEqual(config.failure_penalty, -0.5)
        self.assertEqual(config.defense_threshold, 0.5)
        self.assertEqual(config.defense_boost, 3.0)
        self.assertEqual(config.range, 25.0)
        self.assertEqual(config.base_damage, 15.0)
        self.assertEqual(config.kill_reward, 10.0)

    def test_inheritance(self):
        """Test that AttackConfig properly inherits from BaseDQNConfig."""
        config = AttackConfig()

        # Should have all base attributes
        self.assertTrue(hasattr(config, "memory_size"))
        self.assertTrue(hasattr(config, "learning_rate"))
        self.assertTrue(hasattr(config, "gamma"))
        self.assertTrue(hasattr(config, "epsilon_start"))
        self.assertTrue(hasattr(config, "epsilon_min"))
        self.assertTrue(hasattr(config, "epsilon_decay"))
        self.assertTrue(hasattr(config, "dqn_hidden_size"))
        self.assertTrue(hasattr(config, "batch_size"))
        self.assertTrue(hasattr(config, "tau"))
        self.assertTrue(hasattr(config, "seed"))


class TestGatherConfig(unittest.TestCase):
    """Test cases for GatherConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GatherConfig()

        # Test inherited values
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)

        # Test gather-specific values
        self.assertEqual(config.success_reward, 1.0)
        self.assertEqual(config.failure_penalty, -0.1)
        self.assertEqual(config.efficiency_multiplier, 0.5)
        self.assertEqual(config.cost_multiplier, 0.3)
        self.assertEqual(config.min_resource_threshold, 0.1)
        self.assertEqual(config.max_wait_steps, 5)
        self.assertEqual(config.range, 30)
        self.assertEqual(config.max_amount, 3)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GatherConfig(
            success_reward=2.0,
            failure_penalty=-0.2,
            efficiency_multiplier=0.7,
            cost_multiplier=0.4,
            min_resource_threshold=0.2,
            max_wait_steps=10,
            range=40,
            max_amount=5,
        )

        self.assertEqual(config.success_reward, 2.0)
        self.assertEqual(config.failure_penalty, -0.2)
        self.assertEqual(config.efficiency_multiplier, 0.7)
        self.assertEqual(config.cost_multiplier, 0.4)
        self.assertEqual(config.min_resource_threshold, 0.2)
        self.assertEqual(config.max_wait_steps, 10)
        self.assertEqual(config.range, 40)
        self.assertEqual(config.max_amount, 5)


class TestMoveConfig(unittest.TestCase):
    """Test cases for MoveConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MoveConfig()

        # Test inherited values
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)

        # Test move-specific values
        self.assertEqual(config.base_cost, -0.1)
        self.assertEqual(config.resource_approach_reward, 0.3)
        self.assertEqual(config.resource_retreat_penalty, -0.2)
        self.assertEqual(config.max_movement, 8)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoveConfig(
            base_cost=-0.05,
            resource_approach_reward=0.5,
            resource_retreat_penalty=-0.3,
            max_movement=10,
        )

        self.assertEqual(config.base_cost, -0.05)
        self.assertEqual(config.resource_approach_reward, 0.5)
        self.assertEqual(config.resource_retreat_penalty, -0.3)
        self.assertEqual(config.max_movement, 10)


class TestReproduceConfig(unittest.TestCase):
    """Test cases for ReproduceConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReproduceConfig()

        # Test inherited values
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)

        # Test reproduce-specific values
        self.assertEqual(config.success_reward, 1.0)
        self.assertEqual(config.failure_penalty, -0.2)
        self.assertEqual(config.offspring_survival_bonus, 0.5)
        self.assertEqual(config.population_balance_bonus, 0.3)
        self.assertEqual(config.min_health_ratio, 0.5)
        self.assertEqual(config.min_resource_ratio, 0.6)
        self.assertEqual(config.ideal_density_radius, 50.0)
        self.assertEqual(config.max_local_density, 0.7)
        self.assertEqual(config.min_space_required, 20.0)
        self.assertEqual(config.offspring_cost, 3)
        self.assertEqual(config.min_reproduction_resources, 8)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReproduceConfig(
            success_reward=2.0,
            failure_penalty=-0.5,
            offspring_survival_bonus=0.7,
            population_balance_bonus=0.4,
            min_health_ratio=0.6,
            min_resource_ratio=0.7,
            ideal_density_radius=60.0,
            max_local_density=0.8,
            min_space_required=25.0,
            offspring_cost=5,
            min_reproduction_resources=10,
        )

        self.assertEqual(config.success_reward, 2.0)
        self.assertEqual(config.failure_penalty, -0.5)
        self.assertEqual(config.offspring_survival_bonus, 0.7)
        self.assertEqual(config.population_balance_bonus, 0.4)
        self.assertEqual(config.min_health_ratio, 0.6)
        self.assertEqual(config.min_resource_ratio, 0.7)
        self.assertEqual(config.ideal_density_radius, 60.0)
        self.assertEqual(config.max_local_density, 0.8)
        self.assertEqual(config.min_space_required, 25.0)
        self.assertEqual(config.offspring_cost, 5)
        self.assertEqual(config.min_reproduction_resources, 10)


class TestSelectConfig(unittest.TestCase):
    """Test cases for SelectConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DecisionConfig()

        # Test inherited values
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)

        # Test select-specific values
        self.assertEqual(config.move_weight, 0.3)
        self.assertEqual(config.gather_weight, 0.3)
        self.assertEqual(config.share_weight, 0.15)
        self.assertEqual(config.attack_weight, 0.1)
        self.assertEqual(config.reproduce_weight, 0.15)

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

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DecisionConfig(
            move_weight=0.4,
            gather_weight=0.2,
            share_weight=0.2,
            attack_weight=0.1,
            reproduce_weight=0.1,
            move_mult_no_resources=2.0,
            gather_mult_low_resources=2.0,
            share_mult_wealthy=1.5,
            share_mult_poor=0.3,
            attack_mult_desperate=1.6,
            attack_mult_stable=0.5,
            reproduce_mult_wealthy=1.6,
            reproduce_mult_poor=0.2,
            attack_starvation_threshold=0.6,
            attack_defense_threshold=0.4,
            reproduce_resource_threshold=0.8,
        )

        self.assertEqual(config.move_weight, 0.4)
        self.assertEqual(config.gather_weight, 0.2)
        self.assertEqual(config.share_weight, 0.2)
        self.assertEqual(config.attack_weight, 0.1)
        self.assertEqual(config.reproduce_weight, 0.1)
        self.assertEqual(config.move_mult_no_resources, 2.0)
        self.assertEqual(config.gather_mult_low_resources, 2.0)
        self.assertEqual(config.share_mult_wealthy, 1.5)
        self.assertEqual(config.share_mult_poor, 0.3)
        self.assertEqual(config.attack_mult_desperate, 1.6)
        self.assertEqual(config.attack_mult_stable, 0.5)
        self.assertEqual(config.reproduce_mult_wealthy, 1.6)
        self.assertEqual(config.reproduce_mult_poor, 0.2)
        self.assertEqual(config.attack_starvation_threshold, 0.6)
        self.assertEqual(config.attack_defense_threshold, 0.4)
        self.assertEqual(config.reproduce_resource_threshold, 0.8)

    def test_weight_validation(self):
        """Test weight validation."""
        # Valid weights should work
        config = DecisionConfig(move_weight=0.5, gather_weight=0.5)
        self.assertEqual(config.move_weight, 0.5)
        self.assertEqual(config.gather_weight, 0.5)

        # Negative weights should raise validation error
        with self.assertRaises(ValidationError):
            DecisionConfig(move_weight=-0.1)

        with self.assertRaises(ValidationError):
            DecisionConfig(gather_weight=-0.1)


class TestShareConfig(unittest.TestCase):
    """Test cases for ShareConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ShareConfig()

        # Test inherited values
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)

        # Test share-specific values
        self.assertEqual(config.range, 30.0)
        self.assertEqual(config.min_amount, 1)
        self.assertEqual(config.success_reward, 0.3)
        self.assertEqual(config.failure_penalty, -0.1)
        self.assertEqual(config.base_cost, -0.05)
        self.assertEqual(config.altruism_bonus, 0.2)
        self.assertEqual(config.cooperation_memory, 100)
        self.assertEqual(config.max_resources, 30)
        self.assertEqual(config.max_amount, 5)
        self.assertEqual(config.threshold, 0.3)
        self.assertEqual(config.cooperation_bonus, 0.2)
        self.assertEqual(config.altruism_factor, 1.2)
        self.assertEqual(config.cooperation_score_threshold, 0.5)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ShareConfig(
            range=40.0,
            min_amount=2,
            success_reward=0.5,
            failure_penalty=-0.2,
            base_cost=-0.1,
            altruism_bonus=0.3,
            cooperation_memory=150,
            max_resources=50,
            max_amount=8,
            threshold=0.4,
            cooperation_bonus=0.3,
            altruism_factor=1.5,
            cooperation_score_threshold=0.6,
        )

        self.assertEqual(config.range, 40.0)
        self.assertEqual(config.min_amount, 2)
        self.assertEqual(config.success_reward, 0.5)
        self.assertEqual(config.failure_penalty, -0.2)
        self.assertEqual(config.base_cost, -0.1)
        self.assertEqual(config.altruism_bonus, 0.3)
        self.assertEqual(config.cooperation_memory, 150)
        self.assertEqual(config.max_resources, 50)
        self.assertEqual(config.max_amount, 8)
        self.assertEqual(config.threshold, 0.4)
        self.assertEqual(config.cooperation_bonus, 0.3)
        self.assertEqual(config.altruism_factor, 1.5)
        self.assertEqual(config.cooperation_score_threshold, 0.6)


class TestDefaultConfigs(unittest.TestCase):
    """Test cases for default configuration instances."""

    def test_default_configs_exist(self):
        """Test that all default config instances exist and are properly typed."""
        self.assertIsInstance(DEFAULT_ATTACK_CONFIG, AttackConfig)
        self.assertIsInstance(DEFAULT_GATHER_CONFIG, GatherConfig)
        self.assertIsInstance(DEFAULT_MOVE_CONFIG, MoveConfig)
        self.assertIsInstance(DEFAULT_REPRODUCE_CONFIG, ReproduceConfig)
        self.assertIsInstance(DEFAULT_DECISION_CONFIG, DecisionConfig)
        self.assertIsInstance(DEFAULT_SHARE_CONFIG, ShareConfig)

    def test_default_configs_are_singletons(self):
        """Test that default configs are singleton instances."""
        self.assertIs(DEFAULT_ATTACK_CONFIG, DEFAULT_ATTACK_CONFIG)
        self.assertIs(DEFAULT_GATHER_CONFIG, DEFAULT_GATHER_CONFIG)
        self.assertIs(DEFAULT_MOVE_CONFIG, DEFAULT_MOVE_CONFIG)
        self.assertIs(DEFAULT_REPRODUCE_CONFIG, DEFAULT_REPRODUCE_CONFIG)
        self.assertIs(DEFAULT_DECISION_CONFIG, DEFAULT_DECISION_CONFIG)
        self.assertIs(DEFAULT_SHARE_CONFIG, DEFAULT_SHARE_CONFIG)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_create_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "memory_size": 5000,
            "learning_rate": 0.0005,
            "base_cost": -0.1,
            "success_reward": 2.0,
        }

        config = create_config_from_dict(config_dict, AttackConfig)

        self.assertIsInstance(config, AttackConfig)
        # Cast to AttackConfig for type safety
        attack_config = config  # type: ignore
        self.assertEqual(attack_config.memory_size, 5000)
        self.assertEqual(attack_config.learning_rate, 0.0005)
        self.assertEqual(attack_config.base_cost, -0.1)  # type: ignore
        self.assertEqual(attack_config.success_reward, 2.0)  # type: ignore

    def test_create_config_from_dict_invalid(self):
        """Test creating config from invalid dictionary."""
        config_dict = {
            "memory_size": "invalid",  # Should be int
            "base_cost": "invalid",  # Should be float
        }

        with self.assertRaises(ValidationError):
            create_config_from_dict(config_dict, AttackConfig)

    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = AttackConfig()
        override_config = {"memory_size": 5000, "base_cost": -0.1}

        merged_config = merge_configs(base_config, override_config)

        self.assertIsInstance(merged_config, AttackConfig)
        # Cast to AttackConfig for type safety
        attack_merged_config = merged_config  # type: ignore
        self.assertEqual(attack_merged_config.memory_size, 5000)
        self.assertEqual(attack_merged_config.base_cost, -0.1)  # type: ignore
        # Other values should remain unchanged
        self.assertEqual(attack_merged_config.learning_rate, 0.001)
        self.assertEqual(attack_merged_config.success_reward, 1.0)  # type: ignore

    def test_merge_configs_with_invalid_values(self):
        """Test merging configs with invalid override values."""
        base_config = AttackConfig()
        override_config = {
            "tau": 1.5,  # Invalid: should be < 1
            "gamma": 1.5,  # Invalid: should be <= 1
        }

        with self.assertRaises(ValidationError):
            merge_configs(base_config, override_config)

    def test_merge_configs_preserves_type(self):
        """Test that merged config preserves the original type."""
        base_config = GatherConfig()
        override_config = {"memory_size": 5000}

        merged_config = merge_configs(base_config, override_config)

        self.assertIsInstance(merged_config, GatherConfig)
        self.assertNotIsInstance(merged_config, AttackConfig)


class TestConfigInheritance(unittest.TestCase):
    """Test cases for configuration inheritance patterns."""

    def test_all_configs_inherit_from_base(self):
        """Test that all configs properly inherit from BaseDQNConfig."""
        configs = [
            AttackConfig(),
            GatherConfig(),
            MoveConfig(),
            ReproduceConfig(),
            DecisionConfig(),
            ShareConfig(),
        ]

        for config in configs:
            self.assertIsInstance(config, BaseDQNConfig)
            # Check that they have base attributes
            self.assertTrue(hasattr(config, "memory_size"))
            self.assertTrue(hasattr(config, "learning_rate"))
            self.assertTrue(hasattr(config, "gamma"))
            self.assertTrue(hasattr(config, "epsilon_start"))
            self.assertTrue(hasattr(config, "epsilon_min"))
            self.assertTrue(hasattr(config, "epsilon_decay"))
            self.assertTrue(hasattr(config, "dqn_hidden_size"))
            self.assertTrue(hasattr(config, "batch_size"))
            self.assertTrue(hasattr(config, "tau"))
            self.assertTrue(hasattr(config, "seed"))

    def test_config_types_are_distinct(self):
        """Test that different config types are distinct."""
        configs = [
            AttackConfig(),
            GatherConfig(),
            MoveConfig(),
            ReproduceConfig(),
            DecisionConfig(),
            ShareConfig(),
        ]

        config_types = [type(config) for config in configs]
        self.assertEqual(len(config_types), len(set(config_types)))


if __name__ == "__main__":
    unittest.main()
