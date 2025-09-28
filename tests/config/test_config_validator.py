"""
Unit tests for ConfigurationValidator class.

This module tests the ConfigurationValidator functionality including:
- Basic property validation
- Agent settings validation
- Resource settings validation
- Learning parameter validation
- Environment settings validation
- Performance settings validation
- Business rule validation
- Strict mode validation
"""

import unittest
from farm.config import SimulationConfig
from farm.config.validation import ConfigurationValidator


class TestConfigurationValidator(unittest.TestCase):
    """Test cases for ConfigurationValidator class."""

    def setUp(self):
        """Set up test environment."""
        self.validator = ConfigurationValidator()
        self.valid_config = SimulationConfig()

    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        is_valid, errors, warnings = self.validator.validate_config(self.valid_config)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertGreaterEqual(len(warnings), 0)  # May have warnings

    def test_validate_basic_properties_invalid_dimensions(self):
        """Test validation of invalid environment dimensions."""
        from farm.config.config import EnvironmentConfig
        config = SimulationConfig(environment=EnvironmentConfig(width=0, height=100))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)
        self.assertIn('width/height', errors[0]['field'])
        self.assertIn('positive', errors[0]['error'])

        # Test negative dimensions
        config = SimulationConfig(environment=EnvironmentConfig(width=-10, height=100))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)

    def test_validate_agent_settings_invalid_ratios(self):
        """Test validation of invalid agent ratios."""
        from farm.config.config import PopulationConfig
        # Invalid agent ratios (out of bounds)
        config = SimulationConfig(population=PopulationConfig(agent_type_ratios={'SystemAgent': 2.0, 'IndependentAgent': -1.0, 'ControlAgent': 0.0}))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('agent_type_ratios' in error['field'] for error in errors))

    def test_validate_agent_settings_invalid_ratios_sum(self):
        """Test validation of agent ratios that don't sum to 1."""
        from farm.config.config import PopulationConfig
        config = SimulationConfig(population=PopulationConfig(agent_type_ratios={'SystemAgent': 0.3, 'IndependentAgent': 0.3}))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('sum to 1.0' in error['error'] for error in errors))

    def test_validate_resource_settings_invalid_resources(self):
        """Test validation of invalid resource settings."""
        from farm.config.config import ResourceConfig
        # Negative initial resources
        config = SimulationConfig(resources=ResourceConfig(initial_resources=-10))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('initial_resources' in error['field'] for error in errors))

        # Invalid regen rates
        config = SimulationConfig(resources=ResourceConfig(resource_regen_rate=-0.1))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('resource_regen_rate' in error['field'] for error in errors))

        # Negative regen amount
        config = SimulationConfig(resources=ResourceConfig(resource_regen_amount=-5))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('resource_regen_amount' in error['field'] for error in errors))

    def test_validate_learning_parameters_invalid_learning(self):
        """Test validation of invalid learning parameters."""
        from farm.config.config import LearningConfig
        # Invalid learning rate
        config = SimulationConfig(learning=LearningConfig(learning_rate=-0.1))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('learning_rate' in error['field'] for error in errors))

        # Invalid gamma
        config = SimulationConfig(learning=LearningConfig(gamma=1.5))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('gamma' in error['field'] for error in errors))

        # Invalid epsilon values
        config = SimulationConfig(learning=LearningConfig(epsilon_start=1.5))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('epsilon_start' in error['field'] for error in errors))

    def test_validate_environment_settings_invalid_environment(self):
        """Test validation of invalid environment settings."""
        from farm.config.config import AgentBehaviorConfig
        # Invalid perception radius
        config = SimulationConfig(agent_behavior=AgentBehaviorConfig(perception_radius=-1))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('perception_radius' in error['field'] for error in errors))

        # Invalid movement range
        config = SimulationConfig(agent_behavior=AgentBehaviorConfig(max_movement=0))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('max_movement' in error['field'] for error in errors))

        # Invalid consumption rate
        config = SimulationConfig(agent_behavior=AgentBehaviorConfig(base_consumption_rate=1.5))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('base_consumption_rate' in error['field'] for error in errors))

    def test_validate_performance_settings_invalid_performance(self):
        """Test validation of invalid performance settings."""
        from farm.config.config import DatabaseConfig
        # Invalid database cache size
        config = SimulationConfig(database=DatabaseConfig(db_cache_size_mb=0))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('db_cache_size_mb' in error['field'] for error in errors))

        # Invalid in-memory db memory limit
        config = SimulationConfig(database=DatabaseConfig(use_in_memory_db=True, in_memory_db_memory_limit_mb=-10))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('in_memory_db_memory_limit_mb' in error['field'] for error in errors))

    def test_validate_business_rules_invalid_rules(self):
        """Test validation of business rule violations."""
        from farm.config.config import AgentBehaviorConfig
        # Invalid reproduction resources
        config = SimulationConfig(agent_behavior=AgentBehaviorConfig(min_reproduction_resources=-10))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('min_reproduction_resources' in error['field'] for error in errors))

        # Invalid offspring cost
        config = SimulationConfig(agent_behavior=AgentBehaviorConfig(offspring_cost=0))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('offspring_cost' in error['field'] for error in errors))

        # Invalid starvation threshold
        config = SimulationConfig(agent_behavior=AgentBehaviorConfig(starvation_threshold=-1))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('starvation_threshold' in error['field'] for error in errors))

    def test_strict_mode_validation(self):
        """Test strict mode validation that treats warnings as errors."""
        # Create config that generates warnings (if any exist in valid config)
        config = SimulationConfig()

        # First validate in normal mode
        is_valid_normal, errors_normal, warnings_normal = self.validator.validate_config(config, strict=False)

        # Then validate in strict mode
        is_valid_strict, errors_strict, warnings_strict = self.validator.validate_config(config, strict=True)

        # In strict mode, warnings should become errors
        if warnings_normal:
            self.assertFalse(is_valid_strict)
            self.assertEqual(len(errors_strict), len(warnings_normal))
            self.assertEqual(len(warnings_strict), 0)
        else:
            self.assertEqual(is_valid_normal, is_valid_strict)

    def test_validator_reinitialization(self):
        """Test that validator clears errors/warnings between validations."""
        # First validation with invalid config
        from farm.config.config import EnvironmentConfig
        invalid_config = SimulationConfig(environment=EnvironmentConfig(width=-1))
        is_valid1, errors1, warnings1 = self.validator.validate_config(invalid_config)

        self.assertFalse(is_valid1)
        self.assertGreater(len(errors1), 0)

        # Second validation with valid config
        is_valid2, errors2, warnings2 = self.validator.validate_config(self.valid_config)

        self.assertTrue(is_valid2)
        self.assertEqual(len(errors2), 0)

        # Errors from first validation should not carry over
        self.assertNotEqual(len(errors1), len(errors2))

    def test_validate_agent_type_ratios_missing_keys(self):
        """Test validation when agent_type_ratios is missing required keys."""
        # This should fail because the sum validation will catch that it doesn't sum to 1.0
        # when missing keys (since missing keys default to 0)
        from farm.config.config import PopulationConfig
        config = SimulationConfig(population=PopulationConfig(agent_type_ratios={'SystemAgent': 0.5, 'IndependentAgent': 0.3}))  # Missing ControlAgent
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('agent_type_ratios' in error['field'] for error in errors))

    def test_validate_learning_parameters_invalid_decay(self):
        """Test validation of invalid epsilon decay values."""
        from farm.config.config import LearningConfig
        config = SimulationConfig(learning=LearningConfig(epsilon_decay=1.1))  # Decay > 1
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('epsilon_decay' in error['field'] for error in errors))

    def test_validate_resource_settings_max_amount(self):
        """Test validation of max resource amount."""
        from farm.config.config import ResourceConfig
        config = SimulationConfig(resources=ResourceConfig(max_resource_amount=-1))
        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertTrue(any('max_resource_amount' in error['field'] for error in errors))


if __name__ == '__main__':
    unittest.main()
