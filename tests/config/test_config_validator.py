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

    def test_validate_learning_parameters_invalid_memory_size(self):
        """Validation catches memory_size <= 0."""
        from farm.config.config import LearningConfig

        config = SimulationConfig(learning=LearningConfig(memory_size=0))
        is_valid, errors, _ = self.validator.validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("memory_size" in e["field"] for e in errors))

    def test_validate_learning_parameters_invalid_batch_size(self):
        """Validation catches batch_size <= 0."""
        from farm.config.config import LearningConfig

        config = SimulationConfig(learning=LearningConfig(batch_size=0))
        is_valid, errors, _ = self.validator.validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("batch_size" in e["field"] for e in errors))

    def test_validate_agent_settings_total_agents_zero(self):
        """Zero total agents triggers an error."""
        from farm.config.config import PopulationConfig

        config = SimulationConfig(
            population=PopulationConfig(
                system_agents=0,
                independent_agents=0,
                control_agents=0,
                agent_type_ratios={"SystemAgent": 0.34, "IndependentAgent": 0.33, "ControlAgent": 0.33},
            )
        )
        is_valid, errors, warnings = self.validator.validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("agent_counts" in e["field"] for e in errors))

    def test_validate_agent_settings_total_exceeds_max_population(self):
        """Total agents > max_population produces a warning."""
        from farm.config.config import PopulationConfig

        config = SimulationConfig(
            population=PopulationConfig(
                system_agents=60,
                independent_agents=60,
                control_agents=60,
                max_population=100,
                agent_type_ratios={"SystemAgent": 0.34, "IndependentAgent": 0.33, "ControlAgent": 0.33},
            )
        )
        _, _, warnings = self.validator.validate_config(config)
        self.assertTrue(any("agent_counts" in w["field"] for w in warnings))

    def test_validate_learning_epsilon_start_less_than_min(self):
        """epsilon_start < epsilon_min produces a warning."""
        from farm.config.config import LearningConfig

        config = SimulationConfig(
            learning=LearningConfig(epsilon_start=0.01, epsilon_min=0.5)
        )
        _, _, warnings = self.validator.validate_config(config)
        self.assertTrue(any("epsilon_start" in w["field"] for w in warnings))

    def test_validate_learning_batch_size_exceeds_memory(self):
        """batch_size > memory_size produces a warning."""
        from farm.config.config import LearningConfig

        config = SimulationConfig(
            learning=LearningConfig(memory_size=100, batch_size=200)
        )
        _, _, warnings = self.validator.validate_config(config)
        self.assertTrue(any("batch_size" in w["field"] for w in warnings))

    def test_validate_simulation_steps_exceeds_max(self):
        """simulation_steps > max_steps produces a warning."""
        config = SimulationConfig(simulation_steps=500, max_steps=100)
        _, _, warnings = self.validator.validate_config(config)
        self.assertTrue(any("simulation_steps" in w["field"] for w in warnings))

    def test_validate_max_steps_zero(self):
        """max_steps=0 is an error."""
        config = SimulationConfig(max_steps=0)
        is_valid, errors, _ = self.validator.validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("max_steps" in e["field"] for e in errors))

    def test_validate_db_cache_size_very_large_warning(self):
        """db_cache_size_mb > 10000 produces a warning."""
        from farm.config.config import DatabaseConfig

        config = SimulationConfig(database=DatabaseConfig(db_cache_size_mb=20000))
        _, _, warnings = self.validator.validate_config(config)
        self.assertTrue(any("db_cache_size_mb" in w["field"] for w in warnings))

    def test_validate_reproduction_resources_less_than_offspring_cost(self):
        """min_reproduction_resources < offspring_cost produces a warning."""
        from farm.config.config import AgentBehaviorConfig

        config = SimulationConfig(
            agent_behavior=AgentBehaviorConfig(
                min_reproduction_resources=5, offspring_cost=10
            )
        )
        _, _, warnings = self.validator.validate_config(config)
        self.assertTrue(
            any("min_reproduction_resources" in w["field"] for w in warnings)
        )
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


class TestConfigurationRecovery(unittest.TestCase):
    """Tests for ConfigurationRecovery static methods."""

    def test_create_fallback_config_returns_simulation_config(self):
        """create_fallback_config returns a valid SimulationConfig."""
        from farm.config.validation import ConfigurationRecovery

        config = ConfigurationRecovery.create_fallback_config({})
        self.assertIsInstance(config, SimulationConfig)

    def test_create_fallback_config_is_valid(self):
        """Fallback config passes basic validation."""
        from farm.config.validation import ConfigurationRecovery, ConfigurationValidator

        config = ConfigurationRecovery.create_fallback_config({})
        validator = ConfigurationValidator()
        is_valid, errors, _ = validator.validate_config(config)
        self.assertTrue(is_valid, f"Fallback config failed validation: {errors}")

    def test_attempt_config_repair_learning_rate(self):
        """Repair clamps invalid learning_rate to a valid value."""
        from farm.config.config import LearningConfig
        from farm.config.validation import ConfigurationRecovery

        bad_config = SimulationConfig(learning=LearningConfig(learning_rate=-0.5))
        errors = [{"field": "learning_rate", "value": -0.5}]
        repaired, actions = ConfigurationRecovery.attempt_config_repair(bad_config, errors)

        self.assertGreater(repaired.learning.learning_rate, 0.0)
        self.assertTrue(any("learning_rate" in a for a in actions))

    def test_attempt_config_repair_gamma(self):
        """Repair clamps gamma to [0, 1] range."""
        from farm.config.config import LearningConfig
        from farm.config.validation import ConfigurationRecovery

        bad_config = SimulationConfig(learning=LearningConfig(gamma=1.5))
        errors = [{"field": "gamma", "value": 1.5}]
        repaired, actions = ConfigurationRecovery.attempt_config_repair(bad_config, errors)

        self.assertLessEqual(repaired.learning.gamma, 1.0)

    def test_attempt_config_repair_max_population(self):
        """Repair sets max_population to a positive default."""
        from farm.config.config import PopulationConfig
        from farm.config.validation import ConfigurationRecovery

        bad_config = SimulationConfig(population=PopulationConfig(max_population=0))
        errors = [{"field": "max_population", "value": 0}]
        repaired, actions = ConfigurationRecovery.attempt_config_repair(bad_config, errors)

        self.assertGreater(repaired.population.max_population, 0)

    def test_attempt_config_repair_learning_rate_high(self):
        """Repair clamps high learning_rate (>1.0) to 0.1."""
        from farm.config.config import LearningConfig
        from farm.config.validation import ConfigurationRecovery

        bad_config = SimulationConfig(learning=LearningConfig(learning_rate=2.0))
        errors = [{"field": "learning_rate", "value": 2.0}]
        repaired, actions = ConfigurationRecovery.attempt_config_repair(bad_config, errors)

        self.assertLessEqual(repaired.learning.learning_rate, 1.0)

    def test_attempt_config_repair_memory_size(self):
        """Repair sets memory_size to a positive default."""
        from farm.config.config import LearningConfig
        from farm.config.validation import ConfigurationRecovery

        bad_config = SimulationConfig(learning=LearningConfig(memory_size=0))
        errors = [{"field": "memory_size", "value": 0}]
        repaired, actions = ConfigurationRecovery.attempt_config_repair(bad_config, errors)

        self.assertGreater(repaired.learning.memory_size, 0)

    def test_attempt_config_repair_batch_size(self):
        """Repair sets batch_size to a positive default."""
        from farm.config.config import LearningConfig
        from farm.config.validation import ConfigurationRecovery

        bad_config = SimulationConfig(learning=LearningConfig(batch_size=0))
        errors = [{"field": "batch_size", "value": 0}]
        repaired, actions = ConfigurationRecovery.attempt_config_repair(bad_config, errors)

        self.assertGreater(repaired.learning.batch_size, 0)


class TestValidateConfigFiles(unittest.TestCase):
    """Tests for validate_config_files function."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_validate_config_files_empty_dir(self):
        """Empty directory returns report with no files checked."""
        from farm.config.validation import validate_config_files

        report = validate_config_files(self.tmpdir)
        self.assertIn("config_dir", report)
        self.assertEqual(report["files_checked"], [])
        self.assertEqual(report["total_errors"], 0)

    def test_validate_config_files_with_default_yaml(self):
        """Directory with a valid default.yaml returns successful report."""
        import os
        from farm.config.validation import validate_config_files

        config = SimulationConfig()
        config.to_yaml(os.path.join(self.tmpdir, "default.yaml"))

        report = validate_config_files(self.tmpdir)
        self.assertIn("default.yaml", report["files_checked"])
        self.assertEqual(report["file_reports"]["default.yaml"]["valid"], True)

    def test_validate_config_files_with_invalid_default(self):
        """Invalid default.yaml is reported as having errors."""
        import os
        from farm.config.validation import validate_config_files

        # Write bad YAML that can't be parsed as SimulationConfig
        with open(os.path.join(self.tmpdir, "default.yaml"), "w") as f:
            f.write("[{invalid yaml\n")

        report = validate_config_files(self.tmpdir)
        self.assertGreater(report["total_errors"], 0)

    def test_validate_config_files_with_environment_dir(self):
        """Environment config files in environments/ subdir are validated."""
        import os
        from farm.config.validation import validate_config_files

        env_dir = os.path.join(self.tmpdir, "environments")
        os.makedirs(env_dir, exist_ok=True)
        config = SimulationConfig()
        config.to_yaml(os.path.join(env_dir, "test.yaml"))

        report = validate_config_files(self.tmpdir)
        self.assertTrue(any("environments" in f for f in report["files_checked"]))

    def test_validate_config_files_with_invalid_env_file(self):
        """Invalid YAML in environments/ subdir is captured as an error."""
        import os
        from farm.config.validation import validate_config_files

        env_dir = os.path.join(self.tmpdir, "environments")
        os.makedirs(env_dir, exist_ok=True)
        with open(os.path.join(env_dir, "bad.yaml"), "w") as f:
            f.write("[{invalid yaml\n")

        report = validate_config_files(self.tmpdir)
        self.assertGreater(report["total_errors"], 0)

    def test_validate_config_files_with_profiles_dir(self):
        """Profile config files in profiles/ subdir are validated."""
        import os
        from farm.config.validation import validate_config_files

        profile_dir = os.path.join(self.tmpdir, "profiles")
        os.makedirs(profile_dir, exist_ok=True)
        config = SimulationConfig()
        config.to_yaml(os.path.join(profile_dir, "fast.yaml"))

        report = validate_config_files(self.tmpdir)
        self.assertTrue(any("profiles" in f for f in report["files_checked"]))

    def test_validate_config_files_with_invalid_profile_file(self):
        """Invalid YAML in profiles/ subdir is captured as an error."""
        import os
        from farm.config.validation import validate_config_files

        profile_dir = os.path.join(self.tmpdir, "profiles")
        os.makedirs(profile_dir, exist_ok=True)
        with open(os.path.join(profile_dir, "bad.yaml"), "w") as f:
            f.write("[{invalid yaml\n")

        report = validate_config_files(self.tmpdir)
        self.assertGreater(report["total_errors"], 0)


class TestSafeConfigLoader(unittest.TestCase):
    """Tests for SafeConfigLoader class."""

    def setUp(self):
        from farm.config.validation import SafeConfigLoader
        self.loader = SafeConfigLoader()

    def test_validate_valid_config_dict(self):
        """A valid config dict validates successfully."""
        config = SimulationConfig()
        config_dict = config.to_dict()
        validated_dict, status = self.loader.validate_config_dict(config_dict)
        self.assertTrue(status["success"])
        self.assertEqual(status["errors"], [])

    def test_validate_invalid_config_dict_raises_without_autorepair(self):
        """Invalid config dict with strict_validation raises ValidationError."""
        from farm.config.config import EnvironmentConfig
        from farm.config.validation import ValidationError

        bad_config = SimulationConfig(environment=EnvironmentConfig(width=-1))
        bad_dict = bad_config.to_dict()
        with self.assertRaises(ValidationError):
            self.loader.validate_config_dict(bad_dict, strict_validation=True, auto_repair=False)

    def test_validate_invalid_config_dict_autorepair(self):
        """Invalid config dict with auto_repair=True returns a result without raising."""
        from farm.config.config import LearningConfig

        bad_config = SimulationConfig(learning=LearningConfig(learning_rate=-0.1))
        bad_dict = bad_config.to_dict()
        validated_dict, status = self.loader.validate_config_dict(
            bad_dict, strict_validation=False, auto_repair=True
        )
        # Either repaired or fallback used - no exception
        self.assertIn("success", status)

    def test_validate_config_dict_strict_mode_with_warnings(self):
        """Strict mode causes warnings to be treated as errors."""
        from farm.config.config import EnvironmentConfig
        from farm.config.validation import ValidationError

        # A config with huge dimensions triggers a warning
        big_env = SimulationConfig(environment=EnvironmentConfig(width=20000, height=20000))
        big_dict = big_env.to_dict()
        with self.assertRaises((ValidationError, Exception)):
            self.loader.validate_config_dict(big_dict, strict_validation=True, auto_repair=False)


if __name__ == '__main__':
    unittest.main()
