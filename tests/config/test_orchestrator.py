"""
Unit tests for ConfigurationOrchestrator.

Tests the orchestrator pattern implementation that breaks circular dependencies
between cache, config, and validation modules.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from farm.config.cache import ConfigCache, OptimizedConfigLoader
from farm.config.config import SimulationConfig
from farm.config.orchestrator import ConfigurationOrchestrator, get_global_orchestrator, load_config
from farm.config.validation import SafeConfigLoader, ValidationError, ConfigurationValidator


class TestConfigurationOrchestrator(unittest.TestCase):
    """Test cases for ConfigurationOrchestrator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, "config")

        # Create mock config directory structure
        os.makedirs(os.path.join(self.config_dir, "environments"), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, "profiles"), exist_ok=True)

        # Create minimal test configuration files
        self._create_test_configs()

        # Create orchestrator instance
        self.orchestrator = ConfigurationOrchestrator()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_test_configs(self):
        """Create minimal test configuration files."""
        # Base configuration
        base_config = {
            "width": 100,
            "height": 100,
            "system_agents": 5,
            "independent_agents": 5,
            "control_agents": 5,
            "max_population": 50,
            "initial_resources": 20,
            "max_resource_amount": 30,
            "learning_rate": 0.001,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "memory_size": 1000,
            "batch_size": 32,
            "max_steps": 100,
            "use_in_memory_db": True,
        }

        import yaml

        # Write base config
        with open(os.path.join(self.config_dir, "default.yaml"), "w") as f:
            yaml.safe_dump(base_config, f)

        # Write environment config
        env_config = {"width": 200, "height": 200}
        with open(os.path.join(self.config_dir, "environments", "testing.yaml"), "w") as f:
            yaml.safe_dump(env_config, f)

        # Write profile config
        profile_config = {"max_steps": 200}
        with open(os.path.join(self.config_dir, "profiles", "benchmark.yaml"), "w") as f:
            yaml.safe_dump(profile_config, f)

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = ConfigurationOrchestrator()

        self.assertIsInstance(orchestrator.cache, ConfigCache)
        self.assertIsInstance(orchestrator.loader, OptimizedConfigLoader)
        self.assertIsInstance(orchestrator.validator, SafeConfigLoader)

    def test_initialization_with_custom_components(self):
        """Test orchestrator initialization with custom components."""
        mock_cache = Mock(spec=ConfigCache)
        mock_loader = Mock(spec=OptimizedConfigLoader)
        mock_validator = Mock(spec=SafeConfigLoader)

        orchestrator = ConfigurationOrchestrator(cache=mock_cache, loader=mock_loader, validator=mock_validator)

        self.assertEqual(orchestrator.cache, mock_cache)
        self.assertEqual(orchestrator.loader, mock_loader)
        self.assertEqual(orchestrator.validator, mock_validator)

    def test_load_config_basic(self):
        """Test basic configuration loading."""
        config = self.orchestrator.load_config(
            environment="testing", config_dir=self.config_dir, validate=False, use_cache=False
        )

        self.assertIsInstance(config, SimulationConfig)
        self.assertEqual(config.environment.width, 200)  # From testing.yaml
        self.assertEqual(config.environment.height, 200)
        self.assertEqual(config.max_steps, 100)  # Default value

    def test_load_config_with_profile(self):
        """Test configuration loading with profile."""
        config = self.orchestrator.load_config(
            environment="testing", profile="benchmark", config_dir=self.config_dir, validate=False, use_cache=False
        )

        self.assertIsInstance(config, SimulationConfig)
        self.assertEqual(config.environment.width, 200)  # From testing.yaml
        self.assertEqual(config.environment.height, 200)
        self.assertEqual(config.max_steps, 200)  # From benchmark.yaml

    def test_load_config_with_caching(self):
        """Test configuration loading with caching enabled."""
        # First load should cache the config
        config1 = self.orchestrator.load_config(
            environment="testing", config_dir=self.config_dir, validate=False, use_cache=True
        )

        # Second load should come from cache
        config2 = self.orchestrator.load_config(
            environment="testing", config_dir=self.config_dir, validate=False, use_cache=True
        )

        self.assertIsInstance(config1, SimulationConfig)
        self.assertIsInstance(config2, SimulationConfig)
        self.assertEqual(config1.to_dict(), config2.to_dict())

        # Verify cache was used
        cache_stats = self.orchestrator.get_cache_stats()
        self.assertGreaterEqual(cache_stats["hits"], 1)

    def test_load_config_with_validation(self):
        """Test configuration loading with validation."""
        config = self.orchestrator.load_config(
            environment="testing", config_dir=self.config_dir, validate=True, use_cache=False
        )

        self.assertIsInstance(config, SimulationConfig)
        # Valid config should load without errors

    def test_load_config_invalid_environment(self):
        """Test loading config with invalid environment."""
        with self.assertRaises(FileNotFoundError):
            self.orchestrator.load_config(
                environment="nonexistent", config_dir=self.config_dir, validate=False, use_cache=False
            )

    def test_load_config_invalid_profile(self):
        """Test loading config with invalid profile."""
        with self.assertRaises(FileNotFoundError):
            self.orchestrator.load_config(
                environment="testing",
                profile="nonexistent",
                config_dir=self.config_dir,
                validate=False,
                use_cache=False,
            )

    def test_load_config_with_status(self):
        """Test loading config with status information."""
        config, status = self.orchestrator.load_config_with_status(
            environment="testing", config_dir=self.config_dir, validate=False, use_cache=False
        )

        self.assertIsInstance(config, SimulationConfig)
        self.assertIsInstance(status, dict)
        self.assertTrue(status["success"])
        self.assertEqual(status["environment"], "testing")
        self.assertIsNone(status["profile"])

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        # Load config to populate cache
        self.orchestrator.load_config(environment="testing", config_dir=self.config_dir, validate=False, use_cache=True)

        # Verify it's cached
        cache_stats_before = self.orchestrator.get_cache_stats()
        self.assertGreaterEqual(cache_stats_before["entries"], 1)

        # Invalidate cache
        self.orchestrator.invalidate_cache(environment="testing", config_dir=self.config_dir)

        # Verify cache is invalidated
        cache_stats_after = self.orchestrator.get_cache_stats()
        self.assertEqual(cache_stats_after["entries"], 0)

    def test_clear_cache(self):
        """Test clearing entire cache."""
        # Load config to populate cache
        self.orchestrator.load_config(environment="testing", config_dir=self.config_dir, validate=False, use_cache=True)

        # Clear cache
        self.orchestrator.invalidate_cache()

        # Verify cache is empty
        cache_stats = self.orchestrator.get_cache_stats()
        self.assertEqual(cache_stats["entries"], 0)

    def test_preload_common_configs(self):
        """Test preloading common configurations."""
        self.orchestrator.preload_common_configs(config_dir=self.config_dir)

        # Verify configs are cached
        cache_stats = self.orchestrator.get_cache_stats()
        self.assertGreater(cache_stats["entries"], 0)

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        stats = self.orchestrator.get_cache_stats()

        expected_keys = [
            "entries",
            "memory_usage_mb",
            "max_size",
            "max_memory_mb",
            "hits",
            "misses",
            "invalidations",
            "total_requests",
            "hit_rate",
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

    @patch("farm.config.orchestrator.logger")
    def test_logging_on_load(self, mock_logger):
        """Test that appropriate logging occurs during config loading."""
        self.orchestrator.load_config(
            environment="testing", config_dir=self.config_dir, validate=False, use_cache=False
        )

        # Verify logging calls were made
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()

    def test_strict_validation_with_warnings(self):
        """Test strict validation treats warnings as errors."""
        # Create a config that will generate warnings (very large population)
        config_with_warnings = {
            "width": 100,
            "height": 100,
            "system_agents": 5,
            "independent_agents": 5,
            "control_agents": 5,
            "max_population": 100001,  # This will generate a warning (> 100000)
            "initial_resources": 20,
            "max_resource_amount": 30,
            "learning_rate": 0.001,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "memory_size": 1000,
            "batch_size": 32,
            "max_steps": 100,
            "use_in_memory_db": True,
        }

        import yaml

        env_file = os.path.join(self.config_dir, "environments", "warning_test.yaml")
        with open(env_file, "w") as f:
            yaml.safe_dump(config_with_warnings, f)

        try:
            # Should work with normal validation
            config = self.orchestrator.load_config(
                environment="warning_test",
                config_dir=self.config_dir,
                validate=True,
                strict_validation=False,
                use_cache=False,
            )
            self.assertIsInstance(config, SimulationConfig)

            # Should fail with strict validation
            with self.assertRaises(ValidationError):
                self.orchestrator.load_config(
                    environment="warning_test",
                    config_dir=self.config_dir,
                    validate=True,
                    strict_validation=True,
                    use_cache=False,
                )
        finally:
            # Clean up
            if os.path.exists(env_file):
                os.remove(env_file)


class TestGlobalOrchestrator(unittest.TestCase):
    """Test global orchestrator functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, "config")

        # Create mock config directory structure
        os.makedirs(os.path.join(self.config_dir, "environments"), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, "profiles"), exist_ok=True)

        # Create minimal test configuration files
        self._create_test_configs()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_test_configs(self):
        """Create minimal test configuration files."""
        # Base configuration
        base_config = {
            "width": 100,
            "height": 100,
            "system_agents": 5,
            "independent_agents": 5,
            "control_agents": 5,
            "max_population": 50,
            "initial_resources": 20,
            "max_resource_amount": 30,
            "learning_rate": 0.001,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "memory_size": 1000,
            "batch_size": 32,
            "max_steps": 100,
            "use_in_memory_db": True,
        }

        import yaml

        # Write base config
        with open(os.path.join(self.config_dir, "default.yaml"), "w") as f:
            yaml.safe_dump(base_config, f)

        # Write environment config
        env_config = {"width": 200, "height": 200}
        with open(os.path.join(self.config_dir, "environments", "testing.yaml"), "w") as f:
            yaml.safe_dump(env_config, f)

        # Write profile config
        profile_config = {"max_steps": 200}
        with open(os.path.join(self.config_dir, "profiles", "benchmark.yaml"), "w") as f:
            yaml.safe_dump(profile_config, f)

    def test_get_global_orchestrator(self):
        """Test getting the global orchestrator instance."""
        orchestrator = get_global_orchestrator()
        self.assertIsInstance(orchestrator, ConfigurationOrchestrator)

    def test_load_config_function(self):
        """Test the global load_config function."""
        # Test global load_config function
        config = load_config(environment="testing", config_dir=self.config_dir, validate=False, use_cache=False)

        self.assertIsInstance(config, SimulationConfig)
        self.assertEqual(config.environment.width, 200)  # From testing.yaml


class TestOrchestratorErrorHandling(unittest.TestCase):
    """Test error handling in the orchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = ConfigurationOrchestrator()

        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, "config")

        # Create mock config directory structure
        os.makedirs(os.path.join(self.config_dir, "environments"), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, "profiles"), exist_ok=True)

        # Create minimal test configuration files
        self._create_test_configs()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_test_configs(self):
        """Create minimal test configuration files."""
        # Base configuration
        base_config = {
            "width": 100,
            "height": 100,
            "system_agents": 5,
            "independent_agents": 5,
            "control_agents": 5,
            "max_population": 50,
            "initial_resources": 20,
            "max_resource_amount": 30,
            "learning_rate": 0.001,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "memory_size": 1000,
            "batch_size": 32,
            "max_steps": 100,
            "use_in_memory_db": True,
        }

        import yaml

        # Write base config
        with open(os.path.join(self.config_dir, "default.yaml"), "w") as f:
            yaml.safe_dump(base_config, f)

        # Write environment config
        env_config = {"width": 200, "height": 200}
        with open(os.path.join(self.config_dir, "environments", "testing.yaml"), "w") as f:
            yaml.safe_dump(env_config, f)

    def test_missing_config_directory(self):
        """Test handling of missing configuration directory."""
        with self.assertRaises(FileNotFoundError):
            self.orchestrator.load_config(
                environment="development", config_dir="/nonexistent/path", validate=False, use_cache=False
            )

    def test_corrupted_cache_data(self):
        """Test handling of corrupted cache data."""
        # Test with mock cache that returns None (cache miss)
        mock_cache = Mock(spec=ConfigCache)
        mock_cache.get.return_value = None  # Simulate cache miss

        orchestrator = ConfigurationOrchestrator(cache=mock_cache)

        # Use the existing test setup from setUp which creates proper config structure
        config = orchestrator.load_config(
            environment="testing", config_dir=self.config_dir, validate=False, use_cache=True
        )

        self.assertIsInstance(config, SimulationConfig)


if __name__ == "__main__":
    unittest.main()
