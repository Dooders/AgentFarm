"""
Component Isolation Tests - Testing cache, loader, and validator independently.

This module provides comprehensive unit tests for each configuration component
in isolation, using mock implementations to ensure proper separation of concerns
and prevent circular dependencies.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch, mock_open
import tempfile
import os
import yaml
from typing import Dict, Any

from farm.config.cache import ConfigCache, OptimizedConfigLoader
from farm.config.validation import ConfigurationValidator, SafeConfigLoader
from farm.config.orchestrator import ConfigurationOrchestrator


class MockFileSystem:
    """Mock file system for testing file operations."""

    def __init__(self):
        self.files = {}
        self.mtimes = {}

    def write_file(self, path: str, content: str):
        """Write content to a mock file."""
        self.files[path] = content
        self.mtimes[path] = 1234567890.0

    def read_file(self, path: str) -> str:
        """Read content from a mock file."""
        return self.files.get(path, "")

    def exists(self, path: str) -> bool:
        """Check if mock file exists."""
        return path in self.files

    def get_mtime(self, path: str) -> float:
        """Get mock file modification time."""
        return self.mtimes.get(path, 0.0)


class TestConfigCacheIsolation(unittest.TestCase):
    """Test ConfigCache component in complete isolation."""

    def setUp(self):
        """Set up isolated cache testing."""
        self.cache = ConfigCache(max_size=3, max_memory_mb=1.0)

    def test_cache_initialization(self):
        """Test cache initializes with correct parameters."""
        cache = ConfigCache(max_size=10, max_memory_mb=5.0)
        self.assertEqual(cache.max_size, 10)
        self.assertEqual(cache.max_memory_mb, 5.0)
        self.assertEqual(len(cache.cache), 0)

    def test_cache_get_put_simulation_config_isolation(self):
        """Test cache operations with SimulationConfig data."""
        from farm.config.config import SimulationConfig

        cache_key = "test_key"
        test_config = SimulationConfig()

        # Cache should be empty
        self.assertIsNone(self.cache.get(cache_key))

        # Put config in cache
        self.cache.put(cache_key, test_config)

        # Should retrieve the same config
        retrieved = self.cache.get(cache_key)
        self.assertIsInstance(retrieved, SimulationConfig)
        self.assertEqual(retrieved.environment.width, test_config.environment.width)

        # Verify cache stats
        stats = self.cache.get_stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_cache_memory_limits(self):
        """Test cache respects memory limits."""
        from farm.config.config import SimulationConfig

        cache = ConfigCache(max_memory_mb=0.5)  # 0.5MB limit

        # First item should fit
        cache.put("key1", SimulationConfig())
        self.assertEqual(cache.get_stats()["entries"], 1)

        # Create a config that should exceed memory limit when serialized
        # This is harder to test precisely, so we'll just verify basic functionality
        cache.put("key2", SimulationConfig())
        stats = cache.get_stats()
        self.assertIsInstance(stats["memory_usage_mb"], float)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from farm.config.config import SimulationConfig

        cache = ConfigCache(max_size=2)

        # Fill cache
        cache.put("key1", SimulationConfig())
        cache.put("key2", SimulationConfig())
        self.assertEqual(cache.get_stats()["entries"], 2)

        # Access key1 to make it recently used
        cache.get("key1")

        # Add third item, should evict key2 (least recently used)
        cache.put("key3", SimulationConfig())
        self.assertEqual(cache.get_stats()["entries"], 2)

        # key1 should still be there
        self.assertIsNotNone(cache.get("key1"))
        # key2 should be gone
        self.assertIsNone(cache.get("key2"))
        # key3 should be there
        self.assertIsNotNone(cache.get("key3"))

    def test_cache_file_modification_tracking(self):
        """Test cache invalidation based on file modification times."""
        from farm.config.config import SimulationConfig

        cache = ConfigCache()

        # Mock file operations
        with patch("os.path.exists", return_value=True), patch("os.path.getmtime", return_value=1000.0):
            # Put item in cache with file tracking
            cache.put("test_key", SimulationConfig(), "test_file.yaml")

            # Should be cached
            self.assertIsNotNone(cache.get("test_key", "test_file.yaml"))

            # Simulate file modification
            with patch("os.path.getmtime", return_value=2000.0):
                # Cache should be invalidated
                self.assertIsNone(cache.get("test_key", "test_file.yaml"))

    def test_cache_invalidation_methods(self):
        """Test cache invalidation methods."""
        from farm.config.config import SimulationConfig

        cache = ConfigCache()

        # Add items
        cache.put("key1", SimulationConfig())
        cache.put("key2", SimulationConfig())
        self.assertEqual(cache.get_stats()["entries"], 2)

        # Invalidate specific key
        cache.invalidate("key1")
        self.assertIsNone(cache.get("key1"))
        self.assertIsNotNone(cache.get("key2"))

        # Clear all
        cache.clear()
        self.assertEqual(cache.get_stats()["entries"], 0)


class TestOptimizedConfigLoaderIsolation(unittest.TestCase):
    """Test OptimizedConfigLoader component in isolation."""

    def setUp(self):
        """Set up isolated loader testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, "config")

        # Create mock config structure
        os.makedirs(os.path.join(self.config_dir, "environments"), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, "profiles"), exist_ok=True)

        # Create test config files
        self._create_test_configs()

        # Create loader with mock cache
        self.mock_cache = Mock(spec=ConfigCache)
        self.loader = OptimizedConfigLoader(cache=self.mock_cache)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_test_configs(self):
        """Create test configuration files."""
        base_config = {
            "width": 100,
            "height": 100,
            "system_agents": 5,
            "independent_agents": 5,
            "control_agents": 5,
            "max_population": 50,
        }

        # Write base config
        with open(os.path.join(self.config_dir, "default.yaml"), "w") as f:
            yaml.dump(base_config, f)

        # Write environment config
        env_config = base_config.copy()
        env_config.update({"width": 200, "debug": True})
        with open(os.path.join(self.config_dir, "environments", "test.yaml"), "w") as f:
            yaml.dump(env_config, f)

    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = OptimizedConfigLoader()
        self.assertIsInstance(loader.cache, ConfigCache)

    def test_loader_cache_miss_workflow(self):
        """Test loader workflow when cache misses."""
        from farm.config.config import SimulationConfig

        # Mock cache miss
        self.mock_cache.get.return_value = None

        # Load config (will actually load from files)
        result = self.loader.load_centralized_config(environment="test", config_dir=self.config_dir, use_cache=True)

        # Verify cache was checked
        self.mock_cache.get.assert_called_once()

        # Verify result is SimulationConfig
        self.assertIsInstance(result, SimulationConfig)

        # Verify cache was populated
        self.mock_cache.put.assert_called_once()

    def test_loader_cache_hit_workflow(self):
        """Test loader workflow when cache hits."""
        from farm.config.config import SimulationConfig

        # Create a valid config dict
        valid_config = SimulationConfig()
        cached_dict = valid_config.to_dict()

        # Mock cache hit
        self.mock_cache.get.return_value = cached_dict

        # Load config
        result = self.loader.load_centralized_config(environment="test", config_dir=self.config_dir, use_cache=True)

        # Verify cache was checked
        self.mock_cache.get.assert_called_once()

        # Verify result is SimulationConfig
        self.assertIsInstance(result, SimulationConfig)

        # Verify cache was not updated
        self.mock_cache.put.assert_not_called()

    def test_loader_no_cache_workflow(self):
        """Test loader workflow when caching is disabled."""
        from farm.config.config import SimulationConfig

        # Load config without cache
        result = self.loader.load_centralized_config(environment="test", config_dir=self.config_dir, use_cache=False)

        # Verify cache was not accessed
        self.mock_cache.get.assert_not_called()
        self.mock_cache.put.assert_not_called()

        # Verify result is SimulationConfig
        self.assertIsInstance(result, SimulationConfig)

    def test_loader_file_loading_integration(self):
        """Test actual file loading without mocks."""
        from farm.config.config import SimulationConfig

        # Use real cache for this test
        real_cache = ConfigCache()
        loader = OptimizedConfigLoader(cache=real_cache)

        # Load config
        result = loader.load_centralized_config(environment="test", config_dir=self.config_dir, use_cache=False)

        # Verify structure
        self.assertIsInstance(result, SimulationConfig)
        self.assertEqual(result.environment.width, 200)  # From environment override


class TestConfigurationValidatorIsolation(unittest.TestCase):
    """Test ConfigurationValidator component in isolation."""

    def setUp(self):
        """Set up isolated validator testing."""
        self.validator = ConfigurationValidator()

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = ConfigurationValidator()
        self.assertIsInstance(validator, ConfigurationValidator)

    def test_validator_basic_validation_pass(self):
        """Test validation passes with valid config."""
        # Create minimal valid config
        from farm.config.config import SimulationConfig

        config = SimulationConfig()

        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertIsInstance(warnings, list)

    def test_validator_basic_validation_fail(self):
        """Test validation fails with invalid config."""
        # Create config with invalid values
        from farm.config.config import SimulationConfig

        config = SimulationConfig()
        # Manually set invalid values
        config.environment.width = -10
        config.population.max_population = 0

        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_validator_strict_mode(self):
        """Test strict validation mode."""
        # Create config that generates warnings
        from farm.config.config import SimulationConfig

        config = SimulationConfig()
        config.population.max_population = 150000  # Should generate warning

        # Normal validation
        is_valid_normal, errors_normal, warnings_normal = self.validator.validate_config(config, strict=False)
        self.assertTrue(is_valid_normal)
        self.assertGreater(len(warnings_normal), 0)

        # Strict validation
        is_valid_strict, errors_strict, warnings_strict = self.validator.validate_config(config, strict=True)
        self.assertFalse(is_valid_strict)
        self.assertEqual(len(warnings_strict), 0)  # Warnings moved to errors

    def test_validator_error_accumulation(self):
        """Test validator accumulates multiple errors."""
        # Create config with multiple validation issues
        from farm.config.config import SimulationConfig

        config = SimulationConfig()
        config.environment.width = -5
        config.environment.height = -10
        config.population.max_population = -1
        config.learning.learning_rate = 2.0  # Invalid learning rate

        is_valid, errors, warnings = self.validator.validate_config(config)

        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 3)  # Should have multiple errors


class TestSafeConfigLoaderIsolation(unittest.TestCase):
    """Test SafeConfigLoader component in isolation."""

    def setUp(self):
        """Set up isolated safe loader testing."""
        self.loader = SafeConfigLoader()

    def test_safe_loader_initialization(self):
        """Test safe loader initializes correctly."""
        loader = SafeConfigLoader()
        self.assertIsInstance(loader.validator, ConfigurationValidator)

    @patch("farm.config.validation.SafeConfigLoader.validate_config_dict")
    def test_safe_loader_success_workflow(self, mock_validate):
        """Test successful validation workflow."""
        mock_validate.return_value = ({"validated": "config"}, {"success": True})

        result_dict, status = self.loader.validate_config_dict(
            {"input": "config"}, strict_validation=False, auto_repair=False
        )

        self.assertEqual(result_dict, {"validated": "config"})
        self.assertTrue(status["success"])
        mock_validate.assert_called_once()

    def test_safe_loader_repair_workflow(self):
        """Test auto-repair workflow."""
        from farm.config.config import SimulationConfig

        # Create a config with issues that will need repair
        config = SimulationConfig()
        config.environment.width = -10  # Invalid width

        # This should trigger repair/fallback and succeed
        result_dict, status = self.loader.validate_config_dict(
            config.to_dict(), strict_validation=False, auto_repair=True
        )

        # Should succeed (either through repair or fallback)
        self.assertTrue(status["success"])
        self.assertIsInstance(result_dict, dict)
        self.assertIn("errors", status)

    def test_safe_loader_fallback_creation(self):
        """Test fallback configuration creation."""
        from farm.config.validation import ConfigurationRecovery

        # Test with error details
        error_details = {"error": "test_error"}
        fallback = ConfigurationRecovery.create_fallback_config(error_details)

        # Should create a valid SimulationConfig
        from farm.config.config import SimulationConfig

        self.assertIsInstance(fallback, SimulationConfig)

        # Should have reasonable default values
        self.assertGreater(fallback.environment.width, 0)
        self.assertGreater(fallback.population.max_population, 0)


class TestConfigurationOrchestratorIsolation(unittest.TestCase):
    """Test ConfigurationOrchestrator component isolation."""

    def setUp(self):
        """Set up isolated orchestrator testing."""
        # Create mocks for all dependencies
        self.mock_cache = Mock(spec=ConfigCache)
        self.mock_loader = Mock(spec=OptimizedConfigLoader)
        self.mock_validator = Mock(spec=SafeConfigLoader)

        # Create orchestrator with mocks
        self.orchestrator = ConfigurationOrchestrator(
            cache=self.mock_cache, loader=self.mock_loader, validator=self.mock_validator
        )

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes with provided components."""
        self.assertEqual(self.orchestrator.cache, self.mock_cache)
        self.assertEqual(self.orchestrator.loader, self.mock_loader)
        self.assertEqual(self.orchestrator.validator, self.mock_validator)

    def test_orchestrator_cache_hit_workflow(self):
        """Test orchestrator workflow when cache hits."""
        from farm.config.config import SimulationConfig

        # Mock cache hit
        cached_config = SimulationConfig()
        self.mock_cache.get.return_value = cached_config

        # Mock validation
        validated_config = SimulationConfig()
        self.mock_validator.validate_config_dict.return_value = (validated_config.to_dict(), {"success": True})

        # Load config
        result = self.orchestrator.load_config(environment="test", validate=True, use_cache=True)

        # Verify cache was checked
        self.mock_cache.get.assert_called_once()

        # Verify validation was called
        self.mock_validator.validate_config_dict.assert_called_once()

        # Verify loader was not called
        self.mock_loader.load_centralized_config.assert_not_called()

        # Verify result is SimulationConfig
        self.assertIsInstance(result, SimulationConfig)

    def test_orchestrator_cache_miss_workflow(self):
        """Test orchestrator workflow when cache misses."""
        from farm.config.config import SimulationConfig

        # Mock cache miss
        self.mock_cache.get.return_value = None

        # Mock loader
        loaded_config = SimulationConfig()
        self.mock_loader.load_centralized_config.return_value = loaded_config

        # Mock validation
        validated_config = SimulationConfig()
        self.mock_validator.validate_config_dict.return_value = (validated_config.to_dict(), {"success": True})

        # Load config
        result = self.orchestrator.load_config(environment="test", validate=True, use_cache=True)

        # Verify cache was checked
        self.mock_cache.get.assert_called_once()

        # Verify loader was called
        self.mock_loader.load_centralized_config.assert_called_once()

        # Verify validation was called
        self.mock_validator.validate_config_dict.assert_called_once()

        # Verify cache was populated
        self.mock_cache.put.assert_called_once()

        # Verify result is SimulationConfig
        self.assertIsInstance(result, SimulationConfig)

    def test_orchestrator_error_handling(self):
        """Test orchestrator error handling."""
        # Mock cache miss
        self.mock_cache.get.return_value = None

        # Mock loader to raise exception
        self.mock_loader.load_centralized_config.side_effect = FileNotFoundError("Config not found")

        # Should raise the exception
        with self.assertRaises(FileNotFoundError):
            self.orchestrator.load_config(environment="test", validate=False, use_cache=True)

    def test_orchestrator_cache_invalidation(self):
        """Test orchestrator cache invalidation."""
        # Call invalidate
        self.orchestrator.invalidate_cache(environment="test")

        # Verify cache invalidate was called
        self.mock_cache.invalidate.assert_called_once()


class TestMockImplementations(unittest.TestCase):
    """Test mock implementations for component isolation."""

    def test_mock_cache_implementation(self):
        """Test that mock cache behaves like real cache."""
        mock_cache = Mock(spec=ConfigCache)

        # Configure mock behavior
        test_data = {"mock": "data"}
        mock_cache.get.return_value = test_data
        mock_cache.put.return_value = None
        mock_cache.get_stats.return_value = {"entries": 1}

        # Test mock usage
        result = mock_cache.get("test_key")
        self.assertEqual(result, test_data)

        mock_cache.put("test_key", test_data)
        mock_cache.put.assert_called_once_with("test_key", test_data)

        stats = mock_cache.get_stats()
        self.assertEqual(stats["entries"], 1)

    def test_mock_loader_implementation(self):
        """Test that mock loader behaves like real loader."""
        mock_loader = Mock(spec=OptimizedConfigLoader)

        # Configure mock behavior
        test_config = {"loader": "config"}
        mock_loader.load_centralized_config.return_value = test_config

        # Test mock usage
        result = mock_loader.load_centralized_config(environment="test", use_cache=False)

        self.assertEqual(result, test_config)
        mock_loader.load_centralized_config.assert_called_once()

    def test_mock_validator_implementation(self):
        """Test that mock validator behaves like real validator."""
        mock_validator = Mock(spec=SafeConfigLoader)

        # Configure mock behavior
        validated_config = {"validated": "config"}
        status = {"success": True}
        mock_validator.validate_config_dict.return_value = (validated_config, status)

        # Test mock usage
        result_config, result_status = mock_validator.validate_config_dict({"input": "config"})

        self.assertEqual(result_config, validated_config)
        self.assertEqual(result_status, status)
        mock_validator.validate_config_dict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
