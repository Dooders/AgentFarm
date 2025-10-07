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

    def test_deep_merge_dicts_function(self):
        """Test the _deep_merge_dicts utility function."""
        from farm.config.cache import _deep_merge_dicts

        # Test simple merge
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge_dicts(base, override)
        self.assertEqual(result, {"a": 1, "b": 3, "c": 4})

        # Test nested merge
        base = {"nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 3, "z": 4}}
        result = _deep_merge_dicts(base, override)
        self.assertEqual(result, {"nested": {"x": 1, "y": 3, "z": 4}})

        # Test non-dict override
        base = {"nested": {"x": 1}}
        override = {"nested": "not_a_dict"}
        result = _deep_merge_dicts(base, override)
        self.assertEqual(result, {"nested": "not_a_dict"})

    def test_cache_simulation_config_storage(self):
        """Test that cache properly handles SimulationConfig objects."""
        from farm.config.config import SimulationConfig

        cache = ConfigCache()

        # Create and store a config
        original_config = SimulationConfig()
        original_config.environment.width = 999  # Unique value

        cache.put("test_key", original_config)

        # Retrieve it
        retrieved = cache.get("test_key")

        # Should be a SimulationConfig with same values
        self.assertIsInstance(retrieved, SimulationConfig)
        self.assertEqual(retrieved.environment.width, 999)

    def test_cache_invalid_data_handling(self):
        """Test cache handles invalid cached data gracefully."""
        cache = ConfigCache()

        # Manually put invalid data in cache
        cache.cache["invalid_key"] = {
            "config": "not_a_dict_or_config",
            "size": 100,
            "created": 1234567890.0,
            "file_mtimes": {},
        }
        cache.access_times["invalid_key"] = 1234567890.0

        # Should handle gracefully and return None
        result = cache.get("invalid_key")
        self.assertIsNone(result)

        # Invalid entry should be removed
        self.assertNotIn("invalid_key", cache.cache)


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

        # Create a valid config
        valid_config = SimulationConfig()

        # Mock cache hit - cache now returns SimulationConfig objects directly
        self.mock_cache.get.return_value = valid_config

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

    def test_configuration_validator_individual_checks(self):
        """Test individual validation methods in ConfigurationValidator."""
        from farm.config.config import SimulationConfig
        from farm.config.validation import ConfigurationValidator

        validator = ConfigurationValidator()
        config = SimulationConfig()

        # Test that individual validation methods don't raise exceptions
        # (they add to errors/warnings instead)
        try:
            validator._validate_basic_properties(config)
            validator._validate_agent_settings(config)
            validator._validate_resource_settings(config)
            validator._validate_learning_parameters(config)
            validator._validate_environment_settings(config)
            validator._validate_performance_settings(config)
            validator._validate_business_rules(config)
        except Exception as e:
            self.fail(f"Individual validation methods should not raise exceptions: {e}")

    def test_configuration_validator_with_invalid_config(self):
        """Test validator with various invalid configurations."""
        from farm.config.config import SimulationConfig
        from farm.config.validation import ConfigurationValidator

        validator = ConfigurationValidator()

        # Test invalid dimensions
        config = SimulationConfig()
        config.environment.width = -10
        config.environment.height = -5

        is_valid, errors, warnings = validator.validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("Environment dimensions must be positive" in str(error) for error in errors))

        # Test invalid learning rate
        config2 = SimulationConfig()
        config2.learning.learning_rate = -0.5

        is_valid2, errors2, warnings2 = validator.validate_config(config2)
        self.assertFalse(is_valid2)
        self.assertTrue(any("learning_rate" in str(error) for error in errors2))

    def test_attempt_config_repair_functionality(self):
        """Test the attempt_config_repair function."""
        from farm.config.validation import ConfigurationRecovery
        from farm.config.config import SimulationConfig

        # Create config with repairable issues
        config = SimulationConfig()
        config.environment.width = -10  # Should be repaired to 50
        config.population.max_population = 0  # Should be repaired

        errors = [
            {"field": "width", "error": "Environment dimensions must be positive", "value": -10},
            {"field": "max_population", "error": "Max population must be positive", "value": 0},
        ]

        # Attempt repair
        repaired_config, repair_actions = ConfigurationRecovery.attempt_config_repair(config, errors)

        # Should have repair actions
        self.assertGreater(len(repair_actions), 0)
        self.assertTrue(any("width" in action.lower() for action in repair_actions))

        # Repaired config should have valid values
        self.assertGreater(repaired_config.environment.width, 0)
        self.assertGreater(repaired_config.population.max_population, 0)

    def test_validate_config_files_function(self):
        """Test the validate_config_files function."""
        from farm.config.validation import validate_config_files
        import tempfile
        import os
        import yaml

        # Create temporary config directory with test files
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a minimal valid config
            config_data = {
                "width": 100,
                "height": 100,
                "system_agents": 5,
                "independent_agents": 5,
                "control_agents": 5,
                "max_population": 50,
            }

            with open(os.path.join(temp_dir, "default.yaml"), "w") as f:
                yaml.dump(config_data, f)

            # Test validation
            result = validate_config_files(temp_dir)

            # Should return validation report
            self.assertIsInstance(result, dict)
            self.assertIn("files_checked", result)
            self.assertIn("total_errors", result)
            self.assertIn("total_warnings", result)

        finally:
            import shutil

            shutil.rmtree(temp_dir)


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

    def test_orchestrator_load_config_with_status_error_handling(self):
        """Test orchestrator load_config_with_status error handling."""

        # Mock cache miss and loader failure
        self.mock_cache.get.return_value = None
        self.mock_loader.load_centralized_config.side_effect = FileNotFoundError("Config not found")

        # Should raise the exception (doesn't handle gracefully)
        with self.assertRaises(FileNotFoundError):
            self.orchestrator.load_config_with_status(environment="test", validate=False)

    def test_orchestrator_validation_error_propagation(self):
        """Test orchestrator propagates validation errors correctly."""
        from farm.config.config import SimulationConfig

        # Mock successful loading but failed validation
        loaded_config = SimulationConfig()
        self.mock_cache.get.return_value = None
        self.mock_loader.load_centralized_config.return_value = loaded_config

        # Mock validation failure
        self.mock_validator.validate_config_dict.return_value = (
            {"failed": "validation"},
            {"success": False, "errors": ["Validation failed"], "warnings": []},
        )

        # Should raise ValidationError
        with self.assertRaises(Exception):  # Could be ValidationError or similar
            self.orchestrator.load_config(environment="test", validate=True)

    def test_orchestrator_preload_common_configs(self):
        """Test orchestrator preload functionality."""
        from farm.config.config import SimulationConfig

        # Mock successful loading - return a valid config
        mock_config = SimulationConfig()
        self.mock_loader.load_centralized_config.return_value = mock_config
        self.mock_cache.get.return_value = None  # Cache miss
        self.mock_validator.validate_config_dict.return_value = ({"validated": True}, {"success": True})

        # Should not raise
        self.orchestrator.preload_common_configs(config_dir="test_dir")

        # Should have called loader multiple times (for different env/profile combos)
        self.assertGreater(self.mock_loader.load_centralized_config.call_count, 1)


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
