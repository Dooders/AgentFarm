"""
Unit tests for OptimizedConfigLoader class.

This module tests the OptimizedConfigLoader functionality including:
- Cached and non-cached loading
- File-based configuration loading
- Preloading functionality
- Cache key generation
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import yaml

from farm.config import SimulationConfig
from farm.config.cache import ConfigCache, OptimizedConfigLoader


class TestOptimizedConfigLoader(unittest.TestCase):
    """Test cases for OptimizedConfigLoader class."""

    def setUp(self):
        """Set up test environment with temporary config files."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, "config")
        self.environments_dir = os.path.join(self.config_dir, "environments")
        self.profiles_dir = os.path.join(self.config_dir, "profiles")

        # Create directories
        os.makedirs(self.config_dir)
        os.makedirs(self.environments_dir)
        os.makedirs(self.profiles_dir)

        # Create base config
        self.base_config = {
            "width": 100,
            "height": 100,
            "system_agents": 5,
            "independent_agents": 5,
            "control_agents": 5,
            "max_population": 50,
            "seed": 42
        }

        with open(os.path.join(self.config_dir, "default.yaml"), 'w') as f:
            yaml.dump(self.base_config, f)

        # Create environment configs
        dev_config = self.base_config.copy()
        dev_config.update({"debug": True, "width": 50})
        with open(os.path.join(self.environments_dir, "development.yaml"), 'w') as f:
            yaml.dump(dev_config, f)

        prod_config = self.base_config.copy()
        prod_config.update({"debug": False, "width": 200, "max_population": 500})
        with open(os.path.join(self.environments_dir, "production.yaml"), 'w') as f:
            yaml.dump(prod_config, f)

        # Create profile configs
        bench_config = {"learning_rate": 0.01, "batch_size": 64, "width": 100}
        with open(os.path.join(self.profiles_dir, "benchmark.yaml"), 'w') as f:
            yaml.dump(bench_config, f)

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test OptimizedConfigLoader initialization."""
        # Test with default cache
        loader = OptimizedConfigLoader()
        self.assertIsNotNone(loader.cache)

        # Test with custom cache
        custom_cache = ConfigCache(max_size=10)
        loader = OptimizedConfigLoader(cache=custom_cache)
        self.assertIs(loader.cache, custom_cache)

    def test_load_centralized_config_without_cache(self):
        """Test loading config without caching."""
        loader = OptimizedConfigLoader()

        config = loader.load_centralized_config(
            environment="development",
            config_dir=self.config_dir,
            use_cache=False
        )

        self.assertIsInstance(config, SimulationConfig)
        self.assertEqual(config.width, 50)  # Development override
        self.assertEqual(config.debug, True)  # Development override
        self.assertEqual(config.seed, 42)  # Default value

    def test_load_centralized_config_with_cache(self):
        """Test loading config with caching enabled."""
        cache = ConfigCache()
        loader = OptimizedConfigLoader(cache=cache)

        # First load - should load from files
        config1 = loader.load_centralized_config(
            environment="development",
            config_dir=self.config_dir,
            use_cache=True
        )

        # Second load - should come from cache
        config2 = loader.load_centralized_config(
            environment="development",
            config_dir=self.config_dir,
            use_cache=True
        )

        self.assertEqual(config1.width, config2.width)
        self.assertEqual(config1.debug, config2.debug)

        # Check that cache contains the entry
        cache_key = loader._create_cache_key("development", None, self.config_dir)
        base_config_path = os.path.join(self.config_dir, "default.yaml")
        cached_config = cache.get(cache_key, base_config_path)
        self.assertIsNotNone(cached_config)

    def test_load_with_profile(self):
        """Test loading config with profile overrides."""
        loader = OptimizedConfigLoader()

        config = loader.load_centralized_config(
            environment="development",
            profile="benchmark",
            config_dir=self.config_dir,
            use_cache=False
        )

        self.assertEqual(config.width, 100)  # Profile overrides environment
        self.assertEqual(config.debug, True)  # Environment setting preserved
        self.assertEqual(config.learning_rate, 0.01)  # Profile setting
        self.assertEqual(config.batch_size, 64)  # Profile setting

    def test_missing_base_config_error(self):
        """Test error when base config file is missing."""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)

        loader = OptimizedConfigLoader()

        with self.assertRaises(FileNotFoundError) as cm:
            loader.load_centralized_config(
                environment="development",
                config_dir=empty_dir,
                use_cache=False
            )

        self.assertIn("Base configuration not found", str(cm.exception))

    def test_missing_environment_config_error(self):
        """Test error when environment config file is missing."""
        loader = OptimizedConfigLoader()

        with self.assertRaises(FileNotFoundError) as cm:
            loader.load_centralized_config(
                environment="nonexistent",
                config_dir=self.config_dir,
                use_cache=False
            )

        self.assertIn("Environment configuration not found", str(cm.exception))

    def test_missing_profile_config_error(self):
        """Test error when profile config file is missing."""
        loader = OptimizedConfigLoader()

        with self.assertRaises(FileNotFoundError) as cm:
            loader.load_centralized_config(
                environment="development",
                profile="nonexistent",
                config_dir=self.config_dir,
                use_cache=False
            )

        self.assertIn("Profile configuration not found", str(cm.exception))

    def test_preload_common_configs(self):
        """Test preloading common configurations."""
        cache = ConfigCache()
        loader = OptimizedConfigLoader(cache=cache)

        # Should not raise errors even if some configs don't exist
        loader.preload_common_configs(config_dir=self.config_dir)

        # Check that some configs were cached
        # (development should exist, testing may not)
        dev_key = loader._create_cache_key("development", None, self.config_dir)
        dev_config = cache.get(dev_key)
        self.assertIsNotNone(dev_config)

    def test_cache_key_generation(self):
        """Test cache key generation includes file modification times."""
        loader = OptimizedConfigLoader()

        key1 = loader._create_cache_key("development", None, self.config_dir)
        key2 = loader._create_cache_key("development", None, self.config_dir)

        # Keys should be the same for same inputs
        self.assertEqual(key1, key2)

        # Keys should be different for different inputs
        key3 = loader._create_cache_key("production", None, self.config_dir)
        self.assertNotEqual(key1, key3)

        key4 = loader._create_cache_key("development", "benchmark", self.config_dir)
        self.assertNotEqual(key1, key4)

    def test_cache_invalidation_on_file_change(self):
        """Test that cache is invalidated when config files change."""
        cache = ConfigCache()
        loader = OptimizedConfigLoader(cache=cache)

        # Load config
        config1 = loader.load_centralized_config(
            environment="development",
            config_dir=self.config_dir,
            use_cache=True
        )

        # Modify the config file
        import time
        time.sleep(0.1)  # Ensure timestamp difference
        modified_config = self.base_config.copy()
        modified_config["width"] = 999
        with open(os.path.join(self.environments_dir, "development.yaml"), 'w') as f:
            yaml.dump(modified_config, f)

        # Load again - should get updated config
        config2 = loader.load_centralized_config(
            environment="development",
            config_dir=self.config_dir,
            use_cache=True
        )

        # Should have different width due to file change
        self.assertNotEqual(config1.width, config2.width)
        self.assertEqual(config2.width, 999)


if __name__ == '__main__':
    unittest.main()
