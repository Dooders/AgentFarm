"""
Unit tests for LazyConfigLoader class.

This module tests the LazyConfigLoader functionality including:
- Lazy loading behavior
- Configuration method chaining
- Attribute delegation
- Cache invalidation on reload
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

import yaml

from farm.config import SimulationConfig
from farm.config.cache import LazyConfigLoader, OptimizedConfigLoader


class TestLazyConfigLoader(unittest.TestCase):
    """Test cases for LazyConfigLoader class."""

    def setUp(self):
        """Set up test environment with temporary config files."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, "config")
        self.environments_dir = os.path.join(self.config_dir, "environments")

        # Create directories
        os.makedirs(self.config_dir)
        os.makedirs(self.environments_dir)

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

        # Create environment config
        dev_config = self.base_config.copy()
        dev_config.update({"debug": True, "width": 50})
        with open(os.path.join(self.environments_dir, "development.yaml"), 'w') as f:
            yaml.dump(dev_config, f)

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test LazyConfigLoader initialization."""
        # Test with default loader
        lazy_loader = LazyConfigLoader()
        self.assertIsInstance(lazy_loader.loader, OptimizedConfigLoader)
        self.assertIsNone(lazy_loader._config)
        self.assertIsNone(lazy_loader._load_params)

        # Test with custom loader
        custom_loader = OptimizedConfigLoader()
        lazy_loader = LazyConfigLoader(loader=custom_loader)
        self.assertIs(lazy_loader.loader, custom_loader)

    def test_configure_method_chaining(self):
        """Test that configure returns self for method chaining."""
        lazy_loader = LazyConfigLoader()

        result = lazy_loader.configure(
            environment="development",
            profile=None,
            config_dir=self.config_dir
        )

        self.assertIs(result, lazy_loader)
        self.assertEqual(lazy_loader._load_params['environment'], "development")
        self.assertIsNone(lazy_loader._load_params['profile'])
        self.assertEqual(lazy_loader._load_params['config_dir'], self.config_dir)
        self.assertIsNone(lazy_loader._config)  # Should be invalidated

    def test_get_config_lazy_loading(self):
        """Test that config is loaded lazily on first access."""
        lazy_loader = LazyConfigLoader()
        lazy_loader.configure(
            environment="development",
            config_dir=self.config_dir
        )

        # Config should not be loaded yet
        self.assertIsNone(lazy_loader._config)

        # First call to get_config should load it
        config = lazy_loader.get_config()
        self.assertIsInstance(config, SimulationConfig)
        self.assertIsNotNone(lazy_loader._config)
        self.assertEqual(config.width, 50)  # Development override

        # Second call should return cached config
        config2 = lazy_loader.get_config()
        self.assertIs(config2, lazy_loader._config)

    def test_get_config_without_configuration(self):
        """Test that get_config raises error when not configured."""
        lazy_loader = LazyConfigLoader()

        with self.assertRaises(ValueError) as cm:
            lazy_loader.get_config()

        self.assertIn("Loader not configured", str(cm.exception))

    def test_attribute_delegation(self):
        """Test that attribute access is delegated to the config."""
        lazy_loader = LazyConfigLoader()
        lazy_loader.configure(
            environment="development",
            config_dir=self.config_dir
        )

        # Access attributes that should be delegated to config
        self.assertEqual(lazy_loader.width, 50)
        self.assertEqual(lazy_loader.debug, True)
        self.assertEqual(lazy_loader.seed, 42)

        # Config should be loaded after attribute access
        self.assertIsNotNone(lazy_loader._config)

    def test_reload_functionality(self):
        """Test that reload forces fresh loading."""
        lazy_loader = LazyConfigLoader()
        lazy_loader.configure(
            environment="development",
            config_dir=self.config_dir
        )

        # Load config initially
        config1 = lazy_loader.get_config()
        initial_width = config1.width

        # Modify the config file
        import time
        time.sleep(0.1)  # Ensure timestamp difference
        modified_config = self.base_config.copy()
        modified_config.update({"debug": True, "width": 999})
        with open(os.path.join(self.environments_dir, "development.yaml"), 'w') as f:
            yaml.dump(modified_config, f)

        # Reload should get updated config
        config2 = lazy_loader.reload()
        self.assertEqual(config2.width, 999)
        self.assertIs(config2, lazy_loader._config)  # Should be cached again

    def test_reload_without_configuration(self):
        """Test that reload raises error when not configured."""
        lazy_loader = LazyConfigLoader()

        with self.assertRaises(ValueError) as cm:
            lazy_loader.reload()

        self.assertIn("Loader not configured", str(cm.exception))

    def test_reload_invalidates_cache(self):
        """Test that reload invalidates the underlying cache."""
        # Mock the loader's cache
        mock_loader = MagicMock(spec=OptimizedConfigLoader)
        mock_cache = MagicMock()
        mock_loader.cache = mock_cache
        mock_loader._create_cache_key.return_value = "test_key"
        mock_loader.load_centralized_config.return_value = SimulationConfig()

        lazy_loader = LazyConfigLoader(loader=mock_loader)
        lazy_loader._load_params = {
            'environment': 'development',
            'profile': None,
            'config_dir': self.config_dir
        }

        # Reload should invalidate cache
        lazy_loader.reload()

        # Check that cache.invalidate was called
        mock_cache.invalidate.assert_called_once_with("test_key")
        # Check that load_centralized_config was called
        mock_loader.load_centralized_config.assert_called_once()

    def test_configure_invalidates_cached_config(self):
        """Test that calling configure invalidates any cached config."""
        lazy_loader = LazyConfigLoader()
        lazy_loader.configure(
            environment="development",
            config_dir=self.config_dir
        )

        # Load config
        lazy_loader.get_config()
        self.assertIsNotNone(lazy_loader._config)

        # Reconfigure
        lazy_loader.configure(
            environment="development",
            config_dir=self.config_dir
        )

        # Config should be invalidated
        self.assertIsNone(lazy_loader._config)

    def test_multiple_configurations(self):
        """Test switching between different configurations."""
        lazy_loader = LazyConfigLoader()
        lazy_loader.configure(
            environment="development",
            config_dir=self.config_dir
        )

        config1 = lazy_loader.get_config()
        self.assertEqual(config1.width, 50)

        # Switch configuration (though we only have one environment)
        # This tests that reconfigure works
        lazy_loader.configure(
            environment="development",
            config_dir=self.config_dir
        )

        config2 = lazy_loader.get_config()
        self.assertEqual(config2.width, 50)


if __name__ == '__main__':
    unittest.main()
