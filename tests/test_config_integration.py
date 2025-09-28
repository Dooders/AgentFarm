"""
Comprehensive integration tests for the centralized configuration system.

This module tests the entire configuration ecosystem including:
- Centralized config loading
- Environment and profile overrides
- Versioning system
- Templating system
- Runtime reloading
- CLI tools
- Error handling and edge cases
"""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from farm.core.config import SimulationConfig
from farm.core.config_template import ConfigTemplate, ConfigTemplateManager
from farm.core.config_watcher import ConfigWatcher, create_reloadable_config


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for the complete configuration system."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, "config")
        self.versions_dir = os.path.join(self.test_dir, "versions")
        self.templates_dir = os.path.join(self.test_dir, "templates")

        # Create directories
        os.makedirs(self.config_dir)
        os.makedirs(self.versions_dir)
        os.makedirs(self.templates_dir)

        # Copy test config files
        self._setup_test_configs()

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_test_configs(self):
        """Set up test configuration files."""
        import yaml

        # Create minimal test config
        test_config = {
            "width": 100,
            "height": 100,
            "system_agents": 5,
            "independent_agents": 5,
            "control_agents": 5,
            "max_population": 50,
            "seed": 42
        }

        # Create subdirectories
        os.makedirs(os.path.join(self.config_dir, "environments"), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, "profiles"), exist_ok=True)

        # Write default config
        with open(os.path.join(self.config_dir, "default.yaml"), 'w') as f:
            yaml.dump(test_config, f)

        # Write environment configs
        dev_config = test_config.copy()
        dev_config.update({"debug": True, "width": 50})
        with open(os.path.join(self.config_dir, "environments", "development.yaml"), 'w') as f:
            yaml.dump(dev_config, f)

        prod_config = test_config.copy()
        prod_config.update({"debug": False, "width": 200, "max_population": 500})
        with open(os.path.join(self.config_dir, "environments", "production.yaml"), 'w') as f:
            yaml.dump(prod_config, f)

        # Write profile configs
        bench_config = test_config.copy()
        bench_config.update({"learning_rate": 0.01, "batch_size": 64})
        with open(os.path.join(self.config_dir, "profiles", "benchmark.yaml"), 'w') as f:
            yaml.dump(bench_config, f)

    def test_full_config_workflow(self):
        """Test complete configuration workflow from creation to versioning."""
        # 1. Load centralized config
        config = SimulationConfig.from_centralized_config(
            config_dir=self.config_dir,
            environment="development",
            validate=False  # Skip validation for minimal test config
        )

        self.assertEqual(config.width, 50)  # Should use development override
        self.assertEqual(config.debug, True)  # Should use development override
        self.assertEqual(config.seed, 42)  # Should use default

        # 2. Apply profile override
        config_with_profile = SimulationConfig.from_centralized_config(
            config_dir=self.config_dir,
            environment="development",
            profile="benchmark",
            validate=False
        )

        self.assertEqual(config_with_profile.width, 100)  # Profile override
        self.assertEqual(config_with_profile.learning_rate, 0.01)  # Profile override
        self.assertEqual(config_with_profile.debug, True)  # Development override

        # 3. Version the configuration
        versioned = config_with_profile.version_config("Integration test config")
        self.assertIsNotNone(versioned.config_version)
        self.assertEqual(versioned.config_description, "Integration test config")

        # 4. Save versioned config
        filepath = versioned.save_versioned_config(self.versions_dir)
        self.assertTrue(os.path.exists(filepath))

        # 5. List versions
        versions = SimulationConfig.list_config_versions(self.versions_dir)
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]['description'], "Integration test config")

        # 6. Load versioned config
        loaded = SimulationConfig.load_versioned_config(self.versions_dir, versioned.config_version)
        self.assertEqual(loaded.config_version, versioned.config_version)
        self.assertEqual(loaded.width, 100)

        # 7. Compare configurations
        diff = config.diff_config(config_with_profile)
        self.assertIn('learning_rate', diff)  # Should show profile differences

    def test_template_workflow(self):
        """Test complete template workflow."""
        manager = ConfigTemplateManager(self.templates_dir)

        # 1. Create template from config
        base_config = SimulationConfig.from_centralized_config(config_dir=self.config_dir)
        template = ConfigTemplate.from_config(base_config)

        # Modify template to add variables
        template_dict = template.template_dict
        template_dict['width'] = '{{env_width}}'
        template_dict['system_agents'] = '{{agent_count}}'
        template = ConfigTemplate(template_dict)

        # 2. Save template
        template_path = manager.save_template("test_template", template, "Integration test template")
        self.assertTrue(os.path.exists(template_path))

        # 3. Load template
        loaded_template = manager.load_template("test_template")
        required_vars = loaded_template.get_required_variables()
        self.assertIn('env_width', required_vars)
        self.assertIn('agent_count', required_vars)

        # 4. Instantiate template
        variables = {'env_width': 150, 'agent_count': 20}
        config = loaded_template.instantiate(variables)
        self.assertEqual(config.width, 150)
        self.assertEqual(config.system_agents, 20)

        # 5. Batch instantiate
        variable_sets = [
            {'env_width': 100, 'agent_count': 10},
            {'env_width': 200, 'agent_count': 25}
        ]

        batch_dir = os.path.join(self.templates_dir, "batch_output")
        config_paths = manager.create_experiment_configs("test_template", variable_sets, batch_dir)

        self.assertEqual(len(config_paths), 2)
        for path in config_paths:
            self.assertTrue(os.path.exists(path))

            # Verify config was created correctly
            config = SimulationConfig.from_yaml(path)
            self.assertIsNotNone(config.config_version)  # Should be versioned

    def test_runtime_reloading_integration(self):
        """Test runtime reloading integration."""
        # Create test config file
        config_path = os.path.join(self.test_dir, "reload_test.yaml")
        initial_config = SimulationConfig.from_centralized_config(config_dir=self.config_dir)
        initial_config.to_yaml(config_path)

        # Create reloadable config
        reloadable = create_reloadable_config(config_path)

        # Track changes
        changes = []
        def change_callback(new_config):
            changes.append(new_config.width)

        reloadable.add_change_callback(change_callback)

        # Modify config file
        modified_config = SimulationConfig.from_yaml(config_path)
        modified_config.width = 999
        modified_config.to_yaml(config_path)

        # Wait for watcher to detect change
        timeout = 5
        start_time = time.time()
        while not changes and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Verify change was detected
        self.assertTrue(len(changes) > 0, "Config change was not detected")
        self.assertEqual(changes[0], 999)

        reloadable.stop_watching()

    def test_error_handling(self):
        """Test error handling throughout the system."""
        # Test invalid environment
        with self.assertRaises(FileNotFoundError):
            SimulationConfig.from_centralized_config(
                config_dir=self.config_dir,
                environment="nonexistent",
                validate=False
            )

        # Test invalid profile
        with self.assertRaises(FileNotFoundError):
            SimulationConfig.from_centralized_config(
                config_dir=self.config_dir,
                environment="development",
                profile="nonexistent",
                validate=False
            )

        # Test invalid version
        with self.assertRaises(FileNotFoundError):
            SimulationConfig.load_versioned_config(self.versions_dir, "invalid_version")

        # Test template with missing variables
        template = ConfigTemplate({"width": "{{size}}", "height": "{{size}}"})
        with self.assertRaises(ValueError):
            template.instantiate({"other_var": 100})

        # Test invalid template loading
        with self.assertRaises(FileNotFoundError):
            manager = ConfigTemplateManager(self.templates_dir)
            manager.load_template("nonexistent_template")

    def test_concurrent_operations(self):
        """Test concurrent configuration operations."""
        results = []

        def worker(worker_id):
            """Worker function for concurrent testing."""
            try:
                # Load config
                config = SimulationConfig.from_centralized_config(config_dir=self.config_dir)

                # Version it
                versioned = config.version_config(f"Worker {worker_id} config")

                # Save version
                versioned.save_versioned_config(self.versions_dir, f"Worker {worker_id}")

                results.append(f"Worker {worker_id}: SUCCESS")
            except Exception as e:
                results.append(f"Worker {worker_id}: ERROR - {e}")

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)

        # Verify all succeeded
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn("SUCCESS", result)

        # Verify versions were created
        versions = SimulationConfig.list_config_versions(self.versions_dir)
        self.assertEqual(len(versions), 5)

    def test_config_validation_and_schema(self):
        """Test configuration validation against schema."""
        # Load a valid config
        config = SimulationConfig.from_centralized_config(config_dir=self.config_dir)

        # Should not raise any validation errors
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('width', config_dict)
        self.assertIn('height', config_dict)

        # Test schema compliance (if schema file exists)
        schema_path = os.path.join(self.config_dir, "schema.json")
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)

            # Basic schema validation
            self.assertIn('sections', schema)
            self.assertIn('simulation', schema['sections'])

    def test_deprecation_warnings(self):
        """Test that deprecation warnings are properly issued."""
        import warnings

        # Test from_yaml deprecation
        config_path = os.path.join(self.config_dir, "default.yaml")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = SimulationConfig.from_yaml(config_path)

            # Should have deprecation warning
            self.assertTrue(len(w) > 0)
            found_deprecation = False
            for warning in w:
                if issubclass(warning.category, DeprecationWarning):
                    self.assertIn("deprecated", str(warning.message).lower())
                    found_deprecation = True
                    break
            self.assertTrue(found_deprecation, "No deprecation warning found")

    def test_performance_baselines(self):
        """Test performance baselines for configuration operations."""
        import time

        # Test config loading performance
        start_time = time.time()
        for _ in range(100):
            SimulationConfig.from_centralized_config(config_dir=self.config_dir)
        load_time = time.time() - start_time

        # Should load 100 configs in reasonable time (< 5 seconds)
        self.assertLess(load_time, 5.0, f"Config loading too slow: {load_time:.2f}s for 100 loads")

        # Test versioning performance
        config = SimulationConfig.from_centralized_config(config_dir=self.config_dir)
        start_time = time.time()
        for _ in range(50):
            config.version_config("Performance test")
        version_time = time.time() - start_time

        # Should version 50 configs quickly (< 1 second)
        self.assertLess(version_time, 1.0, f"Versioning too slow: {version_time:.2f}s for 50 versions")


if __name__ == '__main__':
    unittest.main()
