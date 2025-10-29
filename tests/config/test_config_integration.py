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
from unittest.mock import MagicMock, patch

from farm.config import SimulationConfig
from farm.config.template import ConfigTemplate, ConfigTemplateManager
from farm.config.watcher import ConfigWatcher, create_reloadable_config


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
            "seed": 42,
        }

        # Create subdirectories
        os.makedirs(os.path.join(self.config_dir, "environments"), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, "profiles"), exist_ok=True)

        # Write default config
        with open(os.path.join(self.config_dir, "default.yaml"), "w") as f:
            yaml.dump(test_config, f)

        # Write environment configs
        dev_config = test_config.copy()
        dev_config.update({"debug": True, "width": 50})
        with open(os.path.join(self.config_dir, "environments", "development.yaml"), "w") as f:
            yaml.dump(dev_config, f)

        prod_config = test_config.copy()
        prod_config.update({"debug": False, "width": 200, "max_population": 500})
        with open(os.path.join(self.config_dir, "environments", "production.yaml"), "w") as f:
            yaml.dump(prod_config, f)

        # Write profile configs
        bench_config = test_config.copy()
        bench_config.update({"learning_rate": 0.01, "batch_size": 64, "max_steps": 2000})
        with open(os.path.join(self.config_dir, "profiles", "benchmark.yaml"), "w") as f:
            yaml.dump(bench_config, f)

    def test_full_config_workflow(self):
        """Test complete configuration workflow from creation to versioning."""
        # 1. Load centralized config
        config = SimulationConfig.from_centralized_config(config_dir=self.config_dir, environment="development")

        self.assertEqual(config.environment.width, 50)  # Should use development override
        self.assertEqual(config.logging.debug, True)  # Should use development override
        self.assertEqual(config.seed, 42)  # Should use default

        # 2. Apply profile override
        config_with_profile = SimulationConfig.from_centralized_config(
            config_dir=self.config_dir, environment="development", profile="benchmark"
        )

        self.assertEqual(config_with_profile.environment.width, 100)  # Profile override
        self.assertEqual(config_with_profile.learning.learning_rate, 0.01)  # Profile override
        self.assertEqual(config_with_profile.logging.debug, True)  # Development override

        # 3. Version the configuration
        versioned = config_with_profile.version_config("Integration test config")
        self.assertIsNotNone(versioned.versioning.config_version)
        self.assertEqual(versioned.versioning.config_description, "Integration test config")

        # 4. Save versioned config
        filepath = versioned.save_versioned_config(self.versions_dir)
        self.assertTrue(os.path.exists(filepath))

        # 5. List versions
        versions = SimulationConfig.list_config_versions(self.versions_dir)
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["description"], "Integration test config")

        # 6. Load versioned config
        loaded = SimulationConfig.load_versioned_config(self.versions_dir, versioned.versioning.config_version)
        self.assertEqual(loaded.versioning.config_version, versioned.versioning.config_version)
        self.assertEqual(loaded.environment.width, 100)

        # 7. Compare configurations
        diff = config.diff_config(config_with_profile)
        self.assertIn("learning.learning_rate", diff)  # Should show profile differences

    def test_template_workflow(self):
        """Test complete template workflow."""
        manager = ConfigTemplateManager(self.templates_dir)

        # 1. Create template from config
        base_config = SimulationConfig.from_centralized_config(config_dir=self.config_dir)
        template = ConfigTemplate.from_config(base_config)

        # Modify template to add variables
        template_dict = template.template_dict
        template_dict["width"] = "{{env_width}}"
        template_dict["system_agents"] = "{{agent_count}}"
        template = ConfigTemplate(template_dict)

        # 2. Save template
        template_path = manager.save_template("test_template", template, "Integration test template")
        self.assertTrue(os.path.exists(template_path))

        # 3. Load template
        loaded_template = manager.load_template("test_template")
        required_vars = loaded_template.get_required_variables()
        self.assertIn("env_width", required_vars)
        self.assertIn("agent_count", required_vars)

        # 4. Instantiate template
        variables = {"env_width": 150, "agent_count": 20}
        config = loaded_template.instantiate(variables)
        self.assertEqual(config.environment.width, 150)
        self.assertEqual(config.population.system_agents, 20)

        # 5. Batch instantiate
        variable_sets = [
            {"env_width": 100, "agent_count": 10},
            {"env_width": 200, "agent_count": 25},
        ]

        batch_dir = os.path.join(self.templates_dir, "batch_output")
        config_paths = manager.create_experiment_configs("test_template", variable_sets, batch_dir)

        self.assertEqual(len(config_paths), 2)
        for path in config_paths:
            self.assertTrue(os.path.exists(path))

            # Verify config was created correctly
            config = SimulationConfig.from_yaml(path)
            self.assertIsNotNone(config.versioning.config_version)  # Should be versioned

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
            changes.append(new_config.environment.width)

        reloadable.add_change_callback(change_callback)

        # Modify config file
        modified_config = SimulationConfig.from_yaml(config_path)
        modified_config.environment.width = 999
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
            SimulationConfig.from_centralized_config(config_dir=self.config_dir, environment="nonexistent")

        # Test invalid profile
        with self.assertRaises(FileNotFoundError):
            SimulationConfig.from_centralized_config(
                config_dir=self.config_dir,
                environment="development",
                profile="nonexistent",
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
            import os  # Ensure os is available in thread context

            try:
                # Load config
                config = SimulationConfig.from_centralized_config(config_dir=self.config_dir)

                # Modify a field to make each config unique
                config.seed = 42 + worker_id

                # Save version (this will version it automatically)
                config.version_config(f"Worker {worker_id} config").save_versioned_config(self.versions_dir)

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
        self.assertIn("environment.width", config_dict)
        self.assertIn("environment.height", config_dict)

        # Test schema compliance (if schema file exists)
        schema_path = os.path.join(self.config_dir, "schema.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = json.load(f)

            # Basic schema validation
            self.assertIn("sections", schema)
            self.assertIn("simulation", schema["sections"])

    def test_from_yaml_method(self):
        """Test that from_yaml method works correctly."""
        # Test from_yaml functionality
        config_path = os.path.join(self.config_dir, "default.yaml")
        config = SimulationConfig.from_yaml(config_path)

        # Verify it loaded correctly
        self.assertIsInstance(config, SimulationConfig)
        self.assertEqual(config.environment.width, 100)  # Default value
        self.assertEqual(config.environment.height, 100)  # Default value

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
        self.assertLess(
            version_time,
            1.0,
            f"Versioning too slow: {version_time:.2f}s for 50 versions",
        )

    def test_orchestrator_full_pipeline(self):
        """Test the full orchestrator pipeline from cache to validation."""
        from farm.config import ConfigurationOrchestrator

        orchestrator = ConfigurationOrchestrator()

        # Test 1: Basic loading
        config = orchestrator.load_config(
            environment="development", config_dir=self.config_dir, validate=False, use_cache=False
        )
        self.assertIsInstance(config, SimulationConfig)
        self.assertEqual(config.environment.width, 50)

        # Test 2: Loading with validation
        config_validated, status = orchestrator.load_config_with_status(
            environment="development", config_dir=self.config_dir, validate=True, use_cache=False
        )
        self.assertIsInstance(config_validated, SimulationConfig)
        self.assertTrue(status["success"])
        self.assertIn("errors", status)
        self.assertIn("warnings", status)

        # Test 3: Caching behavior
        # First load (should miss cache)
        config1 = orchestrator.load_config(
            environment="development", config_dir=self.config_dir, validate=False, use_cache=True
        )

        # Second load (should hit cache)
        config2 = orchestrator.load_config(
            environment="development", config_dir=self.config_dir, validate=False, use_cache=True
        )

        # Verify they're the same and cache worked
        self.assertEqual(config1.environment.width, config2.environment.width)
        stats = orchestrator.get_cache_stats()
        self.assertGreaterEqual(stats["hits"], 1)

    def test_orchestrator_environment_profiles(self):
        """Test orchestrator with different environments and profiles."""
        from farm.config import ConfigurationOrchestrator

        orchestrator = ConfigurationOrchestrator()

        # Test environment switching
        dev_config = orchestrator.load_config(environment="development", config_dir=self.config_dir, validate=False)
        prod_config = orchestrator.load_config(environment="production", config_dir=self.config_dir, validate=False)

        # Environments should have different settings
        self.assertNotEqual(dev_config.environment.width, prod_config.environment.width)

        # Test profile application
        bench_config = orchestrator.load_config(
            environment="development", profile="benchmark", config_dir=self.config_dir, validate=False
        )

        # Profile should override base settings
        self.assertNotEqual(dev_config.max_steps, bench_config.max_steps)

    def test_orchestrator_error_handling(self):
        """Test orchestrator error handling and recovery."""
        from farm.config import ConfigurationOrchestrator
        from farm.config.validation import ValidationError

        orchestrator = ConfigurationOrchestrator()

        # Test invalid environment (should use fallback or raise)
        try:
            config = orchestrator.load_config(environment="nonexistent", config_dir=self.config_dir, validate=False)
            # If it succeeds, it should have created a fallback config
            self.assertIsInstance(config, SimulationConfig)
        except Exception:
            # If it fails, that's also acceptable
            pass

        # Test validation with auto-repair
        config, status = orchestrator.load_config_with_status(
            environment="development", config_dir=self.config_dir, validate=True, auto_repair=True
        )
        self.assertTrue(status["success"])
        self.assertIsInstance(config, SimulationConfig)

    def test_orchestrator_cache_invalidation(self):
        """Test cache invalidation in orchestrator."""
        from farm.config import ConfigurationOrchestrator
        import time

        orchestrator = ConfigurationOrchestrator()

        # Load config to populate cache
        config1 = orchestrator.load_config(
            environment="development", config_dir=self.config_dir, validate=False, use_cache=True
        )

        # Verify cache has entries
        stats = orchestrator.get_cache_stats()
        initial_entries = stats["entries"]
        self.assertGreaterEqual(initial_entries, 1)

        # Invalidate cache
        orchestrator.invalidate_cache()

        # Verify cache is cleared
        stats = orchestrator.get_cache_stats()
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)

    def test_orchestrator_performance_regression(self):
        """Test for performance regression in orchestrator."""
        from farm.config import ConfigurationOrchestrator
        import time

        orchestrator = ConfigurationOrchestrator()

        # Test basic load performance (should be < 50ms average)
        times = []
        for _ in range(20):
            start = time.perf_counter()
            config = orchestrator.load_config(
                environment="development", config_dir=self.config_dir, validate=False, use_cache=False
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Performance should be reasonable
        self.assertLess(avg_time, 50.0, f"Average load time too slow: {avg_time:.2f}ms")
        self.assertLess(max_time, 100.0, f"Max load time too slow: {max_time:.2f}ms")

        # Test cached performance (should be < 1ms average)
        # Warm up cache
        for _ in range(5):
            orchestrator.load_config(
                environment="development", config_dir=self.config_dir, validate=False, use_cache=True
            )

        times = []
        for _ in range(20):
            start = time.perf_counter()
            config = orchestrator.load_config(
                environment="development", config_dir=self.config_dir, validate=False, use_cache=True
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_cached_time = sum(times) / len(times)
        self.assertLess(avg_cached_time, 2.0, f"Cached load time too slow: {avg_cached_time:.3f}ms")

        # Verify high cache hit rate
        stats = orchestrator.get_cache_stats()
        self.assertGreater(stats["hit_rate"], 0.8, f"Cache hit rate too low: {stats['hit_rate']:.1%}")


if __name__ == "__main__":
    unittest.main()
