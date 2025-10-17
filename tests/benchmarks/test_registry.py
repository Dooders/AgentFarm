"""
Unit tests for the benchmarks.core.registry module.

Tests ExperimentRegistry and register_experiment decorator.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from benchmarks.core.registry import (
    ExperimentRegistry, ExperimentInfo, register_experiment, REGISTRY
)
from benchmarks.core.experiments import Experiment


class TestExperimentInfo(unittest.TestCase):
    """Test ExperimentInfo dataclass."""

    def test_experiment_info_creation(self):
        """Test ExperimentInfo creation with all fields."""
        class TestExperiment(Experiment):
            def execute_once(self, context):
                return {}

        info = ExperimentInfo(
            slug="test_experiment",
            cls=TestExperiment,
            summary="Test experiment description",
            tags=["test", "unit"],
            param_schema={"properties": {"param1": {"type": "string"}}}
        )

        self.assertEqual(info.slug, "test_experiment")
        self.assertEqual(info.cls, TestExperiment)
        self.assertEqual(info.summary, "Test experiment description")
        self.assertEqual(info.tags, ["test", "unit"])
        self.assertEqual(info.param_schema, {"properties": {"param1": {"type": "string"}}})

    def test_experiment_info_defaults(self):
        """Test ExperimentInfo creation with default values."""
        class TestExperiment(Experiment):
            def execute_once(self, context):
                return {}

        info = ExperimentInfo(
            slug="minimal_experiment",
            cls=TestExperiment
        )

        self.assertEqual(info.slug, "minimal_experiment")
        self.assertEqual(info.cls, TestExperiment)
        self.assertEqual(info.summary, "")
        self.assertEqual(info.tags, [])
        self.assertEqual(info.param_schema, {})


class TestExperimentRegistry(unittest.TestCase):
    """Test ExperimentRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ExperimentRegistry()

        # Create test experiment classes
        class TestExperiment1(Experiment):
            def __init__(self, params=None, **kwargs):
                super().__init__(params)
                # Handle any additional kwargs
                if kwargs:
                    self.params.update(kwargs)

            def execute_once(self, context):
                return {"result": "test1"}

        class TestExperiment2(Experiment):
            param_schema = {
                "properties": {
                    "param1": {"type": "string", "default": "default_value"},
                    "param2": {"type": "integer", "default": 42}
                },
                "required": ["param1"]
            }

            def __init__(self, params=None, param1=None, param2=None, **kwargs):
                super().__init__(params)
                # Handle schema parameters
                if param1 is not None:
                    self.params["param1"] = param1
                if param2 is not None:
                    self.params["param2"] = param2
                # Handle any additional kwargs
                if kwargs:
                    self.params.update(kwargs)

            def execute_once(self, context):
                return {"result": "test2", "param1": self.params.get("param1")}

        self.TestExperiment1 = TestExperiment1
        self.TestExperiment2 = TestExperiment2

    def test_initialization(self):
        """Test registry initialization."""
        self.assertEqual(len(self.registry._slug_to_info), 0)

    def test_register_experiment(self):
        """Test registering an experiment."""
        self.registry.register(
            slug="test_exp",
            cls=self.TestExperiment1,
            summary="Test experiment",
            tags=["test"]
        )

        self.assertEqual(len(self.registry._slug_to_info), 1)
        self.assertIn("test_exp", self.registry._slug_to_info)

    def test_register_duplicate_slug_raises_error(self):
        """Test that registering duplicate slug raises ValueError."""
        self.registry.register("test_exp", self.TestExperiment1)

        with self.assertRaises(ValueError) as context:
            self.registry.register("test_exp", self.TestExperiment2)

        self.assertIn("already registered", str(context.exception))

    def test_register_with_docstring_summary(self):
        """Test registering with docstring as summary."""
        class DocstringExperiment(Experiment):
            """This is a test experiment with a docstring."""
            def execute_once(self, context):
                return {}

        self.registry.register("docstring_exp", DocstringExperiment)

        info = self.registry.get("docstring_exp")
        self.assertEqual(info.summary, "This is a test experiment with a docstring.")

    def test_register_without_summary(self):
        """Test registering without summary (should use empty string)."""
        class NoDocstringExperiment(Experiment):
            def execute_once(self, context):
                return {}

        self.registry.register("no_docstring_exp", NoDocstringExperiment)

        info = self.registry.get("no_docstring_exp")
        self.assertEqual(info.summary, "")

    def test_register_with_param_schema(self):
        """Test registering with param_schema."""
        self.registry.register("schema_exp", self.TestExperiment2)

        info = self.registry.get("schema_exp")
        self.assertEqual(info.param_schema, self.TestExperiment2.param_schema)

    def test_list_experiments(self):
        """Test listing all registered experiments."""
        self.registry.register("exp1", self.TestExperiment1, "First experiment")
        self.registry.register("exp2", self.TestExperiment2, "Second experiment")

        experiments = self.registry.list()

        self.assertEqual(len(experiments), 2)
        slugs = [exp.slug for exp in experiments]
        self.assertIn("exp1", slugs)
        self.assertIn("exp2", slugs)

    def test_get_existing_experiment(self):
        """Test getting an existing experiment."""
        self.registry.register("test_exp", self.TestExperiment1, "Test experiment")

        info = self.registry.get("test_exp")

        self.assertEqual(info.slug, "test_exp")
        self.assertEqual(info.cls, self.TestExperiment1)
        self.assertEqual(info.summary, "Test experiment")

    def test_get_nonexistent_experiment_raises_error(self):
        """Test that getting nonexistent experiment raises KeyError."""
        with self.assertRaises(KeyError) as context:
            self.registry.get("nonexistent")

        self.assertIn("Unknown experiment", str(context.exception))

    def test_create_experiment_without_params(self):
        """Test creating experiment without parameters."""
        self.registry.register("test_exp", self.TestExperiment1)

        experiment = self.registry.create("test_exp")

        self.assertIsInstance(experiment, self.TestExperiment1)
        self.assertEqual(experiment.params, {})

    def test_create_experiment_with_params(self):
        """Test creating experiment with parameters."""
        self.registry.register("test_exp", self.TestExperiment1)

        params = {"param1": "value1", "param2": 42}
        experiment = self.registry.create("test_exp", params)

        self.assertIsInstance(experiment, self.TestExperiment1)
        self.assertEqual(experiment.params, params)

    def test_create_experiment_with_schema_defaults(self):
        """Test creating experiment with schema defaults."""
        self.registry.register("schema_exp", self.TestExperiment2)

        experiment = self.registry.create("schema_exp", {"param1": "custom_value"})

        self.assertIsInstance(experiment, self.TestExperiment2)
        self.assertEqual(experiment.params["param1"], "custom_value")
        self.assertEqual(experiment.params["param2"], 42)  # default value

    def test_create_experiment_with_schema_validation(self):
        """Test creating experiment with schema validation."""
        self.registry.register("schema_exp", self.TestExperiment2)

        # Should raise error for missing required parameter
        with self.assertRaises(ValueError) as context:
            self.registry.create("schema_exp")

        self.assertIn("Missing required param", str(context.exception))

    def test_create_experiment_with_schema_override_defaults(self):
        """Test creating experiment overriding schema defaults."""
        self.registry.register("schema_exp", self.TestExperiment2)

        params = {"param1": "required_value", "param2": 100}
        experiment = self.registry.create("schema_exp", params)

        self.assertEqual(experiment.params["param1"], "required_value")
        self.assertEqual(experiment.params["param2"], 100)  # overridden default

    @patch('importlib.import_module')
    @patch('importlib.util.find_spec')
    @patch('os.walk')
    def test_discover_package(self, mock_walk, mock_find_spec, mock_import_module):
        """Test package discovery."""
        # Mock the package spec
        mock_spec = Mock()
        mock_spec.submodule_search_locations = ["/path/to/package"]
        mock_find_spec.return_value = mock_spec

        # Mock os.walk to return some Python files
        mock_walk.return_value = [
            ("/path/to/package", [], ["module1.py", "module2.py", "__init__.py"]),
            ("/path/to/package/subdir", [], ["module3.py"])
        ]

        self.registry.discover_package("test.package")

        # Should import the modules (excluding __init__.py and private modules)
        expected_calls = [
            unittest.mock.call("test.package.module1"),
            unittest.mock.call("test.package.module2"),
            unittest.mock.call("test.package.subdir.module3")
        ]
        mock_import_module.assert_has_calls(expected_calls, any_order=True)

    @patch('importlib.util.find_spec')
    def test_discover_package_nonexistent(self, mock_find_spec):
        """Test discovering nonexistent package."""
        mock_find_spec.return_value = None

        # Should not raise an error
        self.registry.discover_package("nonexistent.package")

        # Should not have registered any experiments
        self.assertEqual(len(self.registry._slug_to_info), 0)

    @patch('importlib.util.find_spec')
    def test_discover_package_no_submodule_locations(self, mock_find_spec):
        """Test discovering package without submodule search locations."""
        mock_spec = Mock()
        mock_spec.submodule_search_locations = None
        mock_find_spec.return_value = mock_spec

        # Should not raise an error
        self.registry.discover_package("test.package")

        # Should not have registered any experiments
        self.assertEqual(len(self.registry._slug_to_info), 0)


class TestRegisterExperimentDecorator(unittest.TestCase):
    """Test register_experiment decorator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh registry for testing
        self.test_registry = ExperimentRegistry()

    def test_decorator_registers_experiment(self):
        """Test that decorator registers experiment correctly."""
        @register_experiment("decorated_exp", "Test decorated experiment", ["test"])
        class DecoratedExperiment(Experiment):
            def execute_once(self, context):
                return {}

        # Check that it was registered in the global registry
        info = REGISTRY.get("decorated_exp")
        self.assertEqual(info.slug, "decorated_exp")
        self.assertEqual(info.summary, "Test decorated experiment")
        self.assertEqual(info.tags, ["test"])
        self.assertEqual(info.cls, DecoratedExperiment)

    def test_decorator_without_summary_and_tags(self):
        """Test decorator without summary and tags."""
        @register_experiment("minimal_exp")
        class MinimalExperiment(Experiment):
            def execute_once(self, context):
                return {}

        info = REGISTRY.get("minimal_exp")
        self.assertEqual(info.slug, "minimal_exp")
        self.assertEqual(info.summary, "")
        self.assertEqual(info.tags, [])
        self.assertEqual(info.cls, MinimalExperiment)

    def test_decorator_with_docstring(self):
        """Test decorator with docstring as summary."""
        @register_experiment("docstring_exp")
        class DocstringExperiment(Experiment):
            """This experiment has a docstring."""
            def execute_once(self, context):
                return {}

        info = REGISTRY.get("docstring_exp")
        self.assertEqual(info.summary, "This experiment has a docstring.")

    def test_decorator_with_param_schema(self):
        """Test decorator with param_schema."""
        @register_experiment("schema_exp")
        class SchemaExperiment(Experiment):
            param_schema = {
                "properties": {
                    "param1": {"type": "string", "default": "default"}
                }
            }

            def execute_once(self, context):
                return {}

        info = REGISTRY.get("schema_exp")
        self.assertEqual(info.param_schema, SchemaExperiment.param_schema)

    def test_decorator_on_non_experiment_class_raises_error(self):
        """Test that decorator on non-Experiment class raises TypeError."""
        with self.assertRaises(TypeError) as context:
            @register_experiment("invalid_exp")
            class NotAnExperiment:
                pass

        self.assertIn("can only decorate Experiment subclasses", str(context.exception))

    def test_decorator_on_non_class_raises_error(self):
        """Test that decorator on non-class raises TypeError."""
        with self.assertRaises(TypeError):
            @register_experiment("invalid_exp")
            def not_a_class():
                pass

    def test_decorator_duplicate_slug_raises_error(self):
        """Test that decorator with duplicate slug raises ValueError."""
        @register_experiment("duplicate_exp")
        class FirstExperiment(Experiment):
            def execute_once(self, context):
                return {}

        with self.assertRaises(ValueError) as context:
            @register_experiment("duplicate_exp")
            class SecondExperiment(Experiment):
                def execute_once(self, context):
                    return {}

        self.assertIn("already registered", str(context.exception))


class TestGlobalRegistry(unittest.TestCase):
    """Test the global REGISTRY instance."""

    def test_global_registry_is_singleton(self):
        """Test that REGISTRY is a singleton instance."""
        from benchmarks.core.registry import REGISTRY as registry1
        from benchmarks.core.registry import REGISTRY as registry2

        self.assertIs(registry1, registry2)
        self.assertIsInstance(registry1, ExperimentRegistry)

    def test_global_registry_persistence(self):
        """Test that experiments registered in global registry persist."""
        # Clear any existing registrations
        original_slugs = list(REGISTRY._slug_to_info.keys())

        @register_experiment("persistent_exp")
        class PersistentExperiment(Experiment):
            def execute_once(self, context):
                return {}

        # Should be able to get the experiment
        info = REGISTRY.get("persistent_exp")
        self.assertEqual(info.cls, PersistentExperiment)

        # Clean up
        if "persistent_exp" in REGISTRY._slug_to_info:
            del REGISTRY._slug_to_info["persistent_exp"]


if __name__ == "__main__":
    unittest.main()
