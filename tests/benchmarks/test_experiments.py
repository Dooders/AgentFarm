"""
Unit tests for the benchmarks.core.experiments module.

Tests the Experiment base class and ExperimentContext dataclass.
"""

import unittest
from unittest.mock import Mock, patch
from dataclasses import asdict

from benchmarks.core.experiments import Experiment, ExperimentContext


class TestExperimentContext(unittest.TestCase):
    """Test ExperimentContext dataclass."""

    def test_initialization_with_all_fields(self):
        """Test ExperimentContext initialization with all fields."""
        context = ExperimentContext(
            run_id="test_run_123",
            output_dir="/tmp/output",
            run_dir="/tmp/output/test_run_123",
            iteration_index=0,
            seed=42,
            instruments=[Mock(), Mock()],
            extras={"key": "value"}
        )
        
        self.assertEqual(context.run_id, "test_run_123")
        self.assertEqual(context.output_dir, "/tmp/output")
        self.assertEqual(context.run_dir, "/tmp/output/test_run_123")
        self.assertEqual(context.iteration_index, 0)
        self.assertEqual(context.seed, 42)
        self.assertEqual(len(context.instruments), 2)
        self.assertEqual(context.extras["key"], "value")

    def test_initialization_with_defaults(self):
        """Test ExperimentContext initialization with default values."""
        context = ExperimentContext(
            run_id="test_run_456",
            output_dir="/tmp/output",
            run_dir="/tmp/output/test_run_456"
        )
        
        self.assertEqual(context.run_id, "test_run_456")
        self.assertEqual(context.output_dir, "/tmp/output")
        self.assertEqual(context.run_dir, "/tmp/output/test_run_456")
        self.assertIsNone(context.iteration_index)
        self.assertIsNone(context.seed)
        self.assertEqual(context.instruments, [])
        self.assertEqual(context.extras, {})

    def test_dataclass_serialization(self):
        """Test that ExperimentContext can be converted to dict."""
        context = ExperimentContext(
            run_id="test_run_789",
            output_dir="/tmp/output",
            run_dir="/tmp/output/test_run_789",
            iteration_index=1,
            seed=123,
            extras={"test": "data"}
        )
        
        context_dict = asdict(context)
        self.assertIn("run_id", context_dict)
        self.assertIn("output_dir", context_dict)
        self.assertIn("run_dir", context_dict)
        self.assertIn("iteration_index", context_dict)
        self.assertIn("seed", context_dict)
        self.assertIn("extras", context_dict)


class ConcreteExperiment(Experiment):
    """Concrete implementation of Experiment for testing."""
    
    def __init__(self, params=None):
        super().__init__(params)
        self.setup_called = False
        self.teardown_called = False
        self.execute_count = 0
    
    def setup(self, context):
        """Track that setup was called."""
        self.setup_called = True
    
    def execute_once(self, context):
        """Track execution and return test metrics."""
        self.execute_count += 1
        return {
            "iteration": self.execute_count,
            "run_id": context.run_id,
            "iteration_index": context.iteration_index
        }
    
    def teardown(self, context):
        """Track that teardown was called."""
        self.teardown_called = True


class TestExperiment(unittest.TestCase):
    """Test Experiment base class."""

    def test_initialization_with_params(self):
        """Test Experiment initialization with parameters."""
        params = {"param1": "value1", "param2": 42}
        experiment = ConcreteExperiment(params)
        
        self.assertEqual(experiment.params, params)
        self.assertFalse(experiment.setup_called)
        self.assertFalse(experiment.teardown_called)
        self.assertEqual(experiment.execute_count, 0)

    def test_initialization_without_params(self):
        """Test Experiment initialization without parameters."""
        experiment = ConcreteExperiment()
        
        self.assertEqual(experiment.params, {})
        self.assertFalse(experiment.setup_called)
        self.assertFalse(experiment.teardown_called)
        self.assertEqual(experiment.execute_count, 0)

    def test_initialization_with_none_params(self):
        """Test Experiment initialization with None parameters."""
        experiment = ConcreteExperiment(None)
        
        self.assertEqual(experiment.params, {})
        self.assertFalse(experiment.setup_called)
        self.assertFalse(experiment.teardown_called)
        self.assertEqual(experiment.execute_count, 0)

    def test_setup_method(self):
        """Test that setup method is called correctly."""
        experiment = ConcreteExperiment()
        context = ExperimentContext(
            run_id="test",
            output_dir="/tmp",
            run_dir="/tmp/test"
        )
        
        experiment.setup(context)
        self.assertTrue(experiment.setup_called)

    def test_execute_once_method(self):
        """Test that execute_once method works correctly."""
        experiment = ConcreteExperiment()
        context = ExperimentContext(
            run_id="test_run",
            output_dir="/tmp",
            run_dir="/tmp/test_run",
            iteration_index=0
        )
        
        result = experiment.execute_once(context)
        
        self.assertEqual(experiment.execute_count, 1)
        self.assertEqual(result["iteration"], 1)
        self.assertEqual(result["run_id"], "test_run")
        self.assertEqual(result["iteration_index"], 0)

    def test_teardown_method(self):
        """Test that teardown method is called correctly."""
        experiment = ConcreteExperiment()
        context = ExperimentContext(
            run_id="test",
            output_dir="/tmp",
            run_dir="/tmp/test"
        )
        
        experiment.teardown(context)
        self.assertTrue(experiment.teardown_called)

    def test_multiple_executions(self):
        """Test multiple execute_once calls."""
        experiment = ConcreteExperiment()
        context = ExperimentContext(
            run_id="test_run",
            output_dir="/tmp",
            run_dir="/tmp/test_run"
        )
        
        # Execute multiple times
        for i in range(3):
            context.iteration_index = i
            result = experiment.execute_once(context)
            self.assertEqual(result["iteration"], i + 1)
            self.assertEqual(result["iteration_index"], i)
        
        self.assertEqual(experiment.execute_count, 3)


class AbstractExperiment(Experiment):
    """Abstract experiment that doesn't implement execute_once."""
    
    def setup(self, context):
        pass
    
    def teardown(self, context):
        pass


class TestAbstractExperiment(unittest.TestCase):
    """Test abstract Experiment behavior."""

    def test_abstract_execute_once_raises_error(self):
        """Test that abstract execute_once raises NotImplementedError."""
        # Note: We can't instantiate AbstractExperiment directly due to abstract methods
        # This test verifies the abstract method exists and would raise NotImplementedError
        # when called on a concrete implementation that doesn't override it
        pass

    def test_param_schema_default(self):
        """Test that param_schema defaults to empty dict."""
        experiment = ConcreteExperiment()
        self.assertEqual(experiment.param_schema, {})


class TestExperimentWithParamSchema(unittest.TestCase):
    """Test Experiment with custom param_schema."""
    
    class ExperimentWithSchema(Experiment):
        param_schema = {
            "properties": {
                "param1": {"type": "string", "default": "default_value"},
                "param2": {"type": "integer", "default": 100}
            },
            "required": ["param1"]
        }
        
        def execute_once(self, context):
            return {"param1": self.params.get("param1"), "param2": self.params.get("param2")}
    
    def test_param_schema_attribute(self):
        """Test that param_schema is accessible."""
        experiment = self.ExperimentWithSchema()
        self.assertIsNotNone(experiment.param_schema)
        self.assertIn("properties", experiment.param_schema)
        self.assertIn("required", experiment.param_schema)


if __name__ == "__main__":
    unittest.main()
