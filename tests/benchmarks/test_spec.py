"""
Unit tests for the benchmarks.core.spec module.

Tests RunSpec loading, validation, and YAML/JSON parsing.
"""

import json
import tempfile
import unittest
from unittest.mock import Mock, patch, mock_open

from benchmarks.core.spec import RunSpec, load_spec, _load_raw, SPEC_DEFAULTS


class TestRunSpec(unittest.TestCase):
    """Test RunSpec dataclass."""

    def test_run_spec_creation(self):
        """Test RunSpec creation with all fields."""
        spec = RunSpec(
            experiment="test_experiment",
            params={"param1": "value1", "param2": 42},
            iterations={"warmup": 1, "measured": 3},
            instrumentation=["timing", "psutil"],
            output_dir="/tmp/results",
            tags=["test", "benchmark"],
            notes="Test run specification",
            seed=42,
            sweep={"param1": ["value1", "value2"]},
            parallelism=2,
            strategy="cartesian",
            samples=100
        )
        
        self.assertEqual(spec.experiment, "test_experiment")
        self.assertEqual(spec.params, {"param1": "value1", "param2": 42})
        self.assertEqual(spec.iterations, {"warmup": 1, "measured": 3})
        self.assertEqual(spec.instrumentation, ["timing", "psutil"])
        self.assertEqual(spec.output_dir, "/tmp/results")
        self.assertEqual(spec.tags, ["test", "benchmark"])
        self.assertEqual(spec.notes, "Test run specification")
        self.assertEqual(spec.seed, 42)
        self.assertEqual(spec.sweep, {"param1": ["value1", "value2"]})
        self.assertEqual(spec.parallelism, 2)
        self.assertEqual(spec.strategy, "cartesian")
        self.assertEqual(spec.samples, 100)

    def test_run_spec_with_none_values(self):
        """Test RunSpec creation with None values."""
        spec = RunSpec(
            experiment="test_experiment",
            params={},
            iterations={"warmup": 0, "measured": 1},
            instrumentation=["timing"],
            output_dir="/tmp/results",
            tags=[],
            notes="",
            seed=None,
            sweep=None,
            parallelism=1,
            strategy="cartesian",
            samples=None
        )
        
        self.assertEqual(spec.experiment, "test_experiment")
        self.assertEqual(spec.params, {})
        self.assertEqual(spec.iterations, {"warmup": 0, "measured": 1})
        self.assertEqual(spec.instrumentation, ["timing"])
        self.assertEqual(spec.output_dir, "/tmp/results")
        self.assertEqual(spec.tags, [])
        self.assertEqual(spec.notes, "")
        self.assertIsNone(spec.seed)
        self.assertIsNone(spec.sweep)
        self.assertEqual(spec.parallelism, 1)
        self.assertEqual(spec.strategy, "cartesian")
        self.assertIsNone(spec.samples)


class TestLoadRaw(unittest.TestCase):
    """Test _load_raw function."""

    def test_load_json_file(self):
        """Test loading JSON file."""
        data = {"experiment": "test", "params": {"param1": "value1"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = _load_raw(temp_path)
            self.assertEqual(result, data)
        finally:
            import os
            os.unlink(temp_path)

    def test_load_yaml_file(self):
        """Test loading YAML file."""
        data = {"experiment": "test", "params": {"param1": "value1"}}
        yaml_content = "experiment: test\nparams:\n  param1: value1\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            result = _load_raw(temp_path)
            self.assertEqual(result, data)
        finally:
            import os
            os.unlink(temp_path)

    def test_load_yml_file(self):
        """Test loading .yml file."""
        data = {"experiment": "test", "params": {"param1": "value1"}}
        yaml_content = "experiment: test\nparams:\n  param1: value1\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            result = _load_raw(temp_path)
            self.assertEqual(result, data)
        finally:
            import os
            os.unlink(temp_path)

    def test_load_fallback_to_json(self):
        """Test fallback to JSON when extension is unknown."""
        data = {"experiment": "test", "params": {"param1": "value1"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.unknown', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = _load_raw(temp_path)
            self.assertEqual(result, data)
        finally:
            import os
            os.unlink(temp_path)

    @patch('builtins.open', side_effect=IOError("File not found"))
    def test_load_raw_file_not_found(self, mock_open):
        """Test _load_raw with file not found."""
        with self.assertRaises(IOError):
            _load_raw("nonexistent.json")

    @patch('yaml.safe_load', side_effect=Exception("YAML parse error"))
    def test_load_yaml_parse_error(self, mock_yaml_load):
        """Test _load_raw with YAML parse error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with self.assertRaises(Exception):
                _load_raw(temp_path)
        finally:
            import os
            os.unlink(temp_path)

    @patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    def test_load_json_parse_error(self, mock_json_load):
        """Test _load_raw with JSON parse error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                _load_raw(temp_path)
        finally:
            import os
            os.unlink(temp_path)


class TestLoadSpec(unittest.TestCase):
    """Test load_spec function."""

    def test_load_spec_minimal(self):
        """Test loading minimal spec."""
        data = {"experiment": "test_experiment"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            spec = load_spec(temp_path)
            
            self.assertEqual(spec.experiment, "test_experiment")
            self.assertEqual(spec.params, {})
            self.assertEqual(spec.iterations, {"warmup": 0, "measured": 1})
            self.assertEqual(spec.instrumentation, ["timing"])
            self.assertEqual(spec.output_dir, "benchmarks/results")
            self.assertEqual(spec.tags, [])
            self.assertEqual(spec.notes, "")
            self.assertIsNone(spec.seed)
            self.assertIsNone(spec.sweep)
            self.assertEqual(spec.parallelism, 1)
            self.assertEqual(spec.strategy, "cartesian")
            self.assertIsNone(spec.samples)
        finally:
            import os
            os.unlink(temp_path)

    def test_load_spec_complete(self):
        """Test loading complete spec."""
        data = {
            "experiment": "complete_experiment",
            "params": {"param1": "value1", "param2": 42},
            "iterations": {"warmup": 2, "measured": 5},
            "instrumentation": ["timing", "psutil", "cprofile"],
            "output_dir": "/custom/results",
            "tags": ["test", "complete"],
            "notes": "Complete test specification",
            "seed": 123,
            "sweep": {"param1": ["value1", "value2"], "param2": [10, 20, 30]},
            "parallelism": 4,
            "strategy": "random",
            "samples": 50
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            spec = load_spec(temp_path)
            
            self.assertEqual(spec.experiment, "complete_experiment")
            self.assertEqual(spec.params, {"param1": "value1", "param2": 42})
            self.assertEqual(spec.iterations, {"warmup": 2, "measured": 5})
            self.assertEqual(spec.instrumentation, ["timing", "psutil", "cprofile"])
            self.assertEqual(spec.output_dir, "/custom/results")
            self.assertEqual(spec.tags, ["test", "complete"])
            self.assertEqual(spec.notes, "Complete test specification")
            self.assertEqual(spec.seed, 123)
            self.assertEqual(spec.sweep, {"param1": ["value1", "value2"], "param2": [10, 20, 30]})
            self.assertEqual(spec.parallelism, 4)
            self.assertEqual(spec.strategy, "random")
            self.assertEqual(spec.samples, 50)
        finally:
            import os
            os.unlink(temp_path)

    def test_load_spec_with_none_values(self):
        """Test loading spec with None values."""
        data = {
            "experiment": "none_test",
            "iterations": None,
            "instrumentation": None,
            "output_dir": None,
            "tags": None,
            "notes": None,
            "seed": None,
            "sweep": None,
            "samples": None
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            spec = load_spec(temp_path)
            
            self.assertEqual(spec.experiment, "none_test")
            self.assertEqual(spec.iterations, {"warmup": 0, "measured": 1})  # defaults
            self.assertEqual(spec.instrumentation, ["timing"])  # defaults
            self.assertEqual(spec.output_dir, "benchmarks/results")  # defaults
            self.assertEqual(spec.tags, [])  # defaults
            self.assertEqual(spec.notes, "None")  # defaults - None is converted to string
            self.assertIsNone(spec.seed)
            self.assertIsNone(spec.sweep)
            self.assertIsNone(spec.samples)
        finally:
            import os
            os.unlink(temp_path)

    def test_load_spec_missing_experiment_raises_error(self):
        """Test that missing experiment field raises ValueError."""
        data = {"params": {"param1": "value1"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError) as context:
                load_spec(temp_path)
            
            self.assertIn("missing required 'experiment' field", str(context.exception))
        finally:
            import os
            os.unlink(temp_path)

    def test_load_spec_invalid_iterations_raises_error(self):
        """Test that invalid iterations raises ValueError."""
        data = {
            "experiment": "test",
            "iterations": {"warmup": 1, "measured": 0}  # measured must be >= 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError) as context:
                load_spec(temp_path)
            
            self.assertIn("'iterations.measured' must be >= 1", str(context.exception))
        finally:
            import os
            os.unlink(temp_path)

    def test_load_spec_negative_measured_iterations_raises_error(self):
        """Test that negative measured iterations raises ValueError."""
        data = {
            "experiment": "test",
            "iterations": {"warmup": 1, "measured": -1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError) as context:
                load_spec(temp_path)
            
            self.assertIn("'iterations.measured' must be >= 1", str(context.exception))
        finally:
            import os
            os.unlink(temp_path)

    def test_load_spec_type_conversions(self):
        """Test that load_spec performs proper type conversions."""
        data = {
            "experiment": "type_test",
            "iterations": {"warmup": 2, "measured": 3},  # integers
            "seed": 42,  # integer
            "parallelism": 4,  # integer
            "samples": 100  # integer
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            spec = load_spec(temp_path)
            
            self.assertEqual(spec.iterations, {"warmup": 2, "measured": 3})
            self.assertEqual(spec.seed, 42)
            self.assertEqual(spec.parallelism, 4)
            self.assertEqual(spec.samples, 100)
        finally:
            import os
            os.unlink(temp_path)

    def test_load_spec_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{}")
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError) as context:
                load_spec(temp_path)
            
            self.assertIn("missing required 'experiment' field", str(context.exception))
        finally:
            import os
            os.unlink(temp_path)

    def test_load_spec_none_file(self):
        """Test loading file with None content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("null")
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError) as context:
                load_spec(temp_path)
            
            self.assertIn("missing required 'experiment' field", str(context.exception))
        finally:
            import os
            os.unlink(temp_path)

    @patch('benchmarks.core.spec._load_raw')
    def test_load_spec_calls_load_raw(self, mock_load_raw):
        """Test that load_spec calls _load_raw."""
        mock_load_raw.return_value = {"experiment": "test"}
        
        spec = load_spec("test.json")
        
        mock_load_raw.assert_called_once_with("test.json")
        self.assertEqual(spec.experiment, "test")


class TestSpecDefaults(unittest.TestCase):
    """Test SPEC_DEFAULTS constant."""

    def test_spec_defaults_structure(self):
        """Test that SPEC_DEFAULTS has expected structure."""
        self.assertIn("iterations", SPEC_DEFAULTS)
        self.assertIn("instrumentation", SPEC_DEFAULTS)
        self.assertIn("output_dir", SPEC_DEFAULTS)
        self.assertIn("strategy", SPEC_DEFAULTS)
        self.assertIn("parallelism", SPEC_DEFAULTS)
        
        self.assertEqual(SPEC_DEFAULTS["iterations"], {"warmup": 0, "measured": 1})
        self.assertEqual(SPEC_DEFAULTS["instrumentation"], ["timing"])
        self.assertEqual(SPEC_DEFAULTS["output_dir"], "benchmarks/results")
        self.assertEqual(SPEC_DEFAULTS["strategy"], "cartesian")
        self.assertEqual(SPEC_DEFAULTS["parallelism"], 1)


if __name__ == "__main__":
    unittest.main()
