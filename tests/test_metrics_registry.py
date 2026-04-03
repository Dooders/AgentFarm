"""Tests for farm/utils/metrics_registry.py."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from farm.utils.metrics_registry import (
    MetricsRegistry,
    get_metrics_registry,
    get_metric_definition,
    get_category_metrics,
    validate_metric_data,
    get_registry_summary,
    get_metrics_by_module,
    get_all_modules,
    get_module_categories,
    search_metrics,
)


def _make_sample_registry():
    """Return a minimal registry dict for testing."""
    return {
        "version": "1.0",
        "tags": {
            "frequency": "How often something happens",
            "reward": "Reward-related metrics",
        },
        "metric_categories": {
            "action_statistics": {
                "description": "Action frequency stats",
                "module": "actions",
                "metrics": {
                    "total_actions": {
                        "name": "total_actions",
                        "description": "Total actions per step",
                        "data_type": "dict",
                        "unit": "count",
                        "range_description": None,
                        "example_value": {"mean": 30},
                        "required": True,
                        "tags": ["frequency", "aggregate"],
                    },
                    "avg_success_rate": {
                        "name": "avg_success_rate",
                        "description": "Average success rate",
                        "data_type": "float",
                        "unit": "ratio",
                        "range_description": "0.0 to 1.0",
                        "example_value": 0.74,
                        "required": True,
                        "tags": ["reward", "success"],
                    },
                    "count": {
                        "name": "count",
                        "description": "Total count",
                        "data_type": "int",
                        "unit": "count",
                        "range_description": None,
                        "example_value": 5,
                        "required": True,
                        "tags": [],
                    },
                },
            },
            "agent_stats": {
                "description": "Agent statistics",
                "module": "agents",
                "metrics": {
                    "population": {
                        "name": "population",
                        "description": "Number of agents",
                        "data_type": "int",
                        "unit": "count",
                        "range_description": None,
                        "example_value": 10,
                        "required": True,
                        "tags": ["frequency"],
                    }
                },
            },
        },
    }


class TestMetricsRegistryWithFile(unittest.TestCase):
    def setUp(self):
        self.registry_data = _make_sample_registry()
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        json.dump(self.registry_data, self.tmp)
        self.tmp.close()
        self.registry = MetricsRegistry(registry_path=self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_load_registry_success(self):
        reg = self.registry.registry
        self.assertIn("metric_categories", reg)

    def test_get_metric_definition_exists(self):
        defn = self.registry.get_metric_definition("action_statistics", "total_actions")
        self.assertIsNotNone(defn)
        self.assertEqual(defn["name"], "total_actions")

    def test_get_metric_definition_missing_category(self):
        result = self.registry.get_metric_definition("nonexistent", "metric")
        self.assertIsNone(result)

    def test_get_metric_definition_missing_metric(self):
        result = self.registry.get_metric_definition("action_statistics", "nonexistent")
        self.assertIsNone(result)

    def test_get_category_metrics_exists(self):
        metrics = self.registry.get_category_metrics("action_statistics")
        self.assertIn("total_actions", metrics)
        self.assertIn("avg_success_rate", metrics)

    def test_get_category_metrics_missing(self):
        metrics = self.registry.get_category_metrics("nonexistent")
        self.assertEqual(metrics, {})

    def test_get_all_categories(self):
        categories = self.registry.get_all_categories()
        self.assertIn("action_statistics", categories)
        self.assertIn("agent_stats", categories)

    def test_get_metrics_by_tag(self):
        results = self.registry.get_metrics_by_tag("frequency")
        self.assertGreater(len(results), 0)
        for key in results:
            self.assertIn(".", key)

    def test_get_metrics_by_tag_missing(self):
        results = self.registry.get_metrics_by_tag("nonexistent_tag")
        self.assertEqual(results, {})

    def test_get_metrics_by_module(self):
        results = self.registry.get_metrics_by_module("actions")
        self.assertIn("action_statistics", results)

    def test_get_metrics_by_module_missing(self):
        results = self.registry.get_metrics_by_module("nonexistent_module")
        self.assertEqual(results, {})

    def test_get_all_modules(self):
        modules = self.registry.get_all_modules()
        self.assertIn("actions", modules)
        self.assertIn("agents", modules)
        # Should be sorted
        self.assertEqual(modules, sorted(modules))

    def test_get_module_categories(self):
        cats = self.registry.get_module_categories("actions")
        self.assertIn("action_statistics", cats)

    def test_get_module_categories_missing(self):
        cats = self.registry.get_module_categories("nonexistent")
        self.assertEqual(cats, [])

    def test_search_metrics_by_name(self):
        results = self.registry.search_metrics("total", search_type="name")
        self.assertIn("action_statistics.total_actions", results)

    def test_search_metrics_by_description(self):
        results = self.registry.search_metrics("success", search_type="description")
        self.assertGreater(len(results), 0)

    def test_search_metrics_by_tags(self):
        results = self.registry.search_metrics("reward", search_type="tags")
        self.assertGreater(len(results), 0)

    def test_search_metrics_any(self):
        results = self.registry.search_metrics("success")
        self.assertGreater(len(results), 0)

    def test_search_metrics_no_match(self):
        results = self.registry.search_metrics("xyzzyzzz")
        self.assertEqual(results, {})

    def test_validate_metric_data_valid_float(self):
        errors = self.registry.validate_metric_data(
            "action_statistics", "avg_success_rate", 0.5
        )
        self.assertEqual(errors, [])

    def test_validate_metric_data_float_out_of_range(self):
        errors = self.registry.validate_metric_data(
            "action_statistics", "avg_success_rate", 1.5
        )
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("outside valid range" in e for e in errors))

    def test_validate_metric_data_wrong_type_float(self):
        errors = self.registry.validate_metric_data(
            "action_statistics", "avg_success_rate", "not_a_float"
        )
        self.assertTrue(len(errors) > 0)

    def test_validate_metric_data_valid_int(self):
        errors = self.registry.validate_metric_data(
            "action_statistics", "count", 42
        )
        self.assertEqual(errors, [])

    def test_validate_metric_data_wrong_type_int(self):
        errors = self.registry.validate_metric_data(
            "action_statistics", "count", "not_int"
        )
        self.assertTrue(len(errors) > 0)

    def test_validate_metric_data_unknown_metric(self):
        errors = self.registry.validate_metric_data(
            "action_statistics", "nonexistent_metric", 42
        )
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("Unknown metric" in e for e in errors))

    def test_get_registry_summary(self):
        summary = self.registry.get_registry_summary()
        self.assertIn("version", summary)
        self.assertIn("total_categories", summary)
        self.assertIn("categories", summary)
        self.assertEqual(summary["version"], "1.0")

    def test_export_category_to_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            self.registry.export_category_to_json("action_statistics", out_path)
            with open(out_path) as f:
                data = json.load(f)
            self.assertEqual(data["category"], "action_statistics")
            self.assertIn("total_actions", data["metrics"])
        finally:
            os.unlink(out_path)

    def test_export_category_missing_does_nothing(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            # Should not raise even for missing category
            self.registry.export_category_to_json("nonexistent", out_path)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_list_all_tags(self):
        tags = self.registry.list_all_tags()
        self.assertIn("frequency", tags)
        self.assertIn("reward", tags)
        self.assertEqual(tags, sorted(tags))

    def test_lazy_load_registry_called_once(self):
        """Registry should only be loaded once (lazy load)."""
        reg1 = self.registry.registry
        reg2 = self.registry.registry
        self.assertIs(reg1, reg2)


class TestMetricsRegistryFileNotFound(unittest.TestCase):
    def test_missing_file_raises(self):
        registry = MetricsRegistry(registry_path="/nonexistent/path/registry.json")
        with self.assertRaises(FileNotFoundError):
            _ = registry.registry


class TestMetricsRegistryDefaultPath(unittest.TestCase):
    def test_default_path_uses_real_registry(self):
        """If metrics_registry.json exists in project root, it should load."""
        project_root = Path(__file__).parent.parent
        registry_path = project_root / "metrics_registry.json"
        if not registry_path.exists():
            self.skipTest("metrics_registry.json not found in project root")

        registry = MetricsRegistry()
        self.assertIsNotNone(registry.registry)
        self.assertIn("metric_categories", registry.registry)


class TestConvenienceFunctions(unittest.TestCase):
    """Test the module-level convenience functions against the real registry."""

    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).parent.parent
        registry_path = project_root / "metrics_registry.json"
        if not registry_path.exists():
            raise unittest.SkipTest("metrics_registry.json not found")

    def test_get_registry_summary_returns_dict(self):
        summary = get_registry_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("total_categories", summary)

    def test_get_all_modules_returns_list(self):
        modules = get_all_modules()
        self.assertIsInstance(modules, list)
        self.assertGreater(len(modules), 0)

    def test_search_metrics_returns_dict(self):
        results = search_metrics("rate")
        self.assertIsInstance(results, dict)

    def test_get_metrics_by_module_returns_dict(self):
        modules = get_all_modules()
        if modules:
            results = get_metrics_by_module(modules[0])
            self.assertIsInstance(results, dict)

    def test_get_module_categories_returns_list(self):
        modules = get_all_modules()
        if modules:
            cats = get_module_categories(modules[0])
            self.assertIsInstance(cats, list)
