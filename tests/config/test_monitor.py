"""Tests for config monitor module (ConfigMonitor, ConfigMetrics, health functions)."""

import json
import os
import tempfile
import time
import unittest

from farm.config.monitor import (
    ConfigMetrics,
    ConfigMonitor,
    get_config_system_health,
    get_global_monitor,
    log_config_system_status,
    monitor_config_operation,
)


class TestConfigMetrics(unittest.TestCase):
    """Tests for ConfigMetrics dataclass."""

    def test_defaults(self):
        """ConfigMetrics sets timestamp automatically when not provided."""
        m = ConfigMetrics(operation="load", duration=0.1, success=True)
        self.assertEqual(m.operation, "load")
        self.assertEqual(m.duration, 0.1)
        self.assertTrue(m.success)
        self.assertFalse(m.cache_hit)
        self.assertEqual(m.config_size, 0)
        self.assertIsNone(m.error_type)
        self.assertIsNone(m.environment)
        self.assertIsNone(m.profile)
        self.assertIsNotNone(m.timestamp)

    def test_explicit_timestamp(self):
        """ConfigMetrics accepts an explicit timestamp."""
        ts = 1234567890.0
        m = ConfigMetrics(operation="save", duration=0.2, success=False, timestamp=ts)
        self.assertEqual(m.timestamp, ts)

    def test_error_type_captured(self):
        """error_type field stores error class name."""
        m = ConfigMetrics(operation="load", duration=0.0, success=False, error_type="ValueError")
        self.assertEqual(m.error_type, "ValueError")


class TestConfigMonitor(unittest.TestCase):
    """Tests for ConfigMonitor class."""

    def setUp(self):
        self.monitor = ConfigMonitor()

    def test_log_successful_operation(self):
        """Logging a successful operation appends to metrics."""
        self.monitor.log_config_operation("load", duration=0.05, success=True)
        self.assertEqual(len(self.monitor.metrics), 1)
        self.assertTrue(self.monitor.metrics[0].success)

    def test_log_failed_operation(self):
        """Logging a failed operation stores error info."""
        err = ValueError("bad config")
        self.monitor.log_config_operation("save", duration=0.1, success=False, error=err)
        self.assertEqual(len(self.monitor.metrics), 1)
        self.assertFalse(self.monitor.metrics[0].success)
        self.assertEqual(self.monitor.metrics[0].error_type, "ValueError")

    def test_log_with_cache_hit(self):
        """Cache-hit operations are recorded."""
        self.monitor.log_config_operation("load", duration=0.001, success=True, cache_hit=True)
        self.assertTrue(self.monitor.metrics[0].cache_hit)

    def test_log_with_config_object(self):
        """Passing a config object records config size."""
        from farm.config import SimulationConfig

        config = SimulationConfig()
        self.monitor.log_config_operation("load", config=config, duration=0.05, success=True)
        self.assertGreater(self.monitor.metrics[0].config_size, 0)

    def test_metrics_history_capped(self):
        """Metrics history is capped at max_metrics_history."""
        self.monitor.max_metrics_history = 5
        for i in range(10):
            self.monitor.log_config_operation("load", duration=0.01, success=True)
        self.assertLessEqual(len(self.monitor.metrics), 5)

    def test_get_metrics_summary_empty(self):
        """Summary for empty metrics returns total_operations=0."""
        summary = self.monitor.get_metrics_summary()
        self.assertEqual(summary["total_operations"], 0)

    def test_get_metrics_summary(self):
        """Summary accurately reflects logged operations."""
        self.monitor.log_config_operation("load", duration=0.1, success=True)
        self.monitor.log_config_operation("save", duration=0.2, success=False, error=Exception())
        summary = self.monitor.get_metrics_summary()
        self.assertEqual(summary["total_operations"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertIn("operation_stats", summary)

    def test_get_metrics_summary_operation_stats(self):
        """Operation stats break down per-operation metrics."""
        self.monitor.log_config_operation("load", duration=0.1, success=True)
        self.monitor.log_config_operation("load", duration=0.2, success=True)
        self.monitor.log_config_operation("save", duration=0.3, success=False, error=Exception())
        summary = self.monitor.get_metrics_summary()
        ops = summary["operation_stats"]
        self.assertIn("load", ops)
        self.assertEqual(ops["load"]["count"], 2)

    def test_get_recent_errors_empty(self):
        """No errors returns empty list."""
        errors = self.monitor.get_recent_errors()
        self.assertEqual(errors, [])

    def test_get_recent_errors_filters_failures(self):
        """get_recent_errors returns only failed operations."""
        self.monitor.log_config_operation("load", duration=0.1, success=True)
        self.monitor.log_config_operation("save", duration=0.2, success=False, error=Exception())
        errors = self.monitor.get_recent_errors()
        self.assertEqual(len(errors), 1)
        self.assertFalse(errors[0].success)

    def test_get_recent_errors_respects_limit(self):
        """get_recent_errors respects the limit parameter."""
        for _ in range(10):
            self.monitor.log_config_operation("load", duration=0.1, success=False, error=Exception())
        errors = self.monitor.get_recent_errors(limit=3)
        self.assertLessEqual(len(errors), 3)

    def test_get_performance_trends_insufficient_data(self):
        """Trend analysis returns insufficient_data for < 2 samples."""
        self.monitor.log_config_operation("load", duration=0.1, success=True)
        result = self.monitor.get_performance_trends("load")
        self.assertIn("insufficient_data", result)

    def test_get_performance_trends_all_operations(self):
        """Trend analysis works for all operations combined."""
        for i in range(5):
            self.monitor.log_config_operation("load", duration=0.1 * (i + 1), success=True)
        result = self.monitor.get_performance_trends()
        self.assertNotIn("insufficient_data", result)
        self.assertIn("total_samples", result)
        self.assertIn("avg_duration", result)

    def test_get_performance_trends_specific_operation(self):
        """Trend analysis can be filtered by operation name."""
        for i in range(4):
            self.monitor.log_config_operation("load", duration=0.1, success=True)
        result = self.monitor.get_performance_trends("load")
        self.assertEqual(result["operation"], "load")

    def test_export_metrics(self):
        """export_metrics writes a valid JSON file."""
        self.monitor.log_config_operation("load", duration=0.05, success=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            self.monitor.export_metrics(fname)
            self.assertTrue(os.path.exists(fname))
            with open(fname) as f:
                data = json.load(f)
            self.assertIn("exported_at", data)
            self.assertIn("metrics", data)
            self.assertEqual(len(data["metrics"]), 1)
        finally:
            os.unlink(fname)

    def test_measure_operation_context_manager_success(self):
        """measure_operation logs a successful timed operation."""
        with self.monitor.measure_operation("load", environment="dev"):
            pass
        self.assertEqual(len(self.monitor.metrics), 1)
        self.assertTrue(self.monitor.metrics[0].success)

    def test_measure_operation_context_manager_failure(self):
        """measure_operation logs a failed operation when an exception is raised."""
        with self.assertRaises(RuntimeError):
            with self.monitor.measure_operation("save"):
                raise RuntimeError("boom")
        self.assertEqual(len(self.monitor.metrics), 1)
        self.assertFalse(self.monitor.metrics[0].success)


class TestGetGlobalMonitor(unittest.TestCase):
    """Tests for get_global_monitor."""

    def test_returns_monitor_instance(self):
        m = get_global_monitor()
        self.assertIsInstance(m, ConfigMonitor)

    def test_returns_same_instance(self):
        m1 = get_global_monitor()
        m2 = get_global_monitor()
        self.assertIs(m1, m2)


class TestGetConfigSystemHealth(unittest.TestCase):
    """Tests for get_config_system_health."""

    def test_returns_health_dict(self):
        health = get_config_system_health()
        self.assertIn("status", health)
        self.assertIn("success_rate", health)
        self.assertIn("total_operations", health)
        self.assertIn("cache_hit_rate", health)
        self.assertIn("recent_errors", health)

    def test_status_is_valid_string(self):
        health = get_config_system_health()
        self.assertIn(health["status"], ("healthy", "warning", "unhealthy"))


class TestLogConfigSystemStatus(unittest.TestCase):
    """Tests for log_config_system_status."""

    def test_runs_without_error(self):
        """log_config_system_status should not raise."""
        log_config_system_status()


class TestMonitorConfigOperationDecorator(unittest.TestCase):
    """Tests for monitor_config_operation decorator."""

    def test_decorator_wraps_function(self):
        """Decorated function still returns its result."""

        @monitor_config_operation("test_op")
        def my_func():
            return 42

        result = my_func()
        self.assertEqual(result, 42)

    def test_decorator_records_metrics(self):
        """Decorated function records metrics in global monitor."""
        monitor = get_global_monitor()
        initial_count = len(monitor.metrics)

        @monitor_config_operation("decorated_op")
        def my_func():
            pass

        my_func()
        self.assertGreater(len(monitor.metrics), initial_count)


if __name__ == "__main__":
    unittest.main()
