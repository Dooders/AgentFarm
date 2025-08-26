"""
Tests for the MetricsTracker class.
"""

import unittest

from farm.core.metrics_tracker import CumulativeMetrics, MetricsTracker, StepMetrics


class TestMetricsTracker(unittest.TestCase):
    """Test cases for MetricsTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = MetricsTracker()

    def test_initial_state(self):
        """Test that tracker initializes with zero metrics."""
        self.assertEqual(self.tracker.step_metrics.births, 0)
        self.assertEqual(self.tracker.step_metrics.deaths, 0)
        self.assertEqual(self.tracker.step_metrics.combat_encounters, 0)
        self.assertEqual(self.tracker.step_metrics.successful_attacks, 0)
        self.assertEqual(self.tracker.step_metrics.resources_shared, 0.0)

    def test_record_birth(self):
        """Test recording birth events."""
        self.tracker.record_birth()
        self.assertEqual(self.tracker.step_metrics.births, 1)

        self.tracker.record_birth()
        self.assertEqual(self.tracker.step_metrics.births, 2)

    def test_record_death(self):
        """Test recording death events."""
        self.tracker.record_death()
        self.assertEqual(self.tracker.step_metrics.deaths, 1)

        self.tracker.record_death()
        self.assertEqual(self.tracker.step_metrics.deaths, 2)

    def test_record_combat(self):
        """Test recording combat events."""
        self.tracker.record_combat_encounter()
        self.assertEqual(self.tracker.step_metrics.combat_encounters, 1)

        self.tracker.record_successful_attack()
        self.assertEqual(self.tracker.step_metrics.successful_attacks, 1)

    def test_record_resources_shared(self):
        """Test recording resource sharing."""
        self.tracker.record_resources_shared(5.0)
        self.assertEqual(self.tracker.step_metrics.resources_shared, 5.0)

        self.tracker.record_resources_shared(3.5)
        self.assertEqual(self.tracker.step_metrics.resources_shared, 8.5)

    def test_end_step(self):
        """Test ending a step and getting metrics."""
        # Record some events
        self.tracker.record_birth()
        self.tracker.record_death()
        self.tracker.record_combat_encounter()
        self.tracker.record_successful_attack()
        self.tracker.record_resources_shared(10.0)

        # End step
        metrics = self.tracker.end_step()

        # Check that metrics are returned
        self.assertEqual(metrics["births"], 1)
        self.assertEqual(metrics["deaths"], 1)
        self.assertEqual(metrics["combat_encounters"], 1)
        self.assertEqual(metrics["successful_attacks"], 1)
        self.assertEqual(metrics["resources_shared"], 10.0)

        # Check that step metrics are reset
        self.assertEqual(self.tracker.step_metrics.births, 0)
        self.assertEqual(self.tracker.step_metrics.deaths, 0)

        # Check that cumulative metrics are updated
        self.assertEqual(self.tracker.cumulative_metrics.total_births, 1)
        self.assertEqual(self.tracker.cumulative_metrics.total_deaths, 1)

    def test_multiple_steps(self):
        """Test tracking across multiple steps."""
        # Step 1
        self.tracker.record_birth()
        self.tracker.record_combat_encounter()
        metrics1 = self.tracker.end_step()

        # Step 2
        self.tracker.record_death()
        self.tracker.record_successful_attack()
        self.tracker.record_resources_shared(5.0)
        metrics2 = self.tracker.end_step()

        # Check cumulative metrics
        self.assertEqual(metrics2["total_births"], 1)
        self.assertEqual(metrics2["total_deaths"], 1)
        self.assertEqual(metrics2["total_combat_encounters"], 1)
        self.assertEqual(metrics2["total_successful_attacks"], 1)
        self.assertEqual(metrics2["total_resources_shared"], 5.0)

    def test_custom_metrics(self):
        """Test adding custom metrics."""
        self.tracker.add_custom_metric("custom_metric", 42)
        self.tracker.add_custom_metric("another_metric", "test_value")

        metrics = self.tracker.get_step_metrics()
        self.assertEqual(metrics["custom_metric"], 42)
        self.assertEqual(metrics["another_metric"], "test_value")

    def test_reset(self):
        """Test resetting all metrics."""
        # Add some metrics
        self.tracker.record_birth()
        self.tracker.record_death()
        self.tracker.add_custom_metric("test", 123)

        # End step to accumulate
        self.tracker.end_step()

        # Reset
        self.tracker.reset()

        # Check everything is reset
        self.assertEqual(self.tracker.step_metrics.births, 0)
        self.assertEqual(self.tracker.cumulative_metrics.total_births, 0)
        self.assertEqual(len(self.tracker.custom_metrics), 0)

    def test_metrics_summary(self):
        """Test getting metrics summary."""
        # Add some metrics
        self.tracker.record_birth()
        self.tracker.record_birth()
        self.tracker.record_death()
        self.tracker.record_combat_encounter()
        self.tracker.record_combat_encounter()
        self.tracker.record_successful_attack()
        self.tracker.record_resources_shared(10.0)

        self.tracker.end_step()

        summary = self.tracker.get_metrics_summary()

        self.assertEqual(summary["population_growth"], 1)  # 2 births - 1 death
        self.assertEqual(
            summary["combat_success_rate"], 0.5
        )  # 1 success / 2 encounters
        self.assertEqual(summary["total_resources_shared"], 10.0)


class TestStepMetrics(unittest.TestCase):
    """Test cases for StepMetrics dataclass."""

    def test_reset(self):
        """Test resetting step metrics."""
        metrics = StepMetrics()
        metrics.births = 5
        metrics.deaths = 3
        metrics.combat_encounters = 10

        metrics.reset()

        self.assertEqual(metrics.births, 0)
        self.assertEqual(metrics.deaths, 0)
        self.assertEqual(metrics.combat_encounters, 0)


class TestCumulativeMetrics(unittest.TestCase):
    """Test cases for CumulativeMetrics dataclass."""

    def test_update_from_step(self):
        """Test updating cumulative metrics from step metrics."""
        cumulative = CumulativeMetrics()
        step = StepMetrics()

        step.births = 2
        step.deaths = 1
        step.combat_encounters = 5
        step.successful_attacks = 3
        step.resources_shared = 15.0

        cumulative.update_from_step(step)

        self.assertEqual(cumulative.total_births, 2)
        self.assertEqual(cumulative.total_deaths, 1)
        self.assertEqual(cumulative.total_combat_encounters, 5)
        self.assertEqual(cumulative.total_successful_attacks, 3)
        self.assertEqual(cumulative.total_resources_shared, 15.0)


if __name__ == "__main__":
    unittest.main()
