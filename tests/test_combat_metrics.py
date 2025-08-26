"""Test for combat metrics in the environment."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from farm.core.environment import Environment


class TestCombatMetrics(unittest.TestCase):
    """Test case for combat metrics."""

    def setUp(self):
        """Set up test environment."""
        # Create a test environment
        self.env = Environment(
            width=10,
            height=10,
            resource_distribution={"amount": 5},
            db_path=":memory:",  # Use in-memory database
            config=MagicMock(
                max_resource_amount=10,
                resource_regen_rate=0.1,
                resource_regen_amount=1,
                seed=42,  # Add a proper seed value
            ),
        )

    def test_combat_encounters_increment(self):
        """Test that combat_encounters counter increments correctly."""
        # Verify initial counters are zero
        step_metrics = self.env.metrics_tracker.get_step_metrics()
        self.assertEqual(step_metrics["combat_encounters"], 0)

        # Manually increment counters using tracker methods
        self.env.record_combat_encounter()

        # Verify counters incremented
        step_metrics = self.env.metrics_tracker.get_step_metrics()
        self.assertEqual(step_metrics["combat_encounters"], 1)

        # Increment again
        self.env.record_combat_encounter()

        # Verify counters incremented again
        step_metrics = self.env.metrics_tracker.get_step_metrics()
        self.assertEqual(step_metrics["combat_encounters"], 2)

        # End step to reset step metrics and update cumulative
        self.env.metrics_tracker.end_step()

        # Verify cumulative metrics persist but step metrics reset
        cumulative_metrics = self.env.metrics_tracker.get_cumulative_metrics()
        step_metrics = self.env.metrics_tracker.get_step_metrics()
        self.assertEqual(cumulative_metrics["total_combat_encounters"], 2)
        self.assertEqual(step_metrics["combat_encounters"], 0)

    def test_successful_attacks_increment(self):
        """Test that successful_attacks counter increments correctly."""
        # Verify initial counters are zero
        step_metrics = self.env.metrics_tracker.get_step_metrics()
        self.assertEqual(step_metrics["successful_attacks"], 0)

        # Manually increment counters using tracker methods
        self.env.record_successful_attack()

        # Verify counters incremented
        step_metrics = self.env.metrics_tracker.get_step_metrics()
        self.assertEqual(step_metrics["successful_attacks"], 1)

        # End step to reset step metrics and update cumulative
        self.env.metrics_tracker.end_step()

        # Verify cumulative metrics persist but step metrics reset
        cumulative_metrics = self.env.metrics_tracker.get_cumulative_metrics()
        step_metrics = self.env.metrics_tracker.get_step_metrics()
        self.assertEqual(cumulative_metrics["total_successful_attacks"], 1)
        self.assertEqual(step_metrics["successful_attacks"], 0)

    def test_metrics_in_calculate_metrics(self):
        """Test that metrics are included in the calculated metrics dictionary."""
        # Manually increment counters using tracker methods
        for _ in range(3):
            self.env.record_combat_encounter()
        for _ in range(2):
            self.env.record_successful_attack()

        # Calculate metrics
        metrics = self.env._calculate_metrics()

        # Verify metrics include combat counters
        self.assertEqual(metrics["combat_encounters"], 3)
        self.assertEqual(metrics["successful_attacks"], 2)
        self.assertEqual(metrics["combat_encounters_this_step"], 3)
        self.assertEqual(metrics["successful_attacks_this_step"], 2)


if __name__ == "__main__":
    unittest.main()
