"""Test for combat metrics in the environment."""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
            db_path=None,  # Use in-memory database
            config=MagicMock(
                max_resource_amount=10,
                resource_regen_rate=0.1,
                resource_regen_amount=1,
            ),
        )

    def test_combat_encounters_increment(self):
        """Test that combat_encounters counter increments correctly."""
        # Verify initial counters are zero
        self.assertEqual(self.env.combat_encounters, 0)
        self.assertEqual(self.env.combat_encounters_this_step, 0)
        
        # Manually increment counters
        self.env.combat_encounters += 1
        self.env.combat_encounters_this_step += 1
        
        # Verify counters incremented
        self.assertEqual(self.env.combat_encounters, 1)
        self.assertEqual(self.env.combat_encounters_this_step, 1)
        
        # Increment again
        self.env.combat_encounters += 1
        self.env.combat_encounters_this_step += 1
        
        # Verify counters incremented again
        self.assertEqual(self.env.combat_encounters, 2)
        self.assertEqual(self.env.combat_encounters_this_step, 2)
        
        # Update environment to reset this_step counter
        self.env.update()
        
        # Verify total counter persists but this_step counter resets
        self.assertEqual(self.env.combat_encounters, 2)
        self.assertEqual(self.env.combat_encounters_this_step, 0)

    def test_successful_attacks_increment(self):
        """Test that successful_attacks counter increments correctly."""
        # Verify initial counters are zero
        self.assertEqual(self.env.successful_attacks, 0)
        self.assertEqual(self.env.successful_attacks_this_step, 0)
        
        # Manually increment counters
        self.env.successful_attacks += 1
        self.env.successful_attacks_this_step += 1
        
        # Verify counters incremented
        self.assertEqual(self.env.successful_attacks, 1)
        self.assertEqual(self.env.successful_attacks_this_step, 1)
        
        # Update environment to reset this_step counter
        self.env.update()
        
        # Verify total counter persists but this_step counter resets
        self.assertEqual(self.env.successful_attacks, 1)
        self.assertEqual(self.env.successful_attacks_this_step, 0)

    def test_metrics_in_calculate_metrics(self):
        """Test that metrics are included in the calculated metrics dictionary."""
        # Manually increment counters
        self.env.combat_encounters = 3
        self.env.successful_attacks = 2
        self.env.combat_encounters_this_step = 1
        self.env.successful_attacks_this_step = 1
        
        # Calculate metrics
        metrics = self.env._calculate_metrics()
        
        # Verify metrics include combat counters
        self.assertEqual(metrics["combat_encounters"], 3)
        self.assertEqual(metrics["successful_attacks"], 2)
        self.assertEqual(metrics["combat_encounters_this_step"], 1)
        self.assertEqual(metrics["successful_attacks_this_step"], 1)


if __name__ == "__main__":
    unittest.main() 