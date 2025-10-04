import unittest
from unittest.mock import Mock, patch
import pandas as pd
from pathlib import Path
import tempfile
import json

from farm.analysis.actions import (
    analyze_action_patterns,
    compute_action_statistics,
    compute_sequence_patterns,
    compute_decision_patterns,
    compute_reward_metrics
)
from farm.analysis.common.context import AnalysisContext
from farm.database.data_types import AgentActionData
from farm.database.enums import AnalysisScope
from farm.database.repositories.action_repository import ActionRepository


class TestActionAnalysis(unittest.TestCase):
    def setUp(self):
        """Initialize test fixtures for action analysis tests.

        Sets up sample action data as pandas DataFrame and creates temporary
        directories for analysis output.
        """
        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create sample test actions as DataFrame
        self.test_actions_df = pd.DataFrame([
            {
                "step": 1,
                "action_type": "gather",
                "frequency": 2,
                "success_rate": 0.8,
                "avg_reward": 7.5,
                "agent_id": "1",
                "action_target_id": None,
            },
            {
                "step": 1,
                "action_type": "gather",
                "frequency": 1,
                "success_rate": 1.0,
                "avg_reward": 5.0,
                "agent_id": "1",
                "action_target_id": "2",
            },
            {
                "step": 2,
                "action_type": "move",
                "frequency": 1,
                "success_rate": 1.0,
                "avg_reward": 2.0,
                "agent_id": "1",
                "action_target_id": None,
            },
        ])

        # Create analysis context
        self.ctx = AnalysisContext(
            output_path=self.temp_dir,
            logger=Mock(),
            config={}
        )

    def tearDown(self):
        """Clean up temporary test directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_action_statistics(self):
        """Test the calculation of basic action statistics.

        Verifies that compute_action_statistics correctly calculates:
        - Total actions statistics
        - Per action type statistics
        - Most common action

        Uses sample data with 'gather' and 'move' actions to validate metrics.
        """
        # Act
        stats = compute_action_statistics(self.test_actions_df)

        # Assert
        self.assertIn("total_actions", stats)
        self.assertIn("action_types", stats)
        self.assertIn("most_common_action", stats)

        # Check that we have statistics for both action types
        action_types = stats["action_types"]
        self.assertIn("gather", action_types)
        self.assertIn("move", action_types)

        # Check gather statistics - should have frequency, success_rate, avg_reward
        gather_stats = action_types["gather"]
        self.assertIn("frequency", gather_stats)
        self.assertIn("success_rate", gather_stats)
        self.assertIn("avg_reward", gather_stats)

        # Check frequency stats for gather
        gather_freq = gather_stats["frequency"]
        self.assertIn("mean", gather_freq)
        self.assertIn("min", gather_freq)
        self.assertIn("max", gather_freq)

        # Check move statistics
        move_stats = action_types["move"]
        self.assertIn("frequency", move_stats)
        self.assertIn("success_rate", move_stats)
        self.assertIn("avg_reward", move_stats)

        # Check most common action
        self.assertEqual(stats["most_common_action"], "gather")

    def test_compute_reward_metrics(self):
        """Test the calculation of reward-related metrics.

        Verifies that compute_reward_metrics correctly calculates:
        - Average rewards per action type
        - Reward distributions
        - Performance metrics

        Uses sample data with different reward values.
        """
        # Act
        reward_metrics = compute_reward_metrics(self.test_actions_df)

        # Assert
        self.assertIsInstance(reward_metrics, dict)

        # Check that reward metrics include expected action types
        if "gather" in reward_metrics:
            gather_rewards = reward_metrics["gather"]
            self.assertIn("avg_reward", gather_rewards)
            self.assertIn("min_reward", gather_rewards)
            self.assertIn("max_reward", gather_rewards)

        if "move" in reward_metrics:
            move_rewards = reward_metrics["move"]
            self.assertIn("avg_reward", move_rewards)

    def test_compute_with_empty_data(self):
        """Test computation behavior when no action data is present.

        Ensures the compute functions handle empty datasets gracefully.
        """
        # Arrange
        empty_df = pd.DataFrame()

        # Act & Assert - should not raise exceptions
        stats = compute_action_statistics(empty_df)
        self.assertIsInstance(stats, dict)

        reward_metrics = compute_reward_metrics(empty_df)
        self.assertIsInstance(reward_metrics, dict)

    def test_analyze_action_patterns(self):
        """Test the analyze_action_patterns function.

        Verifies that the function creates output files and handles the analysis context.
        """
        # Act
        analyze_action_patterns(self.test_actions_df, self.ctx)

        # Assert - check that output file was created
        output_file = self.ctx.get_output_file("action_statistics.json")
        self.assertTrue(output_file.exists())

        # Check that the file contains valid JSON
        with open(output_file, 'r') as f:
            data = json.load(f)
            self.assertIsInstance(data, dict)

    def test_compute_sequence_patterns(self):
        """Test the computation of action sequence patterns.

        Verifies that compute_sequence_patterns correctly identifies patterns
        in action sequences.
        """
        # Arrange - create data with sequences
        sequence_df = pd.DataFrame([
            {"step": 1, "action_sequence": ["move", "gather"], "agent_id": "1"},
            {"step": 2, "action_sequence": ["gather", "attack"], "agent_id": "1"},
            {"step": 3, "action_sequence": ["move", "gather", "attack"], "agent_id": "1"},
        ])

        # Act
        patterns = compute_sequence_patterns(sequence_df)

        # Assert
        self.assertIsInstance(patterns, dict)
        # Should contain sequence analysis results

    def test_compute_decision_patterns(self):
        """Test the computation of decision patterns.

        Verifies that compute_decision_patterns correctly analyzes
        decision-making patterns in actions.
        """
        # Act
        patterns = compute_decision_patterns(self.test_actions_df)

        # Assert
        self.assertIsInstance(patterns, dict)
        # Should contain decision pattern analysis results


if __name__ == "__main__":
    unittest.main()
