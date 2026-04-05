"""Tests for chart_actions module."""

import json
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from farm.charts import chart_actions


def _make_actions_df():
    """Build a minimal actions DataFrame with all required columns."""
    return pd.DataFrame(
        {
            "action_type": ["move", "eat", "move", "attack", "eat"],
            "reward": [0.1, 0.5, 0.2, -0.1, 0.4],
            "step_number": [1, 1, 2, 2, 3],
            "resources_before": [10.0, 10.0, 9.5, 9.0, 9.5],
            "resources_after": [9.5, 11.0, 9.0, 8.0, 11.0],
            "agent_id": ["a1", "a1", "a1", "a2", "a2"],
            "details": [
                json.dumps({"target_position": [3, 4]}),
                json.dumps({"target_id": "r1"}),
                json.dumps({"target_position": [5, 6]}),
                json.dumps({"target_id": "a2"}),
                json.dumps({"target_id": "r2"}),
            ],
        }
    )


class TestChartActions(unittest.TestCase):
    """Tests for chart_actions plotting functions."""

    @patch("farm.charts.chart_actions.plt")
    def test_plot_action_type_distribution(self, mock_plt):
        """plot_action_type_distribution creates a figure and returns plt."""
        df = _make_actions_df()
        result = chart_actions.plot_action_type_distribution(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_rewards_by_action_type(self, mock_plt):
        """plot_rewards_by_action_type creates a figure and returns plt."""
        df = _make_actions_df()
        result = chart_actions.plot_rewards_by_action_type(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_resource_changes(self, mock_plt):
        """plot_resource_changes creates a histogram figure and returns plt."""
        df = _make_actions_df()
        result = chart_actions.plot_resource_changes(df)
        mock_plt.figure.assert_called_once()
        mock_plt.hist.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_action_frequency_over_time(self, mock_plt):
        """plot_action_frequency_over_time creates a stacked area chart and returns plt."""
        df = _make_actions_df()
        result = chart_actions.plot_action_frequency_over_time(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_position_changes_with_positions(self, mock_plt):
        """plot_position_changes returns plt when position data exists."""
        df = _make_actions_df()
        result = chart_actions.plot_position_changes(df, "a1")
        self.assertIsNotNone(result)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_position_changes_no_target_position(self, mock_plt):
        """plot_position_changes returns None when no target_position in details."""
        df = pd.DataFrame(
            {
                "action_type": ["eat"],
                "reward": [0.5],
                "step_number": [1],
                "resources_before": [10.0],
                "resources_after": [11.0],
                "agent_id": ["a1"],
                "details": [json.dumps({"target_id": "r1"})],
            }
        )
        result = chart_actions.plot_position_changes(df, "a1")
        self.assertIsNone(result)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_position_changes_null_details(self, mock_plt):
        """plot_position_changes returns None when details is null."""
        df = pd.DataFrame(
            {
                "action_type": ["eat"],
                "reward": [0.5],
                "step_number": [1],
                "resources_before": [10.0],
                "resources_after": [11.0],
                "agent_id": ["a1"],
                "details": [None],
            }
        )
        result = chart_actions.plot_position_changes(df, "a1")
        self.assertIsNone(result)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_rewards_over_time(self, mock_plt):
        """plot_rewards_over_time creates a line figure and returns plt."""
        df = _make_actions_df()
        result = chart_actions.plot_rewards_over_time(df)
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_action_target_distribution(self, mock_plt):
        """plot_action_target_distribution creates a bar figure and returns plt."""
        df = _make_actions_df()
        result = chart_actions.plot_action_target_distribution(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_action_target_distribution_null_details(self, mock_plt):
        """plot_action_target_distribution handles null details gracefully."""
        df = pd.DataFrame(
            {
                "action_type": ["eat", "move"],
                "reward": [0.5, 0.1],
                "step_number": [1, 2],
                "resources_before": [10.0, 9.0],
                "resources_after": [11.0, 9.0],
                "agent_id": ["a1", "a1"],
                "details": [None, None],
            }
        )
        # Should not raise even with null details
        result = chart_actions.plot_action_target_distribution(df)
        self.assertIsNotNone(result)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_position_changes_exception_returns_none(self, mock_plt):
        """plot_position_changes returns None when an exception occurs during processing."""
        # Use null target_position which causes tuple(None) TypeError
        df = pd.DataFrame(
            {
                "action_type": ["move"],
                "reward": [0.1],
                "step_number": [1],
                "resources_before": [10.0],
                "resources_after": [9.5],
                "agent_id": ["a1"],
                "details": ['{"target_position": null}'],
            }
        )
        # tuple(None) raises TypeError, which is caught and returns None
        result = chart_actions.plot_position_changes(df, "a1")
        self.assertIsNone(result)

    @patch("farm.charts.chart_actions.plt")
    def test_plot_action_target_distribution_invalid_json(self, mock_plt):
        """plot_action_target_distribution handles invalid JSON in details gracefully."""
        df = pd.DataFrame(
            {
                "action_type": ["eat", "move"],
                "reward": [0.5, 0.1],
                "step_number": [1, 2],
                "resources_before": [10.0, 9.0],
                "resources_after": [11.0, 9.0],
                "agent_id": ["a1", "a1"],
                "details": ["not_valid_json", '{"no_target_key": true}'],
            }
        )
        result = chart_actions.plot_action_target_distribution(df)
        # Should not raise even with invalid JSON
        self.assertIsNotNone(result)

    def test_functions_exist(self):
        """Verify all public chart functions are present."""
        for name in [
            "plot_action_type_distribution",
            "plot_rewards_by_action_type",
            "plot_resource_changes",
            "plot_action_frequency_over_time",
            "plot_position_changes",
            "plot_rewards_over_time",
            "plot_action_target_distribution",
        ]:
            self.assertTrue(hasattr(chart_actions, name), f"Missing function: {name}")


if __name__ == "__main__":
    unittest.main()
