"""Tests for chart_experience module with mocked matplotlib."""

import unittest
from unittest.mock import patch

import pandas as pd

from farm.charts import chart_experience


def _make_experience_df():
    """Build a minimal experience DataFrame with all required columns."""
    return pd.DataFrame(
        {
            "step_number": [1, 2, 3, 4, 5],
            "agent_id": ["a1", "a1", "a1", "a2", "a2"],
            "action_taken": [0, 1, 0, 2, 1],
            "reward": [0.5, 0.7, 0.3, 0.4, 0.6],
            "loss": [0.5, 0.4, 0.35, 0.45, 0.3],
            "module_type": ["A", "B", "A", "C", "B"],
        }
    )


class TestChartExperience(unittest.TestCase):
    """Tests for chart_experience plotting functions."""

    @patch("farm.charts.chart_experience.plt")
    def test_plot_action_rewards(self, mock_plt):
        """Test plot_action_rewards function."""
        df = pd.DataFrame(
            {"action_taken": [0, 1, 2], "reward": [0.5, 0.7, 0.3]}
        )

        chart_experience.plot_action_rewards(df)

        mock_plt.figure.assert_called_once()
        mock_plt.scatter.assert_called_once()

    @patch("farm.charts.chart_experience.plt")
    def test_plot_module_type_distribution(self, mock_plt):
        """Test plot_module_type_distribution function."""
        df = pd.DataFrame({"module_type": ["A", "B", "A", "C"]})

        chart_experience.plot_module_type_distribution(df)

        mock_plt.figure.assert_called_once()

    @patch("farm.charts.chart_experience.plt")
    def test_plot_loss_over_time(self, mock_plt):
        """Test plot_loss_over_time function."""
        df = pd.DataFrame(
            {"step_number": [1, 2, 3], "loss": [0.5, 0.4, 0.3]}
        )

        chart_experience.plot_loss_over_time(df)

        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()

    @patch("farm.charts.chart_experience.plt")
    def test_plot_rewards_over_time(self, mock_plt):
        """Test plot_rewards_over_time creates a cumulative rewards line plot."""
        df = _make_experience_df()
        chart_experience.plot_rewards_over_time(df)
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()

    @patch("farm.charts.chart_experience.plt")
    def test_plot_loss_vs_rewards(self, mock_plt):
        """Test plot_loss_vs_rewards creates a scatter plot."""
        df = _make_experience_df()
        chart_experience.plot_loss_vs_rewards(df)
        mock_plt.figure.assert_called_once()
        mock_plt.scatter.assert_called_once()

    def test_analyze_experience_details_runs_without_error(self):
        """analyze_experience_details prints without raising."""
        df = pd.DataFrame(
            {
                "experience_id": [1, 2],
                "module_type": ["A", "B"],
                "action_taken": [0, 1],
                "reward": [0.5, 0.7],
            }
        )
        # Should not raise
        chart_experience.analyze_experience_details(df)

    @patch("farm.charts.chart_experience.plt")
    def test_plot_state_transitions(self, mock_plt):
        """plot_state_transitions creates a figure for the given agent."""
        df = pd.DataFrame(
            {
                "step_number": [1, 2, 3],
                "agent_id": [1, 1, 1],
                "state_before": ["[1, 0]", "[2, 1]", "[3, 0]"],
                "state_after": ["[2, 1]", "[3, 0]", "[4, 1]"],
            }
        )
        chart_experience.plot_state_transitions(df, agent_id=1)
        mock_plt.figure.assert_called_once()

    def test_functions_exist(self):
        """Verify all public chart functions are present."""
        for name in [
            "plot_action_rewards",
            "plot_module_type_distribution",
            "plot_loss_over_time",
            "plot_rewards_over_time",
            "plot_state_transitions",
            "plot_loss_vs_rewards",
            "analyze_experience_details",
        ]:
            self.assertTrue(
                hasattr(chart_experience, name), f"Missing function: {name}"
            )


if __name__ == "__main__":
    unittest.main()

