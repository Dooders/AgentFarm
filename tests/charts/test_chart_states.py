"""Tests for chart_states module with mocked matplotlib."""

import unittest
from unittest.mock import patch

import pandas as pd

from farm.charts import chart_states


def _make_states_df():
    """Build a minimal agent_states DataFrame with all required columns."""
    return pd.DataFrame(
        {
            "step_number": [1, 1, 2, 2, 3, 3],
            "agent_id": ["a1", "a2", "a1", "a2", "a1", "a2"],
            "age": [1, 2, 2, 3, 3, 4],
            "current_health": [0.9, 0.8, 0.85, 0.75, 0.88, 0.72],
            "resource_level": [5.0, 4.5, 5.2, 4.3, 5.1, 4.6],
            "total_reward": [10.0, 8.0, 12.0, 9.0, 15.0, 11.0],
        }
    )


class TestChartStates(unittest.TestCase):
    """Tests for chart_states plotting functions."""

    @patch("farm.charts.chart_states.plt")
    def test_plotting_functions_exist(self, mock_plt):
        """Test that plotting functions exist."""
        self.assertTrue(hasattr(chart_states, "plot_total_reward_distribution"))
        self.assertTrue(hasattr(chart_states, "plot_average_health_vs_age"))
        self.assertTrue(hasattr(chart_states, "plot_average_resource_vs_age"))

    @patch("farm.charts.chart_states.plt")
    def test_plot_total_reward_distribution(self, mock_plt):
        """plot_total_reward_distribution creates a histogram and returns plt."""
        df = _make_states_df()
        result = chart_states.plot_total_reward_distribution(df)
        mock_plt.figure.assert_called_once()
        mock_plt.hist.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_states.plt")
    def test_plot_average_health_vs_age(self, mock_plt):
        """plot_average_health_vs_age creates a line figure and returns plt."""
        df = _make_states_df()
        result = chart_states.plot_average_health_vs_age(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_states.plt")
    def test_plot_average_resource_vs_age(self, mock_plt):
        """plot_average_resource_vs_age creates a line figure and returns plt."""
        df = _make_states_df()
        result = chart_states.plot_average_resource_vs_age(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_states.plt")
    def test_plot_average_health_over_time(self, mock_plt):
        """plot_average_health_over_time creates a figure with std band and returns plt."""
        df = _make_states_df()
        result = chart_states.plot_average_health_over_time(df)
        mock_plt.figure.assert_called_once()
        mock_plt.fill_between.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_states.plt")
    def test_plot_average_resource_over_time(self, mock_plt):
        """plot_average_resource_over_time creates a figure with std band and returns plt."""
        df = _make_states_df()
        result = chart_states.plot_average_resource_over_time(df)
        mock_plt.figure.assert_called_once()
        mock_plt.fill_between.assert_called_once()
        self.assertIs(result, mock_plt)


if __name__ == "__main__":
    unittest.main()

