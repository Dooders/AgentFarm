"""Tests for chart_experience module with mocked matplotlib."""

import unittest
from unittest.mock import patch

import pandas as pd

from farm.charts import chart_experience


class TestChartExperience(unittest.TestCase):
    """Tests for chart_experience plotting functions."""

    @patch("farm.charts.chart_experience.plt")
    def test_plot_action_rewards(self, mock_plt):
        """Test plot_action_rewards function."""
        df = pd.DataFrame(
            {"action_taken": [0, 1, 2], "reward": [0.5, 0.7, 0.3]}
        )

        chart_experience.plot_action_rewards(df)

        # Verify matplotlib was called
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


if __name__ == "__main__":
    unittest.main()

