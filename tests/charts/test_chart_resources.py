"""Tests for chart_resources module with mocked matplotlib."""

import unittest
from unittest.mock import patch

import pandas as pd

from farm.charts import chart_resources


def _make_resources_df():
    """Build a minimal resource_states DataFrame with all required columns."""
    return pd.DataFrame(
        {
            "step_number": [1, 1, 2, 2, 3, 3],
            "resource_id": [1, 2, 1, 2, 1, 2],
            "amount": [10.0, 8.0, 9.5, 7.5, 9.0, 8.5],
            "position_x": [5.0, 15.0, 5.0, 15.0, 5.0, 15.0],
            "position_y": [5.0, 10.0, 5.0, 10.0, 5.0, 10.0],
        }
    )


class TestChartResources(unittest.TestCase):
    """Tests for chart_resources plotting functions."""

    @patch("farm.charts.chart_resources.plt")
    def test_plotting_functions_exist(self, mock_plt):
        """Test that plotting functions exist and can be called."""
        self.assertTrue(hasattr(chart_resources, "plot_resource_amount_distribution"))
        self.assertTrue(hasattr(chart_resources, "plot_resource_positions"))
        self.assertTrue(hasattr(chart_resources, "plot_total_resources_over_time"))

    @patch("farm.charts.chart_resources.plt")
    def test_plot_resource_amount_distribution(self, mock_plt):
        """plot_resource_amount_distribution creates a histogram."""
        df = _make_resources_df()
        chart_resources.plot_resource_amount_distribution(df)
        mock_plt.figure.assert_called_once()
        mock_plt.hist.assert_called_once()

    @patch("farm.charts.chart_resources.plt")
    def test_plot_resource_positions(self, mock_plt):
        """plot_resource_positions creates a scatter plot."""
        df = _make_resources_df()
        chart_resources.plot_resource_positions(df)
        mock_plt.figure.assert_called_once()
        mock_plt.scatter.assert_called_once()

    @patch("farm.charts.chart_resources.plt")
    def test_plot_total_resources_over_time(self, mock_plt):
        """plot_total_resources_over_time creates a line plot."""
        df = _make_resources_df()
        chart_resources.plot_total_resources_over_time(df)
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()

    @patch("farm.charts.chart_resources.plt")
    def test_plot_resource_distribution_by_step(self, mock_plt):
        """plot_resource_distribution_by_step creates a scatter plot for the given step."""
        df = _make_resources_df()
        chart_resources.plot_resource_distribution_by_step(df, step_number=1)
        mock_plt.figure.assert_called_once()
        mock_plt.scatter.assert_called_once()

    @patch("farm.charts.chart_resources.plt")
    def test_plot_resource_amount_changes(self, mock_plt):
        """plot_resource_amount_changes creates a line plot for the given resource."""
        df = _make_resources_df()
        chart_resources.plot_resource_amount_changes(df, resource_id=1)
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()

    @patch("farm.charts.chart_resources.plt")
    def test_plot_average_resource_density(self, mock_plt):
        """plot_average_resource_density creates an imshow plot."""
        df = _make_resources_df()
        chart_resources.plot_average_resource_density(df, grid_size=10)
        mock_plt.figure.assert_called_once()
        mock_plt.imshow.assert_called_once()


if __name__ == "__main__":
    unittest.main()

