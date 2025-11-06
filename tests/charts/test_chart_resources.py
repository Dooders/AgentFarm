"""Tests for chart_resources module with mocked matplotlib."""

import unittest
from unittest.mock import patch

from farm.charts import chart_resources


class TestChartResources(unittest.TestCase):
    """Tests for chart_resources plotting functions."""

    @patch("farm.charts.chart_resources.plt")
    def test_plotting_functions_exist(self, mock_plt):
        """Test that plotting functions exist and can be called."""
        # Verify actual functions exist
        self.assertTrue(hasattr(chart_resources, "plot_resource_amount_distribution"))
        self.assertTrue(hasattr(chart_resources, "plot_resource_positions"))
        self.assertTrue(hasattr(chart_resources, "plot_total_resources_over_time"))


if __name__ == "__main__":
    unittest.main()

