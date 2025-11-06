"""Tests for chart_states module with mocked matplotlib."""

import unittest
from unittest.mock import patch

from farm.charts import chart_states


class TestChartStates(unittest.TestCase):
    """Tests for chart_states plotting functions."""

    @patch("farm.charts.chart_states.plt")
    def test_plotting_functions_exist(self, mock_plt):
        """Test that plotting functions exist."""
        # Verify actual functions exist
        self.assertTrue(hasattr(chart_states, "plot_total_reward_distribution"))
        self.assertTrue(hasattr(chart_states, "plot_average_health_vs_age"))
        self.assertTrue(hasattr(chart_states, "plot_average_resource_vs_age"))


if __name__ == "__main__":
    unittest.main()

