"""Tests for chart_utils module."""

import unittest

from farm.charts import chart_utils


class TestChartUtils(unittest.TestCase):
    """Tests for chart_utils functions."""

    def test_utils_exist(self):
        """Test that utility functions exist."""
        # Verify module can be imported and has expected functions
        self.assertIsNotNone(chart_utils)
        self.assertTrue(hasattr(chart_utils, "save_plot"))


if __name__ == "__main__":
    unittest.main()

