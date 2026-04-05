"""Tests for chart_utils module."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from farm.charts import chart_utils


class TestChartUtils(unittest.TestCase):
    """Tests for chart_utils functions."""

    def test_utils_exist(self):
        """Test that utility functions exist."""
        # Verify module can be imported and has expected functions
        self.assertIsNotNone(chart_utils)
        self.assertTrue(hasattr(chart_utils, "save_plot"))

    def test_save_plot_writes_under_charts_subdir(self):
        mock_plt = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = chart_utils.save_plot(mock_plt, "mychart", output_dir=tmpdir)
            expected = os.path.join(tmpdir, "charts", "mychart.png")
            self.assertEqual(path, expected)
            mock_plt.savefig.assert_called_once_with(expected)
            mock_plt.close.assert_called_once()

    def test_save_plot_returns_figure_when_no_output_dir(self):
        mock_plt = MagicMock()
        fig = MagicMock()
        mock_plt.gcf.return_value = fig
        result = chart_utils.save_plot(mock_plt, "mychart", output_dir="")
        self.assertIs(result, fig)
        mock_plt.savefig.assert_not_called()
        mock_plt.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()

