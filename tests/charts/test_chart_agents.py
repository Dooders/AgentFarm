"""Tests for chart_agents module."""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from farm.charts import chart_agents


def _make_agents_df():
    """Build a minimal agents DataFrame with all required columns."""
    return pd.DataFrame(
        {
            "birth_time": [0, 0, 5, 10, 15],
            "death_time": [20, 15, 25, 30, 35],
            "genome_id": ["::", "::", "a1::1", "a1::2", "a2:a1:1"],
            "agent_type": [
                "SystemAgent",
                "SystemAgent",
                "IndependentAgent",
                "SystemAgent",
                "IndependentAgent",
            ],
        }
    )


class TestChartAgents(unittest.TestCase):
    """Tests for chart_agents plotting functions."""

    @patch("farm.charts.chart_agents.plt")
    def test_plot_lifespan_distribution(self, mock_plt):
        """plot_lifespan_distribution creates a histogram and returns plt."""
        df = _make_agents_df()
        result = chart_agents.plot_lifespan_distribution(df)
        mock_plt.figure.assert_called_once()
        mock_plt.hist.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_agents.plt")
    def test_plot_lineage_size(self, mock_plt):
        """plot_lineage_size creates a histogram and returns plt."""
        df = _make_agents_df()
        result = chart_agents.plot_lineage_size(df)
        mock_plt.figure.assert_called_once()
        mock_plt.hist.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_agents.plt")
    def test_plot_lineage_size_nan_genome_id(self, mock_plt):
        """plot_lineage_size handles NaN genome_id values."""
        df = pd.DataFrame(
            {
                "birth_time": [0, 5],
                "death_time": [10, 15],
                "genome_id": [None, ""],
                "agent_type": ["SystemAgent", "IndependentAgent"],
            }
        )
        result = chart_agents.plot_lineage_size(df)
        mock_plt.figure.assert_called_once()
        self.assertIsNotNone(result)

    @patch("farm.charts.chart_agents.plt")
    def test_plot_agent_types_over_time(self, mock_plt):
        """plot_agent_types_over_time creates a line figure and returns plt."""
        df = _make_agents_df()
        result = chart_agents.plot_agent_types_over_time(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    def test_get_base_genome_id_variants(self):
        """Test genome_id parsing logic via plot_lineage_size output."""
        # Verify lineage grouping logic via DataFrame transformation
        df = pd.DataFrame(
            {
                "birth_time": [0] * 9,
                "death_time": [10] * 9,
                "genome_id": [
                    "::",  # initial agent (3 parts)
                    "::2",  # initial with counter
                    "agentA:",  # single parent no counter
                    "agentA:1",  # single parent with counter
                    "agentA:agentB",  # two parents no counter
                    "agentA:agentB:3",  # two parents with counter
                    "",  # empty
                    "legacy",  # legacy single part
                    ":",  # single colon → maps to "::"
                ],
                "agent_type": ["SystemAgent"] * 9,
            }
        )
        with patch("farm.charts.chart_agents.plt"):
            result = chart_agents.plot_lineage_size(df)
        self.assertIsNotNone(result)
        # Verify base_genome_id column is created
        self.assertIn("base_genome_id", df.columns)

    @patch("farm.charts.chart_agents.plt")
    def test_plot_reproduction_success_rate(self, mock_plt):
        """plot_reproduction_success_rate creates a figure and returns plt."""
        df = pd.DataFrame(
            {
                "step_number": [1, 2, 3, 4, 5],
                "births": [2, 1, 2, 1, 0],
                "deaths": [1, 0, 1, 2, 1],
            }
        )
        result = chart_agents.plot_reproduction_success_rate(df)
        mock_plt.figure.assert_called_once()
        self.assertIsNotNone(result)

    @patch("farm.charts.chart_agents.plt")
    def test_plot_reproduction_success_rate_exception(self, mock_plt):
        """plot_reproduction_success_rate returns None when data is invalid."""
        df = pd.DataFrame({"wrong_col": [1, 2]})
        result = chart_agents.plot_reproduction_success_rate(df)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
