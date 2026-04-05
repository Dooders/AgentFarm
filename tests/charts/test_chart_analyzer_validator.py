"""Tests for chart_analyzer_validator module."""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from farm.utils import chart_analyzer_validator


def _make_sim_df():
    n = 5
    return pd.DataFrame(
        {
            "step_number": list(range(1, n + 1)),
            "total_agents": [10, 11, 12, 11, 10],
            "system_agents": [5, 5, 6, 6, 5],
            "independent_agents": [3, 4, 4, 3, 3],
            "control_agents": [2, 2, 2, 2, 2],
            "births": [2, 1, 2, 1, 0],
            "deaths": [1, 0, 1, 2, 1],
            "resource_efficiency": [0.7, 0.72, 0.68, 0.71, 0.73],
            "total_resources": [100.0, 98.0, 96.0, 97.0, 95.0],
            "average_agent_health": [0.8, 0.81, 0.79, 0.82, 0.80],
            "average_agent_age": [15.0, 16.0, 17.0, 15.0, 16.0],
            "combat_encounters": [3, 4, 5, 3, 2],
            "successful_attacks": [1, 2, 3, 1, 1],
            "resources_shared": [5.0, 6.0, 4.0, 5.0, 7.0],
            "genetic_diversity": [0.6, 0.62, 0.61, 0.63, 0.64],
            "dominant_genome_ratio": [30.0, 28.0, 29.0, 27.0, 26.0],
            "resource_distribution_entropy": [0.5, 0.52, 0.55, 0.53, 0.56],
            "average_reward": [0.3, 0.32, 0.28, 0.31, 0.35],
            "average_agent_resources": [5.0, 4.9, 5.1, 5.0, 4.8],
        }
    )


class TestChartAnalyzerValidator(unittest.TestCase):
    """Tests for chart_analyzer_validator utility functions."""

    def test_available_charts_is_list(self):
        """AVAILABLE_CHARTS should be a non-empty list."""
        self.assertIsInstance(chart_analyzer_validator.AVAILABLE_CHARTS, list)
        self.assertGreater(len(chart_analyzer_validator.AVAILABLE_CHARTS), 0)

    def test_available_charts_contains_simulation_charts(self):
        """AVAILABLE_CHARTS includes expected simulation chart names."""
        for name in [
            "population_dynamics",
            "births_and_deaths",
            "resource_efficiency",
            "combat_metrics",
            "rewards",
        ]:
            self.assertIn(name, chart_analyzer_validator.AVAILABLE_CHARTS)

    def test_available_charts_contains_action_charts(self):
        """AVAILABLE_CHARTS includes expected action chart names."""
        for name in [
            "action_type_distribution",
            "rewards_by_action_type",
            "resource_changes",
            "action_frequency_over_time",
            "rewards_over_time",
        ]:
            self.assertIn(name, chart_analyzer_validator.AVAILABLE_CHARTS)

    def test_available_charts_contains_agent_charts(self):
        """AVAILABLE_CHARTS includes expected agent chart names."""
        for name in [
            "lifespan_distribution",
            "lineage_size",
            "agent_types_over_time",
        ]:
            self.assertIn(name, chart_analyzer_validator.AVAILABLE_CHARTS)

    def test_list_available_charts_runs(self):
        """list_available_charts() prints without raising."""
        chart_analyzer_validator.list_available_charts()

    @patch("farm.utils.chart_analyzer_validator.ChartAnalyzer")
    @patch("farm.utils.chart_analyzer_validator.SimulationDatabase")
    @patch("farm.utils.chart_analyzer_validator.create_engine")
    @patch("pandas.read_sql")
    def test_validate_chart_unknown_name_prints_error(
        self, mock_read_sql, mock_engine, mock_db, mock_analyzer
    ):
        """validate_chart prints an error message for unknown chart names."""
        sim_df = _make_sim_df()
        mock_read_sql.return_value = sim_df
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance._analyze_simulation_chart.return_value = "Test analysis"
        mock_analyzer.return_value = mock_analyzer_instance

        # Should not raise for an unknown chart name
        chart_analyzer_validator.validate_chart("unknown_chart_xyz", show_plot=False)

    @patch("farm.utils.chart_analyzer_validator.ChartAnalyzer")
    @patch("farm.utils.chart_analyzer_validator.SimulationDatabase")
    @patch("farm.utils.chart_analyzer_validator.create_engine")
    @patch("pandas.read_sql")
    def test_validate_chart_known_chart_no_plot(
        self, mock_read_sql, mock_engine, mock_db, mock_analyzer
    ):
        """validate_chart runs without error for a known chart with show_plot=False."""
        sim_df = _make_sim_df()
        mock_read_sql.return_value = sim_df
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance._analyze_simulation_chart.return_value = "Test analysis"
        mock_analyzer.return_value = mock_analyzer_instance

        # Should not raise
        chart_analyzer_validator.validate_chart("population_dynamics", show_plot=False)
        mock_analyzer_instance._analyze_simulation_chart.assert_called_once_with(
            "population_dynamics", sim_df
        )

    @patch("farm.utils.chart_analyzer_validator.ChartAnalyzer")
    @patch("farm.utils.chart_analyzer_validator.SimulationDatabase")
    @patch("farm.utils.chart_analyzer_validator.create_engine")
    @patch("pandas.read_sql")
    @patch("farm.utils.chart_analyzer_validator.plot_population_dynamics")
    def test_validate_chart_with_plot(
        self,
        mock_plot_fn,
        mock_read_sql,
        mock_engine,
        mock_db,
        mock_analyzer,
    ):
        """validate_chart calls the plot function when show_plot=True."""
        sim_df = _make_sim_df()
        mock_read_sql.return_value = sim_df
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance._analyze_simulation_chart.return_value = "Test analysis"
        mock_analyzer.return_value = mock_analyzer_instance
        mock_plt = MagicMock()
        mock_plot_fn.return_value = mock_plt

        chart_analyzer_validator.validate_chart("population_dynamics", show_plot=True)
        # The plot function should have been called via the lambda
        mock_plot_fn.assert_called()

    @patch("farm.utils.chart_analyzer_validator.ChartAnalyzer")
    @patch("farm.utils.chart_analyzer_validator.SimulationDatabase")
    @patch("farm.utils.chart_analyzer_validator.create_engine")
    @patch("pandas.read_sql")
    @patch("farm.utils.chart_analyzer_validator.plot_population_dynamics")
    def test_validate_chart_plot_returns_none(
        self,
        mock_plot_fn,
        mock_read_sql,
        mock_engine,
        mock_db,
        mock_analyzer,
    ):
        """validate_chart handles plot function returning None gracefully."""
        sim_df = _make_sim_df()
        mock_read_sql.return_value = sim_df
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance._analyze_simulation_chart.return_value = "Test analysis"
        mock_analyzer.return_value = mock_analyzer_instance
        mock_plot_fn.return_value = None  # Plot returns None

        # Should not raise
        chart_analyzer_validator.validate_chart("population_dynamics", show_plot=True)

    @patch("farm.utils.chart_analyzer_validator.ChartAnalyzer")
    @patch("farm.utils.chart_analyzer_validator.SimulationDatabase")
    @patch("farm.utils.chart_analyzer_validator.create_engine")
    @patch("pandas.read_sql")
    @patch("farm.utils.chart_analyzer_validator.plot_population_dynamics")
    def test_validate_chart_plot_exception_handled(
        self,
        mock_plot_fn,
        mock_read_sql,
        mock_engine,
        mock_db,
        mock_analyzer,
    ):
        """validate_chart handles exceptions from plot function gracefully."""
        sim_df = _make_sim_df()
        mock_read_sql.return_value = sim_df
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance._analyze_simulation_chart.return_value = "Test analysis"
        mock_analyzer.return_value = mock_analyzer_instance
        mock_plot_fn.side_effect = RuntimeError("Plot failed")

        # Should not raise - exception is caught internally
        chart_analyzer_validator.validate_chart("population_dynamics", show_plot=True)

    def test_functions_exist(self):
        """Key functions are present in the module."""
        for name in [
            "validate_chart",
            "list_available_charts",
            "validate_all_charts",
            "main",
        ]:
            self.assertTrue(
                hasattr(chart_analyzer_validator, name),
                f"Missing function: {name}",
            )


if __name__ == "__main__":
    unittest.main()
