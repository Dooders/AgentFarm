"""Tests for chart_simulation module."""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from farm.charts import chart_simulation


def _make_simulation_df():
    """Build a minimal simulation_steps DataFrame with all required columns."""
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


class TestChartSimulation(unittest.TestCase):
    """Tests for chart_simulation plotting functions."""

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_population_dynamics(self, mock_plt):
        """plot_population_dynamics creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_population_dynamics(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_births_and_deaths(self, mock_plt):
        """plot_births_and_deaths creates a figure and returns plt."""
        # Need step_number >= 20 for the filter, use larger range
        df = pd.DataFrame(
            {
                "step_number": list(range(1, 31)),
                "births": [2] * 30,
                "deaths": [1] * 30,
            }
        )
        result = chart_simulation.plot_births_and_deaths(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_resource_efficiency(self, mock_plt):
        """plot_resource_efficiency creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_resource_efficiency(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_agent_health_and_age(self, mock_plt):
        """plot_agent_health_and_age creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_agent_health_and_age(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_combat_metrics(self, mock_plt):
        """plot_combat_metrics creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_combat_metrics(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_resource_sharing(self, mock_plt):
        """plot_resource_sharing creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_resource_sharing(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_evolutionary_metrics(self, mock_plt):
        """plot_evolutionary_metrics creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_evolutionary_metrics(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_resource_distribution_entropy(self, mock_plt):
        """plot_resource_distribution_entropy creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_resource_distribution_entropy(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_rewards(self, mock_plt):
        """plot_rewards creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_rewards(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.plt")
    def test_plot_average_resources(self, mock_plt):
        """plot_average_resources creates a figure and returns plt."""
        df = _make_simulation_df()
        result = chart_simulation.plot_average_resources(df)
        mock_plt.figure.assert_called_once()
        self.assertIs(result, mock_plt)

    @patch("farm.charts.chart_simulation.create_engine")
    @patch("farm.charts.chart_simulation.plt")
    def test_plot_agent_lifespan_histogram(self, mock_plt, mock_engine):
        """plot_agent_lifespan_histogram creates a histogram when DB is available."""
        df = _make_simulation_df()
        # Mock DB returning agent lifespan data
        agents_df = pd.DataFrame(
            {"birth_time": [0, 5, 10], "death_time": [20, 15, 25]}
        )
        mock_conn = MagicMock()
        mock_engine.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.return_value.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        with patch("pandas.read_sql", return_value=agents_df):
            result = chart_simulation.plot_agent_lifespan_histogram(df)
        mock_plt.figure.assert_called_once()
        self.assertIsNotNone(result)

    @patch("farm.charts.chart_simulation.create_engine")
    @patch("farm.charts.chart_simulation.plt")
    def test_plot_agent_lifespan_histogram_error(self, mock_plt, mock_engine):
        """plot_agent_lifespan_histogram returns None when DB fails."""
        df = _make_simulation_df()
        mock_engine.side_effect = Exception("DB error")
        result = chart_simulation.plot_agent_lifespan_histogram(df)
        self.assertIsNone(result)

    @patch("farm.charts.chart_simulation.create_engine")
    @patch("farm.charts.chart_simulation.plt")
    def test_plot_births_and_deaths_by_type_with_data(self, mock_plt, mock_engine):
        """plot_births_and_deaths_by_type creates a figure when events data is available."""
        df = _make_simulation_df()
        events_df = pd.DataFrame(
            {
                "step_number": [25, 25, 30, 30],
                "agent_type": ["SystemAgent", "IndependentAgent", "SystemAgent", "IndependentAgent"],
                "births": [2, 1, 1, 2],
                "deaths": [1, 0, 1, 1],
            }
        )
        with patch("pandas.read_sql", return_value=events_df):
            mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock()])
            result = chart_simulation.plot_births_and_deaths_by_type(df)
        # Should return plt (not None) when data is available
        self.assertIsNotNone(result)

    @patch("farm.charts.chart_simulation.create_engine")
    @patch("farm.charts.chart_simulation.plt")
    def test_plot_births_and_deaths_by_type_empty_data(self, mock_plt, mock_engine):
        """plot_births_and_deaths_by_type returns None when events data is empty."""
        df = _make_simulation_df()
        with patch("pandas.read_sql", return_value=pd.DataFrame()):
            result = chart_simulation.plot_births_and_deaths_by_type(df)
        self.assertIsNone(result)

    @patch("farm.charts.chart_simulation.create_engine")
    @patch("farm.charts.chart_simulation.plt")
    def test_plot_births_and_deaths_by_type_no_active_types(self, mock_plt, mock_engine):
        """plot_births_and_deaths_by_type returns None when no active agent types."""
        df = _make_simulation_df()
        events_df = pd.DataFrame(
            {
                "step_number": [25],
                "agent_type": ["UnknownType"],
                "births": [1],
                "deaths": [0],
            }
        )
        with patch("pandas.read_sql", return_value=events_df):
            result = chart_simulation.plot_births_and_deaths_by_type(df)
        self.assertIsNone(result)

    @patch("farm.charts.chart_simulation.create_engine")
    @patch("farm.charts.chart_simulation.plt")
    def test_plot_agent_type_comparison_with_data(self, mock_plt, mock_engine):
        """plot_agent_type_comparison creates figure when data is available."""
        df = _make_simulation_df()
        # Return a DataFrame with agent_type as a regular column (code will set_index it)
        agent_metrics_df = pd.DataFrame(
            {
                "agent_type": ["SystemAgent", "IndependentAgent"],
                "avg_resources": [5.0, 4.0],
                "avg_health": [0.8, 0.75],
                "avg_age": [15.0, 12.0],
                "avg_reward": [0.3, 0.2],
            }
        )
        with patch("pandas.read_sql", return_value=agent_metrics_df):
            mock_fig = MagicMock()
            mock_ax = MagicMock()  # Single polar axis returned by subplots
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            result = chart_simulation.plot_agent_type_comparison(df)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()

    def test_functions_exist(self):
        """Verify all public chart functions are present."""
        for name in [
            "plot_population_dynamics",
            "plot_births_and_deaths",
            "plot_births_and_deaths_by_type",
            "plot_resource_efficiency",
            "plot_agent_health_and_age",
            "plot_combat_metrics",
            "plot_resource_sharing",
            "plot_evolutionary_metrics",
            "plot_resource_distribution_entropy",
            "plot_rewards",
            "plot_average_resources",
            "plot_agent_lifespan_histogram",
            "plot_agent_type_comparison",
        ]:
            self.assertTrue(
                hasattr(chart_simulation, name), f"Missing function: {name}"
            )


if __name__ == "__main__":
    unittest.main()
