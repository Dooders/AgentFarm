"""Tests for chart_analyzer module – analysis methods and _analyze_simulation_chart dispatch."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from farm.charts.chart_analyzer import ChartAnalyzer


def _make_db_mock():
    """Return a minimal DatabaseProtocol mock."""
    db = MagicMock()
    db.engine = MagicMock()
    return db


def _make_simulation_df():
    """Minimal simulation_steps DataFrame."""
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


def _make_actions_df():
    """Minimal agent_actions DataFrame."""
    import json

    return pd.DataFrame(
        {
            "action_type": ["move", "eat", "move", "attack", "eat"],
            "reward": [0.1, 0.5, 0.2, -0.1, 0.4],
            "step_number": [1, 1, 2, 2, 3],
            "resources_before": [10.0, 10.0, 9.5, 9.0, 9.5],
            "resources_after": [9.5, 11.0, 9.0, 8.0, 11.0],
            "agent_id": ["a1", "a1", "a1", "a2", "a2"],
            "details": [
                json.dumps({"target_id": "t1"}),
                json.dumps({"target_position": [3, 4]}),
                json.dumps({"target_id": "t2"}),
                json.dumps({"target_id": "t1"}),
                None,
            ],
        }
    )


def _make_agents_df():
    """Minimal agents DataFrame."""
    return pd.DataFrame(
        {
            "agent_id": ["a1", "a2", "a3"],
            "agent_type": ["SystemAgent", "IndependentAgent", "SystemAgent"],
            "birth_time": [0, 5, 10],
            "death_time": [20, 15, 30],
            "genome_id": ["::", "a1::1", "a2:a1:1"],
        }
    )


@patch("farm.charts.chart_analyzer.LLMClient")
class TestChartAnalyzerInit(unittest.TestCase):
    """Tests for ChartAnalyzer initialisation."""

    def test_init_default(self, mock_llm):
        """ChartAnalyzer can be constructed with a mock DB."""
        db = _make_db_mock()
        analyzer = ChartAnalyzer(db)
        self.assertEqual(analyzer.db, db)
        self.assertTrue(analyzer.save_charts)

    def test_init_save_charts_false(self, mock_llm):
        """ChartAnalyzer respects save_charts=False."""
        db = _make_db_mock()
        analyzer = ChartAnalyzer(db, save_charts=False)
        self.assertFalse(analyzer.save_charts)


@patch("farm.charts.chart_analyzer.LLMClient")
class TestAnalyzeSimulationChart(unittest.TestCase):
    """Tests for _analyze_simulation_chart dispatch."""

    def setUp(self):
        with patch("farm.charts.chart_analyzer.LLMClient"):
            self.analyzer = ChartAnalyzer(_make_db_mock(), save_charts=False)
        self.sim_df = _make_simulation_df()
        self.actions_df = _make_actions_df()
        self.agents_df = _make_agents_df()

    def test_dispatch_population_dynamics(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart("population_dynamics", self.sim_df)
        self.assertIn("Population Dynamics Analysis", result)

    def test_dispatch_births_and_deaths(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart("births_and_deaths", self.sim_df)
        self.assertIn("Population Change Analysis", result)

    def test_dispatch_births_and_deaths_by_type(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "births_and_deaths_by_type", self.sim_df
        )
        self.assertIn("Population Changes by Type Analysis", result)

    def test_dispatch_resource_efficiency(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "resource_efficiency", self.sim_df
        )
        self.assertIn("Resource Efficiency Analysis", result)

    def test_dispatch_agent_health_and_age(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "agent_health_and_age", self.sim_df
        )
        self.assertIn("Health and Age Analysis", result)

    def test_dispatch_combat_metrics(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart("combat_metrics", self.sim_df)
        self.assertIn("Combat Analysis", result)

    def test_dispatch_resource_sharing(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart("resource_sharing", self.sim_df)
        self.assertIn("Resource Sharing Analysis", result)

    def test_dispatch_evolutionary_metrics(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "evolutionary_metrics", self.sim_df
        )
        self.assertIn("Evolutionary Analysis", result)

    def test_dispatch_resource_distribution_entropy(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "resource_distribution_entropy", self.sim_df
        )
        self.assertIn("Resource Distribution Analysis", result)

    def test_dispatch_rewards(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart("rewards", self.sim_df)
        self.assertIn("Reward Analysis", result)

    def test_dispatch_average_resources(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "average_resources", self.sim_df
        )
        self.assertIn("Average Resources Analysis", result)

    def test_dispatch_agent_lifespan_histogram(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "agent_lifespan_histogram", self.sim_df
        )
        self.assertIn("Lifespan Distribution Analysis", result)

    def test_dispatch_agent_type_comparison(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "agent_type_comparison", self.sim_df
        )
        self.assertIn("Agent Type Comparison", result)

    def test_dispatch_action_type_distribution(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "action_type_distribution", self.actions_df
        )
        self.assertIn("Action Type Analysis", result)

    def test_dispatch_rewards_by_action_type(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "rewards_by_action_type", self.actions_df
        )
        self.assertIn("Action Rewards Analysis", result)

    def test_dispatch_resource_changes(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "resource_changes", self.actions_df
        )
        self.assertIn("Resource Change Analysis", result)

    def test_dispatch_action_frequency_over_time(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "action_frequency_over_time", self.actions_df
        )
        self.assertIn("Action Frequency Analysis", result)

    def test_dispatch_rewards_over_time(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "rewards_over_time", self.actions_df
        )
        self.assertIn("Rewards Over Time Analysis", result)

    def test_dispatch_action_target_distribution(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "action_target_distribution", self.actions_df
        )
        self.assertIn("Action Target Analysis", result)

    def test_dispatch_lifespan_distribution(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "lifespan_distribution", self.agents_df
        )
        self.assertIn("Lifespan Distribution Analysis", result)

    def test_dispatch_lineage_size(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "lineage_size", self.agents_df
        )
        self.assertIn("Lineage", result)

    def test_dispatch_agent_types_over_time(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "agent_types_over_time", self.agents_df
        )
        self.assertIn("Analysis", result)

    def test_dispatch_unknown_chart(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "nonexistent_chart", self.sim_df
        )
        self.assertIn("Analysis not implemented", result)

    def test_dispatch_reproduction_success_rate(self, mock_llm):
        """_analyze_reproduction_success_rate does not crash even with no real DB."""
        result = self.analyzer._analyze_simulation_chart(
            "reproduction_success_rate", self.sim_df
        )
        self.assertIsInstance(result, str)

    def test_dispatch_reproduction_resources(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "reproduction_resources", self.sim_df
        )
        self.assertIn("Reproduction Resource Analysis", result)

    def test_dispatch_reproduction_failure_reasons(self, mock_llm):
        result = self.analyzer._analyze_simulation_chart(
            "reproduction_failure_reasons", self.sim_df
        )
        self.assertIn("Reproduction Failure Analysis", result)


@patch("farm.charts.chart_analyzer.LLMClient")
class TestChartAnalyzerAnalyzeAllCharts(unittest.TestCase):
    """Tests for ChartAnalyzer.analyze_all_charts with mocked database."""

    def _make_full_db_mock(self):
        """Build a mock DB with simulation_steps, agent_actions, and agents tables."""
        db = MagicMock()
        engine = MagicMock()
        db.engine = engine
        db.engine.url = "sqlite:///:memory:"

        sim_df = _make_simulation_df()
        actions_df = _make_actions_df()
        agents_df = _make_agents_df()

        def read_sql_side_effect(query, engine, *args, **kwargs):
            q = query.lower()
            if "simulation_steps" in q:
                return sim_df
            elif "agent_actions" in q:
                return actions_df
            elif "agents" in q:
                return agents_df
            return pd.DataFrame()

        return db, read_sql_side_effect

    @patch("farm.charts.chart_analyzer.plt")
    @patch("pandas.read_sql")
    def test_analyze_all_charts_returns_dict(self, mock_read_sql, mock_plt, mock_llm):
        """analyze_all_charts returns a dict of chart analyses."""
        db, side_effect = self._make_full_db_mock()
        mock_read_sql.side_effect = side_effect

        analyzer = ChartAnalyzer(db, save_charts=False)
        result = analyzer.analyze_all_charts()

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    @patch("farm.charts.chart_analyzer.save_plot")
    @patch("farm.charts.chart_analyzer.plt")
    @patch("pandas.read_sql")
    def test_analyze_all_charts_save_charts_true(
        self, mock_read_sql, mock_plt, mock_save_plot, mock_llm
    ):
        """analyze_all_charts saves charts and analyses when save_charts=True."""
        import tempfile
        from pathlib import Path

        db, side_effect = self._make_full_db_mock()
        mock_read_sql.side_effect = side_effect
        mock_save_plot.return_value = "/tmp/test_chart.png"

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ChartAnalyzer(db, output_dir=Path(tmpdir), save_charts=True)
            result = analyzer.analyze_all_charts()

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    @patch("farm.charts.chart_analyzer.plt")
    @patch("pandas.read_sql")
    def test_analyze_all_charts_with_output_path(self, mock_read_sql, mock_plt, mock_llm):
        """analyze_all_charts accepts an output_path parameter."""
        import tempfile
        from pathlib import Path

        db, side_effect = self._make_full_db_mock()
        mock_read_sql.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ChartAnalyzer(db, save_charts=False)
            result = analyzer.analyze_all_charts(output_path=Path(tmpdir))

        self.assertIsInstance(result, dict)

    @patch("farm.charts.chart_analyzer.plt")
    @patch("pandas.read_sql")
    def test_analyze_all_charts_handles_db_error(self, mock_read_sql, mock_plt, mock_llm):
        """analyze_all_charts returns empty dict when DB read fails."""
        db = _make_db_mock()
        mock_read_sql.side_effect = Exception("DB connection failed")

        analyzer = ChartAnalyzer(db, save_charts=False)
        result = analyzer.analyze_all_charts()

        self.assertIsInstance(result, dict)

    @patch("farm.charts.chart_analyzer.plt")
    @patch("pandas.read_sql")
    def test_analyze_all_charts_action_chart_exception(
        self, mock_read_sql, mock_plt, mock_llm
    ):
        """analyze_all_charts catches exceptions from individual action chart functions."""
        db, side_effect = self._make_full_db_mock()
        mock_read_sql.side_effect = side_effect

        with patch("farm.charts.chart_analyzer.plot_action_type_distribution") as mock_fn:
            mock_fn.side_effect = RuntimeError("Action chart failed")
            analyzer = ChartAnalyzer(db, save_charts=False)
            result = analyzer.analyze_all_charts()

        self.assertIsInstance(result, dict)

    @patch("farm.charts.chart_analyzer.plt")
    @patch("pandas.read_sql")
    def test_analyze_all_charts_agent_chart_exception(
        self, mock_read_sql, mock_plt, mock_llm
    ):
        """analyze_all_charts catches exceptions from individual agent chart functions."""
        db, side_effect = self._make_full_db_mock()
        mock_read_sql.side_effect = side_effect

        with patch("farm.charts.chart_analyzer.plot_lifespan_distribution") as mock_fn:
            mock_fn.side_effect = RuntimeError("Agent chart failed")
            analyzer = ChartAnalyzer(db, save_charts=False)
            result = analyzer.analyze_all_charts()

        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
