"""Tests for farm/core/analysis.py – SimulationAnalyzer and analyze_simulation."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from farm.core.analysis import SimulationAnalyzer, analyze_simulation


class TestAnalyzeSimulation:
    """Tests for the standalone analyze_simulation helper."""

    def test_returns_dict(self):
        result = analyze_simulation({"step": 1, "agents": []})
        assert isinstance(result, dict)

    def test_has_metrics_and_statistics_keys(self):
        result = analyze_simulation({})
        assert "metrics" in result
        assert "statistics" in result

    def test_metrics_is_dict(self):
        result = analyze_simulation({"anything": True})
        assert isinstance(result["metrics"], dict)

    def test_statistics_is_dict(self):
        result = analyze_simulation(None)
        assert isinstance(result["statistics"], dict)


class TestSimulationAnalyzerInit:
    """Test SimulationAnalyzer instantiation with mocked database."""

    @patch("farm.core.analysis.SimulationDatabase")
    def test_init_creates_database(self, mock_db_cls):
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        analyzer = SimulationAnalyzer(db_path=":memory:", simulation_id="test_sim")

        mock_db_cls.assert_called_once_with(":memory:", simulation_id="test_sim")
        assert analyzer.db is mock_db

    @patch("farm.core.analysis.SimulationDatabase")
    def test_init_default_params(self, mock_db_cls):
        mock_db_cls.return_value = MagicMock()
        analyzer = SimulationAnalyzer()
        mock_db_cls.assert_called_once_with("simulation.db", simulation_id=None)


class TestSimulationAnalyzerMethods:
    """Test analyzer query methods with mocked DB transaction helper."""

    def _make_analyzer(self):
        with patch("farm.core.analysis.SimulationDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            analyzer = SimulationAnalyzer(db_path=":memory:")
        return analyzer

    def test_calculate_survival_rates_calls_transaction(self):
        analyzer = self._make_analyzer()
        expected_df = pd.DataFrame({"step": [1], "system_alive": [5], "independent_alive": [3]})
        analyzer.db._execute_in_transaction.return_value = expected_df

        result = analyzer.calculate_survival_rates()

        analyzer.db._execute_in_transaction.assert_called_once()
        assert result is expected_df

    def test_analyze_resource_distribution_calls_transaction(self):
        analyzer = self._make_analyzer()
        expected_df = pd.DataFrame({"step": [1], "agent_type": ["system"],
                                    "avg_resources": [5.0], "min_resources": [1.0],
                                    "max_resources": [10.0], "agent_count": [3]})
        analyzer.db._execute_in_transaction.return_value = expected_df

        result = analyzer.analyze_resource_distribution()

        analyzer.db._execute_in_transaction.assert_called_once()
        assert result is expected_df

    def test_analyze_competitive_interactions_calls_transaction(self):
        analyzer = self._make_analyzer()
        expected_df = pd.DataFrame({"step": [1], "competitive_interactions": [2]})
        analyzer.db._execute_in_transaction.return_value = expected_df

        result = analyzer.analyze_competitive_interactions()

        analyzer.db._execute_in_transaction.assert_called_once()
        assert result is expected_df

    def test_analyze_resource_efficiency_calls_transaction(self):
        analyzer = self._make_analyzer()
        expected_df = pd.DataFrame({"step": [1], "efficiency": [0.75]})
        analyzer.db._execute_in_transaction.return_value = expected_df

        result = analyzer.analyze_resource_efficiency()

        analyzer.db._execute_in_transaction.assert_called_once()
        assert result is expected_df

    def test_generate_report_calls_analysis_methods(self, tmp_path):
        analyzer = self._make_analyzer()

        survival_df = pd.DataFrame({"step": [1], "system_alive": [5.0], "independent_alive": [3.0]})
        efficiency_df = pd.DataFrame({"step": [1, 2], "efficiency": [0.5, 0.8]})

        # Patch matplotlib.pyplot.savefig to avoid writing to disk
        with patch("farm.core.analysis.plt") as mock_plt:
            analyzer.db._execute_in_transaction.side_effect = [survival_df, efficiency_df]

            output_file = str(tmp_path / "report.html")
            analyzer.generate_report(output_file=output_file)

        # Verify HTML file was written
        import os
        assert os.path.exists(output_file)
        with open(output_file) as f:
            content = f.read()
        assert "<html>" in content.lower()
