"""Tests for farm/core/analysis.py – SimulationAnalyzer and analyze_simulation."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from farm.core.analysis import SimulationAnalyzer, analyze_simulation
from farm.database.database import SimulationDatabase


class TestAnalyzeSimulation:
    """Tests for the standalone analyze_simulation helper."""

    @patch("farm.core.analysis.SimulationAnalyzer")
    def test_returns_dict_with_metrics_and_statistics(self, mock_analyzer_cls):
        instance = mock_analyzer_cls.return_value
        instance.calculate_survival_rates.return_value = pd.DataFrame(
            {"step": [1], "system_alive": [2], "independent_alive": [1]}
        )
        instance.analyze_resource_distribution.return_value = pd.DataFrame()
        instance.analyze_competitive_interactions.return_value = pd.DataFrame()
        instance.analyze_resource_efficiency.return_value = pd.DataFrame(
            {"step": [1], "efficiency": [0.5]}
        )

        result = analyze_simulation("/tmp/test.db")

        assert isinstance(result, dict)
        assert "metrics" in result
        assert "statistics" in result
        assert isinstance(result["metrics"], dict)
        assert isinstance(result["statistics"], dict)
        assert result["metrics"]["last_step_system_alive"] == 2
        assert result["metrics"]["database_basename"] == "test.db"
        mock_analyzer_cls.assert_called_once_with(db_path="/tmp/test.db", simulation_id=None)

    def test_accepts_simulation_database_instance_empty_schema(self):
        db = SimulationDatabase(":memory:", simulation_id="sim-1")
        try:
            result = analyze_simulation(db)
        finally:
            db.close()

        assert result["metrics"]["survival_rates_row_count"] == 0
        assert result["metrics"]["simulation_id"] == "sim-1"
        assert "database_basename" not in result["metrics"]
        assert result["statistics"] == {}

    def test_rejects_non_db_input(self):
        with pytest.raises(TypeError, match="SimulationDatabase"):
            analyze_simulation({"step": 1})

        with pytest.raises(TypeError, match="SimulationDatabase"):
            analyze_simulation(None)


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
