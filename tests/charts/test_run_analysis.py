"""Tests for farm/utils/run_analysis.py."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestRunAnalysisLoadData(unittest.TestCase):
    """Tests for the load_data helper in run_analysis."""

    @patch("farm.utils.run_analysis.create_engine")
    @patch("pandas.read_sql")
    def test_load_data_returns_dataframe(self, mock_read_sql, mock_engine):
        """load_data returns the DataFrame from the database."""
        from farm.utils.run_analysis import load_data

        expected_df = pd.DataFrame({"action_type": ["move"], "reward": [0.5]})
        mock_read_sql.return_value = expected_df

        result = load_data("some/path/simulation.db")

        mock_engine.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    @patch("farm.utils.run_analysis.create_engine")
    @patch("pandas.read_sql")
    def test_load_data_raises_on_db_error(self, mock_read_sql, mock_engine):
        """load_data re-raises database errors as Exception."""
        from farm.utils.run_analysis import load_data

        mock_read_sql.side_effect = Exception("no such table")
        with self.assertRaises(Exception) as ctx:
            load_data("missing.db")
        self.assertIn("Error loading data", str(ctx.exception))


class TestRunAnalysisSetupEnvironment(unittest.TestCase):
    """Tests for setup_environment in run_analysis."""

    def test_setup_environment_raises_when_no_key(self):
        """setup_environment raises EnvironmentError when API key is missing."""
        from farm.utils.run_analysis import setup_environment

        mock_cfg = MagicMock()
        mock_cfg.get_openai_api_key.return_value = None

        with patch("farm.utils.run_analysis.load_dotenv"):
            with self.assertRaises(EnvironmentError):
                setup_environment(mock_cfg)

    def test_setup_environment_passes_when_key_present(self):
        """setup_environment does not raise when API key exists."""
        from farm.utils.run_analysis import setup_environment

        mock_cfg = MagicMock()
        mock_cfg.get_openai_api_key.return_value = "sk-test"

        with patch("farm.utils.run_analysis.load_dotenv"):
            # Should not raise
            setup_environment(mock_cfg)


class TestRunAnalysisPipeline(unittest.TestCase):
    """Tests for the run_analysis pipeline function."""

    @patch("farm.utils.run_analysis.ChartAnalyzer")
    @patch("farm.utils.run_analysis.SimulationDatabase")
    @patch("farm.utils.run_analysis.load_data")
    @patch("farm.utils.run_analysis.setup_environment")
    @patch("farm.utils.run_analysis.EnvConfigService")
    def test_run_analysis_save_charts_true(
        self, mock_cfg_cls, mock_setup_env, mock_load_data, mock_db_cls, mock_analyzer_cls
    ):
        """run_analysis creates output dir and runs the full pipeline when save_charts=True."""
        import tempfile
        from farm.utils.run_analysis import run_analysis

        mock_load_data.return_value = pd.DataFrame({"action_type": ["move"]})
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_all_charts.return_value = {
            "chart1": "Analysis text for chart1",
            "chart2": "Short",
        }
        mock_analyzer_cls.return_value = mock_analyzer
        mock_db_cls.return_value = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.get_openai_api_key.return_value = "sk-test"
        mock_cfg_cls.return_value = mock_cfg

        with tempfile.TemporaryDirectory() as tmpdir:
            run_analysis("test.db", tmpdir, save_charts=True)
            mock_analyzer.analyze_all_charts.assert_called_once()

    @patch("farm.utils.run_analysis.ChartAnalyzer")
    @patch("farm.utils.run_analysis.SimulationDatabase")
    @patch("farm.utils.run_analysis.load_data")
    @patch("farm.utils.run_analysis.setup_environment")
    @patch("farm.utils.run_analysis.EnvConfigService")
    def test_run_analysis_save_charts_false(
        self, mock_cfg_cls, mock_setup_env, mock_load_data, mock_db_cls, mock_analyzer_cls
    ):
        """run_analysis runs without creating output dir when save_charts=False."""
        from farm.utils.run_analysis import run_analysis

        mock_load_data.return_value = pd.DataFrame({"action_type": ["move"]})
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_all_charts.return_value = {}
        mock_analyzer_cls.return_value = mock_analyzer
        mock_db_cls.return_value = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.get_openai_api_key.return_value = "sk-test"
        mock_cfg_cls.return_value = mock_cfg

        # Should not raise
        run_analysis("test.db", "/tmp/test_output", save_charts=False)
        mock_analyzer.analyze_all_charts.assert_called_once()


class TestRunAnalysisFunctionsExist(unittest.TestCase):
    """Verify public API of run_analysis module."""

    def test_module_functions_exist(self):
        import farm.utils.run_analysis as m

        for name in ["setup_environment", "load_data", "run_analysis", "main"]:
            self.assertTrue(hasattr(m, name), f"Missing function: {name}")


if __name__ == "__main__":
    unittest.main()
