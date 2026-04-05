"""Tests for core CLI module covering command parsing."""

import json
import os
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open

import pytest

from farm.core import cli


class TestCoreCLI(unittest.TestCase):
    """Tests for core CLI command parsing."""

    @patch("farm.core.cli.ExperimentRunner")
    @patch("farm.core.cli.SimulationConfig")
    def test_run_experiment(self, mock_config_class, mock_runner_class):
        """Test run_experiment function."""
        mock_config = Mock()
        mock_config_class.from_centralized_config.return_value = mock_config

        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner

        args = Mock()
        args.environment = "test"
        args.experiment_name = "test_exp"
        args.iterations = 10
        args.variations = None

        cli.run_experiment(args)

        mock_runner.run_iterations.assert_called_once_with(10)

    @patch("farm.core.cli.ExperimentRunner")
    @patch("farm.core.cli.SimulationConfig")
    def test_run_experiment_with_variations(self, mock_config_class, mock_runner_class):
        """Test run_experiment loads JSON variations file when provided."""
        mock_config_class.from_centralized_config.return_value = Mock()
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner

        variations_data = {"param": [1, 2, 3]}
        args = Mock()
        args.environment = "test"
        args.experiment_name = "exp_var"
        args.iterations = 5
        args.variations = "variations.json"

        with patch("builtins.open", mock_open(read_data=json.dumps(variations_data))):
            cli.run_experiment(args)

        mock_runner.run_iterations.assert_called_once_with(5, variations_data)

    def test_main_parser(self):
        """Test main function exists."""
        self.assertTrue(hasattr(cli, "main"))


class TestMainVisualizeModeNoDb:
    """Test --mode visualize when the DB file does not exist."""

    def test_missing_db_logs_error_and_returns(self):
        test_args = ["prog", "--mode", "visualize", "--db-path", "/nonexistent/path.db"]
        with patch.object(sys, "argv", test_args):
            with patch("farm.core.cli.configure_logging"):
                with patch("farm.core.cli.logger") as mock_logger:
                    cli.main()
        mock_logger.error.assert_called()


class TestMainAnalyzeModeNoDb:
    """Test --mode analyze when the DB file does not exist."""

    def test_missing_db_logs_error_and_returns(self):
        test_args = ["prog", "--mode", "analyze", "--db-path", "/nonexistent/path.db"]
        with patch.object(sys, "argv", test_args):
            with patch("farm.core.cli.configure_logging"):
                with patch("farm.core.cli.logger") as mock_logger:
                    cli.main()
        mock_logger.error.assert_called()


class TestMainAnalyzeModeWithDb:
    """Test --mode analyze with an existing DB file."""

    def test_analyze_mode_calls_generate_report(self, tmp_path):
        db_file = tmp_path / "sim.db"
        db_file.write_text("")  # create an empty file so os.path.exists passes
        report_path = str(tmp_path / "report.html")

        test_args = [
            "prog", "--mode", "analyze",
            "--db-path", str(db_file),
            "--report-path", report_path,
        ]
        with patch.object(sys, "argv", test_args):
            with patch("farm.core.cli.configure_logging"):
                with patch("farm.core.cli.SimulationAnalyzer") as mock_analyzer_cls:
                    mock_analyzer = MagicMock()
                    mock_analyzer_cls.return_value = mock_analyzer
                    with patch("farm.core.cli.logger"):
                        cli.main()

        mock_analyzer.generate_report.assert_called_once_with(output_file=report_path)


class TestMainSimulateMode:
    """Test --mode simulate (default) with config override."""

    def test_simulate_mode_overrides_population_config(self):
        test_args = [
            "prog", "--mode", "simulate",
            "--system-agents", "10",
            "--independent-agents", "5",
            "--resources", "20",
        ]
        mock_config = MagicMock()
        mock_config.population = MagicMock()
        mock_config.resources = MagicMock()

        with patch.object(sys, "argv", test_args):
            with patch("farm.core.cli.configure_logging"):
                with patch("farm.core.cli.SimulationConfig") as mock_cfg_cls:
                    mock_cfg_cls.from_centralized_config.return_value = mock_config
                    cli.main()

        assert mock_config.population.system_agents == 10
        assert mock_config.population.independent_agents == 5
        assert mock_config.resources.initial_resources == 20

    def test_simulate_mode_saves_config_when_requested(self, tmp_path):
        save_path = str(tmp_path / "config.yaml")
        test_args = ["prog", "--mode", "simulate", "--save-config", save_path]
        mock_config = MagicMock()

        with patch.object(sys, "argv", test_args):
            with patch("farm.core.cli.configure_logging"):
                with patch("farm.core.cli.SimulationConfig") as mock_cfg_cls:
                    mock_cfg_cls.from_centralized_config.return_value = mock_config
                    cli.main()

        mock_config.to_yaml.assert_called_once_with(save_path)


class TestMainExperimentMode:
    """Test --mode experiment routing."""

    @patch("farm.core.cli.run_experiment")
    def test_experiment_mode_calls_run_experiment(self, mock_run_experiment):
        test_args = ["prog", "--mode", "experiment", "--experiment-name", "myexp"]
        with patch.object(sys, "argv", test_args):
            with patch("farm.core.cli.configure_logging"):
                cli.main()
        mock_run_experiment.assert_called_once()


class TestMainVisualizeModeImportError:
    """Test --mode visualize when tkinter/visualization deps are missing."""

    def test_visualize_import_error_logs_error(self, tmp_path):
        db_file = tmp_path / "sim.db"
        db_file.write_text("")

        test_args = ["prog", "--mode", "visualize", "--db-path", str(db_file)]
        with patch.object(sys, "argv", test_args):
            with patch("farm.core.cli.configure_logging"):
                with patch("farm.core.cli.logger") as mock_logger:
                    # Simulate missing tkinter by making the in-function import raise
                    import builtins
                    original_import = builtins.__import__

                    def mock_import(name, *args, **kwargs):
                        if name == "tkinter":
                            raise ImportError("No module named 'tkinter'")
                        return original_import(name, *args, **kwargs)

                    with patch("builtins.__import__", side_effect=mock_import):
                        cli.main()

        # Should have logged an error about unavailable visualization deps
        mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()

