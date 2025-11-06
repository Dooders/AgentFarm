"""Tests for core CLI module covering command parsing."""

import unittest
from unittest.mock import Mock, patch

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

    def test_main_parser(self):
        """Test main parser creation."""
        # Just verify main function exists
        self.assertTrue(hasattr(cli, "main"))


if __name__ == "__main__":
    unittest.main()

