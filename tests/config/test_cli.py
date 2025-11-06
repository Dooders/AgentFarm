"""Tests for config CLI module covering command parsing."""

import unittest
from unittest.mock import Mock, patch

from farm.config import cli


class TestConfigCLI(unittest.TestCase):
    """Tests for config CLI command parsing."""

    @patch("farm.config.cli.SimulationConfig")
    def test_cmd_version_create(self, mock_config_class):
        """Test version create command."""
        mock_config = Mock()
        mock_config.version_config.return_value.save_versioned_config.return_value = "/path/to/config"
        mock_config_class.from_centralized_config.return_value = mock_config

        args = Mock()
        args.subcommand = "create"
        args.environment = "test"
        args.profile = None
        args.output_dir = "/tmp"
        args.description = "test version"

        # Should not raise
        cli.cmd_version(args)

    def test_cmd_version_list(self):
        """Test version list command."""
        args = Mock()
        args.subcommand = "list"
        args.directory = "/tmp"

        with patch("farm.config.cli.SimulationConfig.list_config_versions") as mock_list:
            mock_list.return_value = []
            cli.cmd_version(args)

            mock_list.assert_called_once_with("/tmp")


if __name__ == "__main__":
    unittest.main()

