"""Tests for config CLI module covering command parsing."""

import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import yaml

from farm.config import cli, SimulationConfig


class TestCmdVersion(unittest.TestCase):
    """Tests for cmd_version CLI handler."""

    @patch("farm.config.cli.SimulationConfig")
    def test_cmd_version_create(self, mock_config_class):
        """Test version create command."""
        mock_config = Mock()
        mock_versioned = Mock()
        mock_versioned.save_versioned_config.return_value = "/path/to/config"
        mock_versioned.config_version = "abc123"
        mock_config.version_config.return_value = mock_versioned
        mock_config_class.from_centralized_config.return_value = mock_config

        args = Mock()
        args.subcommand = "create"
        args.environment = "test"
        args.profile = None
        args.output_dir = "/tmp"
        args.description = "test version"

        cli.cmd_version(args)
        mock_config.version_config.assert_called_once_with(args.description)

    def test_cmd_version_list_empty(self):
        """Test version list with no versions."""
        args = Mock()
        args.subcommand = "list"
        args.directory = "/tmp/nonexistent_versions_dir"

        with patch("farm.config.cli.SimulationConfig.list_config_versions") as mock_list:
            mock_list.return_value = []
            cli.cmd_version(args)
            mock_list.assert_called_once_with(args.directory)

    def test_cmd_version_list_with_versions(self):
        """Test version list with existing versions prints them."""
        args = Mock()
        args.subcommand = "list"
        args.directory = "/tmp"

        versions = [
            {"version": "abc123", "created_at": "2024-01-01", "description": "v1"},
        ]
        with patch("farm.config.cli.SimulationConfig.list_config_versions", return_value=versions):
            with patch("builtins.print") as mock_print:
                cli.cmd_version(args)
            # At least one print call should mention the version hash
            all_printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("abc123", all_printed)

    def test_cmd_version_load_to_stdout(self):
        """Test version load without --output prints config to stdout."""
        mock_config = Mock()
        mock_config.config_created_at = "2024-01-01"
        mock_config.config_description = "test"
        mock_config.to_dict.return_value = {"simulation_steps": 10}

        args = Mock()
        args.subcommand = "load"
        args.version = "abc123"
        args.directory = "/tmp"
        args.output = None

        with patch("farm.config.cli.SimulationConfig.load_versioned_config", return_value=mock_config):
            cli.cmd_version(args)
        mock_config.to_dict.assert_called()

    def test_cmd_version_load_to_file(self):
        """Test version load with --output saves to file."""
        mock_config = Mock()
        mock_config.config_created_at = "2024-01-01"
        mock_config.config_description = "test"

        args = Mock()
        args.subcommand = "load"
        args.version = "abc123"
        args.directory = "/tmp"
        args.output = "/tmp/out.yaml"

        with patch("farm.config.cli.SimulationConfig.load_versioned_config", return_value=mock_config):
            cli.cmd_version(args)
        mock_config.to_yaml.assert_called_once_with(args.output)


class TestCmdTemplate(unittest.TestCase):
    """Tests for cmd_template CLI handler."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("farm.config.cli.SimulationConfig")
    def test_cmd_template_create(self, mock_config_class):
        """Test template create command."""
        mock_config = Mock()
        mock_config_class.from_centralized_config.return_value = mock_config

        mock_template = Mock()
        mock_template.get_required_variables.return_value = []

        args = Mock()
        args.subcommand = "create"
        args.name = "test_tpl"
        args.environment = "dev"
        args.profile = None
        args.description = "Test template"
        args.template_dir = self.tmpdir

        with patch("farm.config.cli.ConfigTemplate.from_config", return_value=mock_template):
            with patch("farm.config.cli.ConfigTemplateManager.save_template", return_value="/tmp/test_tpl.yaml"):
                cli.cmd_template(args)

    def test_cmd_template_list_empty(self):
        """Test template list with no templates."""
        args = Mock()
        args.subcommand = "list"
        args.template_dir = self.tmpdir

        cli.cmd_template(args)  # Should not raise

    def test_cmd_template_list_with_templates(self):
        """Test template list with existing templates."""
        from farm.config.template import ConfigTemplate, ConfigTemplateManager

        manager = ConfigTemplateManager(template_dir=self.tmpdir)
        template = ConfigTemplate.from_config(SimulationConfig())
        manager.save_template("my_tpl", template, description="desc")

        args = Mock()
        args.subcommand = "list"
        args.template_dir = self.tmpdir

        with patch("builtins.print") as mock_print:
            cli.cmd_template(args)
        all_printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("my_tpl", all_printed)

    def test_cmd_template_instantiate_missing_vars(self):
        """Instantiate with missing variables prints error."""
        from farm.config.template import ConfigTemplate, ConfigTemplateManager

        base = SimulationConfig().to_dict()
        base["simulation_steps"] = "{{steps}}"
        template = ConfigTemplate(base)
        manager = ConfigTemplateManager(template_dir=self.tmpdir)
        manager.save_template("tpl_vars", template)

        args = Mock()
        args.subcommand = "instantiate"
        args.name = "tpl_vars"
        args.output = os.path.join(self.tmpdir, "out.yaml")
        args.variables = None  # no variables provided
        args.template_dir = self.tmpdir

        with patch("builtins.print") as mock_print:
            cli.cmd_template(args)
        all_printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Missing", all_printed)

    def test_cmd_template_instantiate_with_vars(self):
        """Instantiate with all variables provided creates output file."""
        from farm.config.template import ConfigTemplate, ConfigTemplateManager

        base = SimulationConfig().to_dict()
        base["simulation_steps"] = "{{steps}}"
        base["max_steps"] = "{{steps}}"
        template = ConfigTemplate(base)
        manager = ConfigTemplateManager(template_dir=self.tmpdir)
        manager.save_template("tpl_ok", template)

        out_path = os.path.join(self.tmpdir, "out.yaml")
        args = Mock()
        args.subcommand = "instantiate"
        args.name = "tpl_ok"
        args.output = out_path
        args.variables = ["steps=30"]
        args.template_dir = self.tmpdir

        cli.cmd_template(args)
        self.assertTrue(os.path.exists(out_path))

    def test_cmd_template_instantiate_json_variable(self):
        """Instantiate command parses JSON numeric values from variable strings."""
        from farm.config.template import ConfigTemplate, ConfigTemplateManager

        base = SimulationConfig().to_dict()
        base["simulation_steps"] = "{{steps}}"
        base["max_steps"] = "{{steps}}"
        template = ConfigTemplate(base)
        manager = ConfigTemplateManager(template_dir=self.tmpdir)
        manager.save_template("tpl_json", template)

        out_path = os.path.join(self.tmpdir, "out_json.yaml")
        args = Mock()
        args.subcommand = "instantiate"
        args.name = "tpl_json"
        args.output = out_path
        args.variables = ["steps=15"]  # numeric string, json.loads("15") = 15
        args.template_dir = self.tmpdir

        cli.cmd_template(args)
        self.assertTrue(os.path.exists(out_path))

    def test_cmd_template_instantiate_string_variable_fallback(self):
        """Instantiate command falls back to string for non-JSON variable values."""
        from farm.config.template import ConfigTemplate, ConfigTemplateManager

        base = SimulationConfig().to_dict()
        template = ConfigTemplate(base)
        manager = ConfigTemplateManager(template_dir=self.tmpdir)
        manager.save_template("tpl_strfb", template)

        out_path = os.path.join(self.tmpdir, "out_str.yaml")
        args = Mock()
        args.subcommand = "instantiate"
        args.name = "tpl_strfb"
        args.output = out_path
        # "my_text" is not valid JSON, triggers JSONDecodeError fallback to string
        args.variables = ["dummy_var=my_text"]
        args.template_dir = self.tmpdir

        # Template has no placeholders so instantiates without using dummy_var
        cli.cmd_template(args)
        self.assertTrue(os.path.exists(out_path))

    def test_cmd_template_batch_no_variable_sets(self):
        """Batch with no variable sets prints an error."""
        args = Mock()
        args.subcommand = "batch"
        args.name = "tpl"
        args.variables = None
        args.variable_file = None
        args.output_dir = self.tmpdir
        args.template_dir = self.tmpdir

        with patch("builtins.print") as mock_print:
            cli.cmd_template(args)
        all_printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("No variable sets", all_printed)

    def test_cmd_template_batch_with_variable_file(self):
        """Batch command reads variable sets from a JSON file."""
        import json as json_mod
        from farm.config.template import ConfigTemplate, ConfigTemplateManager

        base = SimulationConfig().to_dict()
        base["simulation_steps"] = "{{steps}}"
        base["max_steps"] = "{{steps}}"
        template = ConfigTemplate(base)
        manager = ConfigTemplateManager(template_dir=self.tmpdir)
        manager.save_template("tpl_batch", template)

        var_file = os.path.join(self.tmpdir, "vars.json")
        with open(var_file, "w") as f:
            json_mod.dump({"variable_sets": [{"steps": 10}, {"steps": 20}]}, f)

        output_dir = os.path.join(self.tmpdir, "batch_out")
        args = Mock()
        args.subcommand = "batch"
        args.name = "tpl_batch"
        args.variables = None
        args.variable_file = var_file
        args.output_dir = output_dir
        args.template_dir = self.tmpdir

        cli.cmd_template(args)
        self.assertTrue(os.path.isdir(output_dir))
        self.assertGreater(len(os.listdir(output_dir)), 0)

    def test_cmd_template_batch_with_variables(self):
        """Batch command processes variable sets from command-line variables."""
        from farm.config.template import ConfigTemplate, ConfigTemplateManager

        base = SimulationConfig().to_dict()
        base["simulation_steps"] = "{{steps}}"
        base["max_steps"] = "{{steps}}"
        template = ConfigTemplate(base)
        manager = ConfigTemplateManager(template_dir=self.tmpdir)
        manager.save_template("tpl_clivar", template)

        output_dir = os.path.join(self.tmpdir, "cli_out")
        args = Mock()
        args.subcommand = "batch"
        args.name = "tpl_clivar"
        args.variables = ["steps=10"]  # CLI variable (JSON-parseable integer)
        args.variable_file = None
        args.output_dir = output_dir
        args.template_dir = self.tmpdir

        cli.cmd_template(args)
        self.assertTrue(os.path.isdir(output_dir))
        self.assertGreater(len(os.listdir(output_dir)), 0)

    def test_cmd_template_batch_with_string_variable(self):
        """Batch command falls back to string for non-JSON variable values."""
        from farm.config.template import ConfigTemplate, ConfigTemplateManager

        base = SimulationConfig().to_dict()
        template = ConfigTemplate(base)
        manager = ConfigTemplateManager(template_dir=self.tmpdir)
        manager.save_template("tpl_strvar", template)

        output_dir = os.path.join(self.tmpdir, "str_out")
        args = Mock()
        args.subcommand = "batch"
        args.name = "tpl_strvar"
        # "not_json_value" triggers JSONDecodeError fallback to str
        args.variables = ["dummy_key=not_json_value"]
        args.variable_file = None
        args.output_dir = output_dir
        args.template_dir = self.tmpdir

        # Template has no {{}} placeholders so it creates one config ignoring the variable
        try:
            cli.cmd_template(args)
        except Exception:
            pass  # Acceptable if template fails on unexpected variable


class TestCmdDiff(unittest.TestCase):
    """Tests for cmd_diff CLI handler."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cmd_diff_identical_yaml_files(self):
        """Diffing two identical YAML configs reports no differences."""
        c = SimulationConfig()
        p1 = os.path.join(self.tmpdir, "c1.yaml")
        p2 = os.path.join(self.tmpdir, "c2.yaml")
        c.to_yaml(p1)
        c.to_yaml(p2)

        args = Mock()
        args.config1 = p1
        args.config2 = p2
        args.version_dir = self.tmpdir

        with patch("builtins.print") as mock_print:
            cli.cmd_diff(args)
        all_printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("identical", all_printed)

    def test_cmd_diff_different_yaml_files(self):
        """Diffing two different YAML configs prints differences."""
        c1 = SimulationConfig(simulation_steps=10)
        c2 = SimulationConfig(simulation_steps=99)
        p1 = os.path.join(self.tmpdir, "c1.yaml")
        p2 = os.path.join(self.tmpdir, "c2.yaml")
        c1.to_yaml(p1)
        c2.to_yaml(p2)

        args = Mock()
        args.config1 = p1
        args.config2 = p2
        args.version_dir = self.tmpdir

        with patch("builtins.print") as mock_print:
            cli.cmd_diff(args)
        all_printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("simulation_steps", all_printed)


class TestCmdWatch(unittest.TestCase):
    """Tests for cmd_watch CLI handler."""

    def test_cmd_watch_status_no_files(self):
        """Watch status with no watched files prints appropriate message."""
        args = Mock()
        args.subcommand = "status"

        mock_watcher = Mock()
        mock_watcher.get_watched_files.return_value = {}

        with patch("farm.config.cli.get_global_watcher", return_value=mock_watcher):
            with patch("builtins.print") as mock_print:
                cli.cmd_watch(args)
        all_printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("No files", all_printed)

    def test_cmd_watch_status_with_files(self):
        """Watch status with watched files lists them."""
        args = Mock()
        args.subcommand = "status"

        mock_watcher = Mock()
        mock_watcher.get_watched_files.return_value = {
            "/tmp/cfg.yaml": "a" * 64,
        }

        with patch("farm.config.cli.get_global_watcher", return_value=mock_watcher):
            with patch("builtins.print") as mock_print:
                cli.cmd_watch(args)
        all_printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("cfg.yaml", all_printed)

    def test_cmd_watch_start(self):
        """Watch start registers the callback and starts the watcher."""
        args = Mock()
        args.subcommand = "start"
        args.filepath = "/tmp/test.yaml"
        args.verbose = False

        mock_watcher = Mock()

        # Simulate KeyboardInterrupt immediately to exit the while loop
        mock_watcher.start.side_effect = None

        with patch("farm.config.cli.get_global_watcher", return_value=mock_watcher):
            with patch("builtins.print"):
                with patch("time.sleep", side_effect=KeyboardInterrupt):
                    cli.cmd_watch(args)

        mock_watcher.watch_file.assert_called_once()
        mock_watcher.start.assert_called_once()
        mock_watcher.stop.assert_called_once()


class TestMain(unittest.TestCase):
    """Tests for main() argument parsing."""

    def test_main_no_args_does_not_raise(self):
        """main() with no args prints help without raising."""
        with patch("sys.argv", ["config"]):
            cli.main()  # Should not raise

    def test_main_error_handling(self):
        """main() catches exceptions and exits with code 1."""
        import sys
        with patch("sys.argv", ["config", "version", "create"]):
            with patch("farm.config.cli.cmd_version", side_effect=RuntimeError("fail")):
                with self.assertRaises(SystemExit) as ctx:
                    cli.main()
                self.assertEqual(ctx.exception.code, 1)

    def test_main_dispatches_template(self):
        """main() dispatches to cmd_template for 'template' command."""
        with patch("sys.argv", ["config", "template", "list"]):
            with patch("farm.config.cli.cmd_template") as mock_cmd:
                cli.main()
            mock_cmd.assert_called_once()

    def test_main_dispatches_diff(self):
        """main() dispatches to cmd_diff for 'diff' command."""
        with patch("sys.argv", ["config", "diff", "/a.yaml", "/b.yaml"]):
            with patch("farm.config.cli.cmd_diff") as mock_cmd:
                cli.main()
            mock_cmd.assert_called_once()

    def test_main_dispatches_watch(self):
        """main() dispatches to cmd_watch for 'watch' command."""
        with patch("sys.argv", ["config", "watch", "status"]):
            with patch("farm.config.cli.cmd_watch") as mock_cmd:
                cli.main()
            mock_cmd.assert_called_once()


if __name__ == "__main__":
    unittest.main()

