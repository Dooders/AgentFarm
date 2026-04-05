"""
Tests for farm/api/cli.py.

Exercises all demo functions and the main() entry point using a mocked
AgentFarmController so no simulation actually runs.
"""

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# Helper: build a minimal mock controller with the right attributes
from farm.api.models import (
    ExperimentStatus,
    SessionStatus,
    SimulationStatus as SimStatusModel,
)


def _make_simulation_status(step=5, total=10, status="completed"):
    s = Mock()
    s.current_step = step
    s.total_steps = total
    s.progress_percentage = 50.0
    s.status = Mock()
    s.status.value = status
    return s


def _make_experiment_status(iteration=2, total=3, status="completed"):
    s = Mock()
    s.current_iteration = iteration
    s.total_iterations = total
    s.progress_percentage = 66.0
    s.status = Mock()
    s.status.value = status
    return s


def _make_simulation_results():
    r = Mock()
    r.final_agent_count = 10
    r.final_resource_count = 5
    r.data_files = ["file1.db", "file2.db"]
    return r


def _make_experiment_results():
    r = Mock()
    r.completed_iterations = 3
    r.total_iterations = 3
    r.data_files = ["file1.db"]
    return r


def _make_config_template():
    t = Mock()
    t.name = "basic_simulation"
    t.description = "A basic simulation template"
    t.category = Mock()
    t.category.value = "simulation"
    t.required_fields = ["steps"]
    t.optional_fields = ["agents"]
    return t


def _make_validation_result(valid=True, warnings=None, errors=None):
    r = Mock()
    r.is_valid = valid
    r.warnings = warnings or []
    r.errors = errors or []
    return r


@pytest.fixture
def mock_controller():
    ctrl = MagicMock()

    # session
    ctrl.create_session.return_value = "session-abc"

    # simulation happy path
    ctrl.create_simulation.return_value = "sim-xyz"
    # Return completed status so demo loop exits immediately
    ctrl.get_simulation_status.return_value = _make_simulation_status(status="completed")
    ctrl.get_simulation_results.return_value = _make_simulation_results()

    # experiment happy path
    ctrl.create_experiment.return_value = "exp-xyz"
    ctrl.get_experiment_status.return_value = _make_experiment_status(status="completed")
    ctrl.get_experiment_results.return_value = _make_experiment_results()

    # configs
    ctrl.get_available_configs.return_value = [_make_config_template()]

    # validation
    ctrl.validate_config.side_effect = lambda cfg: _make_validation_result(
        valid="steps" in cfg,
        warnings=["Consider more steps"] if "steps" in cfg else [],
        errors=[] if "steps" in cfg else ["Missing 'steps'"],
    )

    # template
    ctrl.create_config_from_template.return_value = {
        "name": "basic_simulation",
        "steps": 200,
        "agents": {"system_agents": 8, "independent_agents": 12},
    }

    ctrl.cleanup = Mock()
    return ctrl


# ---------------------------------------------------------------------------
# Helper config factories
# ---------------------------------------------------------------------------

class TestConfigFactories:
    def test_create_basic_simulation_config(self):
        from farm.api.cli import create_basic_simulation_config
        cfg = create_basic_simulation_config()
        assert "steps" in cfg
        assert "environment" in cfg
        assert "agents" in cfg

    def test_create_basic_experiment_config(self):
        from farm.api.cli import create_basic_experiment_config
        cfg = create_basic_experiment_config()
        assert "iterations" in cfg
        assert "variations" in cfg
        assert len(cfg["variations"]) == 3


# ---------------------------------------------------------------------------
# Individual demo functions
# ---------------------------------------------------------------------------

class TestDemoFunctions:
    def test_run_simulation_demo(self, mock_controller, capsys):
        from farm.api.cli import run_simulation_demo
        run_simulation_demo(mock_controller)
        out = capsys.readouterr().out
        assert "Simulation Demo" in out
        mock_controller.create_session.assert_called_once()
        mock_controller.create_simulation.assert_called_once()
        mock_controller.start_simulation.assert_called_once()

    def test_run_experiment_demo(self, mock_controller, capsys):
        from farm.api.cli import run_experiment_demo
        run_experiment_demo(mock_controller)
        out = capsys.readouterr().out
        assert "Experiment Demo" in out
        mock_controller.create_experiment.assert_called_once()
        mock_controller.start_experiment.assert_called_once()

    def test_list_configs_demo(self, mock_controller, capsys):
        from farm.api.cli import list_configs_demo
        list_configs_demo(mock_controller)
        out = capsys.readouterr().out
        assert "Configurations" in out
        mock_controller.get_available_configs.assert_called_once()

    def test_validate_config_demo_valid(self, mock_controller, capsys):
        from farm.api.cli import validate_config_demo
        validate_config_demo(mock_controller)
        out = capsys.readouterr().out
        assert "True" in out  # valid config

    def test_validate_config_demo_invalid(self, mock_controller, capsys):
        from farm.api.cli import validate_config_demo
        validate_config_demo(mock_controller)
        out = capsys.readouterr().out
        # Invalid config result should also appear
        assert "False" in out

    def test_create_config_from_template_demo(self, mock_controller, capsys):
        from farm.api.cli import create_config_from_template_demo
        create_config_from_template_demo(mock_controller)
        out = capsys.readouterr().out
        assert "basic_simulation" in out
        mock_controller.create_config_from_template.assert_called_once()

    def test_run_simulation_demo_with_warnings(self, mock_controller, capsys):
        """Validates warnings output when warnings are present."""
        from farm.api.cli import validate_config_demo
        # Ensure warnings are non-empty for valid config
        mock_controller.validate_config.side_effect = lambda cfg: (
            _make_validation_result(valid=True, warnings=["warning1"])
            if "steps" in cfg
            else _make_validation_result(valid=False)
        )
        validate_config_demo(mock_controller)
        out = capsys.readouterr().out
        assert "warning1" in out


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------

class TestMain:
    def _run_main(self, args, mock_controller):
        """Patch AgentFarmController and sys.argv, then call main()."""
        from farm.api import cli as cli_module

        with patch("farm.api.cli.AgentFarmController", return_value=mock_controller), \
             patch("farm.api.cli.configure_logging"), \
             patch("sys.argv", ["cli"] + args):
            cli_module.main()

    def test_main_no_demo(self, mock_controller, capsys):
        self._run_main([], mock_controller)
        out = capsys.readouterr().out
        assert "No demo specified" in out
        mock_controller.cleanup.assert_called_once()

    def test_main_demo_simulation(self, mock_controller):
        self._run_main(["--demo", "simulation"], mock_controller)
        mock_controller.create_simulation.assert_called_once()

    def test_main_demo_experiment(self, mock_controller):
        self._run_main(["--demo", "experiment"], mock_controller)
        mock_controller.create_experiment.assert_called_once()

    def test_main_demo_configs(self, mock_controller):
        self._run_main(["--demo", "configs"], mock_controller)
        mock_controller.get_available_configs.assert_called_once()

    def test_main_demo_validate(self, mock_controller):
        self._run_main(["--demo", "validate"], mock_controller)
        assert mock_controller.validate_config.called

    def test_main_demo_template(self, mock_controller):
        self._run_main(["--demo", "template"], mock_controller)
        mock_controller.create_config_from_template.assert_called_once()

    def test_main_all_demos(self, mock_controller):
        self._run_main(["--all"], mock_controller)
        mock_controller.get_available_configs.assert_called_once()
        mock_controller.validate_config.assert_called()
        mock_controller.create_config_from_template.assert_called_once()
        mock_controller.create_simulation.assert_called_once()
        mock_controller.create_experiment.assert_called_once()

    def test_main_cleanup_called_on_exception(self, mock_controller):
        """Cleanup is always called even when a demo raises."""
        from farm.api import cli as cli_module

        mock_controller.create_simulation.side_effect = RuntimeError("sim crash")

        with patch("farm.api.cli.AgentFarmController", return_value=mock_controller), \
             patch("farm.api.cli.configure_logging"), \
             patch("sys.argv", ["cli", "--demo", "simulation"]):
            with pytest.raises(RuntimeError):
                cli_module.main()

        mock_controller.cleanup.assert_called_once()

    def test_main_custom_workspace(self, mock_controller):
        from farm.api import cli as cli_module

        with patch("farm.api.cli.AgentFarmController", return_value=mock_controller) as mock_cls, \
             patch("farm.api.cli.configure_logging"), \
             patch("sys.argv", ["cli", "--workspace", "my_workspace"]):
            cli_module.main()

        mock_cls.assert_called_once_with(workspace_path="my_workspace")
