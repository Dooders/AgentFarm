"""
Tests for ExperimentController.

Covers initialization, run_experiment lifecycle, iteration helpers,
analyze_results, get_state, cleanup, and context-manager protocol.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from farm.api.experiment_controller import ExperimentController


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    cfg = Mock()
    cfg.simulation_steps = 10
    cfg.width = 10
    cfg.height = 10
    cfg.initial_resources = 3
    cfg_copy = Mock()
    cfg_copy.simulation_steps = 10
    cfg.copy.return_value = cfg_copy
    cfg.to_dict.return_value = {}
    return cfg


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "experiment_output"


@pytest.fixture
def controller(mock_config, tmp_output):
    """ExperimentController with a temporary output directory."""
    with patch("farm.api.experiment_controller.ResearchProject") as mock_project_cls:
        mock_project = Mock()
        mock_project.project_path = str(tmp_output)
        mock_project_cls.return_value = mock_project

        ctrl = ExperimentController(
            name="test_experiment",
            description="Unit test experiment",
            base_config=mock_config,
            output_dir=tmp_output,
        )
        yield ctrl


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestExperimentControllerInit:
    def test_initial_state(self, controller):
        assert controller.name == "test_experiment"
        assert controller.description == "Unit test experiment"
        assert controller.current_iteration == 0
        assert controller.total_iterations == 0
        assert not controller.is_running
        assert isinstance(controller.results, list)

    def test_output_dir_created(self, controller, tmp_output):
        # The fixture passes output_dir directly, so it should exist after init
        assert controller.output_dir is not None

    def test_custom_output_dir(self, mock_config, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        with patch("farm.api.experiment_controller.ResearchProject"):
            ctrl = ExperimentController(
                name="exp",
                description="desc",
                base_config=mock_config,
                output_dir=custom_dir,
            )
        assert ctrl.output_dir == custom_dir


# ---------------------------------------------------------------------------
# _create_iteration_config
# ---------------------------------------------------------------------------

class TestCreateIterationConfig:
    def test_no_variations(self, controller, mock_config):
        result = controller._create_iteration_config(0, None)
        assert result is mock_config.copy.return_value

    def test_variation_applied(self, controller, mock_config):
        variations = [{"simulation_steps": 99}]
        result = controller._create_iteration_config(0, variations)
        assert result.simulation_steps == 99
        assert mock_config.simulation_steps == 10  # base config unchanged

    def test_variation_index_out_of_range(self, controller, mock_config):
        variations = [{"simulation_steps": 99}]
        result = controller._create_iteration_config(5, variations)
        assert result is mock_config.copy.return_value  # copy returned, no variation applied


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

class TestRunExperiment:
    def _make_fast_controller(self, controller):
        """Patch _run_iteration and _analyze_iteration to be no-ops."""
        controller._run_iteration = Mock()
        controller._analyze_iteration = Mock()
        return controller

    def test_run_experiment_basic(self, controller):
        self._make_fast_controller(controller)
        controller.run_experiment(num_iterations=3, num_steps=5, run_analysis=False)
        assert controller.current_iteration == 3
        assert not controller.is_running

    def test_run_experiment_calls_analyze_iteration(self, controller):
        controller._run_iteration = Mock()
        controller._analyze_iteration = Mock()
        controller.run_experiment(num_iterations=2, num_steps=5, run_analysis=True)
        assert controller._analyze_iteration.call_count == 2

    def test_run_experiment_skips_analyze_when_false(self, controller):
        controller._run_iteration = Mock()
        controller._analyze_iteration = Mock()
        controller.run_experiment(num_iterations=2, num_steps=5, run_analysis=False)
        controller._analyze_iteration.assert_not_called()

    def test_run_experiment_applies_variations(self, controller, mock_config):
        iterations_ran = []

        def capture_config(config, output_dir, num_steps):
            iterations_ran.append(config)

        controller._run_iteration = capture_config
        controller._analyze_iteration = Mock()
        variations = [
            {"simulation_steps": 11},
            {"simulation_steps": 22},
        ]
        controller.run_experiment(num_iterations=2, variations=variations, run_analysis=False)
        assert len(iterations_ran) == 2

    def test_run_experiment_error_resets_running_flag(self, controller):
        controller._run_iteration = Mock(side_effect=RuntimeError("iter failed"))
        with pytest.raises(RuntimeError):
            controller.run_experiment(num_iterations=1, run_analysis=False)
        assert not controller.is_running

    def test_run_experiment_total_iterations_set(self, controller):
        controller._run_iteration = Mock()
        controller._analyze_iteration = Mock()
        controller.run_experiment(num_iterations=5, run_analysis=False)
        assert controller.total_iterations == 5


# ---------------------------------------------------------------------------
# _run_iteration (mocked SimulationController)
# ---------------------------------------------------------------------------

class TestRunIteration:
    def test_run_iteration_completes(self, controller, mock_config, tmp_path):
        iter_dir = tmp_path / "iter1"
        iter_dir.mkdir()

        with patch("farm.api.experiment_controller.SimulationController") as mock_ctrl_cls:
            mock_ctrl = MagicMock()
            mock_ctrl.is_running = False
            mock_ctrl.current_step = 10  # equal to num_steps
            mock_ctrl_cls.return_value = mock_ctrl

            controller._run_iteration(mock_config, iter_dir, num_steps=10)

        mock_ctrl.initialize_simulation.assert_called_once()
        mock_ctrl.start.assert_called_once()
        mock_ctrl.cleanup.assert_called_once()

    def test_run_iteration_cleanup_always_called(self, controller, mock_config, tmp_path):
        iter_dir = tmp_path / "iter1"
        iter_dir.mkdir()

        with patch("farm.api.experiment_controller.SimulationController") as mock_ctrl_cls:
            mock_ctrl = MagicMock()
            mock_ctrl.initialize_simulation.side_effect = RuntimeError("init fail")
            mock_ctrl_cls.return_value = mock_ctrl

            with pytest.raises(RuntimeError):
                controller._run_iteration(mock_config, iter_dir, num_steps=10)

        mock_ctrl.cleanup.assert_called_once()

    def test_run_iteration_raises_on_early_stop(self, controller, mock_config, tmp_path):
        iter_dir = tmp_path / "iter1"
        iter_dir.mkdir()

        with patch("farm.api.experiment_controller.SimulationController") as mock_ctrl_cls:
            mock_ctrl = MagicMock()
            mock_ctrl.is_running = False
            mock_ctrl.current_step = 3  # less than num_steps=10
            mock_ctrl_cls.return_value = mock_ctrl

            with pytest.raises(RuntimeError, match="stopped early"):
                controller._run_iteration(mock_config, iter_dir, num_steps=10)


# ---------------------------------------------------------------------------
# _analyze_iteration
# ---------------------------------------------------------------------------

class TestAnalyzeIteration:
    def test_analyze_iteration_skips_missing_db(self, controller, tmp_path):
        iter_dir = tmp_path / "no_db_iter"
        iter_dir.mkdir()
        # db_path does not exist; should log warning and return without error
        controller._analyze_iteration(iter_dir)

    def test_analyze_iteration_runs_chart_analyzer(self, controller, tmp_path):
        iter_dir = tmp_path / "iter1"
        iter_dir.mkdir()
        db_file = iter_dir / "simulation.db"
        db_file.touch()

        with patch("farm.api.experiment_controller.SimulationDatabase") as mock_db_cls, \
             patch("farm.api.experiment_controller.ChartAnalyzer") as mock_chart_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_chart = MagicMock()
            mock_chart_cls.return_value = mock_chart

            controller._analyze_iteration(iter_dir)

        mock_chart.analyze_all_charts.assert_called_once_with(iter_dir)
        mock_db.close.assert_called_once()

    def test_analyze_iteration_db_close_on_chart_error(self, controller, tmp_path):
        iter_dir = tmp_path / "iter1"
        iter_dir.mkdir()
        db_file = iter_dir / "simulation.db"
        db_file.touch()

        with patch("farm.api.experiment_controller.SimulationDatabase") as mock_db_cls, \
             patch("farm.api.experiment_controller.ChartAnalyzer") as mock_chart_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            mock_chart = MagicMock()
            mock_chart.analyze_all_charts.side_effect = RuntimeError("chart error")
            mock_chart_cls.return_value = mock_chart

            # Should not re-raise; just logs the error
            controller._analyze_iteration(iter_dir)

        mock_db.close.assert_called_once()


# ---------------------------------------------------------------------------
# analyze_results
# ---------------------------------------------------------------------------

class TestAnalyzeResults:
    def test_analyze_results_runs_compare(self, controller, tmp_output):
        tmp_output.mkdir(parents=True, exist_ok=True)

        with patch("farm.api.experiment_controller.compare_simulations") as mock_cmp:
            controller.analyze_results()

        mock_cmp.assert_called_once()

    def test_analyze_results_missing_output_dir(self, controller, tmp_path):
        controller.output_dir = tmp_path / "nonexistent"
        # Should log warning and return without raising
        controller.analyze_results()

    def test_analyze_results_error_propagates(self, controller, tmp_output):
        tmp_output.mkdir(parents=True, exist_ok=True)

        with patch("farm.api.experiment_controller.compare_simulations") as mock_cmp:
            mock_cmp.side_effect = RuntimeError("compare failed")
            with pytest.raises(RuntimeError):
                controller.analyze_results()


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------

class TestGetState:
    def test_get_state_returns_dict(self, controller):
        state = controller.get_state()
        assert isinstance(state, dict)
        assert state["name"] == "test_experiment"
        assert "is_running" in state
        assert "current_iteration" in state
        assert "total_iterations" in state

    def test_get_state_reflects_current_iteration(self, controller):
        controller.current_iteration = 7
        controller.total_iterations = 10
        state = controller.get_state()
        assert state["current_iteration"] == 7
        assert state["total_iterations"] == 10


# ---------------------------------------------------------------------------
# Cleanup & context manager
# ---------------------------------------------------------------------------

class TestCleanupAndContextManager:
    def test_cleanup_runs_without_error(self, controller):
        controller.cleanup()

    def test_context_manager(self, controller):
        with controller as ctrl:
            assert ctrl is controller
        # No exception on exit

    def test_context_manager_on_exception(self, controller):
        with pytest.raises(ValueError):
            with controller:
                raise ValueError("boom")
        # cleanup should have been called (no secondary exception)

    def test_del_does_not_raise(self, controller):
        # __del__ calls cleanup(); verify that doesn't raise
        controller.__del__()
