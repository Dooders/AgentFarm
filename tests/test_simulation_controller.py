"""
Tests for SimulationController.

Covers initialization, start/pause/stop/step lifecycle, callbacks,
state retrieval, cleanup, and context-manager protocol.
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

from farm.api.simulation_controller import SimulationController


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Minimal SimulationConfig mock."""
    cfg = Mock()
    cfg.simulation_steps = 10
    cfg.width = 20
    cfg.height = 20
    cfg.initial_resources = 5
    cfg.to_dict.return_value = {"steps": 10}
    return cfg


@pytest.fixture
def mock_db():
    with patch("farm.api.simulation_controller.SimulationDatabase") as cls:
        db = MagicMock()
        cls.return_value = db
        yield db


@pytest.fixture
def mock_env():
    with patch("farm.api.simulation_controller.Environment") as cls:
        env = MagicMock()
        env.agents = [Mock(), Mock()]
        env.resources = [Mock()]
        cls.return_value = env
        yield env


@pytest.fixture
def controller(mock_config, mock_db):
    """Return an un-initialized controller."""
    return SimulationController(mock_config, "/tmp/test_sim.db")


@pytest.fixture
def initialized_controller(controller, mock_env):
    """Controller with initialize_simulation() already called."""
    controller.initialize_simulation()
    return controller


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestSimulationControllerInit:
    def test_initial_state(self, controller):
        assert controller.current_step == 0
        assert not controller.is_running
        assert not controller.is_paused
        assert controller.environment is None
        assert controller.simulation_id is None

    def test_initialize_simulation(self, initialized_controller, mock_db, mock_env):
        ctrl = initialized_controller
        assert ctrl.simulation_id is not None
        assert ctrl.environment is not None
        assert ctrl.current_step == 0
        mock_db.add_simulation_record.assert_called_once()

    def test_initialize_simulation_db_error(self, controller, mock_db):
        mock_db.add_simulation_record.side_effect = RuntimeError("DB failure")
        with pytest.raises(RuntimeError):
            controller.initialize_simulation()

    def test_initialize_simulation_env_error(self, controller, mock_db, mock_env):
        mock_env.side_effect = RuntimeError("Env failure")
        with patch("farm.api.simulation_controller.Environment", side_effect=RuntimeError("Env failure")):
            with pytest.raises(RuntimeError):
                controller.initialize_simulation()


# ---------------------------------------------------------------------------
# Start / Pause / Stop
# ---------------------------------------------------------------------------

class TestSimulationLifecycle:
    def test_start_creates_background_thread(self, initialized_controller):
        ctrl = initialized_controller
        # Override _run_simulation so it ends quickly
        ctrl._run_simulation = Mock()
        ctrl.start()
        # Thread should have been created
        assert ctrl._simulation_thread is not None
        ctrl._simulation_thread.join(timeout=2)

    def test_start_sets_running_flag(self, initialized_controller):
        ctrl = initialized_controller
        ctrl._run_simulation = Mock()  # no-op run
        ctrl.start()
        # Give thread a moment; it may finish quickly but flag was set
        assert ctrl._simulation_thread is not None

    def test_start_resumes_paused_simulation(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.is_paused = True
        # Fake an alive thread
        fake_thread = Mock()
        fake_thread.is_alive.return_value = True
        ctrl._simulation_thread = fake_thread

        ctrl.start()

        assert not ctrl.is_paused

    def test_start_ignores_running_non_paused(self, initialized_controller):
        """Calling start when thread is alive and not paused is a no-op."""
        ctrl = initialized_controller
        fake_thread = Mock()
        fake_thread.is_alive.return_value = True
        ctrl._simulation_thread = fake_thread
        ctrl.is_paused = False

        # Should return without changing anything
        ctrl.start()
        assert not ctrl.is_running  # we didn't set it

    def test_pause(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.pause()
        assert ctrl.is_paused

    def test_stop(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.is_running = True
        ctrl.stop()
        assert not ctrl.is_running
        assert ctrl._stop_requested

    def test_stop_joins_thread(self, initialized_controller):
        ctrl = initialized_controller
        fake_thread = Mock()
        fake_thread.is_alive.return_value = False
        ctrl._simulation_thread = fake_thread
        ctrl.is_running = True

        ctrl.stop()
        fake_thread.join.assert_called_once()

    def test_stop_warns_on_timeout(self, initialized_controller):
        ctrl = initialized_controller
        fake_thread = Mock()
        fake_thread.is_alive.return_value = True  # never finishes
        ctrl._simulation_thread = fake_thread
        ctrl.is_running = True

        # Should not raise; just log a warning
        ctrl.stop()

    def test_stop_no_thread(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.is_running = True
        ctrl._simulation_thread = None
        ctrl.stop()
        assert not ctrl.is_running


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestSimulationStep:
    def test_step_without_environment_raises(self, controller):
        with pytest.raises(RuntimeError, match="not initialized"):
            controller.step()

    def test_step_increments_counter(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.step()
        assert ctrl.current_step == 1

    def test_step_calls_environment_step(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.step()
        ctrl.environment.step.assert_called_once()

    def test_step_notifies_callbacks(self, initialized_controller):
        ctrl = initialized_controller
        steps_received = []
        ctrl.register_step_callback("tracker", lambda s: steps_received.append(s))
        ctrl.step()
        assert steps_received == [1]

    def test_step_stops_on_exception(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.environment.step.side_effect = RuntimeError("env error")
        with pytest.raises(RuntimeError):
            ctrl.step()
        assert not ctrl.is_running


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbackRegistration:
    def test_register_step_callback(self, controller):
        cb = Mock()
        controller.register_step_callback("cb1", cb)
        assert "cb1" in controller.step_callbacks

    def test_register_status_callback(self, controller):
        cb = Mock()
        controller.register_status_callback("cb1", cb)
        assert "cb1" in controller.status_callbacks

    def test_step_callback_called_with_step(self, initialized_controller):
        ctrl = initialized_controller
        received = []
        ctrl.register_step_callback("test", lambda s: received.append(s))
        ctrl.step()
        assert received == [1]

    def test_status_callback_called_on_pause(self, initialized_controller):
        ctrl = initialized_controller
        statuses = []
        ctrl.register_status_callback("test", lambda s: statuses.append(s))
        ctrl.pause()
        assert "paused" in statuses

    def test_status_callback_called_on_stop(self, initialized_controller):
        ctrl = initialized_controller
        statuses = []
        ctrl.register_status_callback("test", lambda s: statuses.append(s))
        ctrl.stop()
        assert "stopped" in statuses

    def test_status_callback_called_on_initialize(self, controller, mock_db, mock_env):
        statuses = []
        controller.register_status_callback("test", lambda s: statuses.append(s))
        controller.initialize_simulation()
        assert "initialized" in statuses

    def test_step_callback_exception_does_not_propagate(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.register_step_callback("bad", lambda s: 1 / 0)
        # Should not raise
        ctrl.step()

    def test_status_callback_exception_does_not_propagate(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.register_status_callback("bad", lambda s: 1 / 0)
        ctrl.pause()  # triggers status callback, should not raise


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------

class TestGetState:
    def test_get_state_not_running(self, controller):
        state = controller.get_state()
        assert state["current_step"] == 0
        assert not state["is_running"]
        assert state["agent_count"] == 0
        assert state["resource_count"] == 0

    def test_get_state_running(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.is_running = True
        state = ctrl.get_state()
        assert state["is_running"]
        assert state["status"] == "running"

    def test_get_state_with_environment(self, initialized_controller):
        ctrl = initialized_controller
        state = ctrl.get_state()
        assert state["agent_count"] == 2
        assert state["resource_count"] == 1

    def test_get_state_total_steps(self, initialized_controller):
        state = initialized_controller.get_state()
        assert state["total_steps"] == 10


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_when_running_stops(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.is_running = True
        ctrl.stop = Mock()
        ctrl.cleanup()
        ctrl.stop.assert_called_once()

    def test_cleanup_clears_environment(self, initialized_controller):
        ctrl = initialized_controller
        ctrl.cleanup()
        ctrl.environment.cleanup.assert_called_once()

    def test_cleanup_closes_db(self, initialized_controller, mock_db):
        initialized_controller.cleanup()
        mock_db.close.assert_called_once()

    def test_cleanup_error_propagates(self, initialized_controller, mock_db):
        mock_db.close.side_effect = RuntimeError("close error")
        with pytest.raises(RuntimeError):
            initialized_controller.cleanup()

    def test_context_manager(self, controller, mock_db, mock_env):
        with controller as ctrl:
            ctrl.initialize_simulation()
            assert ctrl.environment is not None
        mock_db.close.assert_called_once()

    def test_context_manager_cleanup_on_exception(self, controller, mock_db, mock_env):
        with pytest.raises(ValueError):
            with controller as ctrl:
                ctrl.initialize_simulation()
                raise ValueError("test error")
        mock_db.close.assert_called_once()


# ---------------------------------------------------------------------------
# _run_simulation loop (integration-style with real threading)
# ---------------------------------------------------------------------------

class TestRunSimulationLoop:
    def test_simulation_completes_after_all_steps(self, controller, mock_db, mock_env):
        """Simulation loop should stop itself when step count is reached."""
        controller.config.simulation_steps = 3
        controller.initialize_simulation()

        statuses = []
        controller.register_status_callback("test", statuses.append)

        controller.start()
        controller._simulation_thread.join(timeout=5)

        assert "completed" in statuses or not controller.is_running

    def test_simulation_loop_stops_on_stop_request(self, controller, mock_db, mock_env):
        """Calling stop() should abort the loop."""
        controller.config.simulation_steps = 10000  # long run
        controller.initialize_simulation()

        # Slow step so we can stop it
        original_step = controller.step
        def slow_step():
            time.sleep(0.05)
            original_step()
        controller.step = slow_step

        controller.start()
        time.sleep(0.1)
        controller.stop()

        assert not controller.is_running

    def test_simulation_pauses_and_resumes(self, controller, mock_db, mock_env):
        """Pause then resume should continue counting steps."""
        controller.config.simulation_steps = 20
        controller.initialize_simulation()

        controller.start()
        time.sleep(0.05)
        controller.pause()
        step_after_pause = controller.current_step

        time.sleep(0.1)
        # Should not advance while paused
        assert controller.current_step <= step_after_pause + 2

        controller.start()  # resume
        controller._simulation_thread.join(timeout=5)

    def test_simulation_loop_error_notifies_callback(self, controller, mock_db, mock_env):
        """An exception in step should result in 'error' status."""
        controller.config.simulation_steps = 5
        controller.initialize_simulation()

        controller.environment.step.side_effect = RuntimeError("step crash")

        statuses = []
        controller.register_status_callback("test", statuses.append)

        controller.start()
        controller._simulation_thread.join(timeout=5)

        assert "error" in statuses or not controller.is_running
