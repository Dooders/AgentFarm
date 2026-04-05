"""
Tests for AnalysisController.

Basic smoke tests to verify controller functionality.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from farm.analysis.service import AnalysisRequest, AnalysisResult
from farm.api.analysis_controller import AnalysisController
from farm.core.services import IConfigService


@pytest.fixture
def mock_config_service():
    """Mock config service."""
    service = Mock(spec=IConfigService)
    return service


@pytest.fixture
def mock_analysis_service():
    """Mock analysis service."""
    with patch('farm.api.analysis_controller.AnalysisService') as mock:
        service = MagicMock()
        mock.return_value = service
        yield service


@pytest.fixture
def controller(mock_config_service, mock_analysis_service):
    """Create controller with mocked dependencies."""
    return AnalysisController(mock_config_service)


def test_controller_initialization(controller):
    """Test controller initializes correctly."""
    assert controller is not None
    assert not controller.is_running
    assert not controller.is_paused
    assert controller.current_progress == 0.0
    assert controller.request is None
    assert controller.result is None


def test_initialize_analysis(controller, mock_analysis_service):
    """Test analysis initialization."""
    # Mock validation
    mock_analysis_service.validate_request = Mock()

    request = AnalysisRequest(
        module_name="test_module",
        experiment_path=Path("/fake/path"),
        output_path=Path("/fake/output"),
        group="all"
    )

    controller.initialize_analysis(request)

    assert controller.request == request
    assert controller.analysis_id is not None
    assert not controller.is_running
    mock_analysis_service.validate_request.assert_called_once()


def test_start_without_initialization_raises_error(controller):
    """Test that starting without initialization raises error."""
    with pytest.raises(RuntimeError, match="not initialized"):
        controller.start()


def test_callbacks_registration(controller):
    """Test callback registration and removal."""
    progress_callback = Mock()
    status_callback = Mock()

    controller.register_progress_callback("test_progress", progress_callback)
    controller.register_status_callback("test_status", status_callback)

    assert "test_progress" in controller.progress_callbacks
    assert "test_status" in controller.status_callbacks

    controller.unregister_progress_callback("test_progress")
    controller.unregister_status_callback("test_status")

    assert "test_progress" not in controller.progress_callbacks
    assert "test_status" not in controller.status_callbacks


def test_get_state(controller, mock_analysis_service):
    """Test get_state returns correct data."""
    mock_analysis_service.validate_request = Mock()

    request = AnalysisRequest(
        module_name="test_module",
        experiment_path=Path("/fake/path"),
        output_path=Path("/fake/output")
    )

    controller.initialize_analysis(request)

    state = controller.get_state()

    assert state["module_name"] == "test_module"
    assert state["is_running"] is False
    assert state["is_paused"] is False
    assert state["progress"] == 0.0
    assert "status" in state


def test_context_manager(mock_config_service, mock_analysis_service):
    """Test controller works as context manager."""
    cleanup_called = False

    with AnalysisController(mock_config_service) as controller:
        assert controller is not None

    # Controller should be cleaned up after exit


def test_list_modules(controller, mock_analysis_service):
    """Test listing available modules."""
    mock_analysis_service.list_modules.return_value = [
        {"name": "module1", "description": "Test module 1"},
        {"name": "module2", "description": "Test module 2"}
    ]

    modules = controller.list_available_modules()

    assert len(modules) == 2
    assert modules[0]["name"] == "module1"
    mock_analysis_service.list_modules.assert_called_once()


def test_get_module_info(controller, mock_analysis_service):
    """Test getting module information."""
    mock_analysis_service.get_module_info.return_value = {
        "name": "test_module",
        "description": "Test description",
        "function_groups": ["all", "plots"],
        "functions": ["func1", "func2"]
    }

    info = controller.get_module_info("test_module")

    assert info["name"] == "test_module"
    assert "function_groups" in info
    mock_analysis_service.get_module_info.assert_called_once_with("test_module")


def test_clear_cache(controller, mock_analysis_service):
    """Test cache clearing."""
    mock_analysis_service.clear_cache.return_value = 5

    cleared = controller.clear_cache()

    assert cleared == 5
    mock_analysis_service.clear_cache.assert_called_once()


def test_progress_handler(controller, mock_analysis_service):
    """Test progress callback handling."""
    mock_analysis_service.validate_request = Mock()

    # Register progress callback
    callback_data = []

    def progress_callback(message, progress):
        callback_data.append((message, progress))

    controller.register_progress_callback("test", progress_callback)

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out")
    )

    controller.initialize_analysis(request)

    # Simulate progress update
    controller._progress_handler("Test message", 0.5)

    assert len(callback_data) == 1
    assert callback_data[0][0] == "Test message"
    assert callback_data[0][1] == 0.5
    assert controller.current_progress == 0.5
    assert controller.current_message == "Test message"


def test_progress_handler_with_stop_requested(controller, mock_analysis_service):
    """Test progress handler respects stop request."""
    mock_analysis_service.validate_request = Mock()

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out")
    )

    controller.initialize_analysis(request)
    controller._stop_requested = True

    with pytest.raises(InterruptedError, match="stop requested"):
        controller._progress_handler("Test", 0.5)


def test_pause_and_resume(controller, mock_analysis_service):
    """Test pause and resume functionality."""
    mock_analysis_service.validate_request = Mock()

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out")
    )

    controller.initialize_analysis(request)

    # Test pause
    controller.pause()
    assert controller.is_paused is True

    # Start should set paused to False if thread already running
    # (in this test, thread isn't really running, so just test the flag)


def test_stop(controller, mock_analysis_service):
    """Test stop functionality."""
    mock_analysis_service.validate_request = Mock()

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out")
    )

    controller.initialize_analysis(request)
    controller.stop()

    assert controller.is_running is False
    assert controller.is_paused is False
    assert controller._stop_requested is True


def test_get_result_before_completion(controller):
    """Test get_result returns None before completion."""
    result = controller.get_result()
    assert result is None


def test_status_callbacks_called(controller, mock_analysis_service):
    """Test status callbacks are invoked."""
    mock_analysis_service.validate_request = Mock()

    status_changes = []

    def status_callback(status):
        status_changes.append(status)

    controller.register_status_callback("test", status_callback)

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out")
    )

    controller.initialize_analysis(request)

    # Should have received "initialized" status
    assert "initialized" in status_changes


def test_wait_for_completion_no_thread(controller):
    """Test wait_for_completion returns immediately when no thread exists."""
    result = controller.wait_for_completion(timeout=1.0)
    assert result is True


def test_wait_for_completion_with_timeout(controller, mock_analysis_service):
    """Test wait_for_completion respects timeout."""
    mock_analysis_service.validate_request = Mock()

    # Create a mock thread that never completes
    mock_thread = Mock()
    mock_thread.is_alive.return_value = True
    controller._analysis_thread = mock_thread

    # Should timeout and return False
    result = controller.wait_for_completion(timeout=0.1)

    # Thread join should have been called with timeout
    mock_thread.join.assert_called_once_with(timeout=0.1)
    assert result is False


def test_wait_for_completion_success(controller, mock_analysis_service):
    """Test wait_for_completion returns True when thread completes."""
    mock_analysis_service.validate_request = Mock()

    # Create a mock thread that completes
    mock_thread = Mock()
    mock_thread.is_alive.return_value = False
    controller._analysis_thread = mock_thread

    result = controller.wait_for_completion(timeout=1.0)

    # When wait_for_completion succeeds immediately, join() should not be called
    mock_thread.join.assert_not_called()
    assert result is True


def test_get_state_with_completed_result(controller, mock_analysis_service):
    """Test get_state includes result info when completed."""
    from farm.analysis.service import AnalysisResult

    mock_analysis_service.validate_request = Mock()

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out")
    )

    controller.initialize_analysis(request)

    # Set a completed result
    controller.result = AnalysisResult(
        success=True,
        module_name="test",
        output_path=Path("/fake/out"),
        execution_time=1.5,
        cache_hit=False
    )

    state = controller.get_state()

    assert state["status"] == "completed"
    assert state["cache_hit"] is False
    assert state["execution_time"] == 1.5


def test_get_state_with_error_result(controller, mock_analysis_service):
    """Test get_state includes error info when failed."""
    from farm.analysis.service import AnalysisResult

    mock_analysis_service.validate_request = Mock()

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out")
    )

    controller.initialize_analysis(request)

    # Set a failed result
    controller.result = AnalysisResult(
        success=False,
        module_name="test",
        output_path=Path("/fake/out"),
        error="Test error message"
    )

    state = controller.get_state()

    assert state["status"] == "error"
    assert state["error"] == "Test error message"


def test_cleanup_multiple_times(controller):
    """Test cleanup can be called multiple times safely."""
    # Should not raise exception
    controller.cleanup()
    controller.cleanup()
    controller.cleanup()


def test_del_calls_cleanup(mock_config_service, mock_analysis_service):
    """Test __del__ attempts cleanup."""
    controller = AnalysisController(mock_config_service)

    # Mock cleanup to track if it's called
    controller.cleanup = Mock()

    # Trigger __del__
    del controller

    # Note: __del__ may not be called immediately, this is just a smoke test


def test_progress_handler_with_pause(controller, mock_analysis_service):
    """Test progress handler respects pause state."""
    import time
    import threading

    mock_analysis_service.validate_request = Mock()

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out")
    )

    controller.initialize_analysis(request)
    controller.is_paused = True

    # Start a thread to unpause after short delay
    def unpause():
        time.sleep(0.1)
        controller.is_paused = False

    threading.Thread(target=unpause, daemon=True).start()

    # This should wait until unpaused
    start = time.time()
    controller._progress_handler("Test", 0.5)
    elapsed = time.time() - start

    # Should have waited at least 0.1 seconds
    assert elapsed >= 0.1


# ---------------------------------------------------------------------------
# Additional coverage for start() paths, _run_analysis(), stop() with thread,
# callback error handling, get_state paused/running, and cleanup/del.
# ---------------------------------------------------------------------------


def test_start_resumes_paused_thread(mock_config_service, mock_analysis_service):
    """start() resumes execution when thread alive and paused."""
    mock_analysis_service.validate_request = Mock()
    controller = AnalysisController(mock_config_service)

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)
    controller.is_paused = True

    # Fake an alive thread
    fake_thread = Mock()
    fake_thread.is_alive.return_value = True
    controller._analysis_thread = fake_thread

    statuses = []
    controller.register_status_callback("test", statuses.append)
    controller.start()

    assert not controller.is_paused
    assert "resumed" in statuses


def test_start_no_op_when_thread_alive_not_paused(mock_config_service, mock_analysis_service):
    """start() is a no-op when thread is alive and not paused."""
    mock_analysis_service.validate_request = Mock()
    controller = AnalysisController(mock_config_service)

    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)

    fake_thread = Mock()
    fake_thread.is_alive.return_value = True
    controller._analysis_thread = fake_thread
    controller.is_paused = False

    # Should not raise or change state
    controller.start()


def test_start_spawns_thread_and_runs_analysis(mock_config_service, mock_analysis_service):
    """start() creates a real thread that calls _run_analysis."""
    from farm.analysis.service import AnalysisResult
    from pathlib import Path as LibPath

    mock_analysis_service.validate_request = Mock()
    mock_result = Mock(spec=AnalysisResult)
    mock_result.success = True
    mock_result.output_path = LibPath("/fake/out")
    mock_result.execution_time = 0.1
    mock_result.cache_hit = False
    mock_analysis_service.run.return_value = mock_result

    controller = AnalysisController(mock_config_service)
    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)

    statuses = []
    controller.register_status_callback("test", statuses.append)
    controller.start()
    controller.wait_for_completion(timeout=5.0)

    assert "completed" in statuses
    assert not controller.is_running
    assert controller.current_progress == 1.0


def test_run_analysis_failure(mock_config_service, mock_analysis_service):
    """_run_analysis notifies 'error' status when result.success is False."""
    from farm.analysis.service import AnalysisResult

    mock_analysis_service.validate_request = Mock()
    mock_result = Mock(spec=AnalysisResult)
    mock_result.success = False
    mock_result.error = "Analysis module failed"
    mock_analysis_service.run.return_value = mock_result

    controller = AnalysisController(mock_config_service)
    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)

    statuses = []
    controller.register_status_callback("test", statuses.append)
    controller.start()
    controller.wait_for_completion(timeout=5.0)

    assert "error" in statuses


def test_run_analysis_exception(mock_config_service, mock_analysis_service):
    """_run_analysis handles unexpected exceptions gracefully."""
    mock_analysis_service.validate_request = Mock()
    mock_analysis_service.run.side_effect = RuntimeError("unexpected crash")

    controller = AnalysisController(mock_config_service)
    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)

    statuses = []
    controller.register_status_callback("test", statuses.append)
    controller.start()
    controller._analysis_thread.join(timeout=5.0)

    assert "error" in statuses
    assert not controller.is_running


def test_stop_joins_thread(mock_config_service, mock_analysis_service):
    """stop() joins the analysis thread when called from a different thread."""
    mock_analysis_service.validate_request = Mock()

    controller = AnalysisController(mock_config_service)
    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)

    fake_thread = Mock()
    fake_thread.is_alive.return_value = False
    controller._analysis_thread = fake_thread

    controller.stop()
    fake_thread.join.assert_called_once()


def test_stop_warns_on_thread_timeout(mock_config_service, mock_analysis_service):
    """stop() logs a warning when thread doesn't terminate within timeout."""
    mock_analysis_service.validate_request = Mock()

    controller = AnalysisController(mock_config_service)
    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)

    fake_thread = Mock()
    fake_thread.is_alive.return_value = True  # never finishes
    controller._analysis_thread = fake_thread

    # Should not raise
    controller.stop()


def test_notify_progress_callback_exception_does_not_propagate(
    controller, mock_analysis_service
):
    """Progress callback exceptions are caught and logged."""
    mock_analysis_service.validate_request = Mock()

    controller.register_progress_callback("bad", lambda msg, p: 1 / 0)
    # Should not raise
    controller._notify_progress("msg", 0.5)


def test_notify_status_callback_exception_does_not_propagate(
    controller, mock_analysis_service
):
    """Status callback exceptions are caught and logged."""
    controller.register_status_callback("bad", lambda s: 1 / 0)
    # Should not raise
    controller._notify_status_change("some_status")


def test_get_state_paused(controller, mock_analysis_service):
    """get_state returns 'paused' status when is_paused is True."""
    mock_analysis_service.validate_request = Mock()
    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)
    controller.is_paused = True
    controller.result = None

    state = controller.get_state()
    assert state["status"] == "paused"


def test_get_state_running(controller, mock_analysis_service):
    """get_state returns 'running' status when is_running is True."""
    mock_analysis_service.validate_request = Mock()
    request = AnalysisRequest(
        module_name="test",
        experiment_path=Path("/fake"),
        output_path=Path("/fake/out"),
    )
    controller.initialize_analysis(request)
    controller.is_running = True
    controller.is_paused = False
    controller.result = None

    state = controller.get_state()
    assert state["status"] == "running"


def test_cleanup_when_running_stops(mock_config_service, mock_analysis_service):
    """cleanup() stops the controller when it is currently running."""
    mock_analysis_service.validate_request = Mock()

    controller = AnalysisController(mock_config_service)
    controller.is_running = True

    stop_called = []
    original_stop = controller.stop
    controller.stop = lambda: stop_called.append(True) or original_stop()

    controller.cleanup()

    assert stop_called


def test_cleanup_raises_on_error(mock_config_service, mock_analysis_service):
    """cleanup() re-raises exceptions."""
    mock_analysis_service.validate_request = Mock()

    controller = AnalysisController(mock_config_service)
    controller.is_running = True
    controller.stop = Mock(side_effect=RuntimeError("stop failed"))

    with pytest.raises(RuntimeError, match="stop failed"):
        controller.cleanup()


def test_del_suppresses_exception(mock_config_service, mock_analysis_service):
    """__del__ swallows exceptions raised during cleanup."""
    controller = AnalysisController(mock_config_service)
    controller.cleanup = Mock(side_effect=RuntimeError("cleanup failed"))
    # Should not raise
    controller.__del__()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
