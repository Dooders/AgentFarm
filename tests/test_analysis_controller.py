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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
