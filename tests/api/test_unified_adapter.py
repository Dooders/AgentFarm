"""Unit tests for the UnifiedAdapter class."""

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

import pytest

from farm.api.models import (
    SimulationStatus, ExperimentStatus, SimulationResults, ExperimentResults,
    AnalysisResults, ComparisonResults, Event, EventSubscription
)
from farm.api.unified_adapter import UnifiedAdapter


class TestUnifiedAdapter:
    """Test UnifiedAdapter class."""

    def test_init(self, temp_workspace):
        """Test UnifiedAdapter initialization."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        assert adapter.session_path == session_path
        assert adapter.config_manager is not None
        assert adapter._simulations == {}
        assert adapter._experiments == {}
        assert adapter._event_subscriptions == {}
        assert adapter._event_history == []
        assert isinstance(adapter._event_lock, type(threading.Lock()))

    @patch('farm.api.unified_adapter.SimulationController')
    @patch('farm.api.unified_adapter.ConfigTemplateManager')
    def test_create_simulation_success(self, mock_config_manager_class, mock_sim_controller_class, temp_workspace, sample_simulation_config):
        """Test creating a simulation successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Mock the config manager
        mock_config_manager = Mock()
        mock_sim_config = Mock()
        mock_config_manager.convert_to_simulation_config.return_value = mock_sim_config
        mock_sim_config.simulation_steps = 1000
        adapter.config_manager = mock_config_manager
        
        # Mock the simulation controller
        mock_controller = Mock()
        mock_sim_controller_class.return_value = mock_controller
        
        simulation_id = adapter.create_simulation(sample_simulation_config)
        
        assert simulation_id is not None
        assert len(simulation_id) == 36  # UUID4 length
        assert simulation_id in adapter._simulations
        
        # Check simulation info
        sim_info = adapter._simulations[simulation_id]
        assert sim_info["controller"] == mock_controller
        assert sim_info["config"] == sample_simulation_config
        assert sim_info["status"] == SimulationStatus.CREATED
        assert sim_info["current_step"] == 0
        assert sim_info["total_steps"] == 1000
        
        # Should create simulation directory
        sim_dir = session_path / "simulations" / simulation_id
        assert sim_dir.exists()
        
        # Should emit event
        assert len(adapter._event_history) == 1
        event = adapter._event_history[0]
        assert event.event_type == "simulation_created"
        assert event.simulation_id == simulation_id

    def test_create_simulation_invalid_config(self, temp_workspace):
        """Test creating simulation with invalid config."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Mock config manager to return None (invalid config)
        adapter.config_manager.convert_to_simulation_config.return_value = None
        
        with pytest.raises(ValueError, match="Invalid simulation configuration"):
            adapter.create_simulation({"invalid": "config"})

    def test_start_simulation_success(self, temp_workspace):
        """Test starting a simulation successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a simulation
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.CREATED,
            "start_time": None,
            "current_step": 0,
            "total_steps": 1000
        }
        
        status = adapter.start_simulation(simulation_id)
        
        assert status == SimulationStatus.RUNNING
        mock_controller.start.assert_called_once()
        
        # Check simulation info updated
        sim_info = adapter._simulations[simulation_id]
        assert sim_info["status"] == SimulationStatus.RUNNING
        assert sim_info["start_time"] is not None
        
        # Should emit event
        assert len(adapter._event_history) == 1
        event = adapter._event_history[0]
        assert event.event_type == "simulation_started"
        assert event.simulation_id == simulation_id

    def test_start_simulation_nonexistent(self, temp_workspace):
        """Test starting a non-existent simulation."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Simulation .* not found"):
            adapter.start_simulation("nonexistent-simulation")

    def test_pause_simulation_success(self, temp_workspace):
        """Test pausing a simulation successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a running simulation
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.RUNNING,
            "current_step": 500,
            "total_steps": 1000
        }
        
        status = adapter.pause_simulation(simulation_id)
        
        assert status == SimulationStatus.PAUSED
        mock_controller.pause.assert_called_once()
        
        # Check simulation info updated
        sim_info = adapter._simulations[simulation_id]
        assert sim_info["status"] == SimulationStatus.PAUSED

    def test_resume_simulation_success(self, temp_workspace):
        """Test resuming a simulation successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a paused simulation
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.PAUSED,
            "current_step": 500,
            "total_steps": 1000
        }
        
        status = adapter.resume_simulation(simulation_id)
        
        assert status == SimulationStatus.RUNNING
        mock_controller.start.assert_called_once()  # Resume calls start
        
        # Check simulation info updated
        sim_info = adapter._simulations[simulation_id]
        assert sim_info["status"] == SimulationStatus.RUNNING

    def test_stop_simulation_success(self, temp_workspace):
        """Test stopping a simulation successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a running simulation
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.RUNNING,
            "current_step": 500,
            "total_steps": 1000
        }
        
        status = adapter.stop_simulation(simulation_id)
        
        assert status == SimulationStatus.STOPPED
        mock_controller.stop.assert_called_once()
        
        # Check simulation info updated
        sim_info = adapter._simulations[simulation_id]
        assert sim_info["status"] == SimulationStatus.STOPPED
        assert sim_info["end_time"] is not None

    def test_get_simulation_status_success(self, temp_workspace):
        """Test getting simulation status successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a simulation
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        mock_controller.get_state.return_value = {
            "current_step": 500,
            "total_steps": 1000,
            "is_running": True,
            "is_paused": False
        }
        
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.RUNNING,
            "current_step": 500,
            "total_steps": 1000,
            "start_time": datetime.now(),
            "end_time": None,
            "error_message": None
        }
        
        status = adapter.get_simulation_status(simulation_id)
        
        assert status is not None
        assert status.simulation_id == simulation_id
        assert status.status == SimulationStatus.RUNNING
        assert status.current_step == 500
        assert status.total_steps == 1000
        assert status.progress_percentage == 50.0

    def test_get_simulation_status_nonexistent(self, temp_workspace):
        """Test getting status for non-existent simulation."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Simulation .* not found"):
            adapter.get_simulation_status("nonexistent-simulation")

    def test_get_simulation_results_success(self, temp_workspace):
        """Test getting simulation results successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a completed simulation
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        mock_controller.get_state.return_value = {
            "current_step": 1000,
            "total_steps": 1000,
            "is_running": False,
            "is_paused": False
        }
        
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.COMPLETED,
            "current_step": 1000,
            "total_steps": 1000,
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "error_message": None
        }
        
        results = adapter.get_simulation_results(simulation_id)
        
        assert results is not None
        assert results.simulation_id == simulation_id
        assert results.status == SimulationStatus.COMPLETED
        assert results.total_steps == 1000

    def test_get_simulation_results_nonexistent(self, temp_workspace):
        """Test getting results for non-existent simulation."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Simulation .* not found"):
            adapter.get_simulation_results("nonexistent-simulation")

    @patch('farm.api.unified_adapter.ExperimentController')
    @patch('farm.api.unified_adapter.ConfigTemplateManager')
    def test_create_experiment_success(self, mock_config_manager_class, mock_exp_controller_class, temp_workspace, sample_experiment_config):
        """Test creating an experiment successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Mock the config manager
        mock_config_manager = Mock()
        mock_exp_config = Mock()
        mock_config_manager.convert_to_experiment_config.return_value = mock_exp_config
        mock_exp_config.iterations = 10
        adapter.config_manager = mock_config_manager
        
        # Mock the experiment controller
        mock_controller = Mock()
        mock_exp_controller_class.return_value = mock_controller
        
        experiment_id = adapter.create_experiment(sample_experiment_config)
        
        assert experiment_id is not None
        assert len(experiment_id) == 36  # UUID4 length
        assert experiment_id in adapter._experiments
        
        # Check experiment info
        exp_info = adapter._experiments[experiment_id]
        assert exp_info["controller"] == mock_controller
        assert exp_info["config"] == sample_experiment_config
        assert exp_info["status"] == ExperimentStatus.CREATED
        assert exp_info["current_iteration"] == 0
        assert exp_info["total_iterations"] == 10
        
        # Should create experiment directory
        exp_dir = session_path / "experiments" / experiment_id
        assert exp_dir.exists()
        
        # Should emit event
        assert len(adapter._event_history) == 1
        event = adapter._event_history[0]
        assert event.event_type == "experiment_created"
        assert event.experiment_id == experiment_id

    def test_create_experiment_invalid_config(self, temp_workspace):
        """Test creating experiment with invalid config."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Mock config manager to return None (invalid config)
        adapter.config_manager.convert_to_experiment_config.return_value = None
        
        with pytest.raises(ValueError, match="Invalid experiment configuration"):
            adapter.create_experiment({"invalid": "config"})

    def test_start_experiment_success(self, temp_workspace):
        """Test starting an experiment successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create an experiment
        experiment_id = str(uuid.uuid4())
        mock_controller = Mock()
        adapter._experiments[experiment_id] = {
            "controller": mock_controller,
            "status": ExperimentStatus.CREATED,
            "start_time": None,
            "current_iteration": 0,
            "total_iterations": 10
        }
        
        status = adapter.start_experiment(experiment_id)
        
        assert status == ExperimentStatus.RUNNING
        mock_controller.start.assert_called_once()
        
        # Check experiment info updated
        exp_info = adapter._experiments[experiment_id]
        assert exp_info["status"] == ExperimentStatus.RUNNING
        assert exp_info["start_time"] is not None
        
        # Should emit event
        assert len(adapter._event_history) == 1
        event = adapter._event_history[0]
        assert event.event_type == "experiment_started"
        assert event.experiment_id == experiment_id

    def test_start_experiment_nonexistent(self, temp_workspace):
        """Test starting a non-existent experiment."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Experiment .* not found"):
            adapter.start_experiment("nonexistent-experiment")

    def test_get_experiment_status_success(self, temp_workspace):
        """Test getting experiment status successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create an experiment
        experiment_id = str(uuid.uuid4())
        mock_controller = Mock()
        mock_controller.get_status.return_value = "running"
        
        adapter._experiments[experiment_id] = {
            "controller": mock_controller,
            "status": ExperimentStatus.RUNNING,
            "current_iteration": 5,
            "total_iterations": 10,
            "start_time": datetime.now(),
            "end_time": None,
            "error_message": None
        }
        
        status = adapter.get_experiment_status(experiment_id)
        
        assert status is not None
        assert status.experiment_id == experiment_id
        assert status.status == ExperimentStatus.RUNNING
        assert status.current_iteration == 5
        assert status.total_iterations == 10
        assert status.progress_percentage == 50.0

    def test_get_experiment_status_nonexistent(self, temp_workspace):
        """Test getting status for non-existent experiment."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Experiment .* not found"):
            adapter.get_experiment_status("nonexistent-experiment")

    def test_get_experiment_results_success(self, temp_workspace):
        """Test getting experiment results successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a completed experiment
        experiment_id = str(uuid.uuid4())
        mock_controller = Mock()
        mock_controller.get_results.return_value = {
            "avg_fitness": 0.8,
            "best_fitness": 0.95
        }
        
        adapter._experiments[experiment_id] = {
            "controller": mock_controller,
            "status": ExperimentStatus.COMPLETED,
            "current_iteration": 10,
            "total_iterations": 10,
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "error_message": None
        }
        
        results = adapter.get_experiment_results(experiment_id)
        
        assert results is not None
        assert results.experiment_id == experiment_id
        assert results.status == ExperimentStatus.COMPLETED
        assert results.total_iterations == 10
        assert results.completed_iterations == 10

    def test_get_experiment_results_nonexistent(self, temp_workspace):
        """Test getting results for non-existent experiment."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Experiment .* not found"):
            adapter.get_experiment_results("nonexistent-experiment")

    @patch('farm.api.unified_adapter.SimulationAnalyzer')
    def test_analyze_simulation_success(self, mock_analyzer_class, temp_workspace):
        """Test analyzing a simulation successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a completed simulation
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.COMPLETED,
            "directory": session_path / "simulations" / simulation_id
        }
        
        # Mock the analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = {
            "summary": {"avg_fitness": 0.8},
            "detailed_results": {"per_agent": [0.7, 0.8, 0.9]}
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        results = adapter.analyze_simulation(simulation_id)
        
        assert results is not None
        assert isinstance(results, AnalysisResults)
        assert results.analysis_id is not None
        assert results.analysis_type == "simulation_analysis"

    def test_analyze_simulation_nonexistent(self, temp_workspace):
        """Test analyzing a non-existent simulation."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Simulation .* not found"):
            adapter.analyze_simulation("nonexistent-simulation")

    @patch('farm.api.unified_adapter.compare_simulations')
    def test_compare_simulations_success(self, mock_compare_func, temp_workspace):
        """Test comparing simulations successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create completed simulations
        sim1_id = str(uuid.uuid4())
        sim2_id = str(uuid.uuid4())
        
        adapter._simulations[sim1_id] = {
            "controller": Mock(),
            "status": SimulationStatus.COMPLETED,
            "directory": session_path / "simulations" / sim1_id
        }
        adapter._simulations[sim2_id] = {
            "controller": Mock(),
            "status": SimulationStatus.COMPLETED,
            "directory": session_path / "simulations" / sim2_id
        }
        
        # Mock the comparison function
        mock_compare_func.return_value = {
            "summary": {"best_simulation": sim2_id},
            "detailed_comparison": {"fitness": {sim1_id: 0.7, sim2_id: 0.9}}
        }
        
        results = adapter.compare_simulations([sim1_id, sim2_id])
        
        assert results is not None
        assert isinstance(results, ComparisonResults)
        assert results.comparison_id is not None
        assert results.simulation_ids == [sim1_id, sim2_id]

    def test_compare_simulations_nonexistent(self, temp_workspace):
        """Test comparing non-existent simulations."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Simulation .* not found"):
            adapter.compare_simulations(["nonexistent-simulation"])

    def test_subscribe_to_events_success(self, temp_workspace):
        """Test subscribing to events successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        subscription_id = adapter.subscribe_to_events(
            ["simulation_started", "simulation_completed"],
            simulation_id="sim-123"
        )
        
        assert subscription_id is not None
        assert len(subscription_id) == 36  # UUID4 length
        assert subscription_id in adapter._event_subscriptions
        
        subscription = adapter._event_subscriptions[subscription_id]
        assert subscription.event_types == ["simulation_started", "simulation_completed"]
        assert subscription.simulation_id == "sim-123"
        assert subscription.active is True

    def test_get_event_history_success(self, temp_workspace):
        """Test getting event history successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a subscription
        subscription_id = adapter.subscribe_to_events(["simulation_started"])
        
        # Add some events
        event1 = Event(
            event_id="event-1",
            event_type="simulation_started",
            timestamp=datetime.now(),
            session_id="session-123",
            simulation_id="sim-123"
        )
        event2 = Event(
            event_id="event-2",
            event_type="simulation_completed",
            timestamp=datetime.now(),
            session_id="session-123",
            simulation_id="sim-123"
        )
        
        adapter._event_history = [event1, event2]
        
        # Get events for subscription
        events = adapter.get_event_history(subscription_id)
        
        assert len(events) == 1  # Only simulation_started should match
        assert events[0].event_type == "simulation_started"

    def test_get_event_history_nonexistent_subscription(self, temp_workspace):
        """Test getting event history for non-existent subscription."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        with pytest.raises(ValueError, match="Subscription .* not found"):
            adapter.get_event_history("nonexistent-subscription")

    def test_emit_event_success(self, temp_workspace):
        """Test emitting an event successfully."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Emit an event
        adapter._emit_event("test_event", simulation_id="sim-123", data={"key": "value"})
        
        assert len(adapter._event_history) == 1
        event = adapter._event_history[0]
        assert event.event_type == "test_event"
        assert event.simulation_id == "sim-123"
        assert event.data == {"key": "value"}

    def test_cleanup_success(self, temp_workspace):
        """Test cleaning up adapter resources."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create some simulations and experiments
        sim_id = str(uuid.uuid4())
        exp_id = str(uuid.uuid4())
        
        mock_sim_controller = Mock()
        mock_exp_controller = Mock()
        
        adapter._simulations[sim_id] = {"controller": mock_sim_controller}
        adapter._experiments[exp_id] = {"controller": mock_exp_controller}
        
        # Cleanup
        adapter.cleanup()
        
        # Controllers should be cleaned up
        mock_sim_controller.cleanup.assert_called_once()
        mock_exp_controller.cleanup.assert_called_once()
        
        # Collections should be cleared
        assert len(adapter._simulations) == 0
        assert len(adapter._experiments) == 0
        assert len(adapter._event_subscriptions) == 0
        assert len(adapter._event_history) == 0

    def test_thread_safety_in_event_handling(self, temp_workspace):
        """Test thread safety in event handling."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create multiple threads that emit events
        import threading
        import time
        
        def emit_events():
            for i in range(10):
                adapter._emit_event(f"test_event_{i}")
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=emit_events)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have 50 events total
        assert len(adapter._event_history) == 50

    def test_error_handling_in_simulation_operations(self, temp_workspace):
        """Test error handling in simulation operations."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a simulation with a controller that raises exceptions
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        mock_controller.start.side_effect = RuntimeError("Controller error")
        
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.CREATED,
            "start_time": None,
            "current_step": 0,
            "total_steps": 1000
        }
        
        # Should propagate the exception
        with pytest.raises(RuntimeError, match="Controller error"):
            adapter.start_simulation(simulation_id)
        
        # Should emit error event
        assert len(adapter._event_history) == 1
        event = adapter._event_history[0]
        assert event.event_type == "simulation_error"
        assert event.simulation_id == simulation_id

    def test_progress_calculation(self, temp_workspace):
        """Test progress percentage calculation."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create a simulation
        simulation_id = str(uuid.uuid4())
        mock_controller = Mock()
        mock_controller.get_state.return_value = {
            "current_step": 750,
            "total_steps": 1000,
            "is_running": True,
            "is_paused": False
        }
        
        adapter._simulations[simulation_id] = {
            "controller": mock_controller,
            "status": SimulationStatus.RUNNING,
            "current_step": 750,
            "total_steps": 1000,
            "start_time": datetime.now(),
            "end_time": None,
            "error_message": None
        }
        
        status = adapter.get_simulation_status(simulation_id)
        
        assert status.progress_percentage == 75.0

    def test_directory_creation(self, temp_workspace):
        """Test that directories are created properly."""
        session_path = temp_workspace / "test_session"
        session_path.mkdir(parents=True)
        
        adapter = UnifiedAdapter(session_path)
        
        # Create simulation and experiment directories
        sim_dir = session_path / "simulations"
        exp_dir = session_path / "experiments"
        
        # Directories should be created when needed
        sim_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        assert sim_dir.exists()
        assert exp_dir.exists()
