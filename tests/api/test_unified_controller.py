"""Unit tests for the AgentFarmController class."""

from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

import pytest

from farm.api.models import (
    SessionInfo, SessionStatus, SimulationStatus, ExperimentStatus,
    SimulationResults, ExperimentResults, AnalysisResults, ComparisonResults,
    ConfigTemplate, ValidationResult, Event
)
from farm.api.unified_controller import AgentFarmController


class TestAgentFarmController:
    """Test AgentFarmController class."""

    def test_init_with_default_workspace(self, temp_workspace):
        """Test AgentFarmController initialization with default workspace."""
        with patch('farm.api.unified_controller.Path') as mock_path:
            mock_path.return_value = temp_workspace
            mock_path.return_value.mkdir = Mock()
            
            controller = AgentFarmController()
            
            # Should create workspace directory
            mock_path.return_value.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            assert controller.workspace_path == mock_path.return_value
            assert controller.session_manager is not None
            assert controller.config_manager is not None
            assert controller._adapters == {}

    def test_init_with_custom_workspace(self, temp_workspace):
        """Test AgentFarmController initialization with custom workspace."""
        custom_path = str(temp_workspace / "custom_workspace")
        
        controller = AgentFarmController(custom_path)
        
        assert controller.workspace_path == Path(custom_path)
        assert controller.workspace_path.exists()

    def test_get_adapter_existing(self, temp_workspace):
        """Test getting an existing adapter."""
        controller = AgentFarmController(str(temp_workspace))
        
        # Create a session first
        session_id = controller.create_session("Test Session")
        
        # Get adapter (should create one)
        adapter = controller._get_adapter(session_id)
        
        assert adapter is not None
        assert session_id in controller._adapters

    def test_get_adapter_nonexistent_session(self, temp_workspace):
        """Test getting adapter for non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        with pytest.raises(ValueError, match="Session .* not found"):
            controller._get_adapter("nonexistent-session")

    def test_create_session_success(self, temp_workspace):
        """Test creating a session successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session", "Test Description")
        
        assert session_id is not None
        assert len(session_id) == 36  # UUID4 length
        
        # Should be able to retrieve the session
        session = controller.get_session(session_id)
        assert session is not None
        assert session.name == "Test Session"
        assert session.description == "Test Description"
        assert session.status == SessionStatus.ACTIVE

    def test_get_session_existing(self, temp_workspace):
        """Test getting an existing session."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        session = controller.get_session(session_id)
        
        assert session is not None
        assert session.session_id == session_id
        assert session.name == "Test Session"

    def test_get_session_nonexistent(self, temp_workspace):
        """Test getting a non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        session = controller.get_session("nonexistent-session")
        assert session is None

    def test_list_sessions(self, temp_workspace):
        """Test listing all sessions."""
        controller = AgentFarmController(str(temp_workspace))
        
        # Create multiple sessions
        session1_id = controller.create_session("Session 1")
        session2_id = controller.create_session("Session 2")
        
        sessions = controller.list_sessions()
        
        assert len(sessions) == 2
        session_ids = [s.session_id for s in sessions]
        assert session1_id in session_ids
        assert session2_id in session_ids

    def test_delete_session_success(self, temp_workspace):
        """Test deleting a session successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Create an adapter for the session
        controller._get_adapter(session_id)
        assert session_id in controller._adapters
        
        success = controller.delete_session(session_id, delete_files=True)
        
        assert success is True
        assert session_id not in controller._adapters
        
        # Session should no longer exist
        session = controller.get_session(session_id)
        assert session is None

    def test_delete_session_nonexistent(self, temp_workspace):
        """Test deleting a non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        success = controller.delete_session("nonexistent-session")
        assert success is False

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_create_simulation_success(self, mock_adapter_class, temp_workspace, sample_simulation_config):
        """Test creating a simulation successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        # Create a session
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.create_simulation.return_value = "sim-123"
        mock_adapter_class.return_value = mock_adapter
        
        simulation_id = controller.create_simulation(session_id, sample_simulation_config)
        
        assert simulation_id == "sim-123"
        mock_adapter.create_simulation.assert_called_once_with(sample_simulation_config)
        
        # Should be added to session
        session = controller.get_session(session_id)
        assert "sim-123" in session.simulations

    def test_create_simulation_nonexistent_session(self, temp_workspace, sample_simulation_config):
        """Test creating simulation for non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        with pytest.raises(ValueError, match="Session .* not found"):
            controller.create_simulation("nonexistent-session", sample_simulation_config)

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_start_simulation_success(self, mock_adapter_class, temp_workspace):
        """Test starting a simulation successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.start_simulation.return_value = SimulationStatus.RUNNING
        mock_adapter_class.return_value = mock_adapter
        
        status = controller.start_simulation(session_id, "sim-123")
        
        assert status == SimulationStatus.RUNNING
        mock_adapter.start_simulation.assert_called_once_with("sim-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_pause_simulation_success(self, mock_adapter_class, temp_workspace):
        """Test pausing a simulation successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.pause_simulation.return_value = SimulationStatus.PAUSED
        mock_adapter_class.return_value = mock_adapter
        
        status = controller.pause_simulation(session_id, "sim-123")
        
        assert status == SimulationStatus.PAUSED
        mock_adapter.pause_simulation.assert_called_once_with("sim-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_resume_simulation_success(self, mock_adapter_class, temp_workspace):
        """Test resuming a simulation successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.resume_simulation.return_value = SimulationStatus.RUNNING
        mock_adapter_class.return_value = mock_adapter
        
        status = controller.resume_simulation(session_id, "sim-123")
        
        assert status == SimulationStatus.RUNNING
        mock_adapter.resume_simulation.assert_called_once_with("sim-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_stop_simulation_success(self, mock_adapter_class, temp_workspace):
        """Test stopping a simulation successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.stop_simulation.return_value = SimulationStatus.STOPPED
        mock_adapter_class.return_value = mock_adapter
        
        status = controller.stop_simulation(session_id, "sim-123")
        
        assert status == SimulationStatus.STOPPED
        mock_adapter.stop_simulation.assert_called_once_with("sim-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_get_simulation_status_success(self, mock_adapter_class, temp_workspace):
        """Test getting simulation status successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.get_simulation_status.return_value = SimulationStatus.RUNNING
        mock_adapter_class.return_value = mock_adapter
        
        status = controller.get_simulation_status(session_id, "sim-123")
        
        assert status == SimulationStatus.RUNNING
        mock_adapter.get_simulation_status.assert_called_once_with("sim-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_get_simulation_results_success(self, mock_adapter_class, temp_workspace):
        """Test getting simulation results successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_results = SimulationResults(
            simulation_id="sim-123",
            status=SimulationStatus.COMPLETED,
            total_steps=1000,
            final_agent_count=20,
            final_resource_count=50
        )
        mock_adapter.get_simulation_results.return_value = mock_results
        mock_adapter_class.return_value = mock_adapter
        
        results = controller.get_simulation_results(session_id, "sim-123")
        
        assert results == mock_results
        mock_adapter.get_simulation_results.assert_called_once_with("sim-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_create_experiment_success(self, mock_adapter_class, temp_workspace, sample_experiment_config):
        """Test creating an experiment successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        # Create a session
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.create_experiment.return_value = "exp-123"
        mock_adapter_class.return_value = mock_adapter
        
        experiment_id = controller.create_experiment(session_id, sample_experiment_config)
        
        assert experiment_id == "exp-123"
        mock_adapter.create_experiment.assert_called_once_with(sample_experiment_config)
        
        # Should be added to session
        session = controller.get_session(session_id)
        assert "exp-123" in session.experiments

    def test_create_experiment_nonexistent_session(self, temp_workspace, sample_experiment_config):
        """Test creating experiment for non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        with pytest.raises(ValueError, match="Session .* not found"):
            controller.create_experiment("nonexistent-session", sample_experiment_config)

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_start_experiment_success(self, mock_adapter_class, temp_workspace):
        """Test starting an experiment successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.start_experiment.return_value = ExperimentStatus.RUNNING
        mock_adapter_class.return_value = mock_adapter
        
        status = controller.start_experiment(session_id, "exp-123")
        
        assert status == ExperimentStatus.RUNNING
        mock_adapter.start_experiment.assert_called_once_with("exp-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_get_experiment_status_success(self, mock_adapter_class, temp_workspace):
        """Test getting experiment status successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.get_experiment_status.return_value = ExperimentStatus.RUNNING
        mock_adapter_class.return_value = mock_adapter
        
        status = controller.get_experiment_status(session_id, "exp-123")
        
        assert status == ExperimentStatus.RUNNING
        mock_adapter.get_experiment_status.assert_called_once_with("exp-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_get_experiment_results_success(self, mock_adapter_class, temp_workspace):
        """Test getting experiment results successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_results = ExperimentResults(
            experiment_id="exp-123",
            status=ExperimentStatus.COMPLETED,
            total_iterations=10,
            completed_iterations=10
        )
        mock_adapter.get_experiment_results.return_value = mock_results
        mock_adapter_class.return_value = mock_adapter
        
        results = controller.get_experiment_results(session_id, "exp-123")
        
        assert results == mock_results
        mock_adapter.get_experiment_results.assert_called_once_with("exp-123")

    def test_get_available_configs(self, temp_workspace):
        """Test getting available configuration templates."""
        controller = AgentFarmController(str(temp_workspace))
        
        configs = controller.get_available_configs()
        
        assert isinstance(configs, list)
        assert len(configs) > 0
        
        for config in configs:
            assert isinstance(config, ConfigTemplate)

    def test_validate_config(self, temp_workspace, sample_simulation_config):
        """Test validating a configuration."""
        controller = AgentFarmController(str(temp_workspace))
        
        result = controller.validate_config(sample_simulation_config)
        
        assert isinstance(result, ValidationResult)

    def test_create_config_from_template_success(self, temp_workspace):
        """Test creating config from template successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        config = controller.create_config_from_template("basic_simulation")
        
        assert config is not None
        assert isinstance(config, dict)
        assert "name" in config
        assert "steps" in config

    def test_create_config_from_template_with_overrides(self, temp_workspace):
        """Test creating config from template with overrides."""
        controller = AgentFarmController(str(temp_workspace))
        
        overrides = {"name": "Custom Simulation", "steps": 2000}
        config = controller.create_config_from_template("basic_simulation", overrides)
        
        assert config is not None
        assert config["name"] == "Custom Simulation"
        assert config["steps"] == 2000

    def test_create_config_from_template_nonexistent(self, temp_workspace):
        """Test creating config from non-existent template."""
        controller = AgentFarmController(str(temp_workspace))
        
        with pytest.raises(ValueError, match="Template .* not found"):
            controller.create_config_from_template("nonexistent_template")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_analyze_simulation_success(self, mock_adapter_class, temp_workspace):
        """Test analyzing a simulation successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_results = AnalysisResults(
            analysis_id="analysis-123",
            analysis_type="fitness_analysis"
        )
        mock_adapter.analyze_simulation.return_value = mock_results
        mock_adapter_class.return_value = mock_adapter
        
        results = controller.analyze_simulation(session_id, "sim-123")
        
        assert results == mock_results
        mock_adapter.analyze_simulation.assert_called_once_with("sim-123")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_compare_simulations_success(self, mock_adapter_class, temp_workspace):
        """Test comparing simulations successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_results = ComparisonResults(
            comparison_id="comp-123",
            simulation_ids=["sim-1", "sim-2"]
        )
        mock_adapter.compare_simulations.return_value = mock_results
        mock_adapter_class.return_value = mock_adapter
        
        results = controller.compare_simulations(session_id, ["sim-1", "sim-2"])
        
        assert results == mock_results
        mock_adapter.compare_simulations.assert_called_once_with(["sim-1", "sim-2"])

    def test_generate_visualization_success(self, temp_workspace):
        """Test generating visualization successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        viz_path = controller.generate_visualization(session_id, "sim-123", "fitness_chart")
        
        assert viz_path is not None
        assert isinstance(viz_path, str)
        
        # Should create the visualization file
        viz_file = Path(viz_path)
        assert viz_file.exists()

    def test_generate_visualization_nonexistent_session(self, temp_workspace):
        """Test generating visualization for non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        with pytest.raises(ValueError, match="Session .* not found"):
            controller.generate_visualization("nonexistent-session", "sim-123", "fitness_chart")

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_subscribe_to_events_success(self, mock_adapter_class, temp_workspace):
        """Test subscribing to events successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_adapter.subscribe_to_events.return_value = "sub-123"
        mock_adapter_class.return_value = mock_adapter
        
        subscription_id = controller.subscribe_to_events(
            session_id,
            ["simulation_started", "simulation_completed"],
            simulation_id="sim-123"
        )
        
        assert subscription_id == "sub-123"
        mock_adapter.subscribe_to_events.assert_called_once_with(
            ["simulation_started", "simulation_completed"],
            "sim-123",
            None
        )

    @patch('farm.api.unified_controller.UnifiedAdapter')
    def test_get_event_history_success(self, mock_adapter_class, temp_workspace):
        """Test getting event history successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock the adapter
        mock_adapter = Mock()
        mock_events = [
            Event(
                event_id="event-1",
                event_type="simulation_started",
                timestamp=controller.session_manager._sessions[session_id].created_at,
                session_id=session_id,
                simulation_id="sim-123"
            )
        ]
        mock_adapter.get_event_history.return_value = mock_events
        mock_adapter_class.return_value = mock_adapter
        
        events = controller.get_event_history(session_id, "sub-123")
        
        assert events == mock_events
        mock_adapter.get_event_history.assert_called_once_with("sub-123")

    def test_get_session_stats_success(self, temp_workspace):
        """Test getting session statistics successfully."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        stats = controller.get_session_stats(session_id)
        
        assert stats is not None
        assert isinstance(stats, dict)
        assert stats["session_id"] == session_id
        assert stats["name"] == "Test Session"

    def test_get_session_stats_nonexistent(self, temp_workspace):
        """Test getting stats for non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        stats = controller.get_session_stats("nonexistent-session")
        assert stats is None

    def test_list_simulations_success(self, temp_workspace):
        """Test listing simulations in a session."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Add some simulations to the session
        session = controller.get_session(session_id)
        session.simulations = ["sim-1", "sim-2", "sim-3"]
        
        simulations = controller.list_simulations(session_id)
        
        assert simulations == ["sim-1", "sim-2", "sim-3"]

    def test_list_simulations_nonexistent_session(self, temp_workspace):
        """Test listing simulations for non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        simulations = controller.list_simulations("nonexistent-session")
        assert simulations == []

    def test_list_experiments_success(self, temp_workspace):
        """Test listing experiments in a session."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Add some experiments to the session
        session = controller.get_session(session_id)
        session.experiments = ["exp-1", "exp-2"]
        
        experiments = controller.list_experiments(session_id)
        
        assert experiments == ["exp-1", "exp-2"]

    def test_list_experiments_nonexistent_session(self, temp_workspace):
        """Test listing experiments for non-existent session."""
        controller = AgentFarmController(str(temp_workspace))
        
        experiments = controller.list_experiments("nonexistent-session")
        assert experiments == []

    def test_cleanup_success(self, temp_workspace):
        """Test cleaning up controller resources."""
        controller = AgentFarmController(str(temp_workspace))
        
        # Create a session and adapter
        session_id = controller.create_session("Test Session")
        controller._get_adapter(session_id)
        
        assert len(controller._adapters) == 1
        
        # Cleanup
        controller.cleanup()
        
        assert len(controller._adapters) == 0

    def test_context_manager(self, temp_workspace):
        """Test using controller as context manager."""
        with AgentFarmController(str(temp_workspace)) as controller:
            session_id = controller.create_session("Test Session")
            assert session_id is not None
        
        # After exiting context, adapters should be cleaned up
        assert len(controller._adapters) == 0

    def test_context_manager_with_exception(self, temp_workspace):
        """Test context manager cleanup even with exception."""
        try:
            with AgentFarmController(str(temp_workspace)) as controller:
                session_id = controller.create_session("Test Session")
                controller._get_adapter(session_id)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Adapters should still be cleaned up
        assert len(controller._adapters) == 0

    @patch('farm.api.unified_controller.get_logger')
    def test_logging_in_initialization(self, mock_get_logger, temp_workspace):
        """Test that initialization logs appropriately."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        controller = AgentFarmController(str(temp_workspace))
        
        # Should log initialization
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Initialized AgentFarmController" in call for call in log_calls)

    @patch('farm.api.unified_controller.get_logger')
    def test_logging_in_cleanup(self, mock_get_logger, temp_workspace):
        """Test that cleanup logs appropriately."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        controller = AgentFarmController(str(temp_workspace))
        controller.cleanup()
        
        # Should log cleanup
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Cleaning up AgentFarmController" in call for call in log_calls)
        assert any("AgentFarmController cleanup complete" in call for call in log_calls)

    def test_adapter_creation_and_caching(self, temp_workspace):
        """Test that adapters are created and cached properly."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # First call should create adapter
        adapter1 = controller._get_adapter(session_id)
        assert session_id in controller._adapters
        
        # Second call should return same adapter
        adapter2 = controller._get_adapter(session_id)
        assert adapter1 is adapter2

    def test_multiple_sessions_with_adapters(self, temp_workspace):
        """Test managing multiple sessions with separate adapters."""
        controller = AgentFarmController(str(temp_workspace))
        
        # Create multiple sessions
        session1_id = controller.create_session("Session 1")
        session2_id = controller.create_session("Session 2")
        
        # Get adapters for both sessions
        adapter1 = controller._get_adapter(session1_id)
        adapter2 = controller._get_adapter(session2_id)
        
        # Should have separate adapters
        assert adapter1 is not adapter2
        assert len(controller._adapters) == 2
        assert session1_id in controller._adapters
        assert session2_id in controller._adapters

    def test_error_handling_in_adapter_operations(self, temp_workspace):
        """Test error handling in adapter operations."""
        controller = AgentFarmController(str(temp_workspace))
        
        session_id = controller.create_session("Test Session")
        
        # Mock adapter to raise exception
        with patch('farm.api.unified_controller.UnifiedAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.start_simulation.side_effect = RuntimeError("Adapter error")
            mock_adapter_class.return_value = mock_adapter
            
            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Adapter error"):
                controller.start_simulation(session_id, "sim-123")
