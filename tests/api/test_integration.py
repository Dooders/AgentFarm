"""Integration tests for the API components working together."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

import pytest

from farm.api import AgentFarmController
from farm.api.models import (
    SessionStatus, SimulationStatus, ExperimentStatus,
    ConfigCategory, ConfigTemplate, ValidationResult
)


class TestAPIIntegration:
    """Integration tests for API components."""

    def test_full_simulation_workflow(self, temp_workspace):
        """Test complete simulation workflow from creation to results."""
        controller = AgentFarmController(str(temp_workspace))

        # 1. Create a session
        session_id = controller.create_session("Integration Test Session", "Testing full workflow")
        assert session_id is not None

        session = controller.get_session(session_id)
        assert session.name == "Integration Test Session"
        assert session.status == SessionStatus.ACTIVE

        # 2. Get available config templates
        configs = controller.get_available_configs()
        assert len(configs) > 0

        basic_sim_config = None
        for config in configs:
            if config.name == "basic_simulation":
                basic_sim_config = config
                break

        assert basic_sim_config is not None
        assert basic_sim_config.category == ConfigCategory.SIMULATION

        # 3. Create config from template
        sim_config = controller.create_config_from_template("basic_simulation", {
            "name": "Integration Test Simulation",
            "steps": 100
        })
        assert sim_config["name"] == "Integration Test Simulation"
        assert sim_config["steps"] == 100

        # 4. Validate the config
        validation_result = controller.validate_config(sim_config)
        assert isinstance(validation_result, ValidationResult)

        # 5. Create simulation (mocked)
        with patch('farm.api.unified_controller.UnifiedAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.create_simulation.return_value = "sim-123"
            mock_adapter_class.return_value = mock_adapter

            simulation_id = controller.create_simulation(session_id, sim_config)
            assert simulation_id == "sim-123"

            # Should be added to session
            session = controller.get_session(session_id)
            assert "sim-123" in session.simulations

            # 6. Start simulation
            mock_adapter.start_simulation.return_value = SimulationStatus.RUNNING
            status = controller.start_simulation(session_id, simulation_id)
            assert status == SimulationStatus.RUNNING

            # 7. Get simulation status
            mock_adapter.get_simulation_status.return_value = SimulationStatus.RUNNING
            status = controller.get_simulation_status(session_id, simulation_id)
            assert status == SimulationStatus.RUNNING

            # 8. Stop simulation
            mock_adapter.stop_simulation.return_value = SimulationStatus.STOPPED
            status = controller.stop_simulation(session_id, simulation_id)
            assert status == SimulationStatus.STOPPED

            # 9. Get simulation results
            from farm.api.models import SimulationResults
            mock_results = SimulationResults(
                simulation_id=simulation_id,
                status=SimulationStatus.COMPLETED,
                total_steps=100,
                final_agent_count=20,
                final_resource_count=50
            )
            mock_adapter.get_simulation_results.return_value = mock_results

            results = controller.get_simulation_results(session_id, simulation_id)
            assert results.simulation_id == simulation_id
            assert results.status == SimulationStatus.COMPLETED

    def test_full_experiment_workflow(self, temp_workspace):
        """Test complete experiment workflow from creation to results."""
        controller = AgentFarmController(str(temp_workspace))

        # 1. Create a session
        session_id = controller.create_session("Experiment Test Session")

        # 2. Create experiment config
        exp_config = {
            "name": "Integration Test Experiment",
            "iterations": 5,
            "simulation_config": {
                "name": "Base Simulation",
                "steps": 50,
                "agents": {"system_agents": 5, "independent_agents": 5}
            },
            "parameters": {
                "learning_rate": [0.001, 0.01],
                "exploration_rate": [0.1, 0.3]
            }
        }

        # 3. Validate experiment config
        validation_result = controller.validate_config(exp_config)
        assert isinstance(validation_result, ValidationResult)

        # 4. Create experiment (mocked)
        with patch('farm.api.unified_controller.UnifiedAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.create_experiment.return_value = "exp-123"
            mock_adapter_class.return_value = mock_adapter

            experiment_id = controller.create_experiment(session_id, exp_config)
            assert experiment_id == "exp-123"

            # Should be added to session
            session = controller.get_session(session_id)
            assert "exp-123" in session.experiments

            # 5. Start experiment
            mock_adapter.start_experiment.return_value = ExperimentStatus.RUNNING
            status = controller.start_experiment(session_id, experiment_id)
            assert status == ExperimentStatus.RUNNING

            # 6. Get experiment status
            mock_adapter.get_experiment_status.return_value = ExperimentStatus.RUNNING
            status = controller.get_experiment_status(session_id, experiment_id)
            assert status == ExperimentStatus.RUNNING

            # 7. Get experiment results
            from farm.api.models import ExperimentResults
            mock_results = ExperimentResults(
                experiment_id=experiment_id,
                status=ExperimentStatus.COMPLETED,
                total_iterations=5,
                completed_iterations=5
            )
            mock_adapter.get_experiment_results.return_value = mock_results

            results = controller.get_experiment_results(session_id, experiment_id)
            assert results.experiment_id == experiment_id
            assert results.status == ExperimentStatus.COMPLETED

    def test_session_management_workflow(self, temp_workspace):
        """Test complete session management workflow."""
        controller = AgentFarmController(str(temp_workspace))

        # 1. Create multiple sessions
        session1_id = controller.create_session("Session 1", "First session")
        session2_id = controller.create_session("Session 2", "Second session")
        session3_id = controller.create_session("Session 3", "Third session")

        # 2. List all sessions
        sessions = controller.list_sessions()
        assert len(sessions) == 3

        session_names = [s.name for s in sessions]
        assert "Session 1" in session_names
        assert "Session 2" in session_names
        assert "Session 3" in session_names

        # 3. Get session statistics
        stats1 = controller.get_session_stats(session1_id)
        assert stats1 is not None
        assert stats1["name"] == "Session 1"
        assert stats1["simulations"] == 0
        assert stats1["experiments"] == 0

        # 4. Archive a session
        session1 = controller.get_session(session1_id)
        assert session1.status == SessionStatus.ACTIVE

        # Archive through session manager
        controller.session_manager.archive_session(session1_id)
        session1 = controller.get_session(session1_id)
        assert session1.status == SessionStatus.ARCHIVED

        # 5. Restore the session
        controller.session_manager.restore_session(session1_id)
        session1 = controller.get_session(session1_id)
        assert session1.status == SessionStatus.ACTIVE

        # 6. Delete a session
        success = controller.delete_session(session3_id, delete_files=True)
        assert success is True

        # Session should no longer exist
        sessions = controller.list_sessions()
        assert len(sessions) == 2
        session_ids = [s.session_id for s in sessions]
        assert session3_id not in session_ids

    def test_config_template_workflow(self, temp_workspace):
        """Test configuration template workflow."""
        controller = AgentFarmController(str(temp_workspace))

        # 1. Get all available templates
        templates = controller.get_available_configs()
        assert len(templates) > 0

        # 2. Find simulation templates
        sim_templates = [t for t in templates if t.category == ConfigCategory.SIMULATION]
        assert len(sim_templates) > 0

        # 3. Get a specific template
        basic_template = None
        for template in templates:
            if template.name == "basic_simulation":
                basic_template = template
                break

        assert basic_template is not None
        assert basic_template.category == ConfigCategory.SIMULATION
        assert "steps" in basic_template.required_fields
        assert "name" in basic_template.required_fields

        # 4. Create config from template with overrides
        config = controller.create_config_from_template("basic_simulation", {
            "name": "Custom Simulation",
            "steps": 2000,
            "agents": {
                "system_agents": 20,
                "independent_agents": 20
            }
        })

        assert config["name"] == "Custom Simulation"
        assert config["steps"] == 2000
        assert config["agents"]["system_agents"] == 20
        assert config["agents"]["independent_agents"] == 20

        # 5. Validate the created config
        validation_result = controller.validate_config(config)
        assert isinstance(validation_result, ValidationResult)

        # 6. Test invalid config
        invalid_config = {"invalid": "config"}
        validation_result = controller.validate_config(invalid_config)
        assert isinstance(validation_result, ValidationResult)

    def test_analysis_and_comparison_workflow(self, temp_workspace):
        """Test analysis and comparison workflow."""
        controller = AgentFarmController(str(temp_workspace))

        # Create a session
        session_id = controller.create_session("Analysis Test Session")

        with patch('farm.api.unified_controller.UnifiedAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter_class.return_value = mock_adapter

            # 1. Create multiple simulations
            sim1_id = "sim-1"
            sim2_id = "sim-2"
            sim3_id = "sim-3"

            mock_adapter.create_simulation.side_effect = [sim1_id, sim2_id, sim3_id]

            config = controller.create_config_from_template("basic_simulation")

            simulation1_id = controller.create_simulation(session_id, config)
            simulation2_id = controller.create_simulation(session_id, config)
            simulation3_id = controller.create_simulation(session_id, config)

            # 2. Analyze individual simulation
            from farm.api.models import AnalysisResults
            mock_analysis = AnalysisResults(
                analysis_id="analysis-1",
                analysis_type="fitness_analysis",
                summary={"avg_fitness": 0.8}
            )
            mock_adapter.analyze_simulation.return_value = mock_analysis

            analysis = controller.analyze_simulation(session_id, simulation1_id)
            assert analysis.analysis_id == "analysis-1"
            assert analysis.analysis_type == "fitness_analysis"

            # 3. Compare simulations
            from farm.api.models import ComparisonResults
            mock_comparison = ComparisonResults(
                comparison_id="comp-1",
                simulation_ids=[simulation1_id, simulation2_id, simulation3_id],
                comparison_summary={"best_simulation": simulation2_id}
            )
            mock_adapter.compare_simulations.return_value = mock_comparison

            comparison = controller.compare_simulations(session_id, [simulation1_id, simulation2_id, simulation3_id])
            assert comparison.comparison_id == "comp-1"
            assert len(comparison.simulation_ids) == 3

    def test_event_subscription_workflow(self, temp_workspace):
        """Test event subscription and monitoring workflow."""
        controller = AgentFarmController(str(temp_workspace))

        # Create a session
        session_id = controller.create_session("Event Test Session")

        with patch('farm.api.unified_controller.UnifiedAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter_class.return_value = mock_adapter

            # 1. Subscribe to events
            mock_adapter.subscribe_to_events.return_value = "sub-123"

            subscription_id = controller.subscribe_to_events(
                session_id,
                ["simulation_started", "simulation_completed", "simulation_error"],
                simulation_id="sim-123"
            )

            assert subscription_id == "sub-123"
            mock_adapter.subscribe_to_events.assert_called_once_with(
                ["simulation_started", "simulation_completed", "simulation_error"],
                "sim-123",
                None
            )

            # 2. Get event history
            from farm.api.models import Event
            mock_events = [
                Event(
                    event_id="event-1",
                    event_type="simulation_started",
                    timestamp=datetime.now(),
                    session_id=session_id,
                    simulation_id="sim-123"
                ),
                Event(
                    event_id="event-2",
                    event_type="simulation_completed",
                    timestamp=datetime.now(),
                    session_id=session_id,
                    simulation_id="sim-123"
                )
            ]
            mock_adapter.get_event_history.return_value = mock_events

            events = controller.get_event_history(session_id, subscription_id)
            assert len(events) == 2
            assert events[0].event_type == "simulation_started"
            assert events[1].event_type == "simulation_completed"

    def test_visualization_workflow(self, temp_workspace):
        """Test visualization generation workflow."""
        controller = AgentFarmController(str(temp_workspace))

        # Create a session
        session_id = controller.create_session("Visualization Test Session")

        # 1. Generate visualization
        viz_path = controller.generate_visualization(session_id, "sim-123", "fitness_chart")

        assert viz_path is not None
        assert isinstance(viz_path, str)

        # Should create the visualization file
        viz_file = Path(viz_path)
        assert viz_file.exists()
        assert "fitness_chart" in viz_file.name
        assert "sim-123" in viz_file.name

    def test_error_handling_workflow(self, temp_workspace):
        """Test error handling across the API."""
        controller = AgentFarmController(str(temp_workspace))

        # 1. Test operations on non-existent session
        with pytest.raises(ValueError, match="Session .* not found"):
            controller.create_simulation("nonexistent-session", {"name": "Test"})

        with pytest.raises(ValueError, match="Session .* not found"):
            controller.start_simulation("nonexistent-session", "sim-123")

        # 2. Test operations on non-existent simulation
        session_id = controller.create_session("Error Test Session")

        with patch('farm.api.unified_controller.UnifiedAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter_class.return_value = mock_adapter

            # Mock adapter to raise ValueError for non-existent simulation
            mock_adapter.start_simulation.side_effect = ValueError("Simulation sim-123 not found")

            with pytest.raises(ValueError, match="Simulation .* not found"):
                controller.start_simulation(session_id, "sim-123")

        # 3. Test invalid template
        with pytest.raises(ValueError, match="Template .* not found"):
            controller.create_config_from_template("nonexistent_template")

        # 4. Test visualization for non-existent session
        with pytest.raises(ValueError, match="Session .* not found"):
            controller.generate_visualization("nonexistent-session", "sim-123", "chart")

    def test_context_manager_workflow(self, temp_workspace):
        """Test using controller as context manager."""
        with AgentFarmController(str(temp_workspace)) as controller:
            # Create session and simulation
            session_id = controller.create_session("Context Test Session")

            with patch('farm.api.unified_controller.UnifiedAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.create_simulation.return_value = "sim-123"
                mock_adapter_class.return_value = mock_adapter

                config = controller.create_config_from_template("basic_simulation")
                simulation_id = controller.create_simulation(session_id, config)

                assert simulation_id == "sim-123"
                assert session_id in controller._adapters

        # After exiting context, adapters should be cleaned up
        assert len(controller._adapters) == 0

    def test_concurrent_operations(self, temp_workspace):
        """Test concurrent operations on the same controller."""
        import threading
        import time

        controller = AgentFarmController(str(temp_workspace))

        # Create multiple sessions concurrently
        session_ids = []
        errors = []

        def create_session(session_num):
            try:
                session_id = controller.create_session(f"Concurrent Session {session_num}")
                session_ids.append(session_id)
            except Exception as e:
                errors.append(e)

        # Create 5 sessions concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_session, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have 5 sessions and no errors
        assert len(session_ids) == 5
        assert len(errors) == 0

        # All sessions should be unique
        assert len(set(session_ids)) == 5

        # Should be able to list all sessions
        sessions = controller.list_sessions()
        assert len(sessions) == 5

    def test_persistence_workflow(self, temp_workspace):
        """Test that session data persists across controller instances."""
        # Create first controller instance
        controller1 = AgentFarmController(str(temp_workspace))

        # Create sessions
        session1_id = controller1.create_session("Persistent Session 1")
        session2_id = controller1.create_session("Persistent Session 2")

        # Add some metadata
        controller1.session_manager.update_session(session1_id, metadata={"test": "data"})

        # Cleanup first controller
        controller1.cleanup()

        # Create second controller instance with same workspace
        controller2 = AgentFarmController(str(temp_workspace))

        # Should be able to retrieve the sessions
        sessions = controller2.list_sessions()
        assert len(sessions) == 2

        session_ids = [s.session_id for s in sessions]
        assert session1_id in session_ids
        assert session2_id in session_ids

        # Should have the metadata
        session1 = controller2.get_session(session1_id)
        assert session1.metadata == {"test": "data"}

        # Cleanup second controller
        controller2.cleanup()

    def test_large_scale_workflow(self, temp_workspace):
        """Test workflow with many sessions and simulations."""
        controller = AgentFarmController(str(temp_workspace))

        # Create many sessions
        session_ids = []
        for i in range(10):
            session_id = controller.create_session(f"Scale Test Session {i}")
            session_ids.append(session_id)

        # Create simulations in each session
        with patch('farm.api.unified_controller.UnifiedAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter_class.return_value = mock_adapter
            mock_adapter.create_simulation.return_value = "sim-123"

            config = controller.create_config_from_template("basic_simulation")

            for session_id in session_ids:
                simulation_id = controller.create_simulation(session_id, config)
                assert simulation_id == "sim-123"

        # List all sessions
        sessions = controller.list_sessions()
        assert len(sessions) == 10

        # Get statistics for all sessions
        for session_id in session_ids:
            stats = controller.get_session_stats(session_id)
            assert stats is not None
            assert stats["simulations"] == 1

        # Cleanup
        controller.cleanup()
