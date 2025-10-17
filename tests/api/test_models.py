"""Unit tests for the API models module."""

import json
from datetime import datetime
from typing import Dict, Any

import pytest

from farm.api.models import (
    SessionStatus, SimulationStatus, ExperimentStatus, ConfigCategory,
    SessionInfo, SimulationStatusInfo, SimulationResults, ExperimentStatusInfo,
    ExperimentResults, ConfigTemplate, ValidationResult, AnalysisResults,
    ComparisonResults, Event, EventSubscription
)


class TestEnums:
    """Test enumeration classes."""

    def test_session_status_enum(self):
        """Test SessionStatus enum values."""
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.ARCHIVED == "archived"
        assert SessionStatus.DELETED == "deleted"

    def test_simulation_status_enum(self):
        """Test SimulationStatus enum values."""
        assert SimulationStatus.CREATED == "created"
        assert SimulationStatus.RUNNING == "running"
        assert SimulationStatus.PAUSED == "paused"
        assert SimulationStatus.COMPLETED == "completed"
        assert SimulationStatus.ERROR == "error"
        assert SimulationStatus.STOPPED == "stopped"

    def test_experiment_status_enum(self):
        """Test ExperimentStatus enum values."""
        assert ExperimentStatus.CREATED == "created"
        assert ExperimentStatus.RUNNING == "running"
        assert ExperimentStatus.COMPLETED == "completed"
        assert ExperimentStatus.ERROR == "error"
        assert ExperimentStatus.STOPPED == "stopped"

    def test_config_category_enum(self):
        """Test ConfigCategory enum values."""
        assert ConfigCategory.SIMULATION == "simulation"
        assert ConfigCategory.EXPERIMENT == "experiment"
        assert ConfigCategory.RESEARCH == "research"


class TestSessionInfo:
    """Test SessionInfo data class."""

    def test_session_info_creation(self):
        """Test creating a SessionInfo instance."""
        now = datetime.now()
        session = SessionInfo(
            session_id="test-123",
            name="Test Session",
            description="A test session",
            created_at=now,
            status=SessionStatus.ACTIVE
        )

        assert session.session_id == "test-123"
        assert session.name == "Test Session"
        assert session.description == "A test session"
        assert session.created_at == now
        assert session.status == SessionStatus.ACTIVE
        assert session.simulations == []
        assert session.experiments == []
        assert session.metadata == {}

    def test_session_info_with_lists(self):
        """Test SessionInfo with simulations and experiments."""
        session = SessionInfo(
            session_id="test-123",
            name="Test Session",
            description="A test session",
            created_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            simulations=["sim-1", "sim-2"],
            experiments=["exp-1"],
            metadata={"key": "value"}
        )

        assert session.simulations == ["sim-1", "sim-2"]
        assert session.experiments == ["exp-1"]
        assert session.metadata == {"key": "value"}

    def test_session_info_to_dict(self):
        """Test SessionInfo to_dict method."""
        now = datetime(2024, 1, 1, 12, 0, 0)
        session = SessionInfo(
            session_id="test-123",
            name="Test Session",
            description="A test session",
            created_at=now,
            status=SessionStatus.ACTIVE,
            simulations=["sim-1"],
            experiments=["exp-1"],
            metadata={"key": "value"}
        )

        result = session.to_dict()

        expected = {
            "session_id": "test-123",
            "name": "Test Session",
            "description": "A test session",
            "created_at": "2024-01-01T12:00:00",
            "status": "active",
            "simulations": ["sim-1"],
            "experiments": ["exp-1"],
            "metadata": {"key": "value"}
        }

        assert result == expected

    def test_session_info_json_serializable(self):
        """Test that SessionInfo is JSON serializable."""
        session = SessionInfo(
            session_id="test-123",
            name="Test Session",
            description="A test session",
            created_at=datetime.now(),
            status=SessionStatus.ACTIVE
        )

        # Should not raise an exception
        json.dumps(session.to_dict())


class TestSimulationStatusInfo:
    """Test SimulationStatusInfo data class."""

    def test_simulation_status_info_creation(self):
        """Test creating a SimulationStatusInfo instance."""
        now = datetime.now()
        status_info = SimulationStatusInfo(
            simulation_id="sim-123",
            status=SimulationStatus.RUNNING,
            current_step=100,
            total_steps=1000,
            progress_percentage=10.0,
            start_time=now,
            end_time=None,
            error_message=None
        )

        assert status_info.simulation_id == "sim-123"
        assert status_info.status == SimulationStatus.RUNNING
        assert status_info.current_step == 100
        assert status_info.total_steps == 1000
        assert status_info.progress_percentage == 10.0
        assert status_info.start_time == now
        assert status_info.end_time is None
        assert status_info.error_message is None
        assert status_info.metadata == {}

    def test_simulation_status_info_to_dict(self):
        """Test SimulationStatusInfo to_dict method."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 13, 0, 0)

        status_info = SimulationStatusInfo(
            simulation_id="sim-123",
            status=SimulationStatus.COMPLETED,
            current_step=1000,
            total_steps=1000,
            progress_percentage=100.0,
            start_time=start_time,
            end_time=end_time,
            error_message=None,
            metadata={"test": True}
        )

        result = status_info.to_dict()

        expected = {
            "simulation_id": "sim-123",
            "status": "completed",
            "current_step": 1000,
            "total_steps": 1000,
            "progress_percentage": 100.0,
            "start_time": "2024-01-01T12:00:00",
            "end_time": "2024-01-01T13:00:00",
            "error_message": None,
            "metadata": {"test": True}
        }

        assert result == expected

    def test_simulation_status_info_with_none_times(self):
        """Test SimulationStatusInfo with None start/end times."""
        status_info = SimulationStatusInfo(
            simulation_id="sim-123",
            status=SimulationStatus.CREATED
        )

        result = status_info.to_dict()

        assert result["start_time"] is None
        assert result["end_time"] is None


class TestSimulationResults:
    """Test SimulationResults data class."""

    def test_simulation_results_creation(self):
        """Test creating a SimulationResults instance."""
        results = SimulationResults(
            simulation_id="sim-123",
            status=SimulationStatus.COMPLETED,
            total_steps=1000,
            final_agent_count=20,
            final_resource_count=50,
            metrics={"fitness": 0.85, "survival_rate": 0.9},
            data_files=["data.csv", "metrics.json"],
            analysis_available=True,
            metadata={"duration": 3600}
        )

        assert results.simulation_id == "sim-123"
        assert results.status == SimulationStatus.COMPLETED
        assert results.total_steps == 1000
        assert results.final_agent_count == 20
        assert results.final_resource_count == 50
        assert results.metrics == {"fitness": 0.85, "survival_rate": 0.9}
        assert results.data_files == ["data.csv", "metrics.json"]
        assert results.analysis_available is True
        assert results.metadata == {"duration": 3600}

    def test_simulation_results_to_dict(self):
        """Test SimulationResults to_dict method."""
        results = SimulationResults(
            simulation_id="sim-123",
            status=SimulationStatus.COMPLETED,
            total_steps=1000,
            final_agent_count=20,
            final_resource_count=50
        )

        result = results.to_dict()

        expected = {
            "simulation_id": "sim-123",
            "status": "completed",
            "total_steps": 1000,
            "final_agent_count": 20,
            "final_resource_count": 50,
            "metrics": {},
            "data_files": [],
            "analysis_available": False,
            "metadata": {}
        }

        assert result == expected


class TestExperimentStatusInfo:
    """Test ExperimentStatusInfo data class."""

    def test_experiment_status_info_creation(self):
        """Test creating an ExperimentStatusInfo instance."""
        now = datetime.now()
        status_info = ExperimentStatusInfo(
            experiment_id="exp-123",
            status=ExperimentStatus.RUNNING,
            current_iteration=5,
            total_iterations=10,
            progress_percentage=50.0,
            start_time=now,
            end_time=None,
            error_message=None
        )

        assert status_info.experiment_id == "exp-123"
        assert status_info.status == ExperimentStatus.RUNNING
        assert status_info.current_iteration == 5
        assert status_info.total_iterations == 10
        assert status_info.progress_percentage == 50.0
        assert status_info.start_time == now
        assert status_info.end_time is None
        assert status_info.error_message is None
        assert status_info.metadata == {}

    def test_experiment_status_info_to_dict(self):
        """Test ExperimentStatusInfo to_dict method."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)

        status_info = ExperimentStatusInfo(
            experiment_id="exp-123",
            status=ExperimentStatus.RUNNING,
            current_iteration=3,
            total_iterations=10,
            progress_percentage=30.0,
            start_time=start_time,
            metadata={"test": True}
        )

        result = status_info.to_dict()

        expected = {
            "experiment_id": "exp-123",
            "status": "running",
            "current_iteration": 3,
            "total_iterations": 10,
            "progress_percentage": 30.0,
            "start_time": "2024-01-01T12:00:00",
            "end_time": None,
            "error_message": None,
            "metadata": {"test": True}
        }

        assert result == expected


class TestExperimentResults:
    """Test ExperimentResults data class."""

    def test_experiment_results_creation(self):
        """Test creating an ExperimentResults instance."""
        results = ExperimentResults(
            experiment_id="exp-123",
            status=ExperimentStatus.COMPLETED,
            total_iterations=10,
            completed_iterations=10,
            results_summary={"avg_fitness": 0.8, "best_fitness": 0.95},
            data_files=["results.csv", "analysis.json"],
            analysis_available=True,
            metadata={"duration": 7200}
        )

        assert results.experiment_id == "exp-123"
        assert results.status == ExperimentStatus.COMPLETED
        assert results.total_iterations == 10
        assert results.completed_iterations == 10
        assert results.results_summary == {"avg_fitness": 0.8, "best_fitness": 0.95}
        assert results.data_files == ["results.csv", "analysis.json"]
        assert results.analysis_available is True
        assert results.metadata == {"duration": 7200}

    def test_experiment_results_to_dict(self):
        """Test ExperimentResults to_dict method."""
        results = ExperimentResults(
            experiment_id="exp-123",
            status=ExperimentStatus.COMPLETED,
            total_iterations=10,
            completed_iterations=10
        )

        result = results.to_dict()

        expected = {
            "experiment_id": "exp-123",
            "status": "completed",
            "total_iterations": 10,
            "completed_iterations": 10,
            "results_summary": {},
            "data_files": [],
            "analysis_available": False,
            "metadata": {}
        }

        assert result == expected


class TestConfigTemplate:
    """Test ConfigTemplate data class."""

    def test_config_template_creation(self):
        """Test creating a ConfigTemplate instance."""
        template = ConfigTemplate(
            name="test_template",
            description="A test template",
            category=ConfigCategory.SIMULATION,
            parameters={"steps": 1000, "agents": 10},
            required_fields=["steps"],
            optional_fields=["agents"],
            examples=[{"steps": 100}, {"steps": 5000}]
        )

        assert template.name == "test_template"
        assert template.description == "A test template"
        assert template.category == ConfigCategory.SIMULATION
        assert template.parameters == {"steps": 1000, "agents": 10}
        assert template.required_fields == ["steps"]
        assert template.optional_fields == ["agents"]
        assert template.examples == [{"steps": 100}, {"steps": 5000}]

    def test_config_template_to_dict(self):
        """Test ConfigTemplate to_dict method."""
        template = ConfigTemplate(
            name="test_template",
            description="A test template",
            category=ConfigCategory.SIMULATION,
            parameters={"steps": 1000},
            required_fields=["steps"],
            optional_fields=["agents"],
            examples=[{"steps": 100}]
        )

        result = template.to_dict()

        expected = {
            "name": "test_template",
            "description": "A test template",
            "category": "simulation",
            "parameters": {"steps": 1000},
            "required_fields": ["steps"],
            "optional_fields": ["agents"],
            "examples": [{"steps": 100}]
        }

        assert result == expected


class TestValidationResult:
    """Test ValidationResult data class."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult instance."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Consider increasing steps"],
            suggestions=["Add more agents"],
            validated_config={"steps": 1000}
        )

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == ["Consider increasing steps"]
        assert result.suggestions == ["Add more agents"]
        assert result.validated_config == {"steps": 1000}

    def test_validation_result_with_errors(self):
        """Test ValidationResult with validation errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Missing required field: steps", "Invalid agent count"],
            warnings=[],
            suggestions=["Add steps field", "Use positive agent count"],
            validated_config=None
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Missing required field: steps" in result.errors
        assert "Invalid agent count" in result.errors
        assert result.validated_config is None

    def test_validation_result_to_dict(self):
        """Test ValidationResult to_dict method."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Warning message"],
            suggestions=["Suggestion message"],
            validated_config={"steps": 1000}
        )

        result_dict = result.to_dict()

        expected = {
            "is_valid": True,
            "errors": [],
            "warnings": ["Warning message"],
            "suggestions": ["Suggestion message"],
            "validated_config": {"steps": 1000}
        }

        assert result_dict == expected


class TestAnalysisResults:
    """Test AnalysisResults data class."""

    def test_analysis_results_creation(self):
        """Test creating an AnalysisResults instance."""
        results = AnalysisResults(
            analysis_id="analysis-123",
            analysis_type="fitness_analysis",
            summary={"avg_fitness": 0.8, "std_fitness": 0.1},
            detailed_results={"per_agent": [0.7, 0.8, 0.9]},
            output_files=["analysis.csv", "charts.png"],
            charts=["fitness_over_time.png"],
            metadata={"algorithm": "dqn"}
        )

        assert results.analysis_id == "analysis-123"
        assert results.analysis_type == "fitness_analysis"
        assert results.summary == {"avg_fitness": 0.8, "std_fitness": 0.1}
        assert results.detailed_results == {"per_agent": [0.7, 0.8, 0.9]}
        assert results.output_files == ["analysis.csv", "charts.png"]
        assert results.charts == ["fitness_over_time.png"]
        assert results.metadata == {"algorithm": "dqn"}

    def test_analysis_results_to_dict(self):
        """Test AnalysisResults to_dict method."""
        results = AnalysisResults(
            analysis_id="analysis-123",
            analysis_type="fitness_analysis"
        )

        result = results.to_dict()

        expected = {
            "analysis_id": "analysis-123",
            "analysis_type": "fitness_analysis",
            "summary": {},
            "detailed_results": {},
            "output_files": [],
            "charts": [],
            "metadata": {}
        }

        assert result == expected


class TestComparisonResults:
    """Test ComparisonResults data class."""

    def test_comparison_results_creation(self):
        """Test creating a ComparisonResults instance."""
        results = ComparisonResults(
            comparison_id="comp-123",
            simulation_ids=["sim-1", "sim-2", "sim-3"],
            comparison_summary={"best_simulation": "sim-2"},
            detailed_comparison={"fitness": {"sim-1": 0.7, "sim-2": 0.9, "sim-3": 0.8}},
            output_files=["comparison.csv"],
            charts=["comparison_chart.png"],
            metadata={"comparison_type": "fitness"}
        )

        assert results.comparison_id == "comp-123"
        assert results.simulation_ids == ["sim-1", "sim-2", "sim-3"]
        assert results.comparison_summary == {"best_simulation": "sim-2"}
        assert results.detailed_comparison == {"fitness": {"sim-1": 0.7, "sim-2": 0.9, "sim-3": 0.8}}
        assert results.output_files == ["comparison.csv"]
        assert results.charts == ["comparison_chart.png"]
        assert results.metadata == {"comparison_type": "fitness"}

    def test_comparison_results_to_dict(self):
        """Test ComparisonResults to_dict method."""
        results = ComparisonResults(
            comparison_id="comp-123",
            simulation_ids=["sim-1", "sim-2"]
        )

        result = results.to_dict()

        expected = {
            "comparison_id": "comp-123",
            "simulation_ids": ["sim-1", "sim-2"],
            "comparison_summary": {},
            "detailed_comparison": {},
            "output_files": [],
            "charts": [],
            "metadata": {}
        }

        assert result == expected


class TestEvent:
    """Test Event data class."""

    def test_event_creation(self):
        """Test creating an Event instance."""
        now = datetime.now()
        event = Event(
            event_id="event-123",
            event_type="simulation_started",
            timestamp=now,
            session_id="session-123",
            simulation_id="sim-123",
            experiment_id=None,
            data={"step": 0, "agents": 10},
            message="Simulation started successfully"
        )

        assert event.event_id == "event-123"
        assert event.event_type == "simulation_started"
        assert event.timestamp == now
        assert event.session_id == "session-123"
        assert event.simulation_id == "sim-123"
        assert event.experiment_id is None
        assert event.data == {"step": 0, "agents": 10}
        assert event.message == "Simulation started successfully"

    def test_event_to_dict(self):
        """Test Event to_dict method."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        event = Event(
            event_id="event-123",
            event_type="simulation_completed",
            timestamp=timestamp,
            session_id="session-123",
            simulation_id="sim-123",
            data={"final_step": 1000},
            message="Simulation completed"
        )

        result = event.to_dict()

        expected = {
            "event_id": "event-123",
            "event_type": "simulation_completed",
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "session-123",
            "simulation_id": "sim-123",
            "experiment_id": None,
            "data": {"final_step": 1000},
            "message": "Simulation completed"
        }

        assert result == expected


class TestEventSubscription:
    """Test EventSubscription data class."""

    def test_event_subscription_creation(self):
        """Test creating an EventSubscription instance."""
        now = datetime.now()
        subscription = EventSubscription(
            subscription_id="sub-123",
            session_id="session-123",
            event_types=["simulation_started", "simulation_completed"],
            simulation_id="sim-123",
            experiment_id=None,
            created_at=now,
            active=True
        )

        assert subscription.subscription_id == "sub-123"
        assert subscription.session_id == "session-123"
        assert subscription.event_types == ["simulation_started", "simulation_completed"]
        assert subscription.simulation_id == "sim-123"
        assert subscription.experiment_id is None
        assert subscription.created_at == now
        assert subscription.active is True

    def test_event_subscription_defaults(self):
        """Test EventSubscription with default values."""
        subscription = EventSubscription(
            subscription_id="sub-123",
            session_id="session-123",
            event_types=["simulation_started"]
        )

        assert subscription.simulation_id is None
        assert subscription.experiment_id is None
        assert subscription.active is True
        # created_at should be set to current time (within a reasonable range)
        assert isinstance(subscription.created_at, datetime)

    def test_event_subscription_to_dict(self):
        """Test EventSubscription to_dict method."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        subscription = EventSubscription(
            subscription_id="sub-123",
            session_id="session-123",
            event_types=["simulation_started"],
            simulation_id="sim-123",
            created_at=timestamp,
            active=False
        )

        result = subscription.to_dict()

        expected = {
            "subscription_id": "sub-123",
            "session_id": "session-123",
            "event_types": ["simulation_started"],
            "simulation_id": "sim-123",
            "experiment_id": None,
            "created_at": "2024-01-01T12:00:00",
            "active": False
        }

        assert result == expected
