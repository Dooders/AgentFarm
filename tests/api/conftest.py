"""Test fixtures and utilities for API tests."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from farm.api.models import (
    ConfigCategory,
    ConfigTemplate,
    ExperimentStatus,
    SessionInfo,
    SessionStatus,
    SimulationStatus,
    ValidationResult,
)
from farm.api.server import app


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_session_info():
    """Create a sample SessionInfo object for testing."""
    return SessionInfo(
        session_id="test-session-123",
        name="Test Session",
        description="A test session for unit tests",
        created_at=datetime.now(),
        status=SessionStatus.ACTIVE,
        simulations=["sim-1", "sim-2"],
        experiments=["exp-1"],
        metadata={"test": True, "version": "1.0"},
    )


@pytest.fixture
def sample_config_template():
    """Create a sample ConfigTemplate object for testing."""
    return ConfigTemplate(
        name="test_template",
        description="A test configuration template",
        category=ConfigCategory.SIMULATION,
        parameters={
            "name": "Test Simulation",
            "steps": 1000,
            "agents": {"system_agents": 10, "independent_agents": 10},
        },
        required_fields=["name", "steps"],
        optional_fields=["agents", "environment"],
        examples=[
            {"name": "Quick Test", "steps": 100},
            {"name": "Long Run", "steps": 5000},
        ],
    )


@pytest.fixture
def sample_validation_result():
    """Create a sample ValidationResult object for testing."""
    return ValidationResult(
        is_valid=True,
        errors=[],
        warnings=["Consider increasing step count for better results"],
        suggestions=["Add more agents for complex scenarios"],
        validated_config={"name": "Valid Config", "steps": 1000},
    )


@pytest.fixture
def sample_simulation_config():
    """Create a sample simulation configuration dictionary."""
    return {
        "name": "Test Simulation",
        "steps": 1000,
        "environment": {"width": 100, "height": 100, "resources": 50},
        "agents": {"system_agents": 10, "independent_agents": 10, "control_agents": 0},
        "learning": {"enabled": True, "algorithm": "dqn"},
    }


@pytest.fixture
def sample_experiment_config():
    """Create a sample experiment configuration dictionary."""
    return {
        "name": "Test Experiment",
        "iterations": 10,
        "simulation_config": {
            "name": "Base Simulation",
            "steps": 500,
            "agents": {"system_agents": 5, "independent_agents": 5},
        },
        "parameters": {
            "learning_rate": [0.001, 0.01, 0.1],
            "exploration_rate": [0.1, 0.3, 0.5],
        },
    }


@pytest.fixture
def mock_simulation_controller():
    """Create a mock SimulationController for testing."""
    mock_controller = Mock()
    mock_controller.is_running = False
    mock_controller.is_paused = False
    mock_controller.current_step = 0
    mock_controller.start = Mock()
    mock_controller.stop = Mock()
    mock_controller.pause = Mock()
    mock_controller.resume = Mock()
    mock_controller.get_state = Mock(
        return_value={
            "current_step": 0,
            "total_steps": 1000,
            "is_running": False,
            "is_paused": False,
        }
    )
    mock_controller.cleanup = Mock()
    return mock_controller


@pytest.fixture
def mock_experiment_controller():
    """Create a mock ExperimentController for testing."""
    mock_controller = Mock()
    mock_controller.is_running = False
    mock_controller.current_iteration = 0
    mock_controller.start = Mock()
    mock_controller.stop = Mock()
    mock_controller.get_status = Mock(return_value="created")
    mock_controller.get_results = Mock(return_value={})
    mock_controller.cleanup = Mock()
    return mock_controller


@pytest.fixture
def mock_database():
    """Create a mock database for testing."""
    mock_db = Mock()
    mock_db.add_simulation_record = Mock()
    mock_db.get_simulation_record = Mock()
    mock_db.close = Mock()
    return mock_db


@pytest.fixture
def mock_environment():
    """Create a mock Environment for testing."""
    mock_env = Mock()
    mock_env.width = 100
    mock_env.height = 100
    mock_env.step = Mock()
    mock_env.get_agent_count = Mock(return_value=20)
    mock_env.get_resource_count = Mock(return_value=50)
    mock_env.cleanup = Mock()
    return mock_env


@pytest.fixture
def sessions_json_file(temp_workspace):
    """Create a temporary sessions.json file with sample data."""
    sessions_data = [
        {
            "session_id": "session-1",
            "name": "Test Session 1",
            "description": "First test session",
            "created_at": "2024-01-01T00:00:00",
            "status": "active",
            "simulations": ["sim-1"],
            "experiments": [],
            "metadata": {},
        },
        {
            "session_id": "session-2",
            "name": "Test Session 2",
            "description": "Second test session",
            "created_at": "2024-01-02T00:00:00",
            "status": "archived",
            "simulations": ["sim-2", "sim-3"],
            "experiments": ["exp-1"],
            "metadata": {"archived": True},
        },
    ]

    sessions_file = temp_workspace / "sessions.json"
    with open(sessions_file, "w", encoding="utf-8") as f:
        json.dump(sessions_data, f, indent=2)

    return sessions_file


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    mock_logger = Mock()
    mock_logger.info = Mock()
    mock_logger.error = Mock()
    mock_logger.warning = Mock()
    mock_logger.debug = Mock()
    return mock_logger


# FastAPI-specific fixtures


@pytest.fixture
def fastapi_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_simulation_create_request():
    """Create a sample simulation creation request."""
    return {"simulation_steps": 100, "num_agents": 10, "environment_size": 100}


@pytest.fixture
def sample_analysis_request():
    """Create a sample analysis request."""
    return {
        "experiment_path": "test/experiments",
        "output_path": "test/output",
        "group": "test_group",
        "processor_kwargs": {"param1": "value1"},
        "analysis_kwargs": {"param2": "value2"},
    }


@pytest.fixture
def mock_simulation_database():
    """Create a mock simulation database."""
    mock_db = Mock()
    mock_db.query.gui_repository.get_simulation_data.return_value = {
        "step": 10,
        "agents": [{"id": 1, "position": [10, 20]}],
        "environment": {"size": 100},
    }
    mock_db.export_data.return_value = None
    return mock_db


@pytest.fixture
def mock_analysis_service():
    """Create a mock analysis service."""
    mock_service = Mock()
    mock_result = Mock()
    mock_result.output_path = Path("test/output/result.csv")
    mock_result.dataframe = Mock()
    mock_result.dataframe.shape = (100, 5)
    mock_service.run.return_value = mock_result
    return mock_service


@pytest.fixture
def active_simulation_data():
    """Create sample active simulation data."""
    return {
        "db_path": "test.db",
        "config": {"steps": 100, "agents": 10},
        "created_at": datetime.now().isoformat(),
        "status": "running",
    }


@pytest.fixture
def websocket_message():
    """Create a sample WebSocket message."""
    return {"type": "subscribe_simulation", "sim_id": "test_sim_123"}
