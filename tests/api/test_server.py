"""Unit tests for the FastAPI server endpoints."""

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient

from farm.api.server import (
    AnalysisRequestModel,
    AnalysisResponse,
    SimulationCreateRequest,
    SimulationResponse,
    SimulationStatus,
    _active_simulations_thread_lock,
    active_simulations,
    app,
    manager,
)


class TestFastAPIServer:
    """Test FastAPI server endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def temp_db_path(self, temp_workspace):
        """Create a temporary database path for testing."""
        return str(temp_workspace / "test_simulation.db")

    @pytest.fixture
    def sample_simulation_config(self):
        """Create a sample simulation configuration."""
        return {"simulation_steps": 100, "num_agents": 10, "environment_size": 100}

    def setup_method(self):
        """Clear active simulations before each test."""
        with _active_simulations_thread_lock:
            active_simulations.clear()

    def teardown_method(self):
        """Clean up after each test."""
        with _active_simulations_thread_lock:
            active_simulations.clear()

    def test_create_simulation_success(
        self, client, sample_simulation_config, temp_workspace
    ):
        """Test successful simulation creation."""
        # Mock the simulation components
        with patch("farm.api.server.SimulationConfig") as mock_config_class, patch(
            "farm.api.server.run_simulation"
        ) as mock_run_sim, patch("os.makedirs") as mock_makedirs, patch(
            "farm.api.server._run_simulation_background"
        ) as mock_background_task:

            # Setup mocks - create a proper dataclass instance
            from dataclasses import dataclass
            from typing import Optional

            @dataclass
            class MockSimulationConfig:
                simulation_steps: Optional[int] = None
                num_agents: Optional[int] = None
                environment_size: Optional[int] = None

            mock_config_instance = MockSimulationConfig()
            mock_config_class.from_centralized_config.return_value = (
                mock_config_instance
            )
            mock_run_sim.return_value = None
            mock_background_task.return_value = None  # Mock the background task

            # Create request data
            request_data = SimulationCreateRequest(**sample_simulation_config)

            # Make request
            response = client.post("/api/simulation/new", json=request_data.dict())

            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "accepted"
            assert "sim_id" in data
            assert data["message"] == "Simulation started"

            # Verify simulation was added to active simulations
            sim_id = data["sim_id"]
            with _active_simulations_thread_lock:
                assert sim_id in active_simulations
                assert active_simulations[sim_id]["status"] == "pending"

    def test_create_simulation_invalid_config(self, client):
        """Test simulation creation with invalid configuration."""
        with patch("farm.api.server.SimulationConfig") as mock_config:
            mock_config.from_centralized_config.side_effect = Exception("Config error")

            request_data = SimulationCreateRequest(simulation_steps=100)
            response = client.post("/api/simulation/new", json=request_data.dict())

            assert response.status_code == 500
            data = response.json()
            assert data["detail"] == "Config error"

    def test_get_step_success(self, client, temp_db_path):
        """Test successful step retrieval."""
        sim_id = "test_sim_123"

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": temp_db_path,
                "config": {},
                "created_at": datetime.now().isoformat(),
                "status": "completed",
            }

        # Mock database and query
        with patch("farm.api.server.SimulationDatabase") as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.query.gui_repository.get_simulation_data.return_value = {
                "step": 10,
                "agents": [{"id": 1, "position": [10, 20]}],
                "environment": {"size": 100},
            }

            response = client.get(f"/api/simulation/{sim_id}/step/10")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            assert data["data"]["step"] == 10

    def test_get_step_simulation_not_found(self, client):
        """Test step retrieval for non-existent simulation."""
        response = client.get("/api/simulation/nonexistent/step/10")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_get_analysis_success(self, client, temp_db_path):
        """Test successful analysis retrieval."""
        sim_id = "test_sim_123"

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": temp_db_path,
                "config": {},
                "created_at": datetime.now().isoformat(),
                "status": "completed",
            }

        # Mock database and analysis
        with patch("farm.api.server.SimulationDatabase") as mock_db_class, patch(
            "farm.api.server.analyze_simulation"
        ) as mock_analyze:

            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_analyze.return_value = {
                "total_steps": 100,
                "agent_count": 10,
                "performance_metrics": {"fitness": 0.85},
            }

            response = client.get(f"/api/simulation/{sim_id}/analysis")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            assert data["data"]["total_steps"] == 100

    def test_run_analysis_module_success(self, client):
        """Test successful analysis module execution (async)."""
        module_name = "test_analysis"
        request_data = AnalysisRequestModel(
            experiment_path="test/experiments",
            output_path="test/output",
            group="test_group",
            processor_kwargs={"param1": "value1"},
            analysis_kwargs={"param2": "value2"},
        )

        # Mock analysis controller and config service
        with patch(
            "farm.api.server.AnalysisController"
        ) as mock_controller_class, patch(
            "farm.api.server.EnvConfigService"
        ) as mock_config_service:

            mock_controller = Mock()
            mock_controller_class.return_value = mock_controller
            mock_controller.initialize_analysis.return_value = None

            response = client.post(
                f"/api/analysis/{module_name}", json=request_data.dict()
            )

            # Should return 202 (accepted) for async analysis
            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "accepted"
            assert "Analysis started with ID:" in data["message"]

            # Verify controller was initialized
            mock_controller.initialize_analysis.assert_called_once()

    def test_list_simulations(self, client):
        """Test listing active simulations."""
        # Add some test simulations
        with _active_simulations_thread_lock:
            active_simulations["sim1"] = {
                "db_path": "test1.db",
                "config": {"steps": 100},
                "created_at": datetime.now().isoformat(),
                "status": "running",
            }
            active_simulations["sim2"] = {
                "db_path": "test2.db",
                "config": {"steps": 200},
                "created_at": datetime.now().isoformat(),
                "status": "completed",
            }

        response = client.get("/api/simulations")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert len(data["data"]) == 2
        assert "sim1" in data["data"]
        assert "sim2" in data["data"]

    def test_export_simulation_success(self, client, temp_db_path):
        """Test successful simulation export."""
        sim_id = "test_sim_123"

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": temp_db_path,
                "config": {},
                "created_at": datetime.now().isoformat(),
                "status": "completed",
            }

        # Mock database export
        with patch("farm.api.server.SimulationDatabase") as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            mock_db.export_data.return_value = None

            response = client.get(f"/api/simulation/{sim_id}/export")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "path" in data
            assert data["message"] == "Data exported successfully"

    def test_get_simulation_status_success(self, client):
        """Test successful simulation status retrieval."""
        sim_id = "test_sim_123"
        sim_data = {
            "db_path": "test.db",
            "config": {"steps": 100},
            "created_at": datetime.now().isoformat(),
            "status": "running",
        }

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = sim_data

        response = client.get(f"/api/simulation/{sim_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert data["data"]["status"] == "running"

    def test_get_simulation_status_not_found(self, client):
        """Test status retrieval for non-existent simulation."""
        response = client.get("/api/simulation/nonexistent/status")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Connection should be established successfully
            assert websocket is not None

    def test_websocket_subscribe_simulation_success(self, client):
        """Test WebSocket simulation subscription."""
        sim_id = "test_sim_123"

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": "test.db",
                "config": {},
                "created_at": datetime.now().isoformat(),
                "status": "running",
            }

        with client.websocket_connect("/ws/test_client") as websocket:
            # Send subscription message
            subscribe_message = {"type": "subscribe_simulation", "sim_id": sim_id}
            websocket.send_text(json.dumps(subscribe_message))

            # Receive response
            response = websocket.receive_text()
            response_data = json.loads(response)

            assert response_data["type"] == "subscription_success"
            assert response_data["sim_id"] == sim_id

    def test_websocket_subscribe_simulation_not_found(self, client):
        """Test WebSocket subscription to non-existent simulation."""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Send subscription message for non-existent simulation
            subscribe_message = {
                "type": "subscribe_simulation",
                "sim_id": "nonexistent",
            }
            websocket.send_text(json.dumps(subscribe_message))

            # Receive error response
            response = websocket.receive_text()
            response_data = json.loads(response)

            assert response_data["type"] == "subscription_error"
            assert "not found" in response_data["message"]

    def test_websocket_invalid_json(self, client):
        """Test WebSocket with invalid JSON message."""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Send invalid JSON
            websocket.send_text("invalid json")

            # Receive error response
            response = websocket.receive_text()
            response_data = json.loads(response)

            assert response_data["type"] == "error"
            assert "Invalid JSON format" in response_data["message"]

    def test_background_simulation_task(self, temp_db_path):
        """Test background simulation task execution."""
        sim_id = "test_sim_123"
        config = Mock()
        config.simulation_steps = 100

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": temp_db_path,
                "config": {},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock run_simulation
        with patch("farm.api.server.run_simulation") as mock_run_sim:
            mock_run_sim.return_value = None

            # Import and run the background function
            from farm.api.server import _run_simulation_background

            _run_simulation_background(sim_id, config, temp_db_path)

            # Verify simulation status was updated
            with _active_simulations_thread_lock:
                assert active_simulations[sim_id]["status"] == "completed"
                assert "ended_at" in active_simulations[sim_id]

    def test_background_simulation_task_error(self, temp_db_path):
        """Test background simulation task error handling."""
        sim_id = "test_sim_123"
        config = Mock()
        config.simulation_steps = 100

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": temp_db_path,
                "config": {},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock run_simulation to raise an error
        with patch("farm.api.server.run_simulation") as mock_run_sim:
            mock_run_sim.side_effect = Exception("Simulation failed")

            # Import and run the background function
            from farm.api.server import _run_simulation_background

            _run_simulation_background(sim_id, config, temp_db_path)

            # Verify error status was set
            with _active_simulations_thread_lock:
                assert active_simulations[sim_id]["status"] == "error"
                assert "error_message" in active_simulations[sim_id]
                assert (
                    active_simulations[sim_id]["error_message"] == "Simulation failed"
                )

    def test_concurrent_simulation_access(self, client):
        """Test concurrent access to simulations."""
        sim_id = "test_sim_123"

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": "test.db",
                "config": {},
                "created_at": datetime.now().isoformat(),
                "status": "running",
            }

        # Test concurrent access from multiple threads
        results = []

        def get_status():
            response = client.get(f"/api/simulation/{sim_id}/status")
            results.append(response.status_code)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_status)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

    def test_api_documentation_endpoints(self, client):
        """Test that API documentation endpoints are accessible."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200

        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200

        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
