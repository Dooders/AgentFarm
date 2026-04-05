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
        by_id = {row["sim_id"]: row for row in data["data"]}
        assert "sim1" in by_id and "sim2" in by_id
        assert by_id["sim1"]["status"] == "running"
        assert "db_path" not in by_id["sim1"]
        assert "config" not in by_id["sim1"]

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
        assert data["data"]["sim_id"] == sim_id
        assert "db_path" not in data["data"]
        assert "config" not in data["data"]

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
                    active_simulations[sim_id]["error_message"] == "Simulation failed. Check server logs for details."
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


class TestAnalysisEndpoints:
    """Tests for analysis-related API endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def clear_state(self):
        from farm.api.server import active_analyses, _active_analyses_thread_lock
        with _active_analyses_thread_lock:
            active_analyses.clear()
        yield
        with _active_analyses_thread_lock:
            active_analyses.clear()

    # ------------------------------------------------------------------
    # /api/analysis/{module_name}  (POST)
    # ------------------------------------------------------------------

    def test_run_analysis_module_success(self, client):
        """POST /api/analysis/{module} returns 202 accepted."""
        request_data = {
            "experiment_path": "test/experiments",
            "output_path": "test/output",
            "group": "all",
        }
        with patch("farm.api.server.AnalysisController") as mock_ctrl_cls, \
             patch("farm.api.server.EnvConfigService"):
            mock_ctrl = Mock()
            mock_ctrl_cls.return_value = mock_ctrl

            response = client.post("/api/analysis/test_module", json=request_data)

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "accepted"
        assert "Analysis started" in data["message"]

    def test_run_analysis_module_error(self, client):
        """POST /api/analysis/{module} returns 500 on error."""
        request_data = {"experiment_path": "test/exp", "output_path": "test/out"}
        with patch("farm.api.server.AnalysisController") as mock_ctrl_cls, \
             patch("farm.api.server.EnvConfigService"):
            mock_ctrl_cls.side_effect = RuntimeError("controller init error")

            response = client.post("/api/analysis/bad_module", json=request_data)

        assert response.status_code == 500

    # ------------------------------------------------------------------
    # /api/analysis/{analysis_id}/status  (GET)
    # ------------------------------------------------------------------

    def test_get_analysis_status_success(self, client):
        """GET /api/analysis/{id}/status returns analysis info."""
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        mock_ctrl = Mock()
        mock_ctrl.get_state.return_value = {"progress": 0.5, "status": "running"}

        with _active_analyses_thread_lock:
            active_analyses["ana1"] = {
                "controller": mock_ctrl,
                "module_name": "genesis",
                "status": "running",
                "created_at": datetime.now().isoformat(),
            }

        response = client.get("/api/analysis/ana1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["progress"] == 0.5

    def test_get_analysis_status_not_found(self, client):
        response = client.get("/api/analysis/nonexistent/status")
        assert response.status_code == 404

    def test_get_analysis_status_no_controller(self, client):
        """Status endpoint works when no live controller is attached."""
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        with _active_analyses_thread_lock:
            active_analyses["ana2"] = {
                "module_name": "genesis",
                "status": "completed",
                "created_at": datetime.now().isoformat(),
            }

        response = client.get("/api/analysis/ana2/status")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "completed"

    # ------------------------------------------------------------------
    # /api/analysis/{analysis_id}/pause  (POST)
    # ------------------------------------------------------------------

    def test_pause_analysis_success(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        mock_ctrl = Mock()
        with _active_analyses_thread_lock:
            active_analyses["ana3"] = {"controller": mock_ctrl, "status": "running"}

        response = client.post("/api/analysis/ana3/pause")
        assert response.status_code == 200
        mock_ctrl.pause.assert_called_once()

    def test_pause_analysis_not_found(self, client):
        response = client.post("/api/analysis/nonexistent/pause")
        assert response.status_code == 404

    def test_pause_analysis_no_controller(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        with _active_analyses_thread_lock:
            active_analyses["ana4"] = {"status": "running"}  # no controller key

        response = client.post("/api/analysis/ana4/pause")
        assert response.status_code == 400

    # ------------------------------------------------------------------
    # /api/analysis/{analysis_id}/resume  (POST)
    # ------------------------------------------------------------------

    def test_resume_analysis_success(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        mock_ctrl = Mock()
        with _active_analyses_thread_lock:
            active_analyses["ana5"] = {"controller": mock_ctrl, "status": "paused"}

        response = client.post("/api/analysis/ana5/resume")
        assert response.status_code == 200
        mock_ctrl.start.assert_called_once()

    def test_resume_analysis_not_found(self, client):
        response = client.post("/api/analysis/nonexistent/resume")
        assert response.status_code == 404

    def test_resume_analysis_no_controller(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        with _active_analyses_thread_lock:
            active_analyses["ana6"] = {"status": "paused"}

        response = client.post("/api/analysis/ana6/resume")
        assert response.status_code == 400

    # ------------------------------------------------------------------
    # /api/analysis/{analysis_id}/stop  (POST)
    # ------------------------------------------------------------------

    def test_stop_analysis_success(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        mock_ctrl = Mock()
        with _active_analyses_thread_lock:
            active_analyses["ana7"] = {"controller": mock_ctrl, "status": "running"}

        response = client.post("/api/analysis/ana7/stop")
        assert response.status_code == 200
        mock_ctrl.stop.assert_called_once()

    def test_stop_analysis_not_found(self, client):
        response = client.post("/api/analysis/nonexistent/stop")
        assert response.status_code == 404

    def test_stop_analysis_no_controller(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        with _active_analyses_thread_lock:
            active_analyses["ana8"] = {"status": "running"}

        response = client.post("/api/analysis/ana8/stop")
        assert response.status_code == 400

    def test_stop_updates_status(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        mock_ctrl = Mock()
        with _active_analyses_thread_lock:
            active_analyses["ana9"] = {"controller": mock_ctrl, "status": "running"}

        client.post("/api/analysis/ana9/stop")
        with _active_analyses_thread_lock:
            assert active_analyses["ana9"]["status"] == "stopped"

    # ------------------------------------------------------------------
    # /api/analyses  (GET)
    # ------------------------------------------------------------------

    def test_list_analyses_empty(self, client):
        response = client.get("/api/analyses")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] == {}

    def test_list_analyses_with_entries(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        mock_ctrl = Mock()
        mock_ctrl.get_state.return_value = {"progress": 0.3}

        with _active_analyses_thread_lock:
            active_analyses["ana10"] = {
                "controller": mock_ctrl,
                "module_name": "test",
                "status": "running",
            }

        response = client.get("/api/analyses")
        assert response.status_code == 200
        data = response.json()
        assert "ana10" in data["data"]
        # controller should not be in the response
        assert "controller" not in data["data"]["ana10"]
        # live state should be merged
        assert data["data"]["ana10"]["progress"] == 0.3

    def test_list_analyses_no_controller(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        with _active_analyses_thread_lock:
            active_analyses["ana11"] = {
                "module_name": "test",
                "status": "completed",
            }

        response = client.get("/api/analyses")
        assert response.status_code == 200

    # ------------------------------------------------------------------
    # /api/analysis/modules  (GET)
    # ------------------------------------------------------------------

    def test_list_analysis_modules(self, client):
        with patch("farm.api.server.AnalysisController") as mock_ctrl_cls, \
             patch("farm.api.server.EnvConfigService"):
            mock_ctrl = Mock()
            mock_ctrl.list_available_modules.return_value = [
                {"name": "genesis", "description": "Genesis analysis"}
            ]
            mock_ctrl_cls.return_value = mock_ctrl

            response = client.get("/api/analysis/modules")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["data"]) == 1

    def test_list_analysis_modules_error(self, client):
        with patch("farm.api.server.AnalysisController") as mock_ctrl_cls, \
             patch("farm.api.server.EnvConfigService"):
            mock_ctrl_cls.side_effect = RuntimeError("module error")

            response = client.get("/api/analysis/modules")

        assert response.status_code == 500

    # ------------------------------------------------------------------
    # /api/analysis/modules/{module_name}  (GET)
    # ------------------------------------------------------------------

    def test_get_module_info(self, client):
        with patch("farm.api.server.AnalysisController") as mock_ctrl_cls, \
             patch("farm.api.server.EnvConfigService"):
            mock_ctrl = Mock()
            mock_ctrl.get_module_info.return_value = {
                "name": "genesis",
                "functions": ["func1"]
            }
            mock_ctrl_cls.return_value = mock_ctrl

            response = client.get("/api/analysis/modules/genesis")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["name"] == "genesis"

    def test_get_module_info_error(self, client):
        with patch("farm.api.server.AnalysisController") as mock_ctrl_cls, \
             patch("farm.api.server.EnvConfigService"):
            mock_ctrl_cls.side_effect = RuntimeError("module not found")

            response = client.get("/api/analysis/modules/unknown")

        assert response.status_code == 500

    # ------------------------------------------------------------------
    # /api/analyses/cleanup  (POST)
    # ------------------------------------------------------------------

    def test_cleanup_analyses_endpoint(self, client):
        response = client.post("/api/analyses/cleanup")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "removed_count" in data

    # ------------------------------------------------------------------
    # /api/analyses/stats  (GET)
    # ------------------------------------------------------------------

    def test_get_analysis_stats_empty(self, client):
        response = client.get("/api/analyses/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        stats = data["data"]
        assert stats["total_analyses"] == 0
        assert stats["running_count"] == 0

    def test_get_analysis_stats_with_entries(self, client):
        from farm.api.server import active_analyses, _active_analyses_thread_lock

        with _active_analyses_thread_lock:
            active_analyses["s1"] = {"status": "running"}
            active_analyses["s2"] = {"status": "completed"}

        response = client.get("/api/analyses/stats")
        data = response.json()
        assert data["data"]["total_analyses"] == 2
        assert data["data"]["running_count"] == 1


class TestCleanupOldAnalyses:
    """Tests for the _cleanup_old_analyses helper function."""

    @pytest.fixture(autouse=True)
    def clear_state(self):
        from farm.api.server import active_analyses, _active_analyses_thread_lock
        with _active_analyses_thread_lock:
            active_analyses.clear()
        yield
        with _active_analyses_thread_lock:
            active_analyses.clear()

    def test_cleanup_removes_expired_analyses(self):
        from farm.api.server import (
            _cleanup_old_analyses,
            active_analyses,
            _active_analyses_thread_lock,
        )
        # Add an old completed analysis
        old_time = "2000-01-01T00:00:00"  # very old
        with _active_analyses_thread_lock:
            active_analyses["old1"] = {
                "status": "completed",
                "ended_at": old_time,
            }

        _cleanup_old_analyses()

        with _active_analyses_thread_lock:
            assert "old1" not in active_analyses

    def test_cleanup_keeps_recent_analyses(self):
        from farm.api.server import (
            _cleanup_old_analyses,
            active_analyses,
            _active_analyses_thread_lock,
        )
        recent_time = datetime.now().isoformat()
        with _active_analyses_thread_lock:
            active_analyses["recent1"] = {
                "status": "completed",
                "ended_at": recent_time,
            }

        _cleanup_old_analyses()

        with _active_analyses_thread_lock:
            assert "recent1" in active_analyses

    def test_cleanup_calls_controller_cleanup(self):
        from farm.api.server import (
            _cleanup_old_analyses,
            active_analyses,
            _active_analyses_thread_lock,
        )
        mock_ctrl = Mock()
        old_time = "2000-01-01T00:00:00"
        with _active_analyses_thread_lock:
            active_analyses["ctrl1"] = {
                "controller": mock_ctrl,
                "status": "completed",
                "ended_at": old_time,
            }

        _cleanup_old_analyses()

        mock_ctrl.cleanup.assert_called_once()

    def test_cleanup_excess_completed_analyses(self):
        """When there are too many completed entries, oldest are pruned."""
        from farm.api.server import (
            _cleanup_old_analyses,
            active_analyses,
            _active_analyses_thread_lock,
            MAX_COMPLETED_ANALYSES,
        )
        recent_time = datetime.now().isoformat()
        with _active_analyses_thread_lock:
            # Fill beyond the limit
            for i in range(MAX_COMPLETED_ANALYSES + 5):
                active_analyses[f"excess_{i}"] = {
                    "status": "completed",
                    "ended_at": recent_time,
                }

        _cleanup_old_analyses()

        with _active_analyses_thread_lock:
            assert len(active_analyses) <= MAX_COMPLETED_ANALYSES


class TestRunAnalysisBackground:
    """Tests for the _run_analysis_background helper function."""

    @pytest.fixture(autouse=True)
    def clear_state(self):
        from farm.api.server import active_analyses, _active_analyses_thread_lock
        with _active_analyses_thread_lock:
            active_analyses.clear()
        yield
        with _active_analyses_thread_lock:
            active_analyses.clear()

    def test_background_analysis_success(self):
        from farm.api.server import (
            _run_analysis_background,
            active_analyses,
            _active_analyses_thread_lock,
        )
        from farm.analysis.service import AnalysisResult
        from pathlib import Path

        mock_ctrl = Mock()
        mock_result = Mock(spec=AnalysisResult)
        mock_result.success = True
        mock_result.output_path = Path("test/output")
        mock_result.execution_time = 1.5
        mock_result.cache_hit = False
        mock_result.dataframe = None
        mock_ctrl.get_result.return_value = mock_result

        with _active_analyses_thread_lock:
            active_analyses["bg1"] = {"status": "pending", "module_name": "test"}

        _run_analysis_background("bg1", mock_ctrl)

        with _active_analyses_thread_lock:
            assert active_analyses["bg1"]["status"] == "completed"
            assert active_analyses["bg1"]["execution_time"] == 1.5

    def test_background_analysis_failure(self):
        from farm.api.server import (
            _run_analysis_background,
            active_analyses,
            _active_analyses_thread_lock,
        )
        from farm.analysis.service import AnalysisResult

        mock_ctrl = Mock()
        mock_result = Mock(spec=AnalysisResult)
        mock_result.success = False
        mock_result.error = "analysis failed"
        mock_ctrl.get_result.return_value = mock_result

        with _active_analyses_thread_lock:
            active_analyses["bg2"] = {"status": "pending", "module_name": "test"}

        _run_analysis_background("bg2", mock_ctrl)

        with _active_analyses_thread_lock:
            assert active_analyses["bg2"]["status"] == "error"

    def test_background_analysis_exception(self):
        from farm.api.server import (
            _run_analysis_background,
            active_analyses,
            _active_analyses_thread_lock,
        )

        mock_ctrl = Mock()
        mock_ctrl.start.side_effect = RuntimeError("crash")

        with _active_analyses_thread_lock:
            active_analyses["bg3"] = {"status": "pending", "module_name": "test"}

        _run_analysis_background("bg3", mock_ctrl)

        with _active_analyses_thread_lock:
            assert active_analyses["bg3"]["status"] == "error"

    def test_background_analysis_rows_counted(self):
        """dataframe rows are counted and stored."""
        from farm.api.server import (
            _run_analysis_background,
            active_analyses,
            _active_analyses_thread_lock,
        )
        from farm.analysis.service import AnalysisResult
        from pathlib import Path

        import pandas as pd

        mock_ctrl = Mock()
        mock_result = Mock(spec=AnalysisResult)
        mock_result.success = True
        mock_result.output_path = Path("test/output")
        mock_result.execution_time = 0.5
        mock_result.cache_hit = True
        mock_result.dataframe = pd.DataFrame({"a": [1, 2, 3]})
        mock_ctrl.get_result.return_value = mock_result

        with _active_analyses_thread_lock:
            active_analyses["bg4"] = {"status": "pending", "module_name": "test"}

        _run_analysis_background("bg4", mock_ctrl)

        with _active_analyses_thread_lock:
            assert active_analyses["bg4"]["rows"] == 3

    def test_background_analysis_controller_cleanup_error(self):
        """Cleanup exception inside background error handler is swallowed."""
        from farm.api.server import (
            _run_analysis_background,
            active_analyses,
            _active_analyses_thread_lock,
        )

        mock_ctrl = Mock()
        mock_ctrl.start.side_effect = RuntimeError("crash")
        mock_ctrl.cleanup.side_effect = RuntimeError("cleanup failed too")

        with _active_analyses_thread_lock:
            active_analyses["bg5"] = {"status": "pending", "module_name": "test"}

        # Should NOT raise; cleanup error is swallowed
        _run_analysis_background("bg5", mock_ctrl)

        with _active_analyses_thread_lock:
            assert active_analyses["bg5"]["status"] == "error"

    def test_background_analysis_cleanup_old_analyses_error(self):
        """Exception in _cleanup_old_analyses (finally block) is swallowed."""
        from farm.api.server import (
            _run_analysis_background,
            active_analyses,
            _active_analyses_thread_lock,
        )
        from farm.analysis.service import AnalysisResult
        from pathlib import Path

        mock_ctrl = Mock()
        mock_result = Mock(spec=AnalysisResult)
        mock_result.success = True
        mock_result.output_path = Path("test/output")
        mock_result.execution_time = 0.1
        mock_result.cache_hit = False
        mock_result.dataframe = None
        mock_ctrl.get_result.return_value = mock_result

        with _active_analyses_thread_lock:
            active_analyses["bg6"] = {"status": "pending", "module_name": "test"}

        with patch("farm.api.server._cleanup_old_analyses", side_effect=RuntimeError("cleanup crash")):
            # Should NOT raise; cleanup error is swallowed
            _run_analysis_background("bg6", mock_ctrl)

        with _active_analyses_thread_lock:
            assert active_analyses["bg6"]["status"] == "completed"


class TestAdditionalCoverage:
    """Additional tests for specific uncovered lines in server.py."""

    @pytest.fixture(autouse=True)
    def clear_state(self):
        from farm.api.server import (
            active_simulations,
            active_analyses,
            _active_simulations_thread_lock,
            _active_analyses_thread_lock,
        )
        with _active_simulations_thread_lock:
            active_simulations.clear()
        with _active_analyses_thread_lock:
            active_analyses.clear()
        yield
        with _active_simulations_thread_lock:
            active_simulations.clear()
        with _active_analyses_thread_lock:
            active_analyses.clear()

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_get_analysis_endpoint_not_found(self, client):
        """GET /api/simulation/{sim_id}/analysis returns 404 for unknown sim."""
        response = client.get("/api/simulation/unknown_sim/analysis")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_simulation_error_status_in_summary(self, client):
        """list_simulations includes error_message for error-status simulations."""
        from farm.api.server import active_simulations, _active_simulations_thread_lock

        with _active_simulations_thread_lock:
            active_simulations["err1"] = {
                "db_path": "x.db",
                "config": {},
                "created_at": datetime.now().isoformat(),
                "status": "error",
                "error_message": "something went wrong",
            }

        response = client.get("/api/simulations")
        assert response.status_code == 200
        items = {r["sim_id"]: r for r in response.json()["data"]}
        assert items["err1"]["error_message"] == "something went wrong"

    def test_cleanup_with_invalid_ended_at_timestamp(self):
        """Cleanup handles invalid ended_at timestamps gracefully."""
        from farm.api.server import (
            _cleanup_old_analyses,
            active_analyses,
            _active_analyses_thread_lock,
        )

        with _active_analyses_thread_lock:
            active_analyses["bad_ts"] = {
                "status": "completed",
                "ended_at": "not-a-valid-timestamp",  # invalid ISO format
            }

        # Should not raise
        _cleanup_old_analyses()

        with _active_analyses_thread_lock:
            # Entry should still be there (not removed; just skipped)
            assert "bad_ts" in active_analyses

    def test_cleanup_with_controller_cleanup_error(self):
        """Cleanup swallows controller.cleanup() exceptions."""
        from farm.api.server import (
            _cleanup_old_analyses,
            active_analyses,
            _active_analyses_thread_lock,
        )

        bad_ctrl = Mock()
        bad_ctrl.cleanup.side_effect = RuntimeError("cleanup error")

        with _active_analyses_thread_lock:
            active_analyses["ctrl_err"] = {
                "controller": bad_ctrl,
                "status": "completed",
                "ended_at": "2000-01-01T00:00:00",  # very old
            }

        # Should not raise
        _cleanup_old_analyses()

        with _active_analyses_thread_lock:
            assert "ctrl_err" not in active_analyses
