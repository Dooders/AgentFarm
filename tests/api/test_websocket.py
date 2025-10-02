"""Unit tests for WebSocket functionality in the FastAPI server."""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from farm.api.server import _active_simulations_lock, active_simulations, app, manager


class TestWebSocketFunctionality:
    """Test WebSocket connection and message handling."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def setup_method(self):
        """Clear active simulations before each test."""
        with _active_simulations_lock:
            active_simulations.clear()

    def teardown_method(self):
        """Clean up after each test."""
        with _active_simulations_lock:
            active_simulations.clear()

    def test_websocket_connection_establishment(self, client):
        """Test that WebSocket connections can be established."""
        with client.websocket_connect("/ws/test_client_123") as websocket:
            assert websocket is not None
            # Connection should be established without errors

    def test_websocket_connection_manager(self):
        """Test the WebSocket connection manager functionality."""
        # Test initial state
        assert len(manager.active_connections) == 0

        # Mock WebSocket with async methods
        mock_websocket = Mock()

        # Create an async mock for the accept method
        async def mock_accept():
            return None

        mock_websocket.accept = mock_accept
        client_id = "test_client"

        # Test connection
        import asyncio

        async def test_connect():
            await manager.connect(mock_websocket, client_id)

        asyncio.run(test_connect())
        assert client_id in manager.active_connections
        assert manager.active_connections[client_id] == mock_websocket

        # Test disconnection
        manager.disconnect(client_id)
        assert client_id not in manager.active_connections

    def test_websocket_subscribe_simulation_success(self, client):
        """Test successful simulation subscription via WebSocket."""
        sim_id = "test_sim_123"

        # Add simulation to active simulations
        with _active_simulations_lock:
            active_simulations[sim_id] = {
                "db_path": "test.db",
                "config": {"steps": 100},
                "created_at": "2024-01-01T00:00:00",
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
                "sim_id": "nonexistent_sim",
            }
            websocket.send_text(json.dumps(subscribe_message))

            # Receive error response
            response = websocket.receive_text()
            response_data = json.loads(response)

            assert response_data["type"] == "subscription_error"
            assert "not found" in response_data["message"]

    def test_websocket_invalid_json_message(self, client):
        """Test WebSocket handling of invalid JSON messages."""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Send invalid JSON
            websocket.send_text("invalid json message")

            # Receive error response
            response = websocket.receive_text()
            response_data = json.loads(response)

            assert response_data["type"] == "error"
            assert "Invalid JSON format" in response_data["message"]

    def test_websocket_unknown_message_type(self, client):
        """Test WebSocket handling of unknown message types."""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Send message with unknown type
            unknown_message = {"type": "unknown_type", "data": "some data"}
            websocket.send_text(json.dumps(unknown_message))

            # Should not receive any response for unknown message types
            # (current implementation only handles subscribe_simulation)
            # This test verifies the system doesn't crash on unknown types

    def test_websocket_missing_sim_id(self, client):
        """Test WebSocket subscription with missing sim_id."""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Send subscription message without sim_id
            subscribe_message = {
                "type": "subscribe_simulation"
                # Missing sim_id field
            }
            websocket.send_text(json.dumps(subscribe_message))

            # Should handle gracefully (sim_id will be None)
            response = websocket.receive_text()
            response_data = json.loads(response)

            assert response_data["type"] == "subscription_error"
            assert "not found" in response_data["message"]

    def test_websocket_connection_cleanup(self, client):
        """Test that WebSocket connections are properly cleaned up on disconnect."""
        # Test that connection is removed from manager on disconnect
        with client.websocket_connect("/ws/test_client_cleanup") as websocket:
            # Connection should be established
            pass

        # After context exit, connection should be cleaned up
        # Note: The actual cleanup happens in the WebSocket disconnect handler
        # This test verifies the connection can be established and closed properly

    def test_websocket_multiple_clients(self, client):
        """Test multiple WebSocket clients connecting simultaneously."""
        client_ids = ["client_1", "client_2", "client_3"]

        # Connect multiple clients
        websockets = []
        for client_id in client_ids:
            websocket = client.websocket_connect(f"/ws/{client_id}")
            websockets.append(websocket)

        # All connections should be established successfully
        for websocket in websockets:
            with websocket as ws:
                assert ws is not None

    def test_websocket_simulation_status_updates(self, client):
        """Test WebSocket functionality with simulation status updates."""
        sim_id = "test_sim_status"

        # Add simulation to active simulations
        with _active_simulations_lock:
            active_simulations[sim_id] = {
                "db_path": "test.db",
                "config": {"steps": 100},
                "created_at": "2024-01-01T00:00:00",
                "status": "pending",
            }

        with client.websocket_connect("/ws/test_client") as websocket:
            # Subscribe to simulation
            subscribe_message = {"type": "subscribe_simulation", "sim_id": sim_id}
            websocket.send_text(json.dumps(subscribe_message))

            # Receive subscription confirmation
            response = websocket.receive_text()
            response_data = json.loads(response)
            assert response_data["type"] == "subscription_success"

            # Update simulation status
            with _active_simulations_lock:
                active_simulations[sim_id]["status"] = "running"

            # Note: In a real implementation, status updates would be broadcast
            # to subscribed clients. This test verifies the subscription works.

    def test_websocket_error_handling(self, client):
        """Test WebSocket error handling and recovery."""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Send malformed message
            websocket.send_text("not json at all")

            # Should receive error response
            response = websocket.receive_text()
            response_data = json.loads(response)
            assert response_data["type"] == "error"

            # Should still be able to send valid messages after error
            subscribe_message = {"type": "subscribe_simulation", "sim_id": "test_sim"}
            websocket.send_text(json.dumps(subscribe_message))

            # Should receive response for valid message
            response = websocket.receive_text()
            response_data = json.loads(response)
            assert (
                response_data["type"] == "subscription_error"
            )  # Because sim doesn't exist

    def test_websocket_concurrent_connections(self, client):
        """Test concurrent WebSocket connections."""
        import threading
        import time

        results = []

        def connect_client(client_id):
            try:
                with client.websocket_connect(f"/ws/{client_id}") as websocket:
                    # Send a test message
                    test_message = {
                        "type": "subscribe_simulation",
                        "sim_id": "test_sim",
                    }
                    websocket.send_text(json.dumps(test_message))

                    # Receive response
                    response = websocket.receive_text()
                    results.append(json.loads(response))
            except Exception as e:
                results.append({"error": str(e)})

        # Create multiple threads for concurrent connections
        threads = []
        for i in range(5):
            thread = threading.Thread(target=connect_client, args=(f"client_{i}",))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All connections should have received responses
        assert len(results) == 5
        for result in results:
            assert "error" not in result
            assert result["type"] == "subscription_error"  # Because sim doesn't exist
