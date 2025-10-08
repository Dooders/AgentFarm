"""Unit tests for background tasks and error handling in the FastAPI server."""

import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from farm.api.server import (
    _active_simulations_thread_lock,
    _run_simulation_background,
    active_simulations,
)


class TestBackgroundTasks:
    """Test background simulation tasks and error handling."""

    def setup_method(self):
        """Clear active simulations before each test."""
        with _active_simulations_thread_lock:
            active_simulations.clear()

    def teardown_method(self):
        """Clean up after each test."""
        with _active_simulations_thread_lock:
            active_simulations.clear()

    def test_background_simulation_success(self, temp_workspace):
        """Test successful background simulation execution."""
        sim_id = "test_sim_success"
        db_path = str(temp_workspace / "test.db")

        # Create mock config
        mock_config = Mock()
        mock_config.simulation_steps = 100

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": db_path,
                "config": {"steps": 100},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock run_simulation to succeed
        with patch("farm.api.server.run_simulation") as mock_run_sim:
            mock_run_sim.return_value = None

            # Run background task
            _run_simulation_background(sim_id, mock_config, db_path)

            # Verify simulation status was updated
            with _active_simulations_thread_lock:
                assert active_simulations[sim_id]["status"] == "completed"
                assert "ended_at" in active_simulations[sim_id]
                assert active_simulations[sim_id]["ended_at"] is not None

    def test_background_simulation_error(self, temp_workspace):
        """Test background simulation error handling."""
        sim_id = "test_sim_error"
        db_path = str(temp_workspace / "test.db")

        # Create mock config
        mock_config = Mock()
        mock_config.simulation_steps = 100

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": db_path,
                "config": {"steps": 100},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock run_simulation to raise an error
        with patch("farm.api.server.run_simulation") as mock_run_sim:
            mock_run_sim.side_effect = Exception("Simulation failed")

            # Run background task
            _run_simulation_background(sim_id, mock_config, db_path)

            # Verify error status was set
            with _active_simulations_thread_lock:
                assert active_simulations[sim_id]["status"] == "error"
                assert "error_message" in active_simulations[sim_id]
                assert (
                    active_simulations[sim_id]["error_message"] == "Simulation failed"
                )

    def test_background_simulation_missing_simulation(self, temp_workspace):
        """Test background simulation when simulation is not in active_simulations."""
        sim_id = "test_sim_missing"
        db_path = str(temp_workspace / "test.db")

        # Create mock config
        mock_config = Mock()
        mock_config.simulation_steps = 100

        # Don't add simulation to active_simulations

        # Mock run_simulation
        with patch("farm.api.server.run_simulation") as mock_run_sim:
            mock_run_sim.return_value = None

            # Run background task - should not crash
            _run_simulation_background(sim_id, mock_config, db_path)

            # run_simulation should still be called
            mock_run_sim.assert_called_once()

    def test_background_simulation_status_updates(self, temp_workspace):
        """Test that simulation status is updated during execution."""
        sim_id = "test_sim_status"
        db_path = str(temp_workspace / "test.db")

        # Create mock config
        mock_config = Mock()
        mock_config.simulation_steps = 100

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": db_path,
                "config": {"steps": 100},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock run_simulation with a delay to test status updates
        def mock_run_simulation(*args, **kwargs):
            # Check that status was updated to "running"
            with _active_simulations_thread_lock:
                if sim_id in active_simulations:
                    assert active_simulations[sim_id]["status"] == "running"
            time.sleep(0.1)  # Small delay to simulate work

        with patch("farm.api.server.run_simulation", side_effect=mock_run_simulation):
            # Run background task
            _run_simulation_background(sim_id, mock_config, db_path)

            # Verify final status
            with _active_simulations_thread_lock:
                assert active_simulations[sim_id]["status"] == "completed"

    def test_background_simulation_thread_safety(self, temp_workspace):
        """Test thread safety of background simulation tasks."""
        sim_ids = ["sim_1", "sim_2", "sim_3"]
        db_paths = [str(temp_workspace / f"test_{i}.db") for i in range(3)]

        # Create mock configs
        mock_configs = [Mock() for _ in range(3)]
        for config in mock_configs:
            config.simulation_steps = 100

        # Add simulations to active simulations
        with _active_simulations_thread_lock:
            for i, sim_id in enumerate(sim_ids):
                active_simulations[sim_id] = {
                    "db_path": db_paths[i],
                    "config": {"steps": 100},
                    "created_at": datetime.now().isoformat(),
                    "status": "pending",
                }

        # Mock run_simulation
        with patch("farm.api.server.run_simulation") as mock_run_sim:
            mock_run_sim.return_value = None

            # Run multiple background tasks concurrently
            threads = []
            for i, sim_id in enumerate(sim_ids):
                thread = threading.Thread(
                    target=_run_simulation_background,
                    args=(sim_id, mock_configs[i], db_paths[i]),
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Verify all simulations completed successfully
            with _active_simulations_thread_lock:
                for sim_id in sim_ids:
                    assert active_simulations[sim_id]["status"] == "completed"
                    assert "ended_at" in active_simulations[sim_id]

    def test_background_simulation_logging(self, temp_workspace):
        """Test that background simulation tasks log appropriately."""
        sim_id = "test_sim_logging"
        db_path = str(temp_workspace / "test.db")

        # Create mock config
        mock_config = Mock()
        mock_config.simulation_steps = 100

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": db_path,
                "config": {"steps": 100},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock logger and run_simulation
        with patch("farm.api.server.logger") as mock_logger, patch(
            "farm.api.server.run_simulation"
        ) as mock_run_sim:

            mock_run_sim.return_value = None

            # Run background task
            _run_simulation_background(sim_id, mock_config, db_path)

            # Verify no error logging occurred (success case)
            mock_logger.error.assert_not_called()

    def test_background_simulation_error_logging(self, temp_workspace):
        """Test that background simulation errors are logged."""
        sim_id = "test_sim_error_logging"
        db_path = str(temp_workspace / "test.db")

        # Create mock config
        mock_config = Mock()
        mock_config.simulation_steps = 100

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": db_path,
                "config": {"steps": 100},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock logger and run_simulation to raise error
        with patch("farm.api.server.logger") as mock_logger, patch(
            "farm.api.server.run_simulation"
        ) as mock_run_sim:

            mock_run_sim.side_effect = Exception("Test error")

            # Run background task
            _run_simulation_background(sim_id, mock_config, db_path)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "background_simulation_failed" in call_args[0][0]
            assert call_args[1]["simulation_id"] == sim_id
            assert call_args[1]["error_message"] == "Test error"

    def test_background_simulation_config_usage(self, temp_workspace):
        """Test that background simulation uses the provided config correctly."""
        sim_id = "test_sim_config"
        db_path = str(temp_workspace / "test.db")

        # Create mock config with specific values
        mock_config = Mock()
        mock_config.simulation_steps = 500

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": db_path,
                "config": {"steps": 500},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock run_simulation to verify it's called with correct parameters
        with patch("farm.api.server.run_simulation") as mock_run_sim:
            mock_run_sim.return_value = None

            # Run background task
            _run_simulation_background(sim_id, mock_config, db_path)

            # Verify run_simulation was called with correct parameters
            mock_run_sim.assert_called_once()
            call_args = mock_run_sim.call_args
            assert call_args[1]["num_steps"] == 500
            assert call_args[1]["config"] == mock_config
            assert call_args[1]["path"] == str(temp_workspace)

    def test_background_simulation_database_path_handling(self, temp_workspace):
        """Test that background simulation handles database paths correctly."""
        sim_id = "test_sim_db_path"
        db_path = str(temp_workspace / "nested" / "test.db")

        # Create mock config
        mock_config = Mock()
        mock_config.simulation_steps = 100

        # Add simulation to active simulations
        with _active_simulations_thread_lock:
            active_simulations[sim_id] = {
                "db_path": db_path,
                "config": {"steps": 100},
                "created_at": datetime.now().isoformat(),
                "status": "pending",
            }

        # Mock run_simulation
        with patch("farm.api.server.run_simulation") as mock_run_sim:
            mock_run_sim.return_value = None

            # Run background task
            _run_simulation_background(sim_id, mock_config, db_path)

            # Verify run_simulation was called with correct path
            mock_run_sim.assert_called_once()
            call_args = mock_run_sim.call_args
            expected_path = str(temp_workspace / "nested")
            assert call_args[1]["path"] == expected_path
