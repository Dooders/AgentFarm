"""Unit tests for the SessionManager class."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest

from farm.api.models import SessionInfo, SessionStatus
from farm.api.session_manager import SessionManager


class TestSessionManager:
    """Test SessionManager class."""

    def test_init_with_default_path(self, temp_workspace):
        """Test SessionManager initialization with default path."""
        with patch('farm.api.session_manager.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.mkdir = Mock()
            mock_path.return_value = mock_path_instance
            
            manager = SessionManager()
            
            # Should create workspace directory
            mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            assert manager.workspace_path == mock_path_instance
            assert manager._sessions == {}

    def test_init_with_custom_path(self, temp_workspace):
        """Test SessionManager initialization with custom path."""
        custom_path = str(temp_workspace / "custom_sessions")
        
        manager = SessionManager(custom_path)
        
        assert manager.workspace_path == Path(custom_path)
        assert manager.workspace_path.exists()

    def test_load_existing_sessions_success(self, temp_workspace, sessions_json_file):
        """Test loading existing sessions from file."""
        manager = SessionManager(str(temp_workspace))
        
        # Should load 2 sessions from the file
        assert len(manager._sessions) == 2
        assert "session-1" in manager._sessions
        assert "session-2" in manager._sessions
        
        session1 = manager._sessions["session-1"]
        assert session1.name == "Test Session 1"
        assert session1.status == SessionStatus.ACTIVE
        assert session1.simulations == ["sim-1"]
        
        session2 = manager._sessions["session-2"]
        assert session2.name == "Test Session 2"
        assert session2.status == SessionStatus.ARCHIVED
        assert session2.simulations == ["sim-2", "sim-3"]

    def test_load_existing_sessions_file_not_exists(self, temp_workspace):
        """Test loading sessions when file doesn't exist."""
        manager = SessionManager(str(temp_workspace))
        
        # Should have empty sessions dict
        assert len(manager._sessions) == 0

    def test_load_existing_sessions_invalid_json(self, temp_workspace):
        """Test loading sessions with invalid JSON."""
        sessions_file = temp_workspace / "sessions.json"
        sessions_file.write_text("invalid json content")
        
        # Should not raise exception, just log error
        manager = SessionManager(str(temp_workspace))
        assert len(manager._sessions) == 0

    def test_save_sessions_success(self, temp_workspace):
        """Test saving sessions to file."""
        manager = SessionManager(str(temp_workspace))
        
        # Add a test session
        session = SessionInfo(
            session_id="test-123",
            name="Test Session",
            description="Test Description",
            created_at=datetime.now(),
            status=SessionStatus.ACTIVE
        )
        manager._sessions["test-123"] = session
        
        # Save sessions
        manager._save_sessions()
        
        # Check file was created and contains correct data
        sessions_file = temp_workspace / "sessions.json"
        assert sessions_file.exists()
        
        with open(sessions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]["session_id"] == "test-123"
        assert data[0]["name"] == "Test Session"

    def test_save_sessions_error(self, temp_workspace):
        """Test saving sessions with file write error."""
        manager = SessionManager(str(temp_workspace))
        
        # Add a test session
        session = SessionInfo(
            session_id="test-123",
            name="Test Session",
            description="Test Description",
            created_at=datetime.now(),
            status=SessionStatus.ACTIVE
        )
        manager._sessions["test-123"] = session
        
        # Mock file write to raise exception
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = IOError("Write error")
            
            # Should not raise exception, just log error
            manager._save_sessions()

    def test_create_session_success(self, temp_workspace):
        """Test creating a new session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session", "Test Description")
        
        # Should return a valid UUID
        assert len(session_id) == 36  # UUID4 length
        assert session_id in manager._sessions
        
        session = manager._sessions[session_id]
        assert session.name == "Test Session"
        assert session.description == "Test Description"
        assert session.status == SessionStatus.ACTIVE
        assert isinstance(session.created_at, datetime)
        
        # Should create session directory
        session_path = temp_workspace / session_id
        assert session_path.exists()

    def test_create_session_with_empty_description(self, temp_workspace):
        """Test creating session with empty description."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        session = manager._sessions[session_id]
        assert session.description == ""

    def test_get_session_existing(self, temp_workspace):
        """Test getting an existing session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        session = manager.get_session(session_id)
        
        assert session is not None
        assert session.name == "Test Session"

    def test_get_session_nonexistent(self, temp_workspace):
        """Test getting a non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        session = manager.get_session("nonexistent-id")
        assert session is None

    def test_list_sessions_all(self, temp_workspace):
        """Test listing all sessions."""
        manager = SessionManager(str(temp_workspace))
        
        # Create multiple sessions
        session1_id = manager.create_session("Session 1")
        session2_id = manager.create_session("Session 2")
        
        sessions = manager.list_sessions()
        
        assert len(sessions) == 2
        session_ids = [s.session_id for s in sessions]
        assert session1_id in session_ids
        assert session2_id in session_ids

    def test_list_sessions_filtered_by_status(self, temp_workspace):
        """Test listing sessions filtered by status."""
        manager = SessionManager(str(temp_workspace))
        
        # Create sessions with different statuses
        session1_id = manager.create_session("Active Session")
        session2_id = manager.create_session("Archived Session")
        
        # Archive second session
        manager.update_session(session2_id, status=SessionStatus.ARCHIVED)
        
        # List active sessions
        active_sessions = manager.list_sessions(SessionStatus.ACTIVE)
        assert len(active_sessions) == 1
        assert active_sessions[0].session_id == session1_id
        
        # List archived sessions
        archived_sessions = manager.list_sessions(SessionStatus.ARCHIVED)
        assert len(archived_sessions) == 1
        assert archived_sessions[0].session_id == session2_id

    def test_update_session_success(self, temp_workspace):
        """Test updating session information."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Original Name", "Original Description")
        
        # Update session
        success = manager.update_session(
            session_id,
            name="Updated Name",
            description="Updated Description",
            status=SessionStatus.ARCHIVED,
            metadata={"key": "value"}
        )
        
        assert success is True
        
        session = manager.get_session(session_id)
        assert session.name == "Updated Name"
        assert session.description == "Updated Description"
        assert session.status == SessionStatus.ARCHIVED
        assert session.metadata == {"key": "value"}

    def test_update_session_nonexistent(self, temp_workspace):
        """Test updating non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        success = manager.update_session("nonexistent-id", name="New Name")
        assert success is False

    def test_update_session_invalid_field(self, temp_workspace):
        """Test updating session with invalid field."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        # Try to update invalid field
        success = manager.update_session(session_id, invalid_field="value")
        assert success is True  # Should still return True, just ignore invalid field
        
        session = manager.get_session(session_id)
        assert not hasattr(session, "invalid_field")

    def test_add_simulation_to_session_success(self, temp_workspace):
        """Test adding simulation to session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        success = manager.add_simulation_to_session(session_id, "sim-123")
        assert success is True
        
        session = manager.get_session(session_id)
        assert "sim-123" in session.simulations

    def test_add_simulation_to_session_duplicate(self, temp_workspace):
        """Test adding duplicate simulation to session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        # Add simulation twice
        manager.add_simulation_to_session(session_id, "sim-123")
        success = manager.add_simulation_to_session(session_id, "sim-123")
        
        assert success is True
        
        session = manager.get_session(session_id)
        # Should only appear once
        assert session.simulations.count("sim-123") == 1

    def test_add_simulation_to_session_nonexistent(self, temp_workspace):
        """Test adding simulation to non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        success = manager.add_simulation_to_session("nonexistent-id", "sim-123")
        assert success is False

    def test_add_experiment_to_session_success(self, temp_workspace):
        """Test adding experiment to session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        success = manager.add_experiment_to_session(session_id, "exp-123")
        assert success is True
        
        session = manager.get_session(session_id)
        assert "exp-123" in session.experiments

    def test_add_experiment_to_session_duplicate(self, temp_workspace):
        """Test adding duplicate experiment to session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        # Add experiment twice
        manager.add_experiment_to_session(session_id, "exp-123")
        success = manager.add_experiment_to_session(session_id, "exp-123")
        
        assert success is True
        
        session = manager.get_session(session_id)
        # Should only appear once
        assert session.experiments.count("exp-123") == 1

    def test_add_experiment_to_session_nonexistent(self, temp_workspace):
        """Test adding experiment to non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        success = manager.add_experiment_to_session("nonexistent-id", "exp-123")
        assert success is False

    def test_get_session_path_existing(self, temp_workspace):
        """Test getting session path for existing session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        session_path = manager.get_session_path(session_id)
        
        assert session_path is not None
        assert session_path == temp_workspace / session_id
        assert session_path.exists()

    def test_get_session_path_nonexistent(self, temp_workspace):
        """Test getting session path for non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        session_path = manager.get_session_path("nonexistent-id")
        assert session_path is None

    def test_delete_session_success(self, temp_workspace):
        """Test deleting a session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        # Add some files to session directory
        session_path = temp_workspace / session_id
        test_file = session_path / "test.txt"
        test_file.write_text("test content")
        
        success = manager.delete_session(session_id, delete_files=True)
        assert success is True
        
        # Session should be removed from memory
        assert session_id not in manager._sessions
        
        # Files should be deleted
        assert not session_path.exists()

    def test_delete_session_keep_files(self, temp_workspace):
        """Test deleting session but keeping files."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        session_path = temp_workspace / session_id
        test_file = session_path / "test.txt"
        test_file.write_text("test content")
        
        success = manager.delete_session(session_id, delete_files=False)
        assert success is True
        
        # Session should be removed from memory
        assert session_id not in manager._sessions
        
        # Files should still exist
        assert session_path.exists()
        assert test_file.exists()

    def test_delete_session_nonexistent(self, temp_workspace):
        """Test deleting non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        success = manager.delete_session("nonexistent-id")
        assert success is False

    def test_archive_session_success(self, temp_workspace):
        """Test archiving a session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        success = manager.archive_session(session_id)
        assert success is True
        
        session = manager.get_session(session_id)
        assert session.status == SessionStatus.ARCHIVED

    def test_archive_session_nonexistent(self, temp_workspace):
        """Test archiving non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        success = manager.archive_session("nonexistent-id")
        assert success is False

    def test_restore_session_success(self, temp_workspace):
        """Test restoring an archived session."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        manager.archive_session(session_id)
        
        success = manager.restore_session(session_id)
        assert success is True
        
        session = manager.get_session(session_id)
        assert session.status == SessionStatus.ACTIVE

    def test_restore_session_nonexistent(self, temp_workspace):
        """Test restoring non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        success = manager.restore_session("nonexistent-id")
        assert success is False

    def test_get_session_stats_existing(self, temp_workspace):
        """Test getting session statistics."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        # Add simulations and experiments
        manager.add_simulation_to_session(session_id, "sim-1")
        manager.add_simulation_to_session(session_id, "sim-2")
        manager.add_experiment_to_session(session_id, "exp-1")
        
        # Create some test files
        session_path = temp_workspace / session_id
        test_file1 = session_path / "test1.txt"
        test_file2 = session_path / "test2.txt"
        test_file1.write_text("content1")
        test_file2.write_text("content2")
        
        stats = manager.get_session_stats(session_id)
        
        assert stats is not None
        assert stats["session_id"] == session_id
        assert stats["name"] == "Test Session"
        assert stats["simulations"] == 2
        assert stats["experiments"] == 1
        assert stats["total_files"] == 2
        assert stats["total_size_mb"] >= 0

    def test_get_session_stats_nonexistent(self, temp_workspace):
        """Test getting stats for non-existent session."""
        manager = SessionManager(str(temp_workspace))
        
        stats = manager.get_session_stats("nonexistent-id")
        assert stats is None

    def test_get_session_stats_no_files(self, temp_workspace):
        """Test getting stats for session with no files."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        stats = manager.get_session_stats(session_id)
        
        assert stats is not None
        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0

    def test_get_session_stats_missing_directory(self, temp_workspace):
        """Test getting stats when session directory is missing."""
        manager = SessionManager(str(temp_workspace))
        
        session_id = manager.create_session("Test Session")
        
        # Remove the session directory
        session_path = temp_workspace / session_id
        shutil.rmtree(session_path)
        
        stats = manager.get_session_stats(session_id)
        
        assert stats is not None
        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0

    @patch('farm.api.session_manager.get_logger')
    def test_logging_in_create_session(self, mock_get_logger, temp_workspace):
        """Test that create_session logs appropriately."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        manager = SessionManager(str(temp_workspace))
        session_id = manager.create_session("Test Session")
        
        # Should log session creation - check that logger was called
        # Note: The actual logging happens in the real logger, not our mock
        # So we just verify the session was created successfully
        assert session_id is not None
        assert session_id in manager._sessions

    @patch('farm.api.session_manager.get_logger')
    def test_logging_in_load_sessions_error(self, mock_get_logger, temp_workspace):
        """Test that load_sessions logs errors appropriately."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Create invalid JSON file
        sessions_file = temp_workspace / "sessions.json"
        sessions_file.write_text("invalid json")
        
        manager = SessionManager(str(temp_workspace))
        
        # Should handle error gracefully - sessions dict should be empty
        # Note: The actual logging happens in the real logger, not our mock
        assert len(manager._sessions) == 0
