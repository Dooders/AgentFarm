"""Session management system for the unified AgentFarm API.

This module provides session management functionality, allowing agentic systems
to organize their work into logical sessions with persistent state.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from farm.api.models import SessionInfo, SessionStatus
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages research sessions and their associated data."""

    def __init__(self, workspace_path: Optional[str] = None):
        """Initialize session manager.

        Args:
            workspace_path: Base path for session storage. If None, uses default.
        """
        self.workspace_path = Path(workspace_path or "sessions")
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # In-memory session cache
        self._sessions: Dict[str, SessionInfo] = {}
        self._load_existing_sessions()

    def _load_existing_sessions(self) -> None:
        """Load existing sessions from disk."""
        try:
            sessions_file = self.workspace_path / "sessions.json"
            if sessions_file.exists():
                with open(sessions_file, "r", encoding="utf-8") as f:
                    sessions_data = json.load(f)

                for session_data in sessions_data:
                    session_info = SessionInfo(
                        session_id=session_data["session_id"],
                        name=session_data["name"],
                        description=session_data["description"],
                        created_at=datetime.fromisoformat(session_data["created_at"]),
                        status=SessionStatus(session_data["status"]),
                        simulations=session_data.get("simulations", []),
                        experiments=session_data.get("experiments", []),
                        metadata=session_data.get("metadata", {}),
                    )
                    self._sessions[session_info.session_id] = session_info

                logger.info(f"Loaded {len(self._sessions)} existing sessions")
        except Exception as e:
            logger.error(f"Error loading existing sessions: {e}")

    def _save_sessions(self) -> None:
        """Save sessions to disk."""
        try:
            sessions_file = self.workspace_path / "sessions.json"
            sessions_data = [session.to_dict() for session in self._sessions.values()]

            with open(sessions_file, "w", encoding="utf-8") as f:
                json.dump(sessions_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving sessions: {e}")

    def create_session(self, name: str, description: str = "") -> str:
        """Create a new session.

        Args:
            name: Name of the session
            description: Optional description

        Returns:
            session_id: Unique identifier for the session
        """
        session_id = str(uuid.uuid4())
        session_path = self.workspace_path / session_id
        session_path.mkdir(exist_ok=True)

        session_info = SessionInfo(
            session_id=session_id,
            name=name,
            description=description,
            created_at=datetime.now(),
            status=SessionStatus.ACTIVE,
        )

        self._sessions[session_id] = session_info
        self._save_sessions()

        logger.info(f"Created session: {name} ({session_id})")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo if found, None otherwise
        """
        return self._sessions.get(session_id)

    def list_sessions(
        self, status: Optional[SessionStatus] = None
    ) -> List[SessionInfo]:
        """List all sessions, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of SessionInfo objects
        """
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session information.

        Args:
            session_id: Session identifier
            **kwargs: Fields to update

        Returns:
            True if updated successfully, False otherwise
        """
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]

        # Update allowed fields
        allowed_fields = ["name", "description", "status", "metadata"]
        for field, value in kwargs.items():
            if field in allowed_fields:
                setattr(session, field, value)

        self._save_sessions()
        logger.info(f"Updated session: {session_id}")
        return True

    def add_simulation_to_session(self, session_id: str, simulation_id: str) -> bool:
        """Add a simulation to a session.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier

        Returns:
            True if added successfully, False otherwise
        """
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        if simulation_id not in session.simulations:
            session.simulations.append(simulation_id)
            self._save_sessions()
            logger.info(f"Added simulation {simulation_id} to session {session_id}")

        return True

    def add_experiment_to_session(self, session_id: str, experiment_id: str) -> bool:
        """Add an experiment to a session.

        Args:
            session_id: Session identifier
            experiment_id: Experiment identifier

        Returns:
            True if added successfully, False otherwise
        """
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        if experiment_id not in session.experiments:
            session.experiments.append(experiment_id)
            self._save_sessions()
            logger.info(f"Added experiment {experiment_id} to session {session_id}")

        return True

    def get_session_path(self, session_id: str) -> Optional[Path]:
        """Get the file system path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path object if session exists, None otherwise
        """
        if session_id not in self._sessions:
            return None

        return self.workspace_path / session_id

    def delete_session(self, session_id: str, delete_files: bool = False) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier
            delete_files: Whether to delete associated files

        Returns:
            True if deleted successfully, False otherwise
        """
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]

        # Delete files if requested
        if delete_files:
            session_path = self.workspace_path / session_id
            if session_path.exists():
                shutil.rmtree(session_path)
                logger.info(f"Deleted session files: {session_path}")

        # Remove from memory
        del self._sessions[session_id]
        self._save_sessions()

        logger.info(f"Deleted session: {session.name} ({session_id})")
        return True

    def archive_session(self, session_id: str) -> bool:
        """Archive a session.

        Args:
            session_id: Session identifier

        Returns:
            True if archived successfully, False otherwise
        """
        return self.update_session(session_id, status=SessionStatus.ARCHIVED)

    def restore_session(self, session_id: str) -> bool:
        """Restore an archived session.

        Args:
            session_id: Session identifier

        Returns:
            True if restored successfully, False otherwise
        """
        return self.update_session(session_id, status=SessionStatus.ACTIVE)

    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics, None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None

        session_path = self.get_session_path(session_id)
        if not session_path or not session_path.exists():
            return {
                "session_id": session_id,
                "name": session.name,
                "simulations": len(session.simulations),
                "experiments": len(session.experiments),
                "total_files": 0,
                "total_size_mb": 0,
            }

        # Calculate file statistics
        total_files = 0
        total_size = 0

        for root, dirs, files in os.walk(session_path):
            total_files += len(files)
            for file in files:
                file_path = Path(root) / file
                if file_path.exists():
                    total_size += file_path.stat().st_size

        return {
            "session_id": session_id,
            "name": session.name,
            "simulations": len(session.simulations),
            "experiments": len(session.experiments),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
