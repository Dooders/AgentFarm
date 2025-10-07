"""
Tests for protocol compliance.

Verifies that concrete implementations properly implement the defined protocols
from farm.core.interfaces. This ensures type safety and proper abstraction.
"""

import os
import tempfile
import unittest
from typing import Type

from farm.core.interfaces import (
    DatabaseProtocol,
    DataLoggerProtocol,
    RepositoryProtocol,
)
from farm.database.database import SimulationDatabase
from farm.database.repositories.action_repository import ActionRepository
from farm.database.repositories.agent_repository import AgentRepository
from farm.database.repositories.resource_repository import ResourceRepository
from farm.database.session_manager import SessionManager


class TestProtocolCompliance(unittest.TestCase):
    """Test that concrete implementations comply with their respective protocols."""

    def setUp(self):
        """Set up test database and session manager."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_file.close()
        self.db_path = self.temp_file.name

        # Create session manager for repositories
        self.session_manager = SessionManager(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _test_protocol_compliance(self, implementation: object, protocol: Type) -> None:
        """Helper method to test if an implementation complies with a protocol."""
        # Check that the implementation is an instance of the protocol
        self.assertIsInstance(implementation, protocol,
                            f"{implementation.__class__.__name__} does not implement {protocol.__name__}")

        # Check that all protocol methods are implemented
        protocol_methods = [method for method in dir(protocol) if not method.startswith('_')]
        implementation_methods = [method for method in dir(implementation.__class__)
                                if not method.startswith('_') and callable(getattr(implementation.__class__, method))]

        missing_methods = set(protocol_methods) - set(implementation_methods)
        self.assertEqual(missing_methods, set(),
                        f"{implementation.__class__.__name__} is missing methods from {protocol.__name__}: {missing_methods}")

    def test_simulation_database_complies_with_database_protocol(self):
        """Test that SimulationDatabase properly implements DatabaseProtocol."""
        db = SimulationDatabase(self.db_path, simulation_id="test_sim")
        try:
            self._test_protocol_compliance(db, DatabaseProtocol)
        finally:
            db.close()

    def test_data_logger_complies_with_data_logger_protocol(self):
        """Test that DataLogger properly implements DataLoggerProtocol."""
        db = SimulationDatabase(self.db_path, simulation_id="test_sim")
        try:
            logger = db.logger
            self._test_protocol_compliance(logger, DataLoggerProtocol)
        finally:
            db.close()

    def test_agent_repository_complies_with_repository_protocol(self):
        """Test that AgentRepository properly implements RepositoryProtocol."""
        repo = AgentRepository(self.session_manager)
        self._test_protocol_compliance(repo, RepositoryProtocol)

    def test_action_repository_complies_with_repository_protocol(self):
        """Test that ActionRepository properly implements RepositoryProtocol."""
        repo = ActionRepository(self.session_manager)
        self._test_protocol_compliance(repo, RepositoryProtocol)

    def test_resource_repository_complies_with_repository_protocol(self):
        """Test that ResourceRepository properly implements RepositoryProtocol."""
        repo = ResourceRepository(self.session_manager)
        self._test_protocol_compliance(repo, RepositoryProtocol)


if __name__ == "__main__":
    unittest.main()
