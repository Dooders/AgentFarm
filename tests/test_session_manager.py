import unittest
from unittest.mock import Mock, patch
import logging
from sqlalchemy.exc import SQLAlchemyError
from farm.database.session_manager import SessionManager


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for SessionManager tests."""
        self.test_db_url = "sqlite:///test.db"
        self.session_manager = SessionManager(self.test_db_url)

    def tearDown(self):
        """Clean up after tests."""
        self.session_manager.cleanup()

    def test_init_with_default_url(self):
        """Test initialization with default database URL."""
        manager = SessionManager()
        self.assertIsNotNone(manager.engine)
        self.assertIsNotNone(manager.Session)
        manager.cleanup()

    def test_init_with_custom_url(self):
        """Test initialization with custom database URL."""
        manager = SessionManager("sqlite:///custom.db")
        self.assertIsNotNone(manager.engine)
        self.assertIsNotNone(manager.Session)
        manager.cleanup()

    def test_create_session(self):
        """Test creation of new database session."""
        session = self.session_manager.create_session()
        self.assertIsNotNone(session)
        session.close()

    def test_close_session(self):
        """Test proper closing of database session."""
        # Create a mock session
        mock_session = Mock()
        
        # Close the session
        self.session_manager.close_session(mock_session)
        
        # Verify close was called
        mock_session.close.assert_called_once()

    def test_close_session_with_error(self):
        """Test session closing with error handling."""
        session = Mock()
        session.close.side_effect = Exception("Test error")

        with self.assertLogs(level="ERROR") as log:
            self.session_manager.close_session(session)
            self.assertIn("Error closing session", log.output[0])

    def test_session_scope_success(self):
        """Test successful use of session_scope context manager."""
        with self.session_manager.session_scope() as session:
            self.assertIsNotNone(session)
            self.assertTrue(session.is_active)

    def test_session_scope_with_sqlalchemy_error(self):
        """Test session_scope handling of SQLAlchemy errors."""
        with self.assertRaises(SQLAlchemyError):
            with self.session_manager.session_scope() as session:
                raise SQLAlchemyError("Test database error")

    def test_session_scope_with_general_error(self):
        """Test session_scope handling of general errors."""
        with self.assertRaises(ValueError):
            with self.session_manager.session_scope() as session:
                raise ValueError("Test error")

    def test_execute_with_retry_success(self):
        """Test successful execution with retry logic."""

        def operation(session):
            return "success"

        result = self.session_manager.execute_with_retry(operation)
        self.assertEqual(result, "success")

    def test_execute_with_retry_failure(self):
        """Test retry logic with persistent failure."""

        def failing_operation(session):
            raise SQLAlchemyError("Database error")

        with self.assertRaises(SQLAlchemyError):
            self.session_manager.execute_with_retry(failing_operation, max_retries=2)

    def test_execute_with_retry_eventual_success(self):
        """Test retry logic with eventual success."""
        attempt_count = 0

        def eventually_successful_operation(session):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise SQLAlchemyError("Temporary error")
            return "success"

        result = self.session_manager.execute_with_retry(
            eventually_successful_operation, max_retries=3
        )
        self.assertEqual(result, "success")
        self.assertEqual(attempt_count, 2)

    def test_context_manager_usage(self):
        """Test using SessionManager as a context manager."""
        with self.session_manager as session:
            self.assertIsNotNone(session)
            self.assertTrue(session.is_active)

    def test_cleanup(self):
        """Test cleanup of database resources."""
        self.session_manager.cleanup()
        # Verify engine is disposed
        self.assertFalse(self.session_manager.engine.pool.checkedin())

    def test_cleanup_with_error(self):
        """Test cleanup error handling."""
        self.session_manager.engine.dispose = Mock(
            side_effect=Exception("Cleanup error")
        )

        with self.assertLogs(level="ERROR") as log:
            self.session_manager.cleanup()
            self.assertIn("Error during cleanup", log.output[0])

    @patch("farm.database.session_manager.create_engine")
    def test_engine_configuration(self, mock_create_engine):
        """Test database engine configuration parameters."""
        SessionManager(self.test_db_url)

        mock_create_engine.assert_called_once_with(
            self.test_db_url,
            poolclass=unittest.mock.ANY,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800,
            echo=False,
        )

    def test_remove_session(self):
        """Test removal of thread-local session."""
        session = self.session_manager.create_session()
        self.session_manager.remove_session()
        # Verify session is removed from thread-local storage
        with self.assertRaises(Exception):
            session.query("test")


if __name__ == "__main__":
    unittest.main()
