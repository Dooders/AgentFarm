"""Session management module for database operations.

This module provides a session manager class to handle SQLAlchemy session lifecycle
and transaction management in a thread-safe way. It includes context manager support
for automatic session cleanup and error handling.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages database session lifecycle and transactions.

    This class provides methods for creating, managing, and cleaning up database
    sessions in a thread-safe way. It supports both context manager and explicit
    session management patterns.

    Attributes:
        engine: SQLAlchemy database engine
        Session: Thread-local session factory
    """

    def __init__(self, path: Optional[str] = None):
        """Initialize the session manager.

        Args:
            path: Path to database directory. If None, uses default path.
        """
        self.engine = create_engine(
            f"sqlite:///{path}/simulation.db",
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800,
            echo=False,
        )
        self.Session = scoped_session(
            sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
        )

    def create_session(self) -> Session:
        """Create and return a new database session.

        Returns:
            Session: New SQLAlchemy session instance

        Note:
            Prefer using session_scope() context manager over this method
            for automatic cleanup.
        """
        return self.Session()

    def close_session(self, session: Session) -> None:
        """Safely close a database session.

        Args:
            session: The session to close

        Note:
            This method handles rolling back uncommitted transactions
            and cleaning up session resources.
        """
        try:
            session.close()
        except Exception as e:
            logger.error(f"Error closing session: {e}")
        finally:
            self.remove_session()

    def remove_session(self) -> None:
        """Remove the current thread-local session.

        This should be called when the session is no longer needed
        to prevent memory leaks.
        """
        self.Session.remove()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations.

        Yields:
            Session: Database session for use in a with statement

        Raises:
            SQLAlchemyError: If a database error occurs
            Exception: If any other error occurs

        Example:
            >>> with session_manager.session_scope() as session:
            ...     results = session.query(Agent).all()
        """
        session = self.create_session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error in session: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error in session: {e}")
            raise
        finally:
            self.close_session(session)

    def execute_with_retry(self, operation, max_retries: int = 3):
        """Execute a database operation with retry logic.

        Args:
            operation: Function that takes a session parameter and performs database operations
            max_retries: Maximum number of retry attempts, defaults to 3

        Returns:
            Result of the operation if successful

        Raises:
            SQLAlchemyError: If operation fails after all retries
        """
        retries = 0
        last_error = None

        while retries < max_retries:
            try:
                with self.session_scope() as session:
                    result = operation(session)
                    return result
            except SQLAlchemyError as e:
                last_error = e
                retries += 1
                logger.warning(
                    f"Database operation failed (attempt {retries}/{max_retries}): {e}"
                )

        logger.error(f"Operation failed after {max_retries} retries: {last_error}")
        raise last_error

    def __enter__(self) -> Session:
        """Enter context manager, creating a new session.

        Returns:
            Session: New database session
        """
        return self.create_session()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, cleaning up session.

        Args:
            exc_type: Type of exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Traceback of exception that occurred, if any
        """
        self.remove_session()

    def cleanup(self) -> None:
        """Clean up database connections and resources."""
        try:
            # Close all sessions
            self.Session.remove()

            # Dispose of the engine connections
            if hasattr(self, "engine"):
                # Close all connections in the pool
                self.engine.dispose()

                # Remove reference to engine
                delattr(self, "engine")

        except Exception as e:
            logging.error(f"Error during session cleanup: {e}", exc_info=True)
