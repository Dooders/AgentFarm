"""Session management module for database operations.

This module provides a session manager class to handle SQLAlchemy session lifecycle
and transaction management in a thread-safe way. It includes context manager support
for automatic session cleanup and error handling.

Example
-------
>>> with SessionManager() as session:
...     results = session.query(Agent).all()

>>> session_manager = SessionManager()
>>> session = session_manager.get_session()
>>> try:
...     results = session.query(Agent).all()
...     session.commit()
... finally:
...     session_manager.close_session(session)
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session, sessionmaker

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages database session lifecycle and transactions.

    This class provides methods for creating, managing, and cleaning up database
    sessions in a thread-safe way. It supports both context manager and explicit
    session management patterns.

    Attributes
    ----------
    engine : Engine
        SQLAlchemy database engine
    Session : scoped_session
        Thread-local session factory

    Methods
    -------
    get_session() -> Session
        Create and return a new database session
    close_session(session: Session) -> None
        Safely close a database session
    remove_session() -> None
        Remove the current thread-local session
    """

    def __init__(self, db_url: Optional[str] = None):
        """Initialize the session manager.

        Parameters
        ----------
        db_url : Optional[str]
            SQLAlchemy database URL. If None, uses SQLite with default path.
        """
        self.engine = create_engine(db_url or "sqlite:///simulation.db")
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def get_session(self) -> Session:
        """Create and return a new database session.

        Returns
        -------
        Session
            New SQLAlchemy session instance

        Notes
        -----
        The caller is responsible for closing the session when finished.
        Consider using the context manager interface for automatic cleanup.
        """
        return self.Session()

    def close_session(self, session: Session) -> None:
        """Safely close a database session.

        Parameters
        ----------
        session : Session
            The session to close

        Notes
        -----
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

        Yields
        ------
        Session
            Database session for use in a with statement

        Notes
        -----
        This method should be used with a 'with' statement to ensure
        proper session cleanup.

        Example
        -------
        >>> with session_manager.session_scope() as session:
        ...     results = session.query(Agent).all()
        """
        session = self.get_session()
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

    def execute_with_retry(
        self, operation: callable, max_retries: int = 3
    ) -> Optional[any]:
        """Execute a database operation with retry logic.

        Parameters
        ----------
        operation : callable
            Function that takes a session parameter and performs database operations
        max_retries : int, optional
            Maximum number of retry attempts, by default 3

        Returns
        -------
        Optional[any]
            Result of the operation if successful

        Raises
        ------
        SQLAlchemyError
            If operation fails after all retries
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

        Returns
        -------
        Session
            New database session
        """
        return self.get_session()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, cleaning up session.

        Parameters
        ----------
        exc_type : Optional[Type[BaseException]]
            Type of exception that occurred, if any
        exc_val : Optional[BaseException]
            Exception instance that occurred, if any
        exc_tb : Optional[TracebackType]
            Traceback of exception that occurred, if any
        """
        self.remove_session()
