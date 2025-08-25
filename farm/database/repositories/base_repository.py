from typing import Any, Generic, List, Optional, Type, TypeVar

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from farm.database.session_manager import SessionManager

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Base repository class with common CRUD operations and session management."""

    def __init__(self, session_manager: SessionManager, model: Type[T]):
        """Initialize the repository with a session manager and model type.
        Parameters
        ----------
        session_manager : SessionManager
            The session manager instance to use for executing queries
        model : Type[T]
            The model class representing the database table
        """
        self.session_manager = session_manager
        self.model = model

    def add(self, entity: T) -> None:
        """Add a new entity to the database.
        Parameters
        ----------
        entity : T
            The entity to add
        """

        def _add(session: Session):
            session.add(entity)

        self._execute_in_transaction(_add)

    def get_by_id(self, entity_id: int) -> Optional[T]:
        """Retrieve an entity by its ID.
        Parameters
        ----------
        entity_id : int
            The ID of the entity to retrieve
        Returns
        -------
        Optional[T]
            The retrieved entity, or None if not found
        """

        def _get_by_id(session: Session) -> Optional[T]:
            return session.query(self.model).get(entity_id)

        return self._execute_in_transaction(_get_by_id)

    def update(self, entity: T) -> None:
        """Update an existing entity in the database.
        Parameters
        ----------
        entity : T
            The entity to update
        """

        def _update(session: Session):
            session.merge(entity)

        self._execute_in_transaction(_update)

    def delete(self, entity: T) -> None:
        """Delete an entity from the database.
        Parameters
        ----------
        entity : T
            The entity to delete
        """

        def _delete(session: Session):
            session.delete(entity)

        self._execute_in_transaction(_delete)

    def _execute_in_transaction(self, func: callable) -> Any:
        """Execute database operations within a transaction with error handling.
        Parameters
        ----------
        func : callable
            Function that takes a session argument and performs database operations
        Returns
        -------
        Any
            Result of the executed function
        Raises
        ------
        SQLAlchemyError
            For database-related errors
        """
        return self.session_manager.execute_with_retry(func)
