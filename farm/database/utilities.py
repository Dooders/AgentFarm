"""Utility functions for database operations and data handling.

This module provides helper functions used across the database package for common
operations like JSON handling, data conversion, and schema management.

Functions
---------
safe_json_loads : Safely parse JSON string with error handling
as_dict : Convert SQLAlchemy model instance to dictionary
format_position : Format position tuple to string
parse_position : Parse position string back to tuple
create_database_schema : Helper for creating database tables and indexes
validate_export_format : Validate requested export format
format_agent_state : Format agent state data for database storage
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def safe_json_loads(data: str) -> Optional[Dict]:
    """Safely parse JSON string with error handling.

    Parameters
    ----------
    data : str
        JSON string to parse

    Returns
    -------
    Optional[Dict]
        Parsed JSON data or None if parsing fails
    """
    if not data:
        return None

    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return None


def as_dict(obj: Any) -> Dict:
    """Convert SQLAlchemy model instance to dictionary.

    Parameters
    ----------
    obj : Any
        SQLAlchemy model instance

    Returns
    -------
    Dict
        Dictionary representation of model
    """
    return {col.name: getattr(obj, col.name) for col in obj.__table__.columns}


def format_position(position: Tuple[float, float]) -> str:
    """Format position tuple to string representation.

    Parameters
    ----------
    position : Tuple[float, float]
        (x, y) position coordinates

    Returns
    -------
    str
        Formatted position string "x, y"
    """
    return f"{position[0]}, {position[1]}"


def parse_position(position_str: str) -> Tuple[float, float]:
    """Parse position string back to coordinate tuple.

    Parameters
    ----------
    position_str : str
        Position string in format "x, y"

    Returns
    -------
    Tuple[float, float]
        Position coordinates (x, y)
    """
    try:
        x, y = position_str.split(",")
        return (float(x.strip()), float(y.strip()))
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing position string '{position_str}': {e}")
        return (0.0, 0.0)


def create_database_schema(engine: Any, base: Any) -> None:
    """Create database tables and indexes.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy database engine
    base : sqlalchemy.ext.declarative.api.DeclarativeMeta
        SQLAlchemy declarative base class

    Raises
    ------
    SQLAlchemyError
        If schema creation fails
    """
    try:
        base.metadata.create_all(engine)
    except SQLAlchemyError as e:
        logger.error(f"Error creating database schema: {e}")
        raise


def validate_export_format(format: str) -> bool:
    """Validate requested export format is supported.

    Parameters
    ----------
    format : str
        Export format to validate

    Returns
    -------
    bool
        True if format is supported, False otherwise
    """
    supported_formats = {"csv", "excel", "json", "parquet"}
    return format.lower() in supported_formats


def format_agent_state(
    agent_id: str, step: int, state_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Format agent state data for database storage.

    Parameters
    ----------
    agent_id : str
        ID of the agent
    step : int
        Current simulation step
    state_data : Dict[str, Any]
        Raw state data to format

    Returns
    -------
    Dict[str, Any]
        Formatted state data ready for database
    """
    position = state_data.get("position", (0.0, 0.0))
    return {
        "step_number": step,
        "agent_id": agent_id,
        "current_health": state_data.get("current_health", 0.0),
        "starting_health": state_data.get("starting_health", 0.0),
        "resource_level": state_data.get("resource_level", 0.0),
        "position_x": position[0],
        "position_y": position[1],
        "is_defending": state_data.get("is_defending", False),
        "total_reward": state_data.get("total_reward", 0.0),
        "starvation_threshold": state_data.get("starvation_threshold", 0),
        "age": step,
    }


def execute_with_retry(
    session: Session, operation: callable, max_retries: int = 3
) -> Any:
    """Execute database operation with retry logic.

    Parameters
    ----------
    session : Session
        SQLAlchemy session
    operation : callable
        Database operation to execute
    max_retries : int, optional
        Maximum number of retry attempts

    Returns
    -------
    Any
        Result of the operation if successful

    Raises
    ------
    SQLAlchemyError
        If operation fails after all retries
    """
    retries = 0
    while retries < max_retries:
        try:
            result = operation()
            session.commit()
            return result
        except SQLAlchemyError as e:
            session.rollback()
            retries += 1
            if retries == max_retries:
                logger.error(f"Operation failed after {max_retries} retries: {e}")
                raise
            logger.warning(f"Retrying operation after error: {e}")

    raise SQLAlchemyError(f"Operation failed after {max_retries} retries")


def execute_query(func):
    """Decorator to execute database queries within a transaction.

    Wraps methods that contain database query logic, executing them within
    the database transaction context.

    Parameters
    ----------
    func : callable
        The method containing the database query logic

    Returns
    -------
    callable
        Wrapped method that executes within a transaction
    """

    def wrapper(self, *args, **kwargs):
        def query(session):
            return func(self, session, *args, **kwargs)

        return self.db._execute_in_transaction(query)

    return wrapper
