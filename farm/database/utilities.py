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
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

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
    session: Session, operation: Callable, max_retries: int = 3
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


def create_prepared_statements(session):
    """Create prepared statements for common database operations.

    Parameters
    ----------
    session : Session
        SQLAlchemy session

    Returns
    -------
    dict
        Dictionary of prepared statements
    """
    connection = session.connection().connection

    statements = {}

    # Prepare statement for agent state insertion
    statements["insert_agent_state"] = connection.prepare(
        """
        INSERT INTO agent_states 
        (id, step_number, agent_id, position_x, position_y, position_z, 
         resource_level, current_health, is_defending, total_reward, age)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
    )

    # Prepare statement for resource state insertion
    statements["insert_resource"] = connection.prepare(
        """
        INSERT INTO resource_states
        (step_number, resource_id, amount, position_x, position_y)
        VALUES (?, ?, ?, ?, ?)
        """
    )

    # Add more prepared statements as needed

    return statements


def setup_db(db_path: Optional[str], simulation_id: str) -> Optional[Any]:
    """Setup database for simulation.

    Handles database file cleanup, creation, and initialization of the appropriate
    database instance. If the database file exists, it will be removed to ensure
    a fresh start. If removal fails, a unique filename will be generated.

    Parameters
    ----------
    db_path : Optional[str]
        Path to the database file. If None, no database is setup.
        If ":memory:", an in-memory database is created.
    simulation_id : str
        The simulation ID to use for database initialization

    Returns
    -------
    Optional[Any]
        The database instance if setup is successful, otherwise None
    """
    # Skip setup if no database requested
    if db_path is None:
        return None

    # Handle in-memory database
    if db_path == ":memory:":
        from datetime import datetime

        from farm.database.database import InMemorySimulationDatabase

        db = InMemorySimulationDatabase(simulation_id=simulation_id)

        # Add simulation record to the in-memory database
        db.add_simulation_record(
            simulation_id=simulation_id,
            start_time=datetime.now(),
            status="running",
            parameters={},
        )

        return db, ":memory:"

    # Handle file-based database
    final_db_path = db_path

    # Try to clean up any existing database connections first
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.close()
    except Exception:
        pass

    # Delete existing database file if it exists
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError as e:
            logger.warning(f"Failed to remove database {db_path}: {e}")
            # Generate unique filename if can't delete
            base, ext = os.path.splitext(db_path)
            final_db_path = f"{base}_{int(time.time())}{ext}"

    # Create the database instance
    from farm.database.database import SimulationDatabase

    db = SimulationDatabase(final_db_path, simulation_id=simulation_id)

    return db
