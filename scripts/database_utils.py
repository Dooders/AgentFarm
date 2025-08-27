#!/usr/bin/env python3
"""
database_utils.py

Shared utility functions for database operations across analysis scripts.
Contains common database connection patterns, session management, and data retrieval functions.
"""

import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from farm.database.database import SimulationDatabase
from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    HealthIncident,
    LearningExperienceModel,
    ReproductionEventModel,
    SimulationStepModel,
)


def create_database_session(db_path: str) -> sqlalchemy.orm.Session:
    """
    Create a database session for a simulation database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file

    Returns
    -------
    sqlalchemy.orm.Session
        Database session object
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    return Session()


def create_simulation_database_session(
    db_path: str, simulation_id: int = 1
) -> Tuple[SimulationDatabase, sqlalchemy.orm.Session]:
    """
    Create a SimulationDatabase session.

    Parameters
    ----------
    db_path : str
        Path to the simulation database
    simulation_id : int
        Simulation ID (default: 1)

    Returns
    -------
    Tuple[SimulationDatabase, sqlalchemy.orm.Session]
        Database instance and session
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    db = SimulationDatabase(db_path, simulation_id=simulation_id)
    session = db.Session()
    return db, session


def get_simulation_folders(experiment_path: str) -> List[str]:
    """
    Get all simulation folders (iteration_*) from an experiment path.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory

    Returns
    -------
    List[str]
        List of simulation folder paths
    """
    return glob.glob(os.path.join(experiment_path, "iteration_*"))


def get_simulation_database_path(folder: str) -> str:
    """
    Get the database path for a simulation folder.

    Parameters
    ----------
    folder : str
        Simulation folder path

    Returns
    -------
    str
        Path to simulation.db
    """
    return os.path.join(folder, "simulation.db")


def get_simulation_config_path(folder: str) -> str:
    """
    Get the config path for a simulation folder.

    Parameters
    ----------
    folder : str
        Simulation folder path

    Returns
    -------
    str
        Path to config.json
    """
    return os.path.join(folder, "config.json")


def get_iteration_number(folder: str) -> int:
    """
    Extract iteration number from folder name.

    Parameters
    ----------
    folder : str
        Simulation folder path

    Returns
    -------
    int
        Iteration number
    """
    folder_name = os.path.basename(folder)
    if folder_name.startswith("iteration_"):
        return int(folder_name.split("_")[1])
    else:
        raise ValueError(f"Invalid folder name format: {folder_name}")


def validate_simulation_folder(folder: str) -> bool:
    """
    Check if a folder contains valid simulation files.

    Parameters
    ----------
    folder : str
        Simulation folder path

    Returns
    -------
    bool
        True if folder contains both database and config files
    """
    db_path = get_simulation_database_path(folder)
    config_path = get_simulation_config_path(folder)
    return os.path.exists(db_path) and os.path.exists(config_path)


def get_final_step_number(session: sqlalchemy.orm.Session) -> int:
    """
    Get the final step number from simulation steps.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session

    Returns
    -------
    int
        Final step number
    """
    final_step = (
        session.query(SimulationStepModel.step_number)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )

    if final_step is None:
        raise ValueError("No simulation steps found")

    return final_step[0]


def get_agent_counts_by_type(session: sqlalchemy.orm.Session) -> Dict[str, int]:
    """
    Get agent counts by type.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session

    Returns
    -------
    Dict[str, int]
        Dictionary mapping agent types to counts
    """
    agent_counts = (
        session.query(AgentModel.agent_type, func.count(AgentModel.agent_id))
        .group_by(AgentModel.agent_type)
        .all()
    )

    return {agent_type: count for agent_type, count in agent_counts}


def get_action_types(session: sqlalchemy.orm.Session) -> List[str]:
    """
    Get all distinct action types in the database.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session

    Returns
    -------
    List[str]
        List of action types
    """
    action_types = session.query(ActionModel.action_type).distinct().all()
    return [a[0] for a in action_types]


def find_action_type(
    session: sqlalchemy.orm.Session, possible_types: List[str]
) -> Optional[str]:
    """
    Find the first matching action type from a list of possibilities.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session
    possible_types : List[str]
        List of possible action type names

    Returns
    -------
    Optional[str]
        Matching action type or None if not found
    """
    available_types = get_action_types(session)

    for action_type in possible_types:
        if action_type in available_types:
            return action_type

    # Try case-insensitive matching
    for action_type in possible_types:
        for available_type in available_types:
            if action_type.lower() in available_type.lower():
                return available_type

    return None


def safe_close_session(session: sqlalchemy.orm.Session) -> None:
    """
    Safely close a database session.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Database session to close
    """
    try:
        if session:
            session.close()
    except Exception as e:
        logging.warning(f"Error closing database session: {e}")


def check_database_schema(engine: sqlalchemy.engine.Engine, table_name: str) -> Dict:
    """
    Check the schema of a specific table in the database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine
    table_name : str
        Name of the table to check

    Returns
    -------
    Dict
        Dictionary with table schema information
    """
    try:
        inspector = sqlalchemy.inspect(engine)

        if table_name not in inspector.get_table_names():
            return {"exists": False}

        columns = inspector.get_columns(table_name)
        column_names = [col["name"] for col in columns]

        # Get primary key
        pk = inspector.get_pk_constraint(table_name)

        # Get foreign keys
        fks = inspector.get_foreign_keys(table_name)

        # Get indexes
        indexes = inspector.get_indexes(table_name)

        return {
            "exists": True,
            "column_count": len(columns),
            "columns": column_names,
            "primary_key": pk,
            "foreign_keys": fks,
            "indexes": indexes,
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}
