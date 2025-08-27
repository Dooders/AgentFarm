#!/usr/bin/env python3
"""
data_extraction.py

Shared utility functions for extracting data from simulation databases.
Contains common data loading patterns for initial states, time series, and analysis data.
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sqlalchemy
from scipy.spatial.distance import euclidean

from .database_utils import create_database_session


def load_simulation_config(config_path: str) -> Dict:
    """
    Load simulation configuration from JSON file.

    Parameters
    ----------
    config_path : str
        Path to config.json file

    Returns
    -------
    Dict
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def get_initial_positions(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract initial positions of agents and resources from the simulation database.

    Parameters
    ----------
    db_path : str
        Path to simulation database

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (agents_df, resources_df) with initial positions
    """
    conn = sqlite3.connect(db_path)

    # Get initial agents (birth_time = 0)
    agents_query = """
    SELECT agent_id, agent_type, position_x, position_y, initial_resources
    FROM agents
    WHERE birth_time = 0
    """
    agents = pd.read_sql_query(agents_query, conn)

    # Get initial resources (step_number = 0)
    resources_query = """
    SELECT resource_id, position_x, position_y, amount
    FROM resource_states
    WHERE step_number = 0
    """
    resources = pd.read_sql_query(resources_query, conn)

    conn.close()
    return agents, resources


def get_initial_state(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Alias for get_initial_positions for backward compatibility.
    """
    return get_initial_positions(db_path)


def get_initial_state_with_config(
    db_path: str, config_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Get initial state along with simulation config.

    Parameters
    ----------
    db_path : str
        Path to simulation database
    config_path : str
        Path to config.json file

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (agents_df, resources_df, config_dict)
    """
    agents, resources = get_initial_positions(db_path)
    config = load_simulation_config(config_path)
    return agents, resources, config


def extract_time_series(
    db_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract time series data from a simulation database.

    Parameters
    ----------
    db_path : str
        Path to simulation database

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (steps_df, repro_df, agents_df) time series data
    """
    conn = sqlite3.connect(db_path)

    # Get simulation steps data
    steps_query = """
    SELECT step_number, total_agents, system_agents, independent_agents, control_agents,
           total_resources, average_agent_resources, births, deaths, average_agent_health,
           average_agent_age, combat_encounters, successful_attacks, resources_shared_this_step AS resources_shared
    FROM simulation_steps
    ORDER BY step_number
    """
    steps_df = pd.read_sql_query(steps_query, conn)

    # Get reproduction events
    repro_query = """
    SELECT step_number, parent_id, offspring_id, success, parent_resources_before,
           parent_resources_after, offspring_initial_resources, parent_generation,
           offspring_generation, parent_position_x, parent_position_y
    FROM reproduction_events
    ORDER BY step_number
    """
    repro_df = pd.read_sql_query(repro_query, conn)

    # Get agent data to map IDs to types
    agents_query = """
    SELECT agent_id, agent_type, birth_time, death_time
    FROM agents
    """
    agents_df = pd.read_sql_query(agents_query, conn)

    conn.close()

    return steps_df, repro_df, agents_df


def get_agent_type_mapping(db_path: str) -> Dict[str, str]:
    """
    Create a mapping of agent_id to agent_type.

    Parameters
    ----------
    db_path : str
        Path to simulation database

    Returns
    -------
    Dict[str, str]
        Mapping of agent_id to agent_type
    """
    agents_df = get_initial_positions(db_path)[0]
    return dict(zip(agents_df["agent_id"], agents_df["agent_type"]))


def calculate_initial_advantages(
    agents_df: pd.DataFrame, resources_df: pd.DataFrame, gathering_range: int = 30
) -> Dict[str, Dict]:
    """
    Calculate initial advantages based on agent positions relative to resources.

    Parameters
    ----------
    agents_df : pd.DataFrame
        DataFrame with agent positions
    resources_df : pd.DataFrame
        DataFrame with resource positions
    gathering_range : int
        Gathering range for agents

    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping agent types to advantage metrics
    """
    advantages = {}

    for _, agent in agents_df.iterrows():
        agent_type = agent["agent_type"]
        agent_pos = (agent["position_x"], agent["position_y"])

        # Calculate distances to all resources
        distances = []
        for _, resource in resources_df.iterrows():
            resource_pos = (resource["position_x"], resource["position_y"])
            distance = euclidean(agent_pos, resource_pos)
            distances.append((distance, resource["amount"]))

        # Calculate metrics
        if distances:
            # Nearest resource distance
            nearest_dist = min(distances, key=lambda x: x[0])[0]

            # Resources within gathering range
            resources_in_range = sum(1 for d, _ in distances if d <= gathering_range)

            # Resource amount within gathering range
            resource_amount_in_range = sum(
                amount for dist, amount in distances if dist <= gathering_range
            )

            advantages[agent_type] = {
                "nearest_resource_dist": nearest_dist,
                "resources_in_range": resources_in_range,
                "resource_amount_in_range": resource_amount_in_range,
            }

    return advantages


def get_state_at_step(
    db_path: str, step_number: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract positions of agents and resources at a specific step.

    Parameters
    ----------
    db_path : str
        Path to simulation database
    step_number : int
        Simulation step number

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (agents_df, resources_df) for the specified step
    """
    conn = sqlite3.connect(db_path)

    # Get agents at this step
    agents_query = """
    SELECT a.agent_id, ag.agent_type, a.position_x, a.position_y, a.resource_level as resources
    FROM agent_states a
    JOIN agents ag ON a.agent_id = ag.agent_id
    WHERE a.step_number = ?
    """
    agents = pd.read_sql_query(agents_query, conn, params=(step_number,))

    # Get resources at this step
    resources_query = """
    SELECT resource_id, position_x, position_y, amount
    FROM resource_states
    WHERE step_number = ?
    """
    resources = pd.read_sql_query(resources_query, conn, params=(step_number,))

    conn.close()
    return agents, resources


def get_simulation_steps_range(db_path: str) -> Tuple[int, int]:
    """
    Get the range of simulation steps available in the database.

    Parameters
    ----------
    db_path : str
        Path to simulation database

    Returns
    -------
    Tuple[int, int]
        (min_step, max_step)
    """
    conn = sqlite3.connect(db_path)

    steps_query = """
    SELECT MIN(step_number), MAX(step_number)
    FROM simulation_steps
    """
    min_step, max_step = conn.execute(steps_query).fetchone()

    conn.close()
    return min_step, max_step


def get_column_data_at_steps(
    db_path: str, column_name: str, steps: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Get specific column data at specified steps.

    Parameters
    ----------
    db_path : str
        Path to simulation database
    column_name : str
        Name of column to retrieve
    steps : Optional[List[int]]
        Specific steps to retrieve (all steps if None)

    Returns
    -------
    pd.DataFrame
        DataFrame with step_number and column data
    """
    conn = sqlite3.connect(db_path)

    if steps:
        placeholders = ",".join(["?"] * len(steps))
        query = f"""
        SELECT step_number, {column_name}
        FROM simulation_steps
        WHERE step_number IN ({placeholders})
        ORDER BY step_number
        """
        df = pd.read_sql_query(query, conn, params=tuple(steps))
    else:
        query = f"""
        SELECT step_number, {column_name}
        FROM simulation_steps
        ORDER BY step_number
        """
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df


def get_agent_states_over_time(
    db_path: str, agent_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get agent states over time for specified agents.

    Parameters
    ----------
    db_path : str
        Path to simulation database
    agent_ids : Optional[List[str]]
        Specific agent IDs to retrieve (all agents if None)

    Returns
    -------
    pd.DataFrame
        DataFrame with agent states over time
    """
    conn = sqlite3.connect(db_path)

    if agent_ids:
        placeholders = ",".join(["?"] * len(agent_ids))
        query = f"""
        SELECT a.step_number, a.agent_id, ag.agent_type, a.position_x, a.position_y,
               a.resource_level, a.health, a.age
        FROM agent_states a
        JOIN agents ag ON a.agent_id = ag.agent_id
        WHERE a.agent_id IN ({placeholders})
        ORDER BY a.step_number, a.agent_id
        """
        df = pd.read_sql_query(query, conn, params=tuple(agent_ids))
    else:
        query = """
        SELECT a.step_number, a.agent_id, ag.agent_type, a.position_x, a.position_y,
               a.resource_level, a.health, a.age
        FROM agent_states a
        JOIN agents ag ON a.agent_id = ag.agent_id
        ORDER BY a.step_number, a.agent_id
        """
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        List of required column names

    Returns
    -------
    bool
        True if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
