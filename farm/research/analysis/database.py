import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sqlalchemy import func

from farm.database.database import SimulationDatabase
from farm.database.utils import extract_agent_counts_from_json
from farm.database.models import (
    ActionModel,
    AgentModel,
    SimulationStepModel,
)
from farm.research.analysis.util import (
    validate_population_data,
    validate_resource_level_data,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def find_simulation_databases(base_path: str) -> List[str]:
    """
    Find all simulation database files in the given directory and its subdirectories.
    Creates the base directory if it doesn't exist.
    """
    base = Path(base_path)
    try:
        base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Searching for databases in: {base.resolve()}")

        db_files = list(base.rglob("simulation.db"))
        if not db_files:
            logger.warning(f"No simulation.db files found in {base}")
        else:
            logger.info(f"Found {len(db_files)} database files:")
            for db_file in db_files:
                logger.info(f"  - {db_file}")
        return sorted(str(path) for path in db_files)
    except Exception as e:
        logger.error(f"Error creating/accessing directory {base}: {str(e)}")
        return []


def get_columns_data(
    experiment_db_path: str, columns: List[str]
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], int]]:
    """
    Retrieve specified columns from the simulation database with robust error handling.

    Args:
        experiment_db_path: Path to the database file
        columns: List of column names to retrieve

    Returns:
        Tuple containing:
        - steps: Array of step numbers
        - populations: Dictionary mapping column names to their data arrays
        - max_steps: Number of steps in the simulation
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return None

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Validate requested columns exist
        for col in columns:
            if not hasattr(SimulationStepModel, col):
                logger.error(f"Column '{col}' not found in database schema")
                return None

        # Build query dynamically based on requested columns
        query_columns = [getattr(SimulationStepModel, col) for col in columns]
        query = (
            session.query(SimulationStepModel.step_number, *query_columns)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if not query:
            logger.warning(f"No data found in database: {experiment_db_path}")
            return None

        # Extract step numbers and column data
        steps = np.array([row[0] for row in query])
        populations = {
            col: np.array([row[i + 1] for row in query])
            for i, col in enumerate(columns)
        }

        # Validate each column's data with the appropriate validation function
        for col, data in populations.items():
            if col == "average_agent_resources":
                # Use resource level validation for resource data
                if not validate_resource_level_data(data, experiment_db_path):
                    logger.error(
                        f"Invalid resource level data for column '{col}' in {experiment_db_path}"
                    )
                    return None
            else:
                # Use population validation for other data
                if not validate_population_data(data, experiment_db_path):
                    logger.error(
                        f"Invalid data for column '{col}' in {experiment_db_path}"
                    )
                    return None

        max_steps = len(steps)

        # Validate steps
        if len(steps) == 0:
            logger.error(f"No steps found in database: {experiment_db_path}")
            return None

        if not np.all(np.diff(steps) >= 0):
            logger.error(
                f"Steps are not monotonically increasing in {experiment_db_path}"
            )
            return None

        return steps, populations, max_steps

    except Exception as e:
        logger.error(f"Error accessing database {experiment_db_path}: {str(e)}")
        return None
    finally:
        if session:
            session.close()
        if db:
            db.close()


def get_data(experiment_db_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Retrieve step numbers and total agents from the simulation database.
    This is a wrapper around get_columns_data for backward compatibility.
    """
    result = get_columns_data(experiment_db_path, ["total_agents"])
    if result is not None:
        steps, populations, max_steps = result
        return steps, populations["total_agents"], max_steps
    return None


def get_columns_data_by_agent_type(
    experiment_db_path: str,
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], int]]:
    """
    Retrieve population data for each agent type from the simulation database.
    """
    # Use agent_type_counts JSON column instead
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return None

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Query steps with agent_type_counts
        query = (
            session.query(
                SimulationStepModel.step_number,
                SimulationStepModel.agent_type_counts
            )
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if not query:
            logger.warning(f"No data found in database: {experiment_db_path}")
            return None

        # Extract step numbers and agent counts from JSON
        steps = np.array([row[0] for row in query])
        system_counts = []
        independent_counts = []
        control_counts = []
        
        for row in query:
            counts = extract_agent_counts_from_json(row[1])
            system_counts.append(counts.get("system", 0))
            independent_counts.append(counts.get("independent", 0))
            control_counts.append(counts.get("control", 0))

        populations = {
            "system_agents": np.array(system_counts),
            "independent_agents": np.array(independent_counts),
            "control_agents": np.array(control_counts),
        }

        max_steps = len(steps)
        return steps, populations, max_steps

    except Exception as e:
        logger.error(f"Error accessing database {experiment_db_path}: {str(e)}")
        return None
    finally:
        if session:
            session.close()
        if db:
            db.close()


def get_resource_consumption_data(
    experiment_db_path: str,
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], int]]:
    """
    Retrieve resource consumption data for each agent type from the simulation database.

    Args:
        experiment_db_path: Path to the database file

    Returns:
        Tuple containing:
        - steps: Array of step numbers
        - consumption: Dictionary mapping agent types to their consumption data arrays
        - max_steps: Number of steps in the simulation
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return None

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Query steps with agent_type_counts and resources_consumed
        query = (
            session.query(
                SimulationStepModel.step_number,
                SimulationStepModel.agent_type_counts,
                SimulationStepModel.resources_consumed
            )
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if not query:
            logger.warning(f"No data found in database: {experiment_db_path}")
            return None

        # Extract data
        steps = np.array([row[0] for row in query])
        system_counts = []
        independent_counts = []
        control_counts = []
        resources_consumed = []
        
        for row in query:
            counts = extract_agent_counts_from_json(row[1])
            system_counts.append(counts.get("system", 0))
            independent_counts.append(counts.get("independent", 0))
            control_counts.append(counts.get("control", 0))
            resources_consumed.append(row[2] or 0)

        # Calculate consumption per agent type
        consumption = {}
        total_agents = np.array(system_counts) + np.array(independent_counts) + np.array(control_counts)
        resources_array = np.array(resources_consumed)
        
        # Calculate consumption proportionally for each agent type
        for agent_type, counts in [("system", system_counts), ("independent", independent_counts), ("control", control_counts)]:
            counts_array = np.array(counts)
            # Avoid division by zero
            safe_total = np.where(total_agents > 0, total_agents, 1)
            consumption[agent_type] = counts_array * resources_array / safe_total

        max_steps = len(steps)
        return steps, consumption, max_steps

    except Exception as e:
        logger.error(f"Error accessing database {experiment_db_path}: {str(e)}")
        return None
    finally:
        if session:
            session.close()
        if db:
            db.close()


def get_action_distribution_data(experiment_db_path: str) -> Dict[str, Dict[str, int]]:
    """
    Retrieve action distribution data for each agent type from the simulation database.

    Args:
        experiment_db_path: Path to the database file

    Returns:
        Dictionary mapping agent types to their action distributions
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return {}

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Query to get agent types
        agent_types = session.query(AgentModel.agent_type).distinct().all()
        agent_types = [t[0] for t in agent_types]

        # Initialize result structure
        result = {agent_type: {} for agent_type in agent_types}

        # For each agent type, get action distribution
        for agent_type in agent_types:
            # Join agents with their actions and count by action type
            query = (
                session.query(
                    ActionModel.action_type,
                    func.count(ActionModel.action_id).label("count"),
                )
                .join(AgentModel, AgentModel.agent_id == ActionModel.agent_id)
                .filter(AgentModel.agent_type == agent_type)
                .group_by(ActionModel.action_type)
                .all()
            )

            # Store results
            action_counts = {action_type: count for action_type, count in query}
            result[agent_type] = action_counts

        return result

    except Exception as e:
        logger.error(f"Error accessing database {experiment_db_path}: {str(e)}")
        return {}
    finally:
        if session:
            session.close()
        if db:
            db.close()


def get_resource_level_data(
    experiment_db_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Retrieve resource level data from the simulation database.

    Args:
        experiment_db_path: Path to the database file

    Returns:
        Tuple containing:
        - steps: Array of step numbers
        - resource_levels: Array of average agent resource levels
        - max_steps: Number of steps in the simulation
    """
    columns = ["average_agent_resources"]
    result = get_columns_data(experiment_db_path, columns)

    if result is None:
        return None

    steps, data, max_steps = result

    return steps, data["average_agent_resources"], max_steps


def get_rewards_by_generation(experiment_db_path: str) -> Dict[int, float]:
    """
    Retrieve average rewards grouped by agent generation from the simulation database.

    Args:
        experiment_db_path: Path to the database file

    Returns:
        Dictionary mapping generation numbers to their average rewards
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return {}

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Query to get average reward by generation
        # Join agents with their actions that have learning module metadata
        query = (
            session.query(
                AgentModel.generation,
                func.avg(ActionModel.reward).label("avg_reward"),
                func.count(ActionModel.action_id).label("experience_count"),
            )
            .join(
                ActionModel,
                ActionModel.agent_id == AgentModel.agent_id,
            )
            .filter(ActionModel.module_type.isnot(None))
            .group_by(AgentModel.generation)
            .having(
                func.count(ActionModel.action_id) > 0
            )  # Only include generations with data
            .order_by(AgentModel.generation)
            .all()
        )

        if not query:
            logger.warning(
                f"No reward data by generation found in database: {experiment_db_path}"
            )
            return {}

        # Convert query results to dictionary
        rewards_by_generation = {
            generation: avg_reward for generation, avg_reward, _ in query
        }

        return rewards_by_generation

    except Exception as e:
        logger.error(f"Error accessing database {experiment_db_path}: {str(e)}")
        return {}
    finally:
        if session:
            session.close()
        if db:
            db.close()
