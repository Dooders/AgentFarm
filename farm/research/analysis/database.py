import logging
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from sqlalchemy import func

from farm.database.database import SimulationDatabase
from farm.database.models import (
    ActionModel,
    AgentModel,
    LearningExperienceModel,
    SimulationStepModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_simulation_databases(base_path: str) -> List[str]:
    """
    Find all SQLite database files in the given base path.

    Args:
        base_path: The base directory to search in

    Returns:
        List of paths to database files
    """
    db_paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".db"):
                db_paths.append(os.path.join(root, file))
    return db_paths


def get_columns_data(
    experiment_db_path: str, columns: List[str]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Get data for specified columns from the database.

    Args:
        experiment_db_path: Path to the database
        columns: List of column names to retrieve

    Returns:
        Tuple containing steps array, data dictionary, and max steps
    """
    db = SimulationDatabase(experiment_db_path)
    session = db.Session()

    try:
        # Get all steps
        steps_query = session.query(SimulationStepModel.step).order_by(
            SimulationStepModel.step
        )
        steps = np.array([step[0] for step in steps_query])

        # Get max step
        max_step = session.query(func.max(SimulationStepModel.step)).scalar() or 0

        # Initialize data dictionary
        data = {column: np.zeros(len(steps)) for column in columns}

        # Get data for each step
        for i, step in enumerate(steps):
            step_data = (
                session.query(SimulationStepModel)
                .filter(SimulationStepModel.step == step)
                .first()
            )

            if step_data:
                for column in columns:
                    if hasattr(step_data, column):
                        data[column][i] = getattr(step_data, column)

        return steps, data, max_step

    finally:
        session.close()


def get_data(experiment_db_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Get population data from the database.

    Args:
        experiment_db_path: Path to the database

    Returns:
        Tuple containing steps array, population array, and max steps
    """
    steps, data, max_step = get_columns_data(experiment_db_path, ["population"])
    return steps, data["population"], max_step


def get_columns_data_by_agent_type(
    experiment_db_path: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Get population data by agent type from the database.

    Args:
        experiment_db_path: Path to the database

    Returns:
        Tuple containing steps array, agent type data dictionary, and max steps
    """
    db = SimulationDatabase(experiment_db_path)
    session = db.Session()

    try:
        # Get all steps
        steps_query = session.query(SimulationStepModel.step).order_by(
            SimulationStepModel.step
        )
        steps = np.array([step[0] for step in steps_query])

        # Get max step
        max_step = session.query(func.max(SimulationStepModel.step)).scalar() or 0

        # Get all agent types
        agent_types_query = session.query(AgentModel.agent_type).distinct()
        agent_types = [agent_type[0] for agent_type in agent_types_query]

        # Initialize data dictionary
        data = {agent_type: np.zeros(len(steps)) for agent_type in agent_types}

        # Get data for each step and agent type
        for i, step in enumerate(steps):
            for agent_type in agent_types:
                count = (
                    session.query(func.count(AgentModel.id))
                    .filter(
                        AgentModel.agent_type == agent_type,
                        AgentModel.created_at <= step,
                        (AgentModel.removed_at > step)
                        | (AgentModel.removed_at.is_(None)),
                    )
                    .scalar()
                ) or 0

                data[agent_type][i] = count

        return steps, data, max_step

    finally:
        session.close()


def get_resource_consumption_data(
    experiment_db_path: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Get resource consumption data by agent type from the database.

    Args:
        experiment_db_path: Path to the database

    Returns:
        Tuple containing steps array, consumption data dictionary, and max steps
    """
    db = SimulationDatabase(experiment_db_path)
    session = db.Session()

    try:
        # Get all steps
        steps_query = session.query(SimulationStepModel.step).order_by(
            SimulationStepModel.step
        )
        steps = np.array([step[0] for step in steps_query])

        # Get max step
        max_step = session.query(func.max(SimulationStepModel.step)).scalar() or 0

        # Get all agent types
        agent_types_query = session.query(AgentModel.agent_type).distinct()
        agent_types = [agent_type[0] for agent_type in agent_types_query]

        # Initialize data dictionary
        data = {agent_type: np.zeros(len(steps)) for agent_type in agent_types}

        # Get data for each step and agent type
        for i, step in enumerate(steps):
            for agent_type in agent_types:
                # Get all agents of this type that were alive at this step
                agents = (
                    session.query(AgentModel)
                    .filter(
                        AgentModel.agent_type == agent_type,
                        AgentModel.created_at <= step,
                        (AgentModel.removed_at > step)
                        | (AgentModel.removed_at.is_(None)),
                    )
                    .all()
                )

                # Sum up resource consumption for these agents at this step
                total_consumption = 0
                for agent in agents:
                    action = (
                        session.query(ActionModel)
                        .filter(
                            ActionModel.agent_id == agent.id,
                            ActionModel.step == step,
                        )
                        .first()
                    )

                    if action and action.resource_delta < 0:
                        total_consumption += abs(action.resource_delta)

                data[agent_type][i] = total_consumption

        return steps, data, max_step

    finally:
        session.close()


def get_action_distribution_data(experiment_db_path: str) -> Dict[str, Dict[str, int]]:
    """
    Get action distribution data by agent type from the database.

    Args:
        experiment_db_path: Path to the database

    Returns:
        Dictionary mapping agent types to action counts
    """
    db = SimulationDatabase(experiment_db_path)
    session = db.Session()

    try:
        # Get all agent types
        agent_types_query = session.query(AgentModel.agent_type).distinct()
        agent_types = [agent_type[0] for agent_type in agent_types_query]

        # Initialize data dictionary
        data = {agent_type: {} for agent_type in agent_types}

        # Get action counts for each agent type
        for agent_type in agent_types:
            # Get all agents of this type
            agents = (
                session.query(AgentModel)
                .filter(AgentModel.agent_type == agent_type)
                .all()
            )

            # Get all actions for these agents
            for agent in agents:
                actions = (
                    session.query(ActionModel)
                    .filter(ActionModel.agent_id == agent.id)
                    .all()
                )

                for action in actions:
                    action_type = action.action_type
                    if action_type not in data[agent_type]:
                        data[agent_type][action_type] = 0
                    data[agent_type][action_type] += 1

        return data

    finally:
        session.close()


def get_resource_level_data(
    experiment_db_path: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Get resource level data from the database.

    Args:
        experiment_db_path: Path to the database

    Returns:
        Tuple containing steps array, resource levels array, and max steps
    """
    steps, data, max_step = get_columns_data(experiment_db_path, ["resource_level"])
    return steps, data["resource_level"], max_step


def get_rewards_by_generation(experiment_db_path: str) -> Dict[int, float]:
    """
    Get average rewards by generation from the database.

    Args:
        experiment_db_path: Path to the database

    Returns:
        Dictionary mapping generations to average rewards
    """
    db = SimulationDatabase(experiment_db_path)
    session = db.Session()

    try:
        # Get all generations
        generations_query = (
            session.query(LearningExperienceModel.generation)
            .distinct()
            .order_by(LearningExperienceModel.generation)
        )
        generations = [gen[0] for gen in generations_query]

        # Initialize data dictionary
        data = {}

        # Get average reward for each generation
        for generation in generations:
            avg_reward = (
                session.query(func.avg(LearningExperienceModel.reward))
                .filter(LearningExperienceModel.generation == generation)
                .scalar()
            )

            if avg_reward is not None:
                data[generation] = float(avg_reward)

        return data

    finally:
        session.close()
