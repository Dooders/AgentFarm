import logging
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from farm.research.analysis.database import (
    find_simulation_databases,
    get_data,
    get_columns_data_by_agent_type,
    get_resource_consumption_data,
    get_resource_level_data,
    get_action_distribution_data,
    get_rewards_by_generation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_population_df(
    all_populations: List[np.ndarray], max_steps: int, is_resource_data: bool = False
) -> pd.DataFrame:
    """
    Create a DataFrame from population data.

    Args:
        all_populations: List of population arrays
        max_steps: Maximum number of steps
        is_resource_data: Whether this is resource data

    Returns:
        DataFrame with population data
    """
    # Create a DataFrame with steps as index
    df = pd.DataFrame(index=range(max_steps + 1))

    # Add each population as a column
    for i, population in enumerate(all_populations):
        # Pad or truncate the population array to match max_steps
        if len(population) < max_steps + 1:
            padded = np.pad(
                population,
                (0, max_steps + 1 - len(population)),
                mode="constant",
                constant_values=np.nan,
            )
            df[f"{'resource' if is_resource_data else 'population'}_{i}"] = padded
        else:
            df[f"{'resource' if is_resource_data else 'population'}_{i}"] = population[
                : max_steps + 1
            ]

    return df


def calculate_statistics(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate statistics from a DataFrame.

    Args:
        df: DataFrame with population data

    Returns:
        Tuple containing mean, median, lower CI, and upper CI arrays
    """
    # Calculate statistics across all simulations
    mean = df.mean(axis=1).values
    median = df.median(axis=1).values
    std = df.std(axis=1).values

    # Calculate 95% confidence interval
    n = df.shape[1]  # Number of simulations
    z = 1.96  # z-score for 95% confidence
    ci_margin = z * std / np.sqrt(n)
    lower_ci = mean - ci_margin
    upper_ci = mean + ci_margin

    return mean, median, lower_ci, upper_ci


def validate_population_data(population: np.ndarray, db_path: str = None) -> bool:
    """
    Validate population data.

    Args:
        population: Population array
        db_path: Path to the database

    Returns:
        True if data is valid, False otherwise
    """
    # Check for NaN values
    if np.isnan(population).any():
        if db_path:
            logger.warning(f"NaN values found in population data from {db_path}")
        return False

    # Check for negative values
    if (population < 0).any():
        if db_path:
            logger.warning(f"Negative values found in population data from {db_path}")
        return False

    # Check for unreasonably large values
    if (population > 1000).any():
        if db_path:
            logger.warning(
                f"Unreasonably large values found in population data from {db_path}"
            )
        return False

    return True


def validate_resource_level_data(
    resource_levels: np.ndarray, db_path: str = None
) -> bool:
    """
    Validate resource level data.

    Args:
        resource_levels: Resource level array
        db_path: Path to the database

    Returns:
        True if data is valid, False otherwise
    """
    # Check for NaN values
    if np.isnan(resource_levels).any():
        if db_path:
            logger.warning(f"NaN values found in resource level data from {db_path}")
        return False

    # Check for negative values
    if (resource_levels < 0).any():
        if db_path:
            logger.warning(
                f"Negative values found in resource level data from {db_path}"
            )
        return False

    # Check for unreasonably large values
    if (resource_levels > 10000).any():
        if db_path:
            logger.warning(
                f"Unreasonably large values found in resource level data from {db_path}"
            )
        return False

    return True


def detect_early_terminations(
    db_paths: List[str], expected_steps: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    Detect simulations that terminated early.

    Args:
        db_paths: List of paths to databases
        expected_steps: Expected number of steps

    Returns:
        Dictionary with information about early terminations
    """
    early_terminations = {}

    # If expected_steps is not provided, find the maximum steps across all simulations
    if expected_steps is None:
        max_steps_list = []
        for db_path in db_paths:
            try:
                _, _, max_step = get_data(db_path)
                max_steps_list.append(max_step)
            except Exception as e:
                logger.error(f"Error getting data from {db_path}: {e}")

        if max_steps_list:
            expected_steps = max(max_steps_list)
        else:
            logger.error("Could not determine expected steps")
            return early_terminations

    # Check each simulation for early termination
    for db_path in db_paths:
        try:
            steps, population, max_step = get_data(db_path)

            if (
                max_step < expected_steps * 0.9
            ):  # Consider it early if less than 90% of expected
                # Get the last population value
                last_population = population[-1] if len(population) > 0 else 0

                # Get the maximum population
                max_population = np.max(population) if len(population) > 0 else 0

                # Calculate the percentage of expected steps completed
                completion_percentage = (max_step / expected_steps) * 100

                early_terminations[db_path] = {
                    "max_step": max_step,
                    "expected_steps": expected_steps,
                    "completion_percentage": completion_percentage,
                    "last_population": last_population,
                    "max_population": max_population,
                }
        except Exception as e:
            logger.error(f"Error checking for early termination in {db_path}: {e}")

    return early_terminations


def analyze_final_agent_counts(experiment_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Analyze final agent counts across simulations.

    Args:
        experiment_data: Dictionary with experiment data

    Returns:
        Dictionary with analysis results
    """
    results = {}

    for agent_type, data in experiment_data.items():
        populations = data.get("populations", [])
        if not populations:
            continue

        # Get final counts for each simulation
        final_counts = []
        for population in populations:
            if len(population) > 0:
                final_counts.append(population[-1])

        if not final_counts:
            continue

        # Calculate statistics
        mean = np.mean(final_counts)
        median = np.median(final_counts)
        std = np.std(final_counts)
        min_count = np.min(final_counts)
        max_count = np.max(final_counts)

        # Calculate 95% confidence interval
        n = len(final_counts)
        z = 1.96  # z-score for 95% confidence
        ci_margin = z * std / np.sqrt(n)
        lower_ci = mean - ci_margin
        upper_ci = mean + ci_margin

        # Calculate survival rate (percentage of simulations with non-zero final count)
        survival_rate = (np.array(final_counts) > 0).mean() * 100

        results[agent_type] = {
            "mean": mean,
            "median": median,
            "std": std,
            "min": min_count,
            "max": max_count,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "survival_rate": survival_rate,
            "sample_size": n,
        }

    return results


def process_experiment(agent_type: str, experiment: str) -> Dict[str, List]:
    """
    Process population data for a single experiment.

    Args:
        agent_type: Agent type
        experiment: Experiment path

    Returns:
        Dictionary with processed data
    """
    result = {"populations": [], "max_steps": 0}

    # Find all database files for this experiment
    db_path = f"results/one_of_a_kind/experiments/data/{experiment}/{agent_type}"
    db_paths = find_simulation_databases(db_path)

    if not db_paths:
        logger.warning(f"No database files found for {agent_type} in {experiment}")
        return result

    # Process each database
    for db_path in db_paths:
        try:
            # Get population data
            steps, population, max_step = get_data(db_path)

            # Validate data
            if validate_population_data(population, db_path):
                result["populations"].append(population)
                result["max_steps"] = max(result["max_steps"], max_step)
        except Exception as e:
            logger.error(f"Error processing {db_path}: {e}")

    return result


def find_experiments(base_path: str) -> Dict[str, List[str]]:
    """
    Find all experiments in the base path.

    Args:
        base_path: Base path to search in

    Returns:
        Dictionary mapping agent types to experiment paths
    """
    experiments = {}

    # Find all directories in the base path
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # Check if this is an agent type directory
            agent_types = []
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    agent_types.append(subitem)

            if agent_types:
                # This is an experiment directory
                for agent_type in agent_types:
                    if agent_type not in experiments:
                        experiments[agent_type] = []
                    experiments[agent_type].append(item)

    return experiments


def process_experiment_by_agent_type(experiment: str) -> Dict[str, Dict]:
    """
    Process population data by agent type for a single experiment.

    Args:
        experiment: Experiment path

    Returns:
        Dictionary with processed data by agent type
    """
    result = {}

    # Find all database files for this experiment
    db_path = f"results/one_of_a_kind/experiments/data/{experiment}"
    db_paths = find_simulation_databases(db_path)

    if not db_paths:
        logger.warning(f"No database files found for {experiment}")
        return result

    # Process each database
    for db_path in db_paths:
        try:
            # Get population data by agent type
            steps, data_by_agent_type, max_step = get_columns_data_by_agent_type(
                db_path
            )

            # Process each agent type
            for agent_type, population in data_by_agent_type.items():
                # Validate data
                if validate_population_data(population, db_path):
                    if agent_type not in result:
                        result[agent_type] = {"populations": [], "max_steps": 0}

                    result[agent_type]["populations"].append(population)
                    result[agent_type]["max_steps"] = max(
                        result[agent_type]["max_steps"], max_step
                    )
        except Exception as e:
            logger.error(f"Error processing {db_path}: {e}")

    return result


def process_experiment_resource_consumption(experiment: str) -> Dict[str, Dict]:
    """
    Process resource consumption data for a single experiment.

    Args:
        experiment: Experiment path

    Returns:
        Dictionary with processed data
    """
    result = {}

    # Find all database files for this experiment
    db_path = f"results/one_of_a_kind/experiments/data/{experiment}"
    db_paths = find_simulation_databases(db_path)

    if not db_paths:
        logger.warning(f"No database files found for {experiment}")
        return result

    # Process each database
    for db_path in db_paths:
        try:
            # Get resource consumption data by agent type
            steps, data_by_agent_type, max_step = get_resource_consumption_data(db_path)

            # Process each agent type
            for agent_type, consumption in data_by_agent_type.items():
                # Validate data
                if validate_population_data(consumption, db_path):
                    if agent_type not in result:
                        result[agent_type] = {"consumption": [], "max_steps": 0}

                    result[agent_type]["consumption"].append(consumption)
                    result[agent_type]["max_steps"] = max(
                        result[agent_type]["max_steps"], max_step
                    )
        except Exception as e:
            logger.error(f"Error processing {db_path}: {e}")

    return result


def process_action_distributions(
    experiment: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Process action distribution data for a single experiment.

    Args:
        experiment: Experiment path

    Returns:
        Dictionary with processed data
    """
    result = {"simulations": {}, "aggregated": {}}

    # Find all database files for this experiment
    db_path = f"results/one_of_a_kind/experiments/data/{experiment}"
    db_paths = find_simulation_databases(db_path)

    if not db_paths:
        logger.warning(f"No database files found for {experiment}")
        return result

    # Process each database
    for i, db_path in enumerate(db_paths):
        try:
            # Get action distribution data
            action_data = get_action_distribution_data(db_path)

            # Store data for this simulation
            sim_name = f"sim_{i}"
            result["simulations"][sim_name] = {}

            # Process each agent type
            for agent_type, actions in action_data.items():
                if agent_type not in result["aggregated"]:
                    result["aggregated"][agent_type] = {}

                # Calculate percentages for this simulation
                total_actions = sum(actions.values())
                if total_actions > 0:
                    percentages = {
                        action: count / total_actions * 100
                        for action, count in actions.items()
                    }
                    result["simulations"][sim_name][agent_type] = percentages

                    # Update aggregated data
                    for action, count in actions.items():
                        if action not in result["aggregated"][agent_type]:
                            result["aggregated"][agent_type][action] = 0
                        result["aggregated"][agent_type][action] += count
        except Exception as e:
            logger.error(f"Error processing {db_path}: {e}")

    # Calculate percentages for aggregated data
    for agent_type, actions in result["aggregated"].items():
        total_actions = sum(actions.values())
        if total_actions > 0:
            result["aggregated"][agent_type] = {
                action: count / total_actions * 100 for action, count in actions.items()
            }

    return result


def process_experiment_resource_levels(experiment: str) -> Dict[str, List]:
    """
    Process resource level data for a single experiment.

    Args:
        experiment: Experiment path

    Returns:
        Dictionary with processed data
    """
    result = {"resource_levels": [], "max_steps": 0}

    # Find all database files for this experiment
    db_path = f"results/one_of_a_kind/experiments/data/{experiment}"
    db_paths = find_simulation_databases(db_path)

    if not db_paths:
        logger.warning(f"No database files found for {experiment}")
        return result

    # Process each database
    for db_path in db_paths:
        try:
            # Get resource level data
            steps, resource_levels, max_step = get_resource_level_data(db_path)

            # Validate data
            if validate_resource_level_data(resource_levels, db_path):
                result["resource_levels"].append(resource_levels)
                result["max_steps"] = max(result["max_steps"], max_step)
        except Exception as e:
            logger.error(f"Error processing {db_path}: {e}")

    return result


def process_experiment_rewards_by_generation(
    experiment: str,
) -> Dict[str, Dict[int, float]]:
    """
    Process rewards by generation for a single experiment.

    Args:
        experiment: Experiment path

    Returns:
        Dictionary with processed data
    """
    result = {"simulations": {}}

    # Find all database files for this experiment
    db_path = f"results/one_of_a_kind/experiments/data/{experiment}"
    db_paths = find_simulation_databases(db_path)

    if not db_paths:
        logger.warning(f"No database files found for {experiment}")
        return result

    # Process each database
    for i, db_path in enumerate(db_paths):
        try:
            # Get rewards by generation
            rewards = get_rewards_by_generation(db_path)

            if rewards:
                # Store data for this simulation
                sim_name = f"sim_{i}"
                result["simulations"][sim_name] = rewards
        except Exception as e:
            logger.error(f"Error processing {db_path}: {e}")

    return result
