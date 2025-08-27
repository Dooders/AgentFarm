import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_population_data(
    population: np.ndarray, db_path: Optional[str] = None
) -> bool:
    """
    Validate population data array for integrity.

    Args:
        population: Array of population values to validate
        db_path: Optional database path for error logging

    Returns:
        bool: True if data is valid, False otherwise
    """
    if population is None:
        logger.warning(f"Population data is None{f' in {db_path}' if db_path else ''}")
        return False

    if len(population) == 0:
        logger.warning(f"Empty population data{f' in {db_path}' if db_path else ''}")
        return False

    if np.all(np.isnan(population)):
        logger.warning(
            f"Population data contains only NaN values{f' in {db_path}' if db_path else ''}"
        )
        return False

    if np.any(population < 0):
        logger.warning(
            f"Population data contains negative values{f' in {db_path}' if db_path else ''}"
        )
        return False

    return True


def validate_resource_level_data(
    resource_levels: np.ndarray, db_path: Optional[str] = None
) -> bool:
    """
    Validate resource level data array for integrity.
    Unlike population data, resource levels can be negative.

    Args:
        resource_levels: Array of resource level values to validate
        db_path: Optional database path for error logging

    Returns:
        bool: True if data is valid, False otherwise
    """
    if resource_levels is None:
        logger.warning(
            f"Resource level data is None{f' in {db_path}' if db_path else ''}"
        )
        return False

    if len(resource_levels) == 0:
        logger.warning(
            f"Empty resource level data{f' in {db_path}' if db_path else ''}"
        )
        return False

    if np.all(np.isnan(resource_levels)):
        logger.warning(
            f"Resource level data contains only NaN values{f' in {db_path}' if db_path else ''}"
        )
        return False

    # We allow negative values for resource levels
    return True


def calculate_statistics(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate population statistics with validation checks.
    """
    if df.empty:
        logger.error("Cannot calculate statistics: DataFrame is empty")
        return np.array([]), np.array([]), np.array([]), np.array([])

    n_simulations = df["simulation_id"].nunique()
    if n_simulations == 0:
        logger.error("No valid simulations found in data")
        return np.array([]), np.array([]), np.array([]), np.array([])

    grouped = df.groupby("step")["population"]

    # Calculate statistics with handling for all-NaN groups
    mean_pop = np.array(grouped.mean().fillna(0).values)
    median_pop = np.array(grouped.median().fillna(0).values)
    std_pop = np.array(grouped.std().fillna(0).values)

    # Avoid division by zero in confidence interval calculation
    confidence_interval = np.where(
        n_simulations > 0, 1.96 * std_pop / np.sqrt(n_simulations), 0
    )

    return mean_pop, median_pop, std_pop, confidence_interval
