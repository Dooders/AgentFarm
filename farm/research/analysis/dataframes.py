from typing import List

import numpy as np
import pandas as pd

from farm.research.analysis.util import (
    validate_population_data,
    validate_resource_level_data,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def create_population_df(
    all_populations: List[np.ndarray], max_steps: int, is_resource_data: bool = False
) -> pd.DataFrame:
    """
    Create a DataFrame from population or resource data with proper padding for missing steps.
    Includes data validation.

    Args:
        all_populations: List of data arrays
        max_steps: Maximum number of steps
        is_resource_data: Whether this is resource level data (allows negative values)

    Returns:
        DataFrame with the processed data
    """
    if not all_populations:
        logger.error("No data provided")
        return pd.DataFrame(columns=["simulation_id", "step", "population"])

    # Validate each data array
    valid_data = []
    for i, pop in enumerate(all_populations):
        if is_resource_data:
            # Use resource validation for resource data
            if validate_resource_level_data(pop):
                valid_data.append(pop)
            else:
                logger.warning(f"Skipping invalid resource data from simulation {i}")
        else:
            # Use population validation for population data
            if validate_population_data(pop):
                valid_data.append(pop)
            else:
                logger.warning(f"Skipping invalid population data from simulation {i}")

    if not valid_data:
        logger.error("No valid data after validation")
        return pd.DataFrame(columns=["simulation_id", "step", "population"])

    # Create DataFrame with valid data
    data = []
    for sim_idx, pop in enumerate(valid_data):
        for step in range(max_steps):
            population = pop[step] if step < len(pop) else np.nan
            data.append((f"sim_{sim_idx}", step, population))

    df = pd.DataFrame(data, columns=["simulation_id", "step", "population"])

    # Final validation of the DataFrame
    if df.empty:
        logger.warning("Created DataFrame is empty")
    elif df["population"].isna().all():
        logger.warning("All values are NaN in the DataFrame")

    return df
