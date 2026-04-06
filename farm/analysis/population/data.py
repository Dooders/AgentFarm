"""
Population data processing for analysis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from farm.analysis.sql_loaders import population_dataframe_from_sqlite
from farm.analysis.common.utils import find_database_path, load_data_with_csv_fallback
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def process_population_data(experiment_path: Path, use_database: bool = True, **kwargs) -> pd.DataFrame:
    """Process population data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with population metrics over time
    """

    def _load_population_from_database() -> pd.DataFrame:
        """Load population data from database."""
        db_path = find_database_path(experiment_path)
        return population_dataframe_from_sqlite(str(db_path))

    return load_data_with_csv_fallback(
        experiment_path=experiment_path,
        csv_filename="population.csv",
        db_loader_func=_load_population_from_database if use_database else None,
        logger=logger,
    )
