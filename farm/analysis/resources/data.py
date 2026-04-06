"""
Resources data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.analysis.sql_loaders import resources_merged_with_positions_from_sqlite
from farm.analysis.common.utils import find_database_path, load_data_with_csv_fallback
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Populated on DB loads for downstream hotspot analysis (avoids a second SQLite open).
RESOURCE_POSITIONS_DATAFRAME_ATTR = "_resource_positions_for_hotspots"


def process_resource_data(experiment_path: Path, use_database: bool = True, **kwargs) -> pd.DataFrame:
    """Process resource data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with resource metrics over time
    """

    def _load_resources_from_database() -> pd.DataFrame:
        """Load resource data from database."""
        db_path = find_database_path(experiment_path)
        logger.info(f"Database path exists: {db_path}")
        df, positions = resources_merged_with_positions_from_sqlite(str(db_path))
        df.attrs[RESOURCE_POSITIONS_DATAFRAME_ATTR] = positions
        logger.info(f"Database loading successful, df shape: {df.shape}")
        return df

    return load_data_with_csv_fallback(
        experiment_path=experiment_path,
        csv_filename="resources.csv",
        db_loader_func=_load_resources_from_database if use_database else None,
        logger=logger,
    )
