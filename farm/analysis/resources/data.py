"""
Resources data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.resource_repository import ResourceRepository
from farm.analysis.common.utils import find_database_path, load_data_with_csv_fallback
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


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
        # Find simulation database
        db_path = find_database_path(experiment_path)
        logger.info(f"Database path exists: {db_path}")

        # Load data using SessionManager directly (like population processor)
        from farm.database.session_manager import SessionManager

        session_manager = SessionManager(f"sqlite:///{db_path}")
        repository = ResourceRepository(session_manager)
        logger.info("Repository created successfully")

        # Get resource data using repository methods
        distribution = repository.resource_distribution()
        consumption = repository.consumption_patterns()
        efficiency = repository.efficiency_metrics()

        logger.info(f"Got distribution data: {len(distribution) if distribution else 0} records")

        # Convert distribution data to DataFrame
        distribution_data = []
        for dist in distribution:
            distribution_data.append(
                {
                    "step": dist.step,
                    "total_resources": dist.total_resources,
                    "average_per_cell": dist.average_per_cell,
                    "distribution_entropy": dist.distribution_entropy,
                }
            )

        df = pd.DataFrame(distribution_data)
        logger.info(f"Created DataFrame with {len(df)} rows")

        # Add consumption data
        df["total_consumed"] = consumption.total_consumed
        df["avg_consumption_rate"] = consumption.avg_consumption_rate
        df["peak_consumption"] = consumption.peak_consumption
        df["consumption_variance"] = consumption.consumption_variance

        # Add efficiency data
        df["utilization_rate"] = efficiency.utilization_rate
        df["distribution_efficiency"] = efficiency.distribution_efficiency
        df["consumption_efficiency"] = efficiency.consumption_efficiency
        df["regeneration_rate"] = efficiency.regeneration_rate

        logger.info(f"Database loading successful, df shape: {df.shape}")
        return df

    return load_data_with_csv_fallback(
        experiment_path=experiment_path,
        csv_filename="resources.csv",
        db_loader_func=_load_resources_from_database if use_database else None,
        logger=logger,
    )
