"""
Population data processing for analysis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from farm.database.repositories.population_repository import PopulationRepository
from farm.database.session_manager import SessionManager
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
        # Find simulation database
        db_path = find_database_path(experiment_path)
        session_manager = SessionManager(f"sqlite:///{db_path}")
        repository = PopulationRepository(session_manager)

        # Get population data over time
        population_data = repository.get_population_over_time()

        # Convert to DataFrame for analysis
        df = pd.DataFrame(
            {
                "step": [p.step_number for p in population_data],
                "total_agents": [p.total_agents for p in population_data],
                "system_agents": [p.system_agents if p.system_agents is not None else 0 for p in population_data],
                "independent_agents": [
                    p.independent_agents if p.independent_agents is not None else 0 for p in population_data
                ],
                "control_agents": [p.control_agents if p.control_agents is not None else 0 for p in population_data],
                "avg_resources": [p.avg_resources if p.avg_resources is not None else 0.0 for p in population_data],
            }
        )
        return df

    return load_data_with_csv_fallback(
        experiment_path=experiment_path,
        csv_filename="population.csv",
        db_loader_func=_load_population_from_database if use_database else None,
        logger=logger,
    )
