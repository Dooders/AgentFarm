"""
Population data processing for analysis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from farm.database.repositories.population_repository import PopulationRepository
from farm.database.session_manager import SessionManager


def process_population_data(experiment_path: Path, use_database: bool = True, **kwargs) -> pd.DataFrame:
    """Process population data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with population metrics over time
    """
    # Try to load from database first
    df = None
    try:
        # Find simulation database
        db_path = experiment_path / "simulation.db"
        if not db_path.exists():
            # Try alternative locations
            db_path = experiment_path / "data" / "simulation.db"

        if db_path.exists():
            # Load data using SessionManager directly
            db_uri = f"sqlite:///{db_path}"
            session_manager = SessionManager(db_uri)
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
                    "control_agents": [
                        p.control_agents if p.control_agents is not None else 0 for p in population_data
                    ],
                    "avg_resources": [p.avg_resources if p.avg_resources is not None else 0.0 for p in population_data],
                }
            )
    except Exception as e:
        # If database loading fails, try to load from CSV files
        pass

    if df is None or df.empty:
        # Fallback to loading from CSV files
        data_dir = experiment_path / "data"
        if data_dir.exists():
            csv_path = data_dir / "population.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"No population data found in {experiment_path}")
        else:
            raise FileNotFoundError(f"No data directory found in {experiment_path}")

    return df
