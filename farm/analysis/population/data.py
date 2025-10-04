"""
Population data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.population_repository import PopulationRepository


def process_population_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process population data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with population metrics over time
    """
    # Find simulation database
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        # Try alternative locations
        db_path = experiment_path / "data" / "simulation.db"

    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")

    # Load data using repository directly
    db = SimulationDatabase(f"sqlite:///{db_path}")
    repository = PopulationRepository(db.session_manager)

    # Get population data over time
    population_data = repository.get_population_over_time()

    # Convert to DataFrame for analysis
    df = pd.DataFrame({
        'step': [p.step for p in population_data],
        'total_agents': [p.total_agents for p in population_data],
        'system_agents': [p.system_agents for p in population_data],
        'independent_agents': [p.independent_agents for p in population_data],
        'control_agents': [p.control_agents for p in population_data],
        'avg_resources': [p.avg_resources for p in population_data],
    })

    return df
