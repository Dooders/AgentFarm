"""
Population data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.population_repository import PopulationRepository
from farm.database.analyzers.population_analyzer import PopulationAnalyzer


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

    # Load data using existing infrastructure
    db = SimulationDatabase(f"sqlite:///{db_path}")
    repository = PopulationRepository(db.session_manager)
    analyzer = PopulationAnalyzer(repository)

    # Get comprehensive statistics
    stats = analyzer.analyze_comprehensive_statistics()

    # Convert to DataFrame for analysis
    df = pd.DataFrame({
        'step': range(len(stats.metrics)),
        'total_agents': [m.total_agents for m in stats.metrics],
        'system_agents': [m.system_agents for m in stats.metrics],
        'independent_agents': [m.independent_agents for m in stats.metrics],
        'control_agents': [m.control_agents for m in stats.metrics],
        'avg_resources': [m.avg_resources for m in stats.metrics],
        # Add more metrics as needed
    })

    return df
