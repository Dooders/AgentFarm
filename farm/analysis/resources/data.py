"""
Resources data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.resource_repository import ResourceRepository
from farm.database.analyzers.resource_analyzer import ResourceAnalyzer


def process_resource_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process resource data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with resource metrics over time
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
    repository = ResourceRepository(db.session_manager)
    analyzer = ResourceAnalyzer(repository)

    # Get comprehensive statistics
    analysis = analyzer.analyze_comprehensive_statistics()

    # Convert distribution data to DataFrame
    distribution_data = []
    for dist in analysis.distribution:
        distribution_data.append({
            'step': dist.step,
            'total_resources': dist.total_resources,
            'average_per_cell': dist.average_per_cell,
            'distribution_entropy': dist.distribution_entropy,
        })

    df = pd.DataFrame(distribution_data)

    # Add consumption data as single row if available
    if hasattr(analysis.consumption, 'total_consumed'):
        df['total_consumed'] = analysis.consumption.total_consumed
        df['avg_consumption_rate'] = analysis.consumption.avg_consumption_rate
        df['peak_consumption'] = analysis.consumption.peak_consumption
        df['consumption_variance'] = analysis.consumption.consumption_variance

    # Add efficiency data as single row if available
    if hasattr(analysis.efficiency, 'utilization_rate'):
        df['utilization_rate'] = analysis.efficiency.utilization_rate
        df['distribution_efficiency'] = analysis.efficiency.distribution_efficiency
        df['consumption_efficiency'] = analysis.efficiency.consumption_efficiency
        df['regeneration_rate'] = analysis.efficiency.regeneration_rate

    return df
