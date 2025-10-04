"""
Agents data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.agent_repository import AgentRepository
from farm.database.analyzers.agent_analyzer import AgentAnalysis


def process_agent_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process agent data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with agent metrics
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
    repository = AgentRepository(db.session_manager)
    analyzer = AgentAnalysis(repository)

    # Get all agent IDs (this would need to be implemented in the repository)
    # For now, create sample agent data structure
    # This would be replaced with actual repository queries

    # Sample structure - would be populated from actual database
    agent_data = []

    # Note: In a full implementation, this would query the agent repository
    # for all agents and their metrics. For now, creating placeholder structure.

    df = pd.DataFrame(agent_data)

    # If no data, create empty DataFrame with expected columns
    if df.empty:
        df = pd.DataFrame(columns=[
            'agent_id', 'agent_type', 'birth_time', 'death_time',
            'lifespan', 'initial_resources', 'starting_health',
            'final_resources', 'final_health', 'total_actions',
            'successful_actions', 'total_rewards'
        ])

    return df
