"""
Agents data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.agent_repository import AgentRepository


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

    # Load data using repository directly
    db = SimulationDatabase(f"sqlite:///{db_path}")
    repository = AgentRepository(db.session_manager)

    # Get a sample agent ID to demonstrate functionality
    sample_agent_id = repository.get_random_agent_id()

    agent_data = []
    if sample_agent_id:
        # Get agent info for the sample agent
        agent_info = repository.get_agent_info(sample_agent_id)
        if agent_info:
            # Get performance metrics
            metrics = repository.get_agent_performance_metrics(sample_agent_id)

            agent_data.append({
                'agent_id': sample_agent_id,
                'agent_type': agent_info.agent_type,
                'birth_time': agent_info.birth_time,
                'death_time': agent_info.death_time,
                'lifespan': getattr(agent_info, 'lifespan', None),
                'initial_resources': getattr(agent_info, 'initial_resources', None),
                'starting_health': getattr(agent_info, 'starting_health', None),
                'final_resources': getattr(metrics, 'final_resources', None),
                'final_health': getattr(metrics, 'final_health', None),
                'total_actions': getattr(metrics, 'total_actions', None),
                'successful_actions': getattr(metrics, 'successful_actions', None),
                'total_rewards': getattr(metrics, 'total_rewards', None),
            })

    df = pd.DataFrame(agent_data)
    return df
