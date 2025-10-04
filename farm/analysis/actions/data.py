"""
Actions data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.action_repository import ActionRepository


def process_action_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process action data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with action metrics over time
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
    repository = ActionRepository(db.session_manager)

    # Get all actions from the experiment
    actions = repository.get_actions_by_scope("experiment")

    # Convert to DataFrame
    action_data = []
    for action in actions:
        action_data.append({
            'agent_id': action.agent_id,
            'action_type': action.action_type,
            'step_number': action.step_number,
            'action_target_id': action.action_target_id,
            'resources_before': action.resources_before,
            'resources_after': action.resources_after,
            'reward': action.reward,
            'details': action.details,
        })

    df = pd.DataFrame(action_data)
    return df
