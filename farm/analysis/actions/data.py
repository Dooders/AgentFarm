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
    # Try to load from database first
    df = None
    try:
        # Find simulation database
        db_path = experiment_path / "simulation.db"
        if not db_path.exists():
            # Try alternative locations
            db_path = experiment_path / "data" / "simulation.db"

        if db_path.exists():
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
    except (FileNotFoundError, ImportError) as e:
        # If database loading fails, try to load from CSV files
        pass

    if df is None or df.empty:
        # Fallback to loading from CSV files
        data_dir = experiment_path / "data"
        if data_dir.exists():
            csv_path = data_dir / "actions.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"No action data found in {experiment_path}")
        else:
            raise FileNotFoundError(f"No data directory found in {experiment_path}")

    return df
