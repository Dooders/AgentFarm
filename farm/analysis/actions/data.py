"""
Actions data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.action_repository import ActionRepository
from farm.analysis.common.utils import find_database_path
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


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
    
    if use_database:
        try:
            # Find simulation database using standardized utility
            db_path = find_database_path(experiment_path, "simulation.db")
            logger.info(f"Loading actions from database: {db_path}")

            # Load data using SessionManager directly (like population processor)
            from farm.database.session_manager import SessionManager
            session_manager = SessionManager(f"sqlite:///{db_path}")
            repository = ActionRepository(session_manager)

            # Get all actions from the simulation
            actions = repository.get_actions_by_scope("simulation")

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
            logger.info(f"Loaded {len(df)} action records from database")

            # Aggregate actions by step and action_type to create frequency data
            if not df.empty:
                # Group by step and action_type, count frequencies
                frequency_df = df.groupby(['step_number', 'action_type']).size().reset_index(name='frequency')
                # Rename step_number to step to match expected column name
                frequency_df = frequency_df.rename(columns={'step_number': 'step'})
                df = frequency_df
                logger.info(f"Aggregated into {len(df)} (step, action_type) frequency records")
            
        except (FileNotFoundError, ImportError) as e:
            logger.exception(f"Database loading failed: {e}. Falling back to CSV files")
            df = None

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
