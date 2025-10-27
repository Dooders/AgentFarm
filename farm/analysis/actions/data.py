"""
Actions data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import json
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.action_repository import ActionRepository
from farm.database.session_manager import SessionManager
from farm.analysis.common.utils import find_database_path, load_data_with_csv_fallback
from farm.utils.logging import get_logger

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
    def _load_actions_from_database() -> pd.DataFrame:
        """Load action data from database."""
        # Find simulation database using standardized utility
        db_path = find_database_path(experiment_path, "simulation.db")
        logger.info(f"Loading actions from database: {db_path}")

        # Load data using SessionManager directly (like population processor)
        session_manager = SessionManager(f"sqlite:///{db_path}")
        repository = ActionRepository(session_manager)

        # Get all actions from the simulation
        actions = repository.get_actions_by_scope("simulation")

        # Convert to DataFrame
        action_data = []
        for action in actions:
            # Extract resource information from details if available
            resources_before = None
            resources_after = None
            if action.details:
                details = json.loads(action.details) if isinstance(action.details, str) else action.details
                resources_before = details.get("agent_resources_before") or details.get("resources_before")
                resources_after = details.get("agent_resources_after") or details.get("resources_after")
            
            action_data.append({
                'agent_id': action.agent_id,
                'action_type': action.action_type,
                'step_number': action.step_number,
                'action_target_id': action.action_target_id,
                'resources_before': resources_before,
                'resources_after': resources_after,
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

        return df

    return load_data_with_csv_fallback(
        experiment_path=experiment_path,
        csv_filename="actions.csv",
        db_loader_func=_load_actions_from_database if use_database else None,
        logger=logger
    )
