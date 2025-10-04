"""
Learning data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.learning_repository import LearningRepository


def process_learning_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process learning data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with learning metrics over time
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
    repository = LearningRepository(db.session_manager)

    try:
        # Get learning experiences
        experiences = repository.get_learning_experiences(
            repository.session_manager.create_session(),
            scope="simulation"
        )

        if not experiences:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=[
                'step', 'agent_id', 'module_type', 'reward', 'action_taken',
                'action_taken_mapped', 'state_delta', 'loss'
            ])

        # Convert to DataFrame
        df = pd.DataFrame(experiences)

        # Ensure required columns exist
        required_cols = ['step', 'agent_id', 'module_type', 'reward', 'action_taken']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # Add derived columns for analysis
        if 'reward' in df.columns:
            df['reward_ma'] = df.groupby('agent_id')['reward'].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean()
            )

        return df

    finally:
        db.close()


def process_learning_progress_data(experiment_path: Path) -> pd.DataFrame:
    """Process learning progress data for time series analysis.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        DataFrame with learning progress metrics
    """
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        db_path = experiment_path / "data" / "simulation.db"

    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")

    db = SimulationDatabase(f"sqlite:///{db_path}")
    repository = LearningRepository(db.session_manager)

    try:
        # Get learning progress
        progress_list = repository.get_learning_progress(
            repository.session_manager.create_session(),
            scope="simulation"
        )

        if not progress_list:
            return pd.DataFrame(columns=['step', 'reward', 'action_count', 'unique_actions'])

        # Convert to DataFrame
        df = pd.DataFrame([{
            'step': p.step,
            'reward': p.reward,
            'action_count': p.action_count,
            'unique_actions': p.unique_actions
        } for p in progress_list])

        return df

    finally:
        db.close()
