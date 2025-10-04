"""
Temporal data processing for analysis.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from farm.database.session_manager import SessionManager
from farm.database.repositories.action_repository import ActionRepository


def process_temporal_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process temporal data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with temporal metrics over time
    """
    # Try to load from database first
    df = None
    try:
        # Find simulation database
        db_path = experiment_path / "simulation.db"
        if not db_path.exists():
            db_path = experiment_path / "data" / "simulation.db"

        if db_path.exists():
            # Load data using SessionManager directly
            db_uri = f"sqlite:///{db_path}"
            session_manager = SessionManager(db_uri)
            action_repo = ActionRepository(session_manager)

            # Get action data over time
            actions = action_repo.get_actions_by_scope("simulation")

            if not actions:
                df = pd.DataFrame(columns=['step', 'agent_id', 'action_type', 'reward'])
            else:
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'step': int(getattr(action, 'step_number', 0)),
                    'agent_id': getattr(action, 'agent_id', None),
                    'action_type': getattr(action, 'action_type', ''),
                    'reward': float(getattr(action, 'reward', 0.0) or 0.0),
                } for action in actions])

                # Ensure required columns exist and have correct types
                required_cols = ['step', 'agent_id', 'action_type', 'reward']
                for col in required_cols:
                    if col not in df.columns:
                        if col == 'step':
                            df[col] = 0
                        elif col == 'agent_id':
                            df[col] = None
                        elif col == 'action_type':
                            df[col] = ''
                        elif col == 'reward':
                            df[col] = 0.0

                # Ensure correct data types
                df['step'] = df['step'].astype(int)
                df['reward'] = df['reward'].astype(float)

    except Exception as e:
        # If database loading fails, return empty DataFrame
        pass

    if df is None:
        # Return empty DataFrame with correct structure
        df = pd.DataFrame(columns=['step', 'agent_id', 'action_type', 'reward'])

    return df


def process_time_series_data(
    experiment_path: Path,
    time_period_size: int = 100,
    **kwargs
) -> pd.DataFrame:
    """Process time series data aggregated by time periods.

    Args:
        experiment_path: Path to experiment directory
        time_period_size: Size of time periods for aggregation
        **kwargs: Additional options

    Returns:
        DataFrame with time series metrics
    """
    df = process_temporal_data(experiment_path)

    if df.empty:
        return pd.DataFrame()

    # Add time period column
    df['time_period'] = df['step'] // time_period_size

    # Aggregate by time period and action type
    time_series = df.groupby(['time_period', 'action_type']).agg({
        'reward': ['count', 'mean', 'sum'],
        'agent_id': 'nunique',
        'step': 'min'
    }).reset_index()

    # Flatten column names
    time_series.columns = ['time_period', 'action_type', 'action_count', 'avg_reward', 'total_reward', 'unique_agents', 'min_step']

    return time_series


def process_event_segmentation_data(
    experiment_path: Path,
    event_steps: Optional[List[int]] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """Process data for event segmentation analysis.

    Args:
        experiment_path: Path to experiment directory
        event_steps: List of step numbers where events occur
        **kwargs: Additional options

    Returns:
        Dictionary with segmentation data
    """
    df = process_temporal_data(experiment_path)

    if df.empty:
        return {}

    if event_steps is None:
        event_steps = []

    # Sort events
    event_steps = sorted(event_steps)

    segments_data = {}

    # Define segment boundaries
    boundaries = [0] + event_steps + [df['step'].max() + 1]

    for i in range(len(boundaries) - 1):
        start_step = boundaries[i]
        end_step = boundaries[i + 1]

        # Extract segment data
        segment_df = df[(df['step'] >= start_step) & (df['step'] < end_step)].copy()
        segment_df['segment_id'] = i
        segment_df['segment_start'] = start_step
        segment_df['segment_end'] = end_step

        segments_data[f'segment_{i}'] = segment_df

    return segments_data


def extract_temporal_patterns(
    experiment_path: Path,
    time_period_size: int = 100,
    rolling_window_size: int = 10,
    **kwargs
) -> pd.DataFrame:
    """Extract temporal patterns from action data.

    Args:
        experiment_path: Path to experiment directory
        time_period_size: Size of time periods
        rolling_window_size: Window size for rolling averages
        **kwargs: Additional options

    Returns:
        DataFrame with temporal patterns
    """
    time_series = process_time_series_data(experiment_path, time_period_size=time_period_size)

    if time_series.empty:
        return pd.DataFrame()

    patterns = []

    for action_type in time_series['action_type'].unique():
        action_data = time_series[time_series['action_type'] == action_type].copy()

        # Calculate rolling averages
        action_data = action_data.sort_values('time_period')
        action_data['rolling_avg_reward'] = action_data['avg_reward'].rolling(
            window=min(rolling_window_size, len(action_data)), center=True
        ).mean()
        action_data['rolling_action_count'] = action_data['action_count'].rolling(
            window=min(rolling_window_size, len(action_data)), center=True
        ).mean()

        action_data['action_type'] = action_type
        patterns.append(action_data)

    if patterns:
        return pd.concat(patterns, ignore_index=True)
    else:
        return pd.DataFrame()
