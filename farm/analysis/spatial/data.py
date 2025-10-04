"""
Spatial data processing for analysis.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from farm.database.database import SimulationDatabase
from farm.database.repositories.agent_repository import AgentRepository
from farm.database.repositories.resource_repository import ResourceRepository


def process_spatial_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process spatial data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with spatial metrics over time
    """
    # Find simulation database
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        db_path = experiment_path / "data" / "simulation.db"

    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")

    # Load data using existing infrastructure
    db = SimulationDatabase(f"sqlite:///{db_path}")
    agent_repo = AgentRepository(db.session_manager)
    resource_repo = ResourceRepository(db.session_manager)

    try:
        # Get agent position data
        agent_data = agent_repo.get_agent_positions_over_time()

        # Get resource position data
        resource_data = resource_repo.get_resource_positions_over_time()

        # Convert to DataFrames
        agent_df = pd.DataFrame(agent_data) if agent_data else pd.DataFrame()
        resource_df = pd.DataFrame(resource_data) if resource_data else pd.DataFrame()

        # Combine spatial data
        spatial_data = {
            'agent_positions': agent_df,
            'resource_positions': resource_df,
        }

        return spatial_data

    finally:
        db.close()


def process_movement_data(
    experiment_path: Path,
    agent_ids: Optional[List[int]] = None,
    **kwargs
) -> pd.DataFrame:
    """Process movement trajectories for analysis.

    Args:
        experiment_path: Path to experiment directory
        agent_ids: Specific agent IDs to analyze (None for all)
        **kwargs: Additional options

    Returns:
        DataFrame with movement trajectories
    """
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        db_path = experiment_path / "data" / "simulation.db"

    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")

    db = SimulationDatabase(f"sqlite:///{db_path}")
    agent_repo = AgentRepository(db.session_manager)

    try:
        # Get movement trajectories
        trajectories = agent_repo.get_agent_trajectories(agent_ids=agent_ids)

        if not trajectories:
            return pd.DataFrame(columns=['agent_id', 'step', 'position_x', 'position_y', 'position_z'])

        # Convert to DataFrame
        df = pd.DataFrame(trajectories)

        # Ensure position columns exist
        required_cols = ['agent_id', 'step', 'position_x', 'position_y']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # Calculate movement distances
        df = df.sort_values(['agent_id', 'step'])
        df['distance'] = df.groupby('agent_id').apply(
            lambda group: _calculate_trajectory_distances(group)
        ).explode().values

        return df

    finally:
        db.close()


def process_location_analysis_data(
    experiment_path: Path,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """Process location-specific analysis data.

    Args:
        experiment_path: Path to experiment directory
        **kwargs: Additional options

    Returns:
        Dictionary with location analysis data
    """
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        db_path = experiment_path / "data" / "simulation.db"

    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")

    db = SimulationDatabase(f"sqlite:///{db_path}")
    agent_repo = AgentRepository(db.session_manager)
    resource_repo = ResourceRepository(db.session_manager)

    try:
        # Get location-based activity data
        location_activity = agent_repo.get_location_activity_data()
        resource_distribution = resource_repo.get_resource_distribution_data()

        return {
            'location_activity': pd.DataFrame(location_activity) if location_activity else pd.DataFrame(),
            'resource_distribution': pd.DataFrame(resource_distribution) if resource_distribution else pd.DataFrame(),
        }

    finally:
        db.close()


def _calculate_trajectory_distances(group: pd.DataFrame) -> List[float]:
    """Calculate distances between consecutive positions in a trajectory.

    Args:
        group: DataFrame group for a single agent

    Returns:
        List of distances
    """
    distances = [0.0]  # First point has no previous distance

    for i in range(1, len(group)):
        prev_pos = (group.iloc[i-1]['position_x'], group.iloc[i-1]['position_y'])
        curr_pos = (group.iloc[i]['position_x'], group.iloc[i]['position_y'])

        if None not in prev_pos and None not in curr_pos:
            distance = np.sqrt(
                (curr_pos[0] - prev_pos[0])**2 +
                (curr_pos[1] - prev_pos[1])**2
            )
            distances.append(distance)
        else:
            distances.append(0.0)

    return distances
