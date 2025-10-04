"""
Combat data processing for analysis.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.action_repository import ActionRepository


def process_combat_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process combat data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with combat metrics over time
    """
    # Find simulation database
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        db_path = experiment_path / "data" / "simulation.db"

    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")

    # Load data using existing infrastructure
    db = SimulationDatabase(f"sqlite:///{db_path}")
    action_repo = ActionRepository(db.session_manager)

    try:
        # Get combat-related actions (attacks)
        actions = action_repo.get_actions_by_scope("simulation")

        # Filter for combat actions
        combat_actions = []
        for action in actions:
            if hasattr(action, 'action_type') and action.action_type == 'attack':
                combat_actions.append({
                    'step': getattr(action, 'step_number', 0),
                    'agent_id': getattr(action, 'agent_id', None),
                    'action_type': getattr(action, 'action_type', ''),
                    'reward': getattr(action, 'reward', 0.0) or 0.0,
                    'details': getattr(action, 'details', {}),
                })

        if not combat_actions:
            return pd.DataFrame(columns=['step', 'agent_id', 'action_type', 'reward', 'damage_dealt', 'target_id'])

        df = pd.DataFrame(combat_actions)

        # Extract combat-specific details
        if 'details' in df.columns:
            df['damage_dealt'] = df['details'].apply(lambda x: x.get('damage_dealt', 0.0) if isinstance(x, dict) else 0.0)
            df['target_id'] = df['details'].apply(lambda x: x.get('target_id', None) if isinstance(x, dict) else None)
            df['attack_range'] = df['details'].apply(lambda x: x.get('attack_range', 0.0) if isinstance(x, dict) else 0.0)
            df.drop('details', axis=1, inplace=True)
        else:
            df['damage_dealt'] = 0.0
            df['target_id'] = None
            df['attack_range'] = 0.0

        return df

    finally:
        db.close()


def process_combat_metrics_data(
    experiment_path: Path,
    **kwargs
) -> pd.DataFrame:
    """Process combat metrics from simulation metrics.

    Args:
        experiment_path: Path to experiment directory
        **kwargs: Additional options

    Returns:
        DataFrame with combat metrics over time
    """
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        db_path = experiment_path / "data" / "simulation.db"

    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")

    db = SimulationDatabase(f"sqlite:///{db_path}")

    try:
        # Query step metrics that contain combat data
        from farm.database.models import StepMetrics
        session = db.session_manager.create_session()

        metrics_data = []
        for step_metric in session.query(StepMetrics).all():
            metrics_dict = step_metric.to_dict()
            if any(key in metrics_dict for key in ['combat_encounters', 'successful_attacks']):
                metrics_data.append({
                    'step': metrics_dict.get('step', 0),
                    'combat_encounters': metrics_dict.get('combat_encounters', 0),
                    'successful_attacks': metrics_dict.get('successful_attacks', 0),
                    'combat_encounters_this_step': metrics_dict.get('combat_encounters_this_step', 0),
                    'successful_attacks_this_step': metrics_dict.get('successful_attacks_this_step', 0),
                })

        session.close()

        if not metrics_data:
            return pd.DataFrame(columns=['step', 'combat_encounters', 'successful_attacks'])

        return pd.DataFrame(metrics_data)

    finally:
        db.close()


def process_agent_combat_stats(
    experiment_path: Path,
    agent_ids: Optional[List[int]] = None,
    **kwargs
) -> pd.DataFrame:
    """Process agent-specific combat statistics.

    Args:
        experiment_path: Path to experiment directory
        agent_ids: Specific agent IDs to analyze (None for all)
        **kwargs: Additional options

    Returns:
        DataFrame with agent combat statistics
    """
    combat_df = process_combat_data(experiment_path)

    if combat_df.empty:
        return pd.DataFrame(columns=['agent_id', 'total_attacks', 'successful_attacks', 'total_damage', 'avg_damage'])

    # Group by agent and calculate stats
    agent_stats = []

    grouped = combat_df.groupby('agent_id')
    for agent_id, group in grouped:
        if agent_ids is None or agent_id in agent_ids:
            total_attacks = len(group)
            successful_attacks = (group['damage_dealt'] > 0).sum()
            total_damage = group['damage_dealt'].sum()
            avg_damage = group['damage_dealt'].mean()
            avg_reward = group['reward'].mean()

            agent_stats.append({
                'agent_id': agent_id,
                'total_attacks': total_attacks,
                'successful_attacks': successful_attacks,
                'success_rate': successful_attacks / total_attacks if total_attacks > 0 else 0.0,
                'total_damage': total_damage,
                'avg_damage': avg_damage,
                'avg_reward': avg_reward,
            })

    return pd.DataFrame(agent_stats)
