"""
Actions data processing for analysis.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.action_repository import ActionRepository
from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer
from farm.database.analyzers.sequence_pattern_analyzer import SequencePatternAnalyzer
from farm.database.analyzers.decision_pattern_analyzer import DecisionPatternAnalyzer


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

    # Load data using existing infrastructure
    db = SimulationDatabase(f"sqlite:///{db_path}")
    repository = ActionRepository(db.session_manager)

    # Get action statistics
    stats_analyzer = ActionStatsAnalyzer(repository)
    action_metrics = stats_analyzer.analyze()

    # Convert to DataFrame
    action_data = []
    for metric in action_metrics:
        action_data.append({
            'step': metric.step,
            'action_type': metric.action_type,
            'frequency': metric.frequency,
            'success_rate': metric.success_rate,
            'avg_reward': metric.avg_reward,
            'total_reward': metric.total_reward,
            'reward_variance': metric.reward_variance,
        })

    df = pd.DataFrame(action_data)

    # Get sequence patterns
    sequence_analyzer = SequencePatternAnalyzer(repository)
    sequence_patterns = sequence_analyzer.analyze()

    # Add sequence data as additional DataFrame columns
    sequence_dict = {}
    for pattern in sequence_patterns:
        key = f"seq_{pattern.sequence.replace('->', '_to_')}"
        sequence_dict[key] = pattern.probability

    # Add sequence probabilities to main DataFrame
    for key, value in sequence_dict.items():
        df[key] = value

    return df
