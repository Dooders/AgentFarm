"""
Action statistical computations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_action_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive action statistics.

    Args:
        df: Action data with columns: step, action_type, frequency, etc.

    Returns:
        Dictionary of computed statistics
    """
    if df.empty:
        return {}

    stats = {}

    # Overall action frequency
    total_actions = df.groupby('step')['frequency'].sum()
    stats['total_actions'] = calculate_statistics(total_actions.values)

    # Per action type statistics
    action_types = df['action_type'].unique()
    stats['action_types'] = {}

    for action_type in action_types:
        type_data = df[df['action_type'] == action_type]
        if not type_data.empty:
            type_stats = {
                'frequency': calculate_statistics(type_data['frequency'].values),
            }

            # Add success_rate if available
            if 'success_rate' in type_data.columns:
                type_stats['success_rate'] = calculate_statistics(type_data['success_rate'].values)

            # Add avg_reward if available
            if 'avg_reward' in type_data.columns:
                type_stats['avg_reward'] = calculate_statistics(type_data['avg_reward'].values)

            stats['action_types'][action_type] = type_stats

    # Most common action
    avg_frequencies = df.groupby('action_type')['frequency'].mean()
    stats['most_common_action'] = avg_frequencies.idxmax()
    stats['most_common_frequency'] = float(avg_frequencies.max())

    return stats


def compute_sequence_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute action sequence pattern statistics.

    Args:
        df: Action data with sequence columns

    Returns:
        Dictionary of sequence pattern metrics
    """
    sequence_cols = [col for col in df.columns if col.startswith('seq_')]

    if not sequence_cols:
        return {}

    patterns = {}
    for col in sequence_cols:
        sequence_name = col.replace('seq_', '').replace('_to_', '->')
        values = df[col].dropna().values
        if len(values) > 0:
            patterns[sequence_name] = {
                'avg_probability': float(np.mean(values)),
                'max_probability': float(np.max(values)),
                'min_probability': float(np.min(values)),
                'probability_trend': calculate_trend(values),
            }

    # Find most common sequence
    if patterns:
        most_common = max(patterns.items(), key=lambda x: x[1]['avg_probability'])
        patterns['most_common_sequence'] = {
            'sequence': most_common[0],
            'probability': most_common[1]['avg_probability']
        }

    return patterns


def compute_decision_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute decision-making pattern statistics.

    Args:
        df: Action data with decision metrics

    Returns:
        Dictionary of decision pattern metrics
    """
    if 'success_rate' not in df.columns:
        return {}

    # Decision success patterns
    success_rates = df.groupby('step')['success_rate'].mean()
    decision_patterns = {
        'avg_success_rate': float(np.mean(success_rates)),
        'success_trend': calculate_trend(success_rates.values),
        'decision_consistency': float(1.0 / (1.0 + np.std(success_rates))),
    }

    # Action diversity (number of different action types per step)
    action_diversity = df.groupby('step')['action_type'].nunique()
    decision_patterns['avg_action_diversity'] = float(np.mean(action_diversity))
    decision_patterns['diversity_trend'] = calculate_trend(action_diversity.values)

    return decision_patterns


def compute_success_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Compute success rates for different action types.

    Args:
        df: Action data with success rate columns

    Returns:
        Dictionary mapping action types to success rates
    """
    if 'success_count' not in df.columns or 'total_attempts' not in df.columns:
        return {}

    success_rates = {}
    for action_type in df['action_type'].unique():
        type_data = df[df['action_type'] == action_type]
        total_success = type_data['success_count'].sum()
        total_attempts = type_data['total_attempts'].sum()

        if total_attempts > 0:
            success_rates[action_type] = total_success / total_attempts
        else:
            success_rates[action_type] = 0.0

    return success_rates


def compute_action_sequences(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute action sequence statistics.

    Args:
        df: Action data with sequence columns

    Returns:
        Dictionary of sequence metrics
    """
    if 'action_sequence' not in df.columns:
        return {}

    sequences = {
        'common_sequences': {},
        'avg_sequence_length': float(df['sequence_length'].mean()) if 'sequence_length' in df.columns else 0.0,
        'max_sequence_length': int(df['sequence_length'].max()) if 'sequence_length' in df.columns else 0,
        'transition_matrix': {}
    }

    # Count common sequences
    if not df.empty:
        sequence_counts = df['action_sequence'].value_counts()
        for seq, count in sequence_counts.items():
            seq_key = '->'.join(seq) if isinstance(seq, list) else str(seq)
            sequences['common_sequences'][seq_key] = int(count)

    return sequences


def compute_reward_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute reward and performance metrics.

    Args:
        df: Action data with reward columns

    Returns:
        Dictionary of reward metrics
    """
    if 'avg_reward' not in df.columns:
        return {}

    reward_data = df['avg_reward'].values

    metrics = {
        'overall_avg_reward': float(np.mean(reward_data)),
        'reward_trend': calculate_trend(reward_data),
        'reward_volatility': float(np.std(reward_data)),
    }

    # Reward distribution statistics
    if 'reward_variance' in df.columns:
        metrics['avg_reward_variance'] = float(df['reward_variance'].mean())

    # Total rewards
    if 'total_reward' in df.columns:
        total_rewards = df.groupby('step')['total_reward'].sum()
        metrics['total_rewards'] = calculate_statistics(total_rewards.values)

    # Best performing actions
    if not df.empty:
        best_action = df.groupby('action_type')['avg_reward'].mean().idxmax()
        metrics['best_performing_action'] = best_action
        metrics['best_action_reward'] = float(df[df['action_type'] == best_action]['avg_reward'].mean())

    return metrics
