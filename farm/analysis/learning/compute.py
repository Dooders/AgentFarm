"""
Learning statistical computations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_learning_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive learning statistics.

    Args:
        df: Learning data with columns: step, agent_id, reward, etc.

    Returns:
        Dictionary of computed statistics
    """
    if df.empty:
        return {
            'total_experiences': 0,
            'avg_reward': 0.0,
            'reward_trend': 0.0,
            'unique_agents': 0,
            'unique_actions': 0,
            'learning_efficiency': 0.0,
        }

    total_experiences = len(df)

    # Reward statistics
    reward_stats = calculate_statistics(df['reward'].values)

    # Learning trend (improvement over time)
    reward_trend = calculate_trend(df['reward'].values)

    # Agent diversity
    unique_agents = df['agent_id'].nunique() if 'agent_id' in df.columns else 0

    # Action diversity
    unique_actions = df['action_taken'].nunique() if 'action_taken' in df.columns else 0
    unique_actions_mapped = df['action_taken_mapped'].nunique() if 'action_taken_mapped' in df.columns else 0

    # Learning efficiency (reward per experience)
    learning_efficiency = reward_stats['mean'] / total_experiences if total_experiences > 0 else 0.0

    # Module performance
    module_performance = {}
    if 'module_type' in df.columns:
        for module, group in df.groupby('module_type'):
            module_performance[str(module)] = {
                'avg_reward': float(group['reward'].mean()),
                'total_experiences': len(group),
                'reward_std': float(group['reward'].std()) if len(group) > 1 else 0.0,
            }

    return {
        'total_experiences': total_experiences,
        'reward': reward_stats,
        'reward_trend': float(reward_trend),
        'unique_agents': unique_agents,
        'unique_actions': unique_actions,
        'unique_actions_mapped': unique_actions_mapped,
        'learning_efficiency': float(learning_efficiency),
        'module_performance': module_performance,
    }


def compute_agent_learning_curves(df: pd.DataFrame) -> Dict[str, List[float]]:
    """Compute learning curves for each agent.

    Args:
        df: Learning data

    Returns:
        Dictionary mapping agent IDs to their reward curves
    """
    if df.empty or 'agent_id' not in df.columns:
        return {}

    curves = {}
    for agent_id, group in df.groupby('agent_id'):
        # Sort by step and compute moving average
        sorted_group = group.sort_values('step')
        if 'reward_ma' in sorted_group.columns:
            curves[str(agent_id)] = sorted_group['reward_ma'].tolist()
        else:
            # Compute simple moving average
            rewards = sorted_group['reward'].rolling(window=10, min_periods=1).mean()
            curves[str(agent_id)] = rewards.tolist()

    return curves


def compute_learning_efficiency_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute learning efficiency metrics.

    Args:
        df: Learning data

    Returns:
        Dictionary of efficiency metrics
    """
    if df.empty:
        return {
            'reward_efficiency': 0.0,
            'action_diversity': 0.0,
            'learning_stability': 0.0,
            'convergence_rate': 0.0,
        }

    # Reward efficiency (average reward, normalized 0-1)
    avg_reward = df['reward'].mean()
    reward_range = df['reward'].max() - df['reward'].min()
    reward_efficiency = (avg_reward - df['reward'].min()) / reward_range if reward_range > 0 else 0.5

    # Action diversity (unique actions / total actions)
    total_actions = len(df)
    unique_actions = df['action_taken_mapped'].nunique() if 'action_taken_mapped' in df.columns else 0
    action_diversity = unique_actions / total_actions if total_actions > 0 else 0

    # Learning stability (inverse of reward variance)
    reward_variance = df['reward'].var() if len(df) > 1 else 0
    learning_stability = 1 / (1 + reward_variance) if reward_variance > 0 else 1.0

    # Convergence rate (how fast rewards stabilize)
    if len(df) > 20:
        first_half = df['reward'].iloc[:len(df)//2].mean()
        second_half = df['reward'].iloc[len(df)//2:].mean()
        convergence_rate = 1 - abs(first_half - second_half) / (df['reward'].max() - df['reward'].min() + 1e-6)
    else:
        convergence_rate = 0.0

    return {
        'reward_efficiency': float(reward_efficiency),
        'action_diversity': float(action_diversity),
        'learning_stability': float(learning_stability),
        'convergence_rate': float(convergence_rate),
    }


def compute_module_performance_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare performance across different learning modules.

    Args:
        df: Learning data

    Returns:
        Dictionary with module comparison metrics
    """
    if df.empty or 'module_type' not in df.columns:
        return {}

    module_stats = {}
    for module, group in df.groupby('module_type'):
        stats = calculate_statistics(group['reward'].values)
        module_stats[str(module)] = {
            'reward_stats': stats,
            'experience_count': len(group),
            'trend': calculate_trend(group['reward'].values),
        }

    return module_stats
