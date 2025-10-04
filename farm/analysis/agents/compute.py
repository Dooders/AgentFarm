"""
Agent statistical computations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_lifespan_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive lifespan statistics.

    Args:
        df: Agent data with lifespan columns

    Returns:
        Dictionary of computed statistics
    """
    if df.empty or 'lifespan' not in df.columns:
        return {}

    lifespans = df['lifespan'].dropna().values

    stats = {
        'lifespan': calculate_statistics(lifespans),
        'total_agents': len(df),
        'survival_rate': len(df[df['death_time'].isna()]) / len(df) if len(df) > 0 else 0.0,
        'mortality_rate': len(df[df['death_time'].notna()]) / len(df) if len(df) > 0 else 0.0,
    }

    # Agent type breakdown
    if 'agent_type' in df.columns:
        type_counts = df['agent_type'].value_counts()
        stats['agent_type_distribution'] = type_counts.to_dict()

        # Lifespan by type
        stats['lifespan_by_type'] = {}
        for agent_type in type_counts.index:
            type_lifespans = df[df['agent_type'] == agent_type]['lifespan'].dropna()
            if len(type_lifespans) > 0:
                stats['lifespan_by_type'][agent_type] = calculate_statistics(type_lifespans.values)

    return stats


def compute_behavior_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute agent behavior pattern statistics.

    Args:
        df: Agent data with behavior metrics

    Returns:
        Dictionary of behavior pattern metrics
    """
    if df.empty:
        return {}

    patterns = {}

    # Action success patterns
    if 'successful_actions' in df.columns and 'total_actions' in df.columns:
        df_copy = df.copy()
        df_copy['success_rate'] = df_copy['successful_actions'] / df_copy['total_actions']
        success_rates = df_copy['success_rate'].dropna().values

        patterns['action_success'] = calculate_statistics(success_rates)

    # Reward patterns
    if 'total_rewards' in df.columns:
        rewards = df['total_rewards'].dropna().values
        patterns['reward_distribution'] = calculate_statistics(rewards)

    # Resource utilization
    if 'initial_resources' in df.columns and 'final_resources' in df.columns:
        df_copy = df.copy()
        df_copy['resource_change'] = df_copy['final_resources'] - df_copy['initial_resources']
        resource_changes = df_copy['resource_change'].dropna().values
        patterns['resource_change'] = calculate_statistics(resource_changes)

    return patterns


def compute_performance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute agent performance metrics.

    Args:
        df: Agent data with performance columns

    Returns:
        Dictionary of performance metrics
    """
    if df.empty:
        return {}

    metrics = {}

    # Overall performance score (composite metric)
    if all(col in df.columns for col in ['successful_actions', 'total_rewards', 'lifespan']):
        df_copy = df.copy()
        # Normalize and combine metrics
        df_copy['success_rate'] = df_copy['successful_actions'] / df_copy['total_actions'].replace(0, 1)
        df_copy['reward_rate'] = df_copy['total_rewards'] / df_copy['lifespan'].replace(0, 1)

        # Create composite score
        df_copy['performance_score'] = (
            df_copy['success_rate'] * 0.4 +
            df_copy['reward_rate'] * 0.4 +
            (df_copy['lifespan'] / df_copy['lifespan'].max()) * 0.2
        )

        performance_scores = df_copy['performance_score'].dropna().values
        metrics['performance_score'] = calculate_statistics(performance_scores)

        # Top performers
        top_performers = df_copy.nlargest(5, 'performance_score')
        metrics['top_performers'] = top_performers[['agent_id', 'performance_score']].to_dict('records')

    return metrics


def compute_learning_curves(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute learning curve statistics.

    Args:
        df: Agent data with learning metrics

    Returns:
        Dictionary of learning curve metrics
    """
    # Learning curves would typically require time-series data per agent
    # For now, return basic learning-related metrics

    if df.empty:
        return {}

    curves = {}

    # Learning efficiency (rewards over time)
    if 'total_rewards' in df.columns and 'lifespan' in df.columns:
        df_copy = df.copy()
        df_copy['learning_efficiency'] = df_copy['total_rewards'] / df_copy['lifespan'].replace(0, 1)
        learning_efficiencies = df_copy['learning_efficiency'].dropna().values

        curves['learning_efficiency'] = calculate_statistics(learning_efficiencies)

    # Adaptation rate (change in performance over time)
    # This would require time-series data, for now use proxy metrics

    return curves
