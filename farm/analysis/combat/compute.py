"""
Combat statistical computations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_combat_statistics(combat_df: pd.DataFrame, metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive combat statistics.

    Args:
        combat_df: Combat action data
        metrics_df: Combat metrics data

    Returns:
        Dictionary of computed combat statistics
    """
    stats = {}

    # Combat action statistics
    if not combat_df.empty:
        stats['combat_actions'] = _compute_combat_action_stats(combat_df)

    # Combat metrics statistics
    if not metrics_df.empty:
        stats['combat_metrics'] = _compute_combat_metrics_stats(metrics_df)

    # Overall combat statistics
    if not combat_df.empty and not metrics_df.empty:
        stats['overall_combat'] = _compute_overall_combat_stats(combat_df, metrics_df)

    return stats


def compute_agent_combat_performance(agent_combat_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute agent-specific combat performance metrics.

    Args:
        agent_combat_df: Agent combat statistics

    Returns:
        Dictionary with agent combat performance
    """
    if agent_combat_df.empty:
        return {}

    # Sort agents by performance
    by_damage = agent_combat_df.sort_values('total_damage', ascending=False)
    by_success_rate = agent_combat_df.sort_values('success_rate', ascending=False)
    by_attacks = agent_combat_df.sort_values('total_attacks', ascending=False)

    # Calculate performance rankings
    rankings = {}
    for i, row in agent_combat_df.iterrows():
        agent_id = row['agent_id']
        rankings[agent_id] = {
            'damage_rank': by_damage[by_damage['agent_id'] == agent_id].index[0] + 1,
            'success_rank': by_success_rate[by_success_rate['agent_id'] == agent_id].index[0] + 1,
            'activity_rank': by_attacks[by_attacks['agent_id'] == agent_id].index[0] + 1,
        }

    # Performance tiers
    damage_thresholds = _calculate_performance_tiers(agent_combat_df['total_damage'])
    success_thresholds = _calculate_performance_tiers(agent_combat_df['success_rate'])

    return {
        'top_performers': {
            'by_damage': by_damage.head(5).to_dict('records'),
            'by_success_rate': by_success_rate.head(5).to_dict('records'),
            'by_activity': by_attacks.head(5).to_dict('records'),
        },
        'rankings': rankings,
        'performance_tiers': {
            'damage_tiers': damage_thresholds,
            'success_tiers': success_thresholds,
        },
    }


def compute_combat_efficiency_metrics(combat_df: pd.DataFrame) -> Dict[str, float]:
    """Compute combat efficiency metrics.

    Args:
        combat_df: Combat action data

    Returns:
        Dictionary with efficiency metrics
    """
    if combat_df.empty:
        return {
            'overall_success_rate': 0.0,
            'damage_efficiency': 0.0,
            'combat_intensity': 0.0,
            'reward_efficiency': 0.0,
        }

    total_attacks = len(combat_df)
    successful_attacks = (combat_df['damage_dealt'] > 0).sum()
    success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0

    total_damage = combat_df['damage_dealt'].sum()
    damage_efficiency = total_damage / total_attacks if total_attacks > 0 else 0.0

    total_reward = combat_df['reward'].sum()
    reward_efficiency = total_reward / total_attacks if total_attacks > 0 else 0.0

    # Combat intensity (attacks per step)
    steps_with_combat = combat_df['step'].nunique()
    combat_intensity = total_attacks / steps_with_combat if steps_with_combat > 0 else 0.0

    return {
        'overall_success_rate': float(success_rate),
        'damage_efficiency': float(damage_efficiency),
        'combat_intensity': float(combat_intensity),
        'reward_efficiency': float(reward_efficiency),
    }


def compute_combat_temporal_patterns(combat_df: pd.DataFrame, metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute temporal patterns in combat behavior.

    Args:
        combat_df: Combat action data
        metrics_df: Combat metrics data

    Returns:
        Dictionary with temporal combat patterns
    """
    if combat_df.empty and metrics_df.empty:
        return {}

    patterns = {}

    # Combat frequency over time
    if not metrics_df.empty:
        combat_freq_trend = calculate_trend(metrics_df['combat_encounters'].values)
        success_rate_trend = calculate_trend(
            (metrics_df['successful_attacks'] / metrics_df['combat_encounters'].replace(0, 1)).values
        )

        patterns['frequency_trend'] = float(combat_freq_trend)
        patterns['success_rate_trend'] = float(success_rate_trend)

    # Damage patterns over time
    if not combat_df.empty:
        damage_trend = calculate_trend(combat_df.groupby('step')['damage_dealt'].mean().values)
        patterns['damage_trend'] = float(damage_trend)

    # Peak combat periods
    if not metrics_df.empty:
        peak_step = metrics_df.loc[metrics_df['combat_encounters'].idxmax()]['step']
        peak_encounters = metrics_df['combat_encounters'].max()

        patterns['peak_combat'] = {
            'step': int(peak_step),
            'encounters': int(peak_encounters),
        }

    return patterns


def _compute_combat_action_stats(combat_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistics for combat actions."""
    total_attacks = len(combat_df)
    successful_attacks = (combat_df['damage_dealt'] > 0).sum()
    success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0

    damage_stats = calculate_statistics(combat_df['damage_dealt'].values)
    reward_stats = calculate_statistics(combat_df['reward'].values)

    unique_attackers = combat_df['agent_id'].nunique()
    unique_targets = combat_df['target_id'].nunique() if 'target_id' in combat_df.columns else 0

    return {
        'total_attacks': total_attacks,
        'successful_attacks': successful_attacks,
        'success_rate': float(success_rate),
        'damage_stats': damage_stats,
        'reward_stats': reward_stats,
        'unique_attackers': unique_attackers,
        'unique_targets': unique_targets,
    }


def _compute_combat_metrics_stats(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistics for combat metrics."""
    total_encounters = metrics_df['combat_encounters'].sum()
    total_successful = metrics_df['successful_attacks'].sum()
    overall_success_rate = total_successful / total_encounters if total_encounters > 0 else 0.0

    encounters_stats = calculate_statistics(metrics_df['combat_encounters'].values)
    successful_stats = calculate_statistics(metrics_df['successful_attacks'].values)

    return {
        'total_encounters': int(total_encounters),
        'total_successful_attacks': int(total_successful),
        'overall_success_rate': float(overall_success_rate),
        'encounters_per_step': encounters_stats,
        'successful_per_step': successful_stats,
    }


def _compute_overall_combat_stats(combat_df: pd.DataFrame, metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall combat statistics combining actions and metrics."""
    # Calculate combat activity rate (encounters per step with combat)
    steps_with_combat = (metrics_df['combat_encounters'] > 0).sum()
    total_steps = len(metrics_df)
    combat_activity_rate = steps_with_combat / total_steps if total_steps > 0 else 0.0

    # Average damage per encounter
    total_damage = combat_df['damage_dealt'].sum()
    total_encounters = metrics_df['combat_encounters'].sum()
    avg_damage_per_encounter = total_damage / total_encounters if total_encounters > 0 else 0.0

    # Combat intensity (damage per step)
    total_steps = len(metrics_df)
    avg_damage_per_step = total_damage / total_steps if total_steps > 0 else 0.0

    return {
        'combat_activity_rate': float(combat_activity_rate),
        'avg_damage_per_encounter': float(avg_damage_per_encounter),
        'avg_damage_per_step': float(avg_damage_per_step),
        'total_combat_steps': steps_with_combat,
        'total_simulation_steps': total_steps,
    }


def _calculate_performance_tiers(values: pd.Series) -> Dict[str, Any]:
    """Calculate performance tier thresholds."""
    if values.empty:
        return {}

    values_sorted = values.sort_values(ascending=False)

    # Quartile-based tiers
    q75 = values_sorted.quantile(0.75)
    q50 = values_sorted.quantile(0.50)
    q25 = values_sorted.quantile(0.25)

    return {
        'elite': float(q75),
        'good': float(q50),
        'average': float(q25),
        'poor': float(values_sorted.min()),
    }
