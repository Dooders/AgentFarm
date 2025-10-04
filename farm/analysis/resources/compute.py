"""
Resource statistical computations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_resource_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive resource statistics.

    Args:
        df: Resource data with columns: step, total_resources, etc.

    Returns:
        Dictionary of computed statistics
    """
    total_resources = df['total_resources'].values

    stats = {
        'total': calculate_statistics(total_resources),
        'peak_step': int(np.argmax(total_resources)),
        'peak_value': float(np.max(total_resources)),
        'final_value': float(total_resources[-1]),
        'trend': calculate_trend(total_resources),
        'resource_stability': float(1.0 / (1.0 + np.std(total_resources) / (np.mean(total_resources) + 1e-6))),
    }

    # Add consumption statistics if available
    if 'consumed_resources' in df.columns:
        consumed = df['consumed_resources'].values
        stats['consumption'] = {
            'total': float(np.sum(consumed)),
            'mean': float(np.mean(consumed)),
            'min': float(np.min(consumed)),
            'max': float(np.max(consumed)),
            'std': float(np.std(consumed)),
        }
        stats['depletion_rate'] = calculate_trend(total_resources)

    # Add efficiency statistics if available
    if 'resource_efficiency' in df.columns:
        efficiency = df['resource_efficiency'].values
        stats['efficiency'] = calculate_statistics(efficiency)

    # Distribution entropy statistics
    if 'distribution_entropy' in df.columns:
        entropy = df['distribution_entropy'].values
        stats['entropy'] = calculate_statistics(entropy)
        stats['avg_distribution_uniformity'] = float(np.mean(entropy))

    # Efficiency metrics
    efficiency_cols = ['utilization_rate', 'distribution_efficiency', 'consumption_efficiency', 'resource_efficiency']
    for col in efficiency_cols:
        if col in df.columns:
            stats[col] = calculate_statistics(df[col].values)

    return stats


def compute_consumption_patterns(df: pd.DataFrame) -> Dict[str, float]:
    """Compute resource consumption patterns.

    Args:
        df: Resource data with consumption metrics

    Returns:
        Dictionary of consumption metrics
    """
    if 'consumed_resources' not in df.columns:
        return {}

    consumed = df['consumed_resources'].values

    return {
        'trend': calculate_trend(consumed),
        'volatility': float(np.std(consumed) / (np.mean(consumed) + 1e-6)),
        'peak_consumption': float(np.max(consumed)),
        'increasing_periods': int(np.sum(np.diff(consumed) > 0)),
    }


def compute_resource_efficiency(df: pd.DataFrame) -> Dict[str, float]:
    """Compute resource utilization efficiency metrics.

    Args:
        df: Resource data with efficiency metrics

    Returns:
        Dictionary of efficiency metrics
    """
    efficiency_metrics = {}

    if 'utilization_rate' in df.columns:
        efficiency_metrics['avg_utilization_rate'] = float(df['utilization_rate'].mean())

    if 'distribution_efficiency' in df.columns:
        efficiency_metrics['avg_distribution_efficiency'] = float(df['distribution_efficiency'].mean())

    if 'consumption_efficiency' in df.columns:
        efficiency_metrics['avg_consumption_efficiency'] = float(df['consumption_efficiency'].mean())

    if 'regeneration_rate' in df.columns:
        efficiency_metrics['avg_regeneration_rate'] = float(df['regeneration_rate'].mean())

    # Overall efficiency score (weighted average)
    if efficiency_metrics:
        weights = {
            'avg_utilization_rate': 0.4,
            'avg_distribution_efficiency': 0.3,
            'avg_consumption_efficiency': 0.3,
        }
        overall_score = 0.0
        total_weight = 0.0
        for metric, weight in weights.items():
            if metric in efficiency_metrics:
                overall_score += efficiency_metrics[metric] * weight
                total_weight += weight

        if total_weight > 0:
            efficiency_metrics['overall_efficiency_score'] = overall_score / total_weight

    return efficiency_metrics


def compute_efficiency_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute efficiency metrics from resource data.

    Args:
        df: Resource data with efficiency columns

    Returns:
        Dictionary of efficiency metrics
    """
    if 'resource_efficiency' not in df.columns:
        return {}

    efficiency = df['resource_efficiency'].values

    return {
        'mean_efficiency': float(np.mean(efficiency)),
        'efficiency_trend': calculate_trend(efficiency),
        'efficiency_volatility': float(np.std(efficiency) / (np.mean(efficiency) + 1e-6)),
        'peak_efficiency': float(np.max(efficiency)),
    }


def compute_resource_hotspots(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze resource hotspot patterns.

    Args:
        df: Resource data (may include hotspot information)

    Returns:
        Dictionary of hotspot analysis metrics
    """
    # For now, return basic hotspot metrics based on distribution
    # In a full implementation, this would analyze spatial hotspot data

    if 'total_resources' not in df.columns:
        return {}

    total_resources = df['total_resources'].values

    # Calculate concentration metrics
    mean_resources = np.mean(total_resources)
    max_resources = np.max(total_resources)
    concentration_ratio = max_resources / (mean_resources + 1e-6)

    return {
        'max_concentration': float(max_resources),
        'avg_concentration': float(mean_resources),
        'concentration_ratio': float(concentration_ratio),
        'hotspot_intensity': float(concentration_ratio - 1.0),  # How much above average
    }
