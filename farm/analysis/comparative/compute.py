"""
Comparative analysis computations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path

from farm.analysis.common.utils import calculate_statistics


def compute_comparison_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comparison metrics between simulations.

    Args:
        df: DataFrame with simulation comparison data

    Returns:
        Dictionary of computed comparison metrics
    """
    if df.empty:
        return {}

    metrics = {}

    # Basic comparison statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col != 'simulation_id':  # Skip ID columns
            values = df[col].values
            metrics[col] = calculate_statistics(values)

    return metrics


def compute_parameter_differences(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute parameter differences between simulations.

    Args:
        df: DataFrame with parameter data from different simulations

    Returns:
        Dictionary of parameter differences
    """
    if df.empty or 'simulation_id' not in df.columns:
        return {}

    differences = {}

    # Group by parameter and compute differences
    if 'parameter_name' in df.columns and 'parameter_value' in df.columns:
        for param_name in df['parameter_name'].unique():
            param_data = df[df['parameter_name'] == param_name]
            if len(param_data) > 1:
                values = param_data['parameter_value'].values
                differences[param_name] = {
                    'values': list(values),
                    'range': float(np.ptp(values)),  # peak-to-peak
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

    return differences


def compute_performance_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute performance comparison metrics.

    Args:
        df: DataFrame with performance metrics

    Returns:
        Dictionary of performance comparison results
    """
    if df.empty:
        return {}

    performance = {}

    # Performance metrics comparison
    performance_cols = [col for col in df.columns if any(term in col.lower()
                        for term in ['reward', 'performance', 'score', 'fitness'])]

    for col in performance_cols:
        values = df[col].dropna().values
        if len(values) > 0:
            performance[col] = calculate_statistics(values)

    return performance
