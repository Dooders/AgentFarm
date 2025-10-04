"""
Population statistical computations.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_population_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive population statistics.

    Args:
        df: Population data with columns: step, total_agents, etc.

    Returns:
        Dictionary of computed statistics
    """
    total = df['total_agents'].values

    stats = {
        'total': calculate_statistics(total),
        'peak_step': int(np.argmax(total)),
        'peak_value': int(np.max(total)),
        'final_value': int(total[-1]),
        'trend': calculate_trend(total),
        'survival_rate': float(np.mean(total > 0)),
    }

    # Per-type statistics
    for agent_type in ['system_agents', 'independent_agents', 'control_agents']:
        if agent_type in df.columns:
            stats[agent_type] = calculate_statistics(df[agent_type].values)

    return stats


def compute_birth_death_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Compute birth and death rates.

    Args:
        df: Population data with births, deaths columns

    Returns:
        Dictionary of rate metrics
    """
    if 'births' not in df.columns or 'deaths' not in df.columns:
        return {}

    total_births = df['births'].sum()
    total_deaths = df['deaths'].sum()
    n_steps = len(df)

    return {
        'total_births': int(total_births),
        'total_deaths': int(total_deaths),
        'birth_rate': float(total_births / n_steps),
        'death_rate': float(total_deaths / n_steps),
        'net_growth': int(total_births - total_deaths),
        'growth_rate': float((total_births - total_deaths) / n_steps),
    }


def compute_population_stability(df: pd.DataFrame, window: int = 50) -> Dict[str, float]:
    """Compute population stability metrics.

    Args:
        df: Population data
        window: Window size for stability calculation

    Returns:
        Stability metrics
    """
    total = df['total_agents'].values

    if len(total) < window:
        window = len(total) // 2

    # Calculate coefficient of variation in windows
    cv_list = []
    for i in range(len(total) - window):
        window_data = total[i:i+window]
        if np.mean(window_data) > 0:
            cv = np.std(window_data) / np.mean(window_data)
            cv_list.append(cv)

    return {
        'mean_cv': float(np.mean(cv_list)) if cv_list else 0.0,
        'stability_score': float(1.0 / (1.0 + np.mean(cv_list))) if cv_list else 1.0,
    }
