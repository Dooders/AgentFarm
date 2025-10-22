"""
Population statistical computations.

Optimized for performance with vectorized operations and comprehensive metrics.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from farm.analysis.common.utils import calculate_statistics, calculate_trend
from farm.analysis.config import get_config
from farm.utils.logging import get_logger

logger = get_logger(__name__)


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


def compute_population_stability(
    df: pd.DataFrame,
    window: Optional[int] = None
) -> Dict[str, float]:
    """Compute population stability metrics using vectorized operations.

    Uses rolling window calculations for improved performance.

    Args:
        df: Population data
        window: Window size for stability calculation (uses config default if None)

    Returns:
        Stability metrics including:
        - mean_cv: Mean coefficient of variation
        - stability_score: Normalized stability score (0-1, higher is more stable)
        - volatility: Standard deviation of population changes
        - max_fluctuation: Maximum single-step population change
    """
    if window is None:
        window = get_config('population').stability_window

    total = df['total_agents']

    # Adjust window if data is too short
    if len(total) < window:
        window = max(2, len(total) // 2)

    # Use pandas rolling for efficient windowed calculations
    rolling = total.rolling(window=window, min_periods=1)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()

    # Calculate coefficient of variation for each window (avoid division by zero)
    cv = np.where(rolling_mean > 0, rolling_std / rolling_mean, 0)
    mean_cv = float(np.mean(cv[window-1:]))  # Skip initial incomplete windows

    # Calculate population changes (first derivative)
    pop_changes = total.diff().fillna(0)
    volatility = float(pop_changes.std())
    max_fluctuation = float(abs(pop_changes).max())

    # Calculate relative changes as percentage
    rel_changes = pop_changes / total.shift(1)
    rel_changes = rel_changes.replace([np.inf, -np.inf], np.nan).fillna(0)

    return {
        'mean_cv': mean_cv,
        'stability_score': float(1.0 / (1.0 + mean_cv)),
        'volatility': volatility,
        'max_fluctuation': max_fluctuation,
        'mean_relative_change': float(abs(rel_changes).mean()),
        'max_relative_change': float(abs(rel_changes).max()),
    }


def compute_growth_rate_analysis(
    df: pd.DataFrame,
    window: Optional[int] = None
) -> Dict[str, Any]:
    """Compute detailed growth rate analysis.

    Analyzes population growth patterns using multiple methods.

    Args:
        df: Population data
        window: Window size for growth rate calculation (uses config default if None)

    Returns:
        Dictionary containing:
        - instantaneous_growth_rate: Step-by-step growth rates
        - average_growth_rate: Mean growth rate across simulation
        - growth_acceleration: Second derivative (rate of change of growth)
        - exponential_fit: Parameters of exponential growth fit (if applicable)
        - doubling_time: Estimated population doubling time (if growing)
        - growth_phases: Identified phases (growth, decline, stable)
    """
    if window is None:
        window = get_config('population').growth_window

    total = df['total_agents']
    steps = df['step']

    # Instantaneous growth rate (percentage change)
    growth_rate = total.pct_change().fillna(0) * 100

    # Smooth growth rate with rolling window
    smoothed_growth = growth_rate.rolling(window=window, min_periods=1).mean()

    # Growth acceleration (second derivative)
    acceleration = growth_rate.diff().fillna(0)

    # Try exponential fit for early growth phase
    exponential_fit = None
    doubling_time = None
    try:
        # Use first half of data for fitting
        mid_idx = len(total) // 2
        x_fit = steps[:mid_idx].values
        y_fit = total[:mid_idx].values

        # Only fit if population is generally increasing
        if y_fit[-1] > y_fit[0] * 1.1:
            # Fit exponential: N(t) = N0 * e^(r*t)
            # Use log transform: ln(N) = ln(N0) + r*t
            log_y = np.log(y_fit + 1)  # Add 1 to avoid log(0)
            slope, intercept = np.polyfit(x_fit, log_y, 1)

            exponential_fit = {
                'rate': float(slope),
                'initial': float(np.exp(intercept)),
                'r_squared': float(np.corrcoef(x_fit, log_y)[0, 1] ** 2)
            }

            # Doubling time: t_d = ln(2) / r
            if slope > 0:
                doubling_time = float(np.log(2) / slope)
    except (ValueError, RuntimeError) as e:
        logger.debug(
            "exponential_fit_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )

    # Identify growth phases using threshold on smoothed growth rate
    phases = []
    growth_threshold = 1.0  # 1% growth
    decline_threshold = -1.0  # 1% decline

    current_phase = None
    phase_start = 0

    for i, rate in enumerate(smoothed_growth):
        if rate > growth_threshold:
            phase = 'growth'
        elif rate < decline_threshold:
            phase = 'decline'
        else:
            phase = 'stable'

        if phase != current_phase:
            if current_phase is not None:
                phases.append({
                    'phase': current_phase,
                    'start_step': int(steps.iloc[phase_start]),
                    'end_step': int(steps.iloc[i]),
                    'duration': i - phase_start
                })
            current_phase = phase
            phase_start = i

    # Add final phase
    if current_phase is not None:
        phases.append({
            'phase': current_phase,
            'start_step': int(steps.iloc[phase_start]),
            'end_step': int(steps.iloc[-1]),
            'duration': len(steps) - phase_start
        })

    return {
        'average_growth_rate': float(growth_rate.mean()),
        'growth_rate_std': float(growth_rate.std()),
        'max_growth_rate': float(growth_rate.max()),
        'min_growth_rate': float(growth_rate.min()),
        'average_acceleration': float(acceleration.mean()),
        'exponential_fit': exponential_fit,
        'doubling_time': doubling_time,
        'growth_phases': phases,
        'time_in_growth': sum(p['duration'] for p in phases if p['phase'] == 'growth'),
        'time_in_decline': sum(p['duration'] for p in phases if p['phase'] == 'decline'),
        'time_stable': sum(p['duration'] for p in phases if p['phase'] == 'stable'),
    }


def compute_demographic_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute demographic composition metrics.

    Analyzes the distribution and dynamics of different agent types.

    Args:
        df: Population data with agent type columns

    Returns:
        Dictionary containing demographic metrics:
        - diversity_index: Shannon diversity index for agent types
        - dominance_index: Simpson's dominance index
        - type_proportions: Average proportion of each type
        - type_stability: Stability of each type over time
        - composition_changes: Significant shifts in composition
    """
    agent_types = ['system_agents', 'independent_agents', 'control_agents']
    available_types = [t for t in agent_types if t in df.columns]

    if not available_types:
        return {}

    # Calculate proportions
    proportions = {}
    for agent_type in available_types:
        proportions[agent_type] = df[agent_type] / df['total_agents']
        proportions[agent_type] = proportions[agent_type].fillna(0)

    # Shannon diversity index: H = -sum(p_i * ln(p_i))
    diversity_over_time = []
    for idx in df.index:
        p_values = [proportions[t].iloc[idx] for t in available_types]
        # Filter out zero proportions to avoid log(0)
        p_values = [p for p in p_values if p > 0]
        if p_values:
            h = -sum(p * np.log(p) for p in p_values)
            diversity_over_time.append(h)
        else:
            diversity_over_time.append(0)

    # Simpson's dominance index: D = sum(p_i^2)
    dominance_over_time = []
    for idx in df.index:
        p_values = [proportions[t].iloc[idx] for t in available_types]
        d = sum(p ** 2 for p in p_values)
        dominance_over_time.append(d)

    # Average proportions
    avg_proportions = {t: float(proportions[t].mean()) for t in available_types}

    # Type stability (inverse of coefficient of variation)
    type_stability = {}
    for agent_type in available_types:
        values = df[agent_type]
        if values.mean() > 0:
            cv = values.std() / values.mean()
            type_stability[agent_type] = float(1.0 / (1.0 + cv))
        else:
            type_stability[agent_type] = 0.0

    # Detect composition changes (using rolling correlation)
    composition_changes = []
    window = 10
    if len(df) >= window * 2:
        for i in range(window, len(df) - window):
            # Compare composition before and after
            before = np.array([proportions[t].iloc[i-window:i].mean() for t in available_types])
            after = np.array([proportions[t].iloc[i:i+window].mean() for t in available_types])

            # Calculate Euclidean distance between compositions
            distance = np.linalg.norm(after - before)

            if distance > 0.1:  # Significant change threshold
                composition_changes.append({
                    'step': int(df['step'].iloc[i]),
                    'distance': float(distance),
                    'before': {t: float(before[j]) for j, t in enumerate(available_types)},
                    'after': {t: float(after[j]) for j, t in enumerate(available_types)}
                })

    return {
        'diversity_index': {
            'mean': float(np.mean(diversity_over_time)),
            'std': float(np.std(diversity_over_time)),
            'min': float(np.min(diversity_over_time)),
            'max': float(np.max(diversity_over_time))
        },
        'dominance_index': {
            'mean': float(np.mean(dominance_over_time)),
            'std': float(np.std(dominance_over_time))
        },
        'type_proportions': avg_proportions,
        'type_stability': type_stability,
        'composition_changes': composition_changes[:5] if composition_changes else [],  # Top 5 changes
        'num_significant_changes': len(composition_changes)
    }
