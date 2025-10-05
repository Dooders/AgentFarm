"""
Temporal statistical computations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from farm.analysis.common.utils import calculate_statistics, calculate_trend, calculate_rolling_mean


def compute_temporal_statistics(time_series_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive temporal statistics.

    Args:
        time_series_df: Time series data with temporal patterns

    Returns:
        Dictionary of computed temporal statistics
    """
    if time_series_df.empty:
        return {
            "total_periods": 0,
            "total_actions": 0,
            "avg_reward_trend": 0.0,
            "action_frequency_trend": 0.0,
        }

    total_periods = time_series_df["time_period"].nunique()
    total_actions = time_series_df["action_count"].sum()

    # Overall trends
    reward_trend = calculate_trend(time_series_df.groupby("time_period")["avg_reward"].mean().values)
    action_trend = calculate_trend(time_series_df.groupby("time_period")["action_count"].mean().values)

    # Action type diversity
    action_types = time_series_df["action_type"].unique()
    action_diversity = len(action_types)

    # Reward volatility
    reward_volatility = time_series_df["avg_reward"].std() if len(time_series_df) > 1 else 0.0

    # Temporal patterns by action type
    action_patterns = {}
    for action_type in action_types:
        action_data = time_series_df[time_series_df["action_type"] == action_type]
        if not action_data.empty:
            action_patterns[action_type] = {
                "frequency": calculate_statistics(action_data["action_count"].values),
                "reward": calculate_statistics(action_data["avg_reward"].values),
                "trend": calculate_trend(action_data["avg_reward"].values),
                "periods_active": len(action_data),
            }

    return {
        "total_periods": total_periods,
        "total_actions": int(total_actions),
        "action_types": list(action_types),
        "action_diversity": action_diversity,
        "avg_reward_trend": float(reward_trend),
        "action_frequency_trend": float(action_trend),
        "reward_volatility": float(reward_volatility),
        "action_patterns": action_patterns,
    }


def compute_event_segmentation_metrics(
    segments_data: Dict[str, pd.DataFrame], event_steps: List[int]
) -> Dict[str, Any]:
    """Compute metrics for event segmentation analysis.

    Args:
        segments_data: Dictionary with segment data
        event_steps: List of event step numbers

    Returns:
        Dictionary with segmentation metrics
    """
    if not segments_data:
        return {}

    segment_metrics = {}

    for segment_name, segment_df in segments_data.items():
        if segment_df.empty:
            segment_metrics[segment_name] = {
                "action_count": 0,
                "unique_actions": 0,
                "avg_reward": 0.0,
                "unique_agents": 0,
            }
            continue

        # Basic metrics
        action_count = len(segment_df)
        unique_actions = segment_df["action_type"].nunique()
        avg_reward = segment_df["reward"].mean()
        unique_agents = segment_df["agent_id"].nunique()

        # Action distribution
        action_distribution = segment_df["action_type"].value_counts().to_dict()

        # Reward distribution by action type
        reward_by_action = {}
        for action_type, group in segment_df.groupby("action_type"):
            reward_by_action[action_type] = {
                "count": len(group),
                "avg_reward": group["reward"].mean(),
                "total_reward": group["reward"].sum(),
            }

        segment_metrics[segment_name] = {
            "action_count": action_count,
            "unique_actions": unique_actions,
            "avg_reward": float(avg_reward),
            "unique_agents": unique_agents,
            "action_distribution": action_distribution,
            "reward_by_action": reward_by_action,
        }

    # Event impact analysis
    event_impacts = _analyze_event_impacts(segment_metrics, event_steps)

    return {
        "segment_metrics": segment_metrics,
        "event_impacts": event_impacts,
    }


def compute_temporal_patterns(patterns_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute detailed temporal patterns analysis.

    Args:
        patterns_df: DataFrame with temporal patterns

    Returns:
        Dictionary with pattern analysis
    """
    if patterns_df.empty:
        return {}

    patterns = {}

    for action_type in patterns_df["action_type"].unique():
        action_patterns = patterns_df[patterns_df["action_type"] == action_type]

        # Peak analysis
        peak_period = action_patterns.loc[action_patterns["action_count"].idxmax()]["time_period"]
        peak_count = action_patterns["action_count"].max()

        # Trend analysis
        reward_trend = calculate_trend(action_patterns["avg_reward"].values)
        count_trend = calculate_trend(action_patterns["action_count"].values)

        # Rolling average analysis
        if "rolling_avg_reward" in action_patterns.columns:
            rolling_stability = 1.0 / (1.0 + action_patterns["rolling_avg_reward"].std())
        else:
            rolling_stability = 1.0

        # Cyclical patterns (simple autocorrelation)
        if len(action_patterns) > 3:
            reward_autocorr = _calculate_autocorrelation(action_patterns["avg_reward"].values)
        else:
            reward_autocorr = 0.0

        patterns[action_type] = {
            "peak_period": int(peak_period),
            "peak_count": int(peak_count),
            "reward_trend": float(reward_trend),
            "count_trend": float(count_trend),
            "rolling_stability": float(rolling_stability),
            "autocorrelation": float(reward_autocorr),
            "periods": len(action_patterns),
        }

    return patterns


def compute_temporal_efficiency_metrics(time_series_df: pd.DataFrame) -> Dict[str, float]:
    """Compute temporal efficiency metrics.

    Args:
        time_series_df: Time series data

    Returns:
        Dictionary with efficiency metrics
    """
    if time_series_df.empty:
        return {
            "avg_actions_per_step": 0.0,
            "avg_reward_per_step": 0.0,
            "avg_success_rate": 0.0,
            "efficiency_trend": 0.0,
        }

    # Check if we have the expected columns for time series data
    if "time_period" in time_series_df.columns and "avg_reward" in time_series_df.columns:
        # This is aggregated time series data
        # Reward efficiency over time (how rewards improve)
        reward_progression = time_series_df.groupby("time_period")["avg_reward"].mean()
        reward_efficiency = calculate_trend(reward_progression.values)

        # Action efficiency trend
        action_efficiency = time_series_df.groupby("time_period")["action_count"].mean()
        action_efficiency_trend = calculate_trend(action_efficiency.values)

        # Temporal stability (inverse of variance over time)
        reward_variance = reward_progression.var() if len(reward_progression) > 1 else 0
        temporal_stability = 1.0 / (1.0 + reward_variance)

        return {
            "reward_efficiency_over_time": float(reward_efficiency),
            "action_efficiency_trend": float(action_efficiency_trend),
            "temporal_stability": float(temporal_stability),
        }
    else:
        # This is raw step-by-step data
        total_steps = len(time_series_df)
        if total_steps == 0:
            return {
                "avg_actions_per_step": 0.0,
                "avg_reward_per_step": 0.0,
                "avg_success_rate": 0.0,
                "efficiency_trend": 0.0,
            }

        # Calculate basic metrics
        avg_actions_per_step = (
            time_series_df["action_count"].mean() if "action_count" in time_series_df.columns else 0.0
        )
        avg_reward_per_step = time_series_df["reward"].mean() if "reward" in time_series_df.columns else 0.0

        # Calculate success rate if success_count is available
        if "success_count" in time_series_df.columns and "action_count" in time_series_df.columns:
            total_actions = time_series_df["action_count"].sum()
            total_successes = time_series_df["success_count"].sum()
            avg_success_rate = total_successes / total_actions if total_actions > 0 else 0.0
        else:
            avg_success_rate = 0.0

        # Calculate efficiency trend (improvement over time)
        if "reward" in time_series_df.columns and len(time_series_df) > 1:
            efficiency_trend = calculate_trend(time_series_df["reward"].values)
        else:
            efficiency_trend = 0.0

        return {
            "avg_actions_per_step": float(avg_actions_per_step),
            "avg_reward_per_step": float(avg_reward_per_step),
            "avg_success_rate": float(avg_success_rate),
            "efficiency_trend": float(efficiency_trend),
        }


def _analyze_event_impacts(segment_metrics: Dict[str, Dict], event_steps: List[int]) -> Dict[str, Any]:
    """Analyze the impact of events on temporal patterns.

    Args:
        segment_metrics: Metrics for each segment
        event_steps: Event step numbers

    Returns:
        Dictionary with event impact analysis
    """
    if len(segment_metrics) < 2:
        return {}

    impacts = {}

    for i, event_step in enumerate(event_steps):
        before_segment = f"segment_{i}"
        after_segment = f"segment_{i + 1}"

        if before_segment in segment_metrics and after_segment in segment_metrics:
            before = segment_metrics[before_segment]
            after = segment_metrics[after_segment]

            # Calculate changes
            reward_change = after["avg_reward"] - before["avg_reward"]
            action_change = after["action_count"] - before["action_count"]

            impacts[f"event_{event_step}"] = {
                "reward_change": float(reward_change),
                "action_change": int(action_change),
                "reward_change_pct": float(reward_change / before["avg_reward"]) if before["avg_reward"] != 0 else 0.0,
                "action_change_pct": float(action_change / before["action_count"])
                if before["action_count"] != 0
                else 0.0,
            }

    return impacts


def _calculate_autocorrelation(values: np.ndarray, lag: int = 1) -> float:
    """Calculate autocorrelation for a time series.

    Args:
        values: Time series values
        lag: Lag for autocorrelation

    Returns:
        Autocorrelation coefficient
    """
    if len(values) <= lag:
        return 0.0

    try:
        corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except (ValueError, IndexError, np.linalg.LinAlgError):
        return 0.0
