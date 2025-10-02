"""
Statistical utilities for benchmarks.
"""

import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.core.results import RunResult


def calculate_statistics(results: RunResult) -> Dict[str, float]:
    """
    Calculate various statistics for benchmark results.

    Parameters
    ----------
    results : RunResult
        Benchmark results to analyze

    Returns
    -------
    Dict[str, float]
        Dictionary of statistics
    """
    # Extract durations from the new result structure
    durations = []
    if results.iteration_metrics:
        durations = [iteration.duration_s for iteration in results.iteration_metrics]

    if not durations:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "variance": 0.0,
            "range": 0.0,
            "iqr": 0.0,
            "cv": 0.0,  # coefficient of variation
        }

    # Calculate basic statistics
    mean = statistics.mean(durations)
    median = statistics.median(durations)
    min_val = min(durations)
    max_val = max(durations)

    # Calculate more advanced statistics
    std = statistics.stdev(durations) if len(durations) > 1 else 0.0
    variance = statistics.variance(durations) if len(durations) > 1 else 0.0
    range_val = max_val - min_val

    # Calculate interquartile range (IQR)
    if len(durations) >= 4:
        sorted_durations = sorted(durations)
        q1 = float(np.percentile(sorted_durations, 25))
        q3 = float(np.percentile(sorted_durations, 75))
        iqr = q3 - q1
    else:
        iqr = 0.0

    # Calculate coefficient of variation (CV)
    cv = (std / mean) * 100 if mean > 0 else 0.0

    return {
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val,
        "std": std,
        "variance": variance,
        "range": range_val,
        "iqr": iqr,
        "cv": cv,
    }


def compare_statistics(results1: RunResult, results2: RunResult) -> Dict[str, float]:
    """
    Compare statistics between two benchmark results.

    Parameters
    ----------
    results1 : RunResult
        First benchmark results
    results2 : RunResult
        Second benchmark results

    Returns
    -------
    Dict[str, float]
        Dictionary of comparison metrics
    """
    stats1 = calculate_statistics(results1)
    stats2 = calculate_statistics(results2)

    # Calculate percentage differences
    comparison = {}

    for key in stats1:
        if stats1[key] == 0 and stats2[key] == 0:
            comparison[f"{key}_diff_pct"] = 0.0
        elif stats1[key] == 0:
            comparison[f"{key}_diff_pct"] = float("inf")
        else:
            comparison[f"{key}_diff_pct"] = (
                (stats2[key] - stats1[key]) / stats1[key]
            ) * 100

    # Add absolute differences
    for key in stats1:
        comparison[f"{key}_diff_abs"] = stats2[key] - stats1[key]

    # Add speedup/slowdown factor for mean and median
    if stats1["mean"] > 0:
        comparison["mean_speedup"] = stats1["mean"] / stats2["mean"]
    else:
        comparison["mean_speedup"] = 0.0

    if stats1["median"] > 0:
        comparison["median_speedup"] = stats1["median"] / stats2["median"]
    else:
        comparison["median_speedup"] = 0.0

    return comparison


def analyze_parameter_impact(
    results_dict: Dict[Any, RunResult], metric: str = "mean_duration"
) -> Dict[str, Any]:
    """
    Analyze the impact of a parameter on benchmark results.

    Parameters
    ----------
    results_dict : Dict[Any, RunResult]
        Dictionary mapping parameter values to benchmark results
    metric : str
        Metric to analyze (e.g., "mean_duration", "median_duration")

    Returns
    -------
    Dict[str, Any]
        Dictionary of analysis results
    """
    if not results_dict:
        return {}

    # Extract parameter values and corresponding metric values
    param_values = list(results_dict.keys())

    if metric == "mean_duration":
        values = [result.get_mean_duration() for result in results_dict.values()]
    elif metric == "median_duration":
        values = [result.get_median_duration() for result in results_dict.values()]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Calculate correlation if parameter values are numeric
    correlation = None
    if all(isinstance(p, (int, float)) for p in param_values):
        if len(param_values) > 1:
            correlation = np.corrcoef(param_values, values)[0, 1]

    # Find minimum and maximum values
    min_value = min(values)
    max_value = max(values)
    min_param = param_values[values.index(min_value)]
    max_param = param_values[values.index(max_value)]

    # Calculate improvement from worst to best
    improvement_pct = (
        ((max_value - min_value) / max_value) * 100 if max_value > 0 else 0.0
    )

    return {
        "correlation": correlation,
        "min_value": min_value,
        "max_value": max_value,
        "min_param": min_param,
        "max_param": max_param,
        "improvement_pct": improvement_pct,
    }
