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

    # Calculate basic descriptive statistics
    mean = statistics.mean(durations)
    median = statistics.median(durations)
    min_val = min(durations)
    max_val = max(durations)

    # Calculate measures of variability
    std = statistics.stdev(durations) if len(durations) > 1 else 0.0
    variance = statistics.variance(durations) if len(durations) > 1 else 0.0
    range_val = max_val - min_val

    # Calculate interquartile range (IQR) - measure of statistical dispersion
    if len(durations) >= 4:
        sorted_durations = sorted(durations)
        q1 = float(np.percentile(sorted_durations, 25))  # First quartile
        q3 = float(np.percentile(sorted_durations, 75))  # Third quartile
        iqr = q3 - q1
    else:
        iqr = 0.0

    # Calculate coefficient of variation (CV) - relative variability measure
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

    # Calculate percentage differences between the two result sets
    comparison = {}

    for key in stats1:
        if stats1[key] == 0 and stats2[key] == 0:
            comparison[f"{key}_diff_pct"] = 0.0
        elif stats1[key] == 0:
            comparison[f"{key}_diff_pct"] = float("inf")  # Infinite improvement
        else:
            comparison[f"{key}_diff_pct"] = (
                (stats2[key] - stats1[key]) / stats1[key]
            ) * 100

    # Calculate absolute differences for each metric
    for key in stats1:
        comparison[f"{key}_diff_abs"] = stats2[key] - stats1[key]

    # Calculate speedup/slowdown factors (higher is better for performance)
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

    This function helps identify how different parameter values affect benchmark
    performance by calculating correlations, finding optimal values, and measuring
    the improvement potential.

    Parameters
    ----------
    results_dict : Dict[Any, RunResult]
        Dictionary mapping parameter values to benchmark results
    metric : str, default="mean_duration"
        Metric to analyze. Supported values:
        - "mean_duration": Average execution time across iterations
        - "median_duration": Median execution time across iterations

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - correlation: Pearson correlation coefficient (if parameters are numeric)
        - min_value: Best (lowest) metric value achieved
        - max_value: Worst (highest) metric value achieved
        - min_param: Parameter value that achieved best performance
        - max_param: Parameter value that achieved worst performance
        - improvement_pct: Percentage improvement from worst to best
    """
    if not results_dict:
        return {}

    # Extract parameter values and their corresponding performance metrics
    param_values = list(results_dict.keys())

    # Map metric names to result extraction methods
    if metric == "mean_duration":
        values = [result.get_mean_duration() for result in results_dict.values()]
    elif metric == "median_duration":
        values = [result.get_median_duration() for result in results_dict.values()]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Calculate Pearson correlation coefficient if parameters are numeric
    correlation = None
    if all(isinstance(p, (int, float)) for p in param_values):
        if len(param_values) > 1:
            correlation = np.corrcoef(param_values, values)[0, 1]

    # Identify optimal parameter values (best and worst performance)
    min_value = min(values)
    max_value = max(values)
    min_param = param_values[values.index(min_value)]  # Best parameter value
    max_param = param_values[values.index(max_value)]  # Worst parameter value

    # Calculate performance improvement potential
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
