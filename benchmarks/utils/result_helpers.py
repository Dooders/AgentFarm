"""Helper functions for processing and analyzing benchmark results.

This module provides common utilities for filtering, grouping, and analyzing
benchmark results to avoid code duplication across benchmark scripts.
"""

from typing import Any, Dict, List


def filter_results_by_field(
    results: List[Dict[str, Any]], required_field: str
) -> List[Dict[str, Any]]:
    """
    Filter results to include only those with a specific field.

    This is useful for filtering out incomplete or malformed benchmark results
    that might be missing critical metrics.

    Args:
        results: List of result dictionaries
        required_field: Field name that must be present in results

    Returns:
        Filtered list of results containing the required field
    """
    return [r for r in results if required_field in r]


def group_results_by_implementation(
    results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group results by implementation name.

    Args:
        results: List of result dictionaries with 'implementation' field

    Returns:
        Dictionary mapping implementation names to their result lists
    """
    by_impl = {}
    for result in results:
        impl = result.get("implementation")
        if impl is not None:
            if impl not in by_impl:
                by_impl[impl] = []
            by_impl[impl].append(result)
    return by_impl


def group_results_by_distribution(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group results by distribution pattern.

    Args:
        results: List of result dictionaries with 'distribution' field

    Returns:
        Dictionary mapping distribution names to their result lists
    """
    by_dist = {}
    for result in results:
        dist = result.get("distribution")
        if dist is not None:
            if dist not in by_dist:
                by_dist[dist] = []
            by_dist[dist].append(result)
    return by_dist


def filter_and_group_results(
    results: List[Dict[str, Any]], required_field: str = "build_time"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter results by required field and group by implementation.

    This is a common pattern used across benchmark scripts to process results.
    It combines filtering and grouping operations that are frequently used together
    when analyzing benchmark results.

    Args:
        results: List of result dictionaries
        required_field: Field that must be present (default: "build_time")

    Returns:
        Dictionary mapping implementation names to filtered result lists
    """
    # First filter out incomplete results
    filtered = filter_results_by_field(results, required_field)
    # Then group by implementation for analysis
    return group_results_by_implementation(filtered)


def get_best_implementation(
    results_by_impl: Dict[str, List[Dict[str, Any]]], metric: str, minimize: bool = True
) -> str:
    """
    Find the best implementation based on a specific metric.

    This function analyzes results grouped by implementation and identifies
    the implementation with the best average performance for the specified metric.
    It handles cases where some implementations may not have the required metric.

    Parameters
    ----------
    results_by_impl : Dict[str, List[Dict[str, Any]]]
        Results grouped by implementation name, where each value is a list
        of result dictionaries from multiple benchmark runs
    metric : str
        Field name to compare (e.g., 'build_time', 'avg_query_time', 'memory_usage')
    minimize : bool, default=True
        If True, lower values are considered better (e.g., for time, memory).
        If False, higher values are considered better (e.g., for throughput).

    Returns
    -------
    str
        Name of the best performing implementation

    Raises
    ------
    ValueError
        If no implementations have the required metric or if results_by_impl is empty
    """
    if not results_by_impl:
        raise ValueError("No implementations to compare")

    # Filter to only implementations that have the required metric in all their results
    valid_impls = {
        impl: results
        for impl, results in results_by_impl.items()
        if results and all(metric in r for r in results)
    }

    if not valid_impls:
        raise ValueError(f"No implementations have metric '{metric}'")

    # Calculate average metric value for each implementation across all runs
    avg_metrics = {
        impl: sum(r[metric] for r in results) / len(results)
        for impl, results in valid_impls.items()
    }

    # Find the best performing implementation based on optimization direction
    if minimize:
        return min(avg_metrics.keys(), key=lambda impl: avg_metrics[impl])
    else:
        return max(avg_metrics.keys(), key=lambda impl: avg_metrics[impl])
