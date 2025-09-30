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
    results: List[Dict[str, Any]]
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
    results: List[Dict[str, Any]], 
    required_field: str = "build_time"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter results by required field and group by implementation.

    This is a common pattern used across benchmark scripts to process results.

    Args:
        results: List of result dictionaries
        required_field: Field that must be present (default: "build_time")

    Returns:
        Dictionary mapping implementation names to filtered result lists
    """
    filtered = filter_results_by_field(results, required_field)
    return group_results_by_implementation(filtered)


def get_best_implementation(
    results_by_impl: Dict[str, List[Dict[str, Any]]], 
    metric: str,
    minimize: bool = True
) -> str:
    """
    Find the best implementation based on a specific metric.

    Args:
        results_by_impl: Results grouped by implementation
        metric: Field name to compare (e.g., 'build_time', 'avg_query_time')
        minimize: If True, lower values are better. If False, higher is better.

    Returns:
        Name of the best performing implementation

    Raises:
        ValueError: If no implementations have the required metric
    """
    if not results_by_impl:
        raise ValueError("No implementations to compare")
    
    # Filter to only implementations with the metric
    valid_impls = {
        impl: results 
        for impl, results in results_by_impl.items()
        if results and all(metric in r for r in results)
    }
    
    if not valid_impls:
        raise ValueError(f"No implementations have metric '{metric}'")
    
    # Calculate average metric for each implementation
    avg_metrics = {
        impl: sum(r[metric] for r in results) / len(results)
        for impl, results in valid_impls.items()
    }
    
    # Find best implementation
    if minimize:
        return min(avg_metrics.keys(), key=lambda impl: avg_metrics[impl])
    else:
        return max(avg_metrics.keys(), key=lambda impl: avg_metrics[impl])
