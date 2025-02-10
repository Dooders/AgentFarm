"""Utility functions for comparing simulation results.

This module provides functions to compare two simulations and analyze their differences
across various metrics, parameters, and results. It supports both programmatic comparison
and formatted output generation.

The module focuses on four main comparison areas:
1. Metadata - Basic simulation properties and status
2. Parameters - Configuration differences between simulations
3. Results - Overall outcome differences
4. Metrics - Statistical differences in step-by-step measurements

Examples
--------
Basic comparison using the main comparison function:
    >>> from farm.database.simulation_comparison import compare_simulations
    >>> differences = compare_simulations(session, sim1_id=1, sim2_id=2)
    >>> print(differences["metrics"]["total_agents"]["mean_difference"])
    5.321

Get a human-readable summary:
    >>> from farm.database.simulation_comparison import summarize_comparison
    >>> summary = summarize_comparison(session, sim1_id=1, sim2_id=2)
    >>> print(summary)
    Comparison Summary: Simulation 1 vs 2
    
    Parameter Changes:
      environment.resource_density: 0.1 -> 0.2
    
    Key Metrics (averages):
      total_agents: 5.32 higher in simulation 2
      average_reward: 0.45 lower in simulation 2

Find significant changes:
    >>> from farm.database.simulation_comparison import get_significant_changes
    >>> significant = get_significant_changes(
    ...     session, sim1_id=1, sim2_id=2, threshold=0.15
    ... )
    >>> for metric, change in significant.items():
    ...     print(f"{metric}: {change:+.2f}")
    total_agents: +5.32
    average_reward: -0.45

Format comparison output:
    >>> from farm.database.simulation_comparison import format_diff_output
    >>> formatted = format_diff_output(differences)
    >>> print(formatted["metrics"]["total_agents"])
    {
        'mean_difference': 5.321,
        'max_difference': 8.0,
        'min_difference': 2.0,
        'std_dev_difference': 1.234
    }

Notes
-----
- All comparison functions require an active database session
- Metric differences are calculated as (sim2 - sim1)
- Statistical comparisons include mean, max, min, and standard deviation
- The module uses DeepDiff for comparing complex nested structures
- Significant changes are determined by relative difference magnitude

See Also
--------
farm.tools.compare_sims : Command-line interface for simulation comparison
farm.database.models.SimulationComparison : Core comparison class
"""

from typing import Optional, Tuple, Dict, Any
from .models import Simulation, SimulationComparison, SimulationDifference

def format_diff_output(diff: SimulationDifference) -> Dict[str, Any]:
    """Format the simulation differences into a more readable structure.
    
    Parameters
    ----------
    diff : SimulationDifference
        Raw difference object from comparison
        
    Returns
    -------
    Dict[str, Any]
        Formatted difference report
    """
    report = {
        "metadata": {},
        "parameters": {},
        "results": {},
        "metrics": {}
    }
    
    # Format metadata differences
    for field, (val1, val2) in diff.metadata_diff.items():
        report["metadata"][field] = {
            "simulation_1": val1,
            "simulation_2": val2
        }
    
    # Format parameter differences
    if diff.parameter_diff:
        report["parameters"] = {
            "added": diff.parameter_diff.get("dictionary_item_added", []),
            "removed": diff.parameter_diff.get("dictionary_item_removed", []),
            "changed": diff.parameter_diff.get("values_changed", {})
        }
    
    # Format results differences
    if diff.results_diff:
        report["results"] = {
            "added": diff.results_diff.get("dictionary_item_added", []),
            "removed": diff.results_diff.get("dictionary_item_removed", []),
            "changed": diff.results_diff.get("values_changed", {})
        }
    
    # Format metrics differences
    for metric, stats in diff.step_metrics_diff.items():
        report["metrics"][metric] = {
            "mean_difference": round(stats["mean_diff"], 3),
            "max_difference": round(stats["max_diff"], 3),
            "min_difference": round(stats["min_diff"], 3),
            "std_dev_difference": round(stats["std_diff"], 3)
        }
    
    return report

def compare_simulations(
    session,
    sim1_id: int,
    sim2_id: int,
    format_output: bool = True
) -> Dict[str, Any]:
    """Compare two simulations and return their differences.
    
    Parameters
    ----------
    session : Session
        SQLAlchemy session for database queries
    sim1_id : int
        ID of first simulation to compare
    sim2_id : int
        ID of second simulation to compare
    format_output : bool, optional
        Whether to format the output into a more readable structure
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the differences between simulations
        
    Examples
    --------
    >>> differences = compare_simulations(session, 1, 2)
    >>> print(differences["metrics"]["total_agents"]["mean_difference"])
    -5.3
    """
    # Load simulations
    sim1 = session.query(Simulation).get(sim1_id)
    sim2 = session.query(Simulation).get(sim2_id)
    
    if not sim1 or not sim2:
        raise ValueError("One or both simulation IDs not found")
    
    # Perform comparison
    comparison = SimulationComparison(sim1, sim2)
    differences = comparison.compare(session)
    
    # Format output if requested
    if format_output:
        return format_diff_output(differences)
    return differences

def summarize_comparison(
    session,
    sim1_id: int,
    sim2_id: int
) -> str:
    """Generate a human-readable summary of simulation differences.
    
    Parameters
    ----------
    session : Session
        SQLAlchemy session for database queries
    sim1_id : int
        ID of first simulation to compare
    sim2_id : int
        ID of second simulation to compare
        
    Returns
    -------
    str
        Formatted string containing key differences
    """
    differences = compare_simulations(session, sim1_id, sim2_id)
    
    summary = [f"Comparison Summary: Simulation {sim1_id} vs {sim2_id}\n"]
    
    # Summarize metadata differences
    if differences["metadata"]:
        summary.append("Metadata Differences:")
        for field, values in differences["metadata"].items():
            summary.append(f"  {field}: {values['simulation_1']} -> {values['simulation_2']}")
    
    # Summarize parameter changes
    if any(differences["parameters"].values()):
        summary.append("\nParameter Changes:")
        if differences["parameters"]["changed"]:
            for path, change in differences["parameters"]["changed"].items():
                summary.append(f"  {path}: {change['old_value']} -> {change['new_value']}")
    
    # Summarize metric differences
    if differences["metrics"]:
        summary.append("\nKey Metrics (averages):")
        for metric, stats in differences["metrics"].items():
            mean_diff = stats["mean_difference"]
            if abs(mean_diff) > 0.001:  # Only show meaningful differences
                direction = "higher" if mean_diff > 0 else "lower"
                summary.append(
                    f"  {metric}: {abs(mean_diff):.2f} {direction} in simulation {sim2_id}"
                )
    
    return "\n".join(summary)

def get_significant_changes(
    session,
    sim1_id: int,
    sim2_id: int,
    threshold: float = 0.1
) -> Dict[str, float]:
    """Identify metrics with significant differences between simulations.
    
    Parameters
    ----------
    session : Session
        SQLAlchemy session for database queries
    sim1_id : int
        ID of first simulation to compare
    sim2_id : int
        ID of second simulation to compare
    threshold : float, optional
        Minimum relative difference to be considered significant
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics with significant differences and their relative changes
    """
    differences = compare_simulations(session, sim1_id, sim2_id)
    
    significant_changes = {}
    
    for metric, stats in differences["metrics"].items():
        mean_diff = stats["mean_difference"]
        if abs(mean_diff) > threshold:
            significant_changes[metric] = mean_diff
            
    return significant_changes 