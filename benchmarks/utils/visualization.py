"""
Visualization utilities for benchmarks.
"""

import os
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.core.results import RunResult


def plot_comparison(results_list: List[RunResult], 
                   metric: str = "mean_duration",
                   title: Optional[str] = None,
                   save_path: Optional[str] = None) -> None:
    """
    Plot a comparison of multiple benchmark results.
    
    Parameters
    ----------
    results_list : List[RunResult]
        List of benchmark results to compare
    metric : str
        Metric to compare (e.g., "mean_duration", "median_duration")
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    if not results_list:
        print("No results to plot")
        return
    
    # Extract data
    names = [result.name for result in results_list]
    
    if metric == "mean_duration":
        values = [result.get_mean_duration() for result in results_list]
        ylabel = "Mean Duration (seconds)"
    elif metric == "median_duration":
        values = [result.get_median_duration() for result in results_list]
        ylabel = "Median Duration (seconds)"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.title(title or f"Benchmark Comparison ({metric})")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # Rotate x-axis labels if there are many benchmarks
    if len(names) > 5:
        plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_multiple_results(results_list: List[RunResult],
                         metric: str = "duration",
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot multiple benchmark results as line plots.
    
    Parameters
    ----------
    results_list : List[RunResult]
        List of benchmark results to plot
    metric : str
        Metric to plot (e.g., "duration")
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    if not results_list:
        print("No results to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    for result in results_list:
        if metric == "duration":
            iterations = range(len(result.iteration_results))
            values = result.get_durations()
            plt.plot(iterations, values, marker='o', label=result.name)
            plt.ylabel("Duration (seconds)")
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    plt.xlabel("Iteration")
    plt.title(title or f"Benchmark Results ({metric})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_parameter_impact(results_dict: Dict[Any, RunResult],
                         parameter_name: str,
                         metric: str = "mean_duration",
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot the impact of a parameter on benchmark results.
    
    Parameters
    ----------
    results_dict : Dict[Any, RunResult]
        Dictionary mapping parameter values to benchmark results
    parameter_name : str
        Name of the parameter being varied
    metric : str
        Metric to plot (e.g., "mean_duration", "median_duration")
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    if not results_dict:
        print("No results to plot")
        return
    
    # Extract data
    param_values = list(results_dict.keys())
    
    if metric == "mean_duration":
        values = [result.get_mean_duration() for result in results_dict.values()]
        ylabel = "Mean Duration (seconds)"
    elif metric == "median_duration":
        values = [result.get_median_duration() for result in results_dict.values()]
        ylabel = "Median Duration (seconds)"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(param_values, values, marker='o')
    plt.xlabel(parameter_name)
    plt.ylabel(ylabel)
    plt.title(title or f"Impact of {parameter_name} on {metric}")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Add values next to points
    for i, v in enumerate(values):
        plt.text(param_values[i], v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show() 