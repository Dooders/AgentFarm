"""
Utility functions for benchmarks.
"""

from benchmarks.utils.visualization import plot_comparison, plot_multiple_results
from benchmarks.utils.statistics import calculate_statistics
from benchmarks.utils.config_helper import (
    configure_for_performance_with_persistence,
    get_recommended_config,
    print_config_recommendations
)

__all__ = [
    "plot_comparison", 
    "plot_multiple_results", 
    "calculate_statistics",
    "configure_for_performance_with_persistence",
    "get_recommended_config",
    "print_config_recommendations"
] 