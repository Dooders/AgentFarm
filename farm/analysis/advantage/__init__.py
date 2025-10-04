"""
Relative Advantage Analysis Package

This package provides tools for analyzing the relative advantages between agent types
and how these advantages contribute to dominance patterns in simulations.
"""

from farm.analysis.advantage.compute import compute_advantages
from farm.analysis.advantage.analyze import analyze_advantage_patterns
from farm.analysis.advantage.plot import (
    plot_advantage_results,
    plot_advantage_correlation_matrix,
    plot_advantage_distribution,
    plot_advantage_timeline,
)

__all__ = [
    'compute_advantages',
    'analyze_advantage_patterns',
    'plot_advantage_results',
    'plot_advantage_correlation_matrix',
    'plot_advantage_distribution',
    'plot_advantage_timeline',
    'advantage_module'
]

# Import the module instance
from farm.analysis.advantage.module import advantage_module 