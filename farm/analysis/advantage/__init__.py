"""
Relative Advantage Analysis Package

This package provides tools for analyzing the relative advantages between agent types
and how these advantages contribute to dominance patterns in simulations.
"""

from farm.analysis.advantage.compute import compute_advantages
from farm.analysis.advantage.analyze import analyze_advantage_patterns
from farm.analysis.advantage.plot import plot_advantage_results

__all__ = [
    'compute_advantages',
    'analyze_advantage_patterns',
    'plot_advantage_results'
] 