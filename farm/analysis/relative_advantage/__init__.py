"""
Relative Advantage Analysis Package

This package provides tools for analyzing the relative advantages between agent types
and how these advantages contribute to dominance patterns in simulations.
"""

from farm.analysis.relative_advantage.compute import compute_relative_advantages
from farm.analysis.relative_advantage.analyze import analyze_relative_advantages
from farm.analysis.relative_advantage.plot import plot_relative_advantage_results

__all__ = [
    'compute_relative_advantages',
    'analyze_relative_advantages',
    'plot_relative_advantage_results'
] 