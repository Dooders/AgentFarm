"""
Genesis Analysis Package

This package provides tools for analyzing how initial states and conditions
impact simulation outcomes, focusing on the relationship between starting
configurations and eventual dominance patterns.
"""

from farm.analysis.genesis.compute import (
    compute_initial_state_metrics,
    compute_genesis_impact_scores,
    compute_critical_period_metrics
)
from farm.analysis.genesis.analyze import (
    analyze_genesis_factors,
    analyze_genesis_across_simulations,
    analyze_critical_period,
    analyze_genesis_patterns,
)
from farm.analysis.genesis.plot import (
    plot_genesis_analysis_results,
    plot_initial_state_comparison,
    plot_critical_period_analysis,
    plot_genesis_patterns,
    plot_genesis_timeline,
)

__all__ = [
    'compute_initial_state_metrics',
    'compute_genesis_impact_scores',
    'compute_critical_period_metrics',
    'analyze_genesis_factors',
    'analyze_genesis_across_simulations',
    'analyze_critical_period',
    'analyze_genesis_patterns',
    'plot_genesis_analysis_results',
    'plot_initial_state_comparison',
    'plot_critical_period_analysis',
    'plot_genesis_patterns',
    'plot_genesis_timeline',
    'genesis_module'
]

# Import the module instance
from farm.analysis.genesis.module import genesis_module 