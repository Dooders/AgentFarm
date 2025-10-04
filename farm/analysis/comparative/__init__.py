"""
Comparative analysis module.

Provides comprehensive analysis for comparing multiple simulations including:
- Statistical comparison of metrics
- Parameter difference analysis
- Performance comparison plots
- Comparative visualization
"""

from farm.analysis.comparative.module import comparative_module, ComparativeModule
from farm.analysis.comparative.compute import (
    compute_comparison_metrics,
    compute_parameter_differences,
    compute_performance_comparison,
)
from farm.analysis.comparative.analyze import (
    analyze_simulation_comparison,
    analyze_parameter_differences,
    analyze_performance_comparison,
)
from farm.analysis.comparative.plot import (
    plot_comparison_metrics,
    plot_parameter_differences,
    plot_performance_comparison,
    plot_simulation_comparison,
)
from farm.analysis.comparative.compare import compare_simulations

__all__ = [
    "comparative_module",
    "ComparativeModule",
    "compare_simulations",
    "compute_comparison_metrics",
    "compute_parameter_differences",
    "compute_performance_comparison",
    "analyze_simulation_comparison",
    "analyze_parameter_differences",
    "analyze_performance_comparison",
    "plot_comparison_metrics",
    "plot_parameter_differences",
    "plot_performance_comparison",
    "plot_simulation_comparison",
]
