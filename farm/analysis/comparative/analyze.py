"""
Comparative analysis functions.
"""

import json
import pandas as pd
from pathlib import Path

from farm.analysis.common.context import AnalysisContext
from farm.analysis.comparative.compute import (
    compute_comparison_metrics,
    compute_parameter_differences,
    compute_performance_comparison,
)


def analyze_simulation_comparison(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze simulation comparison and save results.

    Args:
        df: Comparison data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing simulation comparison...")

    # Compute comparison metrics
    metrics = compute_comparison_metrics(df)

    # Save to file
    output_file = ctx.get_output_file("comparison_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    ctx.logger.info(f"Saved comparison metrics to {output_file}")
    ctx.report_progress("Simulation comparison analysis complete", 0.4)


def analyze_parameter_differences(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze parameter differences between simulations.

    Args:
        df: Parameter data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing parameter differences...")

    # Compute parameter differences
    differences = compute_parameter_differences(df)

    # Save to file
    output_file = ctx.get_output_file("parameter_differences.json")
    with open(output_file, 'w') as f:
        json.dump(differences, f, indent=2)

    ctx.logger.info(f"Saved parameter differences to {output_file}")
    ctx.report_progress("Parameter differences analysis complete", 0.4)


def analyze_performance_comparison(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze performance comparison between simulations.

    Args:
        df: Performance data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing performance comparison...")

    # Compute performance comparison
    performance = compute_performance_comparison(df)

    # Save to file
    output_file = ctx.get_output_file("performance_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(performance, f, indent=2)

    ctx.logger.info(f"Saved performance comparison to {output_file}")
    ctx.report_progress("Performance comparison analysis complete", 0.4)
