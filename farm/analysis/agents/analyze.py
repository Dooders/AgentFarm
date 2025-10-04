"""
Agent analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.agents.compute import (
    compute_lifespan_statistics,
    compute_behavior_patterns,
    compute_performance_metrics,
    compute_learning_curves,
)


def analyze_lifespan_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze agent lifespan patterns and save results.

    Args:
        df: Agent data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing agent lifespan patterns...")

    # Compute statistics
    stats = compute_lifespan_statistics(df)

    # Save to file
    output_file = ctx.get_output_file("lifespan_statistics.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    ctx.logger.info(f"Saved statistics to {output_file}")
    ctx.report_progress("Lifespan analysis complete", 0.4)


def analyze_behavior_clustering(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze agent behavior clustering.

    Args:
        df: Agent data with behavior metrics
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing agent behavior clustering...")

    patterns = compute_behavior_patterns(df)

    # Save behavior analysis
    output_file = ctx.get_output_file("behavior_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(patterns, f, indent=2)

    ctx.logger.info(f"Saved behavior analysis to {output_file}")
    ctx.report_progress("Behavior clustering analysis complete", 0.6)


def analyze_performance_analysis(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze agent performance metrics.

    Args:
        df: Agent data with performance metrics
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing agent performance...")

    performance = compute_performance_metrics(df)

    # Save performance analysis
    output_file = ctx.get_output_file("performance_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(performance, f, indent=2)

    ctx.logger.info(f"Saved performance analysis to {output_file}")
    ctx.report_progress("Performance analysis complete", 0.8)


def analyze_learning_curves(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze agent learning curves.

    Args:
        df: Agent data with learning metrics
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing agent learning curves...")

    curves = compute_learning_curves(df)

    # Save learning analysis
    output_file = ctx.get_output_file("learning_curves.json")
    with open(output_file, 'w') as f:
        json.dump(curves, f, indent=2)

    ctx.logger.info(f"Saved learning analysis to {output_file}")
    ctx.report_progress("Learning curves analysis complete", 0.9)
