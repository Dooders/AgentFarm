"""
Resource analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.resources.compute import (
    compute_resource_statistics,
    compute_consumption_patterns,
    compute_resource_efficiency,
    compute_resource_hotspots,
)


def analyze_resource_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze resource distribution patterns and save results.

    Args:
        df: Resource data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing resource patterns...")

    # Compute statistics
    stats = compute_resource_statistics(df)
    consumption = compute_consumption_patterns(df)
    efficiency = compute_resource_efficiency(df)
    hotspots = compute_resource_hotspots(df)

    # Combine results
    results = {
        'statistics': stats,
        'patterns': consumption,  # Use consumption as patterns
        'efficiency': efficiency,
        'hotspots': hotspots,
    }

    # Save to file
    output_file = ctx.get_output_file("resource_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    ctx.logger.info(f"Saved statistics to {output_file}")
    ctx.report_progress("Resource patterns analysis complete", 0.5)


def analyze_consumption(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze resource consumption patterns in detail.

    Args:
        df: Resource data with consumption metrics
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing resource consumption patterns...")

    # Save consumption analysis as CSV
    output_file = ctx.get_output_file("consumption_patterns.csv")
    df.to_csv(output_file, index=False)

    ctx.logger.info(f"Saved consumption analysis to {output_file}")
    ctx.report_progress("Consumption analysis complete", 0.7)


def analyze_resource_efficiency(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze resource utilization efficiency.

    Args:
        df: Resource data with efficiency metrics
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing resource efficiency...")

    efficiency = compute_resource_efficiency(df)

    # Calculate improvement rate if efficiency_gain is available
    improvement_rate = 0.0
    if 'efficiency_gain' in df.columns:
        improvement_rate = float(df['efficiency_gain'].mean())

    # Save efficiency analysis
    output_file = ctx.get_output_file("efficiency_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': efficiency,
            'improvement_rate': improvement_rate
        }, f, indent=2)

    ctx.logger.info(f"Saved efficiency analysis to {output_file}")
    ctx.report_progress("Efficiency analysis complete", 0.8)


def analyze_hotspots(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze resource hotspot patterns.

    Args:
        df: Resource data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing resource hotspots...")

    hotspots = compute_resource_hotspots(df)

    # Save hotspot analysis
    output_file = ctx.get_output_file("hotspot_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(hotspots, f, indent=2)

    ctx.logger.info(f"Saved hotspot analysis to {output_file}")
    ctx.report_progress("Hotspot analysis complete", 0.9)


