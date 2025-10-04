"""
Spatial analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.spatial.compute import (
    compute_spatial_statistics,
    compute_movement_patterns,
    compute_location_hotspots,
    compute_spatial_distribution_metrics,
)
from farm.analysis.spatial.data import (
    process_spatial_data,
    process_movement_data,
    process_location_analysis_data,
)


def analyze_spatial_overview(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze overall spatial patterns and save results.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing spatial overview...")

    from pathlib import Path
    spatial_data = process_spatial_data(Path(experiment_path))

    # Compute spatial statistics
    stats = compute_spatial_statistics(spatial_data)

    # Save to file
    output_file = ctx.get_output_file("spatial_overview.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    ctx.logger.info(f"Saved spatial overview to {output_file}")
    ctx.report_progress("Spatial overview analysis complete", 0.4)


def analyze_movement_patterns(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze movement patterns and trajectories.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing movement patterns...")

    agent_ids = kwargs.get('agent_ids')
    from pathlib import Path
    movement_df = process_movement_data(Path(experiment_path), agent_ids=agent_ids)

    if movement_df.empty:
        ctx.logger.warning("No movement data found")
        return

    # Compute movement patterns
    patterns = compute_movement_patterns(movement_df)

    # Save to file
    output_file = ctx.get_output_file("movement_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(patterns, f, indent=2, default=str)

    ctx.logger.info(f"Saved movement patterns to {output_file}")
    ctx.report_progress("Movement pattern analysis complete", 0.6)


def analyze_location_hotspots(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze location hotspots and clustering.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing location hotspots...")

    from pathlib import Path
    location_data = process_location_analysis_data(Path(experiment_path))

    # Compute hotspots and clustering
    hotspots = compute_location_hotspots(location_data)

    # Save to file
    output_file = ctx.get_output_file("location_hotspots.json")
    with open(output_file, 'w') as f:
        json.dump(hotspots, f, indent=2, default=str)

    ctx.logger.info(f"Saved location hotspots to {output_file}")
    ctx.report_progress("Location hotspots analysis complete", 0.8)


def analyze_spatial_distribution(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze spatial distribution patterns.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing spatial distribution...")

    from pathlib import Path
    spatial_data = process_spatial_data(Path(experiment_path))

    distribution_metrics = {}

    # Analyze agent distribution
    agent_df = spatial_data.get('agent_positions', pd.DataFrame())
    if not agent_df.empty:
        distribution_metrics['agents'] = compute_spatial_distribution_metrics(agent_df)

    # Analyze resource distribution
    resource_df = spatial_data.get('resource_positions', pd.DataFrame())
    if not resource_df.empty:
        distribution_metrics['resources'] = compute_spatial_distribution_metrics(resource_df)

    # Save to file
    output_file = ctx.get_output_file("spatial_distribution.json")
    with open(output_file, 'w') as f:
        json.dump(distribution_metrics, f, indent=2, default=str)

    ctx.logger.info(f"Saved spatial distribution to {output_file}")
    ctx.report_progress("Spatial distribution analysis complete", 1.0)
