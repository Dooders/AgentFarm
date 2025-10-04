"""
Temporal analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.temporal.compute import (
    compute_temporal_statistics,
    compute_event_segmentation_metrics,
    compute_temporal_patterns,
    compute_temporal_efficiency_metrics,
)
from farm.analysis.temporal.data import (
    process_time_series_data,
    process_event_segmentation_data,
    extract_temporal_patterns,
)


def analyze_temporal_patterns(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze temporal patterns in action data.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options (time_period_size, rolling_window_size)
    """
    ctx.logger.info("Analyzing temporal patterns...")

    time_period_size = kwargs.get('time_period_size', 100)
    rolling_window_size = kwargs.get('rolling_window_size', 10)

    from pathlib import Path
    patterns_df = extract_temporal_patterns(
        Path(experiment_path),
        time_period_size=time_period_size,
        rolling_window_size=rolling_window_size
    )

    if patterns_df.empty:
        ctx.logger.warning("No temporal pattern data found")
        return

    # Compute temporal patterns
    patterns = compute_temporal_patterns(patterns_df)
    efficiency = compute_temporal_efficiency_metrics(patterns_df)

    # Combine results
    results = {
        'patterns': patterns,
        'efficiency': efficiency,
    }

    # Save to file
    output_file = ctx.get_output_file("temporal_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    ctx.logger.info(f"Saved temporal patterns to {output_file}")
    ctx.report_progress("Temporal pattern analysis complete", 0.4)


def analyze_event_segmentation(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze event segmentation in temporal data.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options (event_steps)
    """
    ctx.logger.info("Analyzing event segmentation...")

    event_steps = kwargs.get('event_steps', [])

    from pathlib import Path
    segments_data = process_event_segmentation_data(Path(experiment_path), event_steps=event_steps)

    # Compute segmentation metrics
    segmentation = compute_event_segmentation_metrics(segments_data, event_steps)

    # Save to file
    output_file = ctx.get_output_file("event_segmentation.json")
    with open(output_file, 'w') as f:
        json.dump(segmentation, f, indent=2, default=str)

    ctx.logger.info(f"Saved event segmentation to {output_file}")
    ctx.report_progress("Event segmentation analysis complete", 0.6)


def analyze_time_series_overview(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze overall time series characteristics.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options (time_period_size)
    """
    ctx.logger.info("Analyzing time series overview...")

    time_period_size = kwargs.get('time_period_size', 100)

    from pathlib import Path
    time_series_df = process_time_series_data(Path(experiment_path), time_period_size=time_period_size)

    if time_series_df.empty:
        ctx.logger.warning("No time series data found")
        return

    # Compute temporal statistics
    stats = compute_temporal_statistics(time_series_df)

    # Save to file
    output_file = ctx.get_output_file("time_series_overview.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    ctx.logger.info(f"Saved time series overview to {output_file}")
    ctx.report_progress("Time series overview analysis complete", 0.8)


def analyze_temporal_efficiency(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze temporal efficiency metrics.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing temporal efficiency...")

    from pathlib import Path
    time_series_df = process_time_series_data(Path(experiment_path))

    if time_series_df.empty:
        ctx.logger.warning("No time series data for efficiency analysis")
        return

    # Compute efficiency metrics
    efficiency = compute_temporal_efficiency_metrics(time_series_df)

    # Save to file
    output_file = ctx.get_output_file("temporal_efficiency.json")
    with open(output_file, 'w') as f:
        json.dump(efficiency, f, indent=2, default=str)

    ctx.logger.info(f"Saved temporal efficiency to {output_file}")
    ctx.report_progress("Temporal efficiency analysis complete", 1.0)
