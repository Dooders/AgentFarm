"""
Combat analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.combat.compute import (
    compute_combat_statistics,
    compute_agent_combat_performance,
    compute_combat_efficiency_metrics,
    compute_combat_temporal_patterns,
)
from farm.analysis.combat.data import (
    process_combat_data,
    process_combat_metrics_data,
    process_agent_combat_stats,
)


def analyze_combat_overview(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze overall combat patterns and save results.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing combat overview...")

    from pathlib import Path
    combat_df = process_combat_data(Path(experiment_path))
    metrics_df = process_combat_metrics_data(Path(experiment_path))

    # Compute combat statistics
    stats = compute_combat_statistics(combat_df, metrics_df)

    # Save to file
    output_file = ctx.get_output_file("combat_overview.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    ctx.logger.info(f"Saved combat overview to {output_file}")
    ctx.report_progress("Combat overview analysis complete", 0.3)


def analyze_agent_combat_performance(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze agent-specific combat performance.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options (agent_ids)
    """
    ctx.logger.info("Analyzing agent combat performance...")

    agent_ids = kwargs.get('agent_ids')

    from pathlib import Path
    agent_combat_df = process_agent_combat_stats(Path(experiment_path), agent_ids=agent_ids)

    if agent_combat_df.empty:
        ctx.logger.warning("No agent combat data found")
        return

    # Compute agent performance
    performance = compute_agent_combat_performance(agent_combat_df)

    # Save to file
    output_file = ctx.get_output_file("agent_combat_performance.json")
    with open(output_file, 'w') as f:
        json.dump(performance, f, indent=2, default=str)

    ctx.logger.info(f"Saved agent combat performance to {output_file}")
    ctx.report_progress("Agent combat performance analysis complete", 0.5)


def analyze_combat_efficiency(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze combat efficiency metrics.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing combat efficiency...")

    from pathlib import Path
    combat_df = process_combat_data(Path(experiment_path))

    if combat_df.empty:
        ctx.logger.warning("No combat data for efficiency analysis")
        return

    # Compute efficiency metrics
    efficiency = compute_combat_efficiency_metrics(combat_df)

    # Save to file
    output_file = ctx.get_output_file("combat_efficiency.json")
    with open(output_file, 'w') as f:
        json.dump(efficiency, f, indent=2, default=str)

    ctx.logger.info(f"Saved combat efficiency to {output_file}")
    ctx.report_progress("Combat efficiency analysis complete", 0.7)


def analyze_combat_temporal_patterns(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze temporal patterns in combat behavior.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing combat temporal patterns...")

    from pathlib import Path
    combat_df = process_combat_data(Path(experiment_path))
    metrics_df = process_combat_metrics_data(Path(experiment_path))

    # Compute temporal patterns
    patterns = compute_combat_temporal_patterns(combat_df, metrics_df)

    # Save to file
    output_file = ctx.get_output_file("combat_temporal_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(patterns, f, indent=2, default=str)

    ctx.logger.info(f"Saved combat temporal patterns to {output_file}")
    ctx.report_progress("Combat temporal patterns analysis complete", 1.0)
