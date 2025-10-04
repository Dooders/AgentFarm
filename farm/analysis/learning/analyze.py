"""
Learning analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.learning.compute import (
    compute_learning_statistics,
    compute_agent_learning_curves,
    compute_learning_efficiency_metrics,
    compute_module_performance_comparison,
)
from farm.analysis.learning.data import process_learning_progress_data


def analyze_learning_performance(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze overall learning performance and save results.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing learning performance...")

    # Compute statistics
    stats = compute_learning_statistics(df)
    efficiency = compute_learning_efficiency_metrics(df)

    # Combine results
    results = {
        'statistics': stats,
        'efficiency': efficiency,
    }

    # Save to file
    output_file = ctx.get_output_file("learning_performance.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    ctx.logger.info(f"Saved performance analysis to {output_file}")
    ctx.report_progress("Learning performance analysis complete", 0.3)


def analyze_agent_learning_curves(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze learning curves for individual agents.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing agent learning curves...")

    curves = compute_agent_learning_curves(df)

    # Save curves data
    output_file = ctx.get_output_file("agent_learning_curves.json")
    with open(output_file, 'w') as f:
        json.dump(curves, f, indent=2)

    ctx.logger.info(f"Saved learning curves to {output_file}")
    ctx.report_progress("Learning curves analysis complete", 0.5)


def analyze_module_performance(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze performance across different learning modules.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing module performance comparison...")

    module_comparison = compute_module_performance_comparison(df)

    # Save comparison data
    output_file = ctx.get_output_file("module_performance_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(module_comparison, f, indent=2, default=str)

    ctx.logger.info(f"Saved module comparison to {output_file}")
    ctx.report_progress("Module performance analysis complete", 0.7)


def analyze_learning_progress(experiment_path: str, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze learning progress over time from experiment data.

    Args:
        experiment_path: Path to experiment directory
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing learning progress over time...")

    from pathlib import Path
    progress_df = process_learning_progress_data(Path(experiment_path))

    if progress_df.empty:
        ctx.logger.warning("No learning progress data found")
        return

    # Compute progress metrics
    progress_stats = {
        'total_steps': len(progress_df),
        'avg_reward': float(progress_df['reward'].mean()),
        'reward_trend': float(progress_df['reward'].diff().mean()),
        'avg_action_count': float(progress_df['action_count'].mean()),
        'avg_unique_actions': float(progress_df['unique_actions'].mean()),
    }

    # Save progress data
    output_file = ctx.get_output_file("learning_progress.json")
    with open(output_file, 'w') as f:
        json.dump({
            'progress_data': progress_df.to_dict('records'),
            'statistics': progress_stats,
        }, f, indent=2, default=str)

    ctx.logger.info(f"Saved learning progress to {output_file}")
    ctx.report_progress("Learning progress analysis complete", 0.9)
