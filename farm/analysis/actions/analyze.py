"""
Action analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.actions.compute import (
    compute_action_statistics,
    compute_sequence_patterns,
    compute_decision_patterns,
    compute_reward_metrics,
)


def analyze_action_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze action patterns and save results.

    Args:
        df: Action data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing action patterns...")

    # Compute statistics
    stats = compute_action_statistics(df)

    # Save to file
    output_file = ctx.get_output_file("action_statistics.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    ctx.logger.info(f"Saved statistics to {output_file}")
    ctx.report_progress("Action patterns analysis complete", 0.4)


def analyze_sequence_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze action sequence patterns.

    Args:
        df: Action data with sequence metrics
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing action sequence patterns...")

    sequences = compute_sequence_patterns(df)

    # Save sequence analysis
    output_file = ctx.get_output_file("sequence_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(sequences, f, indent=2)

    ctx.logger.info(f"Saved sequence analysis to {output_file}")
    ctx.report_progress("Sequence patterns analysis complete", 0.6)


def analyze_decision_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze decision-making patterns.

    Args:
        df: Action data with decision metrics
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing decision patterns...")

    decisions = compute_decision_patterns(df)

    # Save decision analysis
    output_file = ctx.get_output_file("decision_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(decisions, f, indent=2)

    ctx.logger.info(f"Saved decision analysis to {output_file}")
    ctx.report_progress("Decision patterns analysis complete", 0.7)


def analyze_reward_analysis(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze reward and performance metrics.

    Args:
        df: Action data with reward metrics
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing reward metrics...")

    rewards = compute_reward_metrics(df)

    # Save reward analysis
    output_file = ctx.get_output_file("reward_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(rewards, f, indent=2)

    ctx.logger.info(f"Saved reward analysis to {output_file}")
    ctx.report_progress("Reward analysis complete", 0.9)
