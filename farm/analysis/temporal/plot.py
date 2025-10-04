"""
Temporal visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

from farm.analysis.common.context import AnalysisContext


def plot_temporal_patterns(patterns_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot temporal patterns over time.

    Args:
        patterns_df: DataFrame with temporal patterns
        ctx: Analysis context
        **kwargs: Plot options
    """
    if patterns_df.empty:
        ctx.logger.warning("No temporal pattern data for plotting")
        return

    ctx.logger.info("Creating temporal patterns plot...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot action counts over time
    for action_type in patterns_df['action_type'].unique():
        action_data = patterns_df[patterns_df['action_type'] == action_type]
        ax1.plot(action_data['time_period'], action_data['action_count'],
                label=action_type, marker='o', alpha=0.7)

    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Action Count')
    ax1.set_title('Action Frequency Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot average rewards over time
    for action_type in patterns_df['action_type'].unique():
        action_data = patterns_df[patterns_df['action_type'] == action_type]
        ax2.plot(action_data['time_period'], action_data['avg_reward'],
                label=action_type, marker='s', alpha=0.7)

    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Reward Progression Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = ctx.get_output_file("temporal_patterns.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved temporal patterns plot to {output_file}")


def plot_rolling_averages(patterns_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot rolling averages of temporal patterns.

    Args:
        patterns_df: DataFrame with temporal patterns
        ctx: Analysis context
        **kwargs: Plot options
    """
    if patterns_df.empty or 'rolling_avg_reward' not in patterns_df.columns:
        ctx.logger.warning("No rolling average data for plotting")
        return

    ctx.logger.info("Creating rolling averages plot...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot rolling average rewards
    for action_type in patterns_df['action_type'].unique():
        action_data = patterns_df[patterns_df['action_type'] == action_type]
        if 'rolling_avg_reward' in action_data.columns:
            ax1.plot(action_data['time_period'], action_data['rolling_avg_reward'],
                    label=action_type, linewidth=2)

    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Rolling Average Reward')
    ax1.set_title('Smoothed Reward Trends')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot rolling average action counts
    for action_type in patterns_df['action_type'].unique():
        action_data = patterns_df[patterns_df['action_type'] == action_type]
        if 'rolling_action_count' in action_data.columns:
            ax2.plot(action_data['time_period'], action_data['rolling_action_count'],
                    label=action_type, linewidth=2)

    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Rolling Average Action Count')
    ax2.set_title('Smoothed Action Frequency Trends')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = ctx.get_output_file("rolling_averages.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved rolling averages plot to {output_file}")


def plot_event_segmentation(segmentation_data: Dict[str, Any], ctx: AnalysisContext, **kwargs) -> None:
    """Plot event segmentation analysis.

    Args:
        segmentation_data: Dictionary with segmentation data
        ctx: Analysis context
        **kwargs: Plot options
    """
    segment_metrics = segmentation_data.get('segment_metrics', {})

    if not segment_metrics:
        ctx.logger.warning("No segmentation data for plotting")
        return

    ctx.logger.info("Creating event segmentation plot...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    segments = list(segment_metrics.keys())
    action_counts = [metrics['action_count'] for metrics in segment_metrics.values()]
    avg_rewards = [metrics['avg_reward'] for metrics in segment_metrics.values()]
    unique_actions = [metrics['unique_actions'] for metrics in segment_metrics.values()]
    unique_agents = [metrics['unique_agents'] for metrics in segment_metrics.values()]

    # Action counts by segment
    ax1.bar(segments, action_counts, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Segment')
    ax1.set_ylabel('Action Count')
    ax1.set_title('Actions by Segment')
    ax1.tick_params(axis='x', rotation=45)

    # Average rewards by segment
    ax2.bar(segments, avg_rewards, alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Segment')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Rewards by Segment')
    ax2.tick_params(axis='x', rotation=45)

    # Unique actions by segment
    ax3.bar(segments, unique_actions, alpha=0.7, color='orange')
    ax3.set_xlabel('Segment')
    ax3.set_ylabel('Unique Actions')
    ax3.set_title('Action Diversity by Segment')
    ax3.tick_params(axis='x', rotation=45)

    # Unique agents by segment
    ax4.bar(segments, unique_agents, alpha=0.7, color='purple')
    ax4.set_xlabel('Segment')
    ax4.set_ylabel('Unique Agents')
    ax4.set_title('Agent Diversity by Segment')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_file = ctx.get_output_file("event_segmentation.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved event segmentation plot to {output_file}")


def plot_temporal_efficiency(efficiency_data: Dict[str, float], ctx: AnalysisContext, **kwargs) -> None:
    """Plot temporal efficiency metrics.

    Args:
        efficiency_data: Dictionary with efficiency metrics
        ctx: Analysis context
        **kwargs: Plot options
    """
    if not efficiency_data:
        ctx.logger.warning("No efficiency data for plotting")
        return

    ctx.logger.info("Creating temporal efficiency plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = list(efficiency_data.keys())
    values = list(efficiency_data.values())

    bars = ax.bar(metrics, values, alpha=0.7, color='green')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom')

    ax.set_ylabel('Efficiency Score')
    ax.set_title('Temporal Efficiency Metrics')
    ax.set_ylim(0, max(1.1, max(values) * 1.1))
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if needed
    ax.tick_params(axis='x', rotation=45)

    output_file = ctx.get_output_file("temporal_efficiency.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved temporal efficiency plot to {output_file}")


def plot_action_type_evolution(patterns_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot evolution of action types over time.

    Args:
        patterns_df: DataFrame with temporal patterns
        ctx: Analysis context
        **kwargs: Plot options
    """
    if patterns_df.empty:
        ctx.logger.warning("No pattern data for evolution plotting")
        return

    ctx.logger.info("Creating action type evolution plot...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked area plot of action frequencies
    pivot_df = patterns_df.pivot(index='time_period', columns='action_type', values='action_count')
    pivot_df = pivot_df.fillna(0)

    ax.stackplot(pivot_df.index, pivot_df.T.values, labels=pivot_df.columns, alpha=0.7)

    ax.set_xlabel('Time Period')
    ax.set_ylabel('Action Count')
    ax.set_title('Action Type Evolution Over Time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("action_evolution.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved action evolution plot to {output_file}")


def plot_reward_trends(patterns_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot reward trends over time.

    Args:
        patterns_df: DataFrame with temporal patterns
        ctx: Analysis context
        **kwargs: Plot options
    """
    if patterns_df.empty:
        ctx.logger.warning("No pattern data for reward trend plotting")
        return

    ctx.logger.info("Creating reward trends plot...")

    fig, ax = plt.subplots(figsize=(14, 6))

    for action_type in patterns_df['action_type'].unique():
        action_data = patterns_df[patterns_df['action_type'] == action_type].sort_values('time_period')
        ax.plot(action_data['time_period'], action_data['avg_reward'],
               label=action_type, marker='o', linewidth=2, alpha=0.8)

    ax.set_xlabel('Time Period')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward Trends by Action Type')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add trend lines
    for action_type in patterns_df['action_type'].unique():
        action_data = patterns_df[patterns_df['action_type'] == action_type].sort_values('time_period')
        if len(action_data) > 1:
            x = action_data['time_period'].values
            y = action_data['avg_reward'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), '--', alpha=0.5, linewidth=1)

    output_file = ctx.get_output_file("reward_trends.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved reward trends plot to {output_file}")
