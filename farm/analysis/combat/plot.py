"""
Combat visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

from farm.analysis.common.context import AnalysisContext


def plot_combat_overview(metrics_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot overview of combat metrics over time.

    Args:
        metrics_df: Combat metrics data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if metrics_df.empty:
        ctx.logger.warning("No combat metrics data for plotting")
        return

    ctx.logger.info("Creating combat overview plot...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot combat encounters over time
    ax1.plot(metrics_df['step'], metrics_df['combat_encounters'],
            label='Combat Encounters', linewidth=2, color='red')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Combat Encounters')
    ax1.set_title('Combat Encounters Over Time')
    ax1.grid(True, alpha=0.3)

    # Plot successful attacks over time
    ax2.plot(metrics_df['step'], metrics_df['successful_attacks'],
            label='Successful Attacks', linewidth=2, color='orange')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Successful Attacks')
    ax2.set_title('Successful Attacks Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = ctx.get_output_file("combat_overview.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved combat overview plot to {output_file}")


def plot_combat_success_rate(metrics_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot combat success rate over time.

    Args:
        metrics_df: Combat metrics data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if metrics_df.empty:
        ctx.logger.warning("No combat metrics data for success rate plotting")
        return

    ctx.logger.info("Creating combat success rate plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate success rate over time
    success_rate = metrics_df['successful_attacks'] / metrics_df['combat_encounters'].replace(0, 1)
    success_rate = success_rate.replace([np.inf, -np.inf], 0).fillna(0)

    ax.plot(metrics_df['step'], success_rate * 100,
           linewidth=2, color='green', label='Success Rate')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Combat Success Rate Over Time')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("combat_success_rate.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved combat success rate plot to {output_file}")


def plot_agent_combat_performance(agent_combat_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot agent combat performance comparison.

    Args:
        agent_combat_df: Agent combat statistics
        ctx: Analysis context
        **kwargs: Plot options
    """
    if agent_combat_df.empty:
        ctx.logger.warning("No agent combat data for performance plotting")
        return

    ctx.logger.info("Creating agent combat performance plot...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by total damage for consistent ordering
    sorted_df = agent_combat_df.sort_values('total_damage', ascending=True)

    # Plot total damage by agent
    ax1.barh(range(len(sorted_df)), sorted_df['total_damage'], color='red', alpha=0.7)
    ax1.set_yticks(range(len(sorted_df)))
    ax1.set_yticklabels([f'Agent {int(id)}' for id in sorted_df['agent_id']])
    ax1.set_xlabel('Total Damage Dealt')
    ax1.set_title('Total Damage by Agent')

    # Plot success rate by agent
    ax2.barh(range(len(sorted_df)), sorted_df['success_rate'] * 100, color='green', alpha=0.7)
    ax2.set_yticks(range(len(sorted_df)))
    ax2.set_yticklabels([f'Agent {int(id)}' for id in sorted_df['agent_id']])
    ax2.set_xlabel('Success Rate (%)')
    ax2.set_title('Success Rate by Agent')

    # Plot total attacks by agent
    ax3.barh(range(len(sorted_df)), sorted_df['total_attacks'], color='blue', alpha=0.7)
    ax3.set_yticks(range(len(sorted_df)))
    ax3.set_yticklabels([f'Agent {int(id)}' for id in sorted_df['agent_id']])
    ax3.set_xlabel('Total Attacks')
    ax3.set_title('Total Attacks by Agent')

    # Plot average damage by agent
    ax4.barh(range(len(sorted_df)), sorted_df['avg_damage'], color='orange', alpha=0.7)
    ax4.set_yticks(range(len(sorted_df)))
    ax4.set_yticklabels([f'Agent {int(id)}' for id in sorted_df['agent_id']])
    ax4.set_xlabel('Average Damage')
    ax4.set_title('Average Damage by Agent')

    plt.tight_layout()

    output_file = ctx.get_output_file("agent_combat_performance.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved agent combat performance plot to {output_file}")


def plot_combat_efficiency(efficiency_data: Dict[str, float], ctx: AnalysisContext, **kwargs) -> None:
    """Plot combat efficiency metrics.

    Args:
        efficiency_data: Dictionary with efficiency metrics
        ctx: Analysis context
        **kwargs: Plot options
    """
    if not efficiency_data:
        ctx.logger.warning("No efficiency data for plotting")
        return

    ctx.logger.info("Creating combat efficiency plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = list(efficiency_data.keys())
    values = list(efficiency_data.values())

    bars = ax.bar(metrics, values, alpha=0.7, color='purple')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
               f'{value:.3f}', ha='center', va='bottom')

    ax.set_ylabel('Efficiency Score')
    ax.set_title('Combat Efficiency Metrics')
    ax.set_ylim(0, max(1.1, max(values) * 1.1))
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if needed
    ax.tick_params(axis='x', rotation=45)

    output_file = ctx.get_output_file("combat_efficiency.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved combat efficiency plot to {output_file}")


def plot_damage_distribution(combat_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot distribution of damage dealt in combat.

    Args:
        combat_df: Combat action data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if combat_df.empty or 'damage_dealt' not in combat_df.columns:
        ctx.logger.warning("No damage data for distribution plotting")
        return

    ctx.logger.info("Creating damage distribution plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of damage dealt
    ax1.hist(combat_df['damage_dealt'], bins=30, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(combat_df['damage_dealt'].mean(), color='blue', linestyle='--',
               label=f'Mean: {combat_df["damage_dealt"].mean():.2f}')
    ax1.set_xlabel('Damage Dealt')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Damage Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot of damage by success
    successful = combat_df[combat_df['damage_dealt'] > 0]['damage_dealt']
    failed = combat_df[combat_df['damage_dealt'] == 0]['damage_dealt']

    ax2.boxplot([successful, failed], labels=['Successful', 'Failed'])
    ax2.set_ylabel('Damage Dealt')
    ax2.set_title('Damage by Attack Success')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = ctx.get_output_file("damage_distribution.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved damage distribution plot to {output_file}")


def plot_combat_temporal_patterns(patterns_data: Dict[str, Any], ctx: AnalysisContext, **kwargs) -> None:
    """Plot temporal patterns in combat behavior.

    Args:
        patterns_data: Dictionary with temporal patterns
        ctx: Analysis context
        **kwargs: Plot options
    """
    if not patterns_data:
        ctx.logger.warning("No temporal patterns data for plotting")
        return

    ctx.logger.info("Creating combat temporal patterns plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot trends as text annotations since we don't have the raw time series
    trend_info = []
    if 'frequency_trend' in patterns_data:
        trend_info.append(f"Frequency Trend: {patterns_data['frequency_trend']:.3f}")
    if 'success_rate_trend' in patterns_data:
        trend_info.append(f"Success Rate Trend: {patterns_data['success_rate_trend']:.3f}")
    if 'damage_trend' in patterns_data:
        trend_info.append(f"Damage Trend: {patterns_data['damage_trend']:.3f}")

    # Create a simple summary plot
    ax.text(0.5, 0.5, '\n'.join(trend_info),
           transform=ax.transAxes, ha='center', va='center',
           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    ax.set_title('Combat Temporal Patterns Summary')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    output_file = ctx.get_output_file("combat_temporal_patterns.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved combat temporal patterns plot to {output_file}")
