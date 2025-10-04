"""
Agent visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from farm.analysis.common.context import AnalysisContext


def plot_lifespan_distributions(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot agent lifespan distributions.

    Args:
        df: Agent data with lifespan columns
        ctx: Analysis context
        **kwargs: Plot options (figsize, dpi, etc.)
    """
    ctx.logger.info("Creating lifespan distributions plot...")

    if df.empty or 'lifespan' not in df.columns:
        ctx.logger.warning("No lifespan data available")
        return

    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    lifespans = df['lifespan'].dropna()

    # Histogram
    ax1.hist(lifespans, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Lifespan')
    ax1.set_ylabel('Number of Agents')
    ax1.set_title('Agent Lifespan Distribution')
    ax1.grid(True, alpha=0.3)

    # Box plot by agent type
    if 'agent_type' in df.columns:
        agent_types = df['agent_type'].unique()
        type_lifespans = [df[df['agent_type'] == t]['lifespan'].dropna() for t in agent_types]
        ax2.boxplot(type_lifespans, labels=agent_types)
        ax2.set_ylabel('Lifespan')
        ax2.set_title('Lifespan by Agent Type')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.hist(lifespans, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Lifespan')
        ax2.set_ylabel('Number of Agents')
        ax2.set_title('Lifespan Distribution (All Agents)')

    plt.tight_layout()

    # Save figure
    output_file = ctx.get_output_file("lifespan_distributions.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_behavior_clusters(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot agent behavior clusters.

    Args:
        df: Agent data with behavior metrics
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating behavior clusters plot...")

    # For now, create a simple scatter plot of behavior metrics
    # In a full implementation, this would use clustering algorithms

    if df.empty or not all(col in df.columns for col in ['successful_actions', 'total_rewards']):
        ctx.logger.warning("Insufficient behavior data for clustering")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Simple scatter plot of success rate vs rewards
    df_copy = df.copy()
    df_copy['success_rate'] = df_copy['successful_actions'] / df_copy['total_actions'].replace(0, 1)

    scatter = ax.scatter(df_copy['success_rate'], df_copy['total_rewards'],
                        alpha=0.6, s=50, c=df_copy['lifespan'], cmap='viridis')

    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Total Rewards')
    ax.set_title('Agent Behavior Patterns\n(Color: Lifespan)')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lifespan')

    output_file = ctx.get_output_file("behavior_clusters.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved cluster plot to {output_file}")


def plot_performance_metrics(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot agent performance metrics.

    Args:
        df: Agent data with performance metrics
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating performance metrics plot...")

    if df.empty:
        ctx.logger.warning("No performance data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot rewards vs lifespan
    if 'total_rewards' in df.columns and 'lifespan' in df.columns:
        scatter = ax.scatter(df['lifespan'], df['total_rewards'],
                           alpha=0.6, s=50, c=df.get('successful_actions', df['total_rewards']),
                           cmap='plasma')

        ax.set_xlabel('Lifespan')
        ax.set_ylabel('Total Rewards')
        ax.set_title('Agent Performance: Rewards vs Lifespan')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Successful Actions')

    output_file = ctx.get_output_file("performance_metrics.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_learning_curves(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot agent learning curves.

    Args:
        df: Agent data with learning metrics
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating learning curves plot...")

    if df.empty or not all(col in df.columns for col in ['total_rewards', 'lifespan']):
        ctx.logger.warning("No learning data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot learning efficiency
    df_copy = df.copy()
    df_copy['learning_efficiency'] = df_copy['total_rewards'] / df_copy['lifespan'].replace(0, 1)

    # Sort by lifespan for curve appearance
    df_sorted = df_copy.sort_values('lifespan')

    ax.plot(df_sorted['lifespan'], df_sorted['learning_efficiency'],
            linewidth=2, marker='o', markersize=4, alpha=0.7)

    ax.set_xlabel('Lifespan')
    ax.set_ylabel('Learning Efficiency (Rewards/Lifespan)')
    ax.set_title('Agent Learning Curves')
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("learning_curves.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")
