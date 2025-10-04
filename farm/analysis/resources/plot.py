"""
Resource visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from farm.analysis.common.context import AnalysisContext


def plot_resource_distribution(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot resource distribution over time.

    Args:
        df: Resource data
        ctx: Analysis context
        **kwargs: Plot options (figsize, dpi, etc.)
    """
    ctx.logger.info("Creating resource distribution plot...")

    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot total resources
    ax.plot(df['step'], df['total_resources'],
            label='Total Resources', linewidth=2, color='green')

    # Plot average per cell if available
    if 'average_per_cell' in df.columns:
        ax.plot(df['step'], df['average_per_cell'],
                label='Average per Cell', linewidth=1.5, color='blue', alpha=0.7)

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Resource Amount')
    ax.set_title('Resource Distribution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    output_file = ctx.get_output_file("resource_distribution.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_consumption_over_time(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot resource consumption patterns over time.

    Args:
        df: Resource data with consumption metrics
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating consumption over time plot...")

    consumption_col = None
    if 'avg_consumption_rate' in df.columns:
        consumption_col = 'avg_consumption_rate'
    elif 'consumption_rate' in df.columns:
        consumption_col = 'consumption_rate'

    if consumption_col is None:
        ctx.logger.warning("No consumption data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step'], df[consumption_col],
            label='Consumption Rate', linewidth=2, color='red')

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Consumption Rate')
    ax.set_title('Resource Consumption Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("consumption_over_time.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_efficiency_metrics(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot resource efficiency metrics over time.

    Args:
        df: Resource data with efficiency metrics
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating efficiency metrics plot...")

    efficiency_cols = ['utilization_rate', 'distribution_efficiency', 'consumption_efficiency', 'resource_efficiency']
    available_cols = [col for col in efficiency_cols if col in df.columns]

    if not available_cols:
        ctx.logger.warning("No efficiency data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['blue', 'green', 'orange', 'red']
    for i, col in enumerate(available_cols):
        ax.plot(df['step'], df[col] * 100,  # Convert to percentage
                label=col.replace('_', ' ').title(),
                linewidth=2, color=colors[i % len(colors)])

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Resource Efficiency Metrics Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    output_file = ctx.get_output_file("efficiency_metrics.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_resource_hotspots(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot resource concentration and hotspot analysis.

    Args:
        df: Resource data
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating resource hotspots plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot resource concentration
    ax.plot(df['step'], df['total_resources'],
            label='Total Resources', linewidth=2, color='purple')

    # Add rolling maximum to show peaks (potential hotspots)
    if len(df) > 10:
        rolling_max = df['total_resources'].rolling(window=10).max()
        ax.plot(df['step'], rolling_max,
                label='Rolling Maximum (10 steps)', linewidth=1.5,
                color='orange', linestyle='--')

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Resource Amount')
    ax.set_title('Resource Concentration and Hotspots')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("resource_hotspots.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


