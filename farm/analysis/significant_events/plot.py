"""
Significant events plotting functions.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional

from farm.analysis.common.context import AnalysisContext


def plot_event_timeline(ctx: AnalysisContext, **kwargs) -> None:
    """Plot timeline of significant events.

    Args:
        ctx: Analysis context
        **kwargs: Additional parameters
    """
    # Load events from the analysis results
    events_file = ctx.get_output_file("significant_events.json")
    if not events_file.exists():
        ctx.logger.warning("No significant events data found")
        return

    import json
    with open(events_file, 'r') as f:
        data = json.load(f)

    events = data.get('events', [])
    if not events:
        ctx.logger.warning("No events to plot")
        return

    df = pd.DataFrame(events)

    if 'step' not in df.columns:
        ctx.logger.warning("No step column in events data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot events as points
    if 'severity' in df.columns:
        # Color by severity
        scatter = ax.scatter(df['step'], [1] * len(df), c=df['severity'],
                           cmap='RdYlGn_r', s=df['severity'] * 100, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Severity')
    else:
        ax.scatter(df['step'], [1] * len(df), alpha=0.7)

    # Add event type labels if available
    if 'type' in df.columns:
        for idx, row in df.iterrows():
            ax.annotate(row['type'], (row['step'], 1), xytext=(0, 10),
                       textcoords='offset points', ha='center', fontsize=8)

    ax.set_xlabel('Step')
    ax.set_title('Significant Events Timeline')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("event_timeline.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    ctx.logger.info(f"Saved event timeline plot to {output_file}")
    ctx.report_progress("Event timeline plotting complete", 0.5)

    plt.close()


def plot_event_severity_distribution(ctx: AnalysisContext, **kwargs) -> None:
    """Plot distribution of event severities.

    Args:
        ctx: Analysis context
        **kwargs: Additional parameters
    """
    # Load events from the analysis results
    events_file = ctx.get_output_file("significant_events.json")
    if not events_file.exists():
        ctx.logger.warning("No significant events data found")
        return

    import json
    with open(events_file, 'r') as f:
        data = json.load(f)

    events = data.get('events', [])
    if not events:
        ctx.logger.warning("No events to plot")
        return

    df = pd.DataFrame(events)

    if 'severity' not in df.columns:
        ctx.logger.warning("No severity column in events data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(df['severity'], bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Severity')
    ax1.set_ylabel('Count')
    ax1.set_title('Event Severity Distribution')
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(df['severity'])
    ax2.set_ylabel('Severity')
    ax2.set_title('Event Severity Box Plot')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = ctx.get_output_file("event_severity_distribution.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    ctx.logger.info(f"Saved event severity distribution plot to {output_file}")
    ctx.report_progress("Event severity distribution plotting complete", 0.5)

    plt.close()


def plot_event_impact_analysis(ctx: AnalysisContext, **kwargs) -> None:
    """Plot analysis of event impacts.

    Args:
        ctx: Analysis context
        **kwargs: Additional parameters
    """
    # Load events from the analysis results
    events_file = ctx.get_output_file("significant_events.json")
    if not events_file.exists():
        ctx.logger.warning("No significant events data found")
        return

    import json
    with open(events_file, 'r') as f:
        data = json.load(f)

    events = data.get('events', [])
    if not events:
        ctx.logger.warning("No events to plot")
        return

    df = pd.DataFrame(events)

    if 'impact_scale' not in df.columns:
        ctx.logger.warning("No impact_scale column in events data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'type' in df.columns:
        # Box plot by event type
        sns.boxplot(data=df, x='type', y='impact_scale', ax=ax)
        ax.set_xlabel('Event Type')
    else:
        # Simple distribution
        ax.hist(df['impact_scale'], bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Impact Scale')

    ax.set_ylabel('Impact Scale')
    ax.set_title('Event Impact Analysis')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)

    output_file = ctx.get_output_file("event_impact_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    ctx.logger.info(f"Saved event impact analysis plot to {output_file}")
    ctx.report_progress("Event impact analysis plotting complete", 0.5)

    plt.close()


def plot_significant_events(ctx: AnalysisContext, **kwargs) -> None:
    """Create comprehensive significant events plot.

    Args:
        ctx: Analysis context
        **kwargs: Additional parameters
    """
    # Create multiple plots for comprehensive analysis
    plot_event_timeline(ctx, **kwargs)
    plot_event_severity_distribution(ctx, **kwargs)
    plot_event_impact_analysis(ctx, **kwargs)

    ctx.logger.info("Comprehensive significant events plotting complete")
