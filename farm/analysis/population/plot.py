"""
Population visualization functions.

Enhanced with comprehensive visualization capabilities including
multi-panel plots, trend analysis, and statistical overlays.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from farm.analysis.common.context import AnalysisContext
from farm.analysis.config import get_config


def plot_population_over_time(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot population dynamics over time.

    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Plot options (figsize, dpi, etc.)
    """
    ctx.logger.info("Creating population over time plot...")

    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot total population
    ax.plot(df['step'], df['total_agents'],
            label='Total Population', linewidth=2, color='black')

    # Plot by type
    colors = {'system_agents': 'blue', 'independent_agents': 'green', 'control_agents': 'red'}
    for agent_type, color in colors.items():
        if agent_type in df.columns:
            ax.plot(df['step'], df[agent_type],
                   label=agent_type.replace('_', ' ').title(),
                   linewidth=1.5, color=color, alpha=0.7)

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Population Count')
    ax.set_title('Population Dynamics Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    output_file = ctx.get_output_file("population_over_time.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_birth_death_rates(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot birth and death rates over time.

    Args:
        df: Population data with births, deaths
        ctx: Analysis context
        **kwargs: Plot options
    """
    if 'births' not in df.columns or 'deaths' not in df.columns:
        ctx.logger.warning("Births/deaths data not available, skipping plot")
        return

    ctx.logger.info("Creating birth/death rates plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['step'], df['births'], label='Births', color='green', linewidth=1.5)
    ax.plot(df['step'], df['deaths'], label='Deaths', color='red', linewidth=1.5)
    ax.fill_between(df['step'], df['births'], alpha=0.3, color='green')
    ax.fill_between(df['step'], df['deaths'], alpha=0.3, color='red')

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Count')
    ax.set_title('Birth and Death Rates Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("birth_death_rates.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_agent_composition(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot agent type composition as stacked area chart.

    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating agent composition plot...")

    agent_types = ['system_agents', 'independent_agents', 'control_agents']
    available_types = [t for t in agent_types if t in df.columns]

    if not available_types:
        ctx.logger.warning("No agent type data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create stacked area
    ax.stackplot(
        df['step'],
        *[df[t] for t in available_types],
        labels=[t.replace('_', ' ').title() for t in available_types],
        alpha=0.7
    )

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Population Count')
    ax.set_title('Agent Type Composition Over Time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("agent_composition.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_population_dashboard(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Create a comprehensive population analysis dashboard.

    Creates a multi-panel visualization showing:
    - Population trends over time
    - Growth rate analysis
    - Agent composition stacked area
    - Stability metrics

    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Plot options (figsize, dpi, etc.)
    """
    ctx.logger.info("Creating population dashboard...")

    figsize = kwargs.get('figsize', (16, 12))
    dpi = kwargs.get('dpi', get_config('global').plot_dpi)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Population trends (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['step'], df['total_agents'], linewidth=2, color='black', label='Total')

    # Add agent types if available
    colors = {'system_agents': 'blue', 'independent_agents': 'green', 'control_agents': 'red'}
    for agent_type, color in colors.items():
        if agent_type in df.columns:
            ax1.plot(df['step'], df[agent_type],
                    linewidth=1.5, color=color, alpha=0.7,
                    label=agent_type.replace('_', ' ').title())

    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Population Count')
    ax1.set_title('Population Dynamics Over Time', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Growth rate (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    growth_rate = df['total_agents'].pct_change() * 100
    smoothed_growth = growth_rate.rolling(window=10, min_periods=1).mean()

    ax2.plot(df['step'], growth_rate, alpha=0.3, color='gray', label='Instantaneous')
    ax2.plot(df['step'], smoothed_growth, linewidth=2, color='darkblue', label='Smoothed (10-step)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(df['step'], 0, smoothed_growth,
                      where=(smoothed_growth > 0), alpha=0.3, color='green', label='Growth')
    ax2.fill_between(df['step'], 0, smoothed_growth,
                      where=(smoothed_growth < 0), alpha=0.3, color='red', label='Decline')

    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Growth Rate (%)')
    ax2.set_title('Population Growth Rate', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. Agent composition stacked area (middle span)
    ax3 = fig.add_subplot(gs[1, :])
    agent_types = ['system_agents', 'independent_agents', 'control_agents']
    available_types = [t for t in agent_types if t in df.columns]

    if available_types:
        ax3.stackplot(
            df['step'],
            *[df[t] for t in available_types],
            labels=[t.replace('_', ' ').title() for t in available_types],
            alpha=0.7
        )
        ax3.set_xlabel('Simulation Step')
        ax3.set_ylabel('Population Count')
        ax3.set_title('Agent Type Composition Over Time', fontweight='bold')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

    # 4. Rolling statistics (bottom-left)
    ax4 = fig.add_subplot(gs[2, 0])
    window = 50
    rolling_mean = df['total_agents'].rolling(window=window, min_periods=1).mean()
    rolling_std = df['total_agents'].rolling(window=window, min_periods=1).std()

    ax4.plot(df['step'], df['total_agents'], alpha=0.4, color='gray', label='Actual')
    ax4.plot(df['step'], rolling_mean, linewidth=2, color='blue', label=f'{window}-step Mean')
    ax4.fill_between(df['step'],
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.2, color='blue', label='± 1 Std Dev')

    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('Population Count')
    ax4.set_title(f'Population with {window}-Step Rolling Statistics', fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # 5. Distribution and histogram (bottom-right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(df['total_agents'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax5.axvline(df['total_agents'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {df["total_agents"].mean():.0f}')
    ax5.axvline(df['total_agents'].median(), color='green', linestyle='--',
                linewidth=2, label=f'Median: {df["total_agents"].median():.0f}')

    ax5.set_xlabel('Population Count')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Population Distribution', fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3, axis='y')

    # Save figure
    output_file = ctx.get_output_file("population_dashboard.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved dashboard to {output_file}")
