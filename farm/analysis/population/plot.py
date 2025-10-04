"""
Population visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from farm.analysis.common.context import AnalysisContext


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
