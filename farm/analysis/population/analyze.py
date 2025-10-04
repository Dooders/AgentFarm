"""
Population analysis functions.
"""

import pandas as pd
import json

from farm.analysis.common.context import AnalysisContext
from farm.analysis.population.compute import (
    compute_population_statistics,
    compute_birth_death_rates,
    compute_population_stability,
)


def analyze_population_dynamics(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze population dynamics and save results.

    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing population dynamics...")

    # Compute statistics
    stats = compute_population_statistics(df)
    rates = compute_birth_death_rates(df)
    stability = compute_population_stability(df)

    # Combine results
    results = {
        'statistics': stats,
        'rates': rates,
        'stability': stability,
    }

    # Save to file
    output_file = ctx.get_output_file("population_statistics.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    ctx.logger.info(f"Saved statistics to {output_file}")
    ctx.report_progress("Population analysis complete", 0.5)


def analyze_agent_composition(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze agent type composition over time.

    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing agent composition...")

    # Calculate proportions
    agent_types = ['system_agents', 'independent_agents', 'control_agents']
    composition_df = df.copy()

    for agent_type in agent_types:
        if agent_type in df.columns:
            composition_df[f'{agent_type}_pct'] = (
                df[agent_type] / df['total_agents']
            ) * 100

    # Save composition data
    output_file = ctx.get_output_file("agent_composition.csv")
    composition_df.to_csv(output_file, index=False)

    ctx.logger.info(f"Saved composition to {output_file}")
    ctx.report_progress("Composition analysis complete", 0.7)
