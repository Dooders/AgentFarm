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
    compute_growth_rate_analysis,
    compute_demographic_metrics,
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


def analyze_comprehensive_population(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Perform comprehensive population analysis with all available metrics.

    This function combines all population analysis capabilities into a single
    comprehensive report including:
    - Basic population statistics
    - Birth/death rates (if available)
    - Population stability metrics
    - Growth rate analysis
    - Demographic composition metrics

    Args:
        df: Population data
        ctx: Analysis context
        **kwargs: Additional options
            - include_growth_analysis: bool (default: True)
            - include_demographic_analysis: bool (default: True)
    """
    ctx.logger.info("Performing comprehensive population analysis...")

    include_growth = kwargs.get('include_growth_analysis', True)
    include_demographics = kwargs.get('include_demographic_analysis', True)

    # Compute all available metrics
    results = {
        'statistics': compute_population_statistics(df),
        'stability': compute_population_stability(df),
    }

    # Add birth/death rates if available
    birth_death = compute_birth_death_rates(df)
    if birth_death:
        results['birth_death_rates'] = birth_death

    # Add growth rate analysis
    if include_growth:
        try:
            results['growth_analysis'] = compute_growth_rate_analysis(df)
        except Exception as e:
            ctx.logger.warning(f"Could not compute growth analysis: {e}")

    # Add demographic metrics
    if include_demographics:
        try:
            demographics = compute_demographic_metrics(df)
            if demographics:
                results['demographics'] = demographics
        except Exception as e:
            ctx.logger.warning(f"Could not compute demographic metrics: {e}")

    # Generate summary statistics
    summary = {
        'simulation_duration': len(df),
        'final_population': int(df['total_agents'].iloc[-1]),
        'peak_population': int(df['total_agents'].max()),
        'average_population': float(df['total_agents'].mean()),
        'population_change': int(df['total_agents'].iloc[-1] - df['total_agents'].iloc[0]),
        'population_change_pct': float(
            ((df['total_agents'].iloc[-1] - df['total_agents'].iloc[0]) / df['total_agents'].iloc[0]) * 100
        ) if df['total_agents'].iloc[0] > 0 else 0.0,
    }

    results['summary'] = summary

    # Save comprehensive results
    output_file = ctx.get_output_file("comprehensive_population_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    ctx.logger.info(f"Saved comprehensive analysis to {output_file}")

    # Also save a human-readable report
    report_file = ctx.get_output_file("population_report.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE POPULATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        for key, value in summary.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        f.write("\n\nSTATISTICS\n")
        f.write("-" * 80 + "\n")
        stats = results['statistics']['total']
        for key, value in stats.items():
            f.write(f"{key.title()}: {value:.2f}\n")

        f.write("\n\nSTABILITY METRICS\n")
        f.write("-" * 80 + "\n")
        for key, value in results['stability'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")

        if 'growth_analysis' in results:
            f.write("\n\nGROWTH ANALYSIS\n")
            f.write("-" * 80 + "\n")
            ga = results['growth_analysis']
            f.write(f"Average Growth Rate: {ga['average_growth_rate']:.2f}%\n")
            f.write(f"Max Growth Rate: {ga['max_growth_rate']:.2f}%\n")
            f.write(f"Min Growth Rate: {ga['min_growth_rate']:.2f}%\n")

            if ga['doubling_time']:
                f.write(f"Population Doubling Time: {ga['doubling_time']:.1f} steps\n")

            f.write(f"\nTime in Growth: {ga['time_in_growth']} steps\n")
            f.write(f"Time in Decline: {ga['time_in_decline']} steps\n")
            f.write(f"Time Stable: {ga['time_stable']} steps\n")

        if 'demographics' in results:
            f.write("\n\nDEMOGRAPHIC COMPOSITION\n")
            f.write("-" * 80 + "\n")
            demo = results['demographics']
            f.write(f"Diversity Index (mean): {demo['diversity_index']['mean']:.4f}\n")
            f.write(f"Dominance Index (mean): {demo['dominance_index']['mean']:.4f}\n")

            f.write("\nType Proportions:\n")
            for agent_type, prop in demo['type_proportions'].items():
                f.write(f"  {agent_type.replace('_', ' ').title()}: {prop*100:.1f}%\n")

    ctx.logger.info(f"Saved human-readable report to {report_file}")
    ctx.report_progress("Comprehensive analysis complete", 1.0)
