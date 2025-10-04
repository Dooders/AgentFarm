"""
Comparative analysis functions.
"""

import json
from typing import Any
import pandas as pd
from pathlib import Path

from farm.analysis.common.context import AnalysisContext
from farm.analysis.comparative.compute import (
    compute_comparison_metrics,
    compute_parameter_differences,
    compute_performance_comparison,
)


def analyze_simulation_comparison(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze simulation comparison and save results.

    Args:
        df: Comparison data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing simulation comparison...")

    # Compute comparison metrics
    metrics = compute_comparison_metrics(df)

    # Save to file
    output_file = ctx.get_output_file("comparison_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    ctx.logger.info(f"Saved comparison metrics to {output_file}")
    ctx.report_progress("Simulation comparison analysis complete", 0.4)


def analyze_parameter_differences(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze parameter differences between simulations.

    Args:
        df: Parameter data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing parameter differences...")

    # Compute parameter differences
    differences = compute_parameter_differences(df)

    # Save to file
    output_file = ctx.get_output_file("parameter_differences.json")
    with open(output_file, 'w') as f:
        json.dump(differences, f, indent=2)

    ctx.logger.info(f"Saved parameter differences to {output_file}")
    ctx.report_progress("Parameter differences analysis complete", 0.4)


def analyze_performance_comparison(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analyze performance comparison between simulations.

    Args:
        df: Performance data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Analyzing performance comparison...")

    # Compute performance comparison
    performance = compute_performance_comparison(df)

    # Save to file
    output_file = ctx.get_output_file("performance_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(performance, f, indent=2)

    ctx.logger.info(f"Saved performance comparison to {output_file}")
    ctx.report_progress("Performance comparison analysis complete", 0.4)


def compare_experiments(df: pd.DataFrame) -> dict:
    """Compare experiments and return statistical results.

    Args:
        df: DataFrame with experiment data

    Returns:
        Dictionary containing comparison results
    """
    if df.empty:
        return {
            'experiment_count': 0,
            'performance_comparison': {},
            'statistical_tests': {}
        }

    # Count experiments
    experiment_count = df['experiment'].nunique() if 'experiment' in df.columns else 0

    # Performance comparison
    performance_cols = [col for col in df.columns if any(term in col.lower()
                        for term in ['performance', 'score', 'reward', 'fitness'])]

    performance_comparison = {}
    for col in performance_cols:
        if col in df.columns:
            exp_stats = df.groupby('experiment' if 'experiment' in df.columns else df.index)[col].agg(['mean', 'std', 'min', 'max'])
            performance_comparison[col] = exp_stats.to_dict('index')

    # Basic statistical tests (placeholder)
    statistical_tests = {
        't_tests': {},
        'anova': {},
        'significant_differences': []
    }

    return {
        'experiment_count': experiment_count,
        'performance_comparison': performance_comparison,
        'statistical_tests': statistical_tests
    }


def process_comparative_data(data: Any, **kwargs) -> pd.DataFrame:
    """
    Process comparative analysis data.

    Parameters
    ----------
    data : Any
        Input data - can be a DataFrame or database session
    **kwargs
        Additional processing parameters

    Returns
    -------
    pd.DataFrame
        Processed data ready for comparative analysis
    """
    if isinstance(data, pd.DataFrame):
        return data
    else:
        # For other data types, return empty DataFrame
        # In a full implementation, this would process database sessions
        return pd.DataFrame()
