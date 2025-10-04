"""
Comparative analysis plotting functions.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from farm.analysis.common.context import AnalysisContext


def plot_comparison_metrics(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot comparison metrics.

    Args:
        df: Comparison data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Plotting comparison metrics...")

    if df.empty:
        ctx.logger.warning("No data to plot")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Simulation Comparison Metrics')

    # Plot 1: Basic metrics comparison
    numeric_cols = df.select_dtypes(include=['number']).columns[:4]  # Limit to 4 plots
    for i, col in enumerate(numeric_cols):
        if i < 4:
            ax = axes[i // 2, i % 2]
            if 'simulation_id' in df.columns:
                sns.boxplot(data=df, x='simulation_id', y=col, ax=ax)
            else:
                df[col].plot(ax=ax, kind='bar')
            ax.set_title(f'{col} Comparison')

    plt.tight_layout()
    output_file = ctx.get_output_file("comparison_metrics.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    ctx.logger.info(f"Saved comparison metrics plot to {output_file}")
    ctx.report_progress("Comparison metrics plotting complete", 0.4)


def plot_parameter_differences(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot parameter differences.

    Args:
        df: Parameter data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Plotting parameter differences...")

    if df.empty or 'parameter_name' not in df.columns:
        ctx.logger.warning("No parameter data to plot")
        return

    # Create parameter difference plots
    unique_params = df['parameter_name'].unique()

    fig, axes = plt.subplots(len(unique_params), 1, figsize=(10, 6 * len(unique_params)))
    if len(unique_params) == 1:
        axes = [axes]

    for i, param in enumerate(unique_params):
        param_data = df[df['parameter_name'] == param]
        ax = axes[i]

        if 'simulation_id' in param_data.columns and 'parameter_value' in param_data.columns:
            sns.barplot(data=param_data, x='simulation_id', y='parameter_value', ax=ax)
            ax.set_title(f'Parameter: {param}')
        else:
            param_data.plot(ax=ax)
            ax.set_title(f'Parameter: {param}')

    plt.tight_layout()
    output_file = ctx.get_output_file("parameter_differences.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    ctx.logger.info(f"Saved parameter differences plot to {output_file}")
    ctx.report_progress("Parameter differences plotting complete", 0.4)


def plot_performance_comparison(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot performance comparison.

    Args:
        df: Performance data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Plotting performance comparison...")

    if df.empty:
        ctx.logger.warning("No performance data to plot")
        return

    # Identify performance columns
    performance_cols = [col for col in df.columns if any(term in col.lower()
                        for term in ['reward', 'performance', 'score', 'fitness'])]

    if not performance_cols:
        ctx.logger.warning("No performance columns found")
        return

    # Create performance comparison plot
    fig, axes = plt.subplots(1, len(performance_cols), figsize=(6 * len(performance_cols), 6))
    if len(performance_cols) == 1:
        axes = [axes]

    for i, col in enumerate(performance_cols):
        ax = axes[i]
        if 'simulation_id' in df.columns:
            sns.boxplot(data=df, x='simulation_id', y=col, ax=ax)
        else:
            df[col].plot(ax=ax, kind='bar')
        ax.set_title(f'Performance: {col}')

    plt.tight_layout()
    output_file = ctx.get_output_file("performance_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    ctx.logger.info(f"Saved performance comparison plot to {output_file}")
    ctx.report_progress("Performance comparison plotting complete", 0.4)


def plot_simulation_comparison(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Create comprehensive simulation comparison plot.

    Args:
        df: Simulation data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Creating comprehensive simulation comparison plot...")

    # This would create a multi-panel comparison plot
    # For now, delegate to individual plot functions
    plot_comparison_metrics(df, ctx, **kwargs)
    plot_performance_comparison(df, ctx, **kwargs)

    ctx.logger.info("Comprehensive simulation comparison plotting complete")
