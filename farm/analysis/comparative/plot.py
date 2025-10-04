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
    fig = plt.figure(figsize=(6 * len(performance_cols), 6))

    for i, col in enumerate(performance_cols):
        ax = plt.subplot(1, len(performance_cols), i + 1)
        if 'simulation_id' in df.columns:
            sns.boxplot(data=df, x='simulation_id', y=col, ax=ax)
        else:
            # Use matplotlib directly to avoid pandas plotting issues with mocks
            ax.bar(range(len(df)), df[col])
            ax.set_xlabel('Index')
            ax.set_ylabel(col)
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


def plot_comparative_analysis(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Create comprehensive comparative analysis plot.

    Args:
        df: Comparative data
        ctx: Analysis context
        **kwargs: Additional options
    """
    ctx.logger.info("Plotting comparative analysis...")

    if df.empty:
        ctx.logger.warning("No comparative data to plot")
        return

    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Comparative Analysis Overview', fontsize=16)

    # Plot 1: Experiment count by iteration (top-left)
    ax1 = plt.subplot(2, 2, 1)
    if 'experiment' in df.columns and 'iteration' in df.columns:
        exp_counts = df.groupby(['experiment', 'iteration']).size().unstack('experiment').fillna(0)
        # Plot each experiment as a line
        for exp in exp_counts.columns:
            ax1.plot(exp_counts.index, exp_counts[exp], label=str(exp), marker='o')
        ax1.set_title('Data Points by Experiment and Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Count')
        ax1.legend()

    # Plot 2: Performance metrics distribution (top-right)
    ax2 = plt.subplot(2, 2, 2)
    performance_cols = [col for col in df.columns if any(term in col.lower()
                        for term in ['performance', 'score', 'reward', 'fitness', 'efficiency'])]

    if performance_cols and 'experiment' in df.columns:
        # Simple bar plot of means by experiment
        experiments = df['experiment'].unique()
        for i, exp in enumerate(experiments):
            exp_data = df[df['experiment'] == exp]
            means = [exp_data[col].mean() for col in performance_cols]
            ax2.bar([i + j*0.2 for j in range(len(performance_cols))], means,
                   width=0.15, label=[f'{exp}-{col}' for col in performance_cols])
        ax2.set_title('Performance Metrics by Experiment')
        ax2.set_xticks(range(len(experiments)))
        ax2.set_xticklabels(experiments)
        ax2.legend()

    # Plot 3: Correlation heatmap (bottom-left)
    ax3 = plt.subplot(2, 2, 3)
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        # Simple correlation matrix plot
        im = ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_title('Metric Correlations')
        ax3.set_xticks(range(len(numeric_cols)))
        ax3.set_yticks(range(len(numeric_cols)))
        ax3.set_xticklabels(numeric_cols, rotation=45)
        ax3.set_yticklabels(numeric_cols)
        plt.colorbar(im, ax=ax3)

    # Plot 4: Time series trends (bottom-right)
    ax4 = plt.subplot(2, 2, 4)
    if 'iteration' in df.columns and performance_cols:
        for col in performance_cols[:3]:  # Limit to 3 metrics
            if 'experiment' in df.columns:
                for exp in df['experiment'].unique():
                    exp_data = df[df['experiment'] == exp]
                    ax4.plot(exp_data['iteration'], exp_data[col], label=f'{exp}-{col}', marker='.')
            else:
                ax4.plot(df['iteration'], df[col], label=col, marker='.')
        ax4.set_title('Performance Trends Over Time')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Value')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    output_file = ctx.get_output_file("comparative_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    ctx.logger.info(f"Saved comparative analysis plot to {output_file}")
    ctx.report_progress("Comparative analysis plotting complete", 0.4)
