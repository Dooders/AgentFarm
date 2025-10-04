"""
Learning visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from farm.analysis.common.context import AnalysisContext


def plot_learning_curves(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot learning curves over time.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Plot options (figsize, dpi, etc.)
    """
    ctx.logger.info("Creating learning curves plot...")

    if df.empty:
        ctx.logger.warning("No learning data available for plotting")
        return

    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Group by agent and plot learning curves
    if 'agent_id' in df.columns:
        for agent_id, group in df.groupby('agent_id'):
            sorted_group = group.sort_values('step')
            if len(sorted_group) > 1:
                ax.plot(sorted_group['step'], sorted_group['reward'],
                       label=f'Agent {agent_id}', alpha=0.7, linewidth=1)
    else:
        # Plot overall learning curve
        ax.plot(df['step'], df['reward'], label='Overall', linewidth=2)

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    output_file = ctx.get_output_file("learning_curves.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved learning curves plot to {output_file}")


def plot_reward_distribution(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot reward distribution histogram.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if df.empty or 'reward' not in df.columns:
        ctx.logger.warning("No reward data available for plotting")
        return

    ctx.logger.info("Creating reward distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['reward'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(df['reward'].mean(), color='red', linestyle='--',
              label=f'Mean: {df["reward"].mean():.3f}')
    ax.axvline(df['reward'].median(), color='green', linestyle='--',
              label=f'Median: {df["reward"].median():.3f}')

    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("reward_distribution.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved reward distribution plot to {output_file}")


def plot_module_performance(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot performance comparison across learning modules.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if df.empty or 'module_type' not in df.columns:
        ctx.logger.warning("No module data available for plotting")
        return

    ctx.logger.info("Creating module performance plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    module_stats = []
    module_names = []

    for module, group in df.groupby('module_type'):
        module_names.append(str(module))
        module_stats.append({
            'mean': group['reward'].mean(),
            'std': group['reward'].std(),
            'count': len(group)
        })

    means = [s['mean'] for s in module_stats]
    stds = [s['std'] for s in module_stats]
    counts = [s['count'] for s in module_stats]

    x_pos = np.arange(len(module_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)

    ax.set_xlabel('Learning Module')
    ax.set_ylabel('Average Reward')
    ax.set_title('Performance by Learning Module')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Add sample size labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i],
               f'n={count}', ha='center', va='bottom')

    output_file = ctx.get_output_file("module_performance.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved module performance plot to {output_file}")


def plot_action_frequencies(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot action frequency distribution.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if df.empty or 'action_taken' not in df.columns:
        ctx.logger.warning("No action data available for plotting")
        return

    ctx.logger.info("Creating action frequencies plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    action_counts = df['action_taken'].value_counts()
    action_counts.plot(kind='bar', ax=ax, alpha=0.7, color='purple')

    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Frequencies')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("action_frequencies.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved action frequencies plot to {output_file}")


def plot_learning_efficiency(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot learning efficiency metrics.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if df.empty:
        ctx.logger.warning("No data available for efficiency plotting")
        return

    ctx.logger.info("Creating learning efficiency plot...")

    # Compute efficiency metrics
    from farm.analysis.learning.compute import compute_learning_efficiency_metrics
    efficiency = compute_learning_efficiency_metrics(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = list(efficiency.keys())
    values = list(efficiency.values())

    bars = ax.bar(metrics, values, alpha=0.7, color='green')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom')

    ax.set_ylabel('Efficiency Score')
    ax.set_title('Learning Efficiency Metrics')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if needed
    ax.tick_params(axis='x', rotation=45)

    output_file = ctx.get_output_file("learning_efficiency.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved learning efficiency plot to {output_file}")


def plot_reward_vs_step(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot reward progression over simulation steps.

    Args:
        df: Learning data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if df.empty or 'step' not in df.columns:
        ctx.logger.warning("No step data available for plotting")
        return

    ctx.logger.info("Creating reward vs step plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by step and plot average reward
    step_rewards = df.groupby('step')['reward'].agg(['mean', 'std']).reset_index()

    ax.plot(step_rewards['step'], step_rewards['mean'],
           linewidth=2, label='Average Reward')
    ax.fill_between(step_rewards['step'],
                   step_rewards['mean'] - step_rewards['std'],
                   step_rewards['mean'] + step_rewards['std'],
                   alpha=0.3, label='±1 Std Dev')

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward Progression Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("reward_progression.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved reward progression plot to {output_file}")
