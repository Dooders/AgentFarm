"""
Action visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from farm.analysis.common.context import AnalysisContext


def plot_action_frequencies(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot action frequencies over time.

    Args:
        df: Action data
        ctx: Analysis context
        **kwargs: Plot options (figsize, dpi, etc.)
    """
    ctx.logger.info("Creating action frequencies plot...")

    if df.empty:
        ctx.logger.warning("No action data available")
        return

    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot frequency by action type
    action_types = df['action_type'].unique()
    for action_type in action_types:
        type_data = df[df['action_type'] == action_type]
        ax.plot(type_data['step'], type_data['frequency'],
                label=action_type, linewidth=2, marker='o', markersize=3)

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Action Frequency')
    ax.set_title('Action Frequencies Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save figure
    output_file = ctx.get_output_file("action_distribution.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_sequence_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot action sequence probabilities.

    Args:
        df: Action data with sequence columns
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating sequence patterns plot...")

    sequence_cols = [col for col in df.columns if col.startswith('seq_')]

    if not sequence_cols:
        ctx.logger.warning("No sequence data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot sequence probabilities
    for col in sequence_cols[:5]:  # Limit to first 5 sequences for readability
        sequence_name = col.replace('seq_', '').replace('_to_', '→')
        values = df[col].dropna()
        if len(values) > 0:
            ax.plot(df['step'].iloc[:len(values)], values,
                    label=sequence_name, linewidth=2)

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Transition Probability')
    ax.set_title('Action Sequence Patterns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    output_file = ctx.get_output_file("sequence_patterns.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_decision_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot decision-making patterns.

    Args:
        df: Action data with success rates
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating decision patterns plot...")

    if 'success_rate' not in df.columns:
        ctx.logger.warning("No decision data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot success rates over time
    success_by_step = df.groupby('step')['success_rate'].mean()
    ax.plot(success_by_step.index, success_by_step.values * 100,
            label='Success Rate', linewidth=2, color='green')

    # Plot action diversity
    diversity_by_step = df.groupby('step')['action_type'].nunique()
    ax2 = ax.twinx()
    ax2.plot(diversity_by_step.index, diversity_by_step.values,
             label='Action Diversity', linewidth=2, color='blue', linestyle='--')

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Success Rate (%)', color='green')
    ax2.set_ylabel('Number of Action Types', color='blue')
    ax.set_title('Decision Patterns: Success Rate and Diversity')
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("decision_patterns.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_reward_distributions(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot reward distributions and trends.

    Args:
        df: Action data with reward columns
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating reward distributions plot...")

    if 'avg_reward' not in df.columns:
        ctx.logger.warning("No reward data available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot reward trends over time
    reward_by_step = df.groupby('step')['avg_reward'].mean()
    ax1.plot(reward_by_step.index, reward_by_step.values,
             label='Average Reward', linewidth=2, color='purple')
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward Trends Over Time')
    ax1.grid(True, alpha=0.3)

    # Plot reward distribution by action type
    action_rewards = df.groupby('action_type')['avg_reward'].mean().sort_values(ascending=True)
    ax2.barh(range(len(action_rewards)), action_rewards.values)
    ax2.set_yticks(range(len(action_rewards)))
    ax2.set_yticklabels(action_rewards.index)
    ax2.set_xlabel('Average Reward')
    ax2.set_title('Rewards by Action Type')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = ctx.get_output_file("reward_distributions.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_action_distribution(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot action distribution.

    Args:
        df: Action data with frequency columns
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating action distribution plot...")

    if df.empty:
        ctx.logger.warning("No action data available")
        return

    figsize = kwargs.get('figsize', (10, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Count actions by type
    if 'frequency' in df.columns:
        action_counts = df.groupby('action_type')['frequency'].sum()
    else:
        action_counts = df['action_type'].value_counts()

    # Create bar plot
    action_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_xlabel('Action Type')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Distribution')
    ax.grid(True, alpha=0.3)

    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    output_file = ctx.get_output_file("action_distribution.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_success_rates(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot success rates by action type.

    Args:
        df: Action data with success rate columns
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating success rates plot...")

    figsize = kwargs.get('figsize', (10, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    if 'success_rate' in df.columns and 'action_type' in df.columns:
        # Group by action type and calculate mean success rate
        success_by_action = df.groupby('action_type')['success_rate'].mean()

        # Create bar plot
        success_by_action.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Action Type')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rates by Action Type')
        ax.grid(True, alpha=0.3)

        # Rotate x labels if needed
        plt.xticks(rotation=45, ha='right')
    else:
        ctx.logger.warning("No success rate data available")

    plt.tight_layout()

    output_file = ctx.get_output_file("success_rates.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")


def plot_action_sequences(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot action sequence length distribution.

    Args:
        df: Action data with sequence length columns
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating action sequences plot...")

    figsize = kwargs.get('figsize', (10, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    if 'sequence_length' in df.columns:
        # Plot histogram of sequence lengths
        ax.hist(df['sequence_length'], bins=range(1, df['sequence_length'].max() + 2),
                edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Frequency')
        ax.set_title('Action Sequence Length Distribution')
        ax.grid(True, alpha=0.3)
    else:
        ctx.logger.warning("No sequence length data available")

    plt.tight_layout()

    output_file = ctx.get_output_file("action_sequences.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved plot to {output_file}")
