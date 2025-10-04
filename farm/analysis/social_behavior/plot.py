"""
Social Behavior Visualization Functions

This module provides visualization functions for social behavior analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from farm.analysis.common.context import AnalysisContext


def plot_social_network_overview(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Create overview visualization of social networks.

    Args:
        df: Social behavior data
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating social network overview plot...")

    figsize = kwargs.get('figsize', (12, 8))
    dpi = kwargs.get('dpi', 300)

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Network density if available
    network_density = df[df['metric_type'] == 'network_density']['value']
    if not network_density.empty:
        ax1.bar(['Network Density'], [network_density.iloc[0]], color='skyblue')
        ax1.set_ylabel('Density')
        ax1.set_title('Social Network Density')
        ax1.grid(True, alpha=0.3)

    # Plot 2: Clustering ratio if available
    clustering = df[df['metric_type'] == 'clustering_ratio']['value']
    if not clustering.empty:
        ax2.bar(['Clustering Ratio'], [clustering.iloc[0]], color='lightgreen')
        ax2.set_ylabel('Ratio')
        ax2.set_title('Spatial Clustering Ratio')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Cooperation-Competition ratio
    coop_comp = df[df['metric_type'] == 'overall_coop_comp_ratio']['value']
    if not coop_comp.empty:
        ax3.bar(['Coop/Comp Ratio'], [coop_comp.iloc[0]], color='orange')
        ax3.set_ylabel('Ratio')
        ax3.set_title('Cooperation vs Competition')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Total social interactions
    interactions = df[df['metric_type'] == 'total_social_interactions']['value']
    if not interactions.empty:
        ax4.bar(['Total Interactions'], [interactions.iloc[0]], color='purple')
        ax4.set_ylabel('Count')
        ax4.set_title('Total Social Interactions')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = ctx.get_output_file("social_network_overview.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved social network overview to {output_file}")


def plot_cooperation_competition_balance(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot cooperation vs competition balance.

    Args:
        df: Social behavior data
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating cooperation-competition balance plot...")

    figsize = kwargs.get('figsize', (10, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Extract agent type cooperation-competition ratios
    coop_comp_ratios = df[df['metric_type'].str.contains('_cooperation_competition_ratio')]

    if not coop_comp_ratios.empty:
        # Parse agent types and ratios
        agent_types = []
        ratios = []

        for _, row in coop_comp_ratios.iterrows():
            metric_type = row['metric_type']
            agent_type = metric_type.split('_')[0]  # Extract agent type from metric name
            agent_types.append(agent_type.capitalize())
            ratios.append(row['value'])

        # Create bar plot
        bars = ax.bar(agent_types, ratios, color=['green' if r > 1 else 'red' for r in ratios])
        ax.set_ylabel('Cooperation/Competition Ratio')
        ax.set_title('Cooperation vs Competition by Agent Type')
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Balanced')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   '.2f', ha='center', va='bottom', fontsize=9)

    else:
        # Fallback: show overall ratio
        overall_ratio = df[df['metric_type'] == 'overall_coop_comp_ratio']['value']
        if not overall_ratio.empty:
            ax.bar(['Overall'], [overall_ratio.iloc[0]],
                  color='blue', alpha=0.7)
            ax.set_ylabel('Cooperation/Competition Ratio')
            ax.set_title('Overall Cooperation vs Competition Balance')
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Balanced')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = ctx.get_output_file("cooperation_competition_balance.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved cooperation-competition balance plot to {output_file}")


def plot_resource_sharing_patterns(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot resource sharing patterns.

    Args:
        df: Social behavior data
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating resource sharing patterns plot...")

    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract sharing metrics by agent type
    sharing_actions = df[df['metric_type'].str.contains('_actions') &
                        df['metric_type'].str.contains('sharing')]

    sharing_resources = df[df['metric_type'].str.contains('_resources')]

    # Plot 1: Sharing actions by agent type
    if not sharing_actions.empty:
        agent_types = []
        actions = []

        for _, row in sharing_actions.iterrows():
            metric_type = row['metric_type']
            # Extract agent type (assuming format: AgentType_sharing_actions)
            agent_type = metric_type.split('_')[0]
            agent_types.append(agent_type.capitalize())
            actions.append(row['value'])

        ax1.bar(agent_types, actions, color='lightblue')
        ax1.set_ylabel('Number of Sharing Actions')
        ax1.set_title('Resource Sharing Actions by Agent Type')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

    # Plot 2: Resources shared by agent type
    if not sharing_resources.empty:
        agent_types = []
        resources = []

        for _, row in sharing_resources.iterrows():
            metric_type = row['metric_type']
            # Extract agent type (assuming format: AgentType_sharing_resources)
            agent_type = metric_type.split('_')[0]
            agent_types.append(agent_type.capitalize())
            resources.append(row['value'])

        ax2.bar(agent_types, resources, color='lightgreen')
        ax2.set_ylabel('Total Resources Shared')
        ax2.set_title('Resources Shared by Agent Type')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = ctx.get_output_file("resource_sharing_patterns.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved resource sharing patterns plot to {output_file}")


def plot_spatial_clustering(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot spatial clustering patterns.

    Args:
        df: Social behavior data
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating spatial clustering plot...")

    figsize = kwargs.get('figsize', (10, 6))
    dpi = kwargs.get('dpi', 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Extract clustering ratios by agent type
    clustering_ratios = df[df['metric_type'].str.contains('_clustering_ratio')]

    if not clustering_ratios.empty:
        agent_types = []
        ratios = []

        for _, row in clustering_ratios.iterrows():
            metric_type = row['metric_type']
            # Extract agent type (assuming format: AgentType_clustering_ratio)
            agent_type = metric_type.split('_')[0]
            agent_types.append(agent_type.capitalize())
            ratios.append(row['value'])

        # Create bar plot
        bars = ax.bar(agent_types, ratios, color='purple', alpha=0.7)
        ax.set_ylabel('Clustering Ratio')
        ax.set_title('Spatial Clustering by Agent Type')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   '.2f', ha='center', va='bottom', fontsize=9)

        # Add reference line at 0.5
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7,
                  label='Moderate Clustering')
        ax.legend()

    else:
        # Fallback: show overall clustering ratio
        overall_clustering = df[df['metric_type'] == 'clustering_ratio']['value']
        if not overall_clustering.empty:
            ax.bar(['Overall'], [overall_clustering.iloc[0]],
                  color='purple', alpha=0.7)
            ax.set_ylabel('Clustering Ratio')
            ax.set_title('Overall Spatial Clustering')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = ctx.get_output_file("spatial_clustering.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved spatial clustering plot to {output_file}")
