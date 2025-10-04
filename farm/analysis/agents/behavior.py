"""
Agent behavior clustering functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from farm.analysis.common.context import AnalysisContext


def cluster_agent_behaviors(df: pd.DataFrame, ctx: AnalysisContext, n_clusters: int = 3, **kwargs) -> None:
    """Cluster agents based on behavior patterns.

    Args:
        df: Agent data with behavior metrics
        ctx: Analysis context
        n_clusters: Number of clusters to create
        **kwargs: Additional options
    """
    ctx.logger.info(f"Clustering agent behaviors into {n_clusters} clusters...")

    if df.empty:
        ctx.logger.warning("No behavior data available")
        return

    # Prepare features for clustering
    feature_cols = ['successful_actions', 'total_actions', 'total_rewards', 'lifespan']
    available_cols = [col for col in feature_cols if col in df.columns]

    if len(available_cols) < 2:
        ctx.logger.warning("Insufficient features for clustering")
        return

    # Prepare data
    cluster_data = df[available_cols].dropna()
    if len(cluster_data) < n_clusters:
        ctx.logger.warning("Not enough data points for clustering")
        return

    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Add cluster labels to data
    cluster_data = cluster_data.copy()
    cluster_data['cluster'] = clusters

    # Save clustering results
    results = {
        'n_clusters': n_clusters,
        'features_used': available_cols,
        'cluster_sizes': cluster_data['cluster'].value_counts().to_dict(),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
    }

    output_file = ctx.get_output_file("behavior_clusters.json")
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    ctx.logger.info(f"Saved clustering results to {output_file}")


def plot_behavior_clusters(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot behavior clusters.

    Args:
        df: Agent data with cluster labels
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating behavior cluster plot...")

    # This would create a visualization of the clusters
    # For now, create a simple placeholder plot

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'cluster' in df.columns and len(df['cluster'].dropna()) > 0:
        # Plot clusters if available
        clusters = df['cluster'].dropna()
        ax.hist(clusters, bins=len(clusters.unique()), alpha=0.7,
               color='lightcoral', edgecolor='black')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Agents')
        ax.set_title('Agent Behavior Clusters')
    else:
        # Placeholder
        ax.text(0.5, 0.5, 'Behavior Clustering\n(Not yet implemented)',
               transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("behavior_clusters.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved cluster plot to {output_file}")
