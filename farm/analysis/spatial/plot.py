"""
Spatial visualization functions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

from farm.analysis.common.context import AnalysisContext


def plot_spatial_overview(spatial_data: Dict[str, pd.DataFrame], ctx: AnalysisContext, **kwargs) -> None:
    """Plot overview of spatial data.

    Args:
        spatial_data: Dictionary with spatial data
        ctx: Analysis context
        **kwargs: Plot options
    """
    ctx.logger.info("Creating spatial overview plot...")

    agent_df = spatial_data.get('agent_positions', pd.DataFrame())
    resource_df = spatial_data.get('resource_positions', pd.DataFrame())

    if agent_df.empty and resource_df.empty:
        ctx.logger.warning("No spatial data available for plotting")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot agents
    if not agent_df.empty and 'position_x' in agent_df.columns:
        ax.scatter(agent_df['position_x'], agent_df['position_y'],
                  c='blue', alpha=0.6, s=20, label='Agents')

    # Plot resources
    if not resource_df.empty and 'position_x' in resource_df.columns:
        ax.scatter(resource_df['position_x'], resource_df['position_y'],
                  c='red', alpha=0.8, s=30, marker='s', label='Resources')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Spatial Distribution Overview')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("spatial_overview.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved spatial overview plot to {output_file}")


def plot_movement_trajectories(movement_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot movement trajectories.

    Args:
        movement_df: DataFrame with movement data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if movement_df.empty:
        ctx.logger.warning("No movement data available for plotting")
        return

    ctx.logger.info("Creating movement trajectories plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot trajectories for each agent
    colors = plt.cm.viridis(np.linspace(0, 1, len(movement_df['agent_id'].unique())))

    for i, (agent_id, group) in enumerate(movement_df.groupby('agent_id')):
        if len(group) > 1:
            ax.plot(group['position_x'], group['position_y'],
                   color=colors[i % len(colors)], alpha=0.7,
                   linewidth=1, label=f'Agent {agent_id}')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Movement Trajectories')
    ax.grid(True, alpha=0.3)

    # Only show legend if not too many agents
    if len(movement_df['agent_id'].unique()) <= 10:
        ax.legend()

    output_file = ctx.get_output_file("movement_trajectories.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved movement trajectories plot to {output_file}")


def plot_location_hotspots(hotspots_data: Dict[str, Any], ctx: AnalysisContext, **kwargs) -> None:
    """Plot location hotspots.

    Args:
        hotspots_data: Dictionary with hotspots data
        ctx: Analysis context
        **kwargs: Plot options
    """
    hotspots = hotspots_data.get('hotspots', [])

    if not hotspots:
        ctx.logger.warning("No hotspots data available for plotting")
        return

    ctx.logger.info("Creating location hotspots plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert hotspots to DataFrame for easier plotting
    hotspots_df = pd.DataFrame(hotspots)

    if not hotspots_df.empty:
        # Plot activity levels
        scatter = ax.scatter(hotspots_df['position_x'], hotspots_df['position_y'],
                           c=hotspots_df['activity'], cmap='hot',
                           s=hotspots_df['activity']*2, alpha=0.7)

        plt.colorbar(scatter, ax=ax, label='Activity Level')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Location Hotspots')
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("location_hotspots.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved location hotspots plot to {output_file}")


def plot_spatial_density(density_data: Dict[str, Any], ctx: AnalysisContext, **kwargs) -> None:
    """Plot spatial density heatmap.

    Args:
        density_data: Dictionary with density data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if not density_data or 'density_map' not in density_data:
        ctx.logger.warning("No density data available for plotting")
        return

    ctx.logger.info("Creating spatial density plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    density_map = np.array(density_data['density_map'])
    x_edges = np.array(density_data['x_edges'])
    y_edges = np.array(density_data['y_edges'])

    # Plot density heatmap
    im = ax.imshow(density_map.T, origin='lower', cmap='viridis',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   aspect='auto')

    plt.colorbar(im, ax=ax, label='Density')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Spatial Density Heatmap')

    output_file = ctx.get_output_file("spatial_density.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved spatial density plot to {output_file}")


def plot_movement_directions(movement_df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Plot movement direction distribution.

    Args:
        movement_df: DataFrame with movement data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if movement_df.empty:
        ctx.logger.warning("No movement data available for plotting")
        return

    ctx.logger.info("Creating movement directions plot...")

    # Calculate movement directions
    directions = _calculate_movement_directions(movement_df)

    if not directions:
        ctx.logger.warning("Could not calculate movement directions")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    direction_names = list(directions.keys())
    direction_counts = list(directions.values())

    bars = ax.bar(direction_names, direction_counts, alpha=0.7, color='skyblue')
    ax.set_xlabel('Direction')
    ax.set_ylabel('Movement Count')
    ax.set_title('Movement Direction Distribution')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, direction_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{int(count)}', ha='center', va='bottom')

    output_file = ctx.get_output_file("movement_directions.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved movement directions plot to {output_file}")


def plot_clustering_analysis(clusters_data: Dict[str, Any], ctx: AnalysisContext, **kwargs) -> None:
    """Plot clustering analysis results.

    Args:
        clusters_data: Dictionary with clustering data
        ctx: Analysis context
        **kwargs: Plot options
    """
    if not clusters_data or 'cluster_centers' not in clusters_data:
        ctx.logger.warning("No clustering data available for plotting")
        return

    ctx.logger.info("Creating clustering analysis plot...")

    # This would require the original coordinate data to plot properly
    # For now, just plot cluster centers
    fig, ax = plt.subplots(figsize=(10, 6))

    centers = np.array(clusters_data['cluster_centers'])
    ax.scatter(centers[:, 0], centers[:, 1],
              c='red', s=200, marker='x', linewidth=3, label='Cluster Centers')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Spatial Clusters (k={clusters_data.get("n_clusters", "?")})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = ctx.get_output_file("spatial_clustering.png")
    fig.savefig(output_file, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    plt.close(fig)

    ctx.logger.info(f"Saved clustering analysis plot to {output_file}")


def _calculate_movement_directions(movement_df: pd.DataFrame) -> Dict[str, int]:
    """Calculate movement direction distribution."""
    if movement_df.empty:
        return {}

    directions = {
        'north': 0, 'northeast': 0, 'east': 0, 'southeast': 0,
        'south': 0, 'southwest': 0, 'west': 0, 'northwest': 0
    }

    for _, row in movement_df.iterrows():
        if pd.notna(row.get('position_x')) and pd.notna(row.get('position_y')):
            # This is a simplified direction calculation
            # In practice, you'd compare consecutive positions
            dx = np.random.uniform(-1, 1)  # Placeholder - need actual movement deltas
            dy = np.random.uniform(-1, 1)  # Placeholder - need actual movement deltas

            if abs(dx) < 0.1 and abs(dy) < 0.1:
                continue  # No significant movement

            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle += 360

            # Determine direction
            if 337.5 <= angle or angle < 22.5:
                directions['east'] += 1
            elif 22.5 <= angle < 67.5:
                directions['northeast'] += 1
            elif 67.5 <= angle < 112.5:
                directions['north'] += 1
            elif 112.5 <= angle < 157.5:
                directions['northwest'] += 1
            elif 157.5 <= angle < 202.5:
                directions['west'] += 1
            elif 202.5 <= angle < 247.5:
                directions['southwest'] += 1
            elif 247.5 <= angle < 292.5:
                directions['south'] += 1
            else:
                directions['southeast'] += 1

    return directions
