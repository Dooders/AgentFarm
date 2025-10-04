"""
Spatial statistical computations.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from farm.analysis.common.utils import calculate_statistics
from farm.analysis.config import spatial_config


def compute_spatial_statistics(spatial_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compute comprehensive spatial statistics.

    Args:
        spatial_data: Dictionary with spatial data (agent_positions, resource_positions)

    Returns:
        Dictionary of computed spatial statistics
    """
    agent_df = spatial_data.get('agent_positions', pd.DataFrame())
    resource_df = spatial_data.get('resource_positions', pd.DataFrame())

    stats = {}

    # Agent spatial statistics
    if not agent_df.empty:
        stats['agent_spatial'] = _compute_agent_spatial_stats(agent_df)

    # Resource spatial statistics
    if not resource_df.empty:
        stats['resource_spatial'] = _compute_resource_spatial_stats(resource_df)

    # Agent-resource interaction statistics
    if not agent_df.empty and not resource_df.empty:
        stats['interaction_spatial'] = _compute_interaction_spatial_stats(agent_df, resource_df)

    return stats


def compute_movement_patterns(movement_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute movement pattern statistics.

    Args:
        movement_df: DataFrame with movement trajectories

    Returns:
        Dictionary of movement pattern metrics
    """
    if movement_df.empty:
        return {
            'total_movements': 0,
            'avg_distance': 0.0,
            'total_distance': 0.0,
            'movement_frequency': 0.0,
        }

    # Basic movement statistics
    total_movements = len(movement_df)
    total_distance = movement_df['distance'].sum()
    avg_distance = movement_df['distance'].mean()

    # Movement frequency per agent per step
    movement_freq = movement_df.groupby('agent_id')['distance'].count().mean()

    # Movement paths analysis
    paths = _analyze_movement_paths(movement_df)

    return {
        'total_movements': total_movements,
        'avg_distance': float(avg_distance),
        'total_distance': float(total_distance),
        'movement_frequency': float(movement_freq),
        'paths': paths,
    }


def compute_location_hotspots(location_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compute location hotspots and clustering analysis.

    Args:
        location_data: Dictionary with location activity data

    Returns:
        Dictionary with hotspot and clustering analysis
    """
    activity_df = location_data.get('location_activity', pd.DataFrame())

    if activity_df.empty:
        return {'hotspots': [], 'clusters': {}}

    # Identify hotspots based on activity density
    hotspots = _identify_hotspots(activity_df)

    # Clustering analysis
    clusters = _compute_location_clusters(activity_df)

    return {
        'hotspots': hotspots,
        'clusters': clusters,
    }


def compute_spatial_distribution_metrics(positions_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute spatial distribution metrics.

    Args:
        positions_df: DataFrame with position data

    Returns:
        Dictionary with distribution metrics
    """
    if positions_df.empty or 'position_x' not in positions_df.columns:
        return {}

    # Extract coordinates
    coords = positions_df[['position_x', 'position_y']].dropna().values

    if len(coords) < 2:
        return {}

    # Centroid
    centroid = np.mean(coords, axis=0)

    # Spread metrics
    distances_from_center = np.sqrt(np.sum((coords - centroid)**2, axis=1))
    spread_stats = calculate_statistics(distances_from_center)

    # Density estimation
    density = _estimate_spatial_density(coords)

    # Clustering coefficient
    clustering = _compute_clustering_coefficient(coords)

    return {
        'centroid': centroid.tolist(),
        'spread': spread_stats,
        'density': density,
        'clustering_coefficient': clustering,
    }


def _compute_agent_spatial_stats(agent_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute spatial statistics for agents."""
    if 'position_x' not in agent_df.columns:
        return {}

    coords = agent_df[['position_x', 'position_y']].dropna()

    return {
        'total_agents': len(agent_df['agent_id'].unique()) if 'agent_id' in agent_df.columns else 0,
        'spatial_distribution': compute_spatial_distribution_metrics(agent_df),
        'position_stats': {
            'x': calculate_statistics(coords['position_x']) if not coords.empty else {},
            'y': calculate_statistics(coords['position_y']) if not coords.empty else {},
        },
    }


def _compute_resource_spatial_stats(resource_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute spatial statistics for resources."""
    if 'position_x' not in resource_df.columns:
        return {}

    coords = resource_df[['position_x', 'position_y']].dropna()

    return {
        'total_resources': len(resource_df),
        'spatial_distribution': compute_spatial_distribution_metrics(resource_df),
        'position_stats': {
            'x': calculate_statistics(coords['position_x']) if not coords.empty else {},
            'y': calculate_statistics(coords['position_y']) if not coords.empty else {},
        },
    }


def _compute_interaction_spatial_stats(agent_df: pd.DataFrame, resource_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute spatial statistics for agent-resource interactions."""
    if agent_df.empty or resource_df.empty:
        return {}

    # Calculate distances between agents and resources
    agent_coords = agent_df[['position_x', 'position_y']].dropna().values
    resource_coords = resource_df[['position_x', 'position_y']].dropna().values

    if len(agent_coords) == 0 or len(resource_coords) == 0:
        return {}

    # Compute all pairwise distances
    distances = []
    for agent_pos in agent_coords:
        for resource_pos in resource_coords:
            dist = np.sqrt(np.sum((agent_pos - resource_pos)**2))
            distances.append(dist)

    distance_stats = calculate_statistics(np.array(distances))

    return {
        'avg_distance_to_resources': float(distance_stats['mean']),
        'distance_stats': distance_stats,
    }


def _analyze_movement_paths(movement_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze movement paths and patterns."""
    if movement_df.empty:
        return {}

    # Group by agent and analyze individual paths
    path_stats = []
    for agent_id, group in movement_df.groupby('agent_id'):
        if len(group) > 1:
            path_length = group['distance'].sum()
            straightness = _compute_path_straightness(group)
            path_stats.append({
                'agent_id': agent_id,
                'length': float(path_length),
                'straightness': straightness,
                'steps': len(group),
            })

    return {
        'individual_paths': path_stats,
        'avg_path_length': np.mean([p['length'] for p in path_stats]) if path_stats else 0.0,
        'avg_straightness': np.mean([p['straightness'] for p in path_stats]) if path_stats else 0.0,
    }


def _compute_path_straightness(path_df: pd.DataFrame) -> float:
    """Compute straightness ratio of a path (actual distance / straight-line distance)."""
    if len(path_df) < 2:
        return 1.0

    # Total path length
    total_distance = path_df['distance'].sum()

    # Straight-line distance from start to end
    start_pos = path_df.iloc[0][['position_x', 'position_y']].values
    end_pos = path_df.iloc[-1][['position_x', 'position_y']].values

    straight_distance = np.sqrt(np.sum((end_pos - start_pos)**2))

    return straight_distance / total_distance if total_distance > 0 else 1.0


def _identify_hotspots(activity_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify spatial hotspots based on activity density."""
    if activity_df.empty:
        return []

    # Group by location and count activity
    location_activity = activity_df.groupby(['position_x', 'position_y']).size().reset_index(name='activity')

    # Calculate activity threshold (mean + std)
    mean_activity = location_activity['activity'].mean()
    std_activity = location_activity['activity'].std()
    threshold = mean_activity + std_activity

    # Identify hotspots
    hotspots = location_activity[location_activity['activity'] > threshold]
    hotspots = hotspots.sort_values('activity', ascending=False)

    return hotspots.to_dict('records')


def _compute_location_clusters(activity_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute location clustering using K-means."""
    min_points = spatial_config.min_clustering_points
    if activity_df.empty or len(activity_df) < min_points:
        return {}

    coords = activity_df[['position_x', 'position_y']].dropna().values

    if len(coords) < min_points:
        return {}

    # Try different numbers of clusters
    max_clusters = min(spatial_config.max_clusters, len(coords))
    best_score = -1
    best_n_clusters = 2

    for n_clusters in range(2, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)
            score = silhouette_score(coords, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
        except (ValueError, RuntimeError) as e:
            # Clustering can fail for certain configurations
            continue

    # Final clustering with best number of clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    return {
        'n_clusters': best_n_clusters,
        'silhouette_score': float(best_score),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'cluster_labels': labels.tolist(),
    }


def _estimate_spatial_density(coords: np.ndarray) -> Dict[str, Any]:
    """Estimate spatial density using kernel density estimation."""
    if len(coords) < 2:
        return {}

    try:
        # Simple density estimation using 2D histogram
        hist, xedges, yedges = np.histogram2d(
            coords[:, 0], coords[:, 1], bins=spatial_config.density_bins, density=True
        )

        return {
            'density_map': hist.tolist(),
            'x_edges': xedges.tolist(),
            'y_edges': yedges.tolist(),
            'max_density': float(np.max(hist)),
            'avg_density': float(np.mean(hist)),
        }
    except (ValueError, TypeError) as e:
        # Can fail if coordinates are invalid or insufficient
        return {}


def _compute_clustering_coefficient(coords: np.ndarray) -> float:
    """Compute a simple clustering coefficient based on spatial proximity."""
    if len(coords) < 3:
        return 0.0

    # Build distance matrix
    n = len(coords)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))

    # Count triangles (simplified clustering coefficient)
    threshold = np.percentile(distances[distances > 0], 50)  # Median distance as threshold

    triangles = 0
    possible_triangles = 0

    for i in range(n):
        neighbors_i = np.where(distances[i] < threshold)[0]
        neighbors_i = neighbors_i[neighbors_i != i]

        for j in neighbors_i:
            if j > i:
                neighbors_j = np.where(distances[j] < threshold)[0]
                neighbors_j = neighbors_j[neighbors_j != j]

                # Check if any common neighbors
                common = np.intersect1d(neighbors_i, neighbors_j)
                if len(common) > 0:
                    triangles += 1

                possible_triangles += len(neighbors_j)

    return triangles / possible_triangles if possible_triangles > 0 else 0.0
