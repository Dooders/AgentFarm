"""
Location analysis submodule for spatial analysis.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from farm.analysis.spatial.compute import compute_location_hotspots, compute_spatial_distribution_metrics


def analyze_location_effects(location_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze how different locations affect agent behavior.

    Args:
        location_data: Dictionary with location activity data

    Returns:
        Dictionary with location effect analysis
    """
    activity_df = location_data.get('location_activity', pd.DataFrame())

    if activity_df.empty:
        return {}

    # Group by location and analyze effects
    location_effects = {}

    for location, group in activity_df.groupby(['position_x', 'position_y']):
        effects = _analyze_single_location_effects(group)
        location_effects[f"{location[0]},{location[1]}"] = effects

    return {
        'location_effects': location_effects,
        'summary': _summarize_location_effects(location_effects),
    }


def analyze_clustering_patterns(location_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze clustering patterns in location data.

    Args:
        location_data: Dictionary with location activity data

    Returns:
        Dictionary with clustering analysis
    """
    activity_df = location_data.get('location_activity', pd.DataFrame())

    if activity_df.empty:
        return {}

    # Compute hotspots
    hotspots = compute_location_hotspots(location_data)

    # Additional clustering analysis
    clustering_metrics = _compute_detailed_clustering(activity_df)

    return {
        'hotspots': hotspots,
        'clustering_metrics': clustering_metrics,
    }


def analyze_resource_location_patterns(location_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze patterns in resource locations.

    Args:
        location_data: Dictionary with location data including resources

    Returns:
        Dictionary with resource location analysis
    """
    resource_df = location_data.get('resource_distribution', pd.DataFrame())

    if resource_df.empty:
        return {}

    # Analyze resource distribution
    distribution = compute_spatial_distribution_metrics(resource_df)

    # Analyze resource clustering
    resource_clustering = _analyze_resource_clustering(resource_df)

    return {
        'distribution': distribution,
        'clustering': resource_clustering,
    }


def _analyze_single_location_effects(location_group: pd.DataFrame) -> Dict[str, Any]:
    """Analyze effects of a single location.

    Args:
        location_group: DataFrame group for single location

    Returns:
        Dictionary with location effect metrics
    """
    if location_group.empty:
        return {}

    # Activity metrics
    activity_count = len(location_group)
    avg_duration = location_group.get('duration', pd.Series()).mean()

    # Reward analysis (if available)
    if 'reward' in location_group.columns:
        avg_reward = location_group['reward'].mean()
        reward_std = location_group['reward'].std()
    else:
        avg_reward = 0.0
        reward_std = 0.0

    # Agent diversity at location
    agent_count = location_group.get('agent_id', pd.Series()).nunique()

    return {
        'activity_count': activity_count,
        'avg_duration': float(avg_duration) if pd.notna(avg_duration) else 0.0,
        'avg_reward': float(avg_reward),
        'reward_std': float(reward_std),
        'agent_diversity': agent_count,
    }


def _summarize_location_effects(location_effects: Dict[str, Dict]) -> Dict[str, Any]:
    """Summarize location effects across all locations.

    Args:
        location_effects: Dictionary of location effect analyses

    Returns:
        Summary statistics
    """
    if not location_effects:
        return {}

    activities = [effect['activity_count'] for effect in location_effects.values()]
    rewards = [effect['avg_reward'] for effect in location_effects.values()]
    diversities = [effect['agent_diversity'] for effect in location_effects.values()]

    return {
        'total_locations': len(location_effects),
        'avg_activity_per_location': float(np.mean(activities)),
        'max_activity_location': max(activities),
        'avg_reward_per_location': float(np.mean(rewards)),
        'reward_variance': float(np.var(rewards)),
        'avg_agent_diversity': float(np.mean(diversities)),
    }


def _compute_detailed_clustering(activity_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute detailed clustering metrics.

    Args:
        activity_df: DataFrame with location activity data

    Returns:
        Dictionary with detailed clustering metrics
    """
    if activity_df.empty:
        return {}

    coords = activity_df[['position_x', 'position_y']].values

    # Calculate pairwise distances
    n_points = len(coords)
    distances = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(n_points):
            distances[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))

    # Nearest neighbor analysis
    nn_distances = []
    for i in range(n_points):
        # Find minimum distance to other points
        min_dist = np.min(distances[i, distances[i] > 0])
        if min_dist > 0:
            nn_distances.append(min_dist)

    if not nn_distances:
        return {}

    # Clustering metrics
    avg_nn_distance = float(np.mean(nn_distances))
    nn_std = float(np.std(nn_distances))

    # Ripley's K function approximation (simplified)
    k_function = _ripleys_k_approximation(coords, max_radius=10.0, n_radii=20)

    return {
        'avg_nearest_neighbor_distance': avg_nn_distance,
        'nearest_neighbor_std': nn_std,
        'ripleys_k': k_function,
        'n_points': n_points,
    }


def _analyze_resource_clustering(resource_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze clustering patterns in resource locations.

    Args:
        resource_df: DataFrame with resource location data

    Returns:
        Dictionary with resource clustering analysis
    """
    if resource_df.empty:
        return {}

    coords = resource_df[['position_x', 'position_y']].values

    # Resource density analysis
    density_stats = _compute_resource_density_stats(coords)

    # Resource proximity analysis
    proximity_stats = _compute_resource_proximity(coords)

    return {
        'density_stats': density_stats,
        'proximity_stats': proximity_stats,
    }


def _compute_resource_density_stats(coords: np.ndarray) -> Dict[str, Any]:
    """Compute resource density statistics.

    Args:
        coords: Array of resource coordinates

    Returns:
        Dictionary with density statistics
    """
    if len(coords) < 2:
        return {}

    # Create 2D histogram for density estimation
    hist, xedges, yedges = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=10, density=True
    )

    return {
        'max_density': float(np.max(hist)),
        'avg_density': float(np.mean(hist)),
        'density_std': float(np.std(hist)),
        'total_resources': len(coords),
    }


def _compute_resource_proximity(coords: np.ndarray) -> Dict[str, Any]:
    """Compute resource proximity statistics.

    Args:
        coords: Array of resource coordinates

    Returns:
        Dictionary with proximity statistics
    """
    if len(coords) < 2:
        return {}

    # Calculate all pairwise distances
    n = len(coords)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
            distances.append(dist)

    distances = np.array(distances)

    return {
        'avg_distance_between_resources': float(np.mean(distances)),
        'min_distance_between_resources': float(np.min(distances)),
        'max_distance_between_resources': float(np.max(distances)),
        'distance_std': float(np.std(distances)),
    }


def _ripleys_k_approximation(coords: np.ndarray, max_radius: float = 10.0, n_radii: int = 20) -> Dict[str, Any]:
    """Approximate Ripley's K function for point pattern analysis.

    Args:
        coords: Array of point coordinates
        max_radius: Maximum radius to analyze
        n_radii: Number of radius values to evaluate

    Returns:
        Dictionary with K function values
    """
    if len(coords) < 2:
        return {}

    radii = np.linspace(0.1, max_radius, n_radii)
    k_values = []

    area = (np.max(coords[:, 0]) - np.min(coords[:, 0])) * (np.max(coords[:, 1]) - np.min(coords[:, 1]))
    n_points = len(coords)
    intensity = n_points / area

    for radius in radii:
        count = 0
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                    if dist <= radius:
                        count += 1

        k_value = count / (n_points * intensity)
        k_values.append(float(k_value))

    return {
        'radii': radii.tolist(),
        'k_values': k_values,
        'max_radius': max_radius,
        'n_radii': n_radii,
    }
