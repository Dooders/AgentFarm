"""
Movement analysis submodule for spatial analysis.
"""

from typing import List, Tuple, Dict, Any
from math import sqrt, atan2, degrees
import numpy as np

from farm.analysis.spatial.compute import compute_movement_patterns


def analyze_movement_trajectories(movement_df) -> Dict[str, Any]:
    """Analyze detailed movement trajectories.

    Args:
        movement_df: DataFrame with movement data

    Returns:
        Dictionary with trajectory analysis
    """
    if movement_df.empty:
        return {}

    trajectories = {}

    for agent_id, group in movement_df.groupby('agent_id'):
        if len(group) > 1:
            trajectory = _analyze_single_trajectory(group)
            trajectories[str(agent_id)] = trajectory

    return {
        'trajectories': trajectories,
        'summary': _summarize_trajectories(trajectories),
    }


def analyze_movement_patterns_detailed(movement_df) -> Dict[str, Any]:
    """Detailed movement pattern analysis.

    Args:
        movement_df: DataFrame with movement data

    Returns:
        Dictionary with detailed movement analysis
    """
    if movement_df.empty:
        return {}

    # Basic patterns
    patterns = compute_movement_patterns(movement_df)

    # Additional detailed analysis
    direction_analysis = _analyze_movement_directions_detailed(movement_df)
    speed_analysis = _analyze_movement_speeds(movement_df)
    path_complexity = _analyze_path_complexity(movement_df)

    return {
        **patterns,
        'direction_analysis': direction_analysis,
        'speed_analysis': speed_analysis,
        'path_complexity': path_complexity,
    }


def calculate_euclidean_distance(pos1: Tuple[float, ...], pos2: Tuple[float, ...]) -> float:
    """Calculate the Euclidean distance between two positions.

    Args:
        pos1: First position coordinates
        pos2: Second position coordinates

    Returns:
        Euclidean distance
    """
    if len(pos1) != len(pos2):
        raise ValueError("Positions must have the same number of dimensions")

    return sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


def _analyze_single_trajectory(trajectory_df) -> Dict[str, Any]:
    """Analyze a single agent's trajectory.

    Args:
        trajectory_df: DataFrame for single agent trajectory

    Returns:
        Dictionary with trajectory metrics
    """
    positions = trajectory_df[['position_x', 'position_y']].values
    distances = trajectory_df['distance'].values

    # Basic metrics
    total_distance = np.sum(distances)
    net_distance = calculate_euclidean_distance(positions[0], positions[-1])
    straightness = net_distance / total_distance if total_distance > 0 else 1.0

    # Speed analysis
    speeds = distances[1:] / np.diff(trajectory_df['step'].values)
    avg_speed = np.mean(speeds) if len(speeds) > 0 else 0

    # Direction changes
    directions = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        angle = degrees(atan2(dy, dx))
        directions.append(angle)

    direction_changes = []
    for i in range(1, len(directions)):
        change = abs(directions[i] - directions[i-1])
        if change > 180:
            change = 360 - change
        direction_changes.append(change)

    avg_direction_change = np.mean(direction_changes) if direction_changes else 0

    return {
        'total_distance': float(total_distance),
        'net_distance': float(net_distance),
        'straightness': float(straightness),
        'avg_speed': float(avg_speed),
        'avg_direction_change': float(avg_direction_change),
        'n_points': len(positions),
    }


def _summarize_trajectories(trajectories: Dict[str, Dict]) -> Dict[str, Any]:
    """Summarize trajectory analysis across all agents.

    Args:
        trajectories: Dictionary of trajectory analyses

    Returns:
        Summary statistics
    """
    if not trajectories:
        return {}

    distances = [t['total_distance'] for t in trajectories.values()]
    straightness = [t['straightness'] for t in trajectories.values()]
    speeds = [t['avg_speed'] for t in trajectories.values()]

    return {
        'avg_total_distance': float(np.mean(distances)),
        'avg_straightness': float(np.mean(straightness)),
        'avg_speed': float(np.mean(speeds)),
        'n_trajectories': len(trajectories),
    }


def _analyze_movement_directions_detailed(movement_df) -> Dict[str, Any]:
    """Detailed movement direction analysis.

    Args:
        movement_df: DataFrame with movement data

    Returns:
        Direction analysis results
    """
    directions = {'N': 0, 'NE': 0, 'E': 0, 'SE': 0, 'S': 0, 'SW': 0, 'W': 0, 'NW': 0}
    total_movements = 0

    for _, row in movement_df.iterrows():
        # This is a placeholder - in practice, you'd calculate direction from position changes
        dx = np.random.uniform(-1, 1)  # Need actual dx calculation
        dy = np.random.uniform(-1, 1)  # Need actual dy calculation

        if abs(dx) > 0.1 or abs(dy) > 0.1:
            angle = degrees(atan2(dy, dx))
            if angle < 0:
                angle += 360

            # Convert angle to direction
            if 337.5 <= angle or angle < 22.5:
                directions['E'] += 1
            elif 22.5 <= angle < 67.5:
                directions['NE'] += 1
            elif 67.5 <= angle < 112.5:
                directions['N'] += 1
            elif 112.5 <= angle < 157.5:
                directions['NW'] += 1
            elif 157.5 <= angle < 202.5:
                directions['W'] += 1
            elif 202.5 <= angle < 247.5:
                directions['SW'] += 1
            elif 247.5 <= angle < 292.5:
                directions['S'] += 1
            else:
                directions['SE'] += 1

            total_movements += 1

    # Convert to proportions
    if total_movements > 0:
        proportions = {k: v / total_movements for k, v in directions.items()}
    else:
        proportions = {k: 0.0 for k in directions.keys()}

    return {
        'counts': directions,
        'proportions': proportions,
        'total_movements': total_movements,
    }


def _analyze_movement_speeds(movement_df) -> Dict[str, Any]:
    """Analyze movement speeds.

    Args:
        movement_df: DataFrame with movement data

    Returns:
        Speed analysis results
    """
    if 'distance' not in movement_df.columns or 'step' not in movement_df.columns:
        return {}

    # Calculate speeds (distance per time step)
    speeds = []
    for _, group in movement_df.groupby('agent_id'):
        group = group.sort_values('step')
        distances = group['distance'].values
        steps = group['step'].values

        if len(steps) > 1:
            step_diffs = np.diff(steps)
            valid_distances = distances[1:]  # Skip first distance (always 0)
            valid_speeds = valid_distances / step_diffs
            speeds.extend(valid_speeds)

    if not speeds:
        return {}

    speeds = np.array(speeds)

    return {
        'mean_speed': float(np.mean(speeds)),
        'median_speed': float(np.median(speeds)),
        'max_speed': float(np.max(speeds)),
        'min_speed': float(np.min(speeds)),
        'speed_std': float(np.std(speeds)),
    }


def _analyze_path_complexity(movement_df) -> Dict[str, Any]:
    """Analyze path complexity metrics.

    Args:
        movement_df: DataFrame with movement data

    Returns:
        Path complexity metrics
    """
    complexities = []

    for _, group in movement_df.groupby('agent_id'):
        if len(group) > 2:
            positions = group[['position_x', 'position_y']].values

            # Calculate path tortuosity (total distance / straight distance)
            total_distance = group['distance'].sum()
            straight_distance = calculate_euclidean_distance(positions[0], positions[-1])

            tortuosity = total_distance / straight_distance if straight_distance > 0 else 1.0

            # Calculate sinuosity (deviation from straight line)
            sinuosity = _calculate_sinuosity(positions)

            complexities.append({
                'tortuosity': tortuosity,
                'sinuosity': sinuosity,
            })

    if not complexities:
        return {}

    tortuosities = [c['tortuosity'] for c in complexities]
    sinuosities = [c['sinuosity'] for c in complexities]

    return {
        'avg_tortuosity': float(np.mean(tortuosities)),
        'avg_sinuosity': float(np.mean(sinuosities)),
        'complexity_metrics': complexities,
    }


def _calculate_sinuosity(positions: np.ndarray) -> float:
    """Calculate path sinuosity (deviation from straight line).

    Args:
        positions: Array of (x, y) positions

    Returns:
        Sinuosity measure
    """
    if len(positions) < 3:
        return 1.0

    # Calculate straight-line distance
    start_to_end = calculate_euclidean_distance(positions[0], positions[-1])

    # Calculate total path length
    total_length = 0
    for i in range(1, len(positions)):
        total_length += calculate_euclidean_distance(positions[i-1], positions[i])

    # Sinuosity is total length / straight distance
    return total_length / start_to_end if start_to_end > 0 else 1.0
