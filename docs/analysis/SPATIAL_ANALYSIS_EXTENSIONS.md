# Spatial Analysis Module - Extension Ideas

## Current Capabilities
- Spatial overview and distribution
- Movement patterns and trajectories
- Location hotspots
- Clustering analysis
- Movement directions
- Density mapping

---

## 1. 🗺️ Territory & Home Range Analysis

### A. Territory Mapping
```python
def analyze_agent_territories(position_data: pd.DataFrame) -> Dict[str, Any]:
    """Identify and map individual agent territories.
    
    Uses minimum convex polygon (MCP) or kernel density estimation (KDE)
    to define territory boundaries.
    """
    from scipy.spatial import ConvexHull
    from sklearn.neighbors import KernelDensity
    
    territories = {}
    for agent_id in position_data['agent_id'].unique():
        agent_positions = position_data[position_data['agent_id'] == agent_id]
        points = agent_positions[['x', 'y']].values
        
        # Minimum Convex Polygon
        if len(points) >= 3:
            hull = ConvexHull(points)
            territory_area = hull.volume  # In 2D, volume gives area
            
            # 95% KDE contour for core area
            kde = KernelDensity(bandwidth=5.0).fit(points)
            
            territories[agent_id] = {
                'area': float(territory_area),
                'perimeter': float(hull.area),  # In 2D, area gives perimeter
                'centroid': points.mean(axis=0).tolist(),
                'core_area_95': calculate_kde_contour(kde, points, 0.95)
            }
    
    # Territory overlap analysis
    overlaps = calculate_territory_overlaps(territories)
    
    return {
        'territories': territories,
        'mean_territory_size': np.mean([t['area'] for t in territories.values()]),
        'territory_overlap_index': overlaps,
        'territorial_agents': len(territories)
    }
```

### B. Home Range Estimation
```python
def compute_home_ranges(position_data: pd.DataFrame, method: str = 'kde') -> Dict[str, Any]:
    """Calculate home range sizes using various methods.
    
    Methods:
    - MCP (Minimum Convex Polygon): 100%, 95%, 50% contours
    - KDE (Kernel Density Estimation): probabilistic core areas
    - Brownian Bridge: movement-based estimation
    """
    from scipy.stats import multivariate_normal
    
    results = {}
    
    for agent_id in position_data['agent_id'].unique():
        positions = position_data[position_data['agent_id'] == agent_id][['x', 'y']].values
        
        if method == 'kde':
            # Estimate 2D density
            kde = KernelDensity(bandwidth='scott', kernel='gaussian')
            kde.fit(positions)
            
            # Create grid
            x_grid = np.linspace(positions[:, 0].min(), positions[:, 0].max(), 100)
            y_grid = np.linspace(positions[:, 1].min(), positions[:, 1].max(), 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_points = np.vstack([X.ravel(), Y.ravel()]).T
            
            # Calculate density
            log_density = kde.score_samples(grid_points)
            density = np.exp(log_density).reshape(X.shape)
            
            # Calculate contour areas (50%, 95%)
            contour_50 = calculate_contour_area(density, 0.50)
            contour_95 = calculate_contour_area(density, 0.95)
            
            results[agent_id] = {
                'core_area_50': contour_50,
                'home_range_95': contour_95,
                'range_ratio': contour_95 / contour_50 if contour_50 > 0 else 0
            }
    
    return results
```

---

## 2. 🎯 Spatial Preference Analysis

### A. Site Fidelity
```python
def analyze_site_fidelity(position_data: pd.DataFrame) -> Dict[str, Any]:
    """Measure how consistently agents return to specific locations."""
    
    from scipy.spatial.distance import cdist
    
    fidelity_scores = {}
    
    for agent_id in position_data['agent_id'].unique():
        positions = position_data[position_data['agent_id'] == agent_id][['x', 'y']].values
        
        if len(positions) < 10:
            continue
        
        # Calculate pairwise distances
        distances = cdist(positions, positions)
        
        # Site fidelity index: how often agent returns to similar locations
        # Low mean distance = high fidelity
        mean_distance_to_past = np.mean(distances[np.triu_indices_from(distances, k=1)])
        
        # Calculate revisit frequency
        # Grid the space and count revisits
        grid_size = 10
        visits_per_cell = discretize_positions(positions, grid_size)
        revisit_rate = (visits_per_cell > 1).sum() / len(visits_per_cell)
        
        fidelity_scores[agent_id] = {
            'mean_displacement': float(mean_distance_to_past),
            'revisit_rate': float(revisit_rate),
            'fidelity_score': float(1.0 / (1.0 + mean_distance_to_past))
        }
    
    return {
        'agent_fidelity': fidelity_scores,
        'mean_fidelity': np.mean([s['fidelity_score'] for s in fidelity_scores.values()]),
        'high_fidelity_agents': sum(1 for s in fidelity_scores.values() if s['fidelity_score'] > 0.7)
    }
```

### B. Resource Selection Function
```python
def compute_resource_selection(agent_positions: pd.DataFrame, 
                               resource_positions: pd.DataFrame,
                               grid_size: int = 10) -> Dict[str, Any]:
    """Analyze which resource types/locations agents prefer.
    
    Uses used vs available analysis.
    """
    
    # Create grid
    x_bins = np.linspace(0, grid_size, 20)
    y_bins = np.linspace(0, grid_size, 20)
    
    # Calculate availability (resources available)
    resource_grid, _, _ = np.histogram2d(
        resource_positions['x'], resource_positions['y'],
        bins=[x_bins, y_bins]
    )
    
    # Calculate usage (where agents go)
    usage_grid, _, _ = np.histogram2d(
        agent_positions['x'], agent_positions['y'],
        bins=[x_bins, y_bins]
    )
    
    # Selection ratio: usage / availability
    # >1 = preference, <1 = avoidance
    with np.errstate(divide='ignore', invalid='ignore'):
        selection_ratio = usage_grid / resource_grid
        selection_ratio[~np.isfinite(selection_ratio)] = 0
    
    # Identify preferred vs avoided areas
    preferred_cells = (selection_ratio > 1.5).sum()
    avoided_cells = (selection_ratio < 0.5).sum()
    
    return {
        'selection_ratio_mean': float(np.mean(selection_ratio[selection_ratio > 0])),
        'preferred_cells': int(preferred_cells),
        'avoided_cells': int(avoided_cells),
        'selection_strength': float(np.std(selection_ratio[selection_ratio > 0])),
        'has_strong_preference': np.max(selection_ratio) > 2.0
    }
```

---

## 3. 🚶 Movement Analysis

### A. Step Length Distribution
```python
def analyze_step_lengths(trajectory_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze movement step length distributions.
    
    Can reveal movement modes (e.g., search vs. transit).
    """
    from scipy import stats
    
    step_lengths = []
    
    for agent_id in trajectory_data['agent_id'].unique():
        agent_traj = trajectory_data[trajectory_data['agent_id'] == agent_id].sort_values('step')
        positions = agent_traj[['x', 'y']].values
        
        # Calculate step lengths
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        step_lengths.extend(steps)
    
    step_lengths = np.array(step_lengths)
    
    # Fit distributions
    # Exponential for random walk
    exp_params = stats.expon.fit(step_lengths)
    # Gamma for correlated random walk
    gamma_params = stats.gamma.fit(step_lengths)
    # Levy for levy flights
    levy_params = stats.levy_stable.fit(step_lengths)
    
    # AIC comparison
    aics = {
        'exponential': compute_aic(stats.expon, step_lengths, exp_params),
        'gamma': compute_aic(stats.gamma, step_lengths, gamma_params),
        'levy': compute_aic(stats.levy_stable, step_lengths, levy_params)
    }
    
    best_fit = min(aics, key=aics.get)
    
    return {
        'mean_step_length': float(np.mean(step_lengths)),
        'std_step_length': float(np.std(step_lengths)),
        'max_step_length': float(np.max(step_lengths)),
        'best_fit_distribution': best_fit,
        'movement_mode': 'ballistic' if best_fit == 'levy' else 'diffusive',
        'step_distribution_params': locals()[f'{best_fit}_params']
    }
```

### B. Turning Angle Analysis
```python
def analyze_turning_angles(trajectory_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze turning angles to understand movement patterns.
    
    Can reveal:
    - Directed movement (angles near 0)
    - Random movement (uniform distribution)
    - Tortuous movement (large angles)
    """
    
    turning_angles = []
    
    for agent_id in trajectory_data['agent_id'].unique():
        agent_traj = trajectory_data[trajectory_data['agent_id'] == agent_id].sort_values('step')
        positions = agent_traj[['x', 'y']].values
        
        if len(positions) < 3:
            continue
        
        # Calculate bearing angles
        diffs = np.diff(positions, axis=0)
        bearings = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        # Calculate turning angles
        turns = np.diff(bearings)
        # Wrap to [-pi, pi]
        turns = np.arctan2(np.sin(turns), np.cos(turns))
        turning_angles.extend(turns)
    
    turning_angles = np.array(turning_angles)
    
    # Statistics
    mean_cosine = np.mean(np.cos(turning_angles))  # Directedness measure
    concentration = np.sqrt(np.mean(np.cos(turning_angles))**2 + np.mean(np.sin(turning_angles))**2)
    
    return {
        'mean_turning_angle': float(np.mean(turning_angles)),
        'turning_angle_std': float(np.std(turning_angles)),
        'mean_cosine': float(mean_cosine),
        'concentration': float(concentration),
        'movement_straightness': 'directed' if concentration > 0.7 else 'tortuous',
        'persistence_score': float((1 + mean_cosine) / 2)  # 0-1 scale
    }
```

### C. Fractal Dimension
```python
def calculate_trajectory_fractal_dimension(trajectory: np.ndarray) -> float:
    """Calculate fractal dimension of trajectory using box-counting.
    
    D ≈ 1: straight line
    D ≈ 2: space-filling curve
    """
    
    # Box-counting algorithm
    scales = np.logspace(0, 3, num=20, dtype=int)
    counts = []
    
    # Normalize trajectory to unit square
    traj_norm = (trajectory - trajectory.min(axis=0)) / (trajectory.max(axis=0) - trajectory.min(axis=0))
    
    for scale in scales:
        # Count boxes containing trajectory points
        boxes = set()
        for point in traj_norm:
            box_coords = tuple((point * scale).astype(int))
            boxes.add(box_coords)
        counts.append(len(boxes))
    
    # Fit log-log relationship
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    fractal_dim = coeffs[0]
    
    return float(fractal_dim)
```

---

## 4. 🌐 Spatial Interaction

### A. Nearest Neighbor Analysis
```python
def analyze_nearest_neighbors(positions: pd.DataFrame) -> Dict[str, Any]:
    """Analyze nearest neighbor distances and patterns."""
    
    from scipy.spatial import cKDTree
    from scipy import stats
    
    coords = positions[['x', 'y']].values
    tree = cKDTree(coords)
    
    # Find nearest neighbor for each point
    distances, indices = tree.query(coords, k=2)  # k=2 to exclude self
    nn_distances = distances[:, 1]  # Take 2nd nearest (1st is self)
    
    # Calculate Clark-Evans index (observed / expected)
    # Expected for CSR (Complete Spatial Randomness)
    n = len(coords)
    area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
    density = n / area
    expected_nn = 0.5 / np.sqrt(density)
    
    clark_evans = np.mean(nn_distances) / expected_nn
    
    # Ripley's K function for multiple scales
    ripleys_k = compute_ripleys_k(coords, radii=np.linspace(1, 20, 10))
    
    return {
        'mean_nn_distance': float(np.mean(nn_distances)),
        'clark_evans_index': float(clark_evans),
        'spatial_pattern': 'clustered' if clark_evans < 1 else 'dispersed' if clark_evans > 1 else 'random',
        'ripleys_k': ripleys_k,
        'aggregation_strength': float(abs(clark_evans - 1))
    }
```

### B. Interaction Zones
```python
def identify_interaction_zones(agent_positions: pd.DataFrame,
                               interaction_distance: float = 5.0) -> Dict[str, Any]:
    """Identify zones where agents frequently interact."""
    
    from scipy.spatial import distance_matrix
    from sklearn.cluster import DBSCAN
    
    # Find all interaction events
    interactions = []
    
    # Group by time step
    for step in agent_positions['step'].unique():
        step_positions = agent_positions[agent_positions['step'] == step]
        coords = step_positions[['x', 'y']].values
        
        # Calculate pairwise distances
        distances = distance_matrix(coords, coords)
        
        # Find interactions (agents within interaction_distance)
        interacting = np.where((distances < interaction_distance) & (distances > 0))
        
        for i, j in zip(*interacting):
            if i < j:  # Avoid double-counting
                midpoint = (coords[i] + coords[j]) / 2
                interactions.append(midpoint)
    
    if not interactions:
        return {'num_interaction_zones': 0}
    
    interactions = np.array(interactions)
    
    # Cluster interaction locations
    clustering = DBSCAN(eps=interaction_distance, min_samples=5).fit(interactions)
    
    zones = []
    for label in set(clustering.labels_):
        if label == -1:
            continue
        zone_points = interactions[clustering.labels_ == label]
        zones.append({
            'centroid': zone_points.mean(axis=0).tolist(),
            'size': len(zone_points),
            'radius': np.std(zone_points)
        })
    
    return {
        'num_interaction_zones': len(zones),
        'interaction_zones': zones,
        'total_interactions': len(interactions),
        'mean_zone_size': np.mean([z['size'] for z in zones]) if zones else 0
    }
```

---

## 5. 🎨 Advanced Clustering

### A. Hierarchical Clustering
```python
def hierarchical_spatial_clustering(positions: pd.DataFrame) -> Dict[str, Any]:
    """Perform hierarchical clustering to understand spatial organization."""
    
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist
    
    coords = positions[['x', 'y']].values
    
    # Compute linkage
    Z = linkage(coords, method='ward')
    
    # Cut dendrogram at different heights
    clusters_2 = fcluster(Z, 2, criterion='maxclust')
    clusters_3 = fcluster(Z, 3, criterion='maxclust')
    clusters_5 = fcluster(Z, 5, criterion='maxclust')
    
    # Calculate silhouette scores
    from sklearn.metrics import silhouette_score
    scores = {
        2: silhouette_score(coords, clusters_2),
        3: silhouette_score(coords, clusters_3),
        5: silhouette_score(coords, clusters_5)
    }
    
    optimal_k = max(scores, key=scores.get)
    
    return {
        'optimal_clusters': optimal_k,
        'silhouette_scores': scores,
        'hierarchical_structure': 'nested' if scores[5] > 0.5 else 'flat'
    }
```

### B. HDBSCAN Clustering
```python
def adaptive_density_clustering(positions: pd.DataFrame) -> Dict[str, Any]:
    """Use HDBSCAN for density-based clustering with varying density."""
    
    import hdbscan
    
    coords = positions[['x', 'y']].values
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    cluster_labels = clusterer.fit_predict(coords)
    
    # Analyze cluster properties
    clusters = []
    for label in set(cluster_labels):
        if label == -1:
            continue  # Noise points
        
        cluster_points = coords[cluster_labels == label]
        clusters.append({
            'size': len(cluster_points),
            'density': len(cluster_points) / cluster_points.std(),
            'centroid': cluster_points.mean(axis=0).tolist()
        })
    
    return {
        'num_clusters': len(clusters),
        'noise_points': (cluster_labels == -1).sum(),
        'cluster_persistence': clusterer.cluster_persistence_.tolist(),
        'clusters': clusters
    }
```

---

## 6. 🔥 Hotspot Analysis

### A. Getis-Ord Gi* Statistics
```python
def calculate_hotspot_statistics(positions: pd.DataFrame,
                                 grid_size: int = 20) -> Dict[str, Any]:
    """Identify statistically significant hotspots and coldspots."""
    
    # Create grid and count
    x_bins = np.linspace(positions['x'].min(), positions['x'].max(), grid_size)
    y_bins = np.linspace(positions['y'].min(), positions['y'].max(), grid_size)
    
    grid, _, _ = np.histogram2d(positions['x'], positions['y'], bins=[x_bins, y_bins])
    
    # Calculate Gi* statistic for each cell
    gi_stats = np.zeros_like(grid)
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Get neighbors (3x3 window)
            neighbors = grid[max(0, i-1):min(grid.shape[0], i+2),
                           max(0, j-1):min(grid.shape[1], j+2)]
            
            # Calculate Gi*
            mean_neighbors = neighbors.mean()
            std_neighbors = neighbors.std()
            
            if std_neighbors > 0:
                gi_stats[i, j] = (grid[i, j] - mean_neighbors) / std_neighbors
    
    # Identify significant hotspots (z-score > 1.96)
    hotspots = gi_stats > 1.96
    coldspots = gi_stats < -1.96
    
    return {
        'num_hotspots': int(hotspots.sum()),
        'num_coldspots': int(coldspots.sum()),
        'hotspot_locations': np.argwhere(hotspots).tolist(),
        'max_gi_statistic': float(gi_stats.max()),
        'clustering_strength': float(hotspots.sum() / grid.size)
    }
```

---

## 7. 🧭 Spatial Autocorrelation

### A. Moran's I
```python
def calculate_morans_i(positions: pd.DataFrame, 
                      values: np.ndarray,
                      distance_threshold: float = 10.0) -> Dict[str, Any]:
    """Calculate Moran's I for spatial autocorrelation.
    
    Useful for analyzing if similar values cluster spatially.
    """
    
    from scipy.spatial import distance_matrix
    
    coords = positions[['x', 'y']].values
    n = len(coords)
    
    # Create spatial weights matrix
    distances = distance_matrix(coords, coords)
    W = (distances < distance_threshold) & (distances > 0)
    W = W.astype(float)
    
    # Normalize weights
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / row_sums
    W[~np.isfinite(W)] = 0
    
    # Calculate Moran's I
    mean_value = values.mean()
    deviations = values - mean_value
    
    numerator = np.sum(W * np.outer(deviations, deviations))
    denominator = np.sum(deviations**2)
    
    morans_i = (n / W.sum()) * (numerator / denominator)
    
    # Calculate expected value and variance for significance test
    expected_i = -1 / (n - 1)
    
    return {
        'morans_i': float(morans_i),
        'expected_i': float(expected_i),
        'z_score': float((morans_i - expected_i) / np.sqrt(1/n)),  # Simplified
        'interpretation': 'clustered' if morans_i > expected_i else 'dispersed',
        'is_significant': abs(morans_i - expected_i) > 0.1
    }
```

---

## 8. 📏 Scale Analysis

### A. Multi-Scale Analysis
```python
def analyze_across_scales(positions: pd.DataFrame,
                         scales: List[float] = [5, 10, 20, 50]) -> Dict[str, Any]:
    """Analyze spatial patterns at multiple scales."""
    
    results = {}
    
    for scale in scales:
        # Grid at this scale
        n_bins = int(100 / scale)
        grid, x_edges, y_edges = np.histogram2d(
            positions['x'], positions['y'],
            bins=n_bins
        )
        
        # Calculate metrics at this scale
        results[scale] = {
            'occupied_cells': int((grid > 0).sum()),
            'mean_density': float(grid[grid > 0].mean()),
            'density_variance': float(grid.var()),
            'spatial_heterogeneity': float(grid.std() / grid.mean()) if grid.mean() > 0 else 0
        }
    
    # Identify characteristic scale (where pattern is strongest)
    heterogeneities = [r['spatial_heterogeneity'] for r in results.values()]
    characteristic_scale = scales[np.argmax(heterogeneities)]
    
    return {
        'scale_analysis': results,
        'characteristic_scale': characteristic_scale,
        'is_scale_dependent': max(heterogeneities) / min(heterogeneities) > 2
    }
```

---

## Summary

These extensions provide:

1. **Territory Analysis** - Home ranges, territory boundaries
2. **Preference Analysis** - Site fidelity, resource selection
3. **Movement Analysis** - Step lengths, turning angles, fractal dimension
4. **Interaction Analysis** - Nearest neighbors, interaction zones
5. **Advanced Clustering** - Hierarchical, HDBSCAN
6. **Hotspot Analysis** - Getis-Ord statistics
7. **Autocorrelation** - Moran's I
8. **Scale Analysis** - Multi-scale patterns

All are directly applicable to spatial ecology, behavioral analysis, and spatial statistics research questions!
