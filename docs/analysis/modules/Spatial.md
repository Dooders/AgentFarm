# Spatial Analysis Module

**Module Name**: `spatial`

Analyze spatial patterns, movement trajectories, clustering, and location-based effects in simulations.

---

## Overview

The Spatial module examines how agents and resources are distributed in space, movement patterns, territorial behavior, and spatial clustering.

### Key Features

- Spatial distribution analysis
- Movement trajectory tracking
- Clustering detection
- Territorial analysis
- Heat map generation
- Spatial autocorrelation

---

## Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="spatial",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/spatial")
))
```

---

## Data Requirements

### Required Columns

- `agent_id` (str): Agent identifier
- `position_x` (float): X coordinate
- `position_y` (float): Y coordinate
- `step` (int): Time step

### Optional Columns

- `agent_type` (str): Type of agent
- `velocity_x` (float): X velocity
- `velocity_y` (float): Y velocity
- `territory` (str): Territory identifier

---

## Analysis Functions

### analyze_spatial_distribution

Analyze how agents are distributed in space.

**Outputs:**
- `spatial_distribution.csv`: Distribution metrics
- Clustering coefficients, spread, density

### analyze_movement_patterns

Analyze movement trajectories and patterns.

**Outputs:**
- `movement_patterns.csv`: Movement statistics
- Speed, direction, path efficiency

### analyze_clustering

Detect and analyze spatial clusters.

**Outputs:**
- `spatial_clusters.csv`: Cluster information
- Cluster centers, sizes, densities

### analyze_territories

Analyze territorial behavior and space usage.

**Outputs:**
- `territorial_analysis.csv`: Territory metrics
- Territory sizes, overlap, stability

---

## Visualization Functions

### plot_spatial_distribution

Plot agent positions and distributions.

**Output:** `spatial_distribution.png`

### plot_movement_trajectories

Visualize movement paths.

**Output:** `movement_trajectories.png`

### plot_spatial_heatmap

Generate density heatmap.

**Output:** `spatial_heatmap.png`

### plot_clusters

Visualize spatial clusters.

**Output:** `spatial_clusters.png`

---

## Examples

### Basic Spatial Analysis

```python
result = service.run(AnalysisRequest(
    module_name="spatial",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/spatial")
))

import pandas as pd
dist_df = pd.read_csv(result.output_path / "spatial_distribution.csv")
print(f"Mean distance from center: {dist_df['mean_distance'].mean():.2f}")
```

### Movement Analysis

```python
result = service.run(AnalysisRequest(
    module_name="spatial",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/movement"),
    group="movement",
    analysis_kwargs={
        "analyze_movement_patterns": {
            "min_distance": 1.0,
            "track_direction": True
        }
    }
))
```

### Cluster Detection

```python
result = service.run(AnalysisRequest(
    module_name="spatial",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/clusters"),
    analysis_kwargs={
        "analyze_clustering": {
            "algorithm": "dbscan",
            "eps": 5.0,
            "min_samples": 10
        }
    }
))
```

---

## See Also

- [API Reference](../API_REFERENCE.md)
- [Resources Module](./Resources.md)
- [Agents Module](./Agents.md)

---

**Module Version**: 2.0.0
