# Resources Analysis Module

**Module Name**: `resources`

Analyze resource distribution, consumption patterns, efficiency, and hotspot locations in simulations.

---

## Overview

The Resources module provides comprehensive analysis of how resources are distributed, consumed, and managed within simulations, including efficiency metrics and spatial hotspot analysis.

### Key Features

- Resource distribution patterns
- Consumption analysis over time
- Resource efficiency metrics
- Hotspot detection and visualization
- Resource depletion patterns
- Consumption rates by agent type

---

## Installation & Usage

### Basic Usage

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/resources")
))
```

### Function Groups

```python
# Run specific analyses
result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results"),
    group="efficiency"  # Options: "all", "analysis", "plots", "basic", "efficiency"
))
```

---

## Data Requirements

### Required Columns

- `step` (int): Simulation step number
- `total_resources` (int/float): Total number of resources

### Optional Columns

- `resource_type` (str): Type of resource
- `consumed` (int/float): Resources consumed this step
- `available` (int/float): Resources available
- `efficiency` (float): Resource utilization efficiency (0-1)
- `position_x` (float): X coordinate for spatial analysis
- `position_y` (float): Y coordinate for spatial analysis
- `agent_type` (str): Type of agent consuming resources

### Example Data Format

```csv
step,total_resources,resource_type,consumed,available,efficiency
0,1000,food,0,1000,0.0
1,950,food,50,950,0.05
2,900,food,50,900,0.055
0,500,water,0,500,0.0
1,480,water,20,480,0.041
```

---

## Analysis Functions

### analyze_patterns

Analyze resource distribution and availability patterns.

**Outputs:**
- `resource_patterns.csv`: Pattern statistics
  - resource_type, mean_available, std, min, max, total_consumed

**Metrics Computed:**
- Mean resource availability
- Resource variance over time
- Distribution patterns
- Stability metrics

### analyze_consumption

Analyze resource consumption rates and trends.

**Outputs:**
- `consumption_analysis.csv`: Consumption statistics
  - step, consumption_rate, cumulative_consumption, efficiency

**Metrics Computed:**
- Consumption rates by step
- Cumulative consumption
- Per-capita consumption
- Consumption trends

### analyze_efficiency

Analyze resource utilization efficiency.

**Outputs:**
- `efficiency_metrics.csv`: Efficiency analysis
  - agent_type, efficiency_mean, efficiency_std, waste_ratio

**Metrics Computed:**
- Utilization efficiency by agent type
- Waste ratios
- Optimal vs actual consumption
- Efficiency trends

### analyze_hotspots

Identify resource hotspots and concentration areas.

**Outputs:**
- `resource_hotspots.csv`: Hotspot locations
  - hotspot_id, center_x, center_y, resource_density, radius

**Metrics Computed:**
- Hotspot locations
- Resource density
- Spatial clustering
- Hotspot persistence

---

## Visualization Functions

### plot_resource

Plot resource distribution over time.

**Output:** `resource_distribution.png`

**Features:**
- Line plot of total resources
- Separate lines by resource type
- Trend lines
- Depletion curves

### plot_consumption

Plot consumption patterns over time.

**Output:** `consumption_over_time.png`

**Features:**
- Consumption rate visualization
- Cumulative consumption
- By agent type (if available)
- Moving averages

### plot_efficiency

Plot efficiency metrics.

**Output:** `efficiency_metrics.png`

**Features:**
- Efficiency over time
- By agent type
- Comparison to optimal
- Efficiency distributions

### plot_hotspots

Visualize resource hotspots on a spatial map.

**Output:** `resource_hotspots.png`

**Features:**
- 2D heatmap of resource density
- Hotspot centers marked
- Concentration gradients
- Temporal evolution (if animated)

---

## Function Groups

### "all"
All analysis and visualization functions.

### "analysis"
Only analysis functions:
- `analyze_patterns`
- `analyze_consumption`
- `analyze_efficiency`
- `analyze_hotspots`

### "plots"
Only visualization functions:
- `plot_resource`
- `plot_consumption`
- `plot_efficiency`
- `plot_hotspots`

### "basic"
Essential functions:
- `analyze_patterns`
- `plot_resource`

### "efficiency"
Efficiency-focused analysis:
- `analyze_efficiency`
- `plot_efficiency`

---

## Output Files

```
output_path/
├── resource_patterns.csv         # Distribution patterns
├── consumption_analysis.csv      # Consumption stats
├── efficiency_metrics.csv        # Efficiency analysis
├── resource_hotspots.csv         # Hotspot locations
├── resource_distribution.png     # Distribution plot
├── consumption_over_time.png     # Consumption plot
├── efficiency_metrics.png        # Efficiency plot
├── resource_hotspots.png         # Hotspot heatmap
└── analysis_summary.json         # Metadata
```

---

## Examples

### Basic Resource Analysis

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/resources")
))

if result.success:
    print(f"Total resources tracked: {len(result.dataframe)}")
    print(f"Resource types: {result.dataframe['resource_type'].unique()}")
```

### Efficiency Analysis

```python
# Focus on efficiency metrics
result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/efficiency"),
    group="efficiency"
))

# Read efficiency metrics
import pandas as pd
efficiency_df = pd.read_csv(result.output_path / "efficiency_metrics.csv")
print(f"Mean efficiency: {efficiency_df['efficiency_mean'].mean():.2%}")
```

### Hotspot Detection

```python
# Analyze resource hotspots
result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/hotspots"),
    analysis_kwargs={
        "analyze_hotspots": {
            "min_density": 10,  # Minimum density threshold
            "radius": 50        # Search radius for clustering
        }
    }
))
```

### Compare Resource Types

```python
# Analyze each resource type separately
result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/resources"),
    processor_kwargs={
        "separate_by_type": True
    }
))

df = result.dataframe
for res_type in df['resource_type'].unique():
    type_df = df[df['resource_type'] == res_type]
    print(f"{res_type}: mean={type_df['available'].mean():.2f}")
```

---

## Integration Examples

### With Population Analysis

```python
# Analyze resource per capita
pop_result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/population")
))

res_result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/resources")
))

# Calculate per-capita resources
pop_df = pop_result.dataframe
res_df = res_result.dataframe

merged = pd.merge(pop_df, res_df, on='step')
merged['resources_per_capita'] = merged['total_resources'] / merged['total_agents']

print(f"Mean resources per capita: {merged['resources_per_capita'].mean():.2f}")
```

### With Spatial Analysis

```python
# Combine resource hotspots with agent movement
spatial_result = service.run(AnalysisRequest(
    module_name="spatial",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/spatial")
))

res_result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/resources"),
    group="hotspots"
))

# Overlay hotspots on movement patterns
# ... spatial overlay code ...
```

---

## Performance Tips

1. **Filter resource types** for focused analysis:
   ```python
   processor_kwargs={"resource_types": ["food", "water"]}
   ```

2. **Limit spatial resolution** for hotspot analysis:
   ```python
   analysis_kwargs={"analyze_hotspots": {"grid_resolution": 10}}
   ```

3. **Use time windows** for efficiency calculations:
   ```python
   processor_kwargs={"window_size": 100}
   ```

4. **Cache results** for iterative analysis:
   ```python
   enable_caching=True
   ```

---

## Troubleshooting

### Common Issues

**"No hotspots detected"**
- Reduce `min_density` threshold
- Check spatial columns exist (`position_x`, `position_y`)
- Verify resource coordinates are valid

**"Efficiency metrics are NaN"**
- Ensure consumption data is available
- Check for division by zero (zero resources)
- Verify efficiency calculation parameters

**"Resource types not separated"**
- Add `resource_type` column to data
- Use `separate_by_type=True` in processor_kwargs

### Debug Spatial Analysis

```python
# Check spatial data
df = result.dataframe
print(f"Has position_x: {'position_x' in df.columns}")
print(f"Has position_y: {'position_y' in df.columns}")
print(f"Position range X: [{df['position_x'].min()}, {df['position_x'].max()}]")
print(f"Position range Y: [{df['position_y'].min()}, {df['position_y'].max()}]")
```

---

## Advanced Usage

### Custom Efficiency Calculation

```python
from farm.analysis.resources.compute import calculate_custom_efficiency

# Override default efficiency calculation
result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results"),
    analysis_kwargs={
        "analyze_efficiency": {
            "efficiency_func": calculate_custom_efficiency,
            "optimal_consumption": 0.8  # 80% utilization target
        }
    }
))
```

### Hotspot Animation

```python
# Generate hotspot evolution over time
result = service.run(AnalysisRequest(
    module_name="resources",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/hotspots"),
    analysis_kwargs={
        "plot_hotspots": {
            "animate": True,
            "frame_step": 10,
            "output_format": "gif"
        }
    }
))
```

---

## See Also

- [API Reference](../API_REFERENCE.md) - Complete API
- [Population Module](./Population.md) - Population analysis
- [Spatial Module](./Spatial.md) - Spatial patterns
- [Quick Reference](../QUICK_REFERENCE.md) - Common patterns

---

**Module Version**: 2.0.0  
**Last Updated**: 2025-10-04
