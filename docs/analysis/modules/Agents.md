# Agents Analysis Module

**Module Name**: `agents`

Analyze individual agent behavior, lifespans, performance, learning patterns, and behavioral clustering.

---

## Overview

The Agents module provides detailed analysis of individual agents, including their lifespans, behavioral patterns, performance metrics, and learning trajectories.

### Key Features

- Agent lifespan analysis
- Behavior clustering
- Performance metrics
- Learning curve analysis
- Individual agent statistics
- Behavioral phenotypes

---

## Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/agents")
))
```

---

## Data Requirements

### Required Columns

- `agent_id` (str): Unique agent identifier

### Optional Columns

- `birth_step` (int): When agent was born
- `death_step` (int): When agent died
- `agent_type` (str): Type of agent
- `performance_score` (float): Performance metric
- `learning_progress` (float): Learning metric
- `behavior_features` (dict/json): Behavioral features

---

## Analysis Functions

### analyze_statistics

Calculate comprehensive agent statistics.

**Outputs:**
- `agent_statistics.csv`: Overall statistics
  - total_agents, mean_lifespan, variance, by_type

### analyze_lifespans

Analyze agent lifespan distributions and patterns.

**Outputs:**
- `lifespan_analysis.csv`: Lifespan statistics
  - agent_type, mean_lifespan, median, std, max

**Metrics:**
- Survival distributions
- Mortality rates
- Lifespan by type

### analyze_behaviors

Cluster agents by behavioral patterns.

**Outputs:**
- `behavior_clusters.csv`: Cluster assignments
  - cluster_id, agent_count, centroid_features

**Features:**
- K-means clustering
- Behavioral phenotypes
- Cluster characteristics

### analyze_performance

Analyze agent performance metrics.

**Outputs:**
- `performance_analysis.csv`: Performance stats
  - agent_id, score, rank, percentile

**Metrics:**
- Performance distributions
- Top/bottom performers
- Performance over time

### analyze_learning

Analyze learning curves and progress.

**Outputs:**
- `learning_curves.csv`: Learning trajectories
  - agent_id, step, learning_score, improvement_rate

**Metrics:**
- Learning rates
- Plateau detection
- Convergence analysis

---

## Visualization Functions

### plot_lifespans

Plot lifespan distributions.

**Output:** `lifespan_distributions.png`

### plot_behaviors

Visualize behavior clusters.

**Output:** `behavior_clusters.png`

### plot_performance

Plot performance metrics and distributions.

**Output:** `performance_metrics.png`

### plot_learning

Plot learning curves.

**Output:** `learning_curves.png`

---

## Function Groups

- **"all"**: All functions
- **"analysis"**: All analysis functions
- **"plots"**: All visualizations
- **"lifespan"**: Lifespan-focused analysis
- **"behavior"**: Behavioral clustering
- **"basic"**: Essential functions

---

## Examples

### Lifespan Analysis

```python
result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/lifespans"),
    group="lifespan"
))

# Read lifespan data
import pandas as pd
lifespan_df = pd.read_csv(result.output_path / "lifespan_analysis.csv")
print(f"Mean lifespan: {lifespan_df['mean_lifespan'].mean():.2f}")
```

### Behavior Clustering

```python
result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/behaviors"),
    group="behavior",
    analysis_kwargs={
        "analyze_behaviors": {
            "n_clusters": 5,
            "features": ["action_diversity", "exploration_rate", "success_rate"]
        }
    }
))

# Analyze clusters
clusters_df = pd.read_csv(result.output_path / "behavior_clusters.csv")
print(f"Found {len(clusters_df)} behavioral clusters")
```

### Performance Analysis

```python
result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/performance"),
    analysis_kwargs={
        "analyze_performance": {
            "metric": "cumulative_reward",
            "top_n": 10
        }
    }
))

# Get top performers
perf_df = pd.read_csv(result.output_path / "performance_analysis.csv")
top_agents = perf_df.nsmallest(10, 'rank')
print(top_agents[['agent_id', 'score', 'rank']])
```

### Learning Curve Analysis

```python
result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/learning"),
    processor_kwargs={
        "window_size": 100,  # Smoothing window
        "min_steps": 500     # Minimum steps for inclusion
    }
))
```

---

## Advanced Examples

### Compare Agent Types

```python
from farm.analysis.common.metrics import split_and_compare_groups

result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/agents")
))

df = result.dataframe

# Compare lifespans by agent type
comparison = split_and_compare_groups(
    df,
    split_column='agent_type',
    metrics=['lifespan', 'performance_score']
)

print(comparison)
```

### Identify Elite Agents

```python
# Find top 5% performers
result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/elite")
))

df = result.dataframe
threshold = df['performance_score'].quantile(0.95)
elite_agents = df[df['performance_score'] >= threshold]

print(f"Elite agents: {len(elite_agents)}")
print(f"Mean elite score: {elite_agents['performance_score'].mean():.3f}")
```

### Behavioral Evolution

```python
# Track behavioral changes over time
result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/evolution"),
    processor_kwargs={
        "time_windows": [0, 500, 1000, 1500, 2000],
        "track_evolution": True
    }
))
```

---

## Integration Examples

### With Population Module

```python
# Analyze relationship between population and individual agents
pop_result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/population")
))

agents_result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/agents")
))

# Correlate population trends with agent lifespans
```

### With Actions Module

```python
# Link agent behaviors to action patterns
actions_result = service.run(AnalysisRequest(
    module_name="actions",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/actions")
))

agents_result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/agents"),
    group="behavior"
))

# Analyze action patterns by behavioral cluster
```

---

## Performance Tips

1. **Filter to active agents:**
   ```python
   processor_kwargs={"min_lifespan": 100}
   ```

2. **Reduce clustering features:**
   ```python
   analysis_kwargs={"analyze_behaviors": {"features": ["key_feature_1", "key_feature_2"]}}
   ```

3. **Sample large populations:**
   ```python
   processor_kwargs={"sample_rate": 0.1}  # 10% sample
   ```

---

## See Also

- [API Reference](../API_REFERENCE.md)
- [Actions Module](./Actions.md)
- [Learning Module](./Learning.md)
- [Population Module](./Population.md)

---

**Module Version**: 2.0.0  
**Last Updated**: 2025-10-04
