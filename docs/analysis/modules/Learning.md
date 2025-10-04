# Learning Analysis Module

**Module Name**: `learning`

Analyze learning performance, agent learning curves, module efficiency, and adaptation patterns.

---

## Overview

The Learning module examines how agents learn over time, tracking performance improvements, learning rates, and the effectiveness of learning algorithms.

### Key Features

- Learning curve analysis
- Performance improvement tracking
- Learning rate calculations
- Convergence detection
- Module efficiency metrics
- Comparative learning analysis

---

## Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/learning")
))
```

---

## Data Requirements

### Required Columns

- `step` (int): Simulation step
- `agent_id` (str): Agent identifier
- `learning_metric` (float): Learning performance metric (e.g., reward, score)

### Optional Columns

- `agent_type` (str): Type of agent
- `learning_module` (str): Learning algorithm/module used
- `exploration_rate` (float): Exploration parameter
- `loss` (float): Training loss
- `accuracy` (float): Performance accuracy

---

## Analysis Functions

### analyze_learning_performance

Analyze overall learning performance trends.

**Outputs:**
- `learning_performance.csv`: Performance statistics
  - step, mean_performance, std, improvement_rate

### analyze_learning_curves

Analyze individual and aggregate learning curves.

**Outputs:**
- `learning_curves.csv`: Learning trajectories
  - agent_id, step, performance, smoothed_performance

### analyze_module_efficiency

Compare efficiency of different learning modules.

**Outputs:**
- `module_efficiency.csv`: Module comparison
  - module_name, final_performance, learning_rate, convergence_step

### analyze_convergence

Detect learning convergence and plateaus.

**Outputs:**
- `convergence_analysis.csv`: Convergence statistics
  - agent_id, convergence_step, plateau_performance, stability

---

## Visualization Functions

### plot_learning_curves

Plot learning curves over time.

**Output:** `learning_curves.png`

**Features:**
- Individual and mean curves
- Confidence intervals
- Smoothed trajectories

### plot_performance_comparison

Compare learning performance across groups.

**Output:** `performance_comparison.png`

### plot_learning_rates

Visualize learning rate evolution.

**Output:** `learning_rates.png`

### plot_module_efficiency

Compare learning module efficiency.

**Output:** `module_efficiency.png`

---

## Function Groups

- **"all"**: All functions
- **"analysis"**: All analysis functions
- **"plots"**: All visualizations
- **"performance"**: Performance-focused analysis
- **"basic"**: Essential functions

---

## Examples

### Basic Learning Analysis

```python
result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/learning"),
    group="basic"
))

# Check learning progress
import pandas as pd
perf_df = pd.read_csv(result.output_path / "learning_performance.csv")
initial = perf_df.iloc[0]['mean_performance']
final = perf_df.iloc[-1]['mean_performance']
improvement = ((final - initial) / initial) * 100

print(f"Learning improvement: {improvement:.1f}%")
```

### Compare Learning Modules

```python
result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/modules"),
    processor_kwargs={
        "group_by_module": True
    }
))

# Compare module performance
modules_df = pd.read_csv(result.output_path / "module_efficiency.csv")
best_module = modules_df.loc[modules_df['final_performance'].idxmax()]
print(f"Best module: {best_module['module_name']}")
print(f"Final performance: {best_module['final_performance']:.3f}")
```

### Detect Learning Plateaus

```python
result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/convergence"),
    analysis_kwargs={
        "analyze_convergence": {
            "window_size": 100,
            "threshold": 0.01  # 1% improvement threshold
        }
    }
))

conv_df = pd.read_csv(result.output_path / "convergence_analysis.csv")
mean_convergence = conv_df['convergence_step'].mean()
print(f"Average convergence at step: {mean_convergence:.0f}")
```

### Learning Rate Analysis

```python
result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/rates"),
    processor_kwargs={
        "calculate_rates": True,
        "rate_window": 50
    }
))
```

---

## Advanced Examples

### Compare Agent Types

```python
result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/by_type"),
    processor_kwargs={
        "group_by_agent_type": True
    }
))

df = result.dataframe
for agent_type in df['agent_type'].unique():
    type_df = df[df['agent_type'] == agent_type]
    final_perf = type_df.groupby('agent_id')['learning_metric'].last().mean()
    print(f"{agent_type}: {final_perf:.3f}")
```

### Early vs Late Learners

```python
result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/timing")
))

conv_df = pd.read_csv(result.output_path / "convergence_analysis.csv")

# Split by convergence speed
median_convergence = conv_df['convergence_step'].median()
early_learners = conv_df[conv_df['convergence_step'] <= median_convergence]
late_learners = conv_df[conv_df['convergence_step'] > median_convergence]

print(f"Early learners: {len(early_learners)}")
print(f"Late learners: {len(late_learners)}")
print(f"Early avg performance: {early_learners['plateau_performance'].mean():.3f}")
print(f"Late avg performance: {late_learners['plateau_performance'].mean():.3f}")
```

---

## Integration Examples

### With Actions Module

```python
# Correlate learning progress with action choices
learning_result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/learning")
))

actions_result = service.run(AnalysisRequest(
    module_name="actions",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/actions")
))

# Analyze how action diversity changes with learning
```

### With Agents Module

```python
# Link learning curves to agent characteristics
agents_result = service.run(AnalysisRequest(
    module_name="agents",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/agents")
))

learning_result = service.run(AnalysisRequest(
    module_name="learning",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/learning")
))

# Identify agent features that predict learning success
```

---

## Performance Tips

1. **Smooth noisy curves:**
   ```python
   processor_kwargs={"smoothing_window": 50}
   ```

2. **Focus on successful learners:**
   ```python
   processor_kwargs={"min_final_performance": 0.5}
   ```

3. **Sample large populations:**
   ```python
   processor_kwargs={"sample_agents": 100}
   ```

---

## See Also

- [API Reference](../API_REFERENCE.md)
- [Agents Module](./Agents.md)
- [Actions Module](./Actions.md)

---

**Module Version**: 2.0.0  
**Last Updated**: 2025-10-04
