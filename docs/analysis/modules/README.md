# Analysis Module Documentation

Complete documentation for all analysis modules.

---

## Core Modules

### [Population Analysis](./Population.md)
**Module**: `population`  
**Purpose**: Population dynamics, births, deaths, agent composition

**Key Features:**
- Population trends over time
- Birth/death rate analysis
- Agent composition breakdown
- Growth rate calculations

**Quick Start:**
```python
AnalysisRequest(module_name="population", ...)
```

---

### [Resources Analysis](./Resources.md)
**Module**: `resources`  
**Purpose**: Resource distribution, consumption, efficiency, hotspots

**Key Features:**
- Distribution patterns
- Consumption analysis
- Efficiency metrics
- Spatial hotspot detection

**Quick Start:**
```python
AnalysisRequest(module_name="resources", ...)
```

---

### [Actions Analysis](./Actions.md)
**Module**: `actions`  
**Purpose**: Action patterns, sequences, decisions, rewards

**Key Features:**
- Action frequency analysis
- Sequence pattern detection
- Decision quality metrics
- Reward distributions

**Quick Start:**
```python
AnalysisRequest(module_name="actions", ...)
```

---

### [Agents Analysis](./Agents.md)
**Module**: `agents`  
**Purpose**: Individual agent behavior, lifespans, performance

**Key Features:**
- Lifespan distributions
- Behavioral clustering
- Performance metrics
- Learning curves

**Quick Start:**
```python
AnalysisRequest(module_name="agents", ...)
```

---

## Specialized Modules

### [Learning Analysis](./Learning.md)
**Module**: `learning`  
**Purpose**: Learning performance, curves, module efficiency

**Key Features:**
- Learning curve analysis
- Convergence detection
- Module comparison
- Improvement rates

**Quick Start:**
```python
AnalysisRequest(module_name="learning", ...)
```

---

### [Spatial Analysis](./Spatial.md)
**Module**: `spatial`  
**Purpose**: Spatial patterns, movement, clustering

**Key Features:**
- Spatial distribution
- Movement trajectories
- Cluster detection
- Territorial analysis

**Quick Start:**
```python
AnalysisRequest(module_name="spatial", ...)
```

---

### [Temporal Analysis](./Temporal.md)
**Module**: `temporal`  
**Purpose**: Temporal patterns, trends, periodicity

**Key Features:**
- Trend detection
- Cycle analysis
- Autocorrelation
- Change points

**Quick Start:**
```python
AnalysisRequest(module_name="temporal", ...)
```

---

### [Combat Analysis](./Combat.md)
**Module**: `combat`  
**Purpose**: Combat metrics, effectiveness, matchups

**Key Features:**
- Combat statistics
- Win/loss ratios
- Matchup analysis
- Damage patterns

**Quick Start:**
```python
AnalysisRequest(module_name="combat", ...)
```

---

## Legacy Modules

### [Dominance Analysis](../Dominance.md)
**Module**: `dominance`  
**Purpose**: Dominance hierarchies and social structures

### [Genesis Analysis](../Genesis.md)
**Module**: `genesis`  
**Purpose**: Initial population generation analysis

### [Advantage Analysis](../Advantage.md)
**Module**: `advantage`  
**Purpose**: Relative advantage analysis between agent types

### [Social Behavior Analysis](../Social.md)
**Module**: `social_behavior`  
**Purpose**: Social interaction patterns

### Significant Events
**Module**: `significant_events`  
**Purpose**: Major event detection and analysis

### Comparative Analysis
**Module**: `comparative`  
**Purpose**: Cross-experiment comparison

---

## Module Comparison

| Module | Focus | Data Required | Output Types |
|--------|-------|---------------|--------------|
| **Population** | Population dynamics | step, total_agents | CSV, plots |
| **Resources** | Resource management | step, total_resources | CSV, plots, heatmaps |
| **Actions** | Agent actions | step, action_type | CSV, plots, sequences |
| **Agents** | Individual agents | agent_id | CSV, plots, clusters |
| **Learning** | Learning progress | step, agent_id, metric | CSV, plots, curves |
| **Spatial** | Spatial patterns | position_x, position_y | CSV, plots, heatmaps |
| **Temporal** | Time series | step, metrics | CSV, plots, trends |
| **Combat** | Combat interactions | attacker_id, defender_id | CSV, plots, matrices |

---

## Common Function Groups

Most modules support these function groups:

- **"all"** - Run all functions
- **"analysis"** - Only analysis functions (CSV outputs)
- **"plots"** - Only visualization functions (PNG outputs)
- **"basic"** - Essential functions only

Example:
```python
AnalysisRequest(
    module_name="population",
    experiment_path=Path("data"),
    output_path=Path("results"),
    group="plots"  # Only generate visualizations
)
```

---

## Choosing the Right Module

### For Overall Simulation Health
- **Population** - Track agent populations
- **Resources** - Monitor resource availability
- **Temporal** - Analyze trends over time

### For Agent Behavior
- **Agents** - Individual agent analysis
- **Actions** - What agents are doing
- **Learning** - How agents improve

### For Spatial Analysis
- **Spatial** - Movement and positioning
- **Resources** - Resource hotspots
- **Combat** - Combat locations

### For Performance Analysis
- **Learning** - Learning effectiveness
- **Actions** - Action rewards
- **Combat** - Combat effectiveness
- **Agents** - Agent performance

---

## Module Integration

### Common Combinations

**Population + Resources**
```python
# Analyze resources per capita
pop_result = service.run(AnalysisRequest(module_name="population", ...))
res_result = service.run(AnalysisRequest(module_name="resources", ...))
# Merge and analyze
```

**Agents + Learning**
```python
# Link agent characteristics to learning success
agents_result = service.run(AnalysisRequest(module_name="agents", ...))
learning_result = service.run(AnalysisRequest(module_name="learning", ...))
```

**Spatial + Resources**
```python
# Overlay resource hotspots with agent positions
spatial_result = service.run(AnalysisRequest(module_name="spatial", ...))
res_result = service.run(AnalysisRequest(module_name="resources", ...))
```

**Actions + Combat**
```python
# Analyze combat-related actions
actions_result = service.run(AnalysisRequest(module_name="actions", ...))
combat_result = service.run(AnalysisRequest(module_name="combat", ...))
```

---

## Quick Reference

### Running Any Module

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="<module_name>",      # See module docs
    experiment_path=Path("data"),
    output_path=Path("results"),
    group="all"                       # or "analysis", "plots", "basic"
))
```

### Listing Available Modules

```python
from farm.analysis.registry import list_modules, get_module_names

# Get module names
print(get_module_names())

# Get detailed listing
print(list_modules())
```

### Getting Module Info

```python
from farm.analysis.registry import get_module

module = get_module("population")
info = module.get_info()

print(f"Name: {info['name']}")
print(f"Description: {info['description']}")
print(f"Functions: {info['functions']}")
print(f"Groups: {info['function_groups']}")
```

---

## See Also

- [API Reference](../API_REFERENCE.md) - Complete API documentation
- [Quick Reference](../QUICK_REFERENCE.md) - Common patterns
- [Architecture](../../../farm/analysis/ARCHITECTURE.md) - System design
- [Main Documentation](../INDEX.md) - Documentation index

---

**Last Updated**: 2025-10-04  
**Version**: 2.0.0
