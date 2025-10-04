# Actions Analysis Module

**Module Name**: `actions`

Analyze action patterns, sequences, decision-making processes, and performance metrics in agent-based simulations.

---

## Overview

The Actions module examines what actions agents take, when they take them, how decisions are made, and how successful those actions are.

### Key Features

- Action frequency analysis
- Sequence pattern detection
- Decision-making analysis
- Reward/performance metrics
- Success/failure rates
- Action transitions

---

## Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="actions",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/actions")
))
```

---

## Data Requirements

### Required Columns

- `step` (int): Simulation step
- `action_type` (str): Type of action taken
- `frequency` (int/float): Number of times action occurred

### Optional Columns

- `agent_id` (str): Agent identifier
- `agent_type` (str): Type of agent
- `success` (bool): Whether action succeeded
- `reward` (float): Reward received
- `previous_action` (str): Previous action (for sequences)
- `state` (str): State when action was taken

---

## Analysis Functions

### analyze_patterns

Analyze action frequency patterns and distributions.

**Outputs:**
- `action_patterns.csv`: Action statistics by type
  - action_type, frequency, percentage, mean_per_step

**Metrics:**
- Most/least common actions
- Action diversity
- Frequency distributions

### analyze_sequences

Detect and analyze action sequences and patterns.

**Outputs:**
- `sequence_patterns.csv`: Common action sequences
  - sequence, frequency, avg_reward, success_rate

**Metrics:**
- Common action chains
- Transition probabilities
- Sequence effectiveness

### analyze_decisions

Analyze decision-making patterns and quality.

**Outputs:**
- `decision_patterns.csv`: Decision analysis
  - state, action_taken, alternative_actions, optimality_score

**Metrics:**
- Decision diversity by state
- Optimal vs actual choices
- Decision consistency

### analyze_rewards

Analyze reward distributions and action performance.

**Outputs:**
- `reward_analysis.csv`: Reward statistics
  - action_type, mean_reward, std_reward, success_rate

**Metrics:**
- Reward by action type
- Risk/reward ratios
- Performance trends

---

## Visualization Functions

### plot_frequencies

Plot action frequency distributions.

**Output:** `action_frequencies.png`

### plot_sequences

Visualize action sequence patterns.

**Output:** `sequence_patterns.png`

### plot_decisions

Plot decision-making patterns and quality.

**Output:** `decision_patterns.png`

### plot_rewards

Visualize reward distributions by action.

**Output:** `reward_distributions.png`

---

## Function Groups

- **"all"**: All functions
- **"analysis"**: All analysis functions
- **"plots"**: All visualization functions
- **"basic"**: `analyze_patterns`, `plot_frequencies`
- **"sequences"**: Sequence-focused analysis
- **"performance"**: Reward and decision analysis

---

## Examples

### Action Frequency Analysis

```python
result = service.run(AnalysisRequest(
    module_name="actions",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/actions"),
    group="basic"
))

# Read action patterns
import pandas as pd
actions_df = pd.read_csv(result.output_path / "action_patterns.csv")
print(actions_df.sort_values('frequency', ascending=False).head())
```

### Sequence Mining

```python
result = service.run(AnalysisRequest(
    module_name="actions",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/sequences"),
    group="sequences",
    analysis_kwargs={
        "analyze_sequences": {
            "min_length": 3,
            "max_length": 5,
            "min_support": 10
        }
    }
))
```

### Performance by Agent Type

```python
result = service.run(AnalysisRequest(
    module_name="actions",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/performance"),
    group="performance",
    processor_kwargs={
        "group_by_agent_type": True
    }
))

# Compare performance
df = result.dataframe
for agent_type in df['agent_type'].unique():
    type_df = df[df['agent_type'] == agent_type]
    mean_reward = type_df['reward'].mean()
    print(f"{agent_type}: {mean_reward:.3f}")
```

---

## Integration Examples

### With Learning Module

```python
# Compare actions before/after learning
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

# Analyze action choices over time
# ... correlation analysis ...
```

---

## See Also

- [API Reference](../API_REFERENCE.md)
- [Learning Module](./Learning.md)
- [Agents Module](./Agents.md)

---

**Module Version**: 2.0.0  
**Last Updated**: 2025-10-04
