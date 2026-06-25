# Combat Analysis Module

**Module Name**: `combat`

Analyze combat metrics, win/loss ratios, damage patterns, and combat effectiveness.

---

## Overview

The Combat module analyzes agent combat interactions, tracking performance, strategies, and outcomes.

### Key Features

- Combat statistics
- Win/loss ratio analysis
- Damage patterns
- Combat effectiveness
- Strategy analysis
- Performance by agent type

---

## Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

result = service.run(AnalysisRequest(
    module_name="combat",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/combat")
))
```

---

## Data Requirements

### Required Columns

- `step` (int): Time step
- `attacker_id` (str): Attacker identifier
- `defender_id` (str): Defender identifier

### Optional Columns

- `damage` (float): Damage dealt
- `outcome` (str): Combat outcome ("win", "loss", "draw")
- `attacker_type` (str): Attacker agent type
- `defender_type` (str): Defender agent type
- `combat_duration` (int): Duration of combat

---

## Analysis Functions

### analyze_combat_statistics

Calculate overall combat statistics.

**Outputs:**
- `combat_statistics.csv`: Combat stats
- Total combats, win rates, damage dealt

### analyze_matchups

Analyze combat outcomes by agent type matchups.

**Outputs:**
- `matchup_analysis.csv`: Matchup statistics
- Win rates per matchup, damage ratios

### analyze_combat_effectiveness

Measure combat effectiveness metrics.

**Outputs:**
- `effectiveness_metrics.csv`: Effectiveness stats
- Kill/death ratios, damage efficiency

### analyze_strategy_patterns

Detect combat strategy patterns.

**Outputs:**
- `strategy_patterns.csv`: Strategy analysis
- Common patterns, success rates

---

## Visualization Functions

### plot_combat_statistics

Plot combat statistics over time.

**Output:** `combat_statistics.png`

### plot_matchup_matrix

Visualize matchup win rates.

**Output:** `matchup_matrix.png`

### plot_damage_distributions

Plot damage distributions.

**Output:** `damage_distributions.png`

### plot_effectiveness

Plot effectiveness metrics.

**Output:** `effectiveness_metrics.png`

---

## Examples

### Basic Combat Analysis

```python
result = service.run(AnalysisRequest(
    module_name="combat",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/combat")
))

import pandas as pd
stats_df = pd.read_csv(result.output_path / "combat_statistics.csv")
print(f"Total combats: {stats_df['total_combats'].sum()}")
print(f"Overall win rate: {stats_df['win_rate'].mean():.1%}")
```

### Matchup Analysis

```python
result = service.run(AnalysisRequest(
    module_name="combat",
    experiment_path=Path("data/experiment"),
    output_path=Path("results/matchups"),
    analysis_kwargs={
        "analyze_matchups": {
            "group_by_type": True,
            "min_combats": 10
        }
    }
))

matchups_df = pd.read_csv(result.output_path / "matchup_analysis.csv")
best_matchup = matchups_df.loc[matchups_df['win_rate'].idxmax()]
print(f"Best matchup: {best_matchup['attacker_type']} vs {best_matchup['defender_type']}")
print(f"Win rate: {best_matchup['win_rate']:.1%}")
```

---

## See Also

- [API Reference](../API_REFERENCE.md)
- [Agents Module](./Agents.md)
- [Actions Module](./Actions.md)

---

**Module Version**: 2.0.0
