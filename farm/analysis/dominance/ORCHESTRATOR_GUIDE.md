# Dominance Analysis Orchestrator Guide

## Overview

The `DominanceAnalysisOrchestrator` provides a unified, protocol-based API for dominance analysis operations. It manages the coordination between computation, analysis, and data provider components while eliminating circular dependencies through runtime dependency injection.

---

## Quick Start

### Simple Usage (Recommended)

```python
from farm.analysis.dominance import get_orchestrator

# Get pre-configured orchestrator
orchestrator = get_orchestrator()

# Compute dominance metrics
population_dom = orchestrator.compute_population_dominance(session)
survival_dom = orchestrator.compute_survival_dominance(session)
comprehensive = orchestrator.compute_comprehensive_dominance(session)

# Analyze DataFrame
df = orchestrator.analyze_dataframe_comprehensively(df)
```

### Factory Pattern

```python
from farm.analysis.dominance import create_dominance_orchestrator

# Create with default implementations
orchestrator = create_dominance_orchestrator()

# Or with custom implementations
from my_custom import MyCustomComputer
orchestrator = create_dominance_orchestrator(
    custom_computer=MyCustomComputer()
)
```

---

## API Reference

### Computation Methods

#### `compute_population_dominance(sim_session)`
Compute the dominant agent type by final population.

**Parameters:**
- `sim_session`: SQLAlchemy database session

**Returns:** `Optional[str]` - Dominant agent type or None

**Example:**
```python
orchestrator = get_orchestrator()
dominant_type = orchestrator.compute_population_dominance(session)
print(f"Population dominant: {dominant_type}")
```

#### `compute_survival_dominance(sim_session)`
Compute the dominant agent type by average survival time.

**Parameters:**
- `sim_session`: SQLAlchemy database session

**Returns:** `Optional[str]` - Dominant agent type or None

#### `compute_comprehensive_dominance(sim_session)`
Compute comprehensive dominance scores using multiple metrics.

**Parameters:**
- `sim_session`: SQLAlchemy database session

**Returns:** `Optional[Dict[str, Any]]` - Dictionary with dominance scores

**Example:**
```python
result = orchestrator.compute_comprehensive_dominance(session)
print(f"Dominant type: {result['dominant_type']}")
print(f"Scores: {result['scores']}")
```

#### `compute_dominance_switches(sim_session)`
Analyze dominance switching patterns during simulation.

**Parameters:**
- `sim_session`: SQLAlchemy database session

**Returns:** `Optional[Dict[str, Any]]` - Dictionary with switching statistics

**Example:**
```python
switches = orchestrator.compute_dominance_switches(session)
print(f"Total switches: {switches['total_switches']}")
print(f"Switches per step: {switches['switches_per_step']}")
```

#### `compute_dominance_switch_factors(df)`
Calculate factors that correlate with dominance switching.

**Parameters:**
- `df`: pandas DataFrame with simulation results

**Returns:** `Optional[Dict[str, Any]]` - Correlation analysis results

---

### Analysis Methods

#### `analyze_by_agent_type(df, numeric_repro_cols)`
Analyze reproduction metrics by dominant agent type.

**Parameters:**
- `df`: pandas DataFrame with simulation results
- `numeric_repro_cols`: List of numeric reproduction column names

**Returns:** `pd.DataFrame` - Input DataFrame with added analysis columns

#### `analyze_high_vs_low_switching(df, numeric_repro_cols)`
Compare reproduction metrics between high and low switching groups.

#### `analyze_reproduction_advantage(df, numeric_repro_cols)`
Analyze reproduction advantage and dominance switching.

#### `analyze_reproduction_efficiency(df, numeric_repro_cols)`
Analyze reproduction efficiency correlation with dominance stability.

#### `analyze_reproduction_timing(df, numeric_repro_cols)`
Analyze first reproduction timing and dominance switching.

#### `analyze_dominance_switch_factors(df)`
Analyze factors correlating with dominance switching patterns.

#### `analyze_reproduction_dominance_switching(df)`
Analyze relationship between reproduction and dominance switching.

---

### Data Provider Methods

#### `get_final_population_counts(sim_session)`
Get final population counts by agent type.

**Returns:** `Optional[Dict[str, int]]`

#### `get_agent_survival_stats(sim_session)`
Get agent survival statistics by type.

**Returns:** `Optional[Dict[str, Any]]`

#### `get_reproduction_stats(sim_session)`
Get reproduction statistics by agent type.

**Returns:** `Optional[Dict[str, Any]]`

#### `get_initial_positions_and_resources(sim_session, config)`
Get initial positioning and resource data.

**Returns:** `Optional[Dict[str, Any]]`

---

### High-Level Orchestration Methods

#### `run_full_analysis(sim_session, config)`
Run complete dominance analysis workflow.

**Parameters:**
- `sim_session`: SQLAlchemy database session
- `config`: Simulation configuration dictionary

**Returns:** `Dict[str, Any]` - Complete analysis results

**Example:**
```python
orchestrator = get_orchestrator()
results = orchestrator.run_full_analysis(session, config)

# Access comprehensive results
print(f"Population dominance: {results['population_dominance']}")
print(f"Survival dominance: {results['survival_dominance']}")
print(f"Switches: {results['dominance_switches']['total_switches']}")
print(f"Initial data: {results['initial_data']}")
```

#### `analyze_dataframe_comprehensively(df, numeric_repro_cols=None)`
Run comprehensive analysis on simulation results DataFrame.

**Parameters:**
- `df`: pandas DataFrame with simulation results
- `numeric_repro_cols`: Optional list of numeric reproduction columns (auto-detected if None)

**Returns:** `pd.DataFrame` - Input DataFrame with all analysis columns

**Example:**
```python
orchestrator = get_orchestrator()

# Auto-detect reproduction columns
df = orchestrator.analyze_dataframe_comprehensively(df)

# Or specify columns explicitly
numeric_cols = ['system_reproduction_success_rate', 'independent_reproduction_success_rate']
df = orchestrator.analyze_dataframe_comprehensively(df, numeric_cols)
```

---

## Advanced Usage

### Custom Implementations

You can provide custom implementations of any protocol:

```python
from farm.analysis.dominance import create_dominance_orchestrator
from farm.analysis.dominance.interfaces import DominanceComputerProtocol

class MyCustomComputer:
    """Custom computer with additional logging."""
    
    def compute_population_dominance(self, sim_session):
        print("Computing population dominance with custom logic...")
        # Your custom implementation
        return "system"
    
    # Implement other protocol methods...

# Use custom implementation
orchestrator = create_dominance_orchestrator(
    custom_computer=MyCustomComputer()
)
```

### Direct Class Instantiation

For maximum control, instantiate classes directly:

```python
from farm.analysis.dominance import (
    DominanceAnalysisOrchestrator,
    DominanceComputer,
    DominanceAnalyzer,
    DominanceDataProvider
)

# Create components
computer = DominanceComputer()
analyzer = DominanceAnalyzer()
data_provider = DominanceDataProvider()

# Create orchestrator
orchestrator = DominanceAnalysisOrchestrator(
    computer=computer,
    analyzer=analyzer,
    data_provider=data_provider
)
```

### Testing with Mocks

The orchestrator makes testing easy with protocol-based mocks:

```python
from unittest.mock import Mock
from farm.analysis.dominance import DominanceAnalysisOrchestrator

# Create mock computer
mock_computer = Mock()
mock_computer.compute_population_dominance.return_value = "system"
mock_computer.compute_survival_dominance.return_value = "independent"

# Create orchestrator with mock
orchestrator = DominanceAnalysisOrchestrator(computer=mock_computer)

# Test
result = orchestrator.compute_population_dominance(session)
assert result == "system"
mock_computer.compute_population_dominance.assert_called_once_with(session)
```

---

## Complete Workflow Example

```python
from farm.analysis.dominance import get_orchestrator
import pandas as pd

# Initialize orchestrator
orchestrator = get_orchestrator()

# 1. Compute dominance metrics from simulation
results = orchestrator.run_full_analysis(session, config)

# 2. Create DataFrame from results
data = []
for sim_id, session in simulation_sessions.items():
    row = {
        'simulation_id': sim_id,
        'population_dominance': orchestrator.compute_population_dominance(session),
        'survival_dominance': orchestrator.compute_survival_dominance(session),
    }
    
    # Add comprehensive dominance
    comp_dom = orchestrator.compute_comprehensive_dominance(session)
    if comp_dom:
        row['comprehensive_dominance'] = comp_dom['dominant_type']
        row.update(comp_dom['scores'])
    
    # Add switching metrics
    switches = orchestrator.compute_dominance_switches(session)
    if switches:
        row['total_switches'] = switches['total_switches']
        row['switches_per_step'] = switches['switches_per_step']
    
    # Add data provider metrics
    row.update(orchestrator.get_agent_survival_stats(session) or {})
    row.update(orchestrator.get_reproduction_stats(session) or {})
    
    data.append(row)

df = pd.DataFrame(data)

# 3. Analyze DataFrame comprehensively
df = orchestrator.analyze_dataframe_comprehensively(df)

# 4. Access results
print(f"Total simulations: {len(df)}")
print(f"Columns added: {len(df.columns)}")
print(df[['simulation_id', 'population_dominance', 'total_switches']].head())
```

---

## Best Practices

### 1. Use the Default Orchestrator
For most use cases, use `get_orchestrator()`:
```python
from farm.analysis.dominance import get_orchestrator
orchestrator = get_orchestrator()
```

### 2. Reuse Orchestrator Instances
Create once, use many times:
```python
# Good - reuse instance
orchestrator = get_orchestrator()
for session in sessions:
    result = orchestrator.compute_population_dominance(session)

# Avoid - creates new instance each time
for session in sessions:
    orchestrator = create_dominance_orchestrator()  # Wasteful
    result = orchestrator.compute_population_dominance(session)
```

### 3. Use High-Level Methods
Prefer orchestration methods over manual coordination:
```python
# Good - orchestrated workflow
results = orchestrator.run_full_analysis(session, config)

# Avoid - manual coordination (more error-prone)
pop_dom = orchestrator.compute_population_dominance(session)
surv_dom = orchestrator.compute_survival_dominance(session)
comp_dom = orchestrator.compute_comprehensive_dominance(session)
# ... manually combining results
```

### 4. Let Auto-Detection Work
The orchestrator can auto-detect many parameters:
```python
# Good - auto-detect reproduction columns
df = orchestrator.analyze_dataframe_comprehensively(df)

# Only specify if you need precise control
df = orchestrator.analyze_dataframe_comprehensively(df, specific_columns)
```

---

## Migration from Legacy API

See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for detailed instructions on migrating from the function-based API to the orchestrator.

---

## Architecture

The orchestrator uses runtime dependency injection to eliminate circular dependencies:

```
┌─────────────────────────────────────────────────┐
│ DominanceAnalysisOrchestrator                   │
│                                                  │
│  Components:                                     │
│  ├── computer: DominanceComputer                │
│  ├── analyzer: DominanceAnalyzer                │
│  └── data_provider: DominanceDataProvider       │
│                                                  │
│  Wiring (at runtime):                            │
│  ├── computer.analyzer = analyzer               │
│  └── analyzer.computer = computer               │
│                                                  │
│  Methods: 19 delegation + 2 orchestration        │
└─────────────────────────────────────────────────┘
```

This design:
- ✅ Eliminates circular import dependencies
- ✅ Enables independent testing of components
- ✅ Supports custom implementations via protocols
- ✅ Maintains clean separation of concerns

---

## Troubleshooting

### Issue: "No analyzer injected" warning
**Cause:** Computer trying to use analyzer methods without injection  
**Solution:** Use orchestrator instead of creating components directly

### Issue: Results missing expected fields
**Cause:** Using partial analysis methods  
**Solution:** Use `run_full_analysis()` or `analyze_dataframe_comprehensively()`

### Issue: Auto-detection not finding reproduction columns
**Cause:** Column names don't contain "reproduction"  
**Solution:** Explicitly pass `numeric_repro_cols` parameter

---

## See Also

- [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) - Migration from legacy API
- [interfaces.py](./interfaces.py) - Protocol definitions
- [orchestrator.py](./orchestrator.py) - Implementation details
