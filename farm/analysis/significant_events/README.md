# Significant Events Analysis Module

A comprehensive analysis module for detecting, analyzing, and visualizing significant events in agent-based simulations.

## Overview

The Significant Events module automatically detects and analyzes critical moments in simulation runs, including population dynamics, resource crises, combat events, and individual agent milestones. It provides statistical analysis and visualization tools to help researchers understand the key turning points and patterns in their simulations.

## Features

### 🔍 Event Detection

Automatically detects 7 types of significant events from the simulation database:

- **Agent Deaths**: Individual agent mortality with impact based on type and generation
- **Agent Births**: Reproduction events tracking evolutionary progress
- **Population Crashes**: Rapid population decline (>30% decrease)
- **Population Booms**: Rapid population growth (>40% increase)
- **Critical Health Incidents**: Severe health drops (>50% loss or <20% remaining)
- **Mass Combat Events**: Large-scale conflicts (>20% population or >10 encounters)
- **Resource Depletion**: Critical resource shortages (>60% drop or <5 per agent)

### 📊 Statistical Analysis

- **Severity Scoring**: Automatic severity computation and categorization (low/medium/high)
- **Pattern Analysis**: Temporal patterns, event frequency, and inter-event timing
- **Impact Assessment**: Quantitative impact metrics by event type
- **Type Distribution**: Event type frequency and clustering

### 📈 Visualization

- **Event Timeline**: Chronological view of events with severity indicators
- **Severity Distribution**: Histograms and box plots of event severity
- **Impact Analysis**: Visual comparison of impact across event types
- **Comprehensive Plots**: Combined multi-panel visualization

## Installation

The module is part of the farm analysis framework and requires:

```bash
pip install sqlalchemy pandas numpy matplotlib seaborn
```

## Quick Start

```python
from farm.analysis.significant_events import (
    significant_events_module,
    detect_significant_events,
    compute_event_severity,
)
from farm.database.session_manager import SessionManager

# Connect to simulation database
session_manager = SessionManager('simulation.db')

# Detect events from simulation
events = detect_significant_events(
    session_manager,
    start_step=0,      # Optional: filter by time range
    end_step=1000,     # Optional: filter by time range
    min_severity=0.3   # Optional: minimum severity threshold
)

# Compute severity scores
events_with_severity = compute_event_severity(events)

# Filter high-severity events
critical_events = [
    e for e in events_with_severity 
    if e['severity_category'] == 'high'
]

print(f"Detected {len(critical_events)} critical events")
for event in critical_events:
    print(f"  Step {event['step']}: {event['type']} "
          f"(severity={event['severity']:.2f})")
```

## Usage

### Event Detection

#### Basic Detection

```python
from farm.analysis.significant_events import detect_significant_events

# Detect all events
events = detect_significant_events(db_connection)

# Filter by time range
events = detect_significant_events(
    db_connection,
    start_step=100,
    end_step=500
)

# Filter by minimum severity
events = detect_significant_events(
    db_connection,
    min_severity=0.5  # Only medium-high severity
)
```

#### Event Structure

Each detected event contains:

```python
{
    'type': str,           # Event type (e.g., 'population_crash')
    'step': int,           # Simulation step when event occurred
    'impact_scale': float, # Quantified impact (0.0-1.0)
    'details': dict,       # Event-specific information
}
```

### Analysis Functions

#### Analyze Events

```python
from farm.analysis.common.context import AnalysisContext
from farm.analysis.significant_events import analyze_significant_events

ctx = AnalysisContext(output_path='./analysis_output')

analyze_significant_events(
    ctx,
    db_connection=session_manager,
    min_severity=0.3
)
# Saves results to: analysis_output/significant_events.json
```

#### Analyze Patterns

```python
from farm.analysis.significant_events import analyze_event_patterns

analyze_event_patterns(
    ctx,
    db_connection=session_manager,
    min_severity=0.3
)
# Saves results to: analysis_output/event_patterns.json
```

#### Analyze Impact

```python
from farm.analysis.significant_events import analyze_event_impact

analyze_event_impact(
    ctx,
    db_connection=session_manager,
    min_severity=0.3
)
# Saves results to: analysis_output/event_impact.json
```

### Visualization

#### Timeline Plot

```python
from farm.analysis.significant_events import plot_event_timeline

plot_event_timeline(ctx)
# Saves plot to: analysis_output/event_timeline.png
```

#### Severity Distribution

```python
from farm.analysis.significant_events import plot_event_severity_distribution

plot_event_severity_distribution(ctx)
# Saves plot to: analysis_output/event_severity_distribution.png
```

#### Impact Analysis

```python
from farm.analysis.significant_events import plot_event_impact_analysis

plot_event_impact_analysis(ctx)
# Saves plot to: analysis_output/event_impact_analysis.png
```

#### Comprehensive Plots

```python
from farm.analysis.significant_events import plot_significant_events

plot_significant_events(ctx)
# Creates all three plots
```

### Using the Module API

```python
from farm.analysis.significant_events import significant_events_module

# Get available functions
functions = significant_events_module.get_function_names()
# Returns: ['analyze_events', 'analyze_patterns', 'analyze_impact', 
#           'plot_timeline', 'plot_severity', 'plot_impact']

# Get function groups
groups = significant_events_module.get_function_groups()
# Returns: ['all', 'analysis', 'plots', 'basic', 'patterns', 'impact']

# Run specific function
func = significant_events_module.get_function('analyze_events')
func(ctx, db_connection=session_manager)

# Run function group
for func in significant_events_module.get_functions_in_group('analysis'):
    func(ctx, db_connection=session_manager)
```

## Event Types

### Agent Deaths

**Detection Criteria**: Any agent with `death_time` set

**Impact Factors**:
- Base impact: 0.4
- System agents: +0.2 impact
- High generation (>5): +0.03 per generation (max +0.3)

**Details**:
```python
{
    'agent_id': str,
    'agent_type': str,
    'generation': int,
}
```

### Agent Births

**Detection Criteria**: Successful reproduction events

**Impact Factors**:
- Base impact: 0.2
- High generation (>5): +0.02 per generation (max +0.3)

**Details**:
```python
{
    'offspring_id': str,
    'parent_id': str,
    'generation': int,
}
```

### Population Crashes

**Detection Criteria**: Population decrease >30% in one step

**Impact**: Proportional to change rate (0.0-1.0)

**Details**:
```python
{
    'population_before': int,
    'population_after': int,
    'change_rate': float,
    'deaths': int,
}
```

### Population Booms

**Detection Criteria**: Population increase >40% in one step

**Impact**: 70% of change rate (booms less impactful than crashes)

**Details**:
```python
{
    'population_before': int,
    'population_after': int,
    'change_rate': float,
    'births': int,
}
```

### Critical Health Incidents

**Detection Criteria**: 
- Health drop >50%, OR
- Health falls below 20%

**Impact**: 0.4 + (drop_rate * 0.5), capped at 1.0

**Details**:
```python
{
    'agent_id': str,
    'health_before': float,
    'health_after': float,
    'cause': str,
    'drop_rate': float,
}
```

### Mass Combat Events

**Detection Criteria**:
- Combat rate >20% of population, OR
- More than 10 combat encounters

**Impact**: 0.5 + (combat_rate * 0.5), capped at 1.0

**Details**:
```python
{
    'combat_encounters': int,
    'successful_attacks': int,
    'total_agents': int,
    'combat_rate': float,
}
```

### Resource Depletion

**Detection Criteria**:
- Resource drop >60% in one step, OR
- Average resources per agent <5

**Impact**: Based on drop rate or scarcity level

**Details**:
```python
{
    'total_resources_before': float,
    'total_resources_after': float,
    'average_per_agent': float,
    'drop_rate': float,
}
```

## Severity Scoring

### Base Severity by Event Type

| Event Type | Base Severity | Typical Range |
|------------|---------------|---------------|
| Population Crash | 0.9 | 0.7-1.0 |
| Mass Combat | 0.8 | 0.5-1.0 |
| Resource Depletion | 0.8 | 0.5-1.0 |
| Health Critical | 0.7 | 0.4-0.9 |
| Population Boom | 0.6 | 0.4-0.7 |
| Agent Death | 0.5 | 0.4-0.6 |
| Agent Birth | 0.3 | 0.2-0.5 |

### Severity Categories

- **High Severity** (>0.7): Critical events requiring immediate attention
- **Medium Severity** (0.4-0.7): Important events with notable impact
- **Low Severity** (≤0.4): Minor events or routine occurrences

### Severity Computation

```python
severity = base_severity * impact_scale
severity = min(1.0, severity)  # Cap at 1.0
```

## Analysis Outputs

### Event Analysis Output

File: `significant_events.json`

```json
{
  "total_events_detected": 150,
  "significant_events": 87,
  "min_severity_threshold": 0.3,
  "events": [
    {
      "type": "population_crash",
      "step": 450,
      "impact_scale": 0.85,
      "severity": 0.765,
      "severity_category": "high",
      "details": {
        "population_before": 200,
        "population_after": 30,
        "change_rate": 0.85,
        "deaths": 170
      }
    }
  ]
}
```

### Pattern Analysis Output

File: `event_patterns.json`

```json
{
  "event_frequency": {
    "mean": 2.5,
    "std": 1.8,
    "min": 1,
    "max": 8
  },
  "inter_event_times": {
    "mean": 15.3,
    "std": 12.7,
    "min": 1,
    "max": 89
  },
  "event_types": {
    "agent_death": 45,
    "agent_birth": 32,
    "population_crash": 5,
    "mass_combat": 3,
    "resource_depletion": 2
  },
  "severity_distribution": {
    "mean": 0.52,
    "std": 0.18,
    "min": 0.21,
    "max": 0.95
  }
}
```

### Impact Analysis Output

File: `event_impact.json`

```json
{
  "impact_by_type": {
    "population_crash": {
      "mean": 0.812,
      "std": 0.145,
      "count": 5
    },
    "agent_death": {
      "mean": 0.487,
      "std": 0.092,
      "count": 45
    }
  },
  "overall_impact": {
    "mean": 0.634,
    "std": 0.203,
    "min": 0.210,
    "max": 0.950
  }
}
```

## Advanced Usage

### Custom Event Filtering

```python
# Get only population-related events
population_events = [
    e for e in events 
    if e['type'] in ['population_crash', 'population_boom']
]

# Get events in specific time window
early_events = [e for e in events if e['step'] < 500]
late_events = [e for e in events if e['step'] >= 500]

# Get events affecting specific agent types
system_deaths = [
    e for e in events 
    if e['type'] == 'agent_death' 
    and e['details'].get('agent_type') == 'system'
]
```

### Event Correlation Analysis

```python
import pandas as pd

# Convert to DataFrame for analysis
df = pd.DataFrame(events_with_severity)

# Find events that occur together
df['step_bin'] = df['step'] // 10  # Group by 10-step bins
correlations = df.groupby('step_bin')['type'].value_counts()

# Identify crisis periods (multiple high-severity events)
high_severity = df[df['severity_category'] == 'high']
crisis_steps = high_severity.groupby('step').size()
crisis_periods = crisis_steps[crisis_steps > 2]  # 3+ critical events
```

### Time Series Analysis

```python
# Event rate over time
event_counts = df.groupby('step').size()
rolling_avg = event_counts.rolling(window=50).mean()

# Severity trends
severity_by_step = df.groupby('step')['severity'].mean()
```

## Integration with Other Modules

### With Population Analysis

```python
from farm.analysis.population import population_module

# Run both analyses
population_module.run_group(ctx, 'all', db_connection=db)
significant_events_module.run_group(ctx, 'all', db_connection=db)

# Compare population metrics with events
# (Cross-reference population.json with significant_events.json)
```

### With Combat Analysis

```python
from farm.analysis.combat import combat_module

# Analyze combat in detail
combat_module.run_group(ctx, 'all', db_connection=db)

# Compare with mass combat events
mass_combat_events = [e for e in events if e['type'] == 'mass_combat']
```

### With Experiment Runner

```python
from farm.runners.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    config=experiment_config,
    db_path='experiment.db'
)

# Events are automatically detected during experiment runs
runner.run_experiment(
    iterations=10,
    analysis_modules=['significant_events']
)
```

## Performance Considerations

### Database Optimization

- Events are detected using indexed queries on:
  - `agents.death_time`
  - `reproduction_events.step_number`
  - `simulation_steps.step_number`
  - `health_incidents.step_number`

- For large simulations (>10,000 steps), consider:
  - Using time range filters (`start_step`, `end_step`)
  - Analyzing in batches
  - Setting higher severity thresholds

### Memory Usage

```python
# For very large event sets, process in chunks
chunk_size = 1000
for i in range(0, len(events), chunk_size):
    chunk = events[i:i+chunk_size]
    chunk_with_severity = compute_event_severity(chunk)
    # Process chunk...
```

## Troubleshooting

### No Events Detected

**Possible causes**:
- `min_severity` threshold too high
- Simulation too stable (no significant changes)
- Database connection issues
- Insufficient data in time range

**Solutions**:
```python
# Lower severity threshold
events = detect_significant_events(db, min_severity=0.1)

# Check database has data
print(f"Steps in database: {db.get_step_count()}")

# Try full time range
events = detect_significant_events(db, start_step=0)
```

### Too Many Events

**Possible causes**:
- `min_severity` threshold too low
- Very dynamic simulation
- Including low-impact events

**Solutions**:
```python
# Increase severity threshold
events = detect_significant_events(db, min_severity=0.5)

# Filter after detection
high_impact = [e for e in events if e['impact_scale'] > 0.7]
```

### Performance Issues

**Solutions**:
```python
# Use time range filtering
events = detect_significant_events(
    db, 
    start_step=recent_start,
    end_step=recent_end
)

# Process in parallel (for multiple simulations)
from multiprocessing import Pool

def analyze_simulation(sim_id):
    db = get_db_connection(sim_id)
    return detect_significant_events(db)

with Pool() as pool:
    all_events = pool.map(analyze_simulation, simulation_ids)
```

## API Reference

### Module Functions

```python
# Detection
detect_significant_events(db_connection, start_step=0, end_step=None, min_severity=0.3)

# Computation
compute_event_severity(events: List[Dict]) -> List[Dict]
compute_event_patterns(events: List[Dict]) -> Dict
compute_event_impact(events: List[Dict]) -> Dict

# Analysis
analyze_significant_events(ctx: AnalysisContext, **kwargs) -> None
analyze_event_patterns(ctx: AnalysisContext, **kwargs) -> None
analyze_event_impact(ctx: AnalysisContext, **kwargs) -> None

# Visualization
plot_event_timeline(ctx: AnalysisContext, **kwargs) -> None
plot_event_severity_distribution(ctx: AnalysisContext, **kwargs) -> None
plot_event_impact_analysis(ctx: AnalysisContext, **kwargs) -> None
plot_significant_events(ctx: AnalysisContext, **kwargs) -> None
```

### Module Object

```python
significant_events_module: SignificantEventsModule
    .name -> str
    .description -> str
    .get_function_names() -> List[str]
    .get_function_groups() -> List[str]
    .get_function(name: str) -> Callable
    .get_functions_in_group(group: str) -> List[Callable]
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/analysis/test_significant_events.py -v

# Specific test class
pytest tests/analysis/test_significant_events.py::TestSignificantEventsComputations -v

# Specific test
pytest tests/analysis/test_significant_events.py::test_detect_agent_deaths -v
```

## Contributing

When adding new event types:

1. Create detection function in `compute.py`:
   ```python
   def _detect_new_event_type(query_func, start_step, end_step):
       """Detect new event type."""
       # Implementation
       return events
   ```

2. Add to `detect_significant_events()`:
   ```python
   events.extend(_detect_new_event_type(query_func, start_step, end_step))
   ```

3. Update severity scoring in `compute_event_severity()`:
   ```python
   base_severity = {
       # ...existing types...
       "new_event_type": 0.7,
   }
   ```

4. Add tests in `test_significant_events.py`

## License

Part of the farm simulation analysis framework.

## Related Documentation

- [Analysis Architecture](../ARCHITECTURE.md)
- [Database Schema](../../../docs/data/database_schema.md)
- [Module Overview](../../../docs/analysis/modules/README.md)
- [Experiment Analysis](../../../docs/experiment_analysis.md)

## Support

For issues, questions, or contributions, please refer to the main project repository.
