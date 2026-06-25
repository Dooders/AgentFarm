# Population Analysis Module

**Module Name**: `population`

Analyze population dynamics, births, deaths, and agent composition in simulations.

---

## Overview

The Population module tracks and analyzes how agent populations change over time, including birth and death rates, population growth patterns, and species composition.

### Key Features

- Population dynamics over time
- Birth and death rate analysis
- Agent composition by type
- Growth rate calculations
- Population stability metrics

---

## Installation & Usage

### Basic Usage

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

# Initialize service
service = AnalysisService(EnvConfigService())

# Run population analysis
result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/population")
))
```

### With Function Groups

```python
# Run only plots
result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results"),
    group="plots"  # Options: "all", "analysis", "plots", "basic"
))
```

---

## Data Requirements

### Required Columns

- `step` (int): Simulation step/iteration number
- `total_agents` (int): Total number of agents at this step

### Optional Columns

- `agent_type` (str): Type of agent (e.g., "system", "independent")
- `births` (int): Number of births at this step
- `deaths` (int): Number of deaths at this step
- `birth_rate` (float): Birth rate (births per capita)
- `death_rate` (float): Death rate (deaths per capita)
- `growth_rate` (float): Population growth rate

### Example Data Format

```csv
step,total_agents,agent_type,births,deaths,birth_rate,death_rate
0,100,system,0,0,0.0,0.0
1,105,system,8,3,0.08,0.03
2,110,system,9,4,0.086,0.038
0,50,independent,0,0,0.0,0.0
1,52,independent,4,2,0.08,0.04
```

---

## Analysis Functions

### analyze_dynamics

Analyze population dynamics and calculate statistics.

**Outputs:**
- `population_dynamics.csv`: Time series statistics
  - step, mean, std, min, max, growth_rate

**Metrics Computed:**
- Mean population per step
- Population variance
- Growth rates
- Stability metrics

### analyze_composition

Analyze agent composition by type.

**Outputs:**
- `agent_composition.csv`: Composition breakdown
  - agent_type, count, percentage, mean, std

**Metrics Computed:**
- Population by agent type
- Proportions over time
- Dominant species

---

## Visualization Functions

### plot_population

Plot population over time.

**Output:** `population_over_time.png`

**Features:**
- Line plot of total population
- Separate lines for each agent type (if available)
- Trend lines
- Confidence intervals

**Customization:**
```python
result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data"),
    output_path=Path("results"),
    analysis_kwargs={
        "plot_population": {
            "show_confidence": True,
            "show_trend": True
        }
    }
))
```

### plot_births_deaths

Plot birth and death rates over time.

**Output:** `birth_death_rates.png`

**Features:**
- Dual-axis plot (births and deaths)
- Rolling averages
- Rate comparisons

### plot_composition

Plot agent composition over time.

**Output:** `agent_composition.png`

**Features:**
- Stacked area chart
- Population proportions
- Color-coded by agent type

---

## Function Groups

### "all"
Run all analysis and visualization functions.

### "analysis"
Run only analysis functions:
- `analyze_dynamics`
- `analyze_composition`

### "plots"
Run only visualization functions:
- `plot_population`
- `plot_births_deaths`
- `plot_composition`

### "basic"
Run essential functions:
- `analyze_dynamics`
- `plot_population`

---

## Output Files

After running the analysis, the following files are created:

```
output_path/
├── population_dynamics.csv       # Dynamics statistics
├── agent_composition.csv         # Composition breakdown
├── population_over_time.png      # Population plot
├── birth_death_rates.png         # Birth/death rates
├── agent_composition.png         # Composition plot
└── analysis_summary.json         # Metadata and timing
```

---

## Examples

### Basic Analysis

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

# Run full population analysis
result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/my_experiment"),
    output_path=Path("results/population")
))

if result.success:
    print(f"Analysis completed in {result.execution_time:.2f}s")
    print(f"Data shape: {result.dataframe.shape}")
else:
    print(f"Analysis failed: {result.error}")
```

### With Progress Tracking

```python
def progress_callback(message: str, progress: float):
    print(f"[{progress:.0%}] {message}")

result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results"),
    progress_callback=progress_callback
))
```

### Batch Analysis

```python
# Analyze multiple experiments
requests = [
    AnalysisRequest(
        module_name="population",
        experiment_path=Path(f"data/experiment_{i:03d}"),
        output_path=Path(f"results/exp_{i:03d}")
    )
    for i in range(10)
]

results = service.run_batch(requests)

# Summarize results
successful = sum(1 for r in results if r.success)
print(f"Completed {successful}/{len(results)} analyses")
```

### Custom Data Processing

```python
# Filter to specific agent types
result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results"),
    processor_kwargs={
        "filter_types": ["system", "independent"]
    }
))
```

---

## Integration with Other Modules

### Combined with Resources

```python
# Analyze population and resources together
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

# Compare population and resource trends
pop_df = pop_result.dataframe
res_df = res_result.dataframe
```

---

## Performance Tips

1. **Use caching** for repeated analyses:
   ```python
   request = AnalysisRequest(..., enable_caching=True)
   ```

2. **Filter data early** in processing:
   ```python
   processor_kwargs={"start_step": 100, "end_step": 1000}
   ```

3. **Run specific groups** for faster execution:
   ```python
   group="basic"  # Skip detailed plots
   ```

4. **Batch multiple experiments**:
   ```python
   results = service.run_batch(requests)
   ```

---

## Troubleshooting

### Common Issues

**"Missing required columns"**
- Ensure your data has `step` and `total_agents` columns
- Check column names match exactly (case-sensitive)

**"Insufficient data for analysis"**
- Need at least 1 row of data
- Check that data is not empty after filtering

**"Analysis function failed"**
- Check logs for specific error
- Verify data types are correct
- Ensure no NaN values in required columns

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run analysis with verbose output
result = service.run(request)
```

---

## API Reference

### PopulationModule

```python
from farm.analysis.population.module import population_module

# Get module info
info = population_module.get_info()

# Get function names
functions = population_module.get_function_names()

# Get function groups
groups = population_module.get_function_groups()

# Run analysis directly
output_path, df = population_module.run_analysis(
    experiment_path=Path("data"),
    output_path=Path("results")
)
```

---

## See Also

- [API Reference](../API_REFERENCE.md) - Complete API documentation
- [Agents Module](./Agents.md) - Individual agent analysis
- [Resources Module](./Resources.md) - Resource analysis
- [Quick Reference](../QUICK_REFERENCE.md) - Common patterns

---

**Module Version**: 2.0.0  
**Last Updated**: 2025-10-04
