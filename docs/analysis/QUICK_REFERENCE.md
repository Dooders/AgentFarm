# Analysis Module Quick Reference

**Quick lookup guide for common tasks and APIs.**

---

## Common Tasks

### Running an Analysis

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

# Setup
service = AnalysisService(EnvConfigService())

# Run analysis
result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results")
))
```

### Listing Available Modules

```python
from farm.analysis.registry import list_modules, get_module_names

# Formatted listing
print(list_modules())

# Just names
modules = get_module_names()
```

### Batch Processing

```python
requests = [
    AnalysisRequest(module_name="population", ...),
    AnalysisRequest(module_name="resources", ...),
]

results = service.run_batch(requests)
```

### With Progress Tracking

```python
def progress(msg: str, pct: float):
    print(f"[{pct:.0%}] {msg}")

result = service.run(AnalysisRequest(
    module_name="population",
    experiment_path=Path("data"),
    output_path=Path("results"),
    progress_callback=progress
))
```

### Caching Control

```python
# Use cache (default)
request = AnalysisRequest(..., enable_caching=True)

# Force refresh
request = AnalysisRequest(..., force_refresh=True)

# Clear cache
service.clear_cache()
```

---

## Module Structure

```
farm/analysis/{module}/
├── module.py     # Module class (required)
├── data.py       # Data loading/processing
├── compute.py    # Statistical computations
├── analyze.py    # Analysis functions
└── plot.py       # Visualizations
```

---

## Creating a Module

### Minimal Module

```python
from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function

def process_data(path, **kwargs):
    # Load and process
    return pd.DataFrame(...)

def my_analysis(df, ctx, **kwargs):
    # Analyze data
    results = df.describe()
    
    # Save output
    output_file = ctx.get_output_file("results.csv")
    results.to_csv(output_file)

class MyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(
            name="my_module",
            description="My analysis"
        )
    
    def register_functions(self):
        self._functions = {
            "analyze": make_analysis_function(my_analysis)
        }
        self._groups = {
            "all": list(self._functions.values())
        }
    
    def get_data_processor(self):
        return SimpleDataProcessor(process_data)
```

### With Validation

```python
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

class MyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(...)
        
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['col1', 'col2'],
                column_types={'col1': int, 'col2': float}
            ),
            DataQualityValidator(
                min_rows=10,
                max_null_fraction=0.1
            )
        ])
        self.set_validator(validator)
```

---

## Analysis Functions

### Function Signature

```python
def my_function(
    df: pd.DataFrame,       # Input data
    ctx: AnalysisContext,   # Context with output_path, logger, etc.
    **kwargs: Any           # Optional parameters
) -> Optional[Any]:         # Can return results or None
    pass
```

### Common Patterns

```python
def my_analysis(df, ctx, **kwargs):
    # Log progress
    ctx.logger.info("Starting analysis")
    
    # Report progress
    ctx.report_progress("Processing", 0.5)
    
    # Get config
    threshold = ctx.get_config("threshold", default=0.5)
    
    # Save results
    output_file = ctx.get_output_file("results.csv")
    results.to_csv(output_file)
    
    # Save plot
    plot_file = ctx.get_output_file("chart.png", subdir="plots")
    fig.savefig(plot_file)
```

---

## Validation

### Column Validation

```python
from farm.analysis.validation import ColumnValidator

validator = ColumnValidator(
    required_columns=['col1', 'col2', 'col3'],
    column_types={
        'col1': int,
        'col2': float,
        'col3': str
    }
)

validator.validate(df)  # Raises DataValidationError if invalid
```

### Quality Validation

```python
from farm.analysis.validation import DataQualityValidator

validator = DataQualityValidator(
    min_rows=100,                    # Minimum 100 rows
    max_null_fraction=0.1,           # Max 10% nulls per column
    allow_duplicates=False,          # No duplicate rows
    value_ranges={'score': (0, 1)}   # Score must be 0-1
)

validator.validate(df)
```

### Custom Validation

```python
def custom_check(df):
    if df['value'].sum() == 0:
        raise DataValidationError("Sum cannot be zero")

validator = DataQualityValidator(
    custom_checks=[custom_check]
)
```

---

## Exception Handling

```python
from farm.analysis.exceptions import (
    AnalysisError,
    DataValidationError,
    ModuleNotFoundError,
    InsufficientDataError
)

try:
    result = service.run(request)
except ModuleNotFoundError as e:
    print(f"Module '{e.module_name}' not found")
    print(f"Available: {e.available_modules}")
except DataValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Missing columns: {e.missing_columns}")
    print(f"Invalid columns: {e.invalid_columns}")
except InsufficientDataError as e:
    print(f"Not enough data: {e}")
    print(f"Required: {e.required_rows}, Actual: {e.actual_rows}")
```

---

## Common Utilities

### Statistics

```python
from farm.analysis.common.utils import (
    calculate_statistics,
    calculate_trend,
    calculate_rolling_mean,
    normalize_dict
)

# Comprehensive stats
stats = calculate_statistics(data)  # mean, median, std, min, max, q25, q75

# Linear trend
trend = calculate_trend(data)  # positive = increasing

# Rolling average
smoothed = calculate_rolling_mean(data, window=5)

# Normalize to proportions
normalized = normalize_dict({'a': 10, 'b': 20, 'c': 30})
# {'a': 0.167, 'b': 0.333, 'c': 0.5}
```

### Correlation Analysis

```python
from farm.analysis.common.metrics import (
    analyze_correlations,
    find_top_correlations
)

# Analyze all correlations
corr = analyze_correlations(df, target_column='score')

# Find top 5 positive and negative correlations
top = find_top_correlations(
    df,
    target_column='score',
    top_n=5,
    min_correlation=0.3
)
print(f"Top positive: {top['top_positive']}")
print(f"Top negative: {top['top_negative']}")
```

### Group Comparison

```python
from farm.analysis.common.metrics import split_and_compare_groups

# Split by median and compare metrics
results = split_and_compare_groups(
    df,
    split_column='fitness',
    split_method='median',  # or 'mean'
    metrics=['score', 'survival_time']
)

for metric, stats in results['comparison_results'].items():
    print(f"{metric}:")
    print(f"  High group: {stats['high_group_mean']:.2f}")
    print(f"  Low group: {stats['low_group_mean']:.2f}")
    print(f"  Difference: {stats['difference']:.2f}")
    print(f"  % Difference: {stats['percent_difference']:.1f}%")
```

### Plotting Helpers

```python
from farm.analysis.common.utils import (
    setup_plot_figure,
    save_plot_figure,
    get_agent_type_colors
)

# Standard figure setup
fig, ax = setup_plot_figure()

# Agent type colors
colors = get_agent_type_colors()
ax.plot(data, color=colors['system'])

# Save with standard settings
save_plot_figure(fig, ctx.output_path, "chart.png")
```

---

## Available Modules

| Module | Name | Description |
|--------|------|-------------|
| **Population** | `"population"` | Population dynamics and composition |
| **Resources** | `"resources"` | Resource distribution and consumption |
| **Actions** | `"actions"` | Action patterns and success rates |
| **Agents** | `"agents"` | Individual agent behavior |
| **Learning** | `"learning"` | Learning performance and curves |
| **Spatial** | `"spatial"` | Spatial patterns and movement |
| **Temporal** | `"temporal"` | Temporal patterns and efficiency |
| **Combat** | `"combat"` | Combat metrics and patterns |
| **Dominance** | `"dominance"` | Dominance hierarchies (legacy) |
| **Genesis** | `"genesis"` | Initial population (legacy) |
| **Advantage** | `"advantage"` | Relative advantages (legacy) |
| **Social Behavior** | `"social_behavior"` | Social interactions (legacy) |
| **Significant Events** | `"significant_events"` | Event analysis (legacy) |
| **Comparative** | `"comparative"` | Cross-experiment comparison (legacy) |

---

## Function Groups

Most modules support these function groups:

- `"all"`: Run all functions
- `"plots"`: Only visualization functions
- `"metrics"`: Only computation/analysis functions
- `"basic"`: Basic/essential functions only

**Example:**
```python
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data"),
    output_path=Path("results"),
    group="plots"  # Only run plot functions
)
```

---

## Configuration

### Via Environment Variable

```bash
export FARM_ANALYSIS_MODULES="my_module.module.my_analysis_module,another.module"
```

### Programmatic Registration

```python
from farm.analysis.registry import registry

registry.register(my_module)
```

---

## Debugging

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Check Module Info

```python
from farm.analysis.registry import get_module

module = get_module("population")
info = module.get_info()

print(f"Name: {info['name']}")
print(f"Description: {info['description']}")
print(f"Functions: {info['functions']}")
print(f"Groups: {info['function_groups']}")
```

### Inspect Data Processor

```python
processor = module.get_data_processor()
df = processor.process(experiment_path)
print(f"Processed data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
```

### Test Validation

```python
validator = module.get_validator()
if validator:
    required = validator.get_required_columns()
    print(f"Required columns: {required}")
    
    try:
        validator.validate(df)
        print("✓ Data is valid")
    except DataValidationError as e:
        print(f"✗ Validation failed: {e}")
```

---

## Performance Tips

1. **Use caching** for repeated analyses
   ```python
   request = AnalysisRequest(..., enable_caching=True)
   ```

2. **Filter data early** in data processor
   ```python
   def process_data(path, filter_dead=True, **kwargs):
       df = load_data(path)
       if filter_dead:
           df = df[df['is_alive']]
       return df
   ```

3. **Run specific groups** instead of "all"
   ```python
   request = AnalysisRequest(..., group="metrics")  # Skip plots
   ```

4. **Use batch processing** for multiple experiments
   ```python
   results = service.run_batch(requests)
   ```

5. **Implement progress callbacks** for long-running analyses
   ```python
   def progress(msg, pct):
       print(f"\r{pct:.0%}: {msg}", end="")
   
   request = AnalysisRequest(..., progress_callback=progress)
   ```

---

## Common Patterns

### Load → Process → Validate → Analyze

```python
# 1. Load data
from farm.analysis.data.loaders import CSVLoader
loader = CSVLoader("data.csv")
df = loader.load_data()

# 2. Process
from farm.analysis.data.processors import DataCleaner
cleaner = DataCleaner(handle_missing=True)
df = cleaner.process(df)

# 3. Validate
from farm.analysis.validation import ColumnValidator
validator = ColumnValidator(required_columns=['col1', 'col2'])
validator.validate(df)

# 4. Analyze
stats = df.describe()
```

### Chained Processing

```python
from farm.analysis.core import ChainedDataProcessor
from farm.analysis.data.processors import (
    DataCleaner,
    TimeSeriesProcessor,
    AgentStatsProcessor
)

processor = ChainedDataProcessor([
    DataCleaner(handle_missing=True),
    TimeSeriesProcessor(smooth=True, window_size=5),
    AgentStatsProcessor(include_derived_metrics=True)
])

df = processor.process(raw_data)
```

### Composite Validation

```python
from farm.analysis.validation import CompositeValidator

validator = CompositeValidator([
    ColumnValidator(required_columns=[...]),
    DataQualityValidator(min_rows=10),
    CustomValidator()
])

validator.validate(df)  # Runs all validators
```

---

## Testing

### Test Your Module

```python
import pytest
from pathlib import Path
import pandas as pd
from farm.analysis.common.context import AnalysisContext

def test_my_module(tmp_path):
    # Setup
    module = MyModule()
    ctx = AnalysisContext(output_path=tmp_path)
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    
    # Test data processor
    processor = module.get_data_processor()
    assert processor is not None
    
    # Test validation
    validator = module.get_validator()
    if validator:
        validator.validate(df)
    
    # Test functions
    functions = module.get_analysis_functions("all")
    assert len(functions) > 0
    
    for func in functions:
        func(df, ctx)
    
    # Verify outputs
    assert (tmp_path / "results.csv").exists()
```

---

## See Also

- [API Reference](./API_REFERENCE.md) - Complete API documentation
- [Architecture](../../farm/analysis/ARCHITECTURE.md) - System architecture
- [README](../../farm/analysis/README.md) - User guide
- [Examples](../../examples/analysis_example.py) - Working examples

---

**Version**: 2.0.0  
**Last Updated**: 2025-10-04
