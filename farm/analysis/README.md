## Analysis Module System

A modern, protocol-based architecture for creating and running analysis modules on simulation data.

### Features

✨ **Protocol-Based Architecture** - Type-safe interfaces using Python protocols  
🔌 **Plugin System** - Dynamic module discovery and registration  
✅ **Comprehensive Validation** - Built-in data validation with custom validators  
💾 **Smart Caching** - Automatic caching of analysis results  
📊 **Progress Tracking** - Real-time progress callbacks  
🎯 **Standardized Functions** - Consistent API across all analysis types  
🧪 **Fully Tested** - Comprehensive test suite with >80% coverage  
🔧 **Shared Utilities** - Common statistical and data processing functions  
📋 **Standard Templates** - Consistent module structure templates  

### Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

# Initialize service
config_service = EnvConfigService()
service = AnalysisService(config_service)

# Create analysis request
request = AnalysisRequest(
    module_name="dominance",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/dominance_analysis"),
    group="basic"  # Run only basic plots
)

# Run analysis
result = service.run(request)

if result.success:
    print(f"Analysis complete in {result.execution_time:.2f}s")
    print(f"Results saved to: {result.output_path}")
else:
    print(f"Analysis failed: {result.error}")
```

### Common Utilities

The `farm.analysis.common.utils` module provides shared utilities for analysis:

```python
from farm.analysis.common.utils import (
    calculate_statistics,      # Basic stats (mean, median, std, etc.)
    calculate_trend,          # Linear trend calculation
    calculate_rolling_mean,   # Rolling averages
    normalize_dict,           # Normalize dictionary values to proportions
    validate_required_columns, # Check DataFrame has required columns
    save_analysis_results,    # Save results to JSON
    setup_plot_figure,        # Consistent matplotlib setup
    get_agent_type_colors,    # Standard agent type color scheme
    handle_missing_data,      # Handle NaN values with different strategies
)

# Example usage
import pandas as pd
from farm.analysis.common.utils import calculate_statistics, normalize_dict

# Calculate stats for a data series
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
stats = calculate_statistics(data)
print(f"Mean: {stats['mean']}, Std: {stats['std']}")

# Normalize agent counts
agent_counts = {'system': 25, 'independent': 35, 'control': 40}
normalized = normalize_dict(agent_counts)
print(f"Normalized: {normalized}")
```

### Standard Module Template

Use the standard template for consistent module structure:

```python
from farm.analysis.template.standard_module import MODULEModule

# Copy template and replace 'MODULE' with your analysis type
# Template includes:
# - Data processing function
# - Analysis functions with standard signatures
# - Visualization functions
# - Complete module class with validation
# - Function registration and grouping

class PopulationModule(MODULEModule):
    # Customize as needed
    pass
```

### Creating a New Analysis Module

1. **Define your analysis functions**:

```python
# my_analysis/compute.py
import pandas as pd
from farm.analysis.common.context import AnalysisContext

def analyze_population_dynamics(df: pd.DataFrame, ctx: AnalysisContext) -> None:
    """Analyze population dynamics over time."""
    # Your analysis logic here
    ctx.logger.info("Analyzing population dynamics")
    
    # Save results
    output_file = ctx.get_output_file("population_stats.csv")
    stats = df.groupby('iteration')['population'].agg(['mean', 'std', 'max'])
    stats.to_csv(output_file)
    
    ctx.report_progress("Population analysis complete", 0.5)
```

2. **Create visualization functions**:

```python
# my_analysis/plot.py
import matplotlib.pyplot as plt
import pandas as pd
from farm.analysis.common.context import AnalysisContext

def plot_population_over_time(df: pd.DataFrame, ctx: AnalysisContext) -> None:
    """Plot population trends."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for agent_type in df['agent_type'].unique():
        data = df[df['agent_type'] == agent_type]
        ax.plot(data['iteration'], data['population'], label=agent_type)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Population')
    ax.legend()
    ax.set_title('Population Dynamics')
    
    output_file = ctx.get_output_file("population_trends.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
```

3. **Create data processor**:

```python
# my_analysis/processor.py
import pandas as pd
from pathlib import Path

def process_my_data(experiment_path: Path, **kwargs) -> pd.DataFrame:
    """Process raw data for analysis."""
    # Load and process your data
    data = []
    
    for sim_dir in experiment_path.glob("iteration_*"):
        # Load simulation data
        sim_data = load_simulation(sim_dir)
        data.append(sim_data)
    
    return pd.DataFrame(data)
```

4. **Create module class**:

```python
# my_analysis/module.py
from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator
from my_analysis.processor import process_my_data
from my_analysis.compute import analyze_population_dynamics
from my_analysis.plot import plot_population_over_time

class MyAnalysisModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(
            name="my_analysis",
            description="Custom analysis of simulation dynamics"
        )
        
        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['iteration', 'agent_type', 'population'],
                column_types={'iteration': int, 'population': float}
            ),
            DataQualityValidator(min_rows=1)
        ])
        self.set_validator(validator)
    
    def register_functions(self) -> None:
        """Register analysis functions."""
        self._functions = {
            "analyze_population": make_analysis_function(analyze_population_dynamics),
            "plot_population": make_analysis_function(plot_population_over_time),
        }
        
        self._groups = {
            "all": list(self._functions.values()),
            "plots": [self._functions["plot_population"]],
            "metrics": [self._functions["analyze_population"]],
        }
    
    def get_data_processor(self) -> SimpleDataProcessor:
        """Get data processor."""
        return SimpleDataProcessor(process_my_data)

# Create singleton
my_analysis_module = MyAnalysisModule()
```

5. **Register your module**:

```bash
# Via environment variable
export FARM_ANALYSIS_MODULES="my_analysis.module.my_analysis_module"

# Or programmatically
from farm.analysis.registry import registry
from my_analysis.module import my_analysis_module

registry.register(my_analysis_module)
```

### Advanced Features

#### Custom Validation

```python
from farm.analysis.validation import ColumnValidator, DataQualityValidator

# Column validation
validator = ColumnValidator(
    required_columns=['iteration', 'agent_type', 'score'],
    column_types={
        'iteration': int,
        'score': float,
        'agent_type': str
    }
)

# Data quality validation
quality_validator = DataQualityValidator(
    min_rows=10,
    max_null_fraction=0.1,  # Max 10% nulls per column
    allow_duplicates=False,
    value_ranges={'score': (0.0, 1.0)}  # Score must be 0-1
)

# Combine validators
from farm.analysis.validation import CompositeValidator
combined = CompositeValidator([validator, quality_validator])
```

#### Progress Tracking

```python
def progress_callback(message: str, progress: float):
    print(f"[{progress:.0%}] {message}")

request = AnalysisRequest(
    module_name="dominance",
    experiment_path=Path("data/experiment"),
    output_path=Path("results"),
    progress_callback=progress_callback
)

result = service.run(request)
```

#### Batch Analysis

```python
requests = [
    AnalysisRequest(
        module_name="dominance",
        experiment_path=Path(f"data/experiment_{i:03d}"),
        output_path=Path(f"results/exp_{i:03d}"),
    )
    for i in range(10)
]

results = service.run_batch(requests)

# Check results
for result in results:
    if result.success:
        print(f"✓ {result.module_name}: {result.execution_time:.2f}s")
    else:
        print(f"✗ {result.module_name}: {result.error}")
```

#### Caching

```python
# Enable caching (default)
request = AnalysisRequest(
    module_name="dominance",
    experiment_path=Path("data/experiment"),
    output_path=Path("results"),
    enable_caching=True
)

# First run - computes results
result1 = service.run(request)
print(f"Cache hit: {result1.cache_hit}")  # False

# Second run - uses cached results
result2 = service.run(request)
print(f"Cache hit: {result2.cache_hit}")  # True

# Force refresh
request.force_refresh = True
result3 = service.run(request)
print(f"Cache hit: {result3.cache_hit}")  # False

# Clear cache
service.clear_cache()
```

### Architecture

The analysis system is built on a protocol-based architecture:

```
farm/analysis/
├── protocols.py          # Protocol definitions (interfaces)
├── core.py              # Base implementations
├── exceptions.py        # Custom exceptions
├── validation.py        # Data validators
├── registry.py          # Module registry
├── service.py           # High-level service API
├── common/
│   ├── context.py       # Analysis context
│   ├── metrics.py       # Shared metrics utilities
│   └── utils.py         # Common analysis utilities
├── template/
│   └── standard_module.py # Standard module template
└── {module}/
    ├── __init__.py      # Package exports
    ├── module.py        # Module implementation
    ├── data.py          # Data processing
    ├── compute.py       # Statistical computations
    ├── analyze.py       # Analysis functions
    └── plot.py          # Visualizations
```

### Testing

Run the test suite:

```bash
# All tests
pytest tests/analysis/

# Specific test file
pytest tests/analysis/test_core.py

# With coverage
pytest tests/analysis/ --cov=farm.analysis --cov-report=html
```

### Phase 1 Foundation (Complete)

Phase 1 of the analysis refactoring has established the foundation for migrating all analysis code into the unified module system:

#### ✅ Completed Tasks

- **Common Utilities** (`farm.analysis.common.utils`):
  - Statistical functions (`calculate_statistics`, `calculate_trend`, etc.)
  - Data processing helpers (`validate_required_columns`, `handle_missing_data`)
  - Plotting utilities (`setup_plot_figure`, `save_plot_figure`, `get_agent_type_colors`)
  - File system utilities (`find_database_path`, `create_output_subdirs`)

- **Standard Module Template** (`farm.analysis.template.standard_module`):
  - Complete module structure template
  - Standard function signatures and patterns
  - Consistent validation setup
  - Ready-to-copy implementation

- **Testing Infrastructure** (`tests.analysis.test_common_utils`):
  - Comprehensive test suite for utilities
  - 27 test cases covering all utility functions
  - Integration with existing pytest fixtures

- **Pattern Extraction**:
  - Analyzed existing modules (dominance, genesis, advantage, social_behavior)
  - Identified common patterns for data loading, analysis, and visualization
  - Standardized agent type naming and color schemes

#### 🔄 Next Phases

- **Phase 2**: Migrate core analyzers (population, resources, actions, agents)
- **Phase 3**: Migrate specialized analyzers (learning, spatial, temporal, combat)
- **Phase 4**: Consolidate scripts and update orchestration
- **Phase 5**: Final testing and documentation

### Migration from Old System

The old `base_module.py` system is deprecated. To migrate:

**Old way:**
```python
class MyModule(AnalysisModule):
    def register_analysis(self):  # Old method name
        self._analysis_functions = {...}
```

**New way:**
```python
class MyModule(BaseAnalysisModule):
    def register_functions(self):  # New method name
        self._functions = {...}
```

Key changes:
- `AnalysisModule` → `BaseAnalysisModule`
- `register_analysis()` → `register_functions()`
- `_analysis_functions` → `_functions`
- `_analysis_groups` → `_groups`
- All functions now use standard `(df, ctx, **kwargs)` signature
- Use `make_analysis_function()` to wrap legacy functions

### Best Practices

1. **Use type hints everywhere**
   ```python
   def my_function(df: pd.DataFrame, ctx: AnalysisContext) -> None:
       ...
   ```

2. **Validate input data**
   ```python
   validator = ColumnValidator(required_columns=['col1', 'col2'])
   self.set_validator(validator)
   ```

3. **Report progress for long operations**
   ```python
   ctx.report_progress("Processing step 1", 0.25)
   ctx.report_progress("Processing step 2", 0.50)
   ```

4. **Use context for file paths**
   ```python
   output_file = ctx.get_output_file("results.csv", subdir="metrics")
   ```

5. **Log important events**
   ```python
   ctx.logger.info("Starting analysis")
   ctx.logger.warning("Low data quality detected")
   ```

6. **Handle errors gracefully**
   ```python
   from farm.analysis.exceptions import DataValidationError
   
   try:
       process_data(df)
   except DataValidationError as e:
       ctx.logger.error(f"Validation failed: {e}")
       raise
   ```

### Available Modules

List all registered modules:

```python
from farm.analysis.registry import list_modules

print(list_modules())
```

Get module info:

```python
from farm.analysis.registry import get_module

module = get_module("dominance")
info = module.get_info()

print(f"Name: {info['name']}")
print(f"Description: {info['description']}")
print(f"Function groups: {info['function_groups']}")
print(f"Functions: {info['functions']}")
```

### Contributing

When adding a new module:

1. Follow the structure in `template/module.py`
2. Add comprehensive docstrings
3. Include type hints
4. Write tests in `tests/analysis/test_{module}.py`
5. Update this README with usage examples

### License

See project LICENSE file.
