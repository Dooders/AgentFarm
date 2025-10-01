# Analysis Module System - Quick Reference

## Basic Usage

### Running an Analysis

```python
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService
from pathlib import Path

# Setup
service = AnalysisService(EnvConfigService())

# Run analysis
request = AnalysisRequest(
    module_name="dominance",
    experiment_path=Path("data/experiment"),
    output_path=Path("results")
)
result = service.run(request)

# Check result
if result.success:
    print(f"Done in {result.execution_time:.2f}s")
else:
    print(f"Failed: {result.error}")
```

### Creating a Module

```python
from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.common.context import AnalysisContext
import pandas as pd

class MyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(name="my_module", description="My analysis")
    
    def register_functions(self):
        self._functions = {
            "analyze": make_analysis_function(self.analyze_func)
        }
        self._groups = {"all": list(self._functions.values())}
    
    def get_data_processor(self):
        return SimpleDataProcessor(lambda p, **k: pd.DataFrame(...))
    
    def analyze_func(self, df: pd.DataFrame, ctx: AnalysisContext):
        # Your logic here
        output = ctx.get_output_file("results.csv")
        df.to_csv(output)
        ctx.report_progress("Done", 1.0)

# Register
from farm.analysis.registry import registry
registry.register(MyModule())
```

## Common Patterns

### Validation

```python
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator

# In your module's __init__
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

### Progress Tracking

```python
def progress(msg: str, pct: float):
    print(f"[{pct:>5.1%}] {msg}")

request = AnalysisRequest(
    module_name="dominance",
    experiment_path=Path("data"),
    output_path=Path("results"),
    progress_callback=progress
)
```

### Caching

```python
# Enable caching (default)
request = AnalysisRequest(..., enable_caching=True)

# Force refresh
request = AnalysisRequest(..., force_refresh=True)

# Clear cache
service.clear_cache()
```

### Batch Processing

```python
requests = [
    AnalysisRequest(
        module_name="dominance",
        experiment_path=Path(f"data/exp_{i}"),
        output_path=Path(f"results/exp_{i}")
    )
    for i in range(10)
]

results = service.run_batch(requests)
```

### Custom Parameters

```python
request = AnalysisRequest(
    module_name="dominance",
    experiment_path=Path("data"),
    output_path=Path("results"),
    processor_kwargs={"option": "value"},
    analysis_kwargs={
        "plot_func": {"bins": 50, "figsize": (12, 8)}
    }
)
```

## Error Handling

```python
from farm.analysis.exceptions import (
    ModuleNotFoundError,
    DataValidationError,
    DataProcessingError
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
except DataProcessingError as e:
    print(f"Processing failed at step: {e.step}")
```

## Analysis Functions

### Standard Signature

```python
def my_function(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    # Access context
    ctx.logger.info("Starting")
    output_file = ctx.get_output_file("output.csv")
    config_val = ctx.get_config("key", default="value")
    
    # Your logic
    results = process(df)
    results.to_csv(output_file)
    
    # Report progress
    ctx.report_progress("Complete", 1.0)
```

### Wrapping Legacy Functions

```python
# Legacy function
def old_func(df, output_path, **kwargs):
    df.to_csv(f"{output_path}/output.csv")

# Wrap it
wrapped = make_analysis_function(old_func, name="old_func")
```

## Context API

```python
# Get output paths
csv_file = ctx.get_output_file("data.csv")
plot_file = ctx.get_output_file("plot.png", subdir="plots")

# Report progress
ctx.report_progress("Processing", 0.5)

# Get config
threshold = ctx.get_config("threshold", default=0.5)

# Logging
ctx.logger.info("Info message")
ctx.logger.warning("Warning message")
ctx.logger.error("Error message")
```

## Service API

```python
# List modules
modules = service.list_modules()
for m in modules:
    print(f"{m['name']}: {m['description']}")

# Get module info
info = service.get_module_info("dominance")
print(f"Groups: {info['function_groups']}")
print(f"Functions: {info['functions']}")

# Clear cache
count = service.clear_cache()
print(f"Cleared {count} entries")
```

## Registry API

```python
from farm.analysis.registry import registry, get_module, get_module_names

# Register module
registry.register(my_module)

# Get module
module = get_module("dominance")

# List all
names = get_module_names()
print(f"Available: {', '.join(names)}")

# Module info
info = module.get_info()
```

## Data Validation

### Validators

```python
# Column validator
col_val = ColumnValidator(
    required_columns=['col1', 'col2'],
    column_types={'col1': int}
)

# Quality validator
qual_val = DataQualityValidator(
    min_rows=10,
    max_null_fraction=0.1,
    allow_duplicates=False,
    value_ranges={'score': (0.0, 1.0)}
)

# Composite
composite = CompositeValidator([col_val, qual_val])
composite.validate(df)
```

### Helper Functions

```python
from farm.analysis.validation import (
    validate_numeric_columns,
    validate_simulation_data
)

# Validate numeric columns
valid_cols = validate_numeric_columns(
    df, 
    ['col1', 'col2'],
    allow_missing=True
)

# Validate simulation data
validate_simulation_data(df)  # Checks for simulation_id
```

## Cheat Sheet

| Task | Code |
|------|------|
| Run analysis | `service.run(request)` |
| Batch analysis | `service.run_batch(requests)` |
| List modules | `service.list_modules()` |
| Get module info | `service.get_module_info(name)` |
| Clear cache | `service.clear_cache()` |
| Register module | `registry.register(module)` |
| Get module | `get_module(name)` |
| Validate data | `validator.validate(df)` |
| Create context | `AnalysisContext(output_path=path)` |
| Save output | `ctx.get_output_file("name.csv")` |
| Report progress | `ctx.report_progress(msg, pct)` |

## Import Reference

```python
# Service layer
from farm.analysis.service import (
    AnalysisService,
    AnalysisRequest,
    AnalysisResult
)

# Core
from farm.analysis.core import (
    BaseAnalysisModule,
    SimpleDataProcessor,
    make_analysis_function
)

# Validation
from farm.analysis.validation import (
    ColumnValidator,
    DataQualityValidator,
    CompositeValidator
)

# Exceptions
from farm.analysis.exceptions import (
    AnalysisError,
    DataValidationError,
    ModuleNotFoundError
)

# Context
from farm.analysis.common.context import AnalysisContext

# Registry
from farm.analysis.registry import (
    registry,
    get_module,
    get_module_names
)
```

## File Locations

```
farm/analysis/
├── README.md              # Complete guide
├── ARCHITECTURE.md        # System design
├── QUICK_REFERENCE.md     # This file
├── protocols.py           # Interfaces
├── core.py               # Base classes
├── validation.py         # Validators
├── exceptions.py         # Errors
├── registry.py           # Module registry
├── service.py            # Service API
└── common/
    └── context.py        # Analysis context

examples/
└── analysis_example.py   # 7 complete examples

tests/analysis/
├── conftest.py           # Test fixtures
├── test_*.py            # Test suites
└── test_integration.py  # E2E tests
```

## Environment Variables

```bash
# Module discovery
export FARM_ANALYSIS_MODULES="my.module.path.my_module,other.module.path.other_module"
```

## Tips

1. **Always use type hints** - Enables IDE autocomplete and type checking
2. **Use context for paths** - Don't hardcode paths, use `ctx.get_output_file()`
3. **Validate early** - Set validators in `__init__`
4. **Report progress** - Users appreciate feedback on long operations
5. **Handle errors gracefully** - Use specific exceptions with context
6. **Test thoroughly** - Use provided fixtures in `tests/analysis/conftest.py`
7. **Document functions** - Add docstrings with Args/Returns
8. **Cache when possible** - Enable caching for expensive computations
9. **Use batch processing** - Process multiple analyses efficiently
10. **Follow protocols** - Implement required methods for module compliance
