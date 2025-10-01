# Migration Guide: Analysis Module System

This guide helps you migrate existing analysis modules and code to the new protocol-based architecture.

## Quick Summary

### What Changed

| Old | New |
|-----|-----|
| `farm.analysis.base_module.AnalysisModule` | `farm.analysis.core.BaseAnalysisModule` |
| `register_analysis()` | `register_functions()` |
| `_analysis_functions` | `_functions` |
| `_analysis_groups` | `_groups` |
| Functions with various signatures | Standard `(df, ctx, **kwargs)` |
| `from farm.analysis.base_module import get_valid_numeric_columns` | `from farm.analysis.common.metrics import get_valid_numeric_columns` |

### Import Changes

**Old imports:**
```python
from farm.analysis.base_module import (
    AnalysisModule,
    BaseAnalysisModule,
    get_valid_numeric_columns,
    analyze_correlations,
)
```

**New imports:**
```python
from farm.analysis.core import BaseAnalysisModule
from farm.analysis.common.metrics import (
    get_valid_numeric_columns,
    analyze_correlations,
    split_and_compare_groups,
    group_and_analyze,
    find_top_correlations,
)
```

## Step-by-Step Migration

### 1. Update Module Class

**Before:**
```python
from farm.analysis.base_module import AnalysisModule

class MyModule(AnalysisModule):
    def __init__(self):
        super().__init__(
            name="my_module",
            description="My analysis"
        )
    
    def register_analysis(self):  # ❌ Old method name
        self._analysis_functions = {  # ❌ Old attribute
            "my_func": my_func
        }
        self._analysis_groups = {"all": [...]}  # ❌ Old attribute
    
    def get_data_processor(self):
        return my_processor_func  # ❌ Returns function directly
```

**After:**
```python
from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function

class MyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(
            name="my_module",
            description="My analysis"
        )
    
    def register_functions(self):  # ✅ New method name
        self._functions = {  # ✅ New attribute
            "my_func": make_analysis_function(my_func)  # ✅ Wrapped
        }
        self._groups = {"all": [...]}  # ✅ New attribute
    
    def get_data_processor(self):
        return SimpleDataProcessor(my_processor_func)  # ✅ Returns processor object
```

### 2. Update Analysis Functions

**Before (various signatures):**
```python
# Old signature 1
def my_func(df, output_path):
    df.to_csv(f"{output_path}/results.csv")

# Old signature 2  
def my_func(df, output_path, **kwargs):
    pass

# Old signature 3
def my_func(df):
    pass
```

**After (standard signature):**
```python
from farm.analysis.common.context import AnalysisContext
import pandas as pd

def my_func(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> None:
    """Analysis function with standard signature.
    
    Args:
        df: Input DataFrame
        ctx: Analysis context (output_path, logger, config, etc.)
        **kwargs: Function-specific parameters
    """
    # Use context for output paths
    output_file = ctx.get_output_file("results.csv")
    df.to_csv(output_file)
    
    # Report progress
    ctx.report_progress("Complete", 1.0)
    
    # Access logger
    ctx.logger.info("Analysis complete")
```

### 3. Wrap Legacy Functions

If you can't change the function signature, wrap it:

```python
from farm.analysis.core import make_analysis_function

# Legacy function
def old_function(df, output_path, **kwargs):
    # Old implementation
    pass

# Wrap it to work with new system
wrapped_function = make_analysis_function(old_function, name="old_function")

# Use in module
class MyModule(BaseAnalysisModule):
    def register_functions(self):
        self._functions = {
            "old_function": wrapped_function
        }
```

### 4. Update Validation

**Before (manual validation):**
```python
def process_data(experiment_path):
    df = load_data(experiment_path)
    if 'iteration' not in df.columns:
        raise ValueError("Missing iteration column")
    if len(df) < 10:
        raise ValueError("Not enough data")
    return df
```

**After (use validators):**
```python
from farm.analysis.validation import (
    ColumnValidator,
    DataQualityValidator,
    CompositeValidator
)

class MyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(name="my_module", description="...")
        
        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(
                required_columns=['iteration', 'agent_type'],
                column_types={'iteration': int}
            ),
            DataQualityValidator(
                min_rows=10,
                max_null_fraction=0.1
            )
        ])
        self.set_validator(validator)
```

### 5. Update Service Usage

**Before:**
```python
from farm.analysis.registry import get_module

module = get_module("dominance")
output_path, df = module.run_analysis(
    experiment_path="/path/to/data",
    output_path="/path/to/output",
    group="all"
)
```

**After:**
```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

service = AnalysisService(EnvConfigService())

request = AnalysisRequest(
    module_name="dominance",
    experiment_path=Path("/path/to/data"),
    output_path=Path("/path/to/output"),
    group="all"
)

result = service.run(request)

if result.success:
    print(f"Success: {result.execution_time:.2f}s")
    df = result.dataframe
else:
    print(f"Failed: {result.error}")
```

### 6. Update Error Handling

**Before (generic exceptions):**
```python
try:
    result = analyze_data(df)
except Exception as e:
    print(f"Error: {e}")
```

**After (specific exceptions):**
```python
from farm.analysis.exceptions import (
    DataValidationError,
    ModuleNotFoundError,
    DataProcessingError
)

try:
    result = service.run(request)
except ModuleNotFoundError as e:
    print(f"Module '{e.module_name}' not found")
    print(f"Available: {', '.join(e.available_modules)}")
except DataValidationError as e:
    print(f"Validation failed: {e}")
    if e.missing_columns:
        print(f"Missing: {e.missing_columns}")
    if e.invalid_columns:
        print(f"Invalid: {e.invalid_columns}")
except DataProcessingError as e:
    print(f"Processing failed at step: {e.step}")
```

## Common Migration Patterns

### Pattern 1: Simple Analysis Module

**Before:**
```python
from farm.analysis.base_module import AnalysisModule

class SimpleModule(AnalysisModule):
    def __init__(self):
        super().__init__(name="simple", description="Simple analysis")
    
    def register_analysis(self):
        self._analysis_functions = {
            "analyze": self.analyze_func
        }
        self._analysis_groups = {
            "all": [self.analyze_func]
        }
    
    def get_data_processor(self):
        return self.process_data
    
    def get_db_loader(self):
        return None
    
    def get_db_filename(self):
        return None
    
    def process_data(self, path):
        # Load data
        return pd.DataFrame(...)
    
    def analyze_func(self, df, output_path):
        # Analyze
        df.to_csv(f"{output_path}/results.csv")
```

**After:**
```python
from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.common.context import AnalysisContext
import pandas as pd

class SimpleModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(name="simple", description="Simple analysis")
    
    def register_functions(self):
        self._functions = {
            "analyze": make_analysis_function(self.analyze_func)
        }
        self._groups = {
            "all": [self._functions["analyze"]]
        }
    
    def get_data_processor(self):
        return SimpleDataProcessor(self.process_data)
    
    def process_data(self, path, **kwargs):
        # Load data
        return pd.DataFrame(...)
    
    def analyze_func(self, df: pd.DataFrame, ctx: AnalysisContext):
        # Analyze
        output_file = ctx.get_output_file("results.csv")
        df.to_csv(output_file)
```

### Pattern 2: Module with Database

**Before:**
```python
class DbModule(AnalysisModule):
    def get_db_filename(self):
        return "mymodule.db"
    
    def get_db_loader(self):
        return load_from_db
```

**After:**
```python
class DbModule(BaseAnalysisModule):
    def supports_database(self):
        return True
    
    def get_db_filename(self):
        return "mymodule.db"
    
    def get_db_loader(self):
        return load_from_db
```

### Pattern 3: Module with Function Groups

**Before:**
```python
def register_analysis(self):
    self._analysis_functions = {
        "plot1": plot1,
        "plot2": plot2,
        "metric1": metric1,
    }
    self._analysis_groups = {
        "all": list(self._analysis_functions.values()),
        "plots": [plot1, plot2],
        "metrics": [metric1],
    }
```

**After:**
```python
def register_functions(self):
    self._functions = {
        "plot1": make_analysis_function(plot1),
        "plot2": make_analysis_function(plot2),
        "metric1": make_analysis_function(metric1),
    }
    self._groups = {
        "all": list(self._functions.values()),
        "plots": [self._functions["plot1"], self._functions["plot2"]],
        "metrics": [self._functions["metric1"]],
    }
```

## Checklist

Use this checklist when migrating a module:

- [ ] Update imports
  - [ ] Change `from farm.analysis.base_module import AnalysisModule` to `from farm.analysis.core import BaseAnalysisModule`
  - [ ] Update metric imports to use `farm.analysis.common.metrics`
- [ ] Update class definition
  - [ ] Change `class MyModule(AnalysisModule):` to `class MyModule(BaseAnalysisModule):`
  - [ ] Rename `register_analysis()` to `register_functions()`
  - [ ] Rename `_analysis_functions` to `_functions`
  - [ ] Rename `_analysis_groups` to `_groups`
- [ ] Update function registration
  - [ ] Wrap functions with `make_analysis_function()` if needed
  - [ ] Update function references in groups
- [ ] Update data processor
  - [ ] Return `SimpleDataProcessor(func)` instead of `func`
- [ ] Add validation (optional but recommended)
  - [ ] Create validators in `__init__`
  - [ ] Call `self.set_validator(validator)`
- [ ] Update analysis functions
  - [ ] Change signature to `(df, ctx, **kwargs)`
  - [ ] Use `ctx.get_output_file()` for paths
  - [ ] Use `ctx.logger` for logging
  - [ ] Use `ctx.report_progress()` for progress
- [ ] Update tests
  - [ ] Use new fixtures from `tests/analysis/conftest.py`
  - [ ] Test with `AnalysisService`
- [ ] Update documentation
  - [ ] Update docstrings
  - [ ] Add type hints

## Backwards Compatibility

The old `base_module.py` is still available for backwards compatibility, but it's deprecated. Files using it will continue to work but should be migrated.

### Files Still Using Old System

These files need migration:
- `farm/analysis/dominance/analyze.py` - Has `DominanceAnalysis` class using old `BaseAnalysisModule`
- `farm/analysis/advantage/analyze.py` - Uses old `BaseAnalysisModule`

These are kept for backwards compatibility but new code should use the module system.

## Getting Help

- See `farm/analysis/README.md` for complete documentation
- See `farm/analysis/template/module.py` for a modern template
- See `examples/analysis_example.py` for working examples
- See `tests/analysis/` for test examples
- See `farm/analysis/QUICK_REFERENCE.md` for quick lookups

## Common Issues

### Issue 1: Function Not Called

**Problem:** Function registered but not called during analysis

**Solution:** Make sure you wrap it:
```python
self._functions = {
    "my_func": make_analysis_function(my_func)  # ✅ Wrapped
}
```

### Issue 2: Import Errors

**Problem:** `ImportError: cannot import name 'AnalysisModule'`

**Solution:** Update import:
```python
# Old
from farm.analysis.base_module import AnalysisModule

# New
from farm.analysis.core import BaseAnalysisModule
```

### Issue 3: Path Issues

**Problem:** `TypeError: unsupported operand type(s) for /: 'str' and 'str'`

**Solution:** Use Path objects and context:
```python
# Old
output = f"{output_path}/file.csv"

# New
output = ctx.get_output_file("file.csv")
```

### Issue 4: Missing Metrics

**Problem:** `ImportError: cannot import name 'get_valid_numeric_columns' from 'farm.analysis.base_module'`

**Solution:** Import from common.metrics:
```python
# Old
from farm.analysis.base_module import get_valid_numeric_columns

# New
from farm.analysis.common.metrics import get_valid_numeric_columns
```
