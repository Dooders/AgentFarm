# Analysis Module System

This directory contains the analysis module system for the AgentFarm project. The module system provides a standardized way to create and run different types of analyses on simulation data.

## Overview

The module system consists of:

1. **Base Module** (`base_module.py`): Defines the `AnalysisModule` abstract base class that all analysis modules must inherit from.
2. **Module Registry** (`registry.py`): Provides a central registry for all analysis modules.
3. **Analysis Modules**: Each analysis type (e.g., dominance, reproduction, resources) has its own module that inherits from `AnalysisModule`.

## Creating a New Analysis Module

To create a new analysis module:

1. Create a new directory under `farm/analysis/your_module_name/`
2. Implement your analysis functions (data processor, plots, etc.)
3. Create a module class that inherits from `AnalysisModule`
4. Register your module in the registry

### Directory Structure

```
farm/analysis/
├── base_module.py         # Base module class
├── registry.py            # Module registry
├── dominance/             # Dominance analysis module
│   ├── analyze.py         # Data processing functions
│   ├── plot.py            # Visualization functions
│   ├── module.py          # Module class implementation
│   └── ...
├── your_module_name/      # Your new module
│   ├── analyze.py         # Data processing functions
│   ├── plot.py            # Visualization functions
│   ├── module.py          # Module class implementation
│   └── ...
└── ...
```

### Module Implementation

Here's a template for implementing a new module:

```python
from typing import Callable, Dict, List, Optional

from farm.analysis.base_module import AnalysisModule
from farm.analysis.your_module_name.analyze import process_your_data
from farm.analysis.your_module_name.plot import (
    plot_function1,
    plot_function2,
)


class YourModule(AnalysisModule):
    def __init__(self):
        super().__init__(
            name="your_module_name",
            description="Description of your module",
        )
    
    def register_analysis(self) -> None:
        # Register all functions
        self._analysis_functions.update({
            "plot_function1": plot_function1,
            "plot_function2": plot_function2,
        })
        
        # Define function groups
        self._analysis_groups = {
            "all": list(self._analysis_functions.values()),
            "basic": [plot_function1],
            "advanced": [plot_function2],
        }
    
    def get_data_processor(self) -> Callable:
        return process_your_data
    
    def get_db_loader(self) -> Optional[Callable]:
        # Return None if not using a database
        return None
    
    def get_db_filename(self) -> Optional[str]:
        # Return None if not using a database
        return None


# Create a singleton instance
your_module = YourModule()
```

### Registering Your Module

Add your module to the registry in `farm/analysis/registry.py`:

```python
def register_modules():
    # Import modules
    from farm.analysis.dominance.module import dominance_module
    from farm.analysis.your_module_name.module import your_module
    
    # Register modules
    registry.register_module(dominance_module)
    registry.register_module(your_module)
```

## Running Analysis

To run analysis using the module system, create a script in the `scripts/` directory:

```python
import logging
from analysis_config import run_analysis

def main():
    # Run the analysis using the module system
    output_path, df = run_analysis(
        analysis_type="your_module_name",
        function_group="all",  # Or specify a specific group
    )
    
    if df is not None and not df.empty:
        logging.info(f"Analysis complete. Processed {len(df)} simulations.")

if __name__ == "__main__":
    main()
```

## Function Groups

Function groups allow you to run specific subsets of analysis functions. Common groups include:

- `all`: All analysis functions
- `basic`: Basic analysis functions
- `advanced`: Advanced analysis functions

You can define your own groups in your module's `register_analysis` method.

## Module Interface

Each module provides the following methods:

- `get_analysis_function(name)`: Get a specific analysis function by name
- `get_analysis_functions(group)`: Get a list of analysis functions by group
- `get_function_groups()`: Get a list of available function groups
- `get_function_names()`: Get a list of all available function names
- `get_module_info()`: Get information about the module
- `run_analysis(...)`: Run analysis functions for this module

## Benefits of the Module System

1. **Standardization**: All analysis modules follow the same structure and interface
2. **Discoverability**: Easy to discover available modules and functions
3. **Flexibility**: Run specific groups of functions or individual functions
4. **Maintainability**: Each module is self-contained and can be developed independently
5. **Extensibility**: Easy to add new modules without modifying existing code 