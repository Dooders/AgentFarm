# Analysis Module API Reference

**Version**: 2.0.0  
**Last Updated**: 2025-10-04

Complete API documentation for the `farm.analysis` module system.

---

## Table of Contents

1. [Service Layer](#service-layer)
2. [Core Classes](#core-classes)
3. [Protocols](#protocols)
4. [Validation](#validation)
5. [Exceptions](#exceptions)
6. [Registry](#registry)
7. [Common Utilities](#common-utilities)
8. [Analysis Modules](#analysis-modules)
9. [Type Definitions](#type-definitions)

---

## Service Layer

### AnalysisService

High-level service for running analysis modules.

```python
from farm.analysis.service import AnalysisService
```

#### Constructor

```python
AnalysisService(
    config_service: IConfigService,
    cache_dir: Optional[Path] = None,
    auto_register: bool = True
)
```

**Parameters:**
- `config_service` (IConfigService): Configuration service instance
- `cache_dir` (Optional[Path]): Directory for result caching. Default: `.analysis_cache`
- `auto_register` (bool): Auto-register modules on initialization. Default: `True`

**Example:**
```python
from farm.analysis.service import AnalysisService
from farm.core.services import EnvConfigService

config_service = EnvConfigService()
service = AnalysisService(config_service)
```

#### Methods

##### `run(request: AnalysisRequest) -> AnalysisResult`

Run a single analysis request.

**Parameters:**
- `request` (AnalysisRequest): The analysis request to execute

**Returns:**
- `AnalysisResult`: Result object with execution details

**Example:**
```python
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment"),
    output_path=Path("results")
)

result = service.run(request)
if result.success:
    print(f"Completed in {result.execution_time:.2f}s")
```

##### `run_batch(requests: List[AnalysisRequest], fail_fast: bool = False) -> List[AnalysisResult]`

Run multiple analysis requests in batch.

**Parameters:**
- `requests` (List[AnalysisRequest]): List of analysis requests
- `fail_fast` (bool): Stop on first failure. Default: `False`

**Returns:**
- `List[AnalysisResult]`: List of results for each request

**Example:**
```python
requests = [
    AnalysisRequest(module_name="population", ...),
    AnalysisRequest(module_name="resources", ...),
]

results = service.run_batch(requests)
successful = sum(1 for r in results if r.success)
print(f"{successful}/{len(results)} analyses completed")
```

##### `validate_request(request: AnalysisRequest) -> None`

Validate an analysis request.

**Parameters:**
- `request` (AnalysisRequest): Request to validate

**Raises:**
- `ConfigurationError`: If request is invalid
- `ModuleNotFoundError`: If module doesn't exist

##### `get_module_info(module_name: str) -> Dict[str, Any]`

Get information about a specific module.

**Parameters:**
- `module_name` (str): Name of the module

**Returns:**
- `Dict[str, Any]`: Module metadata

**Example:**
```python
info = service.get_module_info("population")
print(f"Functions: {info['functions']}")
print(f"Groups: {info['function_groups']}")
```

##### `list_modules() -> List[Dict[str, str]]`

List all available modules.

**Returns:**
- `List[Dict[str, str]]`: List of module information dictionaries

**Example:**
```python
modules = service.list_modules()
for mod in modules:
    print(f"{mod['name']}: {mod['description']}")
```

##### `clear_cache() -> int`

Clear the analysis result cache.

**Returns:**
- `int`: Number of cache entries cleared

---

### AnalysisRequest

Request object for analysis execution.

```python
from farm.analysis.service import AnalysisRequest
```

#### Constructor

```python
AnalysisRequest(
    module_name: str,
    experiment_path: Path,
    output_path: Path,
    group: str = "all",
    processor_kwargs: Dict[str, Any] = None,
    analysis_kwargs: Dict[str, Dict[str, Any]] = None,
    enable_caching: bool = True,
    force_refresh: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    metadata: Dict[str, Any] = None
)
```

**Parameters:**
- `module_name` (str): Name of analysis module to run
- `experiment_path` (Path): Path to experiment data
- `output_path` (Path): Path to save results
- `group` (str): Function group to execute. Default: `"all"`
- `processor_kwargs` (Dict): Arguments for data processor
- `analysis_kwargs` (Dict): Arguments for specific analysis functions
- `enable_caching` (bool): Use cached results if available. Default: `True`
- `force_refresh` (bool): Force recomputation. Default: `False`
- `progress_callback` (Callable): Progress update callback
- `metadata` (Dict): Additional metadata

**Example:**
```python
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/exp_001"),
    output_path=Path("results/pop"),
    group="plots",  # Run only plot functions
    processor_kwargs={"filter_dead": True},
    progress_callback=lambda msg, pct: print(f"{pct:.0%}: {msg}")
)
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert request to dictionary for serialization.

##### `get_cache_key() -> str`

Generate cache key for this request.

**Returns:**
- `str`: Hash-based cache key

---

### AnalysisResult

Result object from analysis execution.

```python
from farm.analysis.service import AnalysisResult
```

#### Attributes

- `success` (bool): Whether analysis completed successfully
- `module_name` (str): Name of module that ran
- `output_path` (Path): Path where results were saved
- `dataframe` (Optional[pd.DataFrame]): Processed DataFrame
- `execution_time` (float): Time taken (seconds)
- `error` (Optional[str]): Error message if failed
- `metadata` (Dict[str, Any]): Additional metadata
- `cache_hit` (bool): Whether result was from cache
- `timestamp` (datetime): When analysis was run

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert result to dictionary for serialization.

##### `save_summary(path: Optional[Path] = None) -> Path`

Save result summary to JSON file.

**Parameters:**
- `path` (Optional[Path]): Save path. Default: `output_path/analysis_summary.json`

**Returns:**
- `Path`: Path where summary was saved

---

### AnalysisCache

File-based cache for analysis results.

```python
from farm.analysis.service import AnalysisCache
```

#### Constructor

```python
AnalysisCache(cache_dir: Path)
```

**Parameters:**
- `cache_dir` (Path): Directory to store cached results

#### Methods

##### `has(cache_key: str) -> bool`

Check if cached result exists.

##### `get(cache_key: str) -> Optional[tuple[Path, pd.DataFrame]]`

Get cached result.

**Returns:**
- `Optional[tuple]`: (output_path, dataframe) or None

##### `put(cache_key: str, output_path: Path, dataframe: Optional[pd.DataFrame]) -> None`

Store result in cache.

##### `clear() -> int`

Clear all cached results.

**Returns:**
- `int`: Number of files deleted

---

## Core Classes

### BaseAnalysisModule

Base implementation of an analysis module.

```python
from farm.analysis.core import BaseAnalysisModule
```

#### Constructor

```python
BaseAnalysisModule(name: str, description: str)
```

**Parameters:**
- `name` (str): Unique module identifier
- `description` (str): Human-readable description

#### Properties

- `name` (str): Module name
- `description` (str): Module description

#### Abstract Methods

Must be implemented by subclasses:

##### `register_functions() -> None`

Register all analysis functions for this module.

**Example:**
```python
def register_functions(self):
    self._functions = {
        "plot_population": make_analysis_function(plot_func),
        "compute_stats": make_analysis_function(stats_func),
    }
    
    self._groups = {
        "all": list(self._functions.values()),
        "plots": [self._functions["plot_population"]],
        "metrics": [self._functions["compute_stats"]],
    }
```

##### `get_data_processor() -> DataProcessor`

Get the data processor for this module.

**Returns:**
- `DataProcessor`: Data processor instance

#### Methods

##### `get_validator() -> Optional[DataValidator]`

Get the data validator.

**Returns:**
- `Optional[DataValidator]`: Validator or None

##### `set_validator(validator: DataValidator) -> None`

Set the data validator.

##### `get_analysis_functions(group: str = "all") -> List[AnalysisFunction]`

Get analysis functions by group.

**Parameters:**
- `group` (str): Function group name

**Returns:**
- `List[AnalysisFunction]`: Functions in the group

##### `get_function_groups() -> List[str]`

Get available function group names.

##### `get_function(name: str) -> Optional[AnalysisFunction]`

Get a specific function by name.

##### `get_function_names() -> List[str]`

Get all function names.

##### `supports_database() -> bool`

Whether module uses database for storage.

**Returns:**
- `bool`: True if uses database

##### `run_analysis(...) -> tuple[Path, Optional[pd.DataFrame]]`

Run complete analysis workflow.

**Parameters:**
- `experiment_path` (Path): Path to experiment data
- `output_path` (Path): Path to save results
- `group` (str): Function group to run
- `processor_kwargs` (Optional[Dict]): Data processor arguments
- `analysis_kwargs` (Optional[Dict]): Analysis function arguments
- `progress_callback` (Optional[Callable]): Progress callback

**Returns:**
- `tuple[Path, Optional[pd.DataFrame]]`: (output_path, dataframe)

##### `get_info() -> Dict[str, Any]`

Get module information.

**Returns:**
- `Dict[str, Any]`: Module metadata

---

### SimpleDataProcessor

Simple data processor that applies a processing function.

```python
from farm.analysis.core import SimpleDataProcessor
```

#### Constructor

```python
SimpleDataProcessor(process_func: Callable[[Any, ...], pd.DataFrame])
```

**Parameters:**
- `process_func` (Callable): Function that processes data

**Example:**
```python
def my_processor(data_path: Path, **kwargs) -> pd.DataFrame:
    # Load and process data
    df = pd.read_csv(data_path / "data.csv")
    return df

processor = SimpleDataProcessor(my_processor)
```

#### Methods

##### `process(data: Any, **kwargs) -> pd.DataFrame`

Process data using the configured function.

---

### ChainedDataProcessor

Data processor that chains multiple processors.

```python
from farm.analysis.core import ChainedDataProcessor
```

#### Constructor

```python
ChainedDataProcessor(processors: List[DataProcessor])
```

**Parameters:**
- `processors` (List[DataProcessor]): List of processors to chain

**Example:**
```python
processor = ChainedDataProcessor([
    CleaningProcessor(),
    NormalizationProcessor(),
    FeatureProcessor(),
])
```

#### Methods

##### `process(data: pd.DataFrame, **kwargs) -> pd.DataFrame`

Process data through all processors in sequence.

---

### make_analysis_function

Wrap a function to match the AnalysisFunction protocol.

```python
from farm.analysis.core import make_analysis_function
```

#### Signature

```python
make_analysis_function(
    func: Callable,
    name: Optional[str] = None
) -> AnalysisFunction
```

**Parameters:**
- `func` (Callable): Function to wrap
- `name` (Optional[str]): Custom name for the function

**Returns:**
- `AnalysisFunction`: Wrapped function

**Example:**
```python
# Legacy function with different signature
def old_plot(df, output_path):
    # ...
    pass

# Wrap to modern signature
wrapped = make_analysis_function(old_plot)

# Now can be called with (df, ctx, **kwargs)
wrapped(df, ctx)
```

---

### AnalysisContext

Shared analysis context passed to analysis functions.

```python
from farm.analysis.common.context import AnalysisContext
```

#### Constructor

```python
AnalysisContext(
    output_path: Path,
    config: Dict[str, Any] = None,
    services: Dict[str, Any] = None,
    logger: structlog.stdlib.BoundLogger = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    metadata: Dict[str, Any] = None
)
```

**Parameters:**
- `output_path` (Path): Directory for outputs
- `config` (Dict): Configuration options
- `services` (Dict): Dependency map
- `logger` (Logger): Logger instance
- `progress_callback` (Callable): Progress callback
- `metadata` (Dict): Additional metadata

#### Methods

##### `get_output_file(filename: str, subdir: Optional[str] = None) -> Path`

Get full path for an output file.

**Parameters:**
- `filename` (str): Output filename
- `subdir` (Optional[str]): Subdirectory within output_path

**Returns:**
- `Path`: Full path to output file

**Example:**
```python
# Save to output_path/results.csv
path = ctx.get_output_file("results.csv")

# Save to output_path/plots/chart.png
path = ctx.get_output_file("chart.png", subdir="plots")
```

##### `report_progress(message: str, progress: float = 0.0) -> None`

Report progress if callback is set.

**Parameters:**
- `message` (str): Progress message
- `progress` (float): Progress value (0.0 to 1.0)

##### `get_config(key: str, default: Any = None) -> Any`

Get configuration value with fallback.

**Parameters:**
- `key` (str): Configuration key
- `default` (Any): Default value if not found

**Returns:**
- `Any`: Configuration value or default

---

## Protocols

Protocol definitions for type safety and duck typing.

### AnalysisModule

Protocol for complete analysis modules.

```python
from farm.analysis.protocols import AnalysisModule
```

#### Required Properties

- `name` (str): Module name
- `description` (str): Module description

#### Required Methods

- `get_data_processor() -> DataProcessor`
- `get_validator() -> Optional[DataValidator]`
- `get_analysis_functions(group: str = "all") -> List[AnalysisFunction]`
- `get_function_groups() -> List[str]`
- `supports_database() -> bool`

---

### DataProcessor

Protocol for data transformation.

```python
from farm.analysis.protocols import DataProcessor
```

#### Required Methods

##### `process(data: pd.DataFrame, **kwargs) -> pd.DataFrame`

Process and transform input data.

---

### DataValidator

Protocol for data validation.

```python
from farm.analysis.protocols import DataValidator
```

#### Required Methods

##### `validate(data: pd.DataFrame) -> None`

Validate data, raising exceptions if invalid.

**Raises:**
- `DataValidationError`: If validation fails

##### `get_required_columns() -> List[str]`

Get list of required columns.

---

### AnalysisFunction

Protocol for analysis functions.

```python
from farm.analysis.protocols import AnalysisFunction
```

#### Required Signature

```python
def __call__(
    df: pd.DataFrame,
    ctx: AnalysisContext,
    **kwargs: Any
) -> Optional[Any]
```

**Parameters:**
- `df` (pd.DataFrame): Input data
- `ctx` (AnalysisContext): Analysis context
- `**kwargs`: Additional parameters

**Returns:**
- `Optional[Any]`: Results (can be None for side-effect functions)

---

### DataLoader

Protocol for loading data from sources.

```python
from farm.analysis.protocols import DataLoader
```

#### Required Methods

##### `iter_data(**kwargs) -> Iterator[pd.DataFrame]`

Stream data in chunks.

##### `load_data(**kwargs) -> pd.DataFrame`

Load all data at once.

##### `get_metadata() -> Dict[str, Any]`

Get metadata about the data source.

---

### Analyzer

Protocol for complete analysis implementations.

```python
from farm.analysis.protocols import Analyzer
```

#### Required Methods

##### `analyze(data: pd.DataFrame, **kwargs) -> Dict[str, Any]`

Perform complete analysis.

##### `get_metrics() -> Dict[str, float]`

Get computed metrics.

---

### Visualizer

Protocol for creating visualizations.

```python
from farm.analysis.protocols import Visualizer
```

#### Required Methods

##### `create_charts(data: Dict[str, Any], **kwargs) -> Dict[str, Any]`

Create charts from analysis results.

##### `save_charts(output_dir: Path, prefix: str = "") -> List[Path]`

Save charts to files.

---

## Validation

Data validation utilities.

### ColumnValidator

Validates DataFrame columns meet requirements.

```python
from farm.analysis.validation import ColumnValidator
```

#### Constructor

```python
ColumnValidator(
    required_columns: Optional[List[str]] = None,
    optional_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, type]] = None
)
```

**Parameters:**
- `required_columns` (List[str]): Columns that must be present
- `optional_columns` (List[str]): Columns that may be present
- `column_types` (Dict[str, type]): Expected types for columns

**Example:**
```python
validator = ColumnValidator(
    required_columns=['iteration', 'agent_type', 'score'],
    column_types={
        'iteration': int,
        'score': float,
        'agent_type': str
    }
)

validator.validate(df)  # Raises DataValidationError if invalid
```

#### Methods

##### `validate(df: pd.DataFrame) -> None`

Validate DataFrame.

**Raises:**
- `DataValidationError`: If validation fails

##### `get_required_columns() -> List[str]`

Get list of required columns.

---

### DataQualityValidator

Validates data quality (nulls, duplicates, ranges).

```python
from farm.analysis.validation import DataQualityValidator
```

#### Constructor

```python
DataQualityValidator(
    min_rows: Optional[int] = None,
    max_null_fraction: float = 1.0,
    allow_duplicates: bool = True,
    value_ranges: Optional[Dict[str, tuple]] = None,
    custom_checks: Optional[List[Callable[[pd.DataFrame], None]]] = None
)
```

**Parameters:**
- `min_rows` (int): Minimum rows required
- `max_null_fraction` (float): Max null fraction per column (0.0-1.0)
- `allow_duplicates` (bool): Whether duplicates are allowed
- `value_ranges` (Dict): Expected ranges {column: (min, max)}
- `custom_checks` (List[Callable]): Additional validation functions

**Example:**
```python
validator = DataQualityValidator(
    min_rows=100,
    max_null_fraction=0.1,  # Max 10% nulls
    allow_duplicates=False,
    value_ranges={'score': (0.0, 1.0)}
)

validator.validate(df)
```

#### Methods

##### `validate(df: pd.DataFrame) -> None`

Validate data quality.

**Raises:**
- `InsufficientDataError`: If not enough data
- `DataValidationError`: If quality checks fail

---

### CompositeValidator

Combines multiple validators.

```python
from farm.analysis.validation import CompositeValidator
```

#### Constructor

```python
CompositeValidator(validators: List[Any])
```

**Parameters:**
- `validators` (List): List of validator objects

**Example:**
```python
validator = CompositeValidator([
    ColumnValidator(required_columns=['col1', 'col2']),
    DataQualityValidator(min_rows=10)
])

validator.validate(df)  # Runs all validators
```

#### Methods

##### `validate(df: pd.DataFrame) -> None`

Run all validators in sequence.

##### `get_required_columns() -> List[str]`

Get combined list of required columns.

---

### Utility Functions

#### validate_numeric_columns

```python
from farm.analysis.validation import validate_numeric_columns

validate_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    allow_missing: bool = False
) -> List[str]
```

Validate and filter to only numeric columns.

**Returns:**
- `List[str]`: Valid numeric column names

**Raises:**
- `DataValidationError`: If required columns missing or non-numeric

---

#### validate_simulation_data

```python
from farm.analysis.validation import validate_simulation_data

validate_simulation_data(df: pd.DataFrame) -> None
```

Standard validator for simulation data.

**Raises:**
- `DataValidationError`: If validation fails

---

## Exceptions

Custom exception types for error handling.

### AnalysisError

Base exception for all analysis-related errors.

```python
from farm.analysis.exceptions import AnalysisError
```

---

### DataValidationError

Raised when data doesn't meet requirements.

```python
from farm.analysis.exceptions import DataValidationError
```

#### Constructor

```python
DataValidationError(
    message: str,
    missing_columns: Optional[Set[str]] = None,
    invalid_columns: Optional[Dict[str, str]] = None
)
```

**Attributes:**
- `missing_columns` (Set[str]): Columns that are missing
- `invalid_columns` (Dict[str, str]): Invalid column descriptions

**Example:**
```python
try:
    validator.validate(df)
except DataValidationError as e:
    print(f"Missing: {e.missing_columns}")
    print(f"Invalid: {e.invalid_columns}")
```

---

### ModuleNotFoundError

Raised when requested module doesn't exist.

```python
from farm.analysis.exceptions import ModuleNotFoundError
```

#### Constructor

```python
ModuleNotFoundError(module_name: str, available_modules: List[str])
```

**Attributes:**
- `module_name` (str): The requested module name
- `available_modules` (List[str]): Available module names

---

### DataProcessingError

Raised when data processing fails.

```python
from farm.analysis.exceptions import DataProcessingError
```

#### Constructor

```python
DataProcessingError(message: str, step: Optional[str] = None)
```

**Attributes:**
- `step` (Optional[str]): Processing step that failed

---

### AnalysisFunctionError

Raised when an analysis function fails.

```python
from farm.analysis.exceptions import AnalysisFunctionError
```

#### Constructor

```python
AnalysisFunctionError(function_name: str, original_error: Exception)
```

**Attributes:**
- `function_name` (str): Name of failed function
- `original_error` (Exception): Original exception

---

### InsufficientDataError

Raised when there's not enough data.

```python
from farm.analysis.exceptions import InsufficientDataError
```

#### Constructor

```python
InsufficientDataError(
    message: str,
    required_rows: Optional[int] = None,
    actual_rows: Optional[int] = None
)
```

**Attributes:**
- `required_rows` (Optional[int]): Required number of rows
- `actual_rows` (Optional[int]): Actual number of rows

---

### Other Exceptions

- `ConfigurationError`: Invalid configuration
- `DataLoaderError`: Data loading failure
- `VisualizationError`: Chart/plot creation failure
- `DatabaseError`: Database operation failure

---

## Registry

Module discovery and management.

### ModuleRegistry

Registry for analysis modules.

```python
from farm.analysis.registry import ModuleRegistry
```

#### Methods

##### `register(module: AnalysisModule) -> None`

Register an analysis module.

**Example:**
```python
from farm.analysis.registry import registry

registry.register(my_module)
```

##### `unregister(name: str) -> None`

Unregister a module by name.

##### `get(name: str) -> AnalysisModule`

Get a module by name.

**Raises:**
- `ModuleNotFoundError`: If module not found

##### `get_optional(name: str) -> Optional[AnalysisModule]`

Get a module, returning None if not found.

##### `get_module_names() -> List[str]`

Get list of all registered module names.

##### `get_all() -> Dict[str, AnalysisModule]`

Get all registered modules.

##### `clear() -> None`

Clear all registered modules.

##### `list_modules() -> str`

Get formatted string listing all modules.

---

### Global Registry

```python
from farm.analysis.registry import registry

# Global registry instance
```

---

### Utility Functions

#### register_modules

```python
from farm.analysis.registry import register_modules

register_modules(
    config_env_var: str = "FARM_ANALYSIS_MODULES",
    *,
    config_service: IConfigService
) -> int
```

Register analysis modules from configuration.

**Returns:**
- `int`: Number of successfully registered modules

---

#### Convenience Functions

```python
from farm.analysis.registry import (
    get_module,
    get_module_names,
    list_modules
)

# Get module by name
module = get_module("population")

# Get all module names
names = get_module_names()

# Get formatted listing
listing = list_modules()
```

---

## Common Utilities

Shared utility functions for analysis.

### Statistical Functions

```python
from farm.analysis.common.utils import (
    calculate_statistics,
    calculate_trend,
    calculate_rolling_mean,
    normalize_dict
)
```

#### calculate_statistics

```python
calculate_statistics(data: np.ndarray) -> Dict[str, float]
```

Calculate comprehensive statistics for data.

**Returns:**
- `Dict[str, float]`: Statistics (mean, median, std, min, max, q25, q75)

**Example:**
```python
stats = calculate_statistics(df['population'].values)
print(f"Mean: {stats['mean']}, Std: {stats['std']}")
```

#### calculate_trend

```python
calculate_trend(data: np.ndarray) -> float
```

Calculate linear trend (slope) of data.

**Returns:**
- `float`: Trend coefficient (positive = increasing, negative = decreasing)

#### calculate_rolling_mean

```python
calculate_rolling_mean(data: np.ndarray, window: int) -> np.ndarray
```

Calculate rolling mean with specified window.

**Parameters:**
- `data` (np.ndarray): Input data
- `window` (int): Window size

**Returns:**
- `np.ndarray`: Rolling mean values

#### normalize_dict

```python
normalize_dict(d: Dict[str, float]) -> Dict[str, float]
```

Normalize dictionary values to proportions (sum to 1).

---

### Data Processing Functions

```python
from farm.analysis.common.utils import (
    validate_required_columns,
    align_time_series,
    handle_missing_data
)
```

#### validate_required_columns

```python
validate_required_columns(df: pd.DataFrame, columns: List[str]) -> None
```

Validate DataFrame has required columns.

**Raises:**
- `ValueError`: If columns are missing

#### align_time_series

```python
align_time_series(data_list: List[np.ndarray]) -> np.ndarray
```

Align multiple time series to same length (pad with last value).

**Returns:**
- `np.ndarray`: Aligned time series matrix

#### handle_missing_data

```python
handle_missing_data(
    df: pd.DataFrame,
    strategy: str = 'drop'
) -> pd.DataFrame
```

Handle missing data with specified strategy.

**Parameters:**
- `strategy` (str): 'drop', 'fill_mean', 'fill_zero'

---

### File System Functions

```python
from farm.analysis.common.utils import (
    create_output_subdirs,
    find_database_path,
    save_analysis_results
)
```

#### create_output_subdirs

```python
create_output_subdirs(
    base_path: Path,
    subdirs: List[str]
) -> Dict[str, Path]
```

Create output subdirectories.

**Returns:**
- `Dict[str, Path]`: Mapping of subdir names to paths

#### find_database_path

```python
find_database_path(experiment_path: Path) -> Path
```

Find database path in experiment directory.

**Raises:**
- `FileNotFoundError`: If database not found

#### save_analysis_results

```python
save_analysis_results(
    results: Dict[str, Any],
    filename: str,
    output_path: Path
) -> Path
```

Save analysis results to JSON file.

---

### Plotting Functions

```python
from farm.analysis.common.utils import (
    setup_plot_figure,
    save_plot_figure,
    get_agent_type_colors,
    normalize_agent_type_names
)
```

#### setup_plot_figure

```python
setup_plot_figure(n_plots: int = 1) -> Tuple[Figure, Union[Axes, List[Axes]]]
```

Set up matplotlib figure with consistent styling.

**Returns:**
- `Tuple`: (figure, axes) or (figure, [axes_list])

#### save_plot_figure

```python
save_plot_figure(
    fig: Figure,
    output_path: Path,
    filename: str
) -> Path
```

Save plot figure to file.

**Returns:**
- `Path`: Path where figure was saved

#### get_agent_type_colors

```python
get_agent_type_colors() -> Dict[str, str]
```

Get standard agent type color scheme.

**Returns:**
- `Dict[str, str]`: Mapping of agent types to colors

#### normalize_agent_type_names

```python
normalize_agent_type_names(names: List[str]) -> List[str]
```

Normalize agent type names (e.g., 'SystemAgent' -> 'system').

---

### Metrics Functions

```python
from farm.analysis.common.metrics import (
    get_valid_numeric_columns,
    split_and_compare_groups,
    analyze_correlations,
    group_and_analyze,
    find_top_correlations
)
```

#### get_valid_numeric_columns

```python
get_valid_numeric_columns(
    df: pd.DataFrame,
    column_list: List[str]
) -> List[str]
```

Filter to only numeric columns.

#### split_and_compare_groups

```python
split_and_compare_groups(
    df: pd.DataFrame,
    split_column: str,
    split_value: Optional[float] = None,
    metrics: Optional[List[str]] = None,
    split_method: str = "median"
) -> Dict[str, Dict[str, Dict[str, float]]]
```

Split DataFrame into high/low groups and compare metrics.

**Parameters:**
- `split_method` (str): "median" or "mean"

**Returns:**
- `Dict`: Comparison results with means, differences, percent differences

#### analyze_correlations

```python
analyze_correlations(
    df: pd.DataFrame,
    target_column: str,
    metric_columns: Optional[List[str]] = None,
    min_data_points: int = 5,
    filter_condition: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> Dict[str, float]
```

Analyze correlations between target and metric columns.

**Returns:**
- `Dict[str, float]`: Correlation coefficients

#### group_and_analyze

```python
group_and_analyze(
    df: pd.DataFrame,
    group_column: str,
    group_values: List[str],
    analysis_func: Callable[[pd.DataFrame], Dict[str, Any]],
    min_group_size: int = 5
) -> Dict[str, Dict[str, Any]]
```

Group DataFrame and apply analysis function to each group.

#### find_top_correlations

```python
find_top_correlations(
    df: pd.DataFrame,
    target_column: str,
    metric_columns: Optional[List[str]] = None,
    top_n: int = 5,
    min_correlation: float = 0.1
) -> Dict[str, Dict[str, float]]
```

Find top positive and negative correlations.

**Returns:**
- `Dict`: {"top_positive": {...}, "top_negative": {...}}

---

## Analysis Modules

Built-in analysis modules available in the system.

### Population Module

Analyze population dynamics and agent composition.

```python
from farm.analysis.population.module import population_module
```

**Module Name**: `"population"`

**Function Groups**:
- `"all"`: All functions
- `"plots"`: Visualization functions
- `"metrics"`: Computation functions

---

### Resources Module

Analyze resource distribution and consumption.

```python
from farm.analysis.resources.module import resources_module
```

**Module Name**: `"resources"`

---

### Actions Module

Analyze action patterns and success rates.

```python
from farm.analysis.actions.module import actions_module
```

**Module Name**: `"actions"`

---

### Agents Module

Analyze individual agent behavior and lifespans.

```python
from farm.analysis.agents.module import agents_module
```

**Module Name**: `"agents"`

---

### Learning Module

Analyze learning performance and curves.

```python
from farm.analysis.learning.module import learning_module
```

**Module Name**: `"learning"`

---

### Spatial Module

Analyze spatial patterns and movement.

```python
from farm.analysis.spatial.module import spatial_module
```

**Module Name**: `"spatial"`

---

### Temporal Module

Analyze temporal patterns and efficiency.

```python
from farm.analysis.temporal.module import temporal_module
```

**Module Name**: `"temporal"`

---

### Combat Module

Analyze combat metrics and patterns.

```python
from farm.analysis.combat.module import combat_module
```

**Module Name**: `"combat"`

---

### Legacy Modules

These modules maintain backward compatibility:

- **Dominance** (`"dominance"`)
- **Genesis** (`"genesis"`)
- **Advantage** (`"advantage"`)
- **Social Behavior** (`"social_behavior"`)
- **Significant Events** (`"significant_events"`)
- **Comparative** (`"comparative"`)

---

## Type Definitions

Common type aliases used throughout the API.

```python
from farm.analysis.protocols import (
    AnalysisFunctionDict,
    AnalysisResults,
    ChartDict
)
```

### Type Aliases

```python
AnalysisFunctionDict = Dict[str, AnalysisFunction]
AnalysisResults = Dict[str, Any]
ChartDict = Dict[str, Any]
```

---

## Examples

### Complete Analysis Pipeline

```python
from pathlib import Path
from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.core.services import EnvConfigService

# Initialize service
config_service = EnvConfigService()
service = AnalysisService(config_service)

# Create request
request = AnalysisRequest(
    module_name="population",
    experiment_path=Path("data/experiment_001"),
    output_path=Path("results/population"),
    group="all",
    enable_caching=True
)

# Run analysis
result = service.run(request)

if result.success:
    print(f"✓ Completed in {result.execution_time:.2f}s")
    print(f"Results: {result.output_path}")
    print(f"Data shape: {result.dataframe.shape}")
else:
    print(f"✗ Failed: {result.error}")
```

### Custom Analysis Module

```python
from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor, make_analysis_function
from farm.analysis.validation import ColumnValidator, DataQualityValidator, CompositeValidator
import pandas as pd
from pathlib import Path

# Data processor
def process_custom_data(experiment_path: Path, **kwargs) -> pd.DataFrame:
    # Load and process your data
    return pd.DataFrame(...)

# Analysis function
def analyze_custom(df: pd.DataFrame, ctx, **kwargs):
    # Perform analysis
    results = df.groupby('category').agg({'value': ['mean', 'std']})
    
    # Save results
    output_file = ctx.get_output_file("stats.csv")
    results.to_csv(output_file)
    
    ctx.report_progress("Analysis complete", 1.0)

# Module class
class CustomModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(
            name="custom",
            description="Custom analysis module"
        )
        
        # Set up validation
        validator = CompositeValidator([
            ColumnValidator(required_columns=['category', 'value']),
            DataQualityValidator(min_rows=10)
        ])
        self.set_validator(validator)
    
    def register_functions(self):
        self._functions = {
            "analyze": make_analysis_function(analyze_custom)
        }
        self._groups = {
            "all": list(self._functions.values())
        }
    
    def get_data_processor(self):
        return SimpleDataProcessor(process_custom_data)

# Register and use
from farm.analysis.registry import registry

custom_module = CustomModule()
registry.register(custom_module)
```

---

## See Also

- [README.md](../../farm/analysis/README.md) - User guide and quick start
- [ARCHITECTURE.md](../../farm/analysis/ARCHITECTURE.md) - Architecture documentation
- [Examples](../../examples/analysis_example.py) - Working code examples
- [Tests](../../tests/analysis/) - Test suite with usage examples

---

**Generated**: 2025-10-04  
**Version**: 2.0.0  
**License**: See project LICENSE
