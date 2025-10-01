# Analysis Module Architecture

## Overview

The analysis module system provides a modern, protocol-based architecture for creating and running analysis modules on simulation data. It follows SOLID principles and industry best practices.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Analysis Service                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Request   │  │   Cache      │  │   Result     │      │
│  │  Validation │  │   Manager    │  │   Builder    │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Module Registry                          │
│              (Dynamic Module Discovery)                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Analysis Module                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Base Module                        │   │
│  │  • Function Registration                            │   │
│  │  • Data Processing                                  │   │
│  │  • Validation                                       │   │
│  │  • Execution                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                             │                               │
│         ┌───────────────────┼───────────────────┐           │
│         ▼                   ▼                   ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Data      │    │  Analysis   │    │    Plot     │    │
│  │  Processor  │    │  Functions  │    │  Functions  │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Infrastructure                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Protocols   │  │  Validators  │  │  Exceptions  │     │
│  │  (Contracts) │  │  (Quality)   │  │  (Errors)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Protocols (`protocols.py`)

Define contracts using Python's Protocol system:

```python
@runtime_checkable
class AnalysisModule(Protocol):
    @property
    def name(self) -> str: ...
    
    @property
    def description(self) -> str: ...
    
    def get_data_processor(self) -> DataProcessor: ...
    def get_validator(self) -> Optional[DataValidator]: ...
    def get_analysis_functions(self, group: str) -> List[AnalysisFunction]: ...
```

**Benefits:**
- Structural typing (duck typing with type safety)
- No forced inheritance
- Easy to mock for testing
- IDE autocomplete support

### 2. Core Implementations (`core.py`)

Base classes that implement the protocols:

```python
class BaseAnalysisModule:
    """Base implementation with common functionality."""
    
    def __init__(self, name: str, description: str)
    def register_functions(self) -> None  # Abstract
    def get_data_processor(self) -> DataProcessor  # Abstract
    def run_analysis(...) -> tuple[Path, pd.DataFrame]
```

**Features:**
- Lazy registration
- Standardized execution
- Progress tracking
- Error handling
- Database support

### 3. Validation (`validation.py`)

Comprehensive data validation:

```python
ColumnValidator       # Schema validation
DataQualityValidator  # Quality checks
CompositeValidator    # Combine validators
```

**Capabilities:**
- Required columns check
- Type validation
- Null value limits
- Duplicate detection
- Value range validation
- Custom validators

### 4. Exceptions (`exceptions.py`)

Specific exception types:

```python
AnalysisError              # Base
├── DataValidationError    # Invalid data
├── ModuleNotFoundError    # Missing module
├── DataProcessingError    # Processing failure
├── AnalysisFunctionError  # Function error
├── ConfigurationError     # Config issue
├── InsufficientDataError  # Not enough data
├── VisualizationError     # Plot failure
└── DatabaseError          # DB operation failure
```

### 5. Registry (`registry.py`)

Module discovery and management:

```python
class ModuleRegistry:
    def register(self, module: AnalysisModule)
    def get(self, name: str) -> AnalysisModule
    def get_all(self) -> Dict[str, AnalysisModule]
    def list_modules(self) -> str
```

**Features:**
- Dynamic registration
- Environment-based discovery
- Fallback to built-ins
- Protocol compliance checking

### 6. Service Layer (`service.py`)

High-level API:

```python
class AnalysisService:
    def run(self, request: AnalysisRequest) -> AnalysisResult
    def run_batch(self, requests: List[...]) -> List[...]
    def get_module_info(self, name: str) -> Dict
    def clear_cache(self) -> int
```

**Features:**
- Request validation
- Result caching
- Progress tracking
- Batch processing
- Metadata management

### 7. Analysis Context (`common/context.py`)

Shared execution context:

```python
@dataclass
class AnalysisContext:
    output_path: Path
    config: Dict[str, Any]
    logger: logging.Logger
    progress_callback: Optional[Callable]
    
    def get_output_file(self, filename, subdir=None) -> Path
    def report_progress(self, message, progress) -> None
    def get_config(self, key, default=None) -> Any
```

## Data Flow

### Standard Analysis Flow

```
1. Request Creation
   AnalysisRequest(
       module_name="dominance",
       experiment_path=Path("data"),
       output_path=Path("results")
   )
   
2. Service Validation
   ├── Check module exists
   ├── Validate paths
   ├── Check function groups
   └── Generate cache key

3. Cache Check
   ├── If cached: return cached result
   └── If not: proceed

4. Data Processing
   ├── Load raw data
   ├── Transform data
   └── Validate data

5. Function Execution
   For each function in group:
   ├── Create context
   ├── Call function(df, ctx, **kwargs)
   ├── Handle errors
   └── Track progress

6. Result Building
   ├── Collect outputs
   ├── Cache results
   ├── Save summary
   └── Return AnalysisResult
```

### Function Execution Flow

```
Analysis Function
       │
       ▼
┌──────────────┐
│   Context    │ ← output_path, config, logger
└──────────────┘
       │
       ▼
┌──────────────┐
│  DataFrame   │ ← Validated input data
└──────────────┘
       │
       ▼
┌──────────────┐
│   Analysis   │ ← Custom logic
│     Logic    │
└──────────────┘
       │
       ▼
┌──────────────┐
│   Outputs    │ ← Save to ctx.output_path
└──────────────┘
       │
       ▼
┌──────────────┐
│   Progress   │ ← ctx.report_progress()
└──────────────┘
```

## Extension Points

### 1. Creating a New Module

```python
class MyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__(
            name="my_module",
            description="My custom analysis"
        )
        
        # Optional: Set validator
        self.set_validator(
            ColumnValidator(required_columns=[...])
        )
    
    def register_functions(self):
        self._functions = {
            "analyze": make_analysis_function(analyze_func),
            "plot": make_analysis_function(plot_func),
        }
        
        self._groups = {
            "all": list(self._functions.values()),
            "metrics": [self._functions["analyze"]],
            "plots": [self._functions["plot"]],
        }
    
    def get_data_processor(self):
        return SimpleDataProcessor(process_func)
    
    # Optional: Database support
    def supports_database(self):
        return True
    
    def get_db_filename(self):
        return "my_module.db"
```

### 2. Creating Custom Validators

```python
class CustomValidator:
    def validate(self, df: pd.DataFrame) -> None:
        # Custom validation logic
        if condition_fails:
            raise DataValidationError("Custom error")
    
    def get_required_columns(self) -> List[str]:
        return ["col1", "col2"]
```

### 3. Creating Custom Data Processors

```python
class CustomProcessor:
    def process(self, data: Any, **kwargs) -> pd.DataFrame:
        # Custom processing logic
        return processed_df
```

### 4. Chaining Processors

```python
processor = ChainedDataProcessor([
    CleaningProcessor(),
    NormalizationProcessor(),
    FeatureProcessor(),
])
```

## Design Patterns Used

### 1. Strategy Pattern
- Different validators interchangeable
- Different processors interchangeable
- Functions as strategies

### 2. Template Method Pattern
- `BaseAnalysisModule.run_analysis()` defines skeleton
- Subclasses fill in specific steps

### 3. Registry Pattern
- Central module registry
- Dynamic discovery
- Loose coupling

### 4. Dependency Injection
- Services injected via constructor
- Configuration injected
- No global state

### 5. Facade Pattern
- `AnalysisService` provides simple interface
- Hides complex subsystems

### 6. Composite Pattern
- `CompositeValidator` combines validators
- `ChainedDataProcessor` combines processors

### 7. Command Pattern
- `AnalysisRequest` encapsulates operation
- Can be queued, logged, undone

## Type Safety

### Protocol-Based Typing

```python
def analyze_with_module(module: AnalysisModule):
    # Type checker verifies protocol compliance
    processor = module.get_data_processor()
    validator = module.get_validator()
    functions = module.get_analysis_functions("all")
```

### Runtime Type Checking

```python
@runtime_checkable
class AnalysisModule(Protocol):
    # Can check at runtime
    if isinstance(obj, AnalysisModule):
        ...
```

## Error Handling Strategy

### 1. Fail Fast
- Validate requests immediately
- Check module exists upfront
- Verify paths before processing

### 2. Specific Exceptions
- Each error type has specific exception
- Rich error context
- Easy to catch and handle

### 3. Graceful Degradation
- Individual function failures don't stop analysis
- Errors logged and tracked
- Partial results still usable

### 4. Error Recovery
```python
try:
    result = service.run(request)
except ModuleNotFoundError as e:
    logger.error(f"Module not found: {e.module_name}")
    logger.info(f"Available: {e.available_modules}")
except DataValidationError as e:
    logger.error(f"Validation failed: {e}")
    logger.info(f"Missing: {e.missing_columns}")
    logger.info(f"Invalid: {e.invalid_columns}")
```

## Performance Considerations

### 1. Lazy Registration
- Functions registered on first use
- Faster startup
- Lower memory footprint

### 2. Caching
- File-based result cache
- Hash-based invalidation
- ~100x speedup for cached results

### 3. Batch Processing
- Process multiple analyses
- Better resource utilization
- Progress tracking

### 4. Streaming Support
- Large datasets handled efficiently
- Memory-conscious processing
- Chunked iteration

## Testing Strategy

### 1. Unit Tests
```
tests/analysis/
├── test_protocols.py      # Protocol compliance
├── test_validation.py     # Validator tests
├── test_core.py          # Core functionality
├── test_registry.py      # Registry tests
├── test_service.py       # Service layer tests
└── test_exceptions.py    # Exception tests
```

### 2. Integration Tests
```
tests/analysis/
└── test_integration.py   # End-to-end workflows
```

### 3. Test Fixtures
```
tests/analysis/conftest.py
├── sample_simulation_data
├── temp_output_dir
├── analysis_context
├── minimal_module
└── config_service_mock
```

## Security Considerations

### 1. Path Validation
- All paths validated before use
- No directory traversal
- Sandboxed output paths

### 2. Input Validation
- All data validated before processing
- Type checking
- Range validation

### 3. Error Messages
- No sensitive data in errors
- Safe for logging
- User-friendly messages

## Future Enhancements

### Planned
1. Async execution support
2. Distributed caching (Redis)
3. Real-time streaming
4. Web API wrapper
5. Result aggregation
6. Export to multiple formats

### Under Consideration
1. GPU acceleration
2. Parallel function execution
3. Database query optimization
4. Result versioning
5. A/B testing framework

## References

### Related Documents
- `README.md` - User guide
- `REFACTORING_SUMMARY.md` - What changed
- `examples/analysis_example.py` - Working examples

### Design Principles
- SOLID Principles
- Protocol-Oriented Programming
- Dependency Injection
- Composition over Inheritance

### External Resources
- [PEP 544 - Protocols](https://www.python.org/dev/peps/pep-0544/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
