# AnalysisController API Documentation

The `AnalysisController` provides centralized control over analysis execution with lifecycle management, progress tracking, and thread-safe state management - following the same pattern as `SimulationController`.

## Features

- **Lifecycle Management**: Start, pause, resume, and stop analysis jobs
- **Progress Tracking**: Real-time progress updates via callbacks
- **Thread-Safe**: Run analyses in background threads with safe state access
- **Caching**: Automatic result caching for faster repeated analyses
- **Context Manager**: Automatic resource cleanup
- **Module Discovery**: List and inspect available analysis modules

## Quick Start

```python
from pathlib import Path
from farm.analysis.service import AnalysisRequest
from farm.api.analysis_controller import AnalysisController
from farm.core.services import EnvConfigService

# Create controller
config_service = EnvConfigService()
controller = AnalysisController(config_service)

# Create analysis request
request = AnalysisRequest(
    module_name="genesis",
    experiment_path=Path("results/experiment_1"),
    output_path=Path("results/analysis/genesis"),
    group="all"
)

# Run analysis
try:
    controller.initialize_analysis(request)
    controller.start()
    
    # Wait for completion
    while controller.is_running:
        state = controller.get_state()
        print(f"Progress: {state['progress']*100:.1f}%")
        time.sleep(1)
    
    # Get results
    result = controller.get_result()
    if result.success:
        print(f"Complete! Output: {result.output_path}")
finally:
    controller.cleanup()
```

## API Endpoints

The controller is integrated into the FastAPI server with the following endpoints:

### Run Analysis

```http
POST /api/analysis/{module_name}
```

**Request Body:**
```json
{
  "experiment_path": "results/experiment_1",
  "output_path": "results/analysis/genesis",
  "group": "all",
  "processor_kwargs": {},
  "analysis_kwargs": {}
}
```

**Response:**
```json
{
  "status": "accepted",
  "message": "Analysis started with ID: 20240102_153045"
}
```

### Get Analysis Status

```http
GET /api/analysis/{analysis_id}/status
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "analysis_id": "20240102_153045",
    "module_name": "genesis",
    "status": "running",
    "progress": 0.65,
    "message": "Processing data...",
    "is_running": true,
    "is_paused": false
  }
}
```

### Pause Analysis

```http
POST /api/analysis/{analysis_id}/pause
```

### Resume Analysis

```http
POST /api/analysis/{analysis_id}/resume
```

### Stop Analysis

```http
POST /api/analysis/{analysis_id}/stop
```

### List All Analyses

```http
GET /api/analyses
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "20240102_153045": {
      "module_name": "genesis",
      "status": "completed",
      "progress": 1.0,
      "execution_time": 45.3,
      "output_path": "results/analysis/genesis"
    }
  }
}
```

### List Available Modules

```http
GET /api/analysis/modules
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "name": "genesis",
      "description": "Analysis of agent genesis events",
      "supports_database": true
    },
    {
      "name": "dominance",
      "description": "Dominance hierarchy analysis",
      "supports_database": true
    }
  ]
}
```

### Get Module Info

```http
GET /api/analysis/modules/{module_name}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "name": "genesis",
    "description": "Analysis of agent genesis events",
    "function_groups": ["all", "plots", "metrics"],
    "functions": ["plot_genesis_timeline", "compute_genesis_stats"],
    "supports_database": true
  }
}
```

### Cleanup Old Analyses

```http
POST /api/analyses/cleanup
```

Manually trigger cleanup of completed analyses older than retention period.

**Response:**
```json
{
  "status": "success",
  "message": "Cleaned up 5 old analyses",
  "removed_count": 5,
  "remaining_count": 15
}
```

### Get Analysis Statistics

```http
GET /api/analyses/stats
```

Get system-level statistics about analysis resource usage.

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_analyses": 20,
    "by_status": {
      "completed": 15,
      "running": 3,
      "error": 2
    },
    "concurrent_limit": 10,
    "running_count": 3,
    "available_slots": 7,
    "retention_hours": 24,
    "max_completed_retention": 100
  }
}
```

## Usage Examples

### With Callbacks

```python
def on_progress(message: str, progress: float):
    print(f"[{progress*100:.1f}%] {message}")

def on_status(status: str):
    print(f"Status changed: {status}")

controller = AnalysisController(config_service)
controller.register_progress_callback("progress", on_progress)
controller.register_status_callback("status", on_status)

request = AnalysisRequest(...)
controller.initialize_analysis(request)
controller.start()
```

### With Context Manager

```python
with AnalysisController(config_service) as controller:
    request = AnalysisRequest(...)
    controller.initialize_analysis(request)
    controller.start()
    
    # Use wait_for_completion instead of polling
    if controller.wait_for_completion(timeout=300):
        result = controller.get_result()
        print(f"Complete! {result.output_path}")
    else:
        print("Analysis timed out")
# Cleanup happens automatically
```

### Pause and Resume

```python
controller.start()
time.sleep(5)

# Pause for a moment
controller.pause()
print("Paused - doing something else...")
time.sleep(2)

# Resume
controller.start()
```

### Custom Parameters

```python
request = AnalysisRequest(
    module_name="dominance",
    experiment_path=Path("results/exp_1"),
    output_path=Path("results/analysis/dominance"),
    group="plots",
    processor_kwargs={
        "save_to_db": True,
        "verbose": True
    },
    analysis_kwargs={
        "plot_distribution": {
            "bins": 50,
            "figsize": (12, 8)
        }
    },
    enable_caching=True,
    force_refresh=False
)
```

### Discover Modules

```python
# List all available modules
modules = controller.list_available_modules()
for module in modules:
    print(f"{module['name']}: {module['description']}")

# Get detailed info
info = controller.get_module_info("genesis")
print(f"Function groups: {info['function_groups']}")
print(f"Functions: {info['functions']}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Server                        │
│  (server.py)                                            │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ HTTP/WebSocket
                 │
┌────────────────▼────────────────────────────────────────┐
│              AnalysisController                         │
│  (analysis_controller.py)                               │
│                                                          │
│  • Lifecycle management (start/pause/stop)              │
│  • Progress tracking & callbacks                        │
│  • Thread-safe state management                         │
│  • Background execution                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ delegates to
                 │
┌────────────────▼────────────────────────────────────────┐
│              AnalysisService                            │
│  (analysis/service.py)                                  │
│                                                          │
│  • Request validation                                   │
│  • Result caching                                       │
│  • Batch execution                                      │
│  • Error handling                                       │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ uses
                 │
┌────────────────▼────────────────────────────────────────┐
│           BaseAnalysisModule                            │
│  (analysis/core.py)                                     │
│                                                          │
│  • Data processing                                      │
│  • Function execution                                   │
│  • Database integration                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ implements
                 │
┌────────────────▼────────────────────────────────────────┐
│         Concrete Analysis Modules                       │
│  (analysis/genesis/, analysis/dominance/, etc.)         │
│                                                          │
│  • Module-specific data processors                      │
│  • Analysis functions (plots, metrics)                  │
│  • Function groups                                      │
└─────────────────────────────────────────────────────────┘
```

## Comparison with SimulationController

Both controllers follow the same design pattern:

| Feature | SimulationController | AnalysisController |
|---------|---------------------|-------------------|
| Lifecycle | start/pause/stop | start/pause/stop |
| Threading | Background thread | Background thread |
| Callbacks | step + status | progress + status |
| State | get_state() | get_state() |
| Cleanup | cleanup() / context manager | cleanup() / context manager |
| ID Generation | timestamp-based | timestamp-based |

**Key Differences:**
- **SimulationController**: Manages simulation steps, agent state, environment
- **AnalysisController**: Manages analysis execution, data processing, results

## Best Practices

1. **Always use cleanup**: Use `try/finally` or context manager to ensure cleanup
2. **Register callbacks early**: Set up callbacks before calling `start()`
3. **Check results**: Always check `result.success` before using result data
4. **Use caching wisely**: Set `force_refresh=True` when input data changes
5. **Thread safety**: Don't modify controller state from callbacks
6. **Progress callbacks**: Keep progress callbacks fast (< 100ms)
7. **Error handling**: Catch and handle exceptions in your callbacks

## Error Handling

```python
from farm.analysis.exceptions import (
    ModuleNotFoundError,
    ConfigurationError,
    AnalysisError
)

try:
    controller.initialize_analysis(request)
except ModuleNotFoundError as e:
    print(f"Module not found: {e.module_name}")
    print(f"Available: {e.available_modules}")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Thread Safety

The controller is designed to be thread-safe:

- **Safe to call from any thread**: All public methods use locks
- **Background execution**: Analysis runs in separate thread
- **Callbacks**: May be called from background thread
- **State access**: `get_state()` returns safe copy

## Performance Tips

1. **Enable caching**: Set `enable_caching=True` for repeated analyses
2. **Use appropriate groups**: Run only needed function groups
3. **Batch processing**: Use `AnalysisService.run_batch()` for multiple analyses
4. **Database mode**: Use database for large datasets
5. **Progress callbacks**: Don't spam updates (< 10 Hz recommended)
6. **Resource management**: System automatically cleans up analyses older than 24 hours
7. **Concurrency limits**: Default limit of 10 concurrent analyses prevents resource exhaustion

## Resource Management

The system automatically manages resources to prevent memory leaks:

- **Automatic Cleanup**: Completed analyses older than `ANALYSIS_RETENTION_HOURS` (default: 24) are automatically removed
- **Max Retention**: Maximum of `MAX_COMPLETED_ANALYSES` (default: 100) completed analyses are kept
- **Concurrency Limiting**: Maximum of `MAX_CONCURRENT_ANALYSES` (default: 10) can run simultaneously
- **Manual Cleanup**: Use `POST /api/analyses/cleanup` to manually trigger cleanup
- **Monitoring**: Use `GET /api/analyses/stats` to check resource usage

### Configuration

Configure resource limits in `server.py`:

```python
MAX_COMPLETED_ANALYSES = 100      # Max completed analyses to retain
ANALYSIS_RETENTION_HOURS = 24     # Hours to keep completed analyses
MAX_CONCURRENT_ANALYSES = 10      # Max concurrent running analyses
```

## Troubleshooting

### Analysis stuck in "pending"
- Check if background task started
- Look for initialization errors in logs

### "Module not found" error
- Use `list_available_modules()` to check available modules
- Ensure module is registered in `farm/analysis/registry.py`

### Cache not working
- Ensure `enable_caching=True` in request
- Check cache directory permissions
- Use `force_refresh=True` to bypass cache

### Thread timeout on stop
- Long-running functions may not respond to stop immediately
- Increase timeout or implement progress callbacks

## See Also

- `examples/analysis_controller_example.py` - Complete working examples
- `farm/analysis/service.py` - Service layer implementation
- `farm/analysis/core.py` - Base module implementation
- `farm/api/simulation_controller.py` - Similar pattern for simulations
