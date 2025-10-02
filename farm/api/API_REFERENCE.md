# Unified AgentFarm API — Reference

This reference documents the public Python API provided by the `farm.api` module. It focuses on the programmatic interface used to create sessions, run simulations/experiments, validate/generate configurations, subscribe to events, and retrieve results.

Version: 1.0.0

## Quick Start

```python
from farm.api import AgentFarmController

controller = AgentFarmController(workspace_path="my_workspace")
session_id = controller.create_session("My Research", "Exploring behaviors")

# Create and start a simulation
config = {
    "name": "Basic Test",
    "steps": 200,
    "environment": {"width": 80, "height": 80, "resources": 40},
    "agents": {"system_agents": 10, "independent_agents": 10}
}
simulation_id = controller.create_simulation(session_id, config)
controller.start_simulation(session_id, simulation_id)

status = controller.get_simulation_status(session_id, simulation_id)
print(status.progress_percentage)

results = controller.get_simulation_results(session_id, simulation_id)
print(results.final_agent_count)

controller.cleanup()
```

## Public Surface

Importable symbols from `farm.api`:

- AgentFarmController
- SessionManager
- UnifiedAdapter
- ConfigTemplateManager
- Data models: SessionInfo, SimulationStatus, SimulationStatusInfo, SimulationResults, ExperimentStatus, ExperimentStatusInfo, ExperimentResults, ConfigTemplate, ValidationResult, AnalysisResults, ComparisonResults, Event, EventSubscription

---

## AgentFarmController

```python
AgentFarmController(workspace_path: Optional[str] = None)
```

Top-level orchestrator for sessions, simulations, experiments, configuration, analysis, and events. Stores state under `workspace_path`.

### Session Management
- `create_session(name: str, description: str = "") -> str`
- `get_session(session_id: str) -> Optional[SessionInfo]`
- `list_sessions() -> List[SessionInfo]`
- `delete_session(session_id: str, delete_files: bool = False) -> bool`
- `archive_session(session_id: str) -> bool` (via `SessionManager`)
- `restore_session(session_id: str) -> bool` (via `SessionManager`)
- `get_session_stats(session_id: str) -> Optional[Dict[str, Any]]`
- `list_simulations(session_id: str) -> List[str]`
- `list_experiments(session_id: str) -> List[str]`

### Simulation Control
- `create_simulation(session_id: str, config: Dict[str, Any]) -> str`
- `start_simulation(session_id: str, simulation_id: str) -> SimulationStatusInfo`
- `pause_simulation(session_id: str, simulation_id: str) -> SimulationStatusInfo`
- `resume_simulation(session_id: str, simulation_id: str) -> SimulationStatusInfo`
- `stop_simulation(session_id: str, simulation_id: str) -> SimulationStatusInfo`
- `get_simulation_status(session_id: str, simulation_id: str) -> SimulationStatusInfo`
- `get_simulation_results(session_id: str, simulation_id: str) -> SimulationResults`

### Experiment Control
- `create_experiment(session_id: str, config: Dict[str, Any]) -> str`
- `start_experiment(session_id: str, experiment_id: str) -> ExperimentStatusInfo`
- `get_experiment_status(session_id: str, experiment_id: str) -> ExperimentStatusInfo`
- `get_experiment_results(session_id: str, experiment_id: str) -> ExperimentResults`

### Configuration Management
- `get_available_configs() -> List[ConfigTemplate]`
- `validate_config(config: Dict[str, Any]) -> ValidationResult`
- `create_config_from_template(template_name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`

### Analysis & Comparison
- `analyze_simulation(session_id: str, simulation_id: str) -> AnalysisResults`
- `compare_simulations(session_id: str, simulation_ids: List[str]) -> ComparisonResults`
- `generate_visualization(session_id: str, simulation_id: str, viz_type: str) -> str`

### Event Monitoring
- `subscribe_to_events(session_id: str, event_types: List[str], simulation_id: Optional[str] = None, experiment_id: Optional[str] = None) -> str`
- `get_event_history(session_id: str, subscription_id: str) -> List[Event]`

### Resource Management
- `cleanup() -> None` (cleans up adapters and open resources)

---

## Data Models

All models are dataclasses and provide a `to_dict()` helper where appropriate.

### SessionInfo
```python
session_id: str
name: str
description: str
created_at: datetime
status: SessionStatus  # "active" | "archived" | "deleted"
simulations: List[str]
experiments: List[str]
metadata: Dict[str, Any]
```

### SimulationStatus (Enum)
`"created" | "running" | "paused" | "completed" | "error" | "stopped"`

### SimulationStatusInfo
```python
simulation_id: str
status: SimulationStatus
current_step: int
total_steps: int
progress_percentage: float
start_time: Optional[datetime]
end_time: Optional[datetime]
error_message: Optional[str]
metadata: Dict[str, Any]  # e.g., agent_count, resource_count
```

### SimulationResults
```python
simulation_id: str
status: SimulationStatus
total_steps: int
final_agent_count: int
final_resource_count: int
metrics: Dict[str, Any]  # duration_seconds, steps_per_second, ...
data_files: List[str]
analysis_available: bool
metadata: Dict[str, Any]
```

### ExperimentStatus (Enum)
`"created" | "running" | "completed" | "error" | "stopped"`

### ExperimentStatusInfo
```python
experiment_id: str
status: ExperimentStatus
current_iteration: int
total_iterations: int
progress_percentage: float
start_time: Optional[datetime]
end_time: Optional[datetime]
error_message: Optional[str]
metadata: Dict[str, Any]
```

### ExperimentResults
```python
experiment_id: str
status: ExperimentStatus
total_iterations: int
completed_iterations: int
results_summary: Dict[str, Any]
data_files: List[str]
analysis_available: bool
metadata: Dict[str, Any]
```

### ConfigTemplate
```python
name: str
description: str
category: ConfigCategory  # simulation | experiment | research
parameters: Dict[str, Any]
required_fields: List[str]
optional_fields: List[str]
examples: List[Dict[str, Any]]
```

### ValidationResult
```python
is_valid: bool
errors: List[str]
warnings: List[str]
suggestions: List[str]
validated_config: Optional[Dict[str, Any]]
```

### AnalysisResults
```python
analysis_id: str
analysis_type: str
summary: Dict[str, Any]
detailed_results: Dict[str, Any]
output_files: List[str]
charts: List[str]
metadata: Dict[str, Any]
```

### ComparisonResults
```python
comparison_id: str
simulation_ids: List[str]
comparison_summary: Dict[str, Any]
detailed_comparison: Dict[str, Any]
output_files: List[str]
charts: List[str]
metadata: Dict[str, Any]
```

### Event / EventSubscription
```python
Event: { event_id, event_type, timestamp, session_id, simulation_id?, experiment_id?, data, message }
EventSubscription: { subscription_id, session_id, event_types, simulation_id?, experiment_id?, created_at, active }
```

---

## Configuration Templates and Validation

```python
from farm.api import AgentFarmController

controller = AgentFarmController()
templates = controller.get_available_configs()

config = controller.create_config_from_template(
    "basic_simulation", {"steps": 500, "agents": {"system_agents": 8}}
)

validation = controller.validate_config(config)
assert validation.is_valid, validation.errors
```

Available built-in templates include: `basic_simulation`, `combat_simulation`, `research_simulation`, `basic_experiment`, `parameter_sweep`.

---

## CLI

Run demos without writing code:

```bash
python -m farm.api.cli --all
python -m farm.api.cli --demo simulation
python -m farm.api.cli --workspace my_ws --demo experiment
```

---

## HTTP Server (optional adapter)

`farm/api/server.py` exposes a small Flask/SocketIO server for interactive use.

REST endpoints (subject to change):

- `POST /api/simulation/new` — create and run a simulation (accepts JSON config)
- `GET /api/simulation/<sim_id>/step/<int:step>` — retrieve state for a step
- `GET /api/simulation/<sim_id>/analysis` — run analysis
- `POST /api/analysis/<module_name>` — run analysis module
- `GET /api/simulations` — list active simulations (in-memory)
- `GET /api/simulation/<sim_id>/export` — export data

WebSocket events:

- `connect` / `disconnect`
- `subscribe_simulation` — subscribe by `sim_id`

Note: The HTTP server is a convenience adapter. The canonical programmatic API is `AgentFarmController`.

---

## Serialization

Most response objects are dataclasses with `.to_dict()`; you can also serialize via `dataclasses.asdict()` or custom logic.

---

## Exceptions

- `ValueError` for invalid identifiers or configurations
- `RuntimeError` for execution-time failures
- Other exceptions may propagate from underlying subsystems (database, analysis)

---

## Stability and Versioning

This API aims for semantic versioning. Check `farm/api/__init__.py` for `__version__`.

