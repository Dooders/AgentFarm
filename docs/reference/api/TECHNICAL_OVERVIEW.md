# Unified AgentFarm API — Technical Overview

This document describes the architecture, execution model, components, data flow, and operational considerations for the `farm.api` module.

## Goals

- Provide a clean, unified programmatic interface to simulations and experiments
- Standardize session management and results retrieval
- Offer configuration templates and validation
- Enable analysis, comparison, and basic event monitoring

## High-Level Architecture

Components and responsibilities:

- `AgentFarmController` (orchestrator)
  - Public facade for sessions, simulations, experiments, configuration, analysis, events
  - Persists session metadata via `SessionManager`
  - Delegates execution to `UnifiedAdapter`

- `UnifiedAdapter` (integration layer)
  - Adapts existing controllers to the unified API
  - Tracks simulation/experiment registries under a given session path
  - Emits events, maintains event history, and provides status/results aggregation

- `SessionManager` (persistence for sessions)
  - Stores sessions under a workspace directory
  - Persists an index file (`sessions.json`) with metadata and IDs

- `SimulationController` (single-run executor)
  - Initializes environment and runs a simulation loop on a background thread
  - Coordinates environment, database, and callbacks

- `ExperimentController` (multi-iteration executor)
  - Runs a set of simulation iterations with variations
  - Organizes outputs and can trigger analysis per iteration and across iterations

- `ConfigTemplateManager` (templates and validation)
  - Provides built-in templates and validation logic
  - Converts user-friendly config dictionaries to `SimulationConfig`

- Data models (`models.py`)
  - Typed dataclasses and enums for sessions, statuses, results, analysis, events

- Optional adapters
  - CLI demos (`cli.py`)
  - Flask/SocketIO server (`server.py`)

## Execution Model

### Simulation lifecycle
1. `AgentFarmController.create_simulation()` delegates to `UnifiedAdapter.create_simulation()`
2. Adapter creates directories (e.g., `sessions/<session_id>/simulations/<simulation_id>`)
3. A `SimulationController` is created and stored; DB path is typically `simulation.db`
4. `start_simulation()` initializes and starts the controller’s background thread
5. `get_simulation_status()` returns a `SimulationStatusInfo` snapshot
6. `get_simulation_results()` aggregates environment state and generated artifacts

### Experiment lifecycle
1. `AgentFarmController.create_experiment()` delegates to `UnifiedAdapter`
2. Adapter creates directories (e.g., `sessions/<session_id>/experiments/<experiment_id>`)
3. `ExperimentController` executes configured iterations (optionally with analysis)
4. Status is reported via `ExperimentStatusInfo`; results enumerate files and summaries

### Events
- The adapter emits `Event` objects (e.g., `simulation_created`, `simulation_started`, `simulation_step`, `experiment_completed`)
- Subscribers filter history by event types and optional simulation/experiment IDs
- Event history is bounded (keeps recent items) to avoid unbounded memory growth

## Storage Layout

Within a `workspace_path` and `session_id`:

- `sessions/<session_id>/`
  - `simulations/<simulation_id>/simulation.db`
  - `experiments/<experiment_id>/iteration_*/simulation.db`
  - `comparisons/<comparison_id>/...`
  - `visualizations/`
  - `analysis/`

Session index: `sessions/sessions.json` (maintained by `SessionManager`).

## Configuration

User-facing configuration is a Python `Dict[str, Any]` shaped by templates:

```python
{
  "name": str,
  "steps": int,
  "environment": {"width": int, "height": int, "resources": int},
  "agents": {"system_agents": int, "independent_agents": int, "control_agents": int},
  "learning": {"enabled": bool, "algorithm": str},
  # experiment-level parameters as needed (e.g., iterations, variations)
}
```

The adapter converts this dictionary into a `SimulationConfig` via `ConfigTemplateManager.convert_to_simulation_config()`.

## Threading and Concurrency

- `SimulationController` runs the main loop on a background thread
- Step and status callbacks are invoked from the simulation thread
- `UnifiedAdapter` protects shared event history with a lock
- Long-running operations should not block request-handling threads in adapter/server layers

## Logging and Observability

- Uses `farm.utils.logging_config` for structured logging
- Controllers log lifecycle events (initialized, started, paused, completed)
- Analysis and comparative workflows log output paths and counts
- Consider integrating metrics (e.g., Prometheus) and traces for production deployments

## Error Handling

- Validation errors raise `ValueError`
- Runtime failures may raise `RuntimeError` or propagate exceptions from subsystems
- API-level methods should catch and enrich exceptions where it improves diagnostics

## Extensibility

- Add templates by extending `ConfigTemplateManager`
- Swap analysis modules or add new ones that operate on generated DBs and artifacts
- Introduce new adapters (e.g., job queues, different storage backends) behind the controller facade

## Deployment Considerations

- The included Flask/SocketIO server is suitable for demos/prototyping; production deployments should:
  - Run simulations asynchronously via a job system (threads or workers)
  - Expose health/readiness endpoints and metrics
  - Ensure authentication/authorization and narrow CORS
  - Persist state durably and reconstruct state on restart

## Directory Map (module)

- `farm/api/unified_controller.py` — public facade (`AgentFarmController`)
- `farm/api/unified_adapter.py` — integration, events, status/results
- `farm/api/session_manager.py` — session persistence (workspace, `sessions.json`)
- `farm/api/simulation_controller.py` — background simulation runner
- `farm/api/experiment_controller.py` — multi-iteration orchestration and analysis
- `farm/api/config_templates.py` — templates, validation, conversion helpers
- `farm/api/models.py` — dataclasses and enums
- `farm/api/cli.py` — demos and manual testing
- `farm/api/server.py` — optional HTTP/WS adapter

