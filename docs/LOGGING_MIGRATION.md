# Logging Migration Checklist

This document provides a step-by-step guide for migrating existing code to use the new structured logging system.

## Phase 1: Foundation ✅ COMPLETED

- [x] Add structlog dependencies to requirements.txt
- [x] Create centralized logging configuration module (`farm/utils/logging_config.py`)
- [x] Create logging utilities and helpers (`farm/utils/logging_utils.py`)
- [x] Update `farm/utils/__init__.py` to export logging utilities
- [x] Update `main.py` to use structlog
- [x] Update `run_simulation.py` to use structlog
- [x] Create documentation (`docs/logging_guide.md`)
- [x] Create examples (`examples/logging_examples.py`)

## Phase 2: Core Modules (Next Steps)

### High Priority

#### 2.1 Core Simulation (`farm/core/`)

- [ ] `farm/core/simulation.py`
  - [ ] Replace `logging.basicConfig()` with `configure_logging()`
  - [ ] Replace `logger = logging.getLogger(__name__)` with `get_logger(__name__)`
  - [ ] Update all log calls to use event-style logging
  - [ ] Add simulation context binding
  - [ ] Use `log_simulation()` context manager

- [ ] `farm/core/environment.py`
  - [ ] Replace logger initialization
  - [ ] Add environment context (environment_id, grid_size, etc.)
  - [ ] Structure step logging with `log_step()`
  - [ ] Add performance metrics for step processing

- [ ] `farm/core/agent.py`
  - [ ] Replace logger initialization
  - [ ] Use `AgentLogger` for agent-specific events
  - [ ] Add agent context binding (agent_id, agent_type)
  - [ ] Structure action logging with rich context
  - [ ] Add performance logging for decision-making

#### 2.2 Database Layer (`farm/database/`)

- [ ] `farm/database/session_manager.py`
  - [ ] Replace logger initialization
  - [ ] Structure transaction logging
  - [ ] Add timing for database operations
  - [ ] Add context for session lifecycle

- [ ] `farm/database/database.py`
  - [ ] Replace logger initialization
  - [ ] Structure query logging
  - [ ] Add performance metrics for queries
  - [ ] Add connection pool metrics

- [ ] `farm/database/data_logging.py`
  - [ ] Replace logger initialization
  - [ ] Structure buffer flush logging
  - [ ] Add metrics for buffer operations
  - [ ] Add sampling for high-frequency logs

#### 2.3 API Server (`farm/api/`)

- [ ] `farm/api/server.py`
  - [ ] Replace `logging.basicConfig()` with `configure_logging()`
  - [ ] Add request context binding (request_id, endpoint, method)
  - [ ] Structure HTTP response logging
  - [ ] Add performance metrics for API endpoints
  - [ ] Use Flask middleware for automatic request logging

### Medium Priority

#### 2.4 Runners (`farm/runners/`)

- [ ] `farm/runners/experiment_runner.py`
  - [ ] Replace `_setup_logging()` with `configure_logging()`
  - [ ] Use `log_experiment()` context manager
  - [ ] Add experiment-level context binding
  - [ ] Structure iteration logging

- [ ] `farm/runners/parallel_experiment_runner.py`
  - [ ] Update to use structured logging
  - [ ] Add process-level context
  - [ ] Structure parallel execution logging

- [ ] `farm/runners/batch_runner.py`
  - [ ] Update to use structured logging
  - [ ] Add batch context
  - [ ] Structure batch progress logging

#### 2.5 Controllers (`farm/controllers/`)

- [ ] `farm/controllers/simulation_controller.py`
  - [ ] Replace logger initialization
  - [ ] Add controller context
  - [ ] Structure state transition logging

- [ ] `farm/controllers/experiment_controller.py`
  - [ ] Replace logger initialization
  - [ ] Structure experiment lifecycle logging

#### 2.6 Memory System (`farm/memory/`)

- [ ] `farm/memory/redis_memory.py`
  - [ ] Replace logger initialization
  - [ ] Structure Redis operation logging
  - [ ] Add performance metrics for memory operations
  - [ ] Add sampling for high-frequency operations

### Low Priority

#### 2.7 Analysis Modules (`farm/analysis/`)

- [ ] `farm/analysis/core.py`
  - [ ] Replace logger initialization
  - [ ] Structure analysis pipeline logging

- [ ] `farm/analysis/service.py`
  - [ ] Replace logger initialization
  - [ ] Add analysis context binding

- [ ] `farm/analysis/dominance/pipeline.py`
  - [ ] Update logging calls to structured format
  - [ ] Add analysis stage context

- [ ] All other analysis modules
  - [ ] Systematic update of logging calls

#### 2.8 Decision Modules (`farm/core/decision/`)

- [ ] `farm/core/decision/decision.py`
  - [ ] Replace logger initialization
  - [ ] Structure decision-making logs
  - [ ] Add performance metrics

- [ ] `farm/core/decision/base_dqn.py`
  - [ ] Update to structured logging
  - [ ] Add training metrics

- [ ] `farm/core/decision/algorithms/`
  - [ ] Update all algorithm implementations

#### 2.9 Spatial & Resources

- [ ] `farm/core/spatial/index.py`
  - [ ] Replace logger initialization
  - [ ] Add spatial index rebuild metrics
  - [ ] Structure query logging

- [ ] `farm/core/resource_manager.py`
  - [ ] Update to structured logging
  - [ ] Add resource lifecycle logging

#### 2.10 Utilities & Tools

- [ ] `farm/charts/llm_client.py`
- [ ] `farm/config/monitor.py`
- [ ] Various utility modules

## Phase 3: Scripts & Analysis

### Scripts (`scripts/`)

- [ ] `scripts/validate_observation_pipeline.py`
- [ ] `scripts/significant_events.py`
- [ ] `scripts/reproduction_analysis.py`
- [ ] `scripts/genesis_analysis.py`
- [ ] `scripts/advantage_analysis.py`
- [ ] `scripts/database_utils.py`
- [ ] All other scripts

### Analysis Scripts (`analysis/`)

- [ ] `analysis/reproducibility.py`
- [ ] `analysis/simulation_analysis.py`
- [ ] `analysis/simulation_comparison.py`

## Phase 4: Testing & Documentation

### Testing

- [ ] Create logging configuration tests
- [ ] Test context propagation
- [ ] Test output formats (JSON, console)
- [ ] Performance impact testing
- [ ] Integration tests for logging

### Documentation

- [ ] Update CONTRIBUTING.md with logging guidelines
- [ ] Update README.md with logging information
- [ ] Create troubleshooting guide
- [ ] Document performance considerations

## Migration Guidelines

### For Each File

1. **Replace imports:**
   ```python
   # Old
   import logging
   logger = logging.getLogger(__name__)
   
   # New
   from farm.utils import get_logger
   logger = get_logger(__name__)
   ```

2. **Remove basicConfig:**
   ```python
   # Delete these lines
   logging.basicConfig(...)
   ```

3. **Update log calls:**
   ```python
   # Old
   logger.info(f"Simulation started with {num_agents} agents")
   
   # New
   logger.info("simulation_started", num_agents=num_agents)
   ```

4. **Add context binding:**
   ```python
   # For long-lived objects
   self.logger = get_logger(__name__).bind(
       component="spatial_index",
       version="2.0",
   )
   
   # For scoped operations
   with log_context(simulation_id=sim_id):
       # All logs include simulation_id
       process_simulation()
   ```

5. **Update error logging:**
   ```python
   # Old
   logger.error(f"Error: {e}", exc_info=True)
   
   # New
   logger.error(
       "operation_failed",
       error_type=type(e).__name__,
       error_message=str(e),
       exc_info=True,
   )
   ```

## Validation Checklist

For each migrated file, verify:

- [ ] No `import logging` statements (except in logging_config.py)
- [ ] No `logging.basicConfig()` calls
- [ ] No f-string or % formatting in log calls
- [ ] All log calls use event names (first positional arg)
- [ ] Context is properly bound where appropriate
- [ ] Error logging includes error_type and error_message
- [ ] Performance-critical sections use sampling if needed
- [ ] Sensitive data is not logged

## Testing After Migration

1. **Run the examples:**
   ```bash
   python examples/logging_examples.py
   ```

2. **Test console output:**
   ```bash
   python run_simulation.py --steps 100 --log-level DEBUG
   ```

3. **Test JSON output:**
   ```bash
   python run_simulation.py --steps 100 --json-logs
   cat logs/application.json.log | jq
   ```

4. **Test in production mode:**
   ```bash
   python run_simulation.py --environment production --steps 100
   ```

5. **Verify log analysis:**
   ```python
   import json
   import pandas as pd
   
   with open("logs/application.json.log") as f:
       logs = [json.loads(line) for line in f]
       df = pd.DataFrame(logs)
       print(df.groupby('level').size())
   ```

## Performance Considerations

- Use `LogSampler` for high-frequency logs (per-step, per-agent operations)
- Avoid logging large objects (truncate if necessary)
- Use appropriate log levels (DEBUG for verbose, INFO for important events)
- Consider disabling DEBUG logs in production
- Use context binding instead of including same fields in every log call

## Rollout Strategy

1. **Phase 1** (Complete): Foundation and entry points
2. **Phase 2** (Week 1-2): Core modules (simulation, environment, agent, database)
3. **Phase 3** (Week 3): Service layers and controllers
4. **Phase 4** (Week 4): Analysis modules and utilities
5. **Phase 5** (Week 5): Scripts and remaining modules
6. **Phase 6** (Week 6): Testing, documentation, cleanup

## Progress Tracking

- Total files with logging: **91**
- Files migrated: **2** (main.py, run_simulation.py)
- Remaining: **89**

Update this number as files are migrated.
