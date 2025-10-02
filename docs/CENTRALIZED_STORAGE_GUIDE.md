# Centralized Storage Guide

## Overview

This guide explains how to use a centralized database for storing multiple simulations/experiments instead of creating a separate SQLite database for each run.

## Problem

Previously, each simulation created its own SQLite database file (e.g., `simulation_12345.db`). This approach has several drawbacks:
- **Difficult to compare experiments**: Data is scattered across many files
- **No temporal tracking**: Hard to analyze trends across multiple runs
- **Storage inefficiency**: Each database has overhead for schema and indexes
- **Query complexity**: Need to open multiple databases to analyze results

## Solution: ExperimentDatabase

The `ExperimentDatabase` class allows you to store many simulations in a single database file. Each simulation is tagged with a unique `simulation_id`, making it easy to:
- Query across multiple simulations
- Track experiments over time
- Analyze comparative results
- Store everything in one place

## Quick Start

### Basic Usage

```python
from farm.database.experiment_database import ExperimentDatabase
from farm.config import SimulationConfig

# Create a centralized database for your experiment
experiment_db = ExperimentDatabase(
    db_path="experiments/my_experiment.db",
    experiment_id="exp_001",
    config=config  # Optional: SimulationConfig for DB settings
)

# Run multiple simulations
for i in range(10):
    sim_id = f"sim_{i:03d}"
    
    # Create a context for this simulation
    sim_context = experiment_db.create_simulation_context(
        simulation_id=sim_id,
        parameters={"run": i, "seed": i * 100}
    )
    
    # Use sim_context.logger instead of db.logger
    # Log simulation data
    sim_context.logger.log_step(
        step_number=0,
        agent_states=agent_states,
        resource_states=resource_states,
        metrics=metrics
    )
    
    # When done, flush and update status
    sim_context.flush_all_buffers()
    experiment_db.update_simulation_status(
        simulation_id=sim_id,
        status="completed",
        results_summary={"final_agents": 100}
    )

# Close the database when all simulations are complete
experiment_db.close()
```

### Integration with Simulation Runner

Here's how to modify `run_simulation.py` to use a centralized database:

```python
from farm.database.experiment_database import ExperimentDatabase
import uuid
from datetime import datetime

# Create experiment database
experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
experiment_db = ExperimentDatabase(
    db_path=f"experiments/{experiment_id}.db",
    experiment_id=experiment_id,
    config=config
)

# For each simulation run
simulation_id = f"sim_{uuid.uuid4().hex[:8]}"
sim_context = experiment_db.create_simulation_context(
    simulation_id=simulation_id,
    parameters={"steps": num_steps, "seed": config.seed}
)

# Pass sim_context to your simulation
# Instead of: db = SimulationDatabase("sim.db")
# Use: db = sim_context (it has the same interface)

# After simulation completes
sim_context.flush_all_buffers()
experiment_db.update_simulation_status(
    simulation_id=simulation_id,
    status="completed"
)
```

## Database Schema

### Experiments Table
Stores metadata about each experiment (group of related simulations):

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | String | Unique experiment identifier |
| `name` | String | Human-readable name |
| `description` | String | Purpose of the experiment |
| `creation_date` | DateTime | When created |
| `status` | String | 'running', 'completed', etc. |
| `results_summary` | JSON | High-level findings |

### Simulations Table
Stores metadata about each simulation run:

| Column | Type | Description |
|--------|------|-------------|
| `simulation_id` | String | Unique simulation identifier |
| `experiment_id` | String | Parent experiment |
| `start_time` | DateTime | When started |
| `end_time` | DateTime | When completed |
| `status` | String | 'pending', 'running', 'completed' |
| `parameters` | JSON | Configuration used |
| `results_summary` | JSON | Key metrics |
| `simulation_db_path` | String | Database file path |

### Data Tables
All data tables (agents, agent_states, actions, etc.) include a `simulation_id` column that links them to a specific simulation.

## Querying Data

### Get All Simulations in an Experiment

```python
simulation_ids = experiment_db.get_simulation_ids()
print(f"Found {len(simulation_ids)} simulations")
```

### Query Data for a Specific Simulation

```python
from sqlalchemy import select
from farm.database.models import AgentStateModel

session = experiment_db.Session()

# Get agent states for a specific simulation
states = session.query(AgentStateModel).filter(
    AgentStateModel.simulation_id == "sim_001"
).all()

session.close()
```

### Compare Across Simulations

```python
import pandas as pd

# Get final population for each simulation
session = experiment_db.Session()

results = []
for sim_id in experiment_db.get_simulation_ids():
    final_step = session.query(SimulationStepModel).filter(
        SimulationStepModel.simulation_id == sim_id
    ).order_by(SimulationStepModel.step_number.desc()).first()
    
    if final_step:
        results.append({
            'simulation_id': sim_id,
            'final_population': final_step.total_agents,
            'final_resources': final_step.total_resources
        })

df = pd.DataFrame(results)
print(df.describe())

session.close()
```

## Advanced Usage

### Using In-Memory Database with Persistence

For fast simulations that you want to persist later:

```python
from farm.database.database import InMemorySimulationDatabase

# Run simulation in memory
in_memory_db = InMemorySimulationDatabase(
    memory_limit_mb=1000,
    config=config,
    simulation_id=sim_id
)

# ... run simulation ...

# Persist to centralized database
in_memory_db.persist_to_disk(
    db_path="experiments/exp_001.db"
)
```

### Batch Running Multiple Experiments

```python
from farm.database.experiment_database import ExperimentDatabase
import multiprocessing

def run_single_simulation(args):
    """Run one simulation with given parameters."""
    experiment_db_path, experiment_id, sim_id, params = args
    
    # Connect to shared database
    experiment_db = ExperimentDatabase(
        db_path=experiment_db_path,
        experiment_id=experiment_id
    )
    
    # Create simulation context
    sim_context = experiment_db.create_simulation_context(
        simulation_id=sim_id,
        parameters=params
    )
    
    # Run simulation
    # ... your simulation code ...
    
    # Clean up
    sim_context.flush_all_buffers()
    experiment_db.update_simulation_status(sim_id, "completed")
    experiment_db.close()

# Prepare simulation parameters
experiment_id = "param_sweep_001"
db_path = f"experiments/{experiment_id}.db"

tasks = []
for i, learning_rate in enumerate([0.01, 0.1, 0.5, 1.0]):
    for j, population in enumerate([10, 50, 100]):
        sim_id = f"sim_lr{i}_pop{j}"
        params = {"learning_rate": learning_rate, "population": population}
        tasks.append((db_path, experiment_id, sim_id, params))

# Run in parallel (be careful with SQLite write concurrency)
# Note: SQLite has limited write concurrency, consider sequential execution
for task in tasks:
    run_single_simulation(task)
```

## Migration Guide

### Converting Existing Code

**Before (separate databases):**
```python
from farm.database.database import SimulationDatabase

db = SimulationDatabase(f"simulation_{run_id}.db")
db.logger.log_step(...)
db.close()
```

**After (centralized database):**
```python
from farm.database.experiment_database import ExperimentDatabase

# Create once per experiment
experiment_db = ExperimentDatabase(
    db_path="experiments/my_experiment.db",
    experiment_id="exp_001"
)

# For each simulation
sim_context = experiment_db.create_simulation_context(
    simulation_id=f"sim_{run_id}"
)
sim_context.logger.log_step(...)
sim_context.flush_all_buffers()

# Close once at the end
experiment_db.close()
```

## Best Practices

1. **Use descriptive experiment IDs**: Include date and purpose (e.g., `exp_20250102_learning_rates`)

2. **Store one experiment per database**: Don't mix unrelated experiments in the same file

3. **Keep simulation_ids unique**: Use timestamps, UUIDs, or sequential numbers

4. **Flush buffers regularly**: Call `flush_all_buffers()` after each simulation to prevent data loss

5. **Update status**: Always update simulation status to track progress

6. **Document parameters**: Store all configuration in the `parameters` field for reproducibility

7. **Add results_summary**: Store key metrics for quick access without full queries

8. **Backup large databases**: Export important results to CSV/JSON periodically

## Performance Considerations

### Write Performance
- SQLite handles sequential writes well, but concurrent writes from multiple processes can cause locking
- For parallel simulations, consider using separate databases and merging later
- Use `PRAGMA` settings from config to optimize performance (WAL mode, cache size, etc.)

### Database Size
- Each simulation adds data proportional to agents × steps
- Monitor database file size and split experiments if it grows too large (>10GB)
- Use `VACUUM` periodically to reclaim space

### Query Performance
- The database includes indexes on `simulation_id` for fast filtering
- Always filter by `simulation_id` when querying specific runs
- Use aggregations at the database level instead of loading all data

## Troubleshooting

### Database Locked Errors
If you see "database is locked" errors:
- Ensure only one process writes at a time
- Check that `flush_all_buffers()` is called before closing
- Increase `busy_timeout` in PRAGMA settings

### Large Database Files
If the database file is too large:
- Export old simulations to CSV and remove them
- Split into multiple experiment databases
- Use compression for archived databases

### Query Performance Issues
If queries are slow:
- Ensure you're filtering by `simulation_id`
- Check that indexes exist on commonly queried columns
- Use `EXPLAIN QUERY PLAN` to debug slow queries

## Example: Complete Experiment Runner

```python
#!/usr/bin/env python3
"""Run a complete experiment with multiple simulations."""

from farm.database.experiment_database import ExperimentDatabase
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from datetime import datetime
import os

def main():
    # Setup
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_path = f"experiments/{experiment_id}.db"
    os.makedirs("experiments", exist_ok=True)
    
    # Load config
    config = SimulationConfig.from_centralized_config()
    
    # Create experiment database
    experiment_db = ExperimentDatabase(
        db_path=db_path,
        experiment_id=experiment_id,
        config=config
    )
    
    # Run multiple simulations
    n_simulations = 10
    for i in range(n_simulations):
        print(f"Running simulation {i+1}/{n_simulations}")
        
        # Create simulation context
        sim_id = f"sim_{i:03d}"
        sim_context = experiment_db.create_simulation_context(
            simulation_id=sim_id,
            parameters={"run_number": i, "seed": i * 100}
        )
        
        # Run simulation (modify this to accept sim_context)
        # You'll need to modify run_simulation to accept a database context
        # instead of creating its own database
        
        # For now, you can manually integrate:
        # - Pass sim_context.logger to your simulation
        # - Use it exactly like db.logger in the simulation
        
        # Flush and update status
        sim_context.flush_all_buffers()
        experiment_db.update_simulation_status(
            simulation_id=sim_id,
            status="completed",
            results_summary={"steps": 1000}
        )
    
    # Mark experiment as complete
    experiment_db.update_experiment_status(
        status="completed",
        results_summary={"total_simulations": n_simulations}
    )
    
    # Cleanup
    experiment_db.close()
    
    print(f"\nExperiment complete! Database: {db_path}")

if __name__ == "__main__":
    main()
```

## See Also

- `farm/database/experiment_database.py` - Implementation
- `farm/database/database.py` - Base database classes
- `farm/database/models.py` - Database schema
- `farm/database/data_logging.py` - Logging functionality
