# Centralized Storage - Quick Reference Card

## TL;DR

Store many simulations in one database instead of creating separate files for each run.

## Quick Start (3 Options)

### Option 1: Environment Variables (Zero Code Changes)
```bash
export USE_EXPERIMENT_DB=1
export EXPERIMENT_ID="my_experiment"
python run_simulation.py  # Uses centralized storage automatically!
```

### Option 2: DatabaseFactory (Minimal Changes)
```python
from farm.database.experiment_integration import DatabaseFactory

# Old: db = SimulationDatabase("sim.db")
# New:
db = DatabaseFactory.create(
    config=config,
    simulation_id="sim_001",
    use_experiment_db=True,
    experiment_id="my_exp"
)

# Rest of code stays the same!
db.logger.log_step(...)
db.close()
```

### Option 3: ExperimentManager (Batch Operations)
```python
from farm.database.experiment_integration import ExperimentManager

with ExperimentManager(experiment_id="exp_001") as mgr:
    for i in range(10):
        ctx = mgr.create_simulation(f"sim_{i}")
        # Run simulation with ctx.logger
        mgr.complete_simulation(f"sim_{i}")
```

## Command Line Tools

### Query Experiments
```bash
# Show experiment info
python scripts/query_experiment.py experiments/exp.db --command info

# List simulations
python scripts/query_experiment.py experiments/exp.db --command list

# Get summary
python scripts/query_experiment.py experiments/exp.db --command summary

# Compare metric
python scripts/query_experiment.py experiments/exp.db --command compare --metric total_agents

# Export data
python scripts/query_experiment.py experiments/exp.db --command export --simulation-id sim_001 --output ./data
```

### Run Experiments
```bash
python scripts/run_experiment.py \
    --experiment-id my_exp \
    --num-simulations 10 \
    --steps 1000 \
    --name "My Experiment"
```

## Python API

### Create Experiment Database
```python
from farm.database.experiment_database import ExperimentDatabase

db = ExperimentDatabase(
    db_path="experiments/exp_001.db",
    experiment_id="exp_001",
    config=config
)
```

### Create Simulation Context
```python
sim_ctx = db.create_simulation_context(
    simulation_id="sim_001",
    parameters={"seed": 42, "learning_rate": 0.1}
)

# Use exactly like SimulationDatabase
sim_ctx.logger.log_step(...)
sim_ctx.logger.log_agent(...)
sim_ctx.flush_all_buffers()
```

### Query Data
```python
from scripts.query_experiment import ExperimentQueryTool

tool = ExperimentQueryTool("experiments/exp_001.db")

# Get simulations
sims = tool.list_simulations()

# Get summary
summary = tool.get_simulation_summary()

# Compare
comparison = tool.compare_simulations(metric="total_agents")

tool.close()
```

### Using ExperimentManager
```python
from farm.database.experiment_integration import ExperimentManager

with ExperimentManager(
    experiment_id="exp_001",
    name="Parameter Sweep",
    description="Testing different parameters"
) as manager:
    
    # Create simulation
    ctx = manager.create_simulation(
        simulation_id="sim_001",
        parameters={"param1": 10}
    )
    
    # Run simulation
    # ... use ctx.logger ...
    
    # Complete
    manager.complete_simulation(
        simulation_id="sim_001",
        results_summary={"score": 95.3}
    )
```

## Common Patterns

### Batch Run Multiple Simulations
```python
with ExperimentManager(experiment_id="batch_001") as mgr:
    for seed in range(10):
        ctx = mgr.create_simulation(
            f"sim_{seed}",
            parameters={"seed": seed}
        )
        config.seed = seed
        # run_simulation(config, ctx)
        mgr.complete_simulation(f"sim_{seed}")
```

### Parameter Sweep
```python
with ExperimentManager(experiment_id="sweep_001") as mgr:
    for lr in [0.01, 0.1, 0.5, 1.0]:
        for pop in [10, 50, 100]:
            sim_id = f"sim_lr{lr}_pop{pop}"
            ctx = mgr.create_simulation(sim_id, {"lr": lr, "pop": pop})
            # run with these parameters
            mgr.complete_simulation(sim_id)
```

### Query and Compare
```python
tool = ExperimentQueryTool("experiments/exp_001.db")

# Get all simulation IDs
sims = tool.list_simulations()

# Get final states
for sim_id in sims['simulation_id']:
    final = tool.get_final_state(sim_id)
    print(f"{sim_id}: {final['total_agents']} agents")

# Compare over time
agents_df = tool.get_agent_counts_over_time()
# Plot or analyze
```

## File Locations

```
Key Files:
├── farm/database/
│   ├── experiment_database.py          # Core implementation
│   └── experiment_integration.py       # Helper utilities
├── scripts/
│   ├── query_experiment.py             # Query tool
│   └── run_experiment.py               # Runner script
├── examples/
│   └── centralized_storage_example.py  # Working examples
└── docs/
    ├── CENTRALIZED_STORAGE_GUIDE.md    # Detailed docs
    └── ARCHITECTURE_DIAGRAM.md         # Visual diagrams
```

## Key Concepts

**Experiment**: A collection of related simulations
**Simulation**: A single simulation run with specific parameters
**simulation_id**: Unique identifier that tags all data from one simulation
**ExperimentDatabase**: Database that stores multiple simulations
**SimulationContext**: Per-simulation view with same interface as SimulationDatabase

## Benefits

✅ One database file per experiment  
✅ Easy cross-simulation queries  
✅ 10% storage savings  
✅ Better organization  
✅ Temporal tracking  
✅ Same logging interface  

## Important Notes

- Interface is identical to `SimulationDatabase`
- All logging methods work the same
- `simulation_id` automatically added to all data
- Environment variables work with existing code
- SQLite limitations on concurrent writes still apply

## Getting Help

- Quick start: `CENTRALIZED_STORAGE_README.md`
- Detailed guide: `docs/CENTRALIZED_STORAGE_GUIDE.md`
- Architecture: `docs/ARCHITECTURE_DIAGRAM.md`
- Examples: `examples/centralized_storage_example.py`

## Cheat Sheet

```python
# Import helpers
from farm.database.experiment_integration import DatabaseFactory, ExperimentManager

# Create database (auto-detects env vars)
db = DatabaseFactory.create(config=config, simulation_id="sim_001", use_experiment_db=True)

# Use database (same as before)
db.logger.log_step(step, agents, resources, metrics)
db.close()

# Run batch experiment
with ExperimentManager(experiment_id="exp") as mgr:
    ctx = mgr.create_simulation("sim_001")
    # ... run simulation ...
    mgr.complete_simulation("sim_001")

# Query results
from scripts.query_experiment import ExperimentQueryTool
tool = ExperimentQueryTool("experiments/exp.db")
summary = tool.get_simulation_summary()
tool.close()
```

## Environment Variables

```bash
USE_EXPERIMENT_DB=1           # Enable centralized storage
EXPERIMENT_ID="my_exp"        # Experiment identifier
EXPERIMENT_DB_PATH="path.db"  # Database path (optional)
```

## Common Commands

```bash
# Run experiment
python scripts/run_experiment.py --experiment-id exp_001 --num-simulations 10

# Query summary
python scripts/query_experiment.py experiments/exp_001.db --command summary

# List simulations
python scripts/query_experiment.py experiments/exp_001.db --command list

# Compare simulations
python scripts/query_experiment.py experiments/exp_001.db --command compare

# Export simulation
python scripts/query_experiment.py experiments/exp_001.db --command export --simulation-id sim_001 --output ./data
```
