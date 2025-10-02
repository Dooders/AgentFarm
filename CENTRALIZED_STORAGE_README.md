# Centralized Storage Solution

## Overview

This solution allows you to store many simulations in a **single SQLite database** instead of creating separate database files for each run. This makes it much easier to:

- **Compare experiments**: All data in one place
- **Track temporal trends**: See how results evolve over time
- **Query efficiently**: No need to open multiple database files
- **Save disk space**: Reduce database overhead

## What's New

### 1. ExperimentDatabase Class
Located in `farm/database/experiment_database.py`, this class extends `SimulationDatabase` to support multiple simulations in one database file. Each simulation is tagged with a unique `simulation_id`.

### 2. DatabaseFactory Helper
Located in `farm/database/experiment_integration.py`, this helper makes it easy to switch between:
- **Traditional mode**: One database per simulation
- **Centralized mode**: Many simulations per database

### 3. Query Tools
`scripts/query_experiment.py` - Command-line tool for exploring experiment databases:
```bash
# Show experiment info
python scripts/query_experiment.py experiments/my_exp.db --command info

# List all simulations
python scripts/query_experiment.py experiments/my_exp.db --command list

# Compare simulations
python scripts/query_experiment.py experiments/my_exp.db --command compare

# Export simulation data
python scripts/query_experiment.py experiments/my_exp.db --command export \
    --simulation-id sim_001 --output ./output
```

### 4. Complete Examples
See `examples/centralized_storage_example.py` for working examples.

## Quick Start

### Option 1: Environment Variables (Easiest)

Set environment variables before running simulations:

```bash
export USE_EXPERIMENT_DB=1
export EXPERIMENT_ID="my_experiment_001"
export EXPERIMENT_DB_PATH="experiments/my_experiment_001.db"

# Now your existing code will use centralized storage
python run_simulation.py
```

### Option 2: DatabaseFactory (Programmatic)

Update your simulation code to use the DatabaseFactory:

```python
from farm.database.experiment_integration import DatabaseFactory

# Old way (separate database files)
# db = SimulationDatabase("simulation.db")

# New way (centralized storage)
db = DatabaseFactory.create(
    config=config,
    simulation_id="sim_001",
    use_experiment_db=True,
    experiment_id="exp_001"
)

# Use db exactly the same way as before!
db.logger.log_step(...)
db.close()
```

### Option 3: ExperimentManager (High-Level)

For running complete experiments:

```python
from farm.database.experiment_integration import ExperimentManager

with ExperimentManager(
    experiment_id="learning_rate_sweep",
    name="Learning Rate Comparison",
    description="Testing different learning rates"
) as manager:
    
    for lr in [0.01, 0.1, 0.5, 1.0]:
        sim_id = f"sim_lr_{lr}"
        
        # Create simulation context
        sim_context = manager.create_simulation(
            simulation_id=sim_id,
            parameters={"learning_rate": lr}
        )
        
        # Run your simulation with sim_context
        # (use sim_context.logger.log_step() etc.)
        
        # Mark as complete
        manager.complete_simulation(
            simulation_id=sim_id,
            results_summary={"final_agents": 100}
        )
```

## Database Schema

The centralized database includes:

### Experiments Table
Stores metadata about each experiment (collection of related simulations).

### Simulations Table
Stores metadata about each simulation run within an experiment.

### Data Tables
All existing tables (agents, agent_states, actions, resources, etc.) now include a `simulation_id` column to separate data from different simulations.

## Migration Guide

### Updating Existing Code

**Before:**
```python
from farm.database.database import SimulationDatabase

db = SimulationDatabase(f"simulations/sim_{timestamp}.db")
db.logger.log_step(...)
db.close()
```

**After:**
```python
from farm.database.experiment_integration import DatabaseFactory

db = DatabaseFactory.create(
    config=config,
    simulation_id=f"sim_{timestamp}",
    use_experiment_db=True,
    experiment_id="my_experiment"
)
db.logger.log_step(...)  # Same interface!
db.close()
```

The interface is **identical**, so your existing logging code doesn't need to change!

## File Structure

```
workspace/
├── farm/database/
│   ├── experiment_database.py        # ExperimentDatabase class
│   └── experiment_integration.py     # Helper utilities
├── scripts/
│   ├── run_experiment.py             # Run multiple simulations
│   └── query_experiment.py           # Query experiment databases
├── examples/
│   └── centralized_storage_example.py # Working examples
├── docs/
│   └── CENTRALIZED_STORAGE_GUIDE.md  # Detailed documentation
└── CENTRALIZED_STORAGE_README.md     # This file
```

## Documentation

- **Quick Reference**: This file (CENTRALIZED_STORAGE_README.md)
- **Detailed Guide**: `docs/CENTRALIZED_STORAGE_GUIDE.md`
- **Code Examples**: `examples/centralized_storage_example.py`
- **Query Tool**: `scripts/query_experiment.py --help`

## Usage Examples

### Run 10 Simulations in One Database

```python
from farm.database.experiment_integration import ExperimentManager

with ExperimentManager(experiment_id="exp_001") as manager:
    for i in range(10):
        sim_context = manager.create_simulation(f"sim_{i:03d}")
        
        # Your simulation code here
        # Use sim_context.logger.log_step() etc.
        
        manager.complete_simulation(f"sim_{i:03d}")
```

### Query Results

```python
from scripts.query_experiment import ExperimentQueryTool

tool = ExperimentQueryTool("experiments/exp_001.db")

# List simulations
sims = tool.list_simulations()
print(sims)

# Get summary
summary = tool.get_simulation_summary()
print(summary)

# Compare metric across simulations
comparison = tool.compare_simulations(metric="total_agents")
print(comparison.head())

tool.close()
```

### Using Environment Variables

```bash
# Set up centralized storage
export USE_EXPERIMENT_DB=1
export EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"

# Run your simulations (they'll all go to the same database)
for i in {1..10}; do
    python run_simulation.py --seed $i
done

# Query results
python scripts/query_experiment.py \
    experiments/${EXPERIMENT_ID}.db \
    --command summary
```

## Benefits

### Before (Separate Databases)
```
simulations/
├── sim_001.db (5 MB)
├── sim_002.db (5 MB)
├── sim_003.db (5 MB)
├── ...
└── sim_100.db (5 MB)
Total: ~500 MB, 100 files
```

### After (Centralized Storage)
```
experiments/
└── exp_001.db (450 MB)
Total: 450 MB, 1 file
```

**Advantages:**
- ✅ 10% smaller (reduced schema overhead)
- ✅ 1 file instead of 100
- ✅ Easy cross-simulation queries
- ✅ Better organization
- ✅ Temporal tracking

## Performance Considerations

### Write Performance
- ✅ Similar to separate databases for sequential writes
- ⚠️ Be careful with parallel writes (SQLite limitation)
- ✅ Use WAL mode (enabled by default) for better concurrency

### Query Performance
- ✅ **Faster** for cross-simulation queries
- ✅ Single database = single connection
- ✅ Indexes on `simulation_id` for efficient filtering

### Storage
- ✅ More efficient (less schema duplication)
- ✅ Single file easier to manage
- ⚠️ Monitor size for very large experiments (>10GB)

## Best Practices

1. **Use descriptive experiment IDs**: Include date and purpose
   ```python
   experiment_id = "exp_20250102_learning_rate_sweep"
   ```

2. **One experiment per database**: Don't mix unrelated experiments
   ```python
   # Good: One focused experiment
   exp_learning = ExperimentDatabase(
       db_path="experiments/learning_rates.db",
       experiment_id="learning_exp"
   )
   
   # Good: Separate database for different experiment
   exp_population = ExperimentDatabase(
       db_path="experiments/population.db",
       experiment_id="population_exp"
   )
   ```

3. **Store parameters**: Always save configuration
   ```python
   sim_context = manager.create_simulation(
       simulation_id="sim_001",
       parameters={
           "learning_rate": 0.1,
           "population": 100,
           "seed": 42
       }
   )
   ```

4. **Update status**: Track progress
   ```python
   manager.complete_simulation(
       simulation_id="sim_001",
       results_summary={
           "final_agents": 150,
           "total_steps": 1000,
           "outcome": "success"
       }
   )
   ```

5. **Regular backups**: Export important data
   ```bash
   python scripts/query_experiment.py experiments/exp_001.db \
       --command export \
       --simulation-id sim_001 \
       --output ./backups/sim_001/
   ```

## Troubleshooting

### "Database is locked" errors
- Ensure only one process writes at a time
- Call `flush_all_buffers()` before closing
- Check that simulations aren't running in parallel

### Large database files
- Split into multiple experiment databases
- Export and archive old simulations
- Use `VACUUM` to reclaim space:
  ```bash
  sqlite3 experiments/exp_001.db "VACUUM;"
  ```

### Slow queries
- Always filter by `simulation_id`
- Check that indexes exist
- Use `EXPLAIN QUERY PLAN` for debugging

## Testing

Run the example to verify everything works:

```bash
python examples/centralized_storage_example.py
```

This will create example databases you can explore:
- `experiments/example_experiment.db`
- `experiments/example_with_config.db`
- `experiments/comparison_experiment.db`

## Next Steps

1. **Read the detailed guide**: `docs/CENTRALIZED_STORAGE_GUIDE.md`
2. **Run the examples**: `python examples/centralized_storage_example.py`
3. **Try the query tool**: `python scripts/query_experiment.py`
4. **Update your code**: Use `DatabaseFactory` or environment variables
5. **Run experiments**: Use `ExperimentManager` for batch runs

## Support

For more information:
- See `docs/CENTRALIZED_STORAGE_GUIDE.md` for detailed documentation
- Check `examples/centralized_storage_example.py` for code examples
- Review `farm/database/experiment_database.py` for implementation details

## Summary

You now have a complete solution for storing multiple simulations in a single database:

✅ **ExperimentDatabase**: Core functionality for centralized storage  
✅ **DatabaseFactory**: Easy integration with existing code  
✅ **ExperimentManager**: High-level API for running experiments  
✅ **Query Tools**: Command-line and programmatic data access  
✅ **Examples**: Working code you can run immediately  
✅ **Documentation**: Comprehensive guides and references  

**The best part?** Your existing simulation code needs minimal changes - just update how you create the database, and everything else stays the same!
