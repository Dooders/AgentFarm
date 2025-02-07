# Running Simulation Experiments

The experiment module provides functionality to run multiple simulation iterations with different parameters and analyze the results. This document explains how to use the experiment features.

## Overview

The experiment system allows you to:
- Run multiple iterations of simulations
- Test different parameter configurations
- Collect and analyze results
- Generate summary reports
- Compare outcomes across iterations

## Usage

### Basic Example

```python
from experiment import ExperimentRunner
from config import SimulationConfig

# Create base configuration
base_config = SimulationConfig(
    num_steps=1000,
    num_agents=100,
    # ... other configuration parameters
)

# Initialize experiment runner
runner = ExperimentRunner(
    base_config=base_config,
    experiment_name="my_experiment"
)

# Run multiple iterations
runner.run_iterations(num_iterations=5)

# Generate summary report
runner.generate_report()
```

### Parameter Variations

You can test different parameter configurations across iterations:

```python
# Define parameter variations
variations = [
    {"num_agents": 50},
    {"num_agents": 150},
    {"resource_rate": 2.0},
    {"system_threshold": 0.8},
]

# Run iterations with variations
runner.run_iterations(
    num_iterations=len(variations),
    config_variations=variations
)
```

## Results Analysis

The experiment runner generates two main result files:

1. `{experiment_name}_results.csv`: Detailed results for each iteration, including:
   - Final number of system and independent agents
   - Average resources per agent type
   - Timestamp
   - Configuration variation used
   - Iteration number

2. `{experiment_name}_summary.csv`: Statistical summary of results across all iterations, including:
   - Mean, standard deviation, min, max values
   - Quartile distributions
   - Count of successful iterations

## Logging

Each experiment maintains its own log file (`{experiment_name}.log`) that captures:
- Experiment progress
- Iteration status
- Error messages
- Configuration details

The log file can be found in the experiment's root directory.

## Best Practices

1. Use meaningful experiment names that reflect the purpose of the test
2. Start with a small number of iterations to validate configuration
3. Save base configurations for reproducibility
4. Document parameter variations used in experiments
5. Review logs for any warnings or errors before analyzing results

## Error Handling

The experiment runner includes built-in error handling to:
- Continue running if individual iterations fail
- Log error details for debugging
- Mark failed iterations in results

Failed iterations will be logged but won't prevent the completion of the overall experiment.

## ExperimentDatabase Class

The `ExperimentDatabase` class is designed to manage multiple simulation runs and aggregate results across simulations. It provides methods for adding, updating, retrieving, listing, and deleting simulation records, ensuring thread-safe access to the database, exporting data to CSV, and aggregating results.

### SQLite Schema

The `ExperimentDatabase` class uses a SQLite schema to store experiment metadata, simulation parameters, and results summaries. The schema includes a `Simulations` table with the following columns:

- `simulation_id`: Primary key.
- `start_time`: Unix timestamp for the start of the simulation.
- `end_time`: Unix timestamp for the end of the simulation.
- `status`: Text field for simulation status (`pending`, `running`, `completed`, `failed`).
- `parameters`: JSON-encoded string of simulation parameters.
- `results_summary`: JSON-encoded string summarizing simulation results.
- `simulation_db_path`: File path to the corresponding `SimulationDatabase`.

### Methods

The `ExperimentDatabase` class provides the following methods:

- `add_simulation(parameters: dict, simulation_db_path: str) -> int`: Adds a new simulation record to the database.
- `update_simulation_status(simulation_id: int, status: str, results_summary: dict = None)`: Updates the status and results of a simulation.
- `get_simulation(simulation_id: int) -> dict`: Retrieves details of a specific simulation.
- `list_simulations(status: str = None) -> list`: Lists all simulations, optionally filtered by status.
- `delete_simulation(simulation_id: int)`: Deletes a simulation record.
- `export_experiment_data(filepath: str)`: Exports all experiment data to a CSV file.
- `get_aggregate_results() -> dict`: Aggregates results across completed simulations.

### Example Usage

```python
from core.experiment_database import ExperimentDatabase

# Initialize ExperimentDatabase
experiment_db = ExperimentDatabase("sqlite:///experiment.db")

# Add a new simulation
simulation_id = experiment_db.add_simulation(
    parameters={"num_agents": 100, "steps": 500},
    simulation_db_path="simulations/simulation_1.db",
)
print(f"New simulation added with ID: {simulation_id}")

# Update simulation status
experiment_db.update_simulation_status(
    simulation_id=simulation_id,
    status="completed",
    results_summary={"total_agents": 100, "average_lifespan": 50.5},
)

# Retrieve a simulation
simulation = experiment_db.get_simulation(simulation_id)
print(f"Simulation Details: {simulation}")

# List all completed simulations
completed_sims = experiment_db.list_simulations(status="completed")
print(f"Completed Simulations: {completed_sims}")

# Export data to CSV
experiment_db.export_experiment_data("experiment_results.csv")

# Aggregate results
aggregate_results = experiment_db.get_aggregate_results()
print(f"Aggregate Results: {aggregate_results}")

# Delete a simulation
experiment_db.delete_simulation(simulation_id)
```
