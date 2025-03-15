## Step 1: Prepare Your Environment

Ensure you have the necessary prerequisites installed (Python, pip, Git) and set up a virtual environment as outlined in the [Quickstart Guide](https://github.com/Dooders/AgentFarm/blob/main/docs/SimulationQuickStart.md).

## Step 2: Clone the Repository

If you haven't already, clone the AgentFarm repository to your local machine:

```bash
git clone https://github.com/Dooders/AgentFarm.git
cd AgentFarm
```

## Step 3: Create a New Script

Create a new script, say `run_multiple_experiments.py`, in the root directory of the repository. This script will utilize the `ExperimentConfig` and `Research` classes to run multiple simulations.

## Step 4: Implement the Script

Here’s an example of how to implement the script:

```python
from run_experiment import ExperimentConfig, Research

def run_experiments():
    """Run a series of simulation experiments with different configurations."""
    research_project_name = "my_experiment_series"
    research_project_description = "Testing various agent configurations"
    num_iterations = 250
    num_steps = 2000
    use_in_memory_db = True
    in_memory_db_memory_limit_mb = None
    use_parallel = True
    num_jobs = -1

    # Create research project
    research = Research(
        name=research_project_name,
        description=research_project_description,
    )

    # Define multiple experiments with different configurations
    experiments = [
        ExperimentConfig(
            name="experiment_with_one_agent",
            variations=[
                {
                    "control_agents": 1, 
                    "system_agents": 0, 
                    "independent_agents": 0,
                    "use_in_memory_db": use_in_memory_db,
                    "in_memory_db_memory_limit_mb": in_memory_db_memory_limit_mb,
                }
            ],
            num_iterations=num_iterations,
            num_steps=num_steps,
            n_jobs=num_jobs,
            use_parallel=use_parallel,
        ),
        ExperimentConfig(
            name="experiment_with_two_agents",
            variations=[
                {
                    "control_agents": 1, 
                    "system_agents": 1, 
                    "independent_agents": 0,
                    "use_in_memory_db": use_in_memory_db,
                    "in_memory_db_memory_limit_mb": in_memory_db_memory_limit_mb,
                }
            ],
            num_iterations=num_iterations,
            num_steps=num_steps,
            n_jobs=num_jobs,
            use_parallel=use_parallel,
        ),
        ExperimentConfig(
            name="experiment_with_three_agents",
            variations=[
                {
                    "control_agents": 1, 
                    "system_agents": 1, 
                    "independent_agents": 1,
                    "use_in_memory_db": use_in_memory_db,
                    "in_memory_db_memory_limit_mb": in_memory_db_memory_limit_mb,
                }
            ],
            num_iterations=num_iterations,
            num_steps=num_steps,
            n_jobs=num_jobs,
            use_parallel=use_parallel,
        ),
    ]

    # Run the experiments
    research.run_experiments(experiments)
    research.compare_results()

if __name__ == "__main__":
    run_experiments()
```

## Step 5: Run the Script

Execute the `run_multiple_experiments.py` script to run the series of simulations with the specified configurations:

```bash
python run_multiple_experiments.py
```

By following these steps, you can run a series of simulations with different configurations using the `ExperimentConfig` and `Research` classes from `run_experiment.py` in a separate script without using command-line argument parsing.
