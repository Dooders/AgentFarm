"""
Script to run simulation experiments with different configurations.
"""

import logging
import os
from typing import Dict, List

from farm.analysis.comparative_analysis import find_simulation_databases
from farm.analysis.comparative_analysis import main as compare_simulations
from farm.core.config import SimulationConfig
from farm.core.experiment_runner import ExperimentRunner

logging.basicConfig(level=logging.INFO)


def run_experiment(
    name: str,
    variations: List[Dict] = None,
    num_iterations: int = 10,
    num_steps: int = 1500,
) -> None:
    """Run experiment with given variations."""
    logging.info(f"Starting {name} experiment...")

    try:
        base_config = SimulationConfig.from_yaml("config.yaml")
        #! make it easier to change config when running experiments
        base_config.independent_agents = 0
        base_config.system_agents = 0
        base_config.control_agents = 30
        experiment = ExperimentRunner(base_config, name)


        experiment.run_iterations(num_iterations, variations, num_steps)
        # experiment.generate_report()

    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
    finally:
        if hasattr(experiment, "db"):
            experiment.db.close()

    logging.info(f"{name} experiment completed")


def main():
    """Run simulation experiments."""
    if not os.path.exists("experiments"):
        os.makedirs("experiments")

    # Define experiment configurations
    experiments = {
        "resource_distribution_test": [
            {"initial_resources": 10, "num_steps": 1000},
            {"initial_resources": 20, "num_steps": 1000},
            {"initial_resources": 30, "num_steps": 1000},
        ],
        "population_ratio_test": [
            {
                "initial_system_agents": 10,
                "initial_independent_agents": 40,
                "num_steps": 1000,
            },
            {
                "initial_system_agents": 25,
                "initial_independent_agents": 25,
                "num_steps": 1000,
            },
            {
                "initial_system_agents": 40,
                "initial_independent_agents": 10,
                "num_steps": 1000,
            },
        ],
    }

    print("Starting experiments...")

    # Run selected experiments
    # run_experiment("only_control_agents", num_iterations=100, num_steps=1500)
    # run_experiment("population_ratio_test", experiments["population_ratio_test"])


    print("Experiments completed! Check the experiments directory for results.")

    # Compare simulations
    compare_simulations("experiments/only_control_agents/databases")


if __name__ == "__main__":
    main()
