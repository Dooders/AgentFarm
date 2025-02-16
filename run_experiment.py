"""
Script to run simulation experiments with different configurations.
"""

#! why is experiments folder still being created for each experiment?

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from farm.analysis.comparative_analysis import compare_simulations
from farm.core.config import SimulationConfig
from farm.core.experiment_runner import ExperimentRunner
from farm.research.research import ResearchProject

logging.basicConfig(level=logging.INFO)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    variations: List[Dict]
    num_iterations: int = 10
    num_steps: int = 1500


class Research:
    """Manager for running and organizing related experiments."""

    def __init__(self, name: str, description: str = ""):
        """Initialize research project.

        Args:
            name: Name of the research project
            description: Optional description of research purpose
        """
        # Create or load existing research project using ResearchProject
        self.research_project = ResearchProject(name, description)
        self.name = name
        self.description = description
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.research_dir = self.research_project.project_path
        self.logger = self.research_project.logger

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for this research project."""
        # Use the logger from ResearchProject
        return self.research_project.logger

    def run_experiment(
        self, experiment_config: ExperimentConfig, folder_name: str
    ) -> None:
        """Run a single experiment with given configuration."""
        self.logger.info(f"Starting experiment: {experiment_config.name}")

        # Create experiment in research project
        base_config = SimulationConfig.from_yaml("config.yaml")

        # Apply variations to base config
        if experiment_config.variations:
            for variation in experiment_config.variations:
                for key, value in variation.items():
                    setattr(base_config, key, value)

        # Create experiment in research project and get experiment path
        exp_path = self.research_project.create_experiment(
            experiment_config.name,
            f"Experiment with variations: {experiment_config.variations}",
            base_config,
            folder_name,
        )

        experiment = None
        try:
            # Create experiment runner with the experiment path
            exp_path = Path(exp_path)
            db_path = exp_path / "simulation.db"

            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)

            experiment = ExperimentRunner(
                base_config, experiment_config.name, db_path=db_path
            )

            # Run experiment iterations
            experiment.run_iterations(
                experiment_config.num_iterations,
                None,  # Don't pass variations here since we already applied them
                experiment_config.num_steps,
                exp_path,  # Pass the Path object
            )

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise
        finally:
            if experiment:
                try:
                    # Ensure proper database cleanup
                    if hasattr(experiment, "db"):
                        experiment.db.close()
                        delattr(experiment, "db")

                    # Add small delay to ensure file handles are released
                    time.sleep(0.1)
                except Exception as cleanup_error:
                    self.logger.error(f"Error during database cleanup: {cleanup_error}")

        self.logger.info(f"Experiment {experiment_config.name} completed")

    def run_experiments(self, experiments: List[ExperimentConfig]) -> None:
        """Run multiple experiments in sequence."""
        self.logger.info(f"Starting research batch with {len(experiments)} experiments")

        for experiment in experiments:
            folder_name = experiment.name + "_" + self.timestamp
            self.run_experiment(experiment, folder_name)

        self.logger.info("All experiments completed")

    def compare_results(self) -> None:
        """Compare results across all experiments in this research."""
        # Create analysis directory if it doesn't exist
        analysis_dir = os.path.join(str(self.research_dir), "experiments", "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Pass both the experiment directory (for finding DBs) and analysis directory
        compare_simulations(
            search_path=str(self.research_dir), analysis_path=analysis_dir
        )


def main():
    """Run research project testing individual agent types."""
    research = Research(
        name="one_of_a_kind",
        description="3 experiments with 100 iterations and 1000 steps, each with a different agent type (1 control, 1 system, 1 independent)",
    )

    # Create experiments for each agent type
    experiments = [
        ExperimentConfig(
            name="single_control_agent",
            variations=[
                {"control_agents": 1, "system_agents": 0, "independent_agents": 0}
            ],
            num_iterations=100,
            num_steps=1000,
        ),
        ExperimentConfig(
            name="single_system_agent",
            variations=[
                {"control_agents": 0, "system_agents": 1, "independent_agents": 0}
            ],
            num_iterations=100,
            num_steps=1000,
        ),
        ExperimentConfig(
            name="single_independent_agent",
            variations=[
                {"control_agents": 0, "system_agents": 0, "independent_agents": 1}
            ],
            num_iterations=100,
            num_steps=1000,
        ),
    ]

    research.run_experiments(experiments)
    research.compare_results()


if __name__ == "__main__":
    main()
