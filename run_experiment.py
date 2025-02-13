"""
Script to run simulation experiments with different configurations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import os

from farm.analysis.comparative_analysis import compare_simulations
from farm.core.config import SimulationConfig
from farm.core.experiment_runner import ExperimentRunner
from research.research import ResearchProject

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

    def run_experiment(self, experiment_config: ExperimentConfig) -> None:
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
            base_config
        )

        experiment = None
        try:
            # Create experiment runner with the experiment path
            # Convert exp_path to Path object and join with simulation.db
            db_path = Path(exp_path) / "simulation.db"
            experiment = ExperimentRunner(
                base_config, 
                experiment_config.name,
                db_path=db_path  # Pass the properly constructed Path object
            )

            experiment.run_iterations(
                experiment_config.num_iterations,
                None,  # Don't pass variations here since we already applied them
                experiment_config.num_steps,
            )

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise
        finally:
            if experiment and hasattr(experiment, "db"):
                experiment.db.close()

        self.logger.info(f"Experiment {experiment_config.name} completed")

    def run_experiments(self, experiments: List[ExperimentConfig]) -> None:
        """Run multiple experiments in sequence."""
        self.logger.info(f"Starting research batch with {len(experiments)} experiments")

        for experiment in experiments:
            self.run_experiment(experiment)

        self.logger.info("All experiments completed")

    def compare_results(self) -> None:
        """Compare results across all experiments in this research."""
        # Create analysis directory if it doesn't exist
        analysis_dir = os.path.join(str(self.research_dir), "experiments", "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Pass both the experiment directory (for finding DBs) and analysis directory
        compare_simulations(
            search_path=str(self.research_dir), 
            analysis_path=analysis_dir
        )


def main():
    """Run research project testing individual agent types."""
    research = Research(
        name="testing_research_experiment",
        description="Investigating behavior of individual agent types in isolation",
    )

    # Create experiments for each agent type
    experiments = [
        ExperimentConfig(
            name="single_control_agent",
            variations=[
                {"control_agents": 1, "system_agents": 0, "independent_agents": 0}
            ],
            num_iterations=1,
            num_steps=1000,
        ),
        ExperimentConfig(
            name="single_system_agent",
            variations=[
                {"control_agents": 0, "system_agents": 1, "independent_agents": 0}
            ],
            num_iterations=1,
            num_steps=1000,
        ),
        ExperimentConfig(
            name="single_independent_agent",
            variations=[
                {"control_agents": 0, "system_agents": 0, "independent_agents": 1}
            ],
            num_iterations=1,
            num_steps=1000,
        ),
    ]

    research.run_experiments(experiments)
    research.compare_results()


if __name__ == "__main__":
    main()
