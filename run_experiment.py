"""
Script to run simulation experiments with different configurations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from farm.analysis.comparative_analysis import main as compare_simulations
from farm.core.config import SimulationConfig
from farm.core.experiment_runner import ExperimentRunner

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
        self.name = name
        self.description = description
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.research_dir = Path("research") / f"{self.name}_{self.timestamp}"
        self.research_dir.mkdir(parents=True, exist_ok=True)

        # Save research metadata
        with open(self.research_dir / "metadata.txt", "w") as f:
            f.write(f"Research: {name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Description: {description}\n")

        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for this research project."""
        logger = logging.getLogger(self.name)
        fh = logging.FileHandler(self.research_dir / "research.log")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def run_experiment(self, experiment_config: ExperimentConfig) -> None:
        """Run a single experiment with given configuration."""
        self.logger.info(f"Starting experiment: {experiment_config.name}")

        experiment = None
        try:
            base_config = SimulationConfig.from_yaml("config.yaml")

            # Apply variations to base config
            if experiment_config.variations:
                for variation in experiment_config.variations:
                    for key, value in variation.items():
                        setattr(base_config, key, value)

            # Create experiment runner without output_dir parameter
            experiment = ExperimentRunner(base_config, experiment_config.name)

            experiment.run_iterations(
                experiment_config.num_iterations,
                None,  # Don't pass variations here since we already applied them
                experiment_config.num_steps,
            )

        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise  # Re-raise the exception for debugging
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
        compare_simulations(str(self.research_dir))


def main():
    """Run research project testing individual agent types."""
    research = Research(
        name="single_agent_study",
        description="Investigating behavior of individual agent types in isolation",
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
