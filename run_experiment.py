"""
Script to run simulation experiments with different configurations.
"""

#! why is experiments folder still being created for each experiment?

import argparse
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from farm.analysis.comparative_analysis import compare_simulations
from farm.core.config import SimulationConfig
from farm.runners.experiment_runner import ExperimentRunner
from farm.runners.parallel_experiment_runner import ParallelExperimentRunner
from farm.research.research import ResearchProject

logging.basicConfig(level=logging.INFO)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    variations: List[Dict]
    num_iterations: int = 10
    num_steps: int = 1500
    n_jobs: int = -1  # Use all available cores by default
    use_parallel: bool = True  # Whether to use parallel execution


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

            # Determine whether to use parallel or sequential execution
            if experiment_config.use_parallel:
                self.logger.info(f"Using parallel execution with {experiment_config.n_jobs} workers")
                
                # Extract in-memory DB settings from base_config
                use_in_memory_db = getattr(base_config, "use_in_memory_db", True)
                in_memory_db_memory_limit_mb = getattr(base_config, "in_memory_db_memory_limit_mb", None)
                
                experiment = ParallelExperimentRunner(
                    base_config, 
                    experiment_config.name, 
                    n_jobs=experiment_config.n_jobs,
                    db_path=db_path,
                    use_in_memory_db=use_in_memory_db,
                    in_memory_db_memory_limit_mb=in_memory_db_memory_limit_mb
                )
            else:
                self.logger.info("Using sequential execution")
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

        # Create progress bar for experiments
        progress_bar = tqdm(experiments, desc=f"Research: {self.name}", unit="experiment")
        
        for experiment in progress_bar:
            progress_bar.set_description(f"Running experiment: {experiment.name}")
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run simulation experiments")
    parser.add_argument(
        "--name", 
        type=str, 
        default="one_of_a_kind",
        help="Name of the research project"
    )
    parser.add_argument(
        "--description", 
        type=str, 
        default="Experiments with different agent configurations",
        help="Description of the research project"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=250,
        help="Number of iterations per experiment"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=2000,
        help="Number of steps per iteration"
    )
    parser.add_argument(
        "--in-memory", 
        action="store_true",
        default=True,
        help="Use in-memory database for improved performance"
    )
    parser.add_argument(
        "--memory-limit", 
        type=int, 
        default=None,
        help="Memory limit in MB for in-memory database (None = no limit)"
    )
    
    # Add mutually exclusive group for parallel/sequential execution
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument(
        "--parallel", 
        action="store_true",
        dest="parallel",
        default=True,
        help="Use parallel execution for experiments (default)"
    )
    parallel_group.add_argument(
        "--no-parallel", 
        action="store_false",
        dest="parallel",
        help="Disable parallel execution and use sequential processing"
    )
    
    parser.add_argument(
        "--jobs", 
        type=int, 
        default=-1,
        help="Number of parallel jobs to use (-1 = all cores)"
    )
    
    args = parser.parse_args()
    
    # Create research project
    research = Research(
        name=args.name,
        description=args.description,
    )

    # Create experiments for each agent type
    experiments = [
        # ExperimentConfig(
        #     name="single_control_agent",
        #     variations=[
        #         {
        #             "control_agents": 1, 
        #             "system_agents": 0, 
        #             "independent_agents": 0,
        #             "use_in_memory_db": args.in_memory,
        #             "in_memory_db_memory_limit_mb": args.memory_limit,
        #         }
        #     ],
        #     num_iterations=args.iterations,
        #     num_steps=args.steps,
        #     n_jobs=args.jobs,
        #     use_parallel=args.parallel,
        # ),
        # ExperimentConfig(
        #     name="single_system_agent",
        #     variations=[
        #         {
        #             "control_agents": 0, 
        #             "system_agents": 1, 
        #             "independent_agents": 0,
        #             "use_in_memory_db": args.in_memory,
        #             "in_memory_db_memory_limit_mb": args.memory_limit,
        #         }
        #     ],
        #     num_iterations=args.iterations,
        #     num_steps=args.steps,
        #     n_jobs=args.jobs,
        #     use_parallel=args.parallel,
        # ),
        # ExperimentConfig(
        #     name="single_independent_agent",
        #     variations=[
        #         {
        #             "control_agents": 0, 
        #             "system_agents": 0, 
        #             "independent_agents": 1,
        #             "use_in_memory_db": args.in_memory,
        #             "in_memory_db_memory_limit_mb": args.memory_limit,
        #         }
        #     ],
        #     num_iterations=args.iterations,
        #     num_steps=args.steps,
        #     n_jobs=args.jobs,
        #     use_parallel=args.parallel,
        # ),
        ExperimentConfig(
            name="one_of_a_kind",
            variations=[
                {
                    "control_agents": 1, 
                    "system_agents": 1, 
                    "independent_agents": 1,
                    "use_in_memory_db": args.in_memory,
                    "in_memory_db_memory_limit_mb": args.memory_limit,
                }
            ],
            num_iterations=args.iterations,
            num_steps=args.steps,
            n_jobs=args.jobs,
            use_parallel=args.parallel,
        ),
    ]

    research.run_experiments(experiments)
    research.compare_results()


if __name__ == "__main__":
    main()
