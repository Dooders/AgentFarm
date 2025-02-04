"""
ExperimentRunner: Module for running multiple simulation experiments and analyzing results.

This module provides functionality to:
- Run multiple simulation iterations with different parameters
- Extract and store statistics from each run
- Compare results across iterations
- Generate summary reports
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from farm.core.config import SimulationConfig
from farm.core.simulation import run_simulation

DEFAULT_NUM_STEPS = 1000


class ExperimentRunner:
    """Manages multiple simulation runs and result analysis."""

    def __init__(self, base_config: SimulationConfig, experiment_name: str):
        """
        Initialize experiment runner.

        Parameters
        ----------
        base_config : SimulationConfig
            Base configuration for simulations
        experiment_name : str
            Name of the experiment for organizing results
        """
        self.base_config = base_config
        self.experiment_name = experiment_name
        self.results: List[Dict] = []

        # Setup experiment directories
        self.experiment_dir = os.path.join("experiments", experiment_name)
        self.db_dir = os.path.join(self.experiment_dir, "databases")
        self.results_dir = os.path.join(self.experiment_dir, "results")

        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure experiment-specific logging."""
        log_file = os.path.join(self.experiment_dir, f"{self.experiment_name}.log")

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger(f"experiment.{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

    def run_iterations(
        self,
        num_iterations: int,
        config_variations: Optional[List[Dict]] = None,
        num_steps: int = DEFAULT_NUM_STEPS,
    ) -> None:
        """Run multiple iterations of the simulation."""
        self.logger.info(f"Starting experiment with {num_iterations} iterations")

        for i in range(num_iterations):
            self.logger.info(f"Starting iteration {i+1}/{num_iterations}")

            # Create iteration-specific config
            iteration_config = self._create_iteration_config(i, config_variations)

            # Setup database path for this iteration
            db_path = os.path.join(self.db_dir, f"iteration_{i+1}.db")

            try:
                # Run simulation
                env = run_simulation(
                    num_steps=num_steps,
                    config=iteration_config,
                    db_path=db_path,
                )

                # Ensure all data is flushed
                if env.db:
                    env.logger.flush_all_buffers()

                # Analyze results
                results = self._analyze_iteration(db_path)
                results["iteration"] = i + 1
                results["config_variation"] = str(
                    config_variations[i] if config_variations else "base"
                )

                self.results.append(results)
                self.logger.info(f"Completed iteration {i+1}")

            except Exception as e:
                self.logger.error(f"Error in iteration {i+1}: {str(e)}")
                continue

    def _create_iteration_config(
        self, iteration: int, variations: Optional[List[Dict]] = None
    ) -> SimulationConfig:
        """Create configuration for specific iteration."""
        config = self.base_config.copy()

        if variations and iteration < len(variations):
            # Apply variation to config
            for key, value in variations[iteration].items():
                setattr(config, key, value)

        return config

    # def _analyze_iteration(self, db_path: str) -> Dict:
    #     """
    #     Extract relevant statistics from simulation database.

    #     Parameters
    #     ----------
    #     db_path : str
    #         Path to simulation database

    #     Returns
    #     -------
    #     Dict
    #         Dictionary containing extracted metrics
    #     """
    #     analyzer = SimulationAnalyzer(db_path)

    #     # Get survival rates
    #     survival_data = analyzer.calculate_survival_rates()
    #     final_survival = survival_data.iloc[-1]

    #     # Get resource distribution
    #     resource_data = analyzer.analyze_resource_distribution()
    #     final_resources = resource_data.groupby("agent_type").last()

    #     # Compile results
    #     results = {
    #         "final_system_agents": final_survival["system_alive"],
    #         "final_independent_agents": final_survival["independent_alive"],
    #         "timestamp": datetime.now().isoformat(),
    #     }

    #     # Add resource metrics
    #     for agent_type in final_resources.index:
    #         results[f"{agent_type.lower()}_avg_resources"] = final_resources.loc[
    #             agent_type, "avg_resources"
    #         ]

    #     return results

    # def generate_report(self) -> None:
    #     """Generate summary report of experiment results."""
    #     if not self.results:
    #         self.logger.warning("No results to generate report from")
    #         return

    #     # Convert results to DataFrame
    #     df = pd.DataFrame(self.results)

    #     # Save detailed results
    #     results_file = os.path.join(
    #         self.results_dir, f"{self.experiment_name}_results.csv"
    #     )
    #     df.to_csv(results_file, index=False)

    #     # Generate summary statistics
    #     summary = df.describe()
    #     summary_file = os.path.join(
    #         self.results_dir, f"{self.experiment_name}_summary.csv"
    #     )
    #     summary.to_csv(summary_file)

    #     self.logger.info(f"Report generated: {results_file}")
