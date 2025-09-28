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
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from farm.charts.chart_analyzer import ChartAnalyzer
from farm.core.config_hydra_bridge import HydraHydraSimulationConfig
from farm.core.simulation import run_simulation
from farm.database.database import SimulationDatabase
from scripts.significant_events import SignificantEventAnalyzer

DEFAULT_NUM_STEPS = 1000


class ExperimentRunner:
    """Manages multiple simulation runs and result analysis."""

    def __init__(
        self,
        base_config: HydraSimulationConfig,
        experiment_name: str,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize experiment runner.

        Parameters
        ----------
        base_config : HydraSimulationConfig
            Base configuration for simulations
        experiment_name : str
            Name of the experiment for organizing results
        db_path : Optional[Path]
            Explicit path for the simulation database. If None, uses default location.
        """
        self.base_config = base_config
        self.experiment_name = experiment_name
        self.results: List[Dict] = []

        # Setup experiment directories
        self.experiment_dir = os.path.join("experiments", experiment_name)
        self.db_dir = (
            os.path.dirname(str(db_path))
            if db_path
            else os.path.join(self.experiment_dir, "databases")
        )
        self.results_dir = os.path.join(self.experiment_dir, "results")

        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Store the database path
        self.db_path = db_path

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
        path: Optional[Path] = None,
        run_analysis: bool = True,
    ) -> None:
        """Run multiple iterations of the simulation.

        Parameters
        ----------
        num_iterations : int
            Number of simulation iterations to run
        config_variations : Optional[List[Dict]]
            List of configuration variations to apply to each iteration
        num_steps : int
            Number of simulation steps per iteration
        path : Optional[Path]
            Base path for storing simulation results
        run_analysis : bool
            Whether to run analysis after each iteration
        """
        self.logger.info(f"Starting experiment with {num_iterations} iterations")

        # Check if in-memory database is enabled in base config
        using_in_memory = getattr(self.base_config, "use_in_memory_db", False)
        if using_in_memory:
            self.logger.info("Using in-memory database for improved performance")

            # Log memory limit if configured
            memory_limit = getattr(
                self.base_config, "in_memory_db_memory_limit_mb", None
            )
            if memory_limit:
                self.logger.info(
                    f"Memory limit for in-memory database: {memory_limit} MB"
                )

        # Ensure path is a Path object
        if path and not isinstance(path, Path):
            path = Path(path)

        # Create progress bar for iterations
        progress_bar = tqdm(
            range(num_iterations),
            desc=f"Running {self.experiment_name}",
            unit="iteration",
        )

        for i in progress_bar:
            progress_bar.set_description(f"Running iteration {i+1}/{num_iterations}")
            self.logger.info(f"Starting iteration {i+1}/{num_iterations}")
            iteration_config = self._create_iteration_config(i, config_variations)
            iteration_path = path / f"iteration_{i+1}" if path else None

            try:
                if iteration_path:
                    iteration_path.mkdir(parents=True, exist_ok=True)

                # Run simulation
                env = run_simulation(
                    num_steps=num_steps,
                    config=iteration_config,
                    path=str(iteration_path) if iteration_path else None,
                )

                # Ensure all data is flushed
                if env.db:
                    env.db.logger.flush_all_buffers()

                self.logger.info(f"Completed iteration {i+1}")

                if run_analysis and iteration_path:
                    # Create a new database connection for analysis
                    db_path = iteration_path / "simulation.db"

                    # Skip analysis if using in-memory DB without persistence
                    if using_in_memory and not getattr(
                        iteration_config, "persist_db_on_completion", True
                    ):
                        self.logger.warning(
                            "Skipping analysis for iteration {i+1} - "
                            "in-memory database without persistence enabled"
                        )
                        continue

                    # Ensure database file exists before analysis
                    if not os.path.exists(db_path):
                        self.logger.warning(
                            f"Database file not found at {db_path}, skipping analysis"
                        )
                        continue

                    analysis_db = SimulationDatabase(str(db_path))

                    try:
                        # Run chart analysis with the new connection
                        chart_analyzer = ChartAnalyzer(analysis_db)
                        chart_analyzer.analyze_all_charts(iteration_path)

                        significant_event_analyzer = SignificantEventAnalyzer(
                            analysis_db
                        )
                        significant_event_analyzer.analyze_simulation(
                            start_step=0,
                            end_step=num_steps,
                            min_severity=0.3,
                            path=str(iteration_path),
                        )
                    finally:
                        # Clean up analysis database connection
                        analysis_db.close()

            except Exception as e:
                self.logger.error(f"Error in iteration {i+1}: {str(e)}")
                continue

    def _create_iteration_config(
        self, iteration: int, variations: Optional[List[Dict]] = None
    ) -> HydraSimulationConfig:
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
