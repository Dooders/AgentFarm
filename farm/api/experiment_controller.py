"""Controller for managing simulation experiments.

This module provides a controller for:
- Running multiple simulation iterations with different configurations
- Managing experiment state and results
- Coordinating analysis and reporting
- Organizing experiment files and data
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from farm.analysis.comparative_analysis import compare_simulations
from farm.api.simulation_controller import SimulationController
from farm.charts.chart_analyzer import ChartAnalyzer
from farm.config import SimulationConfig
from farm.database.database import SimulationDatabase
from farm.research.research import ResearchProject
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExperimentController:
    """Controls and manages simulation experiments.

    This controller provides centralized management of simulation experiments including:
    - Running multiple iterations with different configurations
    - Managing experiment state and results
    - Coordinating analysis and reporting
    - Organizing experiment files and data

    Example usage:
        ```python
        # Initialize controller with experiment config
        config = SimulationConfig.from_centralized_config()
        controller = ExperimentController(
            name="agent_comparison",
            description="Compare different agent types",
            base_config=config
        )

        # Define variations for different iterations
        variations = [
            {"control_agents": 1, "system_agents": 0},
            {"control_agents": 0, "system_agents": 1}
        ]

        # Run experiment
        try:
            controller.run_experiment(
                num_iterations=10,
                variations=variations,
                num_steps=1000
            )

            # Generate analysis
            controller.analyze_results()

        finally:
            controller.cleanup()
        ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        base_config: SimulationConfig,
        project: Optional[ResearchProject] = None,
        output_dir: Optional[Path] = None,
    ):
        """Initialize experiment controller.

        Args:
            name: Name of the experiment
            description: Description of experiment purpose
            base_config: Base simulation configuration
            project: Optional research project this experiment belongs to
            output_dir: Optional custom output directory
        """
        self.name = name
        self.description = description
        self.base_config = base_config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup research project
        self.project = project or ResearchProject(name, description)

        # Setup output directory
        self.output_dir = (
            output_dir
            or Path(self.project.project_path)
            / "experiments"
            / f"{name}_{self.timestamp}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Track experiment state
        self.current_iteration = 0
        self.total_iterations = 0
        self.is_running = False
        self.results: List[Dict] = []

    def _setup_logging(self) -> None:
        """Configure experiment-specific logging."""
        from farm.utils.logging_config import get_logger

        # Create experiment-specific logger with bound context
        self.logger = get_logger(f"experiment.{self.name}").bind(
            experiment_name=self.name, experiment_id=self.timestamp
        )

        # Log experiment initialization
        self.logger.info(
            "experiment_logging_configured",
            log_file=str(self.output_dir / f"{self.name}.log"),
            output_dir=str(self.output_dir),
        )

    def run_experiment(
        self,
        num_iterations: int,
        variations: Optional[List[Dict]] = None,
        num_steps: int = 1000,
        run_analysis: bool = True,
    ) -> None:
        """Run experiment with multiple iterations.

        Args:
            num_iterations: Number of iterations to run
            variations: Optional list of config variations for each iteration
            num_steps: Number of steps per iteration
            run_analysis: Whether to run analysis after each iteration
        """
        self.logger.info(
            "experiment_starting",
            experiment_name=self.name,
            num_iterations=num_iterations,
            num_steps=num_steps,
            run_analysis=run_analysis,
        )
        self.is_running = True
        self.total_iterations = num_iterations

        try:
            for i in range(num_iterations):
                self.current_iteration = i + 1
                self.logger.info(
                    "iteration_starting",
                    iteration=self.current_iteration,
                    total_iterations=num_iterations,
                )

                # Create iteration directory
                iteration_dir = self.output_dir / f"iteration_{self.current_iteration}"
                iteration_dir.mkdir(exist_ok=True)

                # Create iteration config
                iteration_config = self._create_iteration_config(i, variations)

                # Run iteration
                self._run_iteration(iteration_config, iteration_dir, num_steps)

                # Run analysis if requested
                if run_analysis:
                    self._analyze_iteration(iteration_dir)

                self.logger.info(
                    "iteration_completed", iteration=self.current_iteration
                )

            self.logger.info(
                "experiment_completed",
                experiment_name=self.name,
                total_iterations=num_iterations,
            )

        except Exception as e:
            self.logger.error(
                "experiment_error",
                experiment_name=self.name,
                iteration=self.current_iteration,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise
        finally:
            self.is_running = False

    def _create_iteration_config(
        self, iteration: int, variations: Optional[List[Dict]]
    ) -> SimulationConfig:
        """Create configuration for specific iteration."""
        config = self.base_config.copy()

        if variations and iteration < len(variations):
            # Apply variation to config
            for key, value in variations[iteration].items():
                setattr(config, key, value)

        return config

    def _run_iteration(
        self, config: SimulationConfig, output_dir: Path, num_steps: int
    ) -> None:
        """Run single iteration with given configuration."""
        # Create simulation controller
        db_path = output_dir / "simulation.db"
        controller = SimulationController(config, str(db_path))

        try:
            # Initialize and run simulation
            controller.initialize_simulation()
            controller.start()

            # Wait for completion
            while controller.is_running:
                time.sleep(0.1)

            if controller.current_step < num_steps:
                raise RuntimeError(
                    f"Simulation stopped early at step {controller.current_step}"
                )

        finally:
            controller.cleanup()

    def _analyze_iteration(self, iteration_dir: Path) -> None:
        """Run analysis on iteration results."""
        db_path = iteration_dir / "simulation.db"
        if not db_path.exists():
            self.logger.warning(f"No database found at {db_path}")
            return

        try:
            # Create database connection for analysis
            db = SimulationDatabase(str(db_path))

            try:
                # Run chart analysis
                chart_analyzer = ChartAnalyzer(db)
                chart_analyzer.analyze_all_charts(iteration_dir)

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"Error analyzing iteration: {str(e)}")

    def analyze_results(self) -> None:
        """Run comparative analysis across all iterations."""
        if not self.output_dir.exists():
            self.logger.warning("No experiment output directory found")
            return

        try:
            # Create analysis directory
            analysis_dir = self.output_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)

            # Run comparative analysis
            compare_simulations(
                search_path=str(self.output_dir), analysis_path=str(analysis_dir)
            )

            self.logger.info("Comparative analysis completed")

        except Exception as e:
            self.logger.error(f"Error in comparative analysis: {str(e)}")
            raise

    def get_state(self) -> Dict:
        """Get current experiment state."""
        return {
            "name": self.name,
            "is_running": self.is_running,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "output_dir": str(self.output_dir),
            "timestamp": self.timestamp,
        }

    def cleanup(self) -> None:
        """Clean up experiment resources."""
        self.logger.info("Cleaning up experiment resources")
        # Additional cleanup if needed

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and ensure cleanup."""
        self.cleanup()
        # Don't suppress exceptions
        return False

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()
