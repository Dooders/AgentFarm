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
import json

import pandas as pd
from tqdm import tqdm

from farm.charts.chart_analyzer import ChartAnalyzer
from farm.core.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.database.database import SimulationDatabase
from farm.database.models import Experiment, ExperimentMetric, Simulation
from scripts.significant_events import SignificantEventAnalyzer
from farm.core.environment import Environment

DEFAULT_NUM_STEPS = 1000


class ExperimentRunner:
    """Manages multiple simulation runs and result analysis."""

    def __init__(
        self,
        base_config: SimulationConfig,
        experiment_name: str,
        db_path: Optional[Path] = None,
    ):
        """Initialize experiment runner.

        Parameters
        ----------
        base_config : SimulationConfig
            Base configuration for simulations
        experiment_name : str
            Name of the experiment for organizing results
        db_path : Optional[Path]
            Path to the database file. If None, uses default location.
        """
        self.base_config = base_config
        self.experiment_name = experiment_name
        
        # Setup database
        self.db_path = db_path or Path("experiments/simulation.db")
        self.db = SimulationDatabase(str(self.db_path))
        
        # Create experiment record
        self.experiment_id = self.db.create_experiment(
            name=experiment_name,
            base_config=base_config.to_dict()
        )
        
        # Setup logging
        self._setup_logging()
        
        # Log experiment creation
        self.db.log_experiment_event('experiment_created', {
            'name': experiment_name,
            'base_config': base_config.to_dict()
        })

    def _setup_logging(self):
        """Configure experiment-specific logging."""
        log_dir = os.path.join("experiments", self.experiment_name)
        log_file = os.path.join(log_dir, f"{self.experiment_name}.log")
        
        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger(f"experiment.{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

    def run_iterations(self, num_iterations=1, config_variations=None):
        """Run multiple iterations of the experiment.
        
        Parameters
        ----------
        num_iterations : int, optional
            Number of iterations to run
        config_variations : List[Dict], optional
            List of configuration variations to apply
        """
        self.db.set_experiment_status('running')
        
        try:
            for i in range(num_iterations):
                self.logger.info(f"Starting iteration {i+1}/{num_iterations}")
                
                # Apply configuration variation if provided
                config = self.base_config
                variation = None
                if config_variations and i < len(config_variations):
                    variation = config_variations[i]
                    config = self._apply_config_variation(config, variation)
                
                # Create a new database connection for each iteration to avoid ID conflicts
                if i > 0:
                    # Close the previous database connection
                    if hasattr(self, '_temp_db') and self._temp_db:
                        self._temp_db.close()
                    
                    # Create a new database connection
                    self._temp_db = SimulationDatabase(self.db_path)
                    self._temp_db.current_experiment_id = self.experiment_id
                    db_for_iteration = self._temp_db
                else:
                    db_for_iteration = self.db
                
                # Create environment with experiment context
                environment = Environment(
                    config=config,
                    database=db_for_iteration,
                    experiment_id=self.experiment_id,
                    iteration_number=i+1
                )
                
                # Run simulation
                try:
                    self._run_iteration(environment, i+1, variation)
                except Exception as e:
                    self.logger.error(f"Error in iteration {i+1}: {e}")
                    self.db.log_experiment_event('iteration_error', {
                        'iteration': i+1,
                        'error': str(e)
                    })
                finally:
                    if hasattr(environment, 'db'):
                        environment.db.close()
                
            # Update experiment status
            self.db.set_experiment_status('completed')
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.db.set_experiment_status('failed')
            raise
            
        # Clean up temporary database connections
        if hasattr(self, '_temp_db') and self._temp_db:
            self._temp_db.close()

    def _run_iteration(self, environment: Environment, iteration: int, variation: Dict = None):
        """Run a single iteration of the experiment.
        
        Parameters
        ----------
        environment : Environment
            The simulation environment
        iteration : int
            Iteration number
        variation : Dict, optional
            Configuration variation applied
        """
        # Log iteration start
        self.db.log_experiment_event('iteration_started', {
            'iteration': iteration,
            'variation': variation
        })
        
        try:
            # Run simulation steps
            for step in range(self.base_config.simulation_steps):
                environment.step()
                
                # Log metrics every N steps
                if step % 100 == 0:
                    metrics = environment.get_metrics()
                    for name, value in metrics.items():
                        self.db.log_experiment_metric(
                            metric_name=name,
                            metric_value=value,
                            metric_type='simulation',
                            metadata={'iteration': iteration, 'step': step}
                        )
                        
            # Update simulation status to completed
            if hasattr(environment, 'simulation_id'):
                self.db._execute_in_transaction(
                    lambda session: session.query(Simulation)
                    .filter_by(simulation_id=environment.simulation_id)
                    .update({"status": "completed"})
                )
                        
            # Log iteration completion
            self.db.log_experiment_event('iteration_completed', {
                'iteration': iteration,
                'final_metrics': environment.get_metrics()
            })
            
        except Exception as e:
            self.logger.error(f"Error in iteration {iteration}: {e}")
            raise

    def _apply_config_variation(self, base_config: SimulationConfig, variation: Dict) -> SimulationConfig:
        """Apply configuration variations to base config.
        
        Parameters
        ----------
        base_config : SimulationConfig
            Base configuration
        variation : Dict
            Configuration variations to apply
            
        Returns
        -------
        SimulationConfig
            Modified configuration
            
        Raises
        ------
        ValueError
            If an invalid configuration parameter is provided
        """
        # Create a copy of the base config
        config_dict = base_config.to_dict()
        
        # Apply variations
        for key, value in variation.items():
            if hasattr(base_config, key):
                config_dict[key] = value
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")
                raise ValueError(f"Invalid configuration parameter: {key}")
                
        return SimulationConfig.from_dict(config_dict)

    def generate_report(self):
        """Generate experiment report."""
        try:
            # Get experiment details
            experiment = self.db.get_experiment(self.experiment_id)
            
            # Get metrics
            def _get_metrics(session):
                return session.query(ExperimentMetric).filter_by(
                    experiment_id=self.experiment_id
                ).all()
            metrics = self.db._execute_in_transaction(_get_metrics)
            
            # Generate report
            report = {
                'experiment': experiment,
                'metrics': [metric.as_dict() for metric in metrics],
                'generated_at': datetime.now().isoformat()
            }
            
            # Save report
            report_path = Path(f"experiments/{self.experiment_name}_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            raise

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
