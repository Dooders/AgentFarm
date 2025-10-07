"""
Parallel experiment runner for executing multiple simulations concurrently.

This module provides a parallel implementation of the experiment runner that
can execute multiple simulations across multiple processes, taking advantage
of multi-core systems for improved performance.
"""

import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import joblib
import psutil
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.database.database import SimulationDatabase
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class ParallelExperimentRunner:
    """Run multiple simulation experiments in parallel.

    This class provides functionality to run multiple simulation experiments
    concurrently using process-based parallelism, with configurable worker
    count and resource management.

    Attributes
    ----------
    base_config : SimulationConfig
        Base configuration for experiments
    experiment_name : str
        Name of the experiment
    n_jobs : int
        Number of parallel workers to use
    db_path : Optional[Path]
        Path to database file
    use_in_memory_db : bool
        Whether to use in-memory database during simulation
    """

    def __init__(
        self,
        base_config: SimulationConfig,
        experiment_name: str,
        n_jobs: int = -1,  # -1 means use all available cores
        db_path: Optional[Path] = None,
        use_in_memory_db: bool = True,
        in_memory_db_memory_limit_mb: Optional[int] = None,
    ):
        """Initialize the parallel experiment runner.

        Parameters
        ----------
        base_config : SimulationConfig
            Base configuration for experiments
        experiment_name : str
            Name of the experiment
        n_jobs : int, optional
            Number of parallel workers to use, by default -1 (all cores)
        db_path : Optional[Path], optional
            Path to database file, by default None
        use_in_memory_db : bool, optional
            Whether to use in-memory database during simulation, by default True
        in_memory_db_memory_limit_mb : Optional[int], optional
            Memory limit for in-memory database in MB, by default None
        """
        self.base_config = base_config
        self.experiment_name = experiment_name
        self.n_jobs = n_jobs
        self.db_path = db_path
        self.use_in_memory_db = use_in_memory_db
        self.in_memory_db_memory_limit_mb = in_memory_db_memory_limit_mb

        # Set up logging
        from farm.utils.logging import get_logger
        self.logger = get_logger(f"parallel_experiment.{experiment_name}")

        # Ensure base_config has in-memory DB settings
        if hasattr(base_config, "use_in_memory_db"):
            base_config.use_in_memory_db = use_in_memory_db
        if hasattr(base_config, "in_memory_db_memory_limit_mb"):
            base_config.in_memory_db_memory_limit_mb = in_memory_db_memory_limit_mb

    def run_single_simulation(
        self,
        config: SimulationConfig,
        num_steps: int,
        output_path: Path,
        seed: Optional[int] = None,
    ) -> Dict:
        """Run a single simulation with the given configuration.

        Parameters
        ----------
        config : SimulationConfig
            Configuration for this simulation
        num_steps : int
            Number of steps to run
        output_path : Path
            Path to save simulation results
        seed : Optional[int], optional
            Random seed for reproducibility, by default None

        Returns
        -------
        Dict
            Results summary from the simulation
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)

            # Run simulation
            environment = run_simulation(
                num_steps=num_steps,
                config=config,
                path=str(output_path),
                save_config=True,
                seed=seed,
            )

            # If using in-memory database, ensure it's persisted to disk
            if (
                self.use_in_memory_db
                and hasattr(environment, "db")
                and environment.db is not None
            ):
                db_file = output_path / "simulation.db"
                if hasattr(environment.db, "persist_to_disk"):
                    environment.db.persist_to_disk(str(db_file))
                    self.logger.info(f"Persisted in-memory database to {db_file}")

            # Collect and return results
            results = {
                "final_agent_count": len(environment.agents),
                "config": (
                    config.to_dict() if hasattr(config, "to_dict") else str(config)
                ),
                "output_path": str(output_path),
                "success": True,
            }

            return results
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            return {"error": str(e), "output_path": str(output_path), "success": False}

    def _run_with_error_handling(self, config, num_steps, output_path, seed):
        """Run a simulation with comprehensive error handling.

        Parameters
        ----------
        config : SimulationConfig
            Configuration for this simulation
        num_steps : int
            Number of steps to run
        output_path : Path
            Path to save simulation results
        seed : Optional[int]
            Random seed for reproducibility

        Returns
        -------
        Dict
            Results or error information
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)

            # Run simulation
            environment = run_simulation(
                num_steps=num_steps,
                config=config,
                path=str(output_path),
                save_config=True,
                seed=seed,
            )

            # If using in-memory database, ensure it's persisted to disk
            if (
                self.use_in_memory_db
                and hasattr(environment, "db")
                and environment.db is not None
            ):
                db_file = output_path / "simulation.db"
                if hasattr(environment.db, "persist_to_disk"):
                    environment.db.persist_to_disk(str(db_file))
                    self.logger.info(f"Persisted in-memory database to {db_file}")

            # Collect and return results
            results = {
                "final_agent_count": len(environment.agents),
                "config": (
                    config.to_dict() if hasattr(config, "to_dict") else str(config)
                ),
                "output_path": str(output_path),
                "success": True,
            }

            return results
        except Exception as e:
            # Log the error
            error_msg = f"Error in simulation: {e}"
            self.logger.error(error_msg)

            # Save error details to file
            os.makedirs(output_path, exist_ok=True)
            error_path = output_path / "error.log"
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Error: {str(e)}\n\n")
                f.write(traceback.format_exc())

            # Return error information
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "output_path": str(output_path),
                "success": False,
            }

    def _configure_process_pool(self):
        """Configure the process pool for optimal performance.

        Returns
        -------
        Dict
            Process pool configuration
        """
        # Get system resources
        total_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        available_memory = psutil.virtual_memory().available

        # Calculate optimal number of workers
        if self.n_jobs == -1:
            # Use all cores by default
            n_jobs = total_cores
        elif self.n_jobs <= 0:
            # Use a percentage of cores
            n_jobs = max(1, int(total_cores * abs(self.n_jobs)))
        else:
            # Use specified number of cores
            n_jobs = min(self.n_jobs, total_cores)

        # Estimate memory per worker
        estimated_memory_per_worker = (
            2 * 1024 * 1024 * 1024
        )  # 2GB per worker (estimate)
        max_workers_by_memory = max(1, available_memory // estimated_memory_per_worker)

        # Adjust workers based on available memory
        n_jobs = min(n_jobs, max_workers_by_memory)

        self.logger.info(f"Configured process pool with {n_jobs} workers")
        self.logger.info(
            f"System has {total_cores} logical cores, {physical_cores} physical cores"
        )
        self.logger.info(f"Available memory: {available_memory / (1024**3):.2f} GB")

        return {
            "n_jobs": n_jobs,
            "backend": "multiprocessing",
            "verbose": 10,
            "timeout": None,  # No timeout
            "pre_dispatch": "2*n_jobs",  # Pre-dispatch twice as many jobs
            "batch_size": "auto",
            "temp_folder": None,
            "max_nbytes": None,
            "mmap_mode": "r",
        }

    def _log_resource_usage(self):
        """Log current resource usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        self.logger.info(
            f"Resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%"
        )

        # Warn if resources are running low
        if memory_percent > 90:
            self.logger.warning("High memory usage detected, consider reducing n_jobs")

    def _cleanup_resources(self):
        """Clean up resources after parallel execution."""
        import gc

        # Force garbage collection
        gc.collect()

        # Close any open database connections
        # This would need to be implemented if we maintain any shared resources

        self.logger.info("Cleaned up resources after parallel execution")

    def _save_summary(self, results: List[Dict], output_dir: Path) -> None:
        """Save summary of all simulation results.

        Parameters
        ----------
        results : List[Dict]
            Results from all simulations
        output_dir : Path
            Directory to save summary
        """
        import json

        # Create summary with basic statistics
        summary = {
            "experiment_name": self.experiment_name,
            "num_iterations": len(results),
            "successful_runs": sum(1 for r in results if r.get("success", False)),
            "failed_runs": sum(1 for r in results if not r.get("success", False)),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }

        # Save summary to file
        summary_path = output_dir / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Experiment summary saved to {summary_path}")

    def run_iterations(
        self,
        num_iterations: int,
        variations: Optional[List[Dict]] = None,
        num_steps: int = 1000,
        output_dir: Optional[Union[str, Path]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Run multiple iterations of the experiment in parallel.

        Parameters
        ----------
        num_iterations : int
            Number of iterations to run
        variations : Optional[List[Dict]], optional
            List of configuration variations, by default None
        num_steps : int, optional
            Number of steps per simulation, by default 1000
        output_dir : Optional[Union[str, Path]], optional
            Directory to save results, by default None
        seeds : Optional[List[int]], optional
            Random seeds for reproducibility, by default None

        Returns
        -------
        List[Dict]
            Results from all simulations
        """
        # Prepare output directory
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"experiments/{self.experiment_name}_{timestamp}")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Running {num_iterations} iterations in parallel")

        # Log initial resource usage
        self._log_resource_usage()

        # Prepare configurations
        configs = []
        for i in range(num_iterations):
            # Create a copy of the base config
            config = (
                self.base_config.copy()
                if hasattr(self.base_config, "copy")
                else self.base_config
            )

            # Apply variations if provided
            if variations and i < len(variations):
                for key, value in variations[i].items():
                    setattr(config, key, value)

            # Get seed if provided
            seed = seeds[i] if seeds and i < len(seeds) else None

            # Create output path for this iteration
            iter_path = output_dir / f"iteration_{i}"

            configs.append((config, num_steps, iter_path, seed))

        try:
            # Configure process pool
            pool_config = self._configure_process_pool()
            n_jobs = pool_config["n_jobs"]
            # Remove n_jobs from pool_config to avoid duplicate keyword argument
            pool_config = {k: v for k, v in pool_config.items() if k != "n_jobs"}

            # Remove the callback parameter if it exists in pool_config
            if "callback" in pool_config:
                del pool_config["callback"]

            # Create a progress bar but don't use it in the Parallel call
            pbar = tqdm(
                total=num_iterations,
                desc=f"Running {self.experiment_name} iterations",
                position=0,
                leave=True,
                unit="iter",
            )

            # Run simulations in parallel without the callback
            results = cast(
                List[Dict],
                list(
                    Parallel(n_jobs=n_jobs, **pool_config)(
                        delayed(self._run_with_error_handling)(
                            config, steps, path, seed
                        )
                        for config, steps, path, seed in configs
                    )
                ),
            )

            # Update the progress bar to show completion
            pbar.update(num_iterations)
            pbar.close()

            # Log final resource usage
            self._log_resource_usage()

            # Save summary of all results
            self._save_summary(results, output_dir)

            return results
        except Exception as e:
            self.logger.error(f"Error in parallel execution: {e}")
            results = [{"error": str(e), "global_error": True, "success": False}]
            return results
        finally:
            # Clean up resources
            self._cleanup_resources()
