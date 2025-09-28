"""
Memory database benchmark implementation.
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

from benchmarks.base.benchmark import Benchmark
from farm.core.config_hydra_models import HydraSimulationConfig
from farm.core.simulation import run_simulation


class MemoryDBBenchmark(Benchmark):
    """
    Benchmark for comparing disk-based and in-memory databases.

    This benchmark runs simulations with both disk-based and in-memory databases
    and compares their performance across different simulation sizes.

    The recommended configuration for production use is in-memory database with
    persistence enabled, which provides a good balance between performance and
    data durability for post-simulation analysis.
    """

    def __init__(
        self,
        num_steps: int = 100,
        num_agents: int = 30,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the memory database benchmark.

        Parameters
        ----------
        num_steps : int
            Number of simulation steps to run
        num_agents : int
            Total number of agents (divided equally among agent types)
        parameters : Dict[str, Any], optional
            Additional parameters for the benchmark
        """
        super().__init__(
            name="memory_db",
            description="Benchmark comparing disk-based and in-memory databases",
            parameters=parameters or {},
        )

        # Set benchmark-specific parameters
        self.parameters.update(
            {
                "num_steps": num_steps,
                "num_agents": num_agents,
            }
        )

        # Initialize benchmark-specific attributes
        self.temp_dir: Optional[str] = None
        self.config: Optional[HydraSimulationConfig] = None

    def setup(self) -> None:
        """
        Set up the benchmark environment.
        """
        # Create temporary directory for benchmark results
        self.temp_dir = tempfile.mkdtemp()

        # Create base configuration
        self.config = HydraSimulationConfig()

        # Set simulation parameters
        self.config.width = 100
        self.config.height = 100

        num_agents = self.parameters["num_agents"]
        self.config.system_agents = num_agents // 3
        self.config.independent_agents = num_agents // 3
        self.config.control_agents = num_agents - (
            2 * (num_agents // 3)
        )  # Ensure total is num_agents

        self.config.initial_resources = 20
        self.config.simulation_steps = self.parameters["num_steps"]

    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.

        Returns
        -------
        Dict[str, Any]
            Raw results from the benchmark run
        """
        num_steps = self.parameters["num_steps"]

        # Results storage
        disk_times = []
        memory_times = []
        memory_persist_times = []

        # Create iteration directory
        if self.temp_dir is None:
            raise RuntimeError("temp_dir not initialized")
        iter_dir = os.path.join(self.temp_dir, "iteration")
        os.makedirs(iter_dir, exist_ok=True)

        # Run with disk database
        print("Running with disk database...")
        disk_dir = os.path.join(iter_dir, "disk")
        os.makedirs(disk_dir, exist_ok=True)

        if self.config is None:
            raise RuntimeError("config not initialized")
        self.config.use_in_memory_db = False

        disk_start = time.time()
        disk_env = run_simulation(
            num_steps=num_steps,
            config=self.config,
            path=disk_dir,
            save_config=True,
        )
        disk_end = time.time()
        disk_time = disk_end - disk_start
        disk_times.append(disk_time)

        print(f"Disk database time: {disk_time:.2f} seconds")

        # Run with in-memory database
        print("Running with in-memory database (no persistence)...")
        memory_dir = os.path.join(iter_dir, "memory")
        os.makedirs(memory_dir, exist_ok=True)

        if self.config is None:
            raise RuntimeError("config not initialized")
        self.config.use_in_memory_db = True
        self.config.persist_db_on_completion = False

        memory_start = time.time()
        memory_env = run_simulation(
            num_steps=num_steps,
            config=self.config,
            path=memory_dir,
            save_config=True,
        )
        memory_end = time.time()
        memory_time = memory_end - memory_start
        memory_times.append(memory_time)

        print(f"In-memory database time: {memory_time:.2f} seconds")

        # Run with in-memory database with persistence (RECOMMENDED FOR PRODUCTION)
        print(
            "Running with in-memory database (with persistence) - RECOMMENDED FOR PRODUCTION..."
        )
        memory_persist_dir = os.path.join(iter_dir, "memory_persist")
        os.makedirs(memory_persist_dir, exist_ok=True)

        # This is the recommended configuration for production use
        # It provides a good balance between performance and data durability
        if self.config is None:
            raise RuntimeError("config not initialized")
        self.config.use_in_memory_db = True
        self.config.persist_db_on_completion = True

        memory_persist_start = time.time()
        memory_persist_env = run_simulation(
            num_steps=num_steps,
            config=self.config,
            path=memory_persist_dir,
            save_config=True,
        )
        memory_persist_end = time.time()
        memory_persist_time = memory_persist_end - memory_persist_start
        memory_persist_times.append(memory_persist_time)

        print(
            f"In-memory database with persistence time: {memory_persist_time:.2f} seconds"
        )
        print(
            "NOTE: This configuration is recommended for production use when post-simulation analysis is required."
        )

        # Calculate speedups
        disk_to_memory_speedup = (
            disk_time / memory_time if memory_time > 0 else float("inf")
        )
        disk_to_memory_persist_speedup = (
            disk_time / memory_persist_time if memory_persist_time > 0 else float("inf")
        )

        print(f"Speedup (disk to memory): {disk_to_memory_speedup:.2f}x")
        print(
            f"Speedup (disk to memory with persistence): {disk_to_memory_persist_speedup:.2f}x"
        )

        # Return results
        return {
            "disk_time": disk_time,
            "memory_time": memory_time,
            "memory_persist_time": memory_persist_time,
            "disk_to_memory_speedup": disk_to_memory_speedup,
            "disk_to_memory_persist_speedup": disk_to_memory_persist_speedup,
        }

    def cleanup(self) -> None:
        """
        Clean up after the benchmark.
        """
        # Remove temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
