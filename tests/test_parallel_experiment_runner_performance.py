"""
Performance tests for the ParallelExperimentRunner class.

This module contains performance benchmarks for the ParallelExperimentRunner,
comparing it with sequential execution and testing scaling with different
numbers of cores.

Note: These tests are marked as slow and should be run separately from the
regular test suite.
"""

import os
import tempfile
import time
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pytest

from farm.core.config import SimulationConfig
from farm.runners.experiment_runner import ExperimentRunner
from farm.runners.parallel_experiment_runner import ParallelExperimentRunner


@pytest.mark.slow
class TestParallelExperimentRunnerPerformance(unittest.TestCase):
    """Performance tests for the ParallelExperimentRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a basic simulation config for testing
        self.config = SimulationConfig()
        self.config.num_system_agents = 5
        self.config.num_independent_agents = 5
        self.config.world_size = (50, 50)
        self.config.use_in_memory_db = True
        self.config.in_memory_db_memory_limit_mb = 100

        # Create a test experiment name
        self.experiment_name = "test_parallel_performance"

        # Output directory for tests
        self.output_dir = Path(self.temp_dir.name) / "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Results directory for plots
        self.results_dir = Path(self.temp_dir.name) / "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_parallel_vs_sequential(self):
        """Benchmark parallel execution against sequential execution."""
        # Skip this test in CI environments
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping performance test in CI environment")

        # Number of iterations and steps
        num_iterations = 8
        num_steps = 20

        # Create sequential runner
        sequential_runner = ExperimentRunner(
            self.config,
            f"{self.experiment_name}_sequential",
            db_path=Path(self.temp_dir.name) / "sequential.db",
        )

        # Create parallel runner with all cores
        parallel_runner = ParallelExperimentRunner(
            self.config,
            f"{self.experiment_name}_parallel",
            n_jobs=-1,  # Use all cores
            db_path=Path(self.temp_dir.name) / "parallel.db",
            use_in_memory_db=True,
        )

        # Time sequential execution
        sequential_start = time.time()
        sequential_runner.run_iterations(
            num_iterations=num_iterations,
            num_steps=num_steps,
            path=self.output_dir / "sequential",
        )
        sequential_duration = time.time() - sequential_start

        # Time parallel execution
        parallel_start = time.time()
        parallel_runner.run_iterations(
            num_iterations=num_iterations,
            num_steps=num_steps,
            output_dir=self.output_dir / "parallel",
        )
        parallel_duration = time.time() - parallel_start

        # Calculate speedup
        speedup = sequential_duration / parallel_duration

        # Print results
        print(f"\nPerformance comparison:")
        print(f"Sequential execution: {sequential_duration:.2f} seconds")
        print(f"Parallel execution: {parallel_duration:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")

        # Assert that parallel execution is faster
        self.assertGreater(
            speedup, 1.0, "Parallel execution should be faster than sequential"
        )

        # Create a bar chart
        labels = ["Sequential", "Parallel"]
        durations = [sequential_duration, parallel_duration]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, durations, color=["blue", "green"])
        plt.ylabel("Execution Time (seconds)")
        plt.title("Sequential vs Parallel Execution Time")

        # Add speedup annotation
        plt.annotate(
            f"Speedup: {speedup:.2f}x",
            xy=(1, parallel_duration),
            xytext=(0.5, max(durations) * 0.7),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

        # Save the plot
        plt.savefig(self.results_dir / "sequential_vs_parallel.png")
        plt.close()

    def test_scaling_with_cores(self):
        """Test how performance scales with increasing number of cores."""
        # Skip this test in CI environments
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping performance test in CI environment")

        # Get the number of available cores
        total_cores = psutil.cpu_count(logical=True)

        # If we have less than 4 cores, skip this test
        if total_cores < 4:
            self.skipTest(f"Not enough cores for scaling test (found {total_cores})")

        # Number of iterations and steps
        num_iterations = 8
        num_steps = 20

        # Test with different numbers of cores
        core_counts = [1]
        if total_cores >= 2:
            core_counts.append(2)
        if total_cores >= 4:
            core_counts.append(4)
        if total_cores >= 8:
            core_counts.append(8)

        # Add all cores
        if total_cores not in core_counts:
            core_counts.append(total_cores)

        # Measure execution time for each core count
        durations = []

        for cores in core_counts:
            # Create runner with specified number of cores
            runner = ParallelExperimentRunner(
                self.config,
                f"{self.experiment_name}_{cores}_cores",
                n_jobs=cores,
                db_path=Path(self.temp_dir.name) / f"cores_{cores}.db",
                use_in_memory_db=True,
            )

            # Time execution
            start_time = time.time()
            runner.run_iterations(
                num_iterations=num_iterations,
                num_steps=num_steps,
                output_dir=self.output_dir / f"cores_{cores}",
            )
            duration = time.time() - start_time

            durations.append(duration)

            print(f"Execution with {cores} cores: {duration:.2f} seconds")

        # Calculate speedups relative to single core
        single_core_time = durations[0]
        speedups = [single_core_time / d for d in durations]

        # Print results
        print("\nScaling with number of cores:")
        for i, cores in enumerate(core_counts):
            print(
                f"{cores} cores: {durations[i]:.2f} seconds, speedup: {speedups[i]:.2f}x"
            )

        # Create a plot
        plt.figure(figsize=(12, 10))

        # Plot execution time vs cores
        plt.subplot(2, 1, 1)
        plt.plot(core_counts, durations, "o-", color="blue")
        plt.xlabel("Number of Cores")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Execution Time vs Number of Cores")
        plt.grid(True)

        # Plot speedup vs cores
        plt.subplot(2, 1, 2)
        plt.plot(core_counts, speedups, "o-", color="green")
        plt.plot(core_counts, core_counts, "--", color="red", label="Linear Speedup")
        plt.xlabel("Number of Cores")
        plt.ylabel("Speedup (relative to 1 core)")
        plt.title("Speedup vs Number of Cores")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.results_dir / "scaling_with_cores.png")
        plt.close()

        # Assert that adding more cores improves performance
        for i in range(1, len(core_counts)):
            self.assertLess(
                durations[i],
                durations[0],
                f"Execution with {core_counts[i]} cores should be faster than with 1 core",
            )

    def test_resource_usage(self):
        """Measure resource usage during parallel execution."""
        # Skip this test in CI environments
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping performance test in CI environment")

        # Number of iterations and steps
        num_iterations = 8
        num_steps = 20

        # Create parallel runner with all cores
        runner = ParallelExperimentRunner(
            self.config,
            f"{self.experiment_name}_resources",
            n_jobs=-1,  # Use all cores
            db_path=Path(self.temp_dir.name) / "resources.db",
            use_in_memory_db=True,
        )

        # Lists to store resource usage over time
        timestamps = []
        cpu_usage = []
        memory_usage = []

        # Function to monitor resource usage
        def monitor_resources():
            process = psutil.Process()
            while monitoring:
                timestamps.append(time.time() - start_time)
                cpu_usage.append(process.cpu_percent(interval=0.1))
                memory_usage.append(process.memory_info().rss / (1024 * 1024))  # MB
                time.sleep(0.5)

        # Start monitoring in a separate thread
        import threading

        monitoring = True
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.daemon = True

        # Start monitoring and run the experiment
        start_time = time.time()
        monitor_thread.start()

        runner.run_iterations(
            num_iterations=num_iterations,
            num_steps=num_steps,
            output_dir=self.output_dir / "resources",
        )

        # Stop monitoring
        monitoring = False
        monitor_thread.join(timeout=1.0)

        # Calculate statistics
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0
        max_cpu = np.max(cpu_usage) if cpu_usage else 0
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        max_memory = np.max(memory_usage) if memory_usage else 0

        # Print results
        print("\nResource usage during parallel execution:")
        print(f"Average CPU usage: {avg_cpu:.2f}%")
        print(f"Maximum CPU usage: {max_cpu:.2f}%")
        print(f"Average memory usage: {avg_memory:.2f} MB")
        print(f"Maximum memory usage: {max_memory:.2f} MB")

        # Create a plot
        plt.figure(figsize=(12, 8))

        # Plot CPU usage
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, cpu_usage, color="blue")
        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU Usage During Parallel Execution")
        plt.grid(True)

        # Plot memory usage
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, memory_usage, color="green")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage During Parallel Execution")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.results_dir / "resource_usage.png")
        plt.close()

        # Assert that we collected some data
        self.assertGreater(
            len(timestamps), 0, "Should have collected resource usage data"
        )

        # Assert that CPU usage is reasonable (should use multiple cores)
        total_cores = psutil.cpu_count(logical=True)
        self.assertGreater(
            max_cpu, 100 / total_cores, "Maximum CPU usage should be significant"
        )

    def test_database_configurations(self):
        """Test performance with different database configurations."""
        # Skip this test in CI environments
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping performance test in CI environment")

        # Number of iterations and steps
        num_iterations = 4
        num_steps = 20

        # Test different database configurations
        configs = [
            {
                "name": "In-memory DB",
                "use_in_memory_db": True,
                "in_memory_db_memory_limit_mb": None,
            },
            {
                "name": "In-memory DB with limit",
                "use_in_memory_db": True,
                "in_memory_db_memory_limit_mb": 50,
            },
            {"name": "Disk DB", "use_in_memory_db": False},
        ]

        durations = []

        for config in configs:
            # Create runner with this configuration
            runner = ParallelExperimentRunner(
                self.config,
                f"{self.experiment_name}_{config['name'].replace(' ', '_')}",
                n_jobs=2,  # Use 2 cores for consistency
                db_path=Path(self.temp_dir.name)
                / f"db_{config['name'].replace(' ', '_')}.db",
                use_in_memory_db=config["use_in_memory_db"],
                in_memory_db_memory_limit_mb=config.get("in_memory_db_memory_limit_mb"),
            )

            # Time execution
            start_time = time.time()
            runner.run_iterations(
                num_iterations=num_iterations,
                num_steps=num_steps,
                output_dir=self.output_dir / f"db_{config['name'].replace(' ', '_')}",
            )
            duration = time.time() - start_time

            durations.append(duration)

            print(f"Execution with {config['name']}: {duration:.2f} seconds")

        # Create a bar chart
        plt.figure(figsize=(12, 6))
        plt.bar([c["name"] for c in configs], durations, color=["blue", "green", "red"])
        plt.ylabel("Execution Time (seconds)")
        plt.xlabel("Database Configuration")
        plt.title("Performance with Different Database Configurations")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(self.results_dir / "database_configurations.png")
        plt.close()

        # Assert that in-memory DB is faster than disk DB
        self.assertLess(
            durations[0],
            durations[2],
            "In-memory database should be faster than disk database",
        )


if __name__ == "__main__":
    unittest.main()
