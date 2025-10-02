#!/usr/bin/env python3
"""
Database Logging Profiler - Phase 2 Component-Level Profiling

Profiles database operations to identify bottlenecks:
- Batch vs. individual inserts
- Different buffer sizes (100, 500, 1000, 5000)
- SQLite PRAGMA configurations
- In-memory vs. disk performance
- Flush time distribution
- Insert throughput

Usage:
    python -m benchmarks.implementations.profiling.database_profiler
"""

import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from farm.database.database import InMemorySimulationDatabase, SimulationDatabase


class DatabaseProfiler:
    """Profile database logging operations."""

    def __init__(self, results_dir: str = None):
        self.results = {}
        self.temp_dir = tempfile.mkdtemp()
        self.timestamp = datetime.now()

        # Set up results directory
        if results_dir is None:
            # Default to benchmarks/results/database_profiling/
            self.results_dir = (
                Path(__file__).parent.parent.parent.parent
                / "benchmarks"
                / "results"
                / "database_profiling"
            )
        else:
            self.results_dir = Path(results_dir)

        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def profile_insert_patterns(self, record_counts: List[int], batch_sizes: List[int]):
        """Profile different insert patterns."""
        print("\n" + "=" * 60)
        print("Profiling Insert Patterns")
        print("=" * 60 + "\n")

        results = {}

        for num_records in record_counts:
            print(f"\nTesting with {num_records} records...")
            record_results = {}

            for batch_size in batch_sizes:
                db_path = os.path.join(
                    self.temp_dir, f"test_{num_records}_{batch_size}.db"
                )

                # Create database
                db = SimulationDatabase(db_path=db_path, simulation_id="test_sim")

                # Create simulation record (required for foreign key constraints)
                db.add_simulation_record(
                    simulation_id="test_sim",
                    start_time=datetime.now(),
                    status="running",
                    parameters={},
                )

                # Configure buffer size
                db.logger.agent_buffer_size = batch_size
                db.logger.step_buffer_size = batch_size

                # Create agent records first (required for foreign key constraint)
                for i in range(100):  # Create 100 unique agents
                    db.logger.log_agent(
                        agent_id=f"agent_{i}",
                        birth_time=0,
                        agent_type="test",
                        position=(0.0, 0.0),
                        initial_resources=10.0,
                        starting_health=100.0,
                        starvation_counter=0,
                        genome_id=f"genome_{i}",
                        generation=0,
                        action_weights={},
                    )

                # Flush agent records
                db.logger.flush_all_buffers()

                # Profile inserts
                start = time.perf_counter()

                for i in range(num_records):
                    db.logger.log_agent_action(
                        step_number=i,
                        agent_id=f"agent_{i % 100}",
                        action_type="move",
                        resources_before=10.0,
                        resources_after=9.5,
                        reward=0.1,
                        details={},
                    )

                # Flush remaining
                db.logger.flush_all_buffers()

                insert_time = time.perf_counter() - start

                record_results[batch_size] = {
                    "total_time": insert_time,
                    "inserts_per_second": (
                        num_records / insert_time if insert_time > 0 else 0
                    ),
                    "time_per_insert": (
                        insert_time / num_records if num_records > 0 else 0
                    ),
                }

                inserts_per_sec = (num_records / insert_time) if insert_time > 0 else 0
                us_per_insert = (
                    (insert_time * 1000000 / num_records) if num_records > 0 else 0
                )
                print(
                    f"  Batch {batch_size:>4}: {insert_time*1000:.2f}ms, "
                    f"{inserts_per_sec:.0f} inserts/s, "
                    f"{us_per_insert:.2f}us per insert"
                )

                # Clean up
                db.close()
                # Wait a moment for file handles to be released
                time.sleep(0.1)
                try:
                    os.remove(db_path)
                except PermissionError:
                    # File might still be locked, skip removal
                    pass

            results[num_records] = record_results

        self.results["insert_patterns"] = results
        return results

    def profile_buffer_sizes(self, num_records: int):
        """Profile different buffer sizes."""
        print("\n" + "=" * 60)
        print(f"Profiling Buffer Sizes ({num_records} records)")
        print("=" * 60 + "\n")

        buffer_sizes = [10, 50, 100, 500, 1000, 5000]
        results = {}

        for buffer_size in buffer_sizes:
            db_path = os.path.join(self.temp_dir, f"test_buffer_{buffer_size}.db")

            # Create database
            db = SimulationDatabase(db_path=db_path, simulation_id="test_sim")

            # Create simulation record (required for foreign key constraints)
            db.add_simulation_record(
                simulation_id="test_sim",
                start_time=datetime.now(),
                status="running",
                parameters={},
            )

            db.logger.agent_buffer_size = buffer_size

            # Create agent records first (required for foreign key constraint)
            for i in range(100):  # Create 100 unique agents
                db.logger.log_agent(
                    agent_id=f"agent_{i}",
                    birth_time=0,
                    agent_type="test",
                    position=(0.0, 0.0),
                    initial_resources=10.0,
                    starting_health=100.0,
                    starvation_counter=0,
                    genome_id=f"genome_{i}",
                    generation=0,
                    action_weights={},
                )

            # Flush agent records
            db.logger.flush_all_buffers()

            # Profile with this buffer size
            start = time.perf_counter()

            for i in range(num_records):
                db.logger.log_agent_action(
                    step_number=i,
                    agent_id=f"agent_{i % 100}",
                    action_type="move",
                    resources_before=10.0,
                    resources_after=9.5,
                    reward=0.1,
                    details={},
                )

            db.logger.flush_all_buffers()
            total_time = time.perf_counter() - start

            results[buffer_size] = {
                "total_time": total_time,
                "inserts_per_second": num_records / total_time if total_time > 0 else 0,
                "time_per_insert": total_time / num_records if num_records > 0 else 0,
            }

            inserts_per_sec = (num_records / total_time) if total_time > 0 else 0
            print(
                f"  Buffer {buffer_size:>5}: {total_time*1000:.2f}ms, "
                f"{inserts_per_sec:.0f} inserts/s"
            )

            # Clean up
            db.close()
            # Wait a moment for file handles to be released
            time.sleep(0.1)
            try:
                os.remove(db_path)
            except PermissionError:
                # File might still be locked, skip removal
                pass

        self.results["buffer_sizes"] = results
        return results

    def profile_memory_vs_disk(self, num_records: int):
        """Compare in-memory vs disk database performance."""
        print("\n" + "=" * 60)
        print(f"Comparing In-Memory vs Disk ({num_records} records)")
        print("=" * 60 + "\n")

        results = {}

        # Test disk database
        print("Testing disk database...")
        db_path = os.path.join(self.temp_dir, "test_disk.db")
        db_disk = SimulationDatabase(db_path=db_path, simulation_id="test_sim")

        # Create simulation record (required for foreign key constraints)
        db_disk.add_simulation_record(
            simulation_id="test_sim",
            start_time=datetime.now(),
            status="running",
            parameters={},
        )

        # Create agent records first (required for foreign key constraint)
        for i in range(100):  # Create 100 unique agents
            db_disk.logger.log_agent(
                agent_id=f"agent_{i}",
                birth_time=0,
                agent_type="test",
                position=(0.0, 0.0),
                initial_resources=10.0,
                starting_health=100.0,
                starvation_counter=0,
                genome_id=f"genome_{i}",
                generation=0,
                action_weights={},
            )

        # Flush agent records
        db_disk.logger.flush_all_buffers()

        start = time.perf_counter()
        for i in range(num_records):
            db_disk.logger.log_agent_action(
                step_number=i,
                agent_id=f"agent_{i % 100}",
                action_type="move",
                resources_before=10.0,
                resources_after=9.5,
                reward=0.1,
                details={},
            )
        db_disk.logger.flush_all_buffers()
        disk_time = time.perf_counter() - start

        results["disk"] = {
            "total_time": disk_time,
            "inserts_per_second": num_records / disk_time if disk_time > 0 else 0,
        }

        inserts_per_sec = (num_records / disk_time) if disk_time > 0 else 0
        print(f"  Disk: {disk_time*1000:.2f}ms, {inserts_per_sec:.0f} inserts/s")

        db_disk.close()
        # Wait a moment for file handles to be released
        time.sleep(0.1)
        try:
            os.remove(db_path)
        except PermissionError:
            # File might still be locked, skip removal
            pass

        # Test in-memory database
        print("\nTesting in-memory database...")
        db_memory = InMemorySimulationDatabase(
            memory_limit_mb=100, simulation_id="test_sim"
        )

        # Create simulation record (required for foreign key constraints)
        db_memory.add_simulation_record(
            simulation_id="test_sim",
            start_time=datetime.now(),
            status="running",
            parameters={},
        )

        # Create agent records first (required for foreign key constraint)
        for i in range(100):  # Create 100 unique agents
            db_memory.logger.log_agent(
                agent_id=f"agent_{i}",
                birth_time=0,
                agent_type="test",
                position=(0.0, 0.0),
                initial_resources=10.0,
                starting_health=100.0,
                starvation_counter=0,
                genome_id=f"genome_{i}",
                generation=0,
                action_weights={},
            )

        # Flush agent records
        db_memory.logger.flush_all_buffers()

        start = time.perf_counter()
        for i in range(num_records):
            db_memory.logger.log_agent_action(
                step_number=i,
                agent_id=f"agent_{i % 100}",
                action_type="move",
                resources_before=10.0,
                resources_after=9.5,
                reward=0.1,
                details={},
            )
        db_memory.logger.flush_all_buffers()
        memory_time = time.perf_counter() - start

        results["memory"] = {
            "total_time": memory_time,
            "inserts_per_second": num_records / memory_time if memory_time > 0 else 0,
        }

        inserts_per_sec = (num_records / memory_time) if memory_time > 0 else 0
        print(f"  Memory: {memory_time*1000:.2f}ms, {inserts_per_sec:.0f} inserts/s")

        # Calculate speedup
        speedup = disk_time / memory_time if memory_time > 0 else 0
        results["speedup"] = speedup

        print(f"\n  Speedup: {speedup:.2f}x (memory faster)")

        db_memory.close()

        self.results["memory_vs_disk"] = results
        return results

    def profile_flush_frequency(self, num_records: int, flush_intervals: List[int]):
        """Profile impact of different flush frequencies."""
        print("\n" + "=" * 60)
        print(f"Profiling Flush Frequency ({num_records} records)")
        print("=" * 60 + "\n")

        results = {}

        for flush_interval in flush_intervals:
            db_path = os.path.join(self.temp_dir, f"test_flush_{flush_interval}.db")
            db = SimulationDatabase(db_path=db_path, simulation_id="test_sim")

            # Create simulation record (required for foreign key constraints)
            db.add_simulation_record(
                simulation_id="test_sim",
                start_time=datetime.now(),
                status="running",
                parameters={},
            )

            # Very large buffer to control flush frequency
            db.logger.agent_buffer_size = 100000

            # Create agent records first (required for foreign key constraint)
            for i in range(100):  # Create 100 unique agents
                db.logger.log_agent(
                    agent_id=f"agent_{i}",
                    birth_time=0,
                    agent_type="test",
                    position=(0.0, 0.0),
                    initial_resources=10.0,
                    starting_health=100.0,
                    starvation_counter=0,
                    genome_id=f"genome_{i}",
                    generation=0,
                    action_weights={},
                )

            # Flush agent records
            db.logger.flush_all_buffers()

            flush_count = 0
            start = time.perf_counter()

            for i in range(num_records):
                db.logger.log_agent_action(
                    step_number=i,
                    agent_id=f"agent_{i % 100}",
                    action_type="move",
                    resources_before=10.0,
                    resources_after=9.5,
                    reward=0.1,
                    details={},
                )

                # Manual flush at intervals
                if (i + 1) % flush_interval == 0:
                    db.logger.flush_all_buffers()
                    flush_count += 1

            # Final flush
            db.logger.flush_all_buffers()
            total_time = time.perf_counter() - start

            results[flush_interval] = {
                "total_time": total_time,
                "flush_count": flush_count,
                "inserts_per_second": num_records / total_time if total_time > 0 else 0,
                "time_per_flush": total_time / flush_count if flush_count > 0 else 0,
            }

            inserts_per_sec = (num_records / total_time) if total_time > 0 else 0
            print(
                f"  Flush every {flush_interval:>5}: {total_time*1000:.2f}ms, "
                f"{flush_count} flushes, {inserts_per_sec:.0f} inserts/s"
            )

            db.close()
            # Wait a moment for file handles to be released
            time.sleep(0.1)
            try:
                os.remove(db_path)
            except PermissionError:
                # File might still be locked, skip removal
                pass

        self.results["flush_frequency"] = results
        return results

    def generate_report(self):
        """Generate a summary report of profiling results."""
        print("\n" + "=" * 60)
        print("Database Logging Profiling Report")
        print("=" * 60 + "\n")

        # Insert patterns
        if "insert_patterns" in self.results:
            print("## Insert Pattern Performance\n")
            for num_records, batch_results in sorted(
                self.results["insert_patterns"].items()
            ):
                print(f"  {num_records} records:")
                for batch_size, data in sorted(batch_results.items()):
                    print(
                        f"    Batch {batch_size:>4}: {data['inserts_per_second']:.0f} inserts/s"
                    )
            print()

        # Buffer sizes
        if "buffer_sizes" in self.results:
            print("## Buffer Size Impact\n")
            for buffer_size, data in sorted(self.results["buffer_sizes"].items()):
                print(
                    f"  Buffer {buffer_size:>5}: {data['inserts_per_second']:.0f} inserts/s"
                )
            print()

        # Memory vs disk
        if "memory_vs_disk" in self.results:
            print("## In-Memory vs Disk Performance\n")
            data = self.results["memory_vs_disk"]
            print(f"  Disk:   {data['disk']['inserts_per_second']:.0f} inserts/s")
            print(f"  Memory: {data['memory']['inserts_per_second']:.0f} inserts/s")
            print(f"  Speedup: {data['speedup']:.2f}x")
            print()

        # Flush frequency
        if "flush_frequency" in self.results:
            print("## Flush Frequency Impact\n")
            for interval, data in sorted(self.results["flush_frequency"].items()):
                print(
                    f"  Every {interval:>5}: {data['inserts_per_second']:.0f} inserts/s "
                    f"({data['flush_count']} flushes)"
                )
            print()

        print("=" * 60 + "\n")

    def save_results_json(self):
        """Save results to JSON file with timestamp."""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"database_profiling_results_{timestamp_str}.json"
        filepath = self.results_dir / filename

        # Prepare data for JSON serialization
        json_data = {"timestamp": self.timestamp.isoformat(), "results": self.results}

        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Results saved to JSON: {filepath}")
        return filepath

    def save_results_csv(self):
        """Save results to CSV files for easy analysis."""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        saved_files = []

        # Save insert patterns results
        if "insert_patterns" in self.results:
            filename = f"insert_patterns_{timestamp_str}.csv"
            filepath = self.results_dir / filename

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "record_count",
                        "batch_size",
                        "total_time_ms",
                        "inserts_per_second",
                        "time_per_insert_us",
                    ]
                )

                for num_records, batch_results in self.results[
                    "insert_patterns"
                ].items():
                    for batch_size, data in batch_results.items():
                        writer.writerow(
                            [
                                num_records,
                                batch_size,
                                data["total_time"] * 1000,  # Convert to ms
                                data["inserts_per_second"],
                                data["time_per_insert"]
                                * 1000000,  # Convert to microseconds
                            ]
                        )

            saved_files.append(filepath)
            print(f"Insert patterns saved to CSV: {filepath}")

        # Save buffer sizes results
        if "buffer_sizes" in self.results:
            filename = f"buffer_sizes_{timestamp_str}.csv"
            filepath = self.results_dir / filename

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "buffer_size",
                        "total_time_ms",
                        "inserts_per_second",
                        "time_per_insert_us",
                    ]
                )

                for buffer_size, data in self.results["buffer_sizes"].items():
                    writer.writerow(
                        [
                            buffer_size,
                            data["total_time"] * 1000,  # Convert to ms
                            data["inserts_per_second"],
                            data["time_per_insert"]
                            * 1000000,  # Convert to microseconds
                        ]
                    )

            saved_files.append(filepath)
            print(f"Buffer sizes saved to CSV: {filepath}")

        # Save memory vs disk results
        if "memory_vs_disk" in self.results:
            filename = f"memory_vs_disk_{timestamp_str}.csv"
            filepath = self.results_dir / filename

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["database_type", "total_time_ms", "inserts_per_second", "speedup"]
                )

                data = self.results["memory_vs_disk"]
                writer.writerow(
                    [
                        "disk",
                        data["disk"]["total_time"] * 1000,
                        data["disk"]["inserts_per_second"],
                        1.0,
                    ]
                )
                writer.writerow(
                    [
                        "memory",
                        data["memory"]["total_time"] * 1000,
                        data["memory"]["inserts_per_second"],
                        data["speedup"],
                    ]
                )

            saved_files.append(filepath)
            print(f"Memory vs disk saved to CSV: {filepath}")

        # Save flush frequency results
        if "flush_frequency" in self.results:
            filename = f"flush_frequency_{timestamp_str}.csv"
            filepath = self.results_dir / filename

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "flush_interval",
                        "total_time_ms",
                        "flush_count",
                        "inserts_per_second",
                        "time_per_flush_ms",
                    ]
                )

                for interval, data in self.results["flush_frequency"].items():
                    writer.writerow(
                        [
                            interval,
                            data["total_time"] * 1000,  # Convert to ms
                            data["flush_count"],
                            data["inserts_per_second"],
                            data["time_per_flush"] * 1000,  # Convert to ms
                        ]
                    )

            saved_files.append(filepath)
            print(f"Flush frequency saved to CSV: {filepath}")

        return saved_files

    def save_summary_report(self):
        """Save a human-readable summary report."""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"database_profiling_summary_{timestamp_str}.txt"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Database Logging Profiling Report\n")
            f.write(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            # Insert patterns
            if "insert_patterns" in self.results:
                f.write("## Insert Pattern Performance\n\n")
                for num_records, batch_results in sorted(
                    self.results["insert_patterns"].items()
                ):
                    f.write(f"  {num_records} records:\n")
                    for batch_size, data in sorted(batch_results.items()):
                        f.write(
                            f"    Batch {batch_size:>4}: {data['inserts_per_second']:.0f} inserts/s\n"
                        )
                f.write("\n")

            # Buffer sizes
            if "buffer_sizes" in self.results:
                f.write("## Buffer Size Impact\n\n")
                for buffer_size, data in sorted(self.results["buffer_sizes"].items()):
                    f.write(
                        f"  Buffer {buffer_size:>5}: {data['inserts_per_second']:.0f} inserts/s\n"
                    )
                f.write("\n")

            # Memory vs disk
            if "memory_vs_disk" in self.results:
                f.write("## In-Memory vs Disk Performance\n\n")
                data = self.results["memory_vs_disk"]
                f.write(
                    f"  Disk:   {data['disk']['inserts_per_second']:.0f} inserts/s\n"
                )
                f.write(
                    f"  Memory: {data['memory']['inserts_per_second']:.0f} inserts/s\n"
                )
                f.write(f"  Speedup: {data['speedup']:.2f}x\n\n")

            # Flush frequency
            if "flush_frequency" in self.results:
                f.write("## Flush Frequency Impact\n\n")
                for interval, data in sorted(self.results["flush_frequency"].items()):
                    f.write(
                        f"  Every {interval:>5}: {data['inserts_per_second']:.0f} inserts/s "
                        f"({data['flush_count']} flushes)\n"
                    )
                f.write("\n")

            f.write("=" * 60 + "\n")

        print(f"Summary report saved to: {filepath}")
        return filepath

    @staticmethod
    def load_and_compare_results(results_dir: str = None, num_recent: int = 5):
        """Load and compare recent profiling results."""
        if results_dir is None:
            results_dir = (
                Path(__file__).parent.parent.parent.parent
                / "benchmarks"
                / "results"
                / "database_profiling"
            )
        else:
            results_dir = Path(results_dir)

        if not results_dir.exists():
            print(f"Results directory does not exist: {results_dir}")
            return

        # Find all JSON result files
        json_files = list(results_dir.glob("database_profiling_results_*.json"))
        json_files.sort(
            key=lambda x: x.stat().st_mtime, reverse=True
        )  # Sort by modification time

        if not json_files:
            print("No profiling results found.")
            return

        # Load recent results
        recent_results = []
        for json_file in json_files[:num_recent]:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    recent_results.append(
                        {
                            "file": json_file.name,
                            "timestamp": data["timestamp"],
                            "results": data["results"],
                        }
                    )
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        if not recent_results:
            print("No valid results found.")
            return

        # Generate comparison report
        comparison_file = (
            results_dir
            / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(comparison_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Database Profiling Results Comparison\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Comparing {len(recent_results)} recent runs\n")
            f.write("=" * 80 + "\n\n")

            # Compare buffer sizes performance
            if all("buffer_sizes" in r["results"] for r in recent_results):
                f.write("## Buffer Size Performance Comparison\n\n")
                f.write(
                    "Run Date                    | Buffer 1000 | Buffer 500  | Buffer 100\n"
                )
                f.write("-" * 80 + "\n")

                for result in recent_results:
                    timestamp = datetime.fromisoformat(result["timestamp"]).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    buffer_1000 = (
                        result["results"]["buffer_sizes"]
                        .get(1000, {})
                        .get("inserts_per_second", 0)
                    )
                    buffer_500 = (
                        result["results"]["buffer_sizes"]
                        .get(500, {})
                        .get("inserts_per_second", 0)
                    )
                    buffer_100 = (
                        result["results"]["buffer_sizes"]
                        .get(100, {})
                        .get("inserts_per_second", 0)
                    )

                    f.write(
                        f"{timestamp:<28} | {buffer_1000:>10.0f} | {buffer_500:>10.0f} | {buffer_100:>10.0f}\n"
                    )
                f.write("\n")

            # Compare memory vs disk performance
            if all("memory_vs_disk" in r["results"] for r in recent_results):
                f.write("## Memory vs Disk Performance Comparison\n\n")
                f.write(
                    "Run Date                    | Disk (ins/s) | Memory (ins/s) | Speedup\n"
                )
                f.write("-" * 80 + "\n")

                for result in recent_results:
                    timestamp = datetime.fromisoformat(result["timestamp"]).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    disk_perf = result["results"]["memory_vs_disk"]["disk"][
                        "inserts_per_second"
                    ]
                    memory_perf = result["results"]["memory_vs_disk"]["memory"][
                        "inserts_per_second"
                    ]
                    speedup = result["results"]["memory_vs_disk"]["speedup"]

                    f.write(
                        f"{timestamp:<28} | {disk_perf:>11.0f} | {memory_perf:>13.0f} | {speedup:>7.2f}x\n"
                    )
                f.write("\n")

            f.write("=" * 80 + "\n")

        print(f"Comparison report saved to: {comparison_file}")
        return comparison_file

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # Some files might still be locked, try to remove individual files
                for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                        except PermissionError:
                            pass
                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except (PermissionError, OSError):
                            pass
                try:
                    os.rmdir(self.temp_dir)
                except (PermissionError, OSError):
                    pass


def main():
    """Run database profiling suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Database Logging Profiler")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare recent profiling results instead of running new profiling",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory to save/load results (default: benchmarks/results/database_profiling)",
    )
    parser.add_argument(
        "--num-recent",
        type=int,
        default=5,
        help="Number of recent results to compare (default: 5)",
    )

    args = parser.parse_args()

    if args.compare:
        # Compare existing results
        DatabaseProfiler.load_and_compare_results(args.results_dir, args.num_recent)
        return

    # Run new profiling
    profiler = DatabaseProfiler(args.results_dir)

    try:
        print("=" * 60)
        print("Database Logging Profiler - Phase 2")
        print("=" * 60)

        # Profile insert patterns
        profiler.profile_insert_patterns(
            record_counts=[100, 1000, 5000], batch_sizes=[1, 10, 100, 500]
        )

        # Profile buffer sizes
        profiler.profile_buffer_sizes(num_records=5000)

        # Compare memory vs disk
        profiler.profile_memory_vs_disk(num_records=5000)

        # Profile flush frequency
        profiler.profile_flush_frequency(
            num_records=5000, flush_intervals=[100, 500, 1000, 2500]
        )

        # Generate report
        profiler.generate_report()

        # Save results to files
        print("\nSaving results to files...")
        json_file = profiler.save_results_json()
        csv_files = profiler.save_results_csv()
        summary_file = profiler.save_summary_report()

        print("\nDatabase profiling complete!")
        print(f"  Results saved in profiler.results")
        print(f"  JSON file: {json_file}")
        print(f"  CSV files: {len(csv_files)} files")
        print(f"  Summary: {summary_file}")
        print(f"  Results directory: {profiler.results_dir}")

    finally:
        profiler.cleanup()


if __name__ == "__main__":
    main()
