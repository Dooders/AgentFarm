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

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from farm.database.database import SimulationDatabase, InMemorySimulationDatabase


class DatabaseProfiler:
    """Profile database logging operations."""

    def __init__(self):
        self.results = {}
        self.temp_dir = tempfile.mkdtemp()

    def profile_insert_patterns(self, record_counts: List[int], batch_sizes: List[int]):
        """Profile different insert patterns."""
        print("\n" + "="*60)
        print("Profiling Insert Patterns")
        print("="*60 + "\n")

        results = {}
        
        for num_records in record_counts:
            print(f"\nTesting with {num_records} records...")
            record_results = {}
            
            for batch_size in batch_sizes:
                db_path = os.path.join(self.temp_dir, f"test_{num_records}_{batch_size}.db")
                
                # Create database
                db = SimulationDatabase(db_path=db_path, simulation_id="test_sim")
                
                # Configure buffer size
                db.logger.agent_buffer_size = batch_size
                db.logger.step_buffer_size = batch_size
                
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
                    "inserts_per_second": num_records / insert_time if insert_time > 0 else 0,
                    "time_per_insert": insert_time / num_records if num_records > 0 else 0,
                }
                
                print(f"  Batch {batch_size:>4}: {insert_time*1000:.2f}ms, "
                      f"{num_records/insert_time:.0f} inserts/s, "
                      f"{insert_time*1000000/num_records:.2f}μs per insert")
                
                # Clean up
                db.close()
                os.remove(db_path)
            
            results[num_records] = record_results
        
        self.results["insert_patterns"] = results
        return results

    def profile_buffer_sizes(self, num_records: int):
        """Profile different buffer sizes."""
        print("\n" + "="*60)
        print(f"Profiling Buffer Sizes ({num_records} records)")
        print("="*60 + "\n")

        buffer_sizes = [10, 50, 100, 500, 1000, 5000]
        results = {}
        
        for buffer_size in buffer_sizes:
            db_path = os.path.join(self.temp_dir, f"test_buffer_{buffer_size}.db")
            
            # Create database
            db = SimulationDatabase(db_path=db_path, simulation_id="test_sim")
            db.logger.agent_buffer_size = buffer_size
            
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
            
            print(f"  Buffer {buffer_size:>5}: {total_time*1000:.2f}ms, "
                  f"{num_records/total_time:.0f} inserts/s")
            
            # Clean up
            db.close()
            os.remove(db_path)
        
        self.results["buffer_sizes"] = results
        return results

    def profile_memory_vs_disk(self, num_records: int):
        """Compare in-memory vs disk database performance."""
        print("\n" + "="*60)
        print(f"Comparing In-Memory vs Disk ({num_records} records)")
        print("="*60 + "\n")

        results = {}
        
        # Test disk database
        print("Testing disk database...")
        db_path = os.path.join(self.temp_dir, "test_disk.db")
        db_disk = SimulationDatabase(db_path=db_path, simulation_id="test_sim")
        
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
        
        print(f"  Disk: {disk_time*1000:.2f}ms, {num_records/disk_time:.0f} inserts/s")
        
        db_disk.close()
        os.remove(db_path)
        
        # Test in-memory database
        print("\nTesting in-memory database...")
        db_memory = InMemorySimulationDatabase(memory_limit_mb=100, simulation_id="test_sim")
        
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
        
        print(f"  Memory: {memory_time*1000:.2f}ms, {num_records/memory_time:.0f} inserts/s")
        
        # Calculate speedup
        speedup = disk_time / memory_time if memory_time > 0 else 0
        results["speedup"] = speedup
        
        print(f"\n  Speedup: {speedup:.2f}x (memory faster)")
        
        db_memory.close()
        
        self.results["memory_vs_disk"] = results
        return results

    def profile_flush_frequency(self, num_records: int, flush_intervals: List[int]):
        """Profile impact of different flush frequencies."""
        print("\n" + "="*60)
        print(f"Profiling Flush Frequency ({num_records} records)")
        print("="*60 + "\n")

        results = {}
        
        for flush_interval in flush_intervals:
            db_path = os.path.join(self.temp_dir, f"test_flush_{flush_interval}.db")
            db = SimulationDatabase(db_path=db_path, simulation_id="test_sim")
            
            # Very large buffer to control flush frequency
            db.logger.agent_buffer_size = 100000
            
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
            
            print(f"  Flush every {flush_interval:>5}: {total_time*1000:.2f}ms, "
                  f"{flush_count} flushes, {num_records/total_time:.0f} inserts/s")
            
            db.close()
            os.remove(db_path)
        
        self.results["flush_frequency"] = results
        return results

    def generate_report(self):
        """Generate a summary report of profiling results."""
        print("\n" + "="*60)
        print("Database Logging Profiling Report")
        print("="*60 + "\n")

        # Insert patterns
        if "insert_patterns" in self.results:
            print("## Insert Pattern Performance\n")
            for num_records, batch_results in sorted(self.results["insert_patterns"].items()):
                print(f"  {num_records} records:")
                for batch_size, data in sorted(batch_results.items()):
                    print(f"    Batch {batch_size:>4}: {data['inserts_per_second']:.0f} inserts/s")
            print()
        
        # Buffer sizes
        if "buffer_sizes" in self.results:
            print("## Buffer Size Impact\n")
            for buffer_size, data in sorted(self.results["buffer_sizes"].items()):
                print(f"  Buffer {buffer_size:>5}: {data['inserts_per_second']:.0f} inserts/s")
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
                print(f"  Every {interval:>5}: {data['inserts_per_second']:.0f} inserts/s "
                      f"({data['flush_count']} flushes)")
            print()
        
        print("="*60 + "\n")

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main():
    """Run database profiling suite."""
    profiler = DatabaseProfiler()
    
    try:
        print("="*60)
        print("Database Logging Profiler - Phase 2")
        print("="*60)
        
        # Profile insert patterns
        profiler.profile_insert_patterns(
            record_counts=[100, 1000, 5000],
            batch_sizes=[1, 10, 100, 500]
        )
        
        # Profile buffer sizes
        profiler.profile_buffer_sizes(num_records=5000)
        
        # Compare memory vs disk
        profiler.profile_memory_vs_disk(num_records=5000)
        
        # Profile flush frequency
        profiler.profile_flush_frequency(
            num_records=5000,
            flush_intervals=[100, 500, 1000, 2500]
        )
        
        # Generate report
        profiler.generate_report()
        
        print("✓ Database profiling complete!")
        print("  Results saved in profiler.results")
    
    finally:
        profiler.cleanup()


if __name__ == "__main__":
    main()
