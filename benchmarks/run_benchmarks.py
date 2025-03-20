#!/usr/bin/env python3
"""
Main script for running benchmarks.
"""

import argparse
import os
import sys
from typing import List, Dict, Any

from benchmarks.base.runner import BenchmarkRunner
from benchmarks.implementations.memory_db_benchmark import MemoryDBBenchmark
from benchmarks.implementations.pragma_profile_benchmark import PragmaProfileBenchmark
from benchmarks.implementations.redis_memory_benchmark import RedisMemoryBenchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmarks for AgentFarm")
    
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["memory_db", "pragma_profile", "redis_memory", "all"],
        default="all",
        help="Benchmark to run",
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps",
    )
    
    parser.add_argument(
        "--agents",
        type=int,
        default=30,
        help="Total number of agents",
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results",
        help="Directory to save results",
    )
    
    # Add pragma profile benchmark specific arguments
    parser.add_argument(
        "--num-records",
        type=int,
        default=100000,
        help="Number of records for pragma profile benchmark",
    )
    
    parser.add_argument(
        "--db-size-mb",
        type=int,
        default=100,
        help="Database size in MB for pragma profile benchmark",
    )
    
    # Add Redis memory benchmark specific arguments
    parser.add_argument(
        "--memory-entries",
        type=int,
        default=1000,
        help="Number of memory entries per agent for Redis benchmark",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for Redis benchmark batch operations",
    )
    
    parser.add_argument(
        "--search-radius",
        type=float,
        default=10.0,
        help="Search radius for Redis benchmark spatial searches",
    )
    
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=5000,
        help="Memory limit per agent for Redis benchmark",
    )
    
    parser.add_argument(
        "--ttl",
        type=int,
        default=3600,
        help="Time-to-live in seconds for Redis benchmark memory entries",
    )
    
    parser.add_argument(
        "--cleanup-interval",
        type=int,
        default=100,
        help="Cleanup interval for Redis benchmark memory entries",
    )
    
    return parser.parse_args()


def main():
    """Run benchmarks."""
    args = parse_args()
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir=args.output)
    
    # Register benchmarks
    if args.benchmark == "memory_db" or args.benchmark == "all":
        memory_db_benchmark = MemoryDBBenchmark(
            num_steps=args.steps,
            num_agents=args.agents,
        )
        runner.register_benchmark(memory_db_benchmark)
    
    if args.benchmark == "pragma_profile" or args.benchmark == "all":
        pragma_profile_benchmark = PragmaProfileBenchmark(
            num_records=args.num_records,
            db_size_mb=args.db_size_mb,
        )
        runner.register_benchmark(pragma_profile_benchmark)
    
    if args.benchmark == "redis_memory" or args.benchmark == "all":
        redis_memory_benchmark = RedisMemoryBenchmark(
            num_agents=args.agents,
            memory_entries=args.memory_entries,
            batch_size=args.batch_size,
            search_radius=args.search_radius,
            memory_limit=args.memory_limit,
            ttl=args.ttl,
            cleanup_interval=args.cleanup_interval,
        )
        runner.register_benchmark(redis_memory_benchmark)
    
    # Run benchmarks
    if args.benchmark == "all":
        results = runner.run_all_benchmarks(iterations=args.iterations)
    else:
        results = {args.benchmark: runner.run_benchmark(args.benchmark, iterations=args.iterations)}
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=================")
    
    for name, result in results.items():
        summary = result.get_summary()
        print(f"\nBenchmark: {name}")
        print(f"  Description: {summary['metadata']['description']}")
        print(f"  Parameters: {summary['parameters']}")
        print(f"  Iterations: {summary['iterations']}")
        print(f"  Mean Duration: {summary['mean_duration']:.2f} seconds")
        print(f"  Median Duration: {summary['median_duration']:.2f} seconds")
        print(f"  Standard Deviation: {summary['std_duration']:.2f} seconds")
        
        # Print additional metrics for Redis memory benchmark
        if name == "redis_memory":
            overall = summary.get("overall", {})
            if overall:
                print("\n  Performance Metrics:")
                print(f"    Write Operations: {overall.get('avg_write_rate', 0):.2f} ops/sec")
                print(f"    Read Operations: {overall.get('avg_read_rate', 0):.2f} ops/sec")
                print(f"    Search Operations: {overall.get('avg_search_rate', 0):.2f} ops/sec")
                print(f"    Batch Operations: {overall.get('batch_throughput', 0):.2f} ops/sec")
                print(f"    Memory Per Entry: {overall.get('memory_efficiency', 0):.2f} bytes")
                print(f"    Cleanup Time: {overall.get('cleanup_time', 0):.6f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 