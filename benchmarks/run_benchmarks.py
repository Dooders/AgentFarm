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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmarks for AgentFarm")
    
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["memory_db", "pragma_profile", "all"],
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 