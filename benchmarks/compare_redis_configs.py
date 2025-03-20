#!/usr/bin/env python3
"""
Script for comparing different Redis configurations for agent memory.

This script runs the Redis memory benchmark with different configurations
and compares the results, showing which configuration provides the best
performance for different types of operations.
"""

import argparse
import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from benchmarks.base.runner import BenchmarkRunner
from benchmarks.implementations.redis_memory_benchmark import RedisMemoryBenchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare Redis memory configurations")
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run for each configuration",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/redis_comparison",
        help="Directory to save results",
    )
    
    parser.add_argument(
        "--memory-entries",
        type=int,
        default=1000,
        help="Base number of memory entries per agent",
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plot of results",
    )
    
    parser.add_argument(
        "--agents",
        type=int,
        nargs="+",
        default=[1, 5, 10, 50, 100],
        help="List of agent counts to test",
    )
    
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 10, 50, 100, 500],
        help="List of batch sizes to test",
    )
    
    parser.add_argument(
        "--memory-limits",
        type=int,
        nargs="+",
        default=[100, 1000, 5000, 10000],
        help="List of memory limits to test",
    )
    
    return parser.parse_args()


def run_agent_count_comparison(
    runner: BenchmarkRunner,
    iterations: int,
    memory_entries: int,
    agent_counts: List[int]
) -> Dict[str, Any]:
    """Run benchmarks with different agent counts."""
    results = {}
    
    for agent_count in agent_counts:
        benchmark_name = f"redis_memory_agents_{agent_count}"
        
        benchmark = RedisMemoryBenchmark(
            name=benchmark_name,
            num_agents=agent_count,
            memory_entries=memory_entries,
        )
        
        runner.register_benchmark(benchmark)
        result = runner.run_benchmark(benchmark_name, iterations=iterations)
        results[benchmark_name] = result
    
    return results


def run_batch_size_comparison(
    runner: BenchmarkRunner,
    iterations: int,
    memory_entries: int,
    batch_sizes: List[int]
) -> Dict[str, Any]:
    """Run benchmarks with different batch sizes."""
    results = {}
    
    for batch_size in batch_sizes:
        benchmark_name = f"redis_memory_batch_{batch_size}"
        
        benchmark = RedisMemoryBenchmark(
            name=benchmark_name,
            batch_size=batch_size,
            memory_entries=memory_entries,
        )
        
        runner.register_benchmark(benchmark)
        result = runner.run_benchmark(benchmark_name, iterations=iterations)
        results[benchmark_name] = result
    
    return results


def run_memory_limit_comparison(
    runner: BenchmarkRunner,
    iterations: int,
    memory_entries: int,
    memory_limits: List[int]
) -> Dict[str, Any]:
    """Run benchmarks with different memory limits."""
    results = {}
    
    for memory_limit in memory_limits:
        benchmark_name = f"redis_memory_limit_{memory_limit}"
        
        benchmark = RedisMemoryBenchmark(
            name=benchmark_name,
            memory_limit=memory_limit,
            memory_entries=min(memory_entries, memory_limit * 2),  # Ensure we hit the limit
        )
        
        runner.register_benchmark(benchmark)
        result = runner.run_benchmark(benchmark_name, iterations=iterations)
        results[benchmark_name] = result
    
    return results


def plot_comparison_results(results: Dict[str, Any], output_dir: str, metric_key: str, title: str, ylabel: str):
    """Plot comparison results."""
    # Extract data
    names = []
    values = []
    
    for name, result in results.items():
        summary = result.get_summary()
        overall = summary.get("overall", {})
        if overall and metric_key in overall:
            names.append(name.split('_')[-1])  # Get just the variable part (e.g., agent count, batch size)
            values.append(overall[metric_key])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def main():
    """Run Redis configuration comparison."""
    args = parse_args()
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir=output_dir)
    
    # Run comparisons
    print("Running Redis configuration comparisons...")
    
    print("\n1. Agent Count Comparison:")
    agent_results = run_agent_count_comparison(
        runner, args.iterations, args.memory_entries, args.agents
    )
    
    print("\n2. Batch Size Comparison:")
    batch_results = run_batch_size_comparison(
        runner, args.iterations, args.memory_entries, args.batch_sizes
    )
    
    print("\n3. Memory Limit Comparison:")
    limit_results = run_memory_limit_comparison(
        runner, args.iterations, args.memory_entries, args.memory_limits
    )
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating comparison plots...")
        
        plot_comparison_results(
            agent_results, 
            output_dir, 
            "avg_write_rate", 
            "Write Rate by Agent Count", 
            "Operations per Second"
        )
        
        plot_comparison_results(
            batch_results, 
            output_dir, 
            "batch_throughput", 
            "Throughput by Batch Size", 
            "Operations per Second"
        )
        
        plot_comparison_results(
            limit_results, 
            output_dir, 
            "memory_efficiency", 
            "Memory Efficiency by Limit", 
            "Bytes per Entry"
        )
    
    # Create summary report
    summary = {
        "timestamp": timestamp,
        "agent_count_comparison": {},
        "batch_size_comparison": {},
        "memory_limit_comparison": {},
    }
    
    # Extract key metrics
    for name, result in agent_results.items():
        summary["agent_count_comparison"][name] = result.get_summary()["overall"]
    
    for name, result in batch_results.items():
        summary["batch_size_comparison"][name] = result.get_summary()["overall"]
    
    for name, result in limit_results.items():
        summary["memory_limit_comparison"][name] = result.get_summary()["overall"]
    
    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComparison complete. Results saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 