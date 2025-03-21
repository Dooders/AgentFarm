#!/usr/bin/env python3
"""
Script to run Redis memory benchmarks and compare different configurations.
"""

import argparse
import os
import sys
import time
import shutil
import subprocess
import json
from pathlib import Path

# Ensure the script can be run from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIRS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "benchmarks"
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Redis memory benchmarks")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "compare", "quick"],
        default="single",
        help="Benchmark mode: single (one config), compare (multiple configs), or quick (fast test)",
    )
    
    parser.add_argument(
        "--agents",
        type=int,
        default=10,
        help="Number of agents to simulate (for single mode)",
    )
    
    parser.add_argument(
        "--memory-entries",
        type=int,
        default=1000,
        help="Number of memory entries per agent",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for batch operations (for single mode)",
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
        default="benchmark_results",
        help="Directory to save results",
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (for compare mode)",
    )
    
    # Agent count range for compare mode
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[1, 5, 10, 50],
        help="List of agent counts to test (for compare mode)",
    )
    
    # Batch size range for compare mode
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 10, 50, 100, 500],
        help="List of batch sizes to test (for compare mode)",
    )
    
    # Check Redis server
    parser.add_argument(
        "--check-redis",
        action="store_true",
        help="Check if Redis server is running before starting",
    )
    
    return parser.parse_args()


def check_redis_server():
    """Check if Redis server is running."""
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        client.ping()
        print("✅ Redis server is running")
        return True
    except ImportError:
        print("❌ Redis package not installed.")
        print("   Please install it with: pip install redis")
        return False
    except Exception as e:
        print("❌ Could not connect to Redis server:")
        print(f"   {str(e)}")
        print("\nPlease ensure Redis server is running:")
        print("  - Windows: Start Redis server from the Redis installation")
        print("  - Linux/macOS: Run 'redis-server' in terminal")
        print("  - Docker: Run 'docker run --name redis -p 6379:6379 -d redis'")
        return False


def find_benchmark_script(script_name):
    """Find the path to a benchmark script."""
    for directory in BENCHMARK_DIRS:
        script_path = directory / script_name
        if script_path.exists():
            return script_path
    
    # Try looking for the module form
    for directory in BENCHMARK_DIRS:
        module_path = directory / "benchmarks" / script_name
        if module_path.exists():
            return module_path
    
    raise FileNotFoundError(f"Could not find benchmark script: {script_name}")


def run_single_benchmark(args):
    """Run a single benchmark with one configuration."""
    print("\n📊 Running single Redis memory benchmark...")
    
    try:
        # Create command with Python module approach
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.run_benchmarks",
            "--benchmark", "redis_memory",
            "--agents", str(args.agents),
            "--memory-entries", str(args.memory_entries),
            "--batch-size", str(args.batch_size),
            "--iterations", str(args.iterations),
            "--output", args.output
        ]
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, check=True)
        
        print("\n✅ Benchmark completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure you're running this script from the AgentFarm directory")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed with exit code {e.returncode}")
        return e.returncode
    
    return 0


def run_comparison_benchmark(args):
    """Run benchmarks with multiple configurations and compare results."""
    print("\n📊 Running Redis configuration comparison benchmark...")
    
    try:
        # Create command with Python module approach
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.compare_redis_configs",
            "--memory-entries", str(args.memory_entries),
            "--iterations", str(args.iterations),
            "--output", args.output,
            "--agents", *map(str, args.agent_counts),
            "--batch-sizes", *map(str, args.batch_sizes)
        ]
        
        if args.plot:
            cmd.append("--plot")
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, check=True)
        
        print("\n✅ Comparison benchmark completed successfully!")
        print(f"Results saved to: {args.output}_<timestamp>")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure you're running this script from the AgentFarm directory")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed with exit code {e.returncode}")
        return e.returncode
    
    return 0


def run_quick_benchmark(args):
    """Run a quick benchmark for fast feedback."""
    print("\n🚀 Running quick Redis memory benchmark...")
    
    # Override some parameters for faster execution
    args.iterations = 1
    args.memory_entries = min(args.memory_entries, 100)
    
    return run_single_benchmark(args)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print banner
    print("=" * 80)
    print("Redis Memory Benchmark Runner".center(80))
    print("=" * 80)
    
    # Check Redis server if requested
    if args.check_redis and not check_redis_server():
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Run the appropriate benchmark mode
    if args.mode == "single":
        return run_single_benchmark(args)
    elif args.mode == "compare":
        return run_comparison_benchmark(args)
    elif args.mode == "quick":
        return run_quick_benchmark(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 