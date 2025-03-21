#!/usr/bin/env python3
"""
Simple self-contained Redis benchmarking tool.
"""

import argparse
import json
import os
import random
import statistics
import string
import sys
import time
from typing import Any, Dict, List, Optional

# Check for required dependencies
required_packages = ["redis", "matplotlib", "numpy", "pandas"]
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("ERROR: Required packages not installed:")
    print(f"Please install them with: pip install {' '.join(missing_packages)}")
    exit(1)

import redis

# Try to import our config module
try:
    from config import BENCHMARK_CONFIG, get_redis_config
except ImportError:
    # Fallback defaults if config.py is not available
    def get_redis_config(environment="DEFAULT"):
        return {"host": "localhost", "port": 6379, "decode_responses": True}

    BENCHMARK_CONFIG = {
        "memory_entries": 1000,
        "batch_size": 100,
        "iterations": 3,
        "output_dir": "results",
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Redis memory benchmarks")

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Redis server host (overrides config)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Redis server port (overrides config)",
    )

    parser.add_argument(
        "--redis-env",
        type=str,
        default="DEFAULT",
        choices=["DEFAULT", "DOCKER", "REMOTE"],
        help="Redis environment to use from config",
    )

    parser.add_argument(
        "--memory-entries",
        type=int,
        default=BENCHMARK_CONFIG["memory_entries"],
        help=f"Number of memory entries to benchmark (default: {BENCHMARK_CONFIG['memory_entries']})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BENCHMARK_CONFIG["batch_size"],
        help=f"Batch size for batch operations (default: {BENCHMARK_CONFIG['batch_size']})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="redis_benchmark_results.json",
        help="File to save results",
    )

    return parser.parse_args()


def check_redis_connection(redis_config: Dict[str, Any]) -> bool:
    """Check if Redis server is running."""
    try:
        client = redis.Redis(**redis_config, socket_connect_timeout=1)
        result = client.ping()
        print(
            f"✅ Redis server is running at {redis_config['host']}:{redis_config['port']} (Ping result: {result})"
        )
        # Debug: show connection info
        info = client.info()
        print(f"Redis version: {info.get('redis_version', 'unknown')}")
        print(f"Redis mode: {info.get('redis_mode', 'unknown')}")
        print(f"Clients connected: {info.get('connected_clients', 'unknown')}")
        return True
    except Exception as e:
        print("❌ Could not connect to Redis server:")
        print(f"   {str(e)}")
        print("\nPlease ensure Redis server is running")
        return False


def generate_test_data(num_entries: int) -> List[Dict[str, Any]]:
    """Generate random test data."""
    print(f"Generating {num_entries} test data entries...")

    test_data = []
    for i in range(num_entries):
        # Create a simulated state
        state = {
            "position": [random.uniform(0, 100), random.uniform(0, 100)],
            "orientation": random.uniform(0, 360),
            "energy": random.uniform(0, 100),
            "health": random.uniform(0, 100),
        }

        # Create simulated perception data
        perception = {
            "visible_agents": random.randint(0, 5),
            "visible_resources": random.randint(0, 10),
        }

        # Create test metadata
        metadata = {
            "test_id": "".join(random.choice(string.ascii_letters) for _ in range(10)),
            "importance": random.uniform(0, 1),
        }

        # Create test action and reward
        actions = [
            "move",
            "eat",
            "sleep",
            "attack",
            "defend",
            "gather",
            "build",
            "communicate",
        ]
        action = random.choice(actions)
        reward = random.uniform(-1, 1)

        test_data.append(
            {
                "step": i,
                "state": state,
                "action": action,
                "reward": reward,
                "perception": perception,
                "metadata": metadata,
                "priority": random.uniform(0, 1),
            }
        )

    return test_data


def benchmark_writes(
    client: redis.Redis, test_data: List[Dict[str, Any]], namespace: str
) -> Dict[str, Any]:
    """Benchmark memory write operations."""
    print("Benchmarking memory writes...")

    durations = []

    # Clear any existing data
    keys = client.keys(f"{namespace}:*")
    if keys:
        client.delete(*keys)

    # Warm-up
    for i in range(min(10, len(test_data))):
        data = test_data[i]
        memory_key = f"{namespace}:memory:{data['step']}"
        client.set(memory_key, json.dumps(data))

    # Benchmark
    start_time = time.time()
    for data in test_data:
        before = time.time()

        # Store data
        memory_key = f"{namespace}:memory:{data['step']}"
        timeline_key = f"{namespace}:timeline"

        client.zadd(timeline_key, {str(data["step"]): data["step"]})
        client.set(memory_key, json.dumps(data))

        after = time.time()
        durations.append(after - before)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    operations = len(test_data)
    operations_per_second = operations / total_time

    print(f"Write operations: {operations_per_second:.2f} operations/second")

    return {
        "total_operations": operations,
        "total_time": total_time,
        "operations_per_second": operations_per_second,
        "avg_operation_time": statistics.mean(durations),
        "min_operation_time": min(durations),
        "max_operation_time": max(durations),
        "std_operation_time": statistics.stdev(durations),
    }


def benchmark_reads(
    client: redis.Redis, test_data: List[Dict[str, Any]], namespace: str
) -> Dict[str, Any]:
    """Benchmark memory read operations."""
    print("Benchmarking memory reads...")

    durations = []

    # Create a random sample of steps to read
    steps_to_read = [data["step"] for data in random.sample(test_data, len(test_data))]

    # Benchmark
    start_time = time.time()
    for step in steps_to_read:
        before = time.time()
        memory_key = f"{namespace}:memory:{step}"
        data = client.get(memory_key)
        after = time.time()
        durations.append(after - before)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    operations = len(steps_to_read)
    operations_per_second = operations / total_time

    print(f"Read operations: {operations_per_second:.2f} operations/second")

    return {
        "total_operations": operations,
        "total_time": total_time,
        "operations_per_second": operations_per_second,
        "avg_operation_time": statistics.mean(durations),
        "min_operation_time": min(durations),
        "max_operation_time": max(durations),
        "std_operation_time": statistics.stdev(durations),
    }


def benchmark_batch_operations(
    client: redis.Redis,
    test_data: List[Dict[str, Any]],
    namespace: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Benchmark batch operations."""
    print("Benchmarking batch operations...")

    # Clear existing data
    keys = client.keys(f"{namespace}:*")
    if keys:
        client.delete(*keys)

    # Create batches
    batches = []
    for i in range(0, len(test_data), batch_size):
        batches.append(test_data[i : i + batch_size])

    durations = []

    # Benchmark
    start_time = time.time()
    for batch in batches:
        before = time.time()
        pipe = client.pipeline()

        for data in batch:
            # Prepare the data for pipeline
            timeline_key = f"{namespace}:timeline"
            memory_key = f"{namespace}:memory:{data['step']}"

            # Add to pipeline
            pipe.zadd(timeline_key, {str(data["step"]): data["step"]})
            pipe.set(memory_key, json.dumps(data))

        # Execute batch
        pipe.execute()
        after = time.time()
        durations.append(after - before)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    operations = sum(len(batch) for batch in batches)
    batches_per_second = len(batches) / total_time if total_time > 0 else 0
    operations_per_second = operations / total_time if total_time > 0 else 0

    print(f"Batch operations: {operations_per_second:.2f} operations/second")
    print(f"Batches: {batches_per_second:.2f} batches/second")

    # Calculate statistics if we have enough data points
    stats = {}
    if len(durations) > 0:
        stats["avg_batch_time"] = statistics.mean(durations)
        stats["min_batch_time"] = min(durations)
        stats["max_batch_time"] = max(durations)
        if len(durations) > 1:
            stats["std_batch_time"] = statistics.stdev(durations)
        else:
            stats["std_batch_time"] = 0

    return {
        "total_operations": operations,
        "total_batches": len(batches),
        "batch_size": batch_size,
        "total_time": total_time,
        "operations_per_second": operations_per_second,
        "batches_per_second": batches_per_second,
        **stats,
    }


def benchmark_memory_usage(
    client: redis.Redis, test_data: List[Dict[str, Any]], namespace: str
) -> Dict[str, Any]:
    """Benchmark memory usage."""
    print("Benchmarking memory usage...")

    # Clear existing data
    keys = client.keys(f"{namespace}:*")
    if keys:
        client.delete(*keys)

    # Get initial memory info
    initial_used_memory = int(client.info()["used_memory"])

    # Insert test data in batches and measure memory usage
    memory_samples = []

    batch_size = 100
    for i in range(0, len(test_data), batch_size):
        # Insert a batch of data
        batch = test_data[i : i + batch_size]
        for data in batch:
            memory_key = f"{namespace}:memory:{data['step']}"
            timeline_key = f"{namespace}:timeline"

            client.zadd(timeline_key, {str(data["step"]): data["step"]})
            client.set(memory_key, json.dumps(data))

        # Measure memory usage
        current_used_memory = int(client.info()["used_memory"])
        entries_count = i + len(batch)
        memory_per_entry = (current_used_memory - initial_used_memory) / max(
            1, entries_count
        )

        memory_samples.append(
            {
                "entries": entries_count,
                "total_memory": current_used_memory - initial_used_memory,
                "memory_per_entry": memory_per_entry,
            }
        )

    # Calculate final metrics
    final_used_memory = int(client.info()["used_memory"])
    total_memory_used = final_used_memory - initial_used_memory
    bytes_per_entry = total_memory_used / len(test_data)

    print(
        f"Memory usage: {total_memory_used} bytes total, {bytes_per_entry:.2f} bytes/entry"
    )

    return {
        "initial_memory": initial_used_memory,
        "final_memory": final_used_memory,
        "total_memory_used": total_memory_used,
        "entries": len(test_data),
        "bytes_per_entry": bytes_per_entry,
        "memory_samples": memory_samples,
        "total_time": 0
    }


def main():
    """Main entry point."""
    args = parse_args()

    # Get Redis configuration from config file, with command-line overrides
    redis_config = get_redis_config(args.redis_env)

    # Override config with command-line arguments if provided
    if args.host:
        redis_config["host"] = args.host
    if args.port:
        redis_config["port"] = args.port

    # Check Redis connection
    if not check_redis_connection(redis_config):
        print("Exiting due to Redis connection failure")
        return 1

    # Generate test data
    test_data = generate_test_data(args.memory_entries)

    # Create a unique namespace for this benchmark
    namespace = f"benchmark:{int(time.time())}"

    # Create Redis client
    client = redis.Redis(**redis_config)

    # Run benchmarks
    try:
        print("\nRunning benchmarks...")

        # Ensure namespace is empty
        if client.exists(namespace):
            client.delete(namespace)

        # Run benchmarks
        write_results = benchmark_writes(client, test_data, namespace)
        read_results = benchmark_reads(client, test_data, namespace)
        batch_results = benchmark_batch_operations(
            client, test_data, namespace, args.batch_size
        )
        memory_results = benchmark_memory_usage(client, test_data, namespace)

        # Clean up
        if client.exists(namespace):
            client.delete(namespace)

        # Print results
        print("\n=== Benchmark Results ===")
        print(f"Memory Entries: {args.memory_entries}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Redis Server: {redis_config['host']}:{redis_config['port']}")
        print("\nWrite Performance:")
        print(
            f"  - Operations per second: {write_results['operations_per_second']:.2f}"
        )
        print("\nRead Performance:")
        print(f"  - Operations per second: {read_results['operations_per_second']:.2f}")
        print("\nBatch Operations:")
        print(f"  - Throughput: {batch_results['operations_per_second']:.2f} ops/sec")
        print("\nMemory Usage:")
        print(f"  - Per entry: {memory_results['bytes_per_entry']:.2f} bytes")

        # Calculate overall metrics
        overall = {
            "writes_per_second": write_results["operations_per_second"],
            "reads_per_second": read_results["operations_per_second"],
            "batch_throughput": batch_results["operations_per_second"],
            "memory_per_entry": memory_results["bytes_per_entry"],
            "total_duration": (
                write_results["total_time"]
                + read_results["total_time"]
                + batch_results["total_time"]
                + memory_results["total_time"]
            ),
        }

        # Create results object
        results = {
            "metadata": {
                "timestamp": time.time(),
                "entries": args.memory_entries,
                "batch_size": args.batch_size,
                "redis_server": f"{redis_config['host']}:{redis_config['port']}",
                "redis_environment": args.redis_env,
            },
            "writes": write_results,
            "reads": read_results,
            "batch_operations": batch_results,
            "memory": memory_results,
            "overall": overall,
        }

        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Benchmark complete! Results saved to {args.output}")
        return 0

    finally:
        # Clean up, even on error
        try:
            if client.exists(namespace):
                client.delete(namespace)
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
