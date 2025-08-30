"""
Redis-backed agent memory system benchmark implementation.
"""

import os
import random
import statistics
import string
import time
from typing import Any, Dict, List, Optional

import redis

from benchmarks.base.benchmark import Benchmark
from farm.core.perception import PerceptionData
from farm.core.state import AgentState
from farm.memory.redis_memory import AgentMemory, AgentMemoryManager, RedisMemoryConfig


class RedisMemoryBenchmark(Benchmark):
    """
    Benchmark for the Redis-backed agent memory system.

    This benchmark measures various performance aspects of the Redis-based agent
    memory system, including:
    - Memory write operations (iterations per second)
    - Memory read operations (iterations per second)
    - Memory search operations (iterations per second)
    - Batch operations throughput
    - Memory usage under different loads
    - Impact of memory limits and cleanup
    """

    def __init__(
        self,
        num_agents: int = 10,
        memory_entries: int = 1000,
        batch_size: int = 100,
        search_radius: float = 10.0,
        memory_limit: int = 5000,
        ttl: int = 3600,
        cleanup_interval: int = 100,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Redis memory benchmark.

        Parameters
        ----------
        num_agents : int
            Number of agents to simulate
        memory_entries : int
            Number of memory entries per agent
        batch_size : int
            Batch size for batch operations
        search_radius : float
            Radius for spatial searches
        memory_limit : int
            Memory limit per agent
        ttl : int
            Time-to-live for memory entries (seconds)
        cleanup_interval : int
            Cleanup interval for memory entries
        parameters : Dict[str, Any], optional
            Additional parameters for the benchmark
        """
        super().__init__(
            name="redis_memory",
            description="Benchmark for Redis-backed agent memory system",
            parameters=parameters or {},
        )

        # Set benchmark-specific parameters
        self.parameters.update(
            {
                "num_agents": num_agents,
                "memory_entries": memory_entries,
                "batch_size": batch_size,
                "search_radius": search_radius,
                "memory_limit": memory_limit,
                "ttl": ttl,
                "cleanup_interval": cleanup_interval,
            }
        )

        # Initialize benchmark-specific attributes
        self.memory_manager = None
        self.memories = []
        self.agent_ids = []
        self.redis_client = None
        self.test_data = []

    def setup(self) -> None:
        """
        Set up the benchmark environment.
        """
        # Create Redis configuration
        redis_config = RedisMemoryConfig(
            host="localhost",
            port=6379,
            db=0,
            memory_limit=self.parameters["memory_limit"],
            ttl=self.parameters["ttl"],
            cleanup_interval=self.parameters["cleanup_interval"],
            namespace=f"agent_memory_benchmark_{int(time.time())}",
        )

        # Initialize Redis client (force synchronous mode)
        redis_config.connection_params["decode_responses"] = True
        self.redis_client = redis.Redis(**redis_config.connection_params)

        try:
            # Test connection
            self.redis_client.ping()
        except redis.ConnectionError:
            print("WARNING: Could not connect to Redis server. Is it running?")
            print("Starting a local Redis server for benchmarking...")
            # For real implementation, you might want to handle this differently

        # Create memory manager
        self.memory_manager = AgentMemoryManager.get_instance(redis_config)

        # Create agent IDs and memories
        self.agent_ids = [f"agent_{i}" for i in range(self.parameters["num_agents"])]
        self.memories = [
            self.memory_manager.get_memory(agent_id) for agent_id in self.agent_ids
        ]

        # Generate test data
        self._generate_test_data()

    def _generate_test_data(self) -> None:
        """Generate test data for the benchmark."""
        num_entries = self.parameters["memory_entries"]

        self.test_data = []
        for step in range(num_entries):
            # Create a test state with all required parameters
            state = AgentState(
                agent_id=f"agent_{step}",
                step_number=step,
                position_x=random.uniform(0, 100),
                position_y=random.uniform(0, 100),
                position_z=random.uniform(0, 100),
                resource_level=random.uniform(0, 100),
                current_health=random.uniform(0, 100),
                is_defending=random.choice([True, False]),
                total_reward=random.uniform(0, 100),
                age=random.randint(0, 100),
            )

            # Create test perception data with numpy array
            import numpy as np

            perception_grid = np.zeros((11, 11), dtype=np.int8)  # 11x11 grid
            perception = PerceptionData(grid=perception_grid)

            # Create test metadata
            metadata = {
                "test_id": "".join(
                    random.choice(string.ascii_letters) for _ in range(10)
                ),
                "importance": random.uniform(0, 1),
            }

            # Create test action
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

            # Create test reward
            reward = random.uniform(-1, 1)

            self.test_data.append(
                {
                    "step": step,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "perception": perception,
                    "metadata": metadata,
                    "priority": random.uniform(0, 1),
                }
            )

    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark.

        Returns
        -------
        Dict[str, Any]
            Raw results from the benchmark run
        """
        results = {}

        # Run write benchmark
        results["write"] = self._benchmark_writes()

        # Run read benchmark
        results["read"] = self._benchmark_reads()

        # Run search benchmark
        results["search"] = self._benchmark_searches()

        # Run batch operations benchmark
        results["batch"] = self._benchmark_batch_operations()

        # Run memory usage benchmark
        results["memory"] = self._benchmark_memory_usage()

        # Run cleanup benchmark
        results["cleanup"] = self._benchmark_cleanup()

        # Calculate overall metrics
        results["overall"] = {
            "avg_write_rate": results["write"]["iterations_per_second"],
            "avg_read_rate": results["read"]["iterations_per_second"],
            "avg_search_rate": results["search"]["iterations_per_second"],
            "batch_throughput": results["batch"]["operations_per_second"],
            "memory_efficiency": results["memory"]["bytes_per_entry"],
            "cleanup_time": results["cleanup"]["cleanup_time"],
        }

        return results

    def _benchmark_writes(self) -> Dict[str, Any]:
        """Benchmark memory write operations."""
        print("Benchmarking memory writes...")

        durations = []
        memory = self.memories[0]

        # Warm-up
        for _ in range(min(10, len(self.test_data))):
            data = self.test_data[_]
            memory.remember_state(**data)

        # Benchmark
        start_time = time.time()
        for data in self.test_data:
            before = time.time()
            memory.remember_state(**data)
            after = time.time()
            durations.append(after - before)
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        operations = len(self.test_data)
        operations_per_second = operations / total_time

        # Calculate percentiles
        percentiles = {
            "p50": statistics.median(durations),
            "p90": statistics.quantiles(durations, n=10)[8],
            "p95": statistics.quantiles(durations, n=20)[18],
            "p99": (
                statistics.quantiles(durations, n=100)[98]
                if len(durations) >= 100
                else None
            ),
        }

        print(f"Write operations: {operations_per_second:.2f} operations/second")

        return {
            "total_operations": operations,
            "total_time": total_time,
            "iterations_per_second": operations_per_second,
            "avg_operation_time": statistics.mean(durations),
            "min_operation_time": min(durations),
            "max_operation_time": max(durations),
            "std_operation_time": statistics.stdev(durations),
            "percentiles": percentiles,
        }

    def _benchmark_reads(self) -> Dict[str, Any]:
        """Benchmark memory read operations."""
        print("Benchmarking memory reads...")

        # Clear memory from previous benchmarks
        self.memories[0].clear_memory()

        # Insert test data first
        for data in self.test_data:
            self.memories[0].remember_state(**data)

        durations = []
        memory = self.memories[0]

        # Warm-up
        for _ in range(min(10, len(self.test_data))):
            step = self.test_data[_]["step"]
            memory.retrieve_state(step)

        # Create a random sample of steps to read
        steps_to_read = [
            data["step"] for data in random.sample(self.test_data, len(self.test_data))
        ]

        # Benchmark
        start_time = time.time()
        for step in steps_to_read:
            before = time.time()
            memories = memory.retrieve_state(step)
            after = time.time()
            durations.append(after - before)
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        operations = len(steps_to_read)
        operations_per_second = operations / total_time

        # Calculate percentiles
        percentiles = {
            "p50": statistics.median(durations),
            "p90": statistics.quantiles(durations, n=10)[8],
            "p95": statistics.quantiles(durations, n=20)[18],
            "p99": (
                statistics.quantiles(durations, n=100)[98]
                if len(durations) >= 100
                else None
            ),
        }

        print(f"Read operations: {operations_per_second:.2f} operations/second")

        return {
            "total_operations": operations,
            "total_time": total_time,
            "iterations_per_second": operations_per_second,
            "avg_operation_time": statistics.mean(durations),
            "min_operation_time": min(durations),
            "max_operation_time": max(durations),
            "std_operation_time": statistics.stdev(durations),
            "percentiles": percentiles,
        }

    def _benchmark_searches(self) -> Dict[str, Any]:
        """Benchmark memory search operations."""
        print("Benchmarking memory searches...")

        durations = []
        memory = self.memories[0]
        search_radius = self.parameters["search_radius"]

        # Generate random positions for searching
        positions = [
            (random.uniform(0, 100), random.uniform(0, 100)) for _ in range(100)
        ]

        # Warm-up
        for _ in range(min(10, len(positions))):
            memory.search_by_position(positions[_], search_radius)

        # Benchmark
        start_time = time.time()
        for position in positions:
            before = time.time()
            results = memory.search_by_position(position, search_radius)
            after = time.time()
            durations.append(after - before)
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        operations = len(positions)
        operations_per_second = operations / total_time

        # Calculate percentiles
        percentiles = {
            "p50": statistics.median(durations),
            "p90": statistics.quantiles(durations, n=10)[8],
            "p95": statistics.quantiles(durations, n=20)[18],
            "p99": (
                statistics.quantiles(durations, n=100)[98]
                if len(durations) >= 100
                else None
            ),
        }

        print(f"Search operations: {operations_per_second:.2f} operations/second")

        return {
            "total_operations": operations,
            "total_time": total_time,
            "iterations_per_second": operations_per_second,
            "avg_operation_time": statistics.mean(durations),
            "min_operation_time": min(durations),
            "max_operation_time": max(durations),
            "std_operation_time": statistics.stdev(durations),
            "percentiles": percentiles,
        }

    def _benchmark_batch_operations(self) -> Dict[str, Any]:
        """Benchmark batch operations."""
        print("Benchmarking batch operations...")

        batch_size = self.parameters["batch_size"]
        memory = self.memories[0]
        memory.clear_memory()

        # Create batches
        batches = []
        for i in range(0, len(self.test_data), batch_size):
            batches.append(self.test_data[i : i + batch_size])

        durations = []

        # Benchmark
        start_time = time.time()
        for batch in batches:
            before = time.time()
            pipe = memory.redis_client.pipeline()

            for data in batch:
                # Prepare the data for pipeline
                timeline_key = f"{memory._agent_key_prefix}:timeline"
                memory_key = f"{memory._agent_key_prefix}:memory:{data['step']}"

                # Simplified version of remember_state for benchmarking
                memory_entry = {
                    "timestamp": time.time(),
                    "step": data["step"],
                    "state": vars(data["state"]),
                    "action": data["action"],
                    "reward": data["reward"],
                    "priority": data["priority"],
                }

                # Add to pipeline
                pipe.zadd(timeline_key, {str(data["step"]): data["step"]})
                pipe.set(memory_key, str(memory_entry))

            # Execute batch
            pipe.execute()
            after = time.time()
            durations.append(after - before)
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        operations = sum(len(batch) for batch in batches)
        batches_per_second = len(batches) / total_time
        operations_per_second = operations / total_time

        # Calculate percentiles
        percentiles = {
            "p50": statistics.median(durations),
            "p90": (
                statistics.quantiles(durations, n=10)[8]
                if len(durations) >= 10
                else None
            ),
            "p95": (
                statistics.quantiles(durations, n=20)[18]
                if len(durations) >= 20
                else None
            ),
            "p99": (
                statistics.quantiles(durations, n=100)[98]
                if len(durations) >= 100
                else None
            ),
        }

        print(f"Batch operations: {operations_per_second:.2f} operations/second")
        print(f"Batches: {batches_per_second:.2f} batches/second")

        return {
            "total_operations": operations,
            "total_batches": len(batches),
            "batch_size": batch_size,
            "total_time": total_time,
            "operations_per_second": operations_per_second,
            "batches_per_second": batches_per_second,
            "avg_batch_time": statistics.mean(durations),
            "min_batch_time": min(durations),
            "max_batch_time": max(durations),
            "std_batch_time": statistics.stdev(durations),
            "percentiles": percentiles,
        }

    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        print("Benchmarking memory usage...")

        memory = self.memories[0]
        memory.clear_memory()

        # Get initial memory info
        if self.redis_client is None:
            print("WARNING: Redis client is None, skipping memory usage benchmark")
            return {
                "initial_memory": 0,
                "final_memory": 0,
                "total_memory_used": 0,
                "entries": len(self.test_data),
                "bytes_per_entry": 0,
                "memory_samples": [],
            }

        initial_used_memory = int(self.redis_client.info()["used_memory"])  # type: ignore

        # Insert test data in batches and measure memory usage
        memory_samples = []

        batch_size = 100
        for i in range(0, len(self.test_data), batch_size):
            # Insert a batch of data
            batch = self.test_data[i : i + batch_size]
            for data in batch:
                memory.remember_state(**data)

            # Measure memory usage
            current_used_memory = int(self.redis_client.info()["used_memory"])  # type: ignore
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
        final_used_memory = int(self.redis_client.info()["used_memory"])  # type: ignore
        total_memory_used = final_used_memory - initial_used_memory
        bytes_per_entry = total_memory_used / len(self.test_data)

        print(
            f"Memory usage: {total_memory_used} bytes total, {bytes_per_entry:.2f} bytes/entry"
        )

        return {
            "initial_memory": initial_used_memory,
            "final_memory": final_used_memory,
            "total_memory_used": total_memory_used,
            "entries": len(self.test_data),
            "bytes_per_entry": bytes_per_entry,
            "memory_samples": memory_samples,
        }

    def _benchmark_cleanup(self) -> Dict[str, Any]:
        """Benchmark memory cleanup operations."""
        print("Benchmarking memory cleanup...")

        memory = self.memories[0]
        memory.clear_memory()

        # Insert test data
        for data in self.test_data:
            memory.remember_state(**data)

        # Force a cleanup
        before = time.time()
        memory._cleanup_old_memories()
        after = time.time()
        cleanup_time = after - before

        # Get memory usage after cleanup
        if self.redis_client is None:
            used_memory_after_cleanup = 0
            remaining_entries = 0
        else:
            memory_info = self.redis_client.info()
            used_memory_after_cleanup = int(memory_info["used_memory"])  # type: ignore

            # Count remaining entries
            timeline_key = f"{memory._agent_key_prefix}:timeline"
            remaining_entries = int(self.redis_client.zcard(timeline_key) or 0)  # type: ignore

        print(f"Cleanup time: {cleanup_time:.6f} seconds")
        print(f"Remaining entries: {remaining_entries} of {len(self.test_data)}")

        return {
            "cleanup_time": cleanup_time,
            "initial_entries": len(self.test_data),
            "remaining_entries": remaining_entries,
            "memory_after_cleanup": used_memory_after_cleanup,
            "entries_removed": len(self.test_data) - remaining_entries,
            "memory_limit": self.parameters["memory_limit"],
        }

    def cleanup(self) -> None:
        """
        Clean up after the benchmark.
        """
        # Clear all agent memories
        for memory in self.memories:
            memory.clear_memory()

        # Delete all benchmark-related keys
        if self.redis_client is not None:
            pattern = f"{self.memories[0].config.namespace}:*"
            keys = self.redis_client.keys(pattern)  # type: ignore
            if keys:
                self.redis_client.delete(*keys)  # type: ignore
            print(f"Cleaned up {len(keys) if keys else 0} Redis keys from benchmark")  # type: ignore
        else:
            print("Redis client is None, skipping cleanup")
