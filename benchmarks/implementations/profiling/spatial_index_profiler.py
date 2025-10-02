#!/usr/bin/env python3
"""
Spatial Index Profiler - Phase 2 Component-Level Profiling

Profiles spatial indexing operations to identify bottlenecks:
- KD-tree build time vs. dataset size
- Query performance (get_nearby, get_nearest, get_nearby_range)
- Batch update performance
- Dirty region tracking overhead
- Memory usage per index type
- Different index types comparison (KD-tree, Quadtree, Spatial Hash)

Usage:
    python -m benchmarks.implementations.profiling.spatial_index_profiler
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from farm.core.resources import Resource
from farm.core.spatial import SpatialIndex


class SpatialIndexProfiler:
    """Profile spatial indexing operations."""

    def __init__(self, width: float = 1000, height: float = 1000):
        self.width = width
        self.height = height
        self.results = {}

    def create_test_entities(self, count: int, entity_type: str = "agent") -> List:
        """Create test entities for profiling."""
        entities = []
        for i in range(count):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)

            if entity_type == "resource":
                entity = Resource(
                    resource_id=f"resource_{i}",
                    position=(x, y),
                    amount=np.random.uniform(1, 10),
                )
            else:
                # Simple agent-like object
                class MockAgent:
                    def __init__(self, agent_id, position):
                        self.agent_id = agent_id
                        self.position = position
                        self.alive = True

                entity = MockAgent(f"agent_{i}", (x, y))

            entities.append(entity)

        return entities

    def profile_index_build(self, entity_counts: List[int]):
        """Profile index build time for different entity counts."""
        print("\n" + "=" * 60)
        print("Profiling Index Build Time")
        print("=" * 60 + "\n")

        results = {}

        for count in entity_counts:
            print(f"Building index with {count} entities...")

            # Create test entities
            agents = self.create_test_entities(count, "agent")
            resources = self.create_test_entities(count // 2, "resource")

            # Create spatial index
            spatial_index = SpatialIndex(
                self.width, self.height, enable_batch_updates=False
            )
            spatial_index.set_references(agents, resources)

            # Profile update (builds the trees)
            start = time.perf_counter()
            spatial_index.update()
            build_time = time.perf_counter() - start

            results[count] = {
                "build_time": build_time,
                "agents": count,
                "resources": count // 2,
                "time_per_entity": build_time / (count + count // 2),
            }

            print(
                f"  Build time: {build_time*1000:.2f}ms ({build_time*1000000/(count+count//2):.2f}us per entity)"
            )

        self.results["build_time"] = results
        return results

    def profile_queries(self, entity_count: int, query_counts: List[int]):
        """Profile query performance."""
        print("\n" + "=" * 60)
        print(f"Profiling Queries (with {entity_count} entities)")
        print("=" * 60 + "\n")

        # Create test entities
        agents = self.create_test_entities(entity_count, "agent")
        resources = self.create_test_entities(entity_count // 2, "resource")

        # Create and build spatial index
        spatial_index = SpatialIndex(
            self.width, self.height, enable_batch_updates=False
        )
        spatial_index.set_references(agents, resources)
        spatial_index.update()

        results = {}

        # Test get_nearby queries
        print("Testing get_nearby queries...")
        for num_queries in query_counts:
            query_positions = [
                (np.random.uniform(0, self.width), np.random.uniform(0, self.height))
                for _ in range(num_queries)
            ]
            radius = 50.0

            start = time.perf_counter()
            for pos in query_positions:
                _ = spatial_index.get_nearby(pos, radius, ["agents", "resources"])
            query_time = time.perf_counter() - start

            results[f"nearby_{num_queries}"] = {
                "total_time": query_time,
                "per_query": (query_time / num_queries) if num_queries > 0 else 0,
                "queries_per_second": num_queries / query_time if query_time > 0 else 0,
            }

            us_per_query = (
                (query_time * 1000000 / num_queries) if num_queries > 0 else 0
            )
            qps = (num_queries / query_time) if query_time > 0 else 0
            print(
                f"  {num_queries} queries: {query_time*1000:.2f}ms "
                f"({us_per_query:.2f}us per query, "
                f"{qps:.0f} qps)"
            )

        # Test get_nearest queries
        print("\nTesting get_nearest queries...")
        for num_queries in query_counts:
            query_positions = [
                (np.random.uniform(0, self.width), np.random.uniform(0, self.height))
                for _ in range(num_queries)
            ]

            start = time.perf_counter()
            for pos in query_positions:
                _ = spatial_index.get_nearest(pos, ["agents"])
            query_time = time.perf_counter() - start

            results[f"nearest_{num_queries}"] = {
                "total_time": query_time,
                "per_query": (query_time / num_queries) if num_queries > 0 else 0,
                "queries_per_second": num_queries / query_time if query_time > 0 else 0,
            }

            us_per_query = (
                (query_time * 1000000 / num_queries) if num_queries > 0 else 0
            )
            qps = (num_queries / query_time) if query_time > 0 else 0
            print(
                f"  {num_queries} queries: {query_time*1000:.2f}ms "
                f"({us_per_query:.2f}us per query, "
                f"{qps:.0f} qps)"
            )

        self.results["queries"] = results
        return results

    def profile_batch_updates(self, entity_count: int, batch_sizes: List[int]):
        """Profile batch update performance."""
        print("\n" + "=" * 60)
        print(f"Profiling Batch Updates (with {entity_count} entities)")
        print("=" * 60 + "\n")

        results = {}

        for batch_size in batch_sizes:
            print(f"Testing batch size {batch_size}...")

            # Create test entities
            agents = self.create_test_entities(entity_count, "agent")
            resources = self.create_test_entities(entity_count // 2, "resource")

            # Create spatial index with batch updates
            spatial_index = SpatialIndex(
                self.width,
                self.height,
                enable_batch_updates=True,
                max_batch_size=batch_size,
                region_size=50.0,
            )
            spatial_index.set_references(agents, resources)
            spatial_index.update()

            # Move half the agents
            num_moves = entity_count // 2
            for i in range(num_moves):
                agents[i].position = (
                    np.random.uniform(0, self.width),
                    np.random.uniform(0, self.height),
                )

            # Profile batch update processing
            spatial_index.mark_positions_dirty()

            start = time.perf_counter()
            spatial_index.update()
            update_time = time.perf_counter() - start

            results[batch_size] = {
                "update_time": update_time,
                "entities_moved": num_moves,
                "time_per_move": update_time / num_moves if num_moves > 0 else 0,
            }

            us_per_move = (update_time * 1000000 / num_moves) if num_moves > 0 else 0
            print(
                f"  Update time: {update_time*1000:.2f}ms "
                f"({us_per_move:.2f}us per moved entity)"
            )

        self.results["batch_updates"] = results
        return results

    def profile_index_types(self, entity_count: int):
        """Compare performance of different index types."""
        print("\n" + "=" * 60)
        print(f"Comparing Index Types (with {entity_count} entities)")
        print("=" * 60 + "\n")

        results = {}

        # Create test entities once
        agents = self.create_test_entities(entity_count, "agent")
        resources = self.create_test_entities(entity_count // 2, "resource")

        # Test different index configurations
        index_configs = [
            (
                "kdtree_only",
                {
                    "agents": {"index_type": "kdtree"},
                    "resources": {"index_type": "kdtree"},
                },
            ),
            (
                "quadtree_only",
                {
                    "agents": {"index_type": "quadtree"},
                    "resources": {"index_type": "quadtree"},
                },
            ),
            (
                "spatial_hash_only",
                {
                    "agents": {"index_type": "spatial_hash", "cell_size": 50.0},
                    "resources": {"index_type": "spatial_hash", "cell_size": 50.0},
                },
            ),
            (
                "mixed_indices",
                {
                    "agents": {"index_type": "kdtree"},
                    "resources": {"index_type": "quadtree"},
                },
            ),
        ]

        for name, index_config in index_configs:
            print(f"Testing {name}...")

            # Create spatial index with custom index configurations
            spatial_index = SpatialIndex(
                self.width,
                self.height,
                index_configs=index_config,
                enable_batch_updates=False,
            )

            spatial_index.set_references(agents, resources)

            # Profile build time
            start = time.perf_counter()
            spatial_index.update()
            build_time = time.perf_counter() - start

            # Profile query time
            query_positions = [
                (np.random.uniform(0, self.width), np.random.uniform(0, self.height))
                for _ in range(100)
            ]
            radius = 50.0

            start = time.perf_counter()
            for pos in query_positions:
                _ = spatial_index.get_nearby(pos, radius, ["agents"])
            query_time = time.perf_counter() - start

            # Profile get_nearest time
            start = time.perf_counter()
            for pos in query_positions:
                _ = spatial_index.get_nearest(pos, ["agents"])
            nearest_time = time.perf_counter() - start

            results[name] = {
                "build_time": build_time,
                "query_time_100": query_time,
                "nearest_time_100": nearest_time,
                "query_per_call": query_time / 100,
                "nearest_per_call": nearest_time / 100,
            }

            print(
                f"  Build: {build_time*1000:.2f}ms, "
                f"Query (100x): {query_time*1000:.2f}ms "
                f"({query_time*10:.2f}us per query), "
                f"Nearest (100x): {nearest_time*1000:.2f}ms "
                f"({nearest_time*10:.2f}us per call)"
            )

        self.results["index_types"] = results
        return results

    def generate_report(self):
        """Generate a summary report of profiling results."""
        print("\n" + "=" * 60)
        print("Spatial Index Profiling Report")
        print("=" * 60 + "\n")

        # Build time summary
        if "build_time" in self.results:
            print("## Build Time Scaling\n")
            for count, data in sorted(self.results["build_time"].items()):
                print(
                    f"  {count:>6} entities: {data['build_time']*1000:>8.2f}ms "
                    f"({data['time_per_entity']*1000000:>6.2f}us per entity)"
                )

        # Query performance summary
        if "queries" in self.results:
            print("\n## Query Performance\n")
            for query_type, data in sorted(self.results["queries"].items()):
                if "nearby" in query_type:
                    num = query_type.split("_")[1]
                    print(
                        f"  get_nearby ({num} queries): {data['per_query']*1000000:.2f}us per query, "
                        f"{data['queries_per_second']:.0f} qps"
                    )

        # Batch update summary
        if "batch_updates" in self.results:
            print("\n## Batch Update Performance\n")
            for batch_size, data in sorted(self.results["batch_updates"].items()):
                print(
                    f"  Batch size {batch_size:>4}: {data['update_time']*1000:.2f}ms "
                    f"({data['time_per_move']*1000000:.2f}us per entity)"
                )

        # Index type comparison
        if "index_types" in self.results:
            print("\n## Index Type Comparison\n")
            for name, data in self.results["index_types"].items():
                print(
                    f"  {name:>20}: Build {data['build_time']*1000:>7.2f}ms, "
                    f"Query {data['query_per_call']*1000000:>6.2f}us, "
                    f"Nearest {data['nearest_per_call']*1000000:>6.2f}us"
                )

        print("\n" + "=" * 60 + "\n")


def main():
    """Run spatial index profiling suite."""
    profiler = SpatialIndexProfiler(width=1000, height=1000)

    print("=" * 60)
    print("Spatial Index Profiler - Phase 2")
    print("=" * 60)

    # Profile index build with different entity counts
    profiler.profile_index_build(entity_counts=[100, 500, 1000, 2000, 5000])

    # Profile query performance
    profiler.profile_queries(entity_count=1000, query_counts=[10, 100, 1000])

    # Profile batch updates
    profiler.profile_batch_updates(entity_count=1000, batch_sizes=[10, 50, 100, 500])

    # Compare index types
    profiler.profile_index_types(entity_count=1000)

    # Generate report
    profiler.generate_report()

    print("+ Spatial index profiling complete!")
    print("  Results saved in profiler.results")


if __name__ == "__main__":
    main()
