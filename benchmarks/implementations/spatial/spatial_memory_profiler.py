"""
Spatial Indexing Memory Profiler and Optimization Analyzer

This tool provides detailed memory usage analysis for spatial indexing implementations,
including memory footprint tracking, garbage collection analysis, and optimization
recommendations.

Features:
- Detailed memory usage tracking during index construction and queries
- Memory leak detection and analysis
- Garbage collection impact measurement
- Memory efficiency comparisons between implementations
- Optimization recommendations based on memory patterns
"""

import gc
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import psutil
# Optional import: memory_profiler is not required at runtime
try:
    from memory_profiler import profile  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    profile = None  # type: ignore

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from farm.core.spatial import Quadtree, SpatialHashGrid, SpatialIndex


class MemorySnapshot:
    """Memory usage snapshot at a specific point in time."""

    def __init__(self, label: str):
        self.label = label
        self.timestamp = time.time()
        self.process = psutil.Process()
        self.memory_info = self.process.memory_info()
        self.memory_percent = self.process.memory_percent()
        self.num_fds = self.process.num_fds() if hasattr(self.process, "num_fds") else 0

        # Get detailed memory breakdown
        try:
            self.memory_maps = list(self.process.memory_maps())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            self.memory_maps = []

    def get_memory_mb(self) -> float:
        """Get memory usage in MB."""
        return self.memory_info.rss / 1024 / 1024

    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return (
            self.memory_info.peak_wss / 1024 / 1024
            if hasattr(self.memory_info, "peak_wss")
            else self.get_memory_mb()
        )

    def diff(self, other: "MemorySnapshot") -> Dict[str, float]:
        """Calculate memory difference with another snapshot."""
        return {
            "rss_diff_mb": (self.memory_info.rss - other.memory_info.rss) / 1024 / 1024,
            "vms_diff_mb": (self.memory_info.vms - other.memory_info.vms) / 1024 / 1024,
            "percent_diff": self.memory_percent - other.memory_percent,
            "fds_diff": self.num_fds - other.num_fds,
        }


class SpatialMemoryProfiler:
    """Memory profiler for spatial indexing implementations."""

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.tracemalloc_enabled = False

    def start_tracing(self):
        """Start memory tracing."""
        if not self.tracemalloc_enabled:
            tracemalloc.start()
            self.tracemalloc_enabled = True

    def stop_tracing(self):
        """Stop memory tracing and return statistics."""
        if self.tracemalloc_enabled:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.tracemalloc_enabled = False
            return {
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024,
            }
        return None

    def take_snapshot(self, label: str) -> MemorySnapshot:
        """Take a memory snapshot."""
        snapshot = MemorySnapshot(label)
        self.snapshots.append(snapshot)
        return snapshot

    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """Get memory usage timeline."""
        timeline = []
        for snapshot in self.snapshots:
            timeline.append(
                {
                    "label": snapshot.label,
                    "timestamp": snapshot.timestamp,
                    "memory_mb": snapshot.get_memory_mb(),
                    "memory_percent": snapshot.memory_percent,
                    "num_fds": snapshot.num_fds,
                }
            )
        return timeline

    def analyze_memory_growth(self) -> Dict[str, Any]:
        """Analyze memory growth patterns."""
        if len(self.snapshots) < 2:
            return {}

        analysis = {
            "total_growth_mb": 0,
            "peak_usage_mb": 0,
            "growth_rate_mb_per_sec": 0,
            "memory_leaks_detected": False,
            "gc_impact": {},
        }

        # Calculate total growth
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        analysis["total_growth_mb"] = (
            last_snapshot.get_memory_mb() - first_snapshot.get_memory_mb()
        )

        # Find peak usage
        analysis["peak_usage_mb"] = max(
            snapshot.get_memory_mb() for snapshot in self.snapshots
        )

        # Calculate growth rate
        time_diff = last_snapshot.timestamp - first_snapshot.timestamp
        if time_diff > 0:
            analysis["growth_rate_mb_per_sec"] = analysis["total_growth_mb"] / time_diff

        # Detect potential memory leaks (continuous growth without corresponding decreases)
        memory_values = [s.get_memory_mb() for s in self.snapshots]
        if len(memory_values) > 3:
            # Simple heuristic: if memory keeps growing without significant decreases
            growth_periods = 0
            for i in range(1, len(memory_values)):
                if memory_values[i] > memory_values[i - 1] * 1.05:  # 5% growth
                    growth_periods += 1

            analysis["memory_leaks_detected"] = (
                growth_periods > len(memory_values) * 0.7
            )

        return analysis


class MockEntity:
    """Mock entity for memory profiling."""

    def __init__(
        self, entity_id: str, position: Tuple[float, float], data_size: int = 100
    ):
        self.entity_id = entity_id
        self.position = position
        self.alive = True
        # Add some data to make entities more realistic
        self.data = np.random.random(data_size)  # configurable number of random floats


class SpatialMemoryBenchmark:
    """Memory benchmark for spatial indexing implementations."""

    def __init__(self):
        self.profiler = SpatialMemoryProfiler()

    def profile_agentfarm_spatial_index(
        self, entities: List[MockEntity], index_type: str = "kdtree"
    ) -> Dict[str, Any]:
        """Profile memory usage of AgentFarm SpatialIndex."""
        results = {
            "index_type": index_type,
            "entity_count": len(entities),
            "memory_snapshots": [],
            "memory_analysis": {},
            "gc_stats": {},
        }

        # Start memory tracing
        self.profiler.start_tracing()

        # Take initial snapshot
        initial_snapshot = self.profiler.take_snapshot("initial")

        # Create spatial index
        spatial_index = SpatialIndex(
            width=1000.0, height=1000.0, enable_batch_updates=True, max_batch_size=100
        )

        after_creation_snapshot = self.profiler.take_snapshot("after_creation")

        # Register index
        spatial_index.register_index(
            name="test_entities",
            data_reference=entities,
            position_getter=lambda e: e.position,
            filter_func=lambda e: e.alive,
            index_type=index_type,
            cell_size=20.0 if index_type == "spatial_hash" else None,
        )

        after_registration_snapshot = self.profiler.take_snapshot("after_registration")

        # Build index
        spatial_index.update()

        after_build_snapshot = self.profiler.take_snapshot("after_build")

        # Perform queries
        for i in range(100):
            x = np.random.uniform(0, 1000)
            y = np.random.uniform(0, 1000)
            radius = np.random.uniform(5, 50)
            nearby = spatial_index.get_nearby((x, y), radius, ["test_entities"])

            if i % 20 == 0:  # Take snapshot every 20 queries
                self.profiler.take_snapshot(f"after_query_{i}")

        after_queries_snapshot = self.profiler.take_snapshot("after_queries")

        # Test batch updates
        entities_to_update = entities[: len(entities) // 10]  # Update 10% of entities
        for entity in entities_to_update:
            old_pos = entity.position
            new_pos = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))
            spatial_index.add_position_update(entity, old_pos, new_pos, "test_entities")

        spatial_index.process_batch_updates(force=True)

        after_updates_snapshot = self.profiler.take_snapshot("after_updates")

        # Force garbage collection
        gc.collect()

        after_gc_snapshot = self.profiler.take_snapshot("after_gc")

        # Stop tracing and get results
        tracemalloc_stats = self.profiler.stop_tracing()

        # Analyze memory usage
        results["memory_snapshots"] = self.profiler.get_memory_timeline()
        results["memory_analysis"] = self.profiler.analyze_memory_growth()
        results["tracemalloc_stats"] = tracemalloc_stats

        # Calculate memory per entity
        build_memory = (
            after_build_snapshot.get_memory_mb()
            - after_registration_snapshot.get_memory_mb()
        )
        results["memory_per_entity_kb"] = (build_memory * 1024) / len(entities)

        # Calculate query memory overhead
        query_memory = (
            after_queries_snapshot.get_memory_mb()
            - after_build_snapshot.get_memory_mb()
        )
        results["query_memory_overhead_mb"] = query_memory

        # Calculate update memory overhead
        update_memory = (
            after_updates_snapshot.get_memory_mb()
            - after_queries_snapshot.get_memory_mb()
        )
        results["update_memory_overhead_mb"] = update_memory

        # GC impact
        gc_memory = (
            after_gc_snapshot.get_memory_mb() - after_updates_snapshot.get_memory_mb()
        )
        results["gc_impact_mb"] = gc_memory

        return results

    def profile_scipy_kdtree(self, entities: List[MockEntity]) -> Dict[str, Any]:
        """Profile memory usage of scipy KDTree."""
        results = {
            "index_type": "scipy_kdtree",
            "entity_count": len(entities),
            "memory_snapshots": [],
            "memory_analysis": {},
        }

        self.profiler.start_tracing()

        # Take initial snapshot
        initial_snapshot = self.profiler.take_snapshot("initial")

        # Extract positions
        positions = np.array([e.position for e in entities])

        after_positions_snapshot = self.profiler.take_snapshot("after_positions")

        # Build KDTree
        from scipy.spatial import cKDTree

        kdtree = cKDTree(positions)

        after_build_snapshot = self.profiler.take_snapshot("after_build")

        # Perform queries
        for i in range(100):
            x = np.random.uniform(0, 1000)
            y = np.random.uniform(0, 1000)
            radius = np.random.uniform(5, 50)
            indices = kdtree.query_ball_point((x, y), radius)

            if i % 20 == 0:
                self.profiler.take_snapshot(f"after_query_{i}")

        after_queries_snapshot = self.profiler.take_snapshot("after_queries")

        # Force garbage collection
        gc.collect()

        after_gc_snapshot = self.profiler.take_snapshot("after_gc")

        # Stop tracing
        tracemalloc_stats = self.profiler.stop_tracing()

        # Analyze memory usage
        results["memory_snapshots"] = self.profiler.get_memory_timeline()
        results["memory_analysis"] = self.profiler.analyze_memory_growth()
        results["tracemalloc_stats"] = tracemalloc_stats

        # Calculate memory per entity
        build_memory = (
            after_build_snapshot.get_memory_mb()
            - after_positions_snapshot.get_memory_mb()
        )
        results["memory_per_entity_kb"] = (build_memory * 1024) / len(entities)

        # Calculate query memory overhead
        query_memory = (
            after_queries_snapshot.get_memory_mb()
            - after_build_snapshot.get_memory_mb()
        )
        results["query_memory_overhead_mb"] = query_memory

        # GC impact
        gc_memory = (
            after_gc_snapshot.get_memory_mb() - after_queries_snapshot.get_memory_mb()
        )
        results["gc_impact_mb"] = gc_memory

        return results

    def profile_memory_scaling(self, data_size: int = 100) -> Dict[str, Any]:
        """Profile memory usage scaling with entity count.

        Args:
            data_size: Size of random data array for each entity (default: 100)
        """
        entity_counts = [100, 500, 1000, 2000, 5000]
        scaling_results = {
            "entity_counts": entity_counts,
            "agentfarm_kdtree": [],
            "agentfarm_quadtree": [],
            "agentfarm_spatial_hash": [],
            "scipy_kdtree": [],
        }

        for count in entity_counts:
            print(f"Profiling memory scaling with {count} entities...")

            # Generate entities
            entities = []
            for i in range(count):
                x = np.random.uniform(0, 1000)
                y = np.random.uniform(0, 1000)
                entities.append(MockEntity(f"entity_{i}", (x, y), data_size))

            # Profile each implementation
            implementations = [
                (
                    "agentfarm_kdtree",
                    lambda entities=entities: self.profile_agentfarm_spatial_index(entities, "kdtree"),
                ),
                (
                    "agentfarm_quadtree",
                    lambda entities=entities: self.profile_agentfarm_spatial_index(entities, "quadtree"),
                ),
                (
                    "agentfarm_spatial_hash",
                    lambda entities=entities: self.profile_agentfarm_spatial_index(
                        entities, "spatial_hash"
                    ),
                ),
                ("scipy_kdtree", lambda entities=entities: self.profile_scipy_kdtree(entities)),
            ]

            for impl_name, impl_func in implementations:
                # Reset profiler
                self.profiler = SpatialMemoryProfiler()

                # Run profiling
                result = impl_func()

                # Store key metrics
                scaling_results[impl_name].append(
                    {
                        "entity_count": count,
                        "memory_per_entity_kb": result["memory_per_entity_kb"],
                        "total_memory_mb": result["memory_snapshots"][-1]["memory_mb"],
                        "peak_memory_mb": result["memory_analysis"].get(
                            "peak_usage_mb", 0
                        ),
                        "gc_impact_mb": result["gc_impact_mb"],
                    }
                )

        return scaling_results

    def analyze_memory_efficiency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory efficiency and provide optimization recommendations."""
        analysis = {
            "efficiency_ranking": [],
            "optimization_recommendations": [],
            "memory_leak_analysis": {},
            "scaling_efficiency": {},
        }

        # Rank implementations by memory efficiency
        implementations = []
        for impl_name, impl_results in results.items():
            if impl_name == "entity_counts":
                continue

            if impl_results:
                # Calculate average memory per entity
                avg_memory_per_entity = np.mean(
                    [r["memory_per_entity_kb"] for r in impl_results]
                )
                implementations.append((impl_name, avg_memory_per_entity))

        # Sort by memory efficiency (lower is better)
        implementations.sort(key=lambda x: x[1])
        analysis["efficiency_ranking"] = implementations

        # Generate optimization recommendations
        if implementations:
            most_efficient = implementations[0]
            least_efficient = implementations[-1]

            analysis["optimization_recommendations"] = [
                f"Most memory-efficient: {most_efficient[0]} ({most_efficient[1]:.2f} KB/entity)",
                f"Least memory-efficient: {least_efficient[0]} ({least_efficient[1]:.2f} KB/entity)",
                f"Memory savings potential: {least_efficient[1] - most_efficient[1]:.2f} KB/entity",
            ]

            # Scaling analysis
            for impl_name, impl_results in results.items():
                if impl_name == "entity_counts":
                    continue

                if len(impl_results) > 1:
                    # Calculate scaling factor
                    first_memory = impl_results[0]["memory_per_entity_kb"]
                    last_memory = impl_results[-1]["memory_per_entity_kb"]
                    scaling_factor = (
                        last_memory / first_memory if first_memory > 0 else 1
                    )

                    analysis["scaling_efficiency"][impl_name] = {
                        "scaling_factor": scaling_factor,
                        "is_linear": abs(scaling_factor - 1.0) < 0.1,
                        "memory_growth_rate": (last_memory - first_memory)
                        / len(impl_results),
                    }

        return analysis

    def generate_memory_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive memory analysis report."""
        report = []
        report.append("# Spatial Indexing Memory Analysis Report")
        report.append("=" * 60)
        report.append("")

        # Executive summary
        report.append("## Executive Summary")
        report.append("")

        analysis = self.analyze_memory_efficiency(results)

        if analysis["efficiency_ranking"]:
            most_efficient = analysis["efficiency_ranking"][0]
            report.append(
                f"**Most Memory-Efficient Implementation**: {most_efficient[0]}"
            )
            report.append(f"- Memory usage: {most_efficient[1]:.2f} KB per entity")
            report.append("")

        # Memory efficiency ranking
        report.append("## Memory Efficiency Ranking")
        report.append("")
        report.append("| Rank | Implementation | Memory per Entity (KB) |")
        report.append("|------|----------------|------------------------|")

        for i, (impl_name, memory_per_entity) in enumerate(
            analysis["efficiency_ranking"], 1
        ):
            report.append(f"| {i} | {impl_name} | {memory_per_entity:.2f} |")

        report.append("")

        # Scaling analysis
        report.append("## Memory Scaling Analysis")
        report.append("")
        report.append(
            "| Implementation | Scaling Factor | Linear Scaling | Growth Rate (KB/step) |"
        )
        report.append(
            "|----------------|----------------|----------------|----------------------|"
        )

        for impl_name, scaling_data in analysis["scaling_efficiency"].items():
            linear = "✓" if scaling_data["is_linear"] else "✗"
            report.append(
                f"| {impl_name} | {scaling_data['scaling_factor']:.2f} | {linear} | {scaling_data['memory_growth_rate']:.2f} |"
            )

        report.append("")

        # Detailed memory breakdown
        report.append("## Detailed Memory Breakdown")
        report.append("")

        for impl_name, impl_results in results.items():
            if impl_name == "entity_counts":
                continue

            if impl_results:
                report.append(f"### {impl_name}")
                report.append("")
                report.append(
                    "| Entity Count | Memory per Entity (KB) | Total Memory (MB) | Peak Memory (MB) | GC Impact (MB) |"
                )
                report.append(
                    "|--------------|------------------------|-------------------|------------------|----------------|"
                )

                for result in impl_results:
                    report.append(
                        f"| {result['entity_count']} | {result['memory_per_entity_kb']:.2f} | {result['total_memory_mb']:.1f} | {result['peak_memory_mb']:.1f} | {result['gc_impact_mb']:.1f} |"
                    )

                report.append("")

        # Optimization recommendations
        report.append("## Optimization Recommendations")
        report.append("")

        for recommendation in analysis["optimization_recommendations"]:
            report.append(f"- {recommendation}")

        report.append("")

        # Memory usage patterns
        report.append("## Memory Usage Patterns")
        report.append("")
        report.append("### Key Insights:")
        report.append("")
        report.append(
            "1. **Memory per entity** varies significantly between implementations"
        )
        report.append(
            "2. **Scaling behavior** shows which implementations maintain efficiency at scale"
        )
        report.append(
            "3. **GC impact** indicates memory fragmentation and cleanup efficiency"
        )
        report.append(
            "4. **Peak memory usage** helps determine memory requirements for production"
        )
        report.append("")

        # Best practices
        report.append("### Memory Optimization Best Practices:")
        report.append("")
        report.append(
            "1. **Choose efficient implementations** for memory-constrained environments"
        )
        report.append("2. **Monitor memory scaling** to ensure linear growth patterns")
        report.append("3. **Use batch updates** to reduce memory fragmentation")
        report.append("4. **Implement periodic GC** for long-running simulations")
        report.append("5. **Profile memory usage** regularly to detect leaks early")
        report.append("")

        return "\n".join(report)


def main():
    """Run memory profiling benchmark."""
    print("Starting Spatial Indexing Memory Profiling")
    print("=" * 50)

    benchmark = SpatialMemoryBenchmark()

    # Run memory scaling analysis
    print("Running memory scaling analysis...")
    scaling_results = benchmark.profile_memory_scaling()

    # Analyze results
    print("Analyzing memory efficiency...")
    analysis = benchmark.analyze_memory_efficiency(scaling_results)

    # Generate report
    print("Generating memory analysis report...")
    report = benchmark.generate_memory_report(scaling_results)

    # Save results
    # Use relative path from the benchmarks directory
    results_dir = os.path.join(os.path.dirname(__file__), "../../results")
    os.makedirs(results_dir, exist_ok=True)

    # Save scaling results
    import json

    scaling_results_path = os.path.join(results_dir, "spatial_memory_scaling.json")
    with open(scaling_results_path, "w", encoding="utf-8") as f:
        json.dump(scaling_results, f, indent=2)

    # Save report
    report_path = os.path.join(results_dir, "spatial_memory_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nMemory profiling completed!")
    print(f"Results saved to: {scaling_results_path}")
    print(f"Report saved to: {report_path}")
    print("\n" + "=" * 50)
    print("MEMORY EFFICIENCY SUMMARY")
    print("=" * 50)

    # Print summary
    if analysis["efficiency_ranking"]:
        print("\nMemory Efficiency Ranking:")
        for i, (impl_name, memory_per_entity) in enumerate(
            analysis["efficiency_ranking"], 1
        ):
            print(f"{i}. {impl_name}: {memory_per_entity:.2f} KB/entity")

    print("\nOptimization Recommendations:")
    for recommendation in analysis["optimization_recommendations"]:
        print(f"- {recommendation}")


if __name__ == "__main__":
    main()
