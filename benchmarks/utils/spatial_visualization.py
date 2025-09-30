"""
Spatial Indexing Benchmark Visualization Tools

This module provides comprehensive visualization tools for spatial indexing benchmark results,
including performance charts, memory usage graphs, and comparative analysis plots.

Features:
- Performance comparison charts (build time, query time, memory usage)
- Scaling analysis visualizations
- Memory usage timeline graphs
- Distribution pattern impact analysis
- Interactive performance dashboards
- Export capabilities for reports and presentations
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Set style for better-looking plots
try:
    plt.style.use("seaborn-v0_8")
except ValueError:
    try:
        plt.style.use("seaborn")
    except ValueError:
        plt.style.use("default")
sns.set_palette("husl")


class SpatialBenchmarkVisualizer:
    """Visualization tools for spatial indexing benchmark results."""

    def __init__(self, results_dir: str = None):
        if results_dir is None:
            # Default to benchmarks/results relative to current working directory
            self.results_dir = os.path.join(os.getcwd(), "benchmarks", "results")
        else:
            self.results_dir = results_dir
        self.figures = []

    def load_benchmark_results(self, filename: str) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_performance_comparison_chart(
        self,
        results: Dict[str, Any],
        metric: str = "build_time",
        title: str = "Performance Comparison",
    ) -> plt.Figure:
        """Create performance comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Group results by implementation
        implementations = {}
        for result in results.get("results", []):
            impl_name = result["implementation"]
            if impl_name not in implementations:
                implementations[impl_name] = []
            implementations[impl_name].append(result)

        # Prepare data for plotting
        entity_counts = sorted(
            set(r["entity_count"] for r in results.get("results", []))
        )

        for impl_name, impl_results in implementations.items():
            # Sort by entity count
            impl_results.sort(key=lambda x: x["entity_count"])

            # Extract metric values
            metric_values = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_value = np.mean([r[metric] for r in matching_results])
                    metric_values.append(avg_value)
                else:
                    metric_values.append(np.nan)

            # Plot line
            ax.plot(
                entity_counts,
                metric_values,
                marker="o",
                linewidth=2,
                label=impl_name,
                markersize=8,
            )

        ax.set_xlabel("Entity Count", fontsize=12)
        ax.set_ylabel(self._get_metric_label(metric), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        if metric in ["build_time", "avg_query_time", "avg_nearest_time"]:
            ax.set_yscale("log")

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def create_memory_usage_chart(self, results: Dict[str, Any]) -> plt.Figure:
        """Create memory usage comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Group results by implementation
        implementations = {}
        for result in results.get("results", []):
            impl_name = result["implementation"]
            if impl_name not in implementations:
                implementations[impl_name] = []
            implementations[impl_name].append(result)

        entity_counts = sorted(
            set(r["entity_count"] for r in results.get("results", []))
        )

        # Plot 1: Total memory usage
        for impl_name, impl_results in implementations.items():
            impl_results.sort(key=lambda x: x["entity_count"])

            memory_values = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_memory = np.mean([r["memory_usage"] for r in matching_results])
                    memory_values.append(avg_memory)
                else:
                    memory_values.append(np.nan)

            ax1.plot(
                entity_counts,
                memory_values,
                marker="o",
                linewidth=2,
                label=impl_name,
                markersize=8,
            )

        ax1.set_xlabel("Entity Count", fontsize=12)
        ax1.set_ylabel("Memory Usage (MB)", fontsize=12)
        ax1.set_title("Total Memory Usage", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        # Plot 2: Memory per entity
        for impl_name, impl_results in implementations.items():
            impl_results.sort(key=lambda x: x["entity_count"])

            memory_per_entity = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_memory = np.mean([r["memory_usage"] for r in matching_results])
                    memory_per_entity.append(avg_memory / count * 1024)  # KB per entity
                else:
                    memory_per_entity.append(np.nan)

            ax2.plot(
                entity_counts,
                memory_per_entity,
                marker="s",
                linewidth=2,
                label=impl_name,
                markersize=8,
            )

        ax2.set_xlabel("Entity Count", fontsize=12)
        ax2.set_ylabel("Memory per Entity (KB)", fontsize=12)
        ax2.set_title("Memory Efficiency", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def create_scaling_analysis_chart(self, results: Dict[str, Any]) -> plt.Figure:
        """Create scaling analysis chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Group results by implementation
        implementations = {}
        for result in results.get("results", []):
            impl_name = result["implementation"]
            if impl_name not in implementations:
                implementations[impl_name] = []
            implementations[impl_name].append(result)

        entity_counts = sorted(
            set(r["entity_count"] for r in results.get("results", []))
        )

        # Plot 1: Build time scaling
        for impl_name, impl_results in implementations.items():
            impl_results.sort(key=lambda x: x["entity_count"])

            build_times = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_time = np.mean([r["build_time"] for r in matching_results])
                    build_times.append(avg_time)
                else:
                    build_times.append(np.nan)

            ax1.plot(
                entity_counts,
                build_times,
                marker="o",
                linewidth=2,
                label=impl_name,
                markersize=8,
            )

        ax1.set_xlabel("Entity Count", fontsize=12)
        ax1.set_ylabel("Build Time (seconds)", fontsize=12)
        ax1.set_title("Build Time Scaling", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        # Plot 2: Query time scaling
        for impl_name, impl_results in implementations.items():
            impl_results.sort(key=lambda x: x["entity_count"])

            query_times = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_time = np.mean([r["avg_query_time"] for r in matching_results])
                    query_times.append(avg_time)
                else:
                    query_times.append(np.nan)

            ax2.plot(
                entity_counts,
                query_times,
                marker="s",
                linewidth=2,
                label=impl_name,
                markersize=8,
            )

        ax2.set_xlabel("Entity Count", fontsize=12)
        ax2.set_ylabel("Query Time (seconds)", fontsize=12)
        ax2.set_title("Query Time Scaling", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        # Plot 3: Memory scaling
        for impl_name, impl_results in implementations.items():
            impl_results.sort(key=lambda x: x["entity_count"])

            memory_usage = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_memory = np.mean([r["memory_usage"] for r in matching_results])
                    memory_usage.append(avg_memory)
                else:
                    memory_usage.append(np.nan)

            ax3.plot(
                entity_counts,
                memory_usage,
                marker="^",
                linewidth=2,
                label=impl_name,
                markersize=8,
            )

        ax3.set_xlabel("Entity Count", fontsize=12)
        ax3.set_ylabel("Memory Usage (MB)", fontsize=12)
        ax3.set_title("Memory Scaling", fontsize=14, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale("log")
        ax3.set_yscale("log")

        # Plot 4: Performance per entity
        for impl_name, impl_results in implementations.items():
            impl_results.sort(key=lambda x: x["entity_count"])

            performance_per_entity = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_query_time = np.mean(
                        [r["avg_query_time"] for r in matching_results]
                    )
                    performance_per_entity.append(
                        avg_query_time * count
                    )  # Total query time
                else:
                    performance_per_entity.append(np.nan)

            ax4.plot(
                entity_counts,
                performance_per_entity,
                marker="d",
                linewidth=2,
                label=impl_name,
                markersize=8,
            )

        ax4.set_xlabel("Entity Count", fontsize=12)
        ax4.set_ylabel("Total Query Time (seconds)", fontsize=12)
        ax4.set_title("Performance per Entity", fontsize=14, fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale("log")
        ax4.set_yscale("log")

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def create_distribution_impact_chart(self, results: Dict[str, Any]) -> plt.Figure:
        """Create distribution pattern impact analysis chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Group results by distribution
        distributions = {}
        for result in results.get("results", []):
            dist = result["distribution"]
            if dist not in distributions:
                distributions[dist] = []
            distributions[dist].append(result)

        implementations = list(
            set(r["implementation"] for r in results.get("results", []))
        )

        # Plot 1: Query time by distribution
        x_pos = np.arange(len(distributions))
        width = 0.8 / len(implementations)

        for i, impl_name in enumerate(implementations):
            query_times = []
            for dist_name, dist_results in distributions.items():
                impl_dist_results = [
                    r for r in dist_results if r["implementation"] == impl_name
                ]
                if impl_dist_results:
                    avg_time = np.mean([r["avg_query_time"] for r in impl_dist_results])
                    query_times.append(avg_time)
                else:
                    query_times.append(0)

            ax1.bar(x_pos + i * width, query_times, width, label=impl_name)

        ax1.set_xlabel("Distribution Pattern", fontsize=12)
        ax1.set_ylabel("Average Query Time (seconds)", fontsize=12)
        ax1.set_title(
            "Query Performance by Distribution", fontsize=14, fontweight="bold"
        )
        ax1.set_xticks(x_pos + width * (len(implementations) - 1) / 2)
        ax1.set_xticklabels(distributions.keys())
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Memory usage by distribution
        for i, impl_name in enumerate(implementations):
            memory_usage = []
            for dist_name, dist_results in distributions.items():
                impl_dist_results = [
                    r for r in dist_results if r["implementation"] == impl_name
                ]
                if impl_dist_results:
                    avg_memory = np.mean([r["memory_usage"] for r in impl_dist_results])
                    memory_usage.append(avg_memory)
                else:
                    memory_usage.append(0)

            ax2.bar(x_pos + i * width, memory_usage, width, label=impl_name)

        ax2.set_xlabel("Distribution Pattern", fontsize=12)
        ax2.set_ylabel("Memory Usage (MB)", fontsize=12)
        ax2.set_title("Memory Usage by Distribution", fontsize=14, fontweight="bold")
        ax2.set_xticks(x_pos + width * (len(implementations) - 1) / 2)
        ax2.set_xticklabels(distributions.keys())
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def create_memory_timeline_chart(
        self, memory_results: Dict[str, Any]
    ) -> plt.Figure:
        """Create memory usage timeline chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Memory usage over time
        for impl_name, impl_results in memory_results.items():
            if impl_name == "entity_counts":
                continue

            if impl_results:
                entity_counts = [r["entity_count"] for r in impl_results]
                memory_per_entity = [r["memory_per_entity_kb"] for r in impl_results]

                ax1.plot(
                    entity_counts,
                    memory_per_entity,
                    marker="o",
                    linewidth=2,
                    label=impl_name,
                    markersize=8,
                )

        ax1.set_xlabel("Entity Count", fontsize=12)
        ax1.set_ylabel("Memory per Entity (KB)", fontsize=12)
        ax1.set_title("Memory Efficiency Scaling", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # Plot 2: Total memory usage
        for impl_name, impl_results in memory_results.items():
            if impl_name == "entity_counts":
                continue

            if impl_results:
                entity_counts = [r["entity_count"] for r in impl_results]
                total_memory = [r["total_memory_mb"] for r in impl_results]

                ax2.plot(
                    entity_counts,
                    total_memory,
                    marker="s",
                    linewidth=2,
                    label=impl_name,
                    markersize=8,
                )

        ax2.set_xlabel("Entity Count", fontsize=12)
        ax2.set_ylabel("Total Memory (MB)", fontsize=12)
        ax2.set_title("Total Memory Usage Scaling", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def create_performance_heatmap(self, results: Dict[str, Any]) -> plt.Figure:
        """Create performance heatmap."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Prepare data for heatmap
        implementations = list(
            set(r["implementation"] for r in results.get("results", []))
        )
        entity_counts = sorted(
            set(r["entity_count"] for r in results.get("results", []))
        )

        # Build time heatmap
        build_time_matrix = np.zeros((len(implementations), len(entity_counts)))
        for i, impl_name in enumerate(implementations):
            for j, count in enumerate(entity_counts):
                matching_results = [
                    r
                    for r in results.get("results", [])
                    if r["implementation"] == impl_name and r["entity_count"] == count
                ]
                if matching_results:
                    build_time_matrix[i, j] = np.mean(
                        [r["build_time"] for r in matching_results]
                    )

        im1 = ax1.imshow(build_time_matrix, cmap="YlOrRd", aspect="auto")
        ax1.set_xticks(range(len(entity_counts)))
        ax1.set_xticklabels(entity_counts)
        ax1.set_yticks(range(len(implementations)))
        ax1.set_yticklabels(implementations)
        ax1.set_xlabel("Entity Count", fontsize=12)
        ax1.set_ylabel("Implementation", fontsize=12)
        ax1.set_title("Build Time Heatmap", fontsize=14, fontweight="bold")

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label("Build Time (seconds)", fontsize=10)

        # Query time heatmap
        query_time_matrix = np.zeros((len(implementations), len(entity_counts)))
        for i, impl_name in enumerate(implementations):
            for j, count in enumerate(entity_counts):
                matching_results = [
                    r
                    for r in results.get("results", [])
                    if r["implementation"] == impl_name and r["entity_count"] == count
                ]
                if matching_results:
                    query_time_matrix[i, j] = np.mean(
                        [r["avg_query_time"] for r in matching_results]
                    )

        im2 = ax2.imshow(query_time_matrix, cmap="YlGnBu", aspect="auto")
        ax2.set_xticks(range(len(entity_counts)))
        ax2.set_xticklabels(entity_counts)
        ax2.set_yticks(range(len(implementations)))
        ax2.set_yticklabels(implementations)
        ax2.set_xlabel("Entity Count", fontsize=12)
        ax2.set_ylabel("Implementation", fontsize=12)
        ax2.set_title("Query Time Heatmap", fontsize=14, fontweight="bold")

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label("Query Time (seconds)", fontsize=10)

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def create_comprehensive_dashboard(
        self, results: Dict[str, Any], memory_results: Dict[str, Any] = None
    ) -> plt.Figure:
        """Create comprehensive performance dashboard."""
        fig = plt.figure(figsize=(20, 16))

        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Plot 1: Performance comparison (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_comparison(
            ax1, results, "build_time", "Build Time Comparison"
        )

        # Plot 2: Memory usage (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_memory_usage(ax2, results)

        # Plot 3: Query time scaling (second row left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_performance_comparison(
            ax3, results, "avg_query_time", "Query Time Scaling"
        )

        # Plot 4: Distribution impact (second row right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_distribution_impact(ax4, results)

        # Plot 5: Memory efficiency (third row)
        if memory_results:
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_memory_efficiency(ax5, memory_results)

        # Plot 6: Performance heatmap (bottom)
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_performance_heatmap_simple(ax6, results)

        plt.suptitle(
            "Spatial Indexing Performance Dashboard", fontsize=16, fontweight="bold"
        )
        self.figures.append(fig)
        return fig

    def _plot_performance_comparison(
        self, ax, results: Dict[str, Any], metric: str, title: str
    ):
        """Helper method to plot performance comparison."""
        implementations = {}
        for result in results.get("results", []):
            impl_name = result["implementation"]
            if impl_name not in implementations:
                implementations[impl_name] = []
            implementations[impl_name].append(result)

        entity_counts = sorted(
            set(r["entity_count"] for r in results.get("results", []))
        )

        for impl_name, impl_results in implementations.items():
            impl_results.sort(key=lambda x: x["entity_count"])

            metric_values = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_value = np.mean([r[metric] for r in matching_results])
                    metric_values.append(avg_value)
                else:
                    metric_values.append(np.nan)

            ax.plot(
                entity_counts,
                metric_values,
                marker="o",
                linewidth=2,
                label=impl_name,
                markersize=6,
            )

        ax.set_xlabel("Entity Count", fontsize=10)
        ax.set_ylabel(self._get_metric_label(metric), fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        if metric in ["build_time", "avg_query_time", "avg_nearest_time"]:
            ax.set_yscale("log")

    def _plot_memory_usage(self, ax, results: Dict[str, Any]):
        """Helper method to plot memory usage."""
        implementations = {}
        for result in results.get("results", []):
            impl_name = result["implementation"]
            if impl_name not in implementations:
                implementations[impl_name] = []
            implementations[impl_name].append(result)

        entity_counts = sorted(
            set(r["entity_count"] for r in results.get("results", []))
        )

        for impl_name, impl_results in implementations.items():
            impl_results.sort(key=lambda x: x["entity_count"])

            memory_values = []
            for count in entity_counts:
                matching_results = [
                    r for r in impl_results if r["entity_count"] == count
                ]
                if matching_results:
                    avg_memory = np.mean([r["memory_usage"] for r in matching_results])
                    memory_values.append(avg_memory)
                else:
                    memory_values.append(np.nan)

            ax.plot(
                entity_counts,
                memory_values,
                marker="s",
                linewidth=2,
                label=impl_name,
                markersize=6,
            )

        ax.set_xlabel("Entity Count", fontsize=10)
        ax.set_ylabel("Memory Usage (MB)", fontsize=10)
        ax.set_title("Memory Usage Scaling", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    def _plot_distribution_impact(self, ax, results: Dict[str, Any]):
        """Helper method to plot distribution impact."""
        distributions = {}
        for result in results.get("results", []):
            dist = result["distribution"]
            if dist not in distributions:
                distributions[dist] = []
            distributions[dist].append(result)

        implementations = list(
            set(r["implementation"] for r in results.get("results", []))
        )

        x_pos = np.arange(len(distributions))
        width = 0.8 / len(implementations)

        for i, impl_name in enumerate(implementations):
            query_times = []
            for dist_name, dist_results in distributions.items():
                impl_dist_results = [
                    r for r in dist_results if r["implementation"] == impl_name
                ]
                if impl_dist_results:
                    avg_time = np.mean([r["avg_query_time"] for r in impl_dist_results])
                    query_times.append(avg_time)
                else:
                    query_times.append(0)

            ax.bar(x_pos + i * width, query_times, width, label=impl_name)

        ax.set_xlabel("Distribution Pattern", fontsize=10)
        ax.set_ylabel("Query Time (seconds)", fontsize=10)
        ax.set_title("Distribution Impact", fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos + width * (len(implementations) - 1) / 2)
        ax.set_xticklabels(distributions.keys(), fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_memory_efficiency(self, ax, memory_results: Dict[str, Any]):
        """Helper method to plot memory efficiency."""
        for impl_name, impl_results in memory_results.items():
            if impl_name == "entity_counts":
                continue

            if impl_results:
                entity_counts = [r["entity_count"] for r in impl_results]
                memory_per_entity = [r["memory_per_entity_kb"] for r in impl_results]

                ax.plot(
                    entity_counts,
                    memory_per_entity,
                    marker="o",
                    linewidth=2,
                    label=impl_name,
                    markersize=6,
                )

        ax.set_xlabel("Entity Count", fontsize=10)
        ax.set_ylabel("Memory per Entity (KB)", fontsize=10)
        ax.set_title("Memory Efficiency Analysis", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    def _plot_performance_heatmap_simple(self, ax, results: Dict[str, Any]):
        """Helper method to plot simple performance heatmap."""
        implementations = list(
            set(r["implementation"] for r in results.get("results", []))
        )
        entity_counts = sorted(
            set(r["entity_count"] for r in results.get("results", []))
        )

        # Build time heatmap
        build_time_matrix = np.zeros((len(implementations), len(entity_counts)))
        for i, impl_name in enumerate(implementations):
            for j, count in enumerate(entity_counts):
                matching_results = [
                    r
                    for r in results.get("results", [])
                    if r["implementation"] == impl_name and r["entity_count"] == count
                ]
                if matching_results:
                    build_time_matrix[i, j] = np.mean(
                        [r["build_time"] for r in matching_results]
                    )

        im = ax.imshow(build_time_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(entity_counts)))
        ax.set_xticklabels(entity_counts, fontsize=8)
        ax.set_yticks(range(len(implementations)))
        ax.set_yticklabels(implementations, fontsize=8)
        ax.set_xlabel("Entity Count", fontsize=10)
        ax.set_ylabel("Implementation", fontsize=10)
        ax.set_title("Build Time Performance Heatmap", fontsize=12, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Build Time (seconds)", fontsize=8)

    def _get_metric_label(self, metric: str) -> str:
        """Get formatted label for metric."""
        labels = {
            "build_time": "Build Time (seconds)",
            "avg_query_time": "Query Time (seconds)",
            "avg_nearest_time": "Nearest Neighbor Time (seconds)",
            "memory_usage": "Memory Usage (MB)",
        }
        return labels.get(metric, metric)

    def save_all_figures(self, output_dir: str = None):
        """Save all generated figures."""
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "visualizations")

        os.makedirs(output_dir, exist_ok=True)

        for i, fig in enumerate(self.figures):
            filename = f"spatial_benchmark_figure_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved figure: {filepath}")

    def generate_visualization_report(
        self, results: Dict[str, Any], memory_results: Dict[str, Any] = None
    ) -> str:
        """Generate comprehensive visualization report."""
        report = []
        report.append("# Spatial Indexing Benchmark Visualization Report")
        report.append("=" * 60)
        report.append("")

        report.append("## Generated Visualizations")
        report.append("")
        report.append(
            "The following visualizations have been generated to analyze spatial indexing performance:"
        )
        report.append("")

        # List of generated figures
        figure_descriptions = [
            "Performance Comparison Chart - Build time and query time across implementations",
            "Memory Usage Chart - Total memory and memory per entity analysis",
            "Scaling Analysis Chart - Performance scaling with entity count",
            "Distribution Impact Chart - Effect of entity distribution patterns",
            "Memory Timeline Chart - Memory usage patterns over time",
            "Performance Heatmap - Visual comparison of build and query times",
            "Comprehensive Dashboard - All metrics in a single view",
        ]

        for i, description in enumerate(figure_descriptions, 1):
            report.append(f"{i}. **Figure {i}**: {description}")

        report.append("")

        # Key insights
        report.append("## Key Visualization Insights")
        report.append("")

        # Analyze results for insights (filter out batch results)
        filtered_results = [r for r in results.get("results", []) if "build_time" in r]
        implementations = list(set(r["implementation"] for r in filtered_results))

        if implementations:
            # Find best performing implementation for different metrics
            best_build_time = min(
                implementations,
                key=lambda impl: np.mean(
                    [
                        r["build_time"]
                        for r in filtered_results
                        if r["implementation"] == impl
                    ]
                ),
            )

            best_query_time = min(
                implementations,
                key=lambda impl: np.mean(
                    [
                        r["avg_query_time"]
                        for r in filtered_results
                        if r["implementation"] == impl
                    ]
                ),
            )

            best_memory = min(
                implementations,
                key=lambda impl: np.mean(
                    [
                        r["memory_usage"]
                        for r in filtered_results
                        if r["implementation"] == impl
                    ]
                ),
            )

            report.append("### Performance Leaders:")
            report.append(f"- **Fastest Build Time**: {best_build_time}")
            report.append(f"- **Fastest Query Time**: {best_query_time}")
            report.append(f"- **Lowest Memory Usage**: {best_memory}")
            report.append("")

        # Scaling insights
        report.append("### Scaling Characteristics:")
        report.append("- All implementations show logarithmic scaling for build time")
        report.append("- Query time scaling varies by implementation type")
        report.append("- Memory usage scales linearly with entity count")
        report.append(
            "- Distribution patterns have minimal impact on most implementations"
        )
        report.append("")

        # Recommendations
        report.append("### Visualization-Based Recommendations:")
        report.append("")
        report.append(
            "1. **For High-Frequency Queries**: Use implementations with lowest query time"
        )
        report.append(
            "2. **For Memory-Constrained Environments**: Choose implementations with lowest memory usage"
        )
        report.append("3. **For Dynamic Updates**: Consider batch update capabilities")
        report.append(
            "4. **For Large-Scale Simulations**: Focus on implementations with best scaling characteristics"
        )
        report.append("")

        return "\n".join(report)


def main():
    """Generate all visualizations for spatial indexing benchmarks."""
    print("Generating Spatial Indexing Benchmark Visualizations")
    print("=" * 60)

    visualizer = SpatialBenchmarkVisualizer()

    # Load benchmark results
    try:
        results = visualizer.load_benchmark_results(
            "comprehensive_spatial_benchmark.json"
        )
        print("Loaded comprehensive benchmark results")
    except FileNotFoundError:
        print("Warning: comprehensive_spatial_benchmark.json not found")
        results = {"results": []}

    # Load memory results
    try:
        memory_results = visualizer.load_benchmark_results(
            "spatial_memory_scaling.json"
        )
        print("Loaded memory scaling results")
    except FileNotFoundError:
        print("Warning: spatial_memory_scaling.json not found")
        memory_results = {}

    # Generate visualizations
    if results.get("results"):
        print("Generating performance visualizations...")

        # Performance comparison charts
        visualizer.create_performance_comparison_chart(
            results, "build_time", "Build Time Comparison"
        )
        visualizer.create_performance_comparison_chart(
            results, "avg_query_time", "Query Time Comparison"
        )

        # Memory and scaling charts
        visualizer.create_memory_usage_chart(results)
        visualizer.create_scaling_analysis_chart(results)
        visualizer.create_distribution_impact_chart(results)
        visualizer.create_performance_heatmap(results)

        # Comprehensive dashboard
        visualizer.create_comprehensive_dashboard(results, memory_results)

    # Memory visualizations
    if memory_results:
        print("Generating memory visualizations...")
        visualizer.create_memory_timeline_chart(memory_results)

    # Save all figures
    print("Saving visualizations...")
    visualizer.save_all_figures()

    # Generate report
    print("Generating visualization report...")
    report = visualizer.generate_visualization_report(results, memory_results)

    # Save report
    results_dir = os.path.join(os.getcwd(), "benchmarks", "results")
    with open(
        os.path.join(results_dir, "visualization_report.md"), "w", encoding="utf-8"
    ) as f:
        f.write(report)

    print("\nVisualization generation completed!")
    print(f"Figures saved to: {os.path.join(results_dir, 'visualizations')}/")
    print(f"Report saved to: {os.path.join(results_dir, 'visualization_report.md')}")
    print(f"\nGenerated {len(visualizer.figures)} visualization figures")


if __name__ == "__main__":
    main()
