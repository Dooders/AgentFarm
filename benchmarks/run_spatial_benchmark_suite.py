#!/usr/bin/env python3
"""
Master Script for Spatial Indexing Benchmark Suite

This script runs the complete spatial indexing benchmark suite, including:
- Comprehensive performance benchmarking
- Memory usage profiling
- Visualization generation
- Performance analysis and documentation

Usage:
    python run_spatial_benchmark_suite.py [--quick] [--memory-only] [--visualize-only]
"""

import argparse
import os
import sys
import time
from pathlib import Path

from benchmarks.implementations.spatial.comprehensive_spatial_benchmark import (
    SpatialBenchmark,
    SpatialBenchmarkConfig,
)
from benchmarks.implementations.spatial.spatial_memory_profiler import (
    SpatialMemoryBenchmark,
)
from benchmarks.implementations.spatial.spatial_performance_analyzer import (
    SpatialPerformanceAnalyzer,
)
from benchmarks.utils.spatial_visualization import SpatialBenchmarkVisualizer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_comprehensive_benchmark(quick_mode: bool = False):
    """Run comprehensive spatial indexing benchmark."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE SPATIAL INDEXING BENCHMARK")
    print("=" * 60)

    # Configure benchmark based on mode
    if quick_mode:
        config = SpatialBenchmarkConfig(
            entity_counts=[100, 500, 1000],  # Reduced for quick testing
            query_radii=[10.0, 20.0],  # Fewer radii
            distributions=["uniform", "clustered"],  # Fewer distributions
            test_iterations=2,  # Fewer iterations
            warmup_iterations=1,
        )
    else:
        config = SpatialBenchmarkConfig(
            entity_counts=[100, 500, 1000, 2000, 5000, 10000],
            query_radii=[5.0, 10.0, 20.0, 50.0, 100.0],
            distributions=["uniform", "clustered", "linear", "sparse"],
            test_iterations=5,
            warmup_iterations=2,
        )

    benchmark = SpatialBenchmark(config)
    results = benchmark.run_comprehensive_benchmark()

    # Create results directory relative to current working directory
    results_dir = os.path.join(os.getcwd(), "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save results
    import json

    with open(
        os.path.join(results_dir, "comprehensive_spatial_benchmark.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    # Generate and save report
    report = benchmark.generate_performance_report(results)
    with open(
        os.path.join(results_dir, "comprehensive_spatial_report.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print("Comprehensive benchmark completed")
    return results


def run_memory_profiling():
    """Run memory usage profiling benchmark."""
    print("=" * 60)
    print("RUNNING MEMORY USAGE PROFILING")
    print("=" * 60)

    benchmark = SpatialMemoryBenchmark()
    scaling_results = benchmark.profile_memory_scaling()

    # Save results
    import json

    results_dir = os.path.join(os.getcwd(), "benchmarks", "results")
    with open(
        os.path.join(results_dir, "spatial_memory_scaling.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(scaling_results, f, indent=2)

    # Generate and save report
    report = benchmark.generate_memory_report(scaling_results)
    with open(
        os.path.join(results_dir, "spatial_memory_report.md"), "w", encoding="utf-8"
    ) as f:
        f.write(report)

    print("Memory profiling completed")
    return scaling_results


def generate_visualizations(comprehensive_results=None, memory_results=None):
    """Generate visualization charts and reports."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    visualizer = SpatialBenchmarkVisualizer()

    # Load results if not provided
    if comprehensive_results is None:
        try:
            comprehensive_results = visualizer.load_benchmark_results(
                "comprehensive_spatial_benchmark.json"
            )
        except FileNotFoundError:
            print("Warning: comprehensive_spatial_benchmark.json not found")
            comprehensive_results = {"results": []}

    if memory_results is None:
        try:
            memory_results = visualizer.load_benchmark_results(
                "spatial_memory_scaling.json"
            )
        except FileNotFoundError:
            print("Warning: spatial_memory_scaling.json not found")
            memory_results = {}

    # Generate visualizations
    if comprehensive_results.get("results"):
        print("Generating performance visualizations...")

        # Filter out batch results for performance metrics
        filtered_results = {
            "results": [
                r for r in comprehensive_results["results"] if "build_time" in r
            ]
        }

        # Performance comparison charts
        visualizer.create_performance_comparison_chart(
            filtered_results, "build_time", "Build Time Comparison"
        )
        visualizer.create_performance_comparison_chart(
            filtered_results, "avg_query_time", "Query Time Comparison"
        )

        # Memory and scaling charts
        visualizer.create_memory_usage_chart(filtered_results)
        visualizer.create_scaling_analysis_chart(filtered_results)
        visualizer.create_distribution_impact_chart(filtered_results)
        visualizer.create_performance_heatmap(filtered_results)

        # Comprehensive dashboard
        visualizer.create_comprehensive_dashboard(filtered_results, memory_results)

    # Memory visualizations
    if memory_results:
        print("Generating memory visualizations...")
        visualizer.create_memory_timeline_chart(memory_results)

    # Save all figures
    visualizer.save_all_figures()

    # Generate visualization report
    report = visualizer.generate_visualization_report(
        comprehensive_results, memory_results
    )
    with open(
        os.path.join(os.getcwd(), "benchmarks", "results", "visualization_report.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print("Visualizations generated")
    return visualizer


def run_performance_analysis(comprehensive_results=None, memory_results=None):
    """Run comprehensive performance analysis."""
    print("=" * 60)
    print("RUNNING PERFORMANCE ANALYSIS")
    print("=" * 60)

    analyzer = SpatialPerformanceAnalyzer()

    # Load results if not provided
    if comprehensive_results is None:
        try:
            comprehensive_results = analyzer.load_benchmark_results(
                "comprehensive_spatial_benchmark.json"
            )
        except FileNotFoundError:
            print("Error: comprehensive_spatial_benchmark.json not found")
            return None

    if memory_results is None:
        try:
            memory_results = analyzer.load_benchmark_results(
                "spatial_memory_scaling.json"
            )
        except FileNotFoundError:
            print("Warning: spatial_memory_scaling.json not found")
            memory_results = None

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(
        comprehensive_results, memory_results
    )

    # Save report
    with open(
        os.path.join(
            os.getcwd(),
            "benchmarks",
            "results",
            "comprehensive_performance_analysis.md",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print("Performance analysis completed")
    return analyzer


def create_summary_report():
    """Create a summary report of all benchmark results."""
    print("=" * 60)
    print("CREATING SUMMARY REPORT")
    print("=" * 60)

    summary = []
    summary.append("# Spatial Indexing Benchmark Suite - Summary Report")
    summary.append("=" * 60)
    summary.append("")
    summary.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")

    # Check which results are available
    results_dir = Path(os.path.join(os.getcwd(), "benchmarks", "results"))
    available_files = list(results_dir.glob("*.json")) + list(results_dir.glob("*.md"))

    summary.append("## Available Results")
    summary.append("")

    if available_files:
        summary.append("| File | Description |")
        summary.append("|------|-------------|")

        file_descriptions = {
            "comprehensive_spatial_benchmark.json": "Comprehensive benchmark results (JSON)",
            "comprehensive_spatial_report.md": "Comprehensive benchmark report",
            "spatial_memory_scaling.json": "Memory scaling analysis results (JSON)",
            "spatial_memory_report.md": "Memory usage analysis report",
            "comprehensive_performance_analysis.md": "Complete performance analysis",
            "visualization_report.md": "Visualization analysis report",
        }

        for file_path in sorted(available_files):
            filename = file_path.name
            description = file_descriptions.get(filename, "Unknown file")
            summary.append(f"| {filename} | {description} |")
    else:
        summary.append("No results files found.")

    summary.append("")

    # Summary of key findings
    summary.append("## Key Findings Summary")
    summary.append("")

    try:
        # Try to load and summarize comprehensive results
        import json

        with open(
            os.path.join(
                os.getcwd(),
                "benchmarks",
                "results",
                "comprehensive_spatial_benchmark.json",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            results = json.load(f)

        if results.get("results"):
            implementations = list(set(r["implementation"] for r in results["results"]))
            summary.append(
                f"- **Total Implementations Tested**: {len(implementations)}"
            )
            summary.append(f"- **Total Test Scenarios**: {len(results['results'])}")
            summary.append("")

            # Find best performing implementations
            by_impl = {}
            for result in results["results"]:
                impl = result["implementation"]
                if impl not in by_impl:
                    by_impl[impl] = []
                by_impl[impl].append(result)

            # Best build time
            best_build = min(
                by_impl.keys(),
                key=lambda impl: sum(r["build_time"] for r in by_impl[impl])
                / len(by_impl[impl]),
            )
            summary.append(f"- **Fastest Build Time**: {best_build}")

            # Best query time
            best_query = min(
                by_impl.keys(),
                key=lambda impl: sum(r["avg_query_time"] for r in by_impl[impl])
                / len(by_impl[impl]),
            )
            summary.append(f"- **Fastest Query Time**: {best_query}")

            # Best memory usage
            best_memory = min(
                by_impl.keys(),
                key=lambda impl: sum(r["memory_usage"] for r in by_impl[impl])
                / len(by_impl[impl]),
            )
            summary.append(f"- **Lowest Memory Usage**: {best_memory}")

    except FileNotFoundError:
        summary.append("- Comprehensive benchmark results not available")

    summary.append("")

    # Recommendations
    summary.append("## Quick Recommendations")
    summary.append("")
    summary.append(
        "1. **For Real-Time Applications**: Use implementations with fastest query times"
    )
    summary.append(
        "2. **For Memory-Constrained Systems**: Choose implementations with lowest memory usage"
    )
    summary.append(
        "3. **For Dynamic Simulations**: Use AgentFarm implementations with batch update support"
    )
    summary.append(
        "4. **For Large-Scale Systems**: Focus on implementations with best scaling characteristics"
    )
    summary.append("")

    # Next steps
    summary.append("## Next Steps")
    summary.append("")
    summary.append("1. Review the detailed reports in the results directory")
    summary.append("2. Examine the visualization charts for performance patterns")
    summary.append("3. Use the optimization recommendations for your specific use case")
    summary.append(
        "4. Consider running additional benchmarks for your specific scenarios"
    )
    summary.append("")

    # Save summary
    summary_text = "\n".join(summary)
    with open(
        os.path.join(os.getcwd(), "benchmarks", "results", "benchmark_summary.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary_text)

    print("Summary report created")
    return summary_text


def main():
    """Main function to run the complete benchmark suite."""
    parser = argparse.ArgumentParser(description="Run spatial indexing benchmark suite")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (fewer tests)"
    )
    parser.add_argument(
        "--memory-only", action="store_true", help="Run only memory profiling"
    )
    parser.add_argument(
        "--visualize-only", action="store_true", help="Generate only visualizations"
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="Run only performance analysis"
    )

    args = parser.parse_args()

    # Create results directory
    os.makedirs(os.path.join(os.getcwd(), "benchmarks", "results"), exist_ok=True)
    os.makedirs(
        os.path.join(os.getcwd(), "benchmarks", "results", "visualizations"),
        exist_ok=True,
    )

    start_time = time.time()

    print("SPATIAL INDEXING BENCHMARK SUITE")
    print("=" * 60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print("")

    comprehensive_results = None
    memory_results = None

    try:
        # Run comprehensive benchmark
        if not args.memory_only and not args.visualize_only and not args.analyze_only:
            comprehensive_results = run_comprehensive_benchmark(args.quick)

        # Run memory profiling
        if not args.visualize_only and not args.analyze_only:
            memory_results = run_memory_profiling()

        # Generate visualizations
        if not args.analyze_only:
            generate_visualizations(comprehensive_results, memory_results)

        # Run performance analysis
        run_performance_analysis(comprehensive_results, memory_results)

        # Create summary report
        create_summary_report()

        # Calculate total time
        total_time = time.time() - start_time

        print("")
        print("=" * 60)
        print("BENCHMARK SUITE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total execution time: {total_time:.2f} seconds")
        print("Results saved to: " + os.path.join(os.getcwd(), "benchmarks", "results"))
        print("")
        print("Generated files:")
        print("- comprehensive_spatial_benchmark.json")
        print("- comprehensive_spatial_report.md")
        print("- spatial_memory_scaling.json")
        print("- spatial_memory_report.md")
        print("- comprehensive_performance_analysis.md")
        print("- visualization_report.md")
        print("- benchmark_summary.md")
        print("- visualizations/ (directory with charts)")
        print("")
        print("Review the benchmark_summary.md file for a quick overview of results.")

    except KeyboardInterrupt:
        print("\nBenchmark suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running benchmark suite: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
