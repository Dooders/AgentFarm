#!/usr/bin/env python3
"""
Example script for running the pragma profile benchmark with visualization.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmarks.implementations.pragma_profile_benchmark import PragmaProfileBenchmark
from benchmarks.base.runner import BenchmarkRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run pragma profile benchmark with visualization")
    
    parser.add_argument(
        "--num-records",
        type=int,
        default=100000,
        help="Number of records for write tests",
    )
    
    parser.add_argument(
        "--db-size-mb",
        type=int,
        default=100,
        help="Database size in MB for read tests",
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
    
    return parser.parse_args()


def visualize_results(results):
    """
    Visualize benchmark results.
    
    Parameters
    ----------
    results : BenchmarkResults
        Benchmark results to visualize
    """
    try:
        # Get raw results from the last iteration
        if not results.iteration_results:
            print("No results to visualize")
            return
        
        # Get the raw results from the last iteration
        raw_data = results.iteration_results[-1]["results"]
        
        if not raw_data or not raw_data.get("raw_results"):
            print("No results to visualize")
            return
        
        raw_results = raw_data["raw_results"]
        normalized_results = raw_data["normalized_results"]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle("SQLite Pragma Profile Benchmark Results", fontsize=16)
        
        # Create regular subplots for the first three charts
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        
        # Create a polar subplot for the radar chart
        ax4 = fig.add_subplot(2, 2, 4, polar=True)
        
        # Profiles and workloads
        profiles = list(raw_results.keys())
        workloads = list(raw_results[profiles[0]].keys())
        
        # Colors for profiles
        default_colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
        colors = {
            "balanced": "blue",
            "performance": "green",
            "safety": "red",
            "memory": "purple"
        }
        
        # Ensure all profiles have a color
        for i, profile in enumerate(profiles):
            if profile not in colors:
                colors[profile] = default_colors[i % len(default_colors)]
        
        # Plot 1: Absolute performance by workload
        ax1.set_title("Absolute Performance by Workload")
        ax1.set_xlabel("Workload Type")
        ax1.set_ylabel("Duration (seconds)")
        
        # Prepare data for grouped bar chart
        x = np.arange(len(workloads))
        width = 0.8 / len(profiles)  # Dynamically calculate width based on number of profiles
        
        # Dynamically calculate offsets based on number of profiles
        half_width = (len(profiles) - 1) * width / 2
        offsets = [width * i - half_width for i in range(len(profiles))]
        
        for i, profile in enumerate(profiles):
            durations = [raw_results[profile][workload] for workload in workloads]
            ax1.bar(x + offsets[i], durations, width, label=profile, color=colors[profile])
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(workloads)
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Plot 2: Relative performance (speedup) by workload
        ax2.set_title("Relative Performance by Workload")
        ax2.set_xlabel("Workload Type")
        ax2.set_ylabel("Speedup Factor (relative to balanced)")
        
        # Get non-balanced profiles
        non_balanced_profiles = [p for p in profiles if p != "balanced"]
        
        # Dynamically calculate offsets for non-balanced profiles
        if non_balanced_profiles:
            width = 0.8 / len(non_balanced_profiles)
            half_width = (len(non_balanced_profiles) - 1) * width / 2
            non_balanced_offsets = [width * i - half_width for i in range(len(non_balanced_profiles))]
        
            for i, profile in enumerate(non_balanced_profiles):
                speedups = [normalized_results[workload][profile] for workload in workloads]
                ax2.bar(x + non_balanced_offsets[i], speedups, width, label=profile, color=colors[profile])
        
        # Add horizontal line at y=1 (baseline)
        ax2.axhline(y=1, color="blue", linestyle="-", label="balanced (baseline)")
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(workloads)
        ax2.legend()
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Plot 3: Performance by profile (grouped by workload)
        ax3.set_title("Performance by Profile")
        ax3.set_xlabel("Profile")
        ax3.set_ylabel("Duration (seconds)")
        
        # Prepare data for grouped bar chart
        x = np.arange(len(profiles))
        width = 0.8 / len(workloads)  # Dynamically calculate width based on number of workloads
        
        # Dynamically calculate offsets based on number of workloads
        half_width = (len(workloads) - 1) * width / 2
        workload_offsets = [width * i - half_width for i in range(len(workloads))]
        
        # Workload colors with defaults
        default_workload_colors = ["darkred", "darkgreen", "darkblue", "darkorange", "darkviolet", "darkcyan"]
        workload_colors = {
            "write_heavy": "darkred",
            "read_heavy": "darkgreen",
            "mixed": "darkblue"
        }
        
        # Ensure all workloads have a color
        for i, workload in enumerate(workloads):
            if workload not in workload_colors:
                workload_colors[workload] = default_workload_colors[i % len(default_workload_colors)]
        
        for i, workload in enumerate(workloads):
            durations = [raw_results[profile][workload] for profile in profiles]
            ax3.bar(x + workload_offsets[i], durations, width, label=workload, color=workload_colors[workload])
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(profiles)
        ax3.legend()
        ax3.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Plot 4: Radar chart of profile characteristics
        ax4.set_title("Profile Characteristics")
        
        # Define characteristics
        characteristics = ["Write Speed", "Read Speed", "Data Safety", "Memory Usage"]
        
        # Define profile scores (subjective based on documentation)
        profile_scores = {
            "balanced": [3, 4, 4, 3],
            "performance": [5, 3, 1, 5],
            "safety": [1, 2, 5, 2],
            "memory": [3, 3, 2, 1]
        }
        
        # Number of characteristics
        N = len(characteristics)
        
        # Create angles for each characteristic
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Configure radar chart
        ax4.set_theta_offset(np.pi / 2)  # Rotate to start at top
        ax4.set_theta_direction(-1)  # Go clockwise
        ax4.set_rlabel_position(0)  # Move radial labels away from plotted line
        
        # Set limits for the radar chart (0 to 5)
        ax4.set_ylim(0, 5)
        
        # Draw one axis per characteristic and add labels
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(characteristics)
        
        # Draw profiles
        for profile in profiles:
            # Use default scores if profile not in predefined scores
            if profile not in profile_scores:
                profile_scores[profile] = [3, 3, 3, 3]  # Default balanced scores
                
            values = profile_scores[profile]
            values += values[:1]  # Close the loop
            ax4.plot(angles, values, linewidth=2, label=profile, color=colors[profile])
            ax4.fill(angles, values, alpha=0.1, color=colors[profile])
        
        ax4.legend(loc="upper right")
        
        # Add timestamp and parameters to the figure
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        params_text = (
            f"Parameters: {results.parameters}\n"
            f"Generated: {timestamp}"
        )
        fig.text(0.5, 0.01, params_text, ha="center", fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        output_dir = os.path.join(args.output, "figures")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"pragma_benchmark_{timestamp}.png")
        plt.savefig(filename, dpi=300)
        
        print(f"Visualization saved to {filename}")
        
        # Show figure
        plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run the pragma profile benchmark with visualization."""
    global args
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create benchmark
    benchmark = PragmaProfileBenchmark(
        num_records=args.num_records,
        db_size_mb=args.db_size_mb,
    )
    
    # Create runner
    runner = BenchmarkRunner(output_dir=args.output)
    runner.register_benchmark(benchmark)
    
    # Run benchmark
    print(f"Running pragma profile benchmark with {args.iterations} iterations...")
    results = runner.run_benchmark("pragma_profile", iterations=args.iterations)
    
    # Print summary
    summary = results.get_summary()
    print("\nBenchmark Summary:")
    print("=================")
    print(f"Description: {summary['metadata']['description']}")
    print(f"Parameters: {summary['parameters']}")
    print(f"Iterations: {summary['iterations']}")
    print(f"Mean Duration: {summary['mean_duration']:.2f} seconds")
    
    # Check if we have valid results before visualizing
    if results.iteration_results and len(results.iteration_results) > 0:
        # Visualize results
        visualize_results(results)
    else:
        print("No valid benchmark results to visualize.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 