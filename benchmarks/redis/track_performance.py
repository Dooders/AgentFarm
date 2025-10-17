#!/usr/bin/env python3
"""
Script to track Redis memory performance changes over time.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import datetime
import argparse
from typing import Dict, Any, List, Tuple


class PerformanceTracker:
    """Tracks Redis memory performance metrics over time."""

    def __init__(self, history_file="redis_performance_history.json"):
        """Initialize the tracker."""
        self.history_file = history_file
        self.history_data = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        """Load performance history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")

        # Create new history structure if file doesn't exist or can't be loaded
        return {
            "timestamps": [],
            "write_ops": [],
            "read_ops": [],
            "batch_ops": [],
            "memory_per_entry": [],
            "batch_sizes": [],
            "notes": []
        }

    def add_benchmark_result(self, result_file: str, note: str = "") -> bool:
        """Add a new benchmark result to the history."""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Extract overall metrics
            overall = data.get('overall', {})
            if not overall:
                print("No overall metrics found in result file")
                return False

            # Get current timestamp
            timestamp = datetime.datetime.now().isoformat()

            # Add data to history
            self.history_data["timestamps"].append(timestamp)
            self.history_data["write_ops"].append(overall.get("writes_per_second", 0))
            self.history_data["read_ops"].append(overall.get("reads_per_second", 0))
            self.history_data["batch_ops"].append(overall.get("batch_throughput", 0))
            self.history_data["memory_per_entry"].append(overall.get("memory_per_entry", 0))
            self.history_data["batch_sizes"].append(overall.get("batch_size", 0))
            self.history_data["notes"].append(note)

            # Save updated history
            self._save_history()

            print(f"Added benchmark from {result_file} to performance history")
            return True

        except Exception as e:
            print(f"Error adding benchmark result: {e}")
            return False

    def _save_history(self) -> bool:
        """Save history data to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
            return False

    def plot_performance_over_time(self, output_file="performance_history.png") -> None:
        """Plot performance metrics over time."""
        if not self.history_data["timestamps"]:
            print("No performance history available to plot")
            return

        # Convert timestamps to readable dates
        dates = [datetime.datetime.fromisoformat(ts).strftime('%m-%d %H:%M')
                 for ts in self.history_data["timestamps"]]

        # Create figure with subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

        # Plot write operations
        axs[0].plot(dates, self.history_data["write_ops"], 'o-', color='blue', linewidth=2)
        axs[0].set_title('Write Operations Performance')
        axs[0].set_ylabel('Operations/second')
        axs[0].grid(True, linestyle='--', alpha=0.7)

        # Plot read operations
        axs[1].plot(dates, self.history_data["read_ops"], 'o-', color='green', linewidth=2)
        axs[1].set_title('Read Operations Performance')
        axs[1].set_ylabel('Operations/second')
        axs[1].grid(True, linestyle='--', alpha=0.7)

        # Plot batch operations
        axs[2].plot(dates, self.history_data["batch_ops"], 'o-', color='purple', linewidth=2)
        axs[2].set_title('Batch Operations Performance')
        axs[2].set_ylabel('Operations/second')

        # Add batch size as annotations
        for i, (x, y, bs) in enumerate(zip(dates, self.history_data["batch_ops"],
                                           self.history_data["batch_sizes"])):
            axs[2].annotate(f"BS:{bs}", (x, y), textcoords="offset points",
                          xytext=(0, 5), ha='center')

        axs[2].grid(True, linestyle='--', alpha=0.7)

        # Plot memory per entry
        axs[3].plot(dates, self.history_data["memory_per_entry"], 'o-', color='red', linewidth=2)
        axs[3].set_title('Memory Usage Efficiency')
        axs[3].set_ylabel('Bytes/entry')
        axs[3].grid(True, linestyle='--', alpha=0.7)

        # Rotate date labels for better readability
        plt.xticks(rotation=45)

        # Add notes as text at the bottom
        for i, (date, note) in enumerate(zip(dates, self.history_data["notes"])):
            if note:  # Only add if there's a note
                plt.figtext(0.01, 0.01 + (i * 0.02), f"{date}: {note}", fontsize=8)

        fig.tight_layout()
        plt.subplots_adjust(bottom=0.1 + min(0.2, len(self.history_data["notes"]) * 0.02))

        # Save the chart
        plt.savefig(output_file)
        print(f"Performance history chart saved to {output_file}")

    def get_performance_change(self, baseline_idx: int = 0) -> Dict[str, Any]:
        """Calculate performance changes compared to a baseline."""
        if len(self.history_data["timestamps"]) <= baseline_idx:
            print("Not enough history data to calculate changes")
            return {}

        # Get the latest data
        latest_idx = len(self.history_data["timestamps"]) - 1

        # Calculate percent changes
        write_change = self._calculate_percent_change(
            self.history_data["write_ops"][baseline_idx],
            self.history_data["write_ops"][latest_idx]
        )

        read_change = self._calculate_percent_change(
            self.history_data["read_ops"][baseline_idx],
            self.history_data["read_ops"][latest_idx]
        )

        batch_change = self._calculate_percent_change(
            self.history_data["batch_ops"][baseline_idx],
            self.history_data["batch_ops"][latest_idx]
        )

        # For memory, lower is better
        memory_change = self._calculate_percent_change(
            self.history_data["memory_per_entry"][baseline_idx],
            self.history_data["memory_per_entry"][latest_idx],
            lower_is_better=True
        )

        return {
            "baseline_date": self.history_data["timestamps"][baseline_idx],
            "latest_date": self.history_data["timestamps"][latest_idx],
            "write_ops_change": write_change,
            "read_ops_change": read_change,
            "batch_ops_change": batch_change,
            "memory_change": memory_change
        }

    def _calculate_percent_change(self, baseline: float, current: float, lower_is_better: bool = False) -> float:
        """Calculate percent change between baseline and current value."""
        if baseline == 0:
            return float('inf') if current > 0 else 0

        change = ((current - baseline) / baseline) * 100

        # Invert if lower is better
        if lower_is_better:
            change = -change

        return change

    def print_summary(self) -> None:
        """Print a summary of the performance history."""
        if not self.history_data["timestamps"]:
            print("No performance history available")
            return

        print("\nRedis Memory Performance History")
        print("================================")

        # Print header
        print(f"{'Date':<20} {'Write (ops/s)':<15} {'Read (ops/s)':<15} {'Batch (ops/s)':<15} {'Memory (B/entry)':<20} {'Note':<30}")
        print("-" * 115)

        # Print each result
        for i in range(len(self.history_data["timestamps"])):
            date = datetime.datetime.fromisoformat(self.history_data["timestamps"][i]).strftime('%Y-%m-%d %H:%M')
            write = self.history_data["write_ops"][i]
            read = self.history_data["read_ops"][i]
            batch = self.history_data["batch_ops"][i]
            memory = self.history_data["memory_per_entry"][i]
            note = self.history_data["notes"][i]

            truncated_note = note[:27] + "..." if len(note) > 30 else note

            print(f"{date:<20} {write:<15.2f} {read:<15.2f} {batch:<15.2f} {memory:<20.2f} {truncated_note:<30}")

        # Print performance changes
        if len(self.history_data["timestamps"]) > 1:
            changes = self.get_performance_change()
            if changes:
                print("\nPerformance Changes (compared to first benchmark):")
                print(f"Write operations: {changes['write_ops_change']:+.2f}%")
                print(f"Read operations: {changes['read_ops_change']:+.2f}%")
                print(f"Batch operations: {changes['batch_ops_change']:+.2f}%")
                print(f"Memory efficiency: {changes['memory_change']:+.2f}%")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Track Redis memory performance over time")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new benchmark result")
    add_parser.add_argument("result_file", help="Path to benchmark result JSON file")
    add_parser.add_argument("--note", type=str, default="", help="Note about this benchmark run")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot performance history")
    plot_parser.add_argument("--output", type=str, default="performance_history.png",
                            help="Output file for the plot")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show performance history")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    tracker = PerformanceTracker()

    if args.command == "add":
        tracker.add_benchmark_result(args.result_file, args.note)
        tracker.print_summary()

    elif args.command == "plot":
        tracker.plot_performance_over_time(args.output)

    elif args.command == "show":
        tracker.print_summary()

    else:
        # If no command provided, show summary
        tracker.print_summary()

    return 0


if __name__ == "__main__":
    exit(main()) 
