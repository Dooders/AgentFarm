#!/usr/bin/env python3
"""
Script to compare and visualize Redis benchmark results.
"""

import json
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, List


def load_results(filenames: List[str]) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON files."""
    results = []
    for filename in filenames:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                # Add the filename as a label (convert from filename to label)
                label = os.path.splitext(os.path.basename(filename))[0]
                if label.startswith('results_batch_'):
                    label = label.replace('results_batch_', 'Batch Size: ')
                elif label == 'redis_benchmark_results':
                    label = 'Default'
                data['label'] = label
                results.append(data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return results


def plot_operations_comparison(results: List[Dict[str, Any]], output_file: str = 'comparison.png'):
    """Plot a comparison of operations per second for different configurations."""
    
    labels = [result['label'] for result in results]
    writes = [result['overall']['writes_per_second'] for result in results]
    reads = [result['overall']['reads_per_second'] for result in results]
    batches = [result['overall']['batch_throughput'] for result in results]
    
    x = range(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar([i - width for i in x], writes, width, label='Write Operations')
    rects2 = ax.bar(x, reads, width, label='Read Operations')
    rects3 = ax.bar([i + width for i in x], batches, width, label='Batch Operations')
    
    ax.set_ylabel('Operations per Second')
    ax.set_title('Redis Memory Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    print(f"Comparison chart saved to {output_file}")
    
    # Show the plot if running interactively
    plt.show()


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of the benchmark results."""
    print("\nRedis Memory Benchmark Comparison")
    print("=================================")
    
    # Print header
    print(f"{'Configuration':<20} {'Writes (ops/s)':<15} {'Reads (ops/s)':<15} {'Batch (ops/s)':<15} {'Memory (bytes/entry)':<20}")
    print("-" * 85)
    
    # Print each result
    for result in results:
        label = result['label']
        writes = result['overall']['writes_per_second']
        reads = result['overall']['reads_per_second']
        batches = result['overall']['batch_throughput']
        memory = result['overall']['memory_per_entry']
        
        print(f"{label:<20} {writes:<15.2f} {reads:<15.2f} {batches:<15.2f} {memory:<20.2f}")
    
    # Print observations
    best_writes = max(results, key=lambda x: x['overall']['writes_per_second'])
    best_reads = max(results, key=lambda x: x['overall']['reads_per_second'])
    best_batches = max(results, key=lambda x: x['overall']['batch_throughput'])
    best_memory = min(results, key=lambda x: x['overall']['memory_per_entry'])
    
    print("\nKey Observations:")
    print(f"- Best write performance: {best_writes['label']} ({best_writes['overall']['writes_per_second']:.2f} ops/s)")
    print(f"- Best read performance: {best_reads['label']} ({best_reads['overall']['reads_per_second']:.2f} ops/s)")
    print(f"- Best batch throughput: {best_batches['label']} ({best_batches['overall']['batch_throughput']:.2f} ops/s)")
    print(f"- Best memory efficiency: {best_memory['label']} ({best_memory['overall']['memory_per_entry']:.2f} bytes/entry)")


def main():
    """Main entry point."""
    # Default result files
    default_files = [
        'redis_benchmark_results.json',  # Default (batch size 100)
        'results_batch_10.json',         # Batch size 10
        'results_batch_500.json'         # Batch size 500
    ]
    
    # Use provided files or defaults
    files_to_compare = sys.argv[1:] if len(sys.argv) > 1 else default_files
    
    # Load results
    results = load_results(files_to_compare)
    
    if not results:
        print("No valid result files found!")
        return 1
    
    # Print summary
    print_summary(results)
    
    # Create visualization
    plot_operations_comparison(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 