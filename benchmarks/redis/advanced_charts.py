#!/usr/bin/env python3
"""
Advanced visualization for Redis benchmark results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
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
                # Add the filename as a label
                label = os.path.splitext(os.path.basename(filename))[0]
                if label.startswith('results_batch_'):
                    batch_size = label.replace('results_batch_', '')
                    label = f"Batch Size: {batch_size}"
                elif label == 'redis_benchmark_results':
                    label = 'Default (Batch 100)'
                data['label'] = label
                # Extract batch size as a number
                if 'overall' in data and 'batch_size' in data['overall']:
                    data['batch_size'] = data['overall']['batch_size']
                else:
                    data['batch_size'] = 100  # default
                results.append(data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return results


def plot_operations_comparison(results: List[Dict[str, Any]], output_file: str = 'comparison_bar.png'):
    """Plot a bar chart comparison of operations per second."""
    
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
    plt.savefig(output_file)
    print(f"Bar chart saved to {output_file}")


def plot_batch_size_vs_throughput(results: List[Dict[str, Any]], output_file: str = 'batch_size_throughput.png'):
    """Plot the effect of batch size on throughput."""
    # Sort results by batch size
    sorted_results = sorted(results, key=lambda x: x['batch_size'])
    
    batch_sizes = [result['batch_size'] for result in sorted_results]
    throughputs = [result['overall']['batch_throughput'] for result in sorted_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8)
    
    # Add labels and improve appearance
    for i, (x, y) in enumerate(zip(batch_sizes, throughputs)):
        ax.annotate(f"{y:.0f} ops/s", (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Operations per Second')
    ax.set_title('Effect of Batch Size on Throughput')
    
    # Set x-axis to use actual batch size values
    ax.set_xticks(batch_sizes)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(output_file)
    print(f"Batch size vs. throughput chart saved to {output_file}")


def plot_operation_time_distribution(results: List[Dict[str, Any]], output_file: str = 'operation_time_dist.png'):
    """Plot the distribution of operation times."""
    fig, axes = plt.subplots(nrows=len(results), ncols=3, figsize=(15, 5 * len(results)))
    
    # If there's only one result, convert axes to 2D array for consistent indexing
    if len(results) == 1:
        axes = np.array([axes])
    
    for i, result in enumerate(results):
        # Get operation times data with fallbacks for missing keys
        # Check if the 'write' key exists in the result dictionary
        if 'write' in result:
            write_data = {
                'times': [
                    result['write']['min_operation_time'],
                    result['write']['avg_operation_time'],
                    result['write']['max_operation_time']
                ],
                'labels': ['Min', 'Avg', 'Max']
            }
        else:
            # Provide default values if 'write' key is missing
            write_data = {
                'times': [0, 0, 0],
                'labels': ['Min', 'Avg', 'Max']
            }
            print(f"Warning: 'write' data missing in result {result.get('label', i)}")
        
        # Check if the 'read' key exists in the result dictionary
        if 'read' in result:
            read_data = {
                'times': [
                    result['read']['min_operation_time'],
                    result['read']['avg_operation_time'],
                    result['read']['max_operation_time']
                ],
                'labels': ['Min', 'Avg', 'Max']
            }
        else:
            # Provide default values if 'read' key is missing
            read_data = {
                'times': [0, 0, 0],
                'labels': ['Min', 'Avg', 'Max']
            }
            print(f"Warning: 'read' data missing in result {result.get('label', i)}")
        
        # Check if the 'batch' key exists in the result dictionary
        if 'batch' in result:
            batch_data = {
                'times': [
                    result['batch'].get('min_batch_time', 0),
                    result['batch'].get('avg_batch_time', 0),
                    result['batch'].get('max_batch_time', 0)
                ],
                'labels': ['Min', 'Avg', 'Max']
            }
        else:
            # Provide default values if 'batch' key is missing
            batch_data = {
                'times': [0, 0, 0],
                'labels': ['Min', 'Avg', 'Max']
            }
            print(f"Warning: 'batch' data missing in result {result.get('label', i)}")
        
        # Convert to milliseconds for better readability
        write_times_ms = [t * 1000 for t in write_data['times']]
        read_times_ms = [t * 1000 for t in read_data['times']]
        batch_times_ms = [t * 1000 for t in batch_data['times']]
        
        # Plot
        axes[i, 0].bar(write_data['labels'], write_times_ms)
        axes[i, 0].set_title(f"{result['label']} - Write Time (ms)")
        axes[i, 0].set_ylabel('Time (ms)')
        
        axes[i, 1].bar(read_data['labels'], read_times_ms)
        axes[i, 1].set_title(f"{result['label']} - Read Time (ms)")
        axes[i, 1].set_ylabel('Time (ms)')
        
        axes[i, 2].bar(batch_data['labels'], batch_times_ms)
        axes[i, 2].set_title(f"{result['label']} - Batch Time (ms)")
        axes[i, 2].set_ylabel('Time (ms)')
    
    fig.tight_layout()
    plt.savefig(output_file)
    print(f"Operation time distribution chart saved to {output_file}")


def plot_radar_chart(results: List[Dict[str, Any]], output_file: str = 'radar_chart.png'):
    """Create a radar chart to compare different configurations."""
    # Number of variables
    categories = ['Write Speed', 'Read Speed', 'Batch Throughput', 'Memory Efficiency']
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add category labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw y-axis lines from center to edge
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    plt.ylim(0, 1)
    
    # Find the max values for each metric to normalize
    max_write = max(result['overall']['writes_per_second'] for result in results)
    max_read = max(result['overall']['reads_per_second'] for result in results)
    max_batch = max(result['overall']['batch_throughput'] for result in results)
    max_memory = max(result['overall']['memory_per_entry'] for result in results)
    
    # For memory, lower is better, so we invert the normalization
    memory_values = [result['overall']['memory_per_entry'] for result in results]
    min_memory = min(memory_values)
    
    # Plot each configuration
    for i, result in enumerate(results):
        values = [
            result['overall']['writes_per_second'] / max_write,
            result['overall']['reads_per_second'] / max_read,
            result['overall']['batch_throughput'] / max_batch,
            # For memory, lower is better so we invert the ratio
            (max_memory - result['overall']['memory_per_entry'] + min_memory) / max_memory
        ]
        
        # Close the loop
        values += values[:1]
        
        # Plot the configuration
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=result['label'])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(output_file)
    print(f"Radar chart saved to {output_file}")


def plot_memory_usage_over_entries(results: List[Dict[str, Any]], output_file: str = 'memory_usage.png'):
    """Plot memory usage as entries increase."""
    # Only use the first result for simplicity
    if not results:
        return
    
    result = results[0]
    
    if 'memory' not in result or 'memory_samples' not in result['memory']:
        print("Memory samples data not available")
        return
    
    samples = result['memory']['memory_samples']
    
    entries = [sample['entries'] for sample in samples]
    memory_per_entry = [sample['memory_per_entry'] for sample in samples]
    total_memory = [sample['total_memory'] for sample in samples]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Total memory usage
    ax1.plot(entries, total_memory, 'o-', linewidth=2, color='blue')
    ax1.set_xlabel('Number of Entries')
    ax1.set_ylabel('Total Memory Usage (bytes)')
    ax1.set_title('Total Memory Usage vs. Number of Entries')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Memory per entry
    ax2.plot(entries, memory_per_entry, 'o-', linewidth=2, color='red')
    ax2.set_xlabel('Number of Entries')
    ax2.set_ylabel('Memory per Entry (bytes)')
    ax2.set_title('Memory per Entry vs. Number of Entries')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(output_file)
    print(f"Memory usage chart saved to {output_file}")


def main():
    """Main entry point."""
    # Default result files
    default_files = [
        'results/redis_benchmark_results.json',  # Default (batch size 100)
        'results/results_batch_10.json',         # Batch size 10
        'results/results_batch_500.json'         # Batch size 500
    ]
    
    # Use provided files or defaults
    files_to_compare = sys.argv[1:] if len(sys.argv) > 1 else default_files
    
    # Create output directory
    os.makedirs('charts', exist_ok=True)
    
    # Load results
    results = load_results(files_to_compare)
    
    if not results:
        print("No valid result files found!")
        return 1
    
    # Generate various charts
    plot_operations_comparison(results, 'charts/comparison_bar.png')
    plot_batch_size_vs_throughput(results, 'charts/batch_size_throughput.png')
    plot_operation_time_distribution(results, 'charts/operation_time_dist.png')
    plot_radar_chart(results, 'charts/radar_chart.png')
    plot_memory_usage_over_entries(results, 'charts/memory_usage.png')
    
    print("\nAll charts have been generated in the 'charts' directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 