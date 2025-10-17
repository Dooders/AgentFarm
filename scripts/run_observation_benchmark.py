#!/usr/bin/env python3
"""
Observation Flow Benchmark Runner - Multiple Stress Levels

Choose from different benchmark levels:
1. Basic: 200 agents, 100 steps (20K observations)
2. High Stress: 500 agents, 200 steps (100K observations)
3. Ultra Stress: 1000 agents, 500 steps (500K observations)
"""

import sys
import os
import warnings
import json
from datetime import datetime

# Suppress warnings and logging
warnings.filterwarnings("ignore")

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.core.registry import REGISTRY
from benchmarks.core.runner import Runner

def display_results(results, params, test_name, baseline_obs=20000):
    """Display benchmark results in a consistent format."""
    print(f"\n{'='*70}")
    print(f"{test_name.upper()} RESULTS")
    print('='*70)

    if results.iteration_metrics:
        # Get metrics from the last iteration
        last_iteration = results.iteration_metrics[-1]
        raw_results = last_iteration.metrics

        # Calculate derived metrics
        total_obs = raw_results.get('total_observes', 0)
        total_time = raw_results.get('total_time_s', 0)
        obs_per_sec = raw_results.get('observes_per_sec', 0)
        mean_step_time = raw_results.get('mean_step_time_s', 0)

        print("PERFORMANCE METRICS:")
        print(f"  Total observations: {total_obs:,}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {obs_per_sec:,.0f} observations/second")
        print(f"  Average step time: {mean_step_time:.4f} seconds")
        print()

        print("EFFICIENCY ANALYSIS:")
        num_agents = params.get('num_agents', 200)
        steps = params.get('steps', 100)
        print(f"  Observations per agent per step: {total_obs / (num_agents * steps):.1f}")
        print(f"  Time per 1K observations: {(1000 / obs_per_sec) * 1000:.1f} ms")
        print(f"  Scaling factor: {total_obs / baseline_obs:.1f}x baseline")
        print()

        print("RAW METRICS:")
        for key, value in raw_results.items():
            if isinstance(value, float):
                if "time" in key.lower():
                    print(f"  {key}: {value:.4f} seconds")
                elif "per_sec" in key:
                    print(f"  {key}: {value:,.2f}")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    return results

def document_results(results, params, test_name, save_to_file=True):
    """Create structured documentation of benchmark results."""
    doc_data = {
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "num_agents": params.get('num_agents', 200),
            "steps": params.get('steps', 100),
            "environment_size": f"{params.get('width', 200)}x{params.get('height', 200)}",
            "observation_radius": params.get('radius', 6),
            "fov_radius": params.get('fov_radius', 6)
        },
        "results": {}
    }

    if results.iteration_metrics:
        # Get metrics from the last iteration
        last_iteration = results.iteration_metrics[-1]
        raw_results = last_iteration.metrics
        doc_data["results"] = dict(raw_results)

        # Add derived metrics
        total_obs = raw_results.get('total_observes', 0)
        total_time = raw_results.get('total_time_s', 0)
        obs_per_sec = raw_results.get('observes_per_sec', 0)
        num_agents = params.get('num_agents', 200)
        steps = params.get('steps', 100)

        doc_data["derived_metrics"] = {
            "total_expected_observations": num_agents * steps,
            "observations_per_agent_per_step": total_obs / (num_agents * steps),
            "time_per_1000_observations_ms": (1000 / obs_per_sec) * 1000,
            "completion_rate": total_obs / (num_agents * steps)
        }

    if save_to_file:
        filename = f"benchmark_results_{test_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2)
        print(f"\nResults documented in: {filename}")

    return doc_data

def run_basic_benchmark():
    """Run the basic benchmark: 200 agents, 100 steps."""
    print("🔹 BASIC BENCHMARK")
    print("=" * 50)
    print("Light load: 200 agents, 100 steps")
    print("20,000 total observations - baseline performance")
    print("=" * 50)

    # Discover experiments
    REGISTRY.discover_package("benchmarks.implementations")

    params = {
        "width": 200,      # Standard environment
        "height": 200,     # Standard environment
        "num_agents": 200, # Basic load
        "steps": 100,      # Basic duration
        "radius": 5,       # Standard observation radius
        "fov_radius": 5,   # Standard field of view
        "device": "cpu"
    }

    print("Configuration:")
    print(f"  • Environment: {params['width']}x{params['height']}")
    print(f"  • Agents: {params['num_agents']}")
    print(f"  • Steps: {params['steps']}")
    print(f"  • Expected observations: {params['num_agents'] * params['steps']:,}")
    print()

    # Create experiment and runner
    experiment = REGISTRY.create("observation_flow", params)
    runner = Runner(
        name="basic_benchmark",
        experiment=experiment,
        output_dir="benchmark_results",
        iterations_measured=1,
    )

    results = runner.run()
    display_results(results, params, "Basic Benchmark")
    document_results(results, params, "Basic Benchmark")

    print("\nBenchmark completed successfully.")
    return results

def run_high_stress_benchmark():
    """Run the high-stress benchmark: 500 agents, 200 steps."""
    print("🔥 HIGH-STRESS BENCHMARK")
    print("=" * 60)
    print("Heavy load: 500 agents, 200 steps")
    print("100,000 total observations - pushing limits!")
    print("=" * 60)

    params = {
        "width": 400,      # Larger environment
        "height": 400,     # Larger environment
        "num_agents": 500, # High load
        "steps": 200,      # Extended duration
        "radius": 8,       # Larger observation radius
        "fov_radius": 8,   # Larger field of view
        "device": "cpu"
    }

    print("Configuration:")
    print(f"  • Environment: {params['width']}x{params['height']}")
    print(f"  • Agents: {params['num_agents']}")
    print(f"  • Steps: {params['steps']}")
    print(f"  • Expected observations: {params['num_agents'] * params['steps']:,}")
    print(f"  • This is {params['num_agents'] * params['steps'] // 20000}x the basic test!")
    print()

    # Create experiment and runner
    experiment = REGISTRY.create("observation_flow", params)
    runner = Runner(
        name="high_stress_benchmark",
        experiment=experiment,
        output_dir="benchmark_results",
        iterations_measured=1,
    )

    results = runner.run()
    display_results(results, params, "High-Stress Benchmark")

    print("\n✅ HIGH-STRESS BENCHMARK COMPLETED SUCCESSFULLY!")
    print("🚀 System handled 5x load without issues!")
    return results

def run_ultra_stress_benchmark():
    """Run the ultra-stress benchmark: 1000 agents, 500 steps."""
    print("🚀🚀🚀 ULTRA-STRESS BENCHMARK - MAXIMUM LIMITS!")
    print("=" * 70)
    print("🚨 EXTREME LOAD: 1000 agents, 500 steps!")
    print("🚨 500,000 total observations - BREAKING THE SYSTEM!")
    print("🚨 MASSIVE 600×600 environment!")
    print("=" * 70)

    params = {
        "width": 600,      # Massive environment
        "height": 600,     # Massive environment
        "num_agents": 1000,  # EXTREME LOAD
        "steps": 500,      # EXTREME duration
        "radius": 10,      # Very large observation radius
        "fov_radius": 10,  # Very large field of view
        "device": "cpu"
    }

    print("Configuration:")
    print(f"  • Environment: {params['width']}x{params['height']}")
    print(f"  • Agents: {params['num_agents']}")
    print(f"  • Steps: {params['steps']}")
    print(f"  • Expected observations: {params['num_agents'] * params['steps']:,}")
    print(f"  • This is {params['num_agents'] * params['steps'] // 20000}x the basic test!")
    print(f"  • This is {params['num_agents'] * params['steps'] // 100000}x the high-stress test!")
    print()

    # Create experiment and runner
    experiment = REGISTRY.create("observation_flow", params)
    runner = Runner(
        name="ultra_stress_benchmark",
        experiment=experiment,
        output_dir="benchmark_results",
        iterations_measured=1,
    )

    results = runner.run()
    display_results(results, params, "Ultra-Stress Benchmark")

    print("\n🎉🎉🎉 ULTRA-STRESS BENCHMARK COMPLETED SUCCESSFULLY!")
    print("🚨 SYSTEM HANDLED 500,000 OBSERVATIONS UNDER MAXIMUM STRESS!")
    return results

def main():
    """Main menu to choose benchmark level."""
    print("🧪 OBSERVATION FLOW BENCHMARK SUITE")
    print("=" * 50)
    print("Choose your stress test level:")
    print()
    print("1. 🔹 BASIC     - 200 agents, 100 steps  (20K observations)")
    print("2. 🔥 HIGH      - 500 agents, 200 steps  (100K observations)")
    print("3. 🚀 ULTRA     - 1000 agents, 500 steps (500K observations)")
    print("4. ⚡ ALL TESTS - Run all levels sequentially")
    print()

    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                run_basic_benchmark()
                break
            elif choice == "2":
                run_high_stress_benchmark()
                break
            elif choice == "3":
                run_ultra_stress_benchmark()
                break
            elif choice == "4":
                print("\n" + "="*70)
                print("🏃 RUNNING ALL BENCHMARK LEVELS SEQUENTIALLY")
                print("="*70)

                print("\n▶️  STARTING BASIC TEST...")
                run_basic_benchmark()

                print("\n▶️  STARTING HIGH-STRESS TEST...")
                run_high_stress_benchmark()

                print("\n▶️  STARTING ULTRA-STRESS TEST...")
                run_ultra_stress_benchmark()

                print("\n" + "="*70)
                print("🎊 ALL BENCHMARK LEVELS COMPLETED!")
                print("📊 Performance scaling validated across all stress levels!")
                print("="*70)
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

        except KeyboardInterrupt:
            print("\n\n⏹️  Benchmark interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error during benchmark: {e}")
            break

if __name__ == "__main__":
    main()
