#!/usr/bin/env python3
"""
Simple script to run a single simulation using the farm simulation framework.
"""

import argparse
import cProfile
import os
import pstats
import time
from io import StringIO

from config_hydra import create_simple_hydra_config_manager
from farm.core.simulation import run_simulation


def run_profiled_simulation(num_steps, config_manager, output_dir):
    """
    Run the simulation with profiling and return the environment.
    """
    return run_simulation(
        num_steps=num_steps, config_manager=config_manager, path=output_dir, save_config=True
    )


def main():
    """
    Main entry point for running a single simulation.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run a single farm simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of simulation steps to run"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling with cProfile and SnakeViz visualization",
    )
    parser.add_argument(
        "--no-snakeviz",
        action="store_true",
        help="Disable automatic SnakeViz visualization when profiling",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        default=True,
        help="Use in-memory database for improved performance",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=None,
        help="Memory limit in MB for in-memory database (None = no limit)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't persist in-memory database to disk after simulation",
    )
    args = parser.parse_args()

    # Ensure simulations directory exists
    output_dir = "simulations"
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    try:
        # Create Hydra configuration manager
        config_manager = create_simple_hydra_config_manager(
            config_dir=os.path.abspath("config_hydra/conf"),
            environment="development",
            agent="system_agent"
        )
        print(f"Loaded configuration from {args.config}")

        # Apply in-memory database settings if requested
        if args.in_memory:
            # Get current config and update it
            config_dict = config_manager.get_config()
            config_dict['use_in_memory_db'] = True
            config_dict['in_memory_db_memory_limit_mb'] = (
                args.memory_limit if args.memory_limit else 1000
            )
            config_dict['persist_db_on_completion'] = not args.no_persist
            print("Using in-memory database for improved performance")
            if args.memory_limit:
                print(f"Memory limit: {args.memory_limit} MB")
            if args.no_persist:
                print("Warning: In-memory database will not be persisted to disk")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Run simulation
    print(f"Starting simulation with {args.steps} steps")
    print(f"Output will be saved to {output_dir}")
    print(
        f"Agent configuration - System: {config_manager.get_simulation_config().system_agents}, "
        f"Independent: {config_manager.get_simulation_config().independent_agents}, "
        f"Control: {config_manager.get_simulation_config().control_agents}"
    )

    try:
        start_time = time.time()

        if args.profile:
            print("Profiling enabled - collecting performance data...")
            # Run with profiling
            profiler = cProfile.Profile()
            profiler.enable()

            environment = run_profiled_simulation(args.steps, config_manager, output_dir)

            profiler.disable()

            # Save profiling results
            profile_path = os.path.join(output_dir, "profile_stats.txt")
            profile_binary_path = os.path.join(output_dir, "profile_stats.prof")

            # Save binary profile data for snakeviz
            profiler.dump_stats(profile_binary_path)

            # Print to console first
            print("\n" + "=" * 50)
            print("PROFILING RESULTS (TOP 20 FUNCTIONS BY TIME)")
            print("=" * 50)

            s = StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            stats.print_stats(20)  # Print top 20 to console
            print(s.getvalue())

            # Save to file
            with open(profile_path, "w") as f:
                stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
                stats.print_stats(50)  # Save top 50 functions to file

            print(f"Full profiling stats saved to {profile_path}")
            print(f"Binary profile data saved to {profile_binary_path}")

            # Open with snakeviz by default unless disabled
            if not args.no_snakeviz:
                try:
                    print("\nLaunching SnakeViz for interactive visualization...")
                    import subprocess

                    subprocess.Popen(["snakeviz", profile_binary_path])
                except (ImportError, FileNotFoundError):
                    print(
                        "\nError: SnakeViz not found. Install with 'pip install snakeviz'"
                    )
                    print(
                        f"Once installed, you can view the profile with: snakeviz {profile_binary_path}"
                    )
        else:
            # Run without profiling
            environment = run_simulation(
                num_steps=args.steps, config_manager=config_manager, path=output_dir, save_config=True
            )

        elapsed_time = time.time() - start_time
        print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
        print(f"Final agent count: {len(environment.agents)}")
        if len(environment.agents) == 0:
            print(
                "WARNING: No agents were created or all agents died during simulation"
            )
        else:
            agent_types = {}
            for agent in environment.agents:
                agent_type = agent.__class__.__name__
                if agent_type in agent_types:
                    agent_types[agent_type] += 1
                else:
                    agent_types[agent_type] = 1
            print(f"Agent types: {agent_types}")
        print(f"Results saved to {output_dir}")
    except Exception as e:
        print(f"Simulation failed: {e}")


if __name__ == "__main__":
    main()
