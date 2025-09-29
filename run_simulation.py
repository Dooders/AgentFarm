#!/usr/bin/env python3
"""
Simple script to run a single simulation using the farm simulation framework.
"""
# Standard library imports
import argparse
import cProfile
import os
import pstats
import time
import warnings
from io import StringIO

# Suppress warnings that might interfere with CI output parsing
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tianshou")

# Local imports
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation


def run_profiled_simulation(num_steps, config, output_dir):
    """
    Run the simulation with profiling and return the environment.
    """
    return run_simulation(
        num_steps=num_steps, config=config, path=output_dir, save_config=True
    )


def main():
    """
    Main entry point for running a single simulation.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run a single farm simulation")
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        choices=["development", "production", "testing"],
        help="Environment to use (development/production/testing)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["benchmark", "simulation", "research"],
        help="Profile to use (benchmark/simulation/research)",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of simulation steps to run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed for the simulation (overrides config if provided)",
    )
    parser.add_argument(
        "--perf-profile",
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
        config = SimulationConfig.from_centralized_config(
            environment=args.environment,
            profile=args.profile
        )
        print(f"Loaded configuration (environment: {args.environment}, profile: {args.profile or 'none'})")

        # Apply in-memory database settings if requested
        if args.in_memory:
            config.use_in_memory_db = True
            config.in_memory_db_memory_limit_mb = (
                args.memory_limit if args.memory_limit else 1000
            )
            config.persist_db_on_completion = not args.no_persist
            print("Using in-memory database for improved performance")
            if args.memory_limit:
                print(f"Memory limit: {args.memory_limit} MB")
            if args.no_persist:
                print("Warning: In-memory database will not be persisted to disk")
        # Apply seed override if provided
        if args.seed is not None:
            try:
                # SimulationConfig should accept seed directly
                config.seed = args.seed
                print(f"Using deterministic seed: {args.seed}")
            except Exception as e:
                print(f"Warning: Failed to set seed on config: {e}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Run simulation
    print(f"Starting simulation with {args.steps} steps")
    print(f"Output will be saved to {output_dir}")
    print(
        f"Agent configuration - System: {config.population.system_agents}, Independent: {config.population.independent_agents}, Control: {config.population.control_agents}"
    )

    try:
        start_time = time.time()

        if args.perf_profile:
            print("Profiling enabled - collecting performance data...")
            # Run with profiling
            profiler = cProfile.Profile()
            profiler.enable()

            environment = run_profiled_simulation(args.steps, config, output_dir)

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
            with open(profile_path, "w", encoding="utf-8") as f:
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
                num_steps=args.steps, config=config, path=output_dir, save_config=True
            )

        elapsed_time = time.time() - start_time
        # Ensure output is flushed for reliable CI detection
        import sys
        print("\n=== SIMULATION COMPLETED SUCCESSFULLY ===", flush=True)
        print(f"Simulation completed in {elapsed_time:.2f} seconds", flush=True)
        print(f"Final agent count: {len(environment.agents)}", flush=True)
        if len(environment.agents) == 0:
            print(
                "WARNING: No agents were created or all agents died during simulation",
                flush=True
            )
        else:
            agent_types = {}
            for agent in environment.agents:
                agent_type = agent.__class__.__name__
                if agent_type in agent_types:
                    agent_types[agent_type] += 1
                else:
                    agent_types[agent_type] = 1
            print(f"Agent types: {agent_types}", flush=True)
        print(f"Results saved to {output_dir}", flush=True)
    except Exception as e:
        print(f"Simulation failed: {e}")


if __name__ == "__main__":
    main()
