#!/usr/bin/env python3
"""
Integrated simulation runner using Hydra configuration system.

This script demonstrates how to run simulations using the new Hydra configuration
system while maintaining compatibility with existing simulation code.
"""

import argparse
import cProfile
import os
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')

from farm.core.config_hydra_bridge_simple import create_simulation_config_from_hydra
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
    Main entry point for running a single simulation with integrated Hydra configuration.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run a single farm simulation with integrated Hydra configuration")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="/workspace/config_hydra/conf",
        help="Directory containing Hydra configuration files",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        choices=["development", "staging", "production"],
        help="Environment to use (development, staging, production)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="system_agent",
        choices=["system_agent", "independent_agent", "control_agent"],
        help="Agent type to use",
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=None, 
        help="Number of simulation steps to run (overrides config)"
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
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Configuration override (e.g., 'max_steps=500' or 'debug=false')",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show current configuration and exit",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit",
    )
    args = parser.parse_args()

    # Ensure simulations directory exists
    output_dir = "simulations"
    os.makedirs(output_dir, exist_ok=True)

    # Create integrated configuration
    try:
        config = create_simulation_config_from_hydra(
            config_dir=args.config_dir,
            environment=args.environment,
            agent=args.agent,
            overrides=args.override
        )
        
        print(f"Loaded integrated Hydra configuration:")
        print(f"  Environment: {args.environment}")
        print(f"  Agent: {args.agent}")
        print(f"  Config directory: {args.config_dir}")
        if args.override:
            print(f"  Overrides: {args.override}")
        
    except Exception as e:
        print(f"Failed to load integrated configuration: {e}")
        return 1

    # Show configuration if requested
    if args.show_config:
        print("\nCurrent Configuration:")
        print("=" * 50)
        print(f"Simulation ID: {config.simulation_id}")
        print(f"Environment: {config.width}x{config.height}")
        print(f"Max steps: {config.max_steps}")
        print(f"Debug mode: {config.debug}")
        print(f"System agents: {config.system_agents}")
        print(f"Independent agents: {config.independent_agents}")
        print(f"Control agents: {config.control_agents}")
        print(f"Use in-memory DB: {config.use_in_memory_db}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Epsilon start: {config.epsilon_start}")
        return 0

    # Validate configuration if requested
    if args.validate_config:
        print("\nValidating Configuration:")
        print("=" * 50)
        try:
            # Basic validation
            assert config.width > 0, "Width must be positive"
            assert config.height > 0, "Height must be positive"
            assert config.max_steps > 0, "Max steps must be positive"
            assert config.system_agents >= 0, "System agents must be non-negative"
            assert config.independent_agents >= 0, "Independent agents must be non-negative"
            assert config.control_agents >= 0, "Control agents must be non-negative"
            
            print("✅ Configuration validation passed!")
            return 0
        except AssertionError as e:
            print(f"❌ Configuration validation failed: {e}")
            return 1

    # Apply command line overrides
    if args.steps:
        config.max_steps = args.steps
        print(f"Overriding max_steps to {args.steps}")

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

    # Run simulation
    print(f"\nStarting simulation:")
    print(f"  Steps: {config.max_steps}")
    print(f"  Environment: {config.width}x{config.height}")
    print(f"  Debug mode: {config.debug}")
    print(f"  Output directory: {output_dir}")
    print(f"  Agent configuration:")
    print(f"    System agents: {config.system_agents}")
    print(f"    Independent agents: {config.independent_agents}")
    print(f"    Control agents: {config.control_agents}")
    print(f"  Learning parameters:")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Epsilon start: {config.epsilon_start}")
    print(f"    Epsilon min: {config.epsilon_min}")

    # Run with or without profiling
    if args.profile:
        print("\nRunning with profiling...")
        start_time = time.time()
        
        # Run profiled simulation
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            env = run_profiled_simulation(
                num_steps=config.max_steps,
                config=config,
                output_dir=output_dir
            )
            
            profiler.disable()
            end_time = time.time()
            
            print(f"Simulation completed in {end_time - start_time:.2f} seconds")
            
            # Save profiling results
            profile_file = os.path.join(output_dir, "profile_results.prof")
            profiler.dump_stats(profile_file)
            print(f"Profiling results saved to {profile_file}")
            
            # Generate SnakeViz visualization if available
            if not args.no_snakeviz:
                try:
                    import subprocess
                    print("Generating SnakeViz visualization...")
                    subprocess.run(["snakeviz", profile_file], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("SnakeViz not available. Install with: pip install snakeviz")
                    print(f"To view results manually: snakeviz {profile_file}")
            
        except Exception as e:
            profiler.disable()
            print(f"Simulation failed: {e}")
            return 1
            
    else:
        print("\nRunning simulation...")
        start_time = time.time()
        
        try:
            env = run_profiled_simulation(
                num_steps=config.max_steps,
                config=config,
                output_dir=output_dir
            )
            
            end_time = time.time()
            print(f"Simulation completed in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            return 1

    print(f"\nSimulation results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())