#!/usr/bin/env python3
"""
Hydra-based script to run a single simulation using the farm simulation framework.
This version uses the new Hydra configuration system.
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

from farm.core.config_hydra_simple import create_simple_hydra_config_manager
from farm.core.simulation import run_simulation

# Constants
DEFAULT_MEMORY_LIMIT_MB = 1000


def run_profiled_simulation(num_steps, config_dict, output_dir):
    """
    Run the simulation with profiling and return the environment.
    """
    # Convert config dict to SimulationConfig if needed
    from farm.core.config_hydra_bridge import HydraSimulationConfig
    config = HydraSimulationConfig.from_dict(config_dict)
    
    return run_simulation(
        num_steps=num_steps, config=config, path=output_dir, save_config=True
    )


def main():
    """
    Main entry point for running a single simulation with Hydra configuration.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run a single farm simulation with Hydra configuration")
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

    # Create Hydra configuration manager
    try:
        config_manager = create_simple_hydra_config_manager(
            config_dir=args.config_dir,
            environment=args.environment,
            agent=args.agent,
            overrides=args.override
        )
        
        print(f"Loaded Hydra configuration:")
        print(f"  Environment: {args.environment}")
        print(f"  Agent: {args.agent}")
        print(f"  Config directory: {args.config_dir}")
        if args.override:
            print(f"  Overrides: {args.override}")
        
    except Exception as e:
        print(f"Failed to load Hydra configuration: {e}")
        return 1

    # Show configuration if requested
    if args.show_config:
        print("\nCurrent Configuration:")
        print("=" * 50)
        config_dict = config_manager.to_dict()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
        return 0

    # Validate configuration if requested
    if args.validate_config:
        print("\nValidating Configuration:")
        print("=" * 50)
        errors = config_manager.validate_configuration()
        if errors:
            print("❌ Configuration validation failed:")
            for area, error_list in errors.items():
                print(f"  {area}:")
                for error in error_list:
                    print(f"    - {error}")
            return 1
        else:
            print("✅ Configuration validation passed!")
            return 0

    # Get configuration
    config_dict = config_manager.to_dict()
    
    # Apply command line overrides
    if args.steps:
        config_dict['max_steps'] = args.steps
        print(f"Overriding max_steps to {args.steps}")

    # Apply in-memory database settings if requested
    if args.in_memory:
        config_dict['use_in_memory_db'] = True
        config_dict['in_memory_db_memory_limit_mb'] = (
            args.memory_limit if args.memory_limit else DEFAULT_MEMORY_LIMIT_MB
        )
        config_dict['persist_db_on_completion'] = not args.no_persist
        print("Using in-memory database for improved performance")
        if args.memory_limit:
            print(f"Memory limit: {args.memory_limit} MB")
        if args.no_persist:
            print("Warning: In-memory database will not be persisted to disk")

    # Run simulation
    print(f"\nStarting simulation:")
    print(f"  Steps: {config_dict.get('max_steps', 'default')}")
    print(f"  Environment: {config_dict.get('environment', 'unknown')}")
    print(f"  Debug mode: {config_dict.get('debug', False)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Agent configuration:")
    agent_params = config_dict.get('agent_parameters', {})
    for agent_type, params in agent_params.items():
        if isinstance(params, dict):
            share_weight = params.get('share_weight', 'N/A')
            attack_weight = params.get('attack_weight', 'N/A')
            print(f"    {agent_type}: share={share_weight}, attack={attack_weight}")

    # Run with or without profiling
    if args.profile:
        print("\nRunning with profiling...")
        start_time = time.time()
        
        # Run profiled simulation
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            env = run_profiled_simulation(
                num_steps=config_dict.get('max_steps', 1000),
                config_dict=config_dict,
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
                num_steps=config_dict.get('max_steps', 1000),
                config_dict=config_dict,
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