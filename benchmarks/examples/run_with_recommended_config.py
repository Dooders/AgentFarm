#!/usr/bin/env python3
"""
Example script demonstrating how to run a simulation with the recommended
configuration for post-simulation analysis.
"""

import os
import argparse
import sys

from farm.core.simulation import run_simulation
from benchmarks.utils.config_helper import get_recommended_config, print_config_recommendations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a simulation with the recommended configuration for post-simulation analysis"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps",
    )
    
    parser.add_argument(
        "--agents",
        type=int,
        default=30,
        help="Total number of agents",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="simulation_results",
        help="Directory to save simulation results",
    )
    
    return parser.parse_args()


def main():
    """Run a simulation with the recommended configuration."""
    args = parse_args()
    
    # Print recommendations
    print_config_recommendations()
    
    print("\nRunning simulation with recommended configuration...")
    
    # Get recommended configuration
    config = get_recommended_config(
        num_agents=args.agents,
        num_steps=args.steps,
    )
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run simulation
    env = run_simulation(
        num_steps=args.steps,
        config=config,
        path=args.output,
        save_config=True,
    )
    
    print(f"\nSimulation completed successfully!")
    print(f"Results saved to {os.path.abspath(args.output)}")
    print("\nYou can now perform post-simulation analysis on the results.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 