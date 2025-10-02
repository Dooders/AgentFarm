#!/usr/bin/env python3
"""
Run multiple simulations in a centralized experiment database.

This script demonstrates how to use ExperimentDatabase to store multiple
simulations in a single database file for easier analysis and comparison.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.config import SimulationConfig
from farm.database.experiment_database import ExperimentDatabase
from farm.utils.logging_config import configure_logging, get_logger


def run_simulation_with_context(sim_context, config, num_steps, logger):
    """
    Run a simulation using a simulation context instead of separate database.
    
    This is a wrapper that would integrate with your existing simulation code.
    You'll need to modify your run_simulation function to accept a database
    context instead of creating its own database.
    
    Parameters
    ----------
    sim_context : SimulationContext
        Database context for this specific simulation
    config : SimulationConfig
        Simulation configuration
    num_steps : int
        Number of steps to run
    logger : Logger
        Logger instance
    """
    # TODO: Integrate with your actual simulation runner
    # For now, this is a placeholder showing the pattern
    
    from farm.core.simulation import Simulation
    
    # Create simulation with the context's logger
    # You'll need to modify the Simulation class to accept a logger parameter
    sim = Simulation(config=config)
    
    # Override the simulation's database with our context
    # This ensures all logging goes to the centralized database
    if hasattr(sim, 'db'):
        original_db = sim.db
        sim.db = sim_context
        sim.logger = sim_context.logger
    
    # Run the simulation
    for step in range(num_steps):
        # Your simulation step logic here
        # The simulation should use sim_context.logger.log_step() etc.
        pass
    
    # Flush any remaining data
    sim_context.flush_all_buffers()
    
    logger.info(
        "simulation_completed",
        simulation_id=sim_context.simulation_id,
        steps=num_steps
    )


def main():
    """Main entry point for running an experiment."""
    parser = argparse.ArgumentParser(
        description="Run multiple simulations in a centralized experiment database"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Experiment",
        help="Human-readable experiment name"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Experiment description"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10,
        help="Number of simulations to run (default: 10)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps per simulation (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Directory to store experiment databases (default: experiments)"
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        choices=["development", "production", "testing"],
        help="Environment to use"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["benchmark", "simulation", "research"],
        help="Configuration profile to use"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed for simulations (each sim gets seed_base + index)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run simulations in parallel (experimental, may cause DB locking)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    configure_logging(
        environment=args.environment,
        log_dir="logs",
        log_level=args.log_level,
        enable_colors=True
    )
    logger = get_logger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate experiment ID if not provided
    if args.experiment_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_id = f"exp_{timestamp}"
    
    # Database path
    db_path = os.path.join(args.output_dir, f"{args.experiment_id}.db")
    
    logger.info(
        "experiment_starting",
        experiment_id=args.experiment_id,
        num_simulations=args.num_simulations,
        steps_per_simulation=args.steps,
        database_path=db_path
    )
    
    # Load configuration
    try:
        config = SimulationConfig.from_centralized_config(
            environment=args.environment,
            profile=args.profile
        )
    except Exception as e:
        logger.error(
            "config_load_failed",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        sys.exit(1)
    
    # Create experiment database
    experiment_db = ExperimentDatabase(
        db_path=db_path,
        experiment_id=args.experiment_id,
        config=config
    )
    
    # Update experiment metadata
    experiment_db.update_experiment_status(
        status="running",
        results_summary={
            "name": args.name,
            "description": args.description,
            "planned_simulations": args.num_simulations,
            "steps_per_simulation": args.steps
        }
    )
    
    logger.info(
        "experiment_initialized",
        database_path=db_path
    )
    
    # Run simulations
    start_time = time.time()
    successful_sims = 0
    failed_sims = 0
    
    try:
        for i in range(args.num_simulations):
            sim_id = f"sim_{i:04d}"
            
            logger.info(
                "simulation_starting",
                simulation_id=sim_id,
                index=i + 1,
                total=args.num_simulations
            )
            
            try:
                # Set seed for this simulation
                sim_seed = args.seed_base + i
                config.seed = sim_seed
                
                # Create simulation context
                sim_context = experiment_db.create_simulation_context(
                    simulation_id=sim_id,
                    parameters={
                        "run_index": i,
                        "seed": sim_seed,
                        "steps": args.steps,
                        "environment": args.environment,
                        "profile": args.profile
                    }
                )
                
                # Run simulation
                sim_start = time.time()
                
                # TODO: Replace this with actual simulation runner
                # For now, just demonstrate the pattern
                logger.warning(
                    "simulation_placeholder",
                    message="Actual simulation integration needed - see run_simulation_with_context"
                )
                
                # Placeholder for actual simulation
                # run_simulation_with_context(sim_context, config, args.steps, logger)
                
                sim_duration = time.time() - sim_start
                
                # Update simulation status
                experiment_db.update_simulation_status(
                    simulation_id=sim_id,
                    status="completed",
                    results_summary={
                        "duration_seconds": sim_duration,
                        "steps": args.steps
                    }
                )
                
                successful_sims += 1
                
                logger.info(
                    "simulation_completed",
                    simulation_id=sim_id,
                    duration_seconds=round(sim_duration, 2)
                )
                
            except Exception as e:
                failed_sims += 1
                logger.error(
                    "simulation_failed",
                    simulation_id=sim_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True
                )
                
                # Update simulation status
                experiment_db.update_simulation_status(
                    simulation_id=sim_id,
                    status="failed",
                    results_summary={
                        "error": str(e)
                    }
                )
        
        # Calculate final statistics
        total_duration = time.time() - start_time
        
        # Update experiment status
        experiment_db.update_experiment_status(
            status="completed",
            results_summary={
                "name": args.name,
                "description": args.description,
                "total_simulations": args.num_simulations,
                "successful_simulations": successful_sims,
                "failed_simulations": failed_sims,
                "total_duration_seconds": total_duration,
                "average_duration_seconds": total_duration / args.num_simulations
            }
        )
        
        logger.info(
            "experiment_completed",
            successful_simulations=successful_sims,
            failed_simulations=failed_sims,
            total_duration_seconds=round(total_duration, 2),
            database_path=db_path
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED")
        print("=" * 70)
        print(f"Experiment ID:    {args.experiment_id}")
        print(f"Database:         {db_path}")
        print(f"Successful runs:  {successful_sims}/{args.num_simulations}")
        print(f"Failed runs:      {failed_sims}/{args.num_simulations}")
        print(f"Total duration:   {total_duration:.2f} seconds")
        print(f"Average per sim:  {total_duration/args.num_simulations:.2f} seconds")
        print("=" * 70)
        print(f"\nDatabase contains all simulation data in: {db_path}")
        print("Use the query tools to analyze results across all simulations.")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        logger.warning("experiment_interrupted", message="Interrupted by user")
        experiment_db.update_experiment_status(
            status="interrupted",
            results_summary={
                "completed_simulations": successful_sims,
                "failed_simulations": failed_sims
            }
        )
    finally:
        # Always close the database
        experiment_db.close()
        logger.info("experiment_database_closed")


if __name__ == "__main__":
    main()
