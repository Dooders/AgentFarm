#!/usr/bin/env python3
"""
Simple script to run a single simulation using the farm simulation framework.
"""

# Standard library imports
import argparse
import cProfile
import os
import pstats
import sys
import time
import warnings
from io import StringIO

# Local imports
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.utils.logging import configure_logging, get_logger

# Suppress warnings that might interfere with CI output parsing
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tianshou")


def run_profiled_simulation(num_steps, config, output_dir, enable_console_logging=False):
    """
    Run the simulation with profiling and return the environment.
    """
    return run_simulation(
        num_steps=num_steps,
        config=config,
        path=output_dir,
        save_config=True,
        disable_console_logging=not enable_console_logging,
    )


def main():
    """
    Main entry point for running a single simulation.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run a single farm simulation")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format",
    )
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
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps to run")
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
    parser.add_argument(
        "--enable-console-logging",
        action="store_true",
        help="Enable console logging during simulation (disabled by default for clean tqdm output)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip database validation after simulation completes",
    )
    args = parser.parse_args()

    # Ensure simulations directory exists
    output_dir = "simulations"
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    configure_logging(
        environment=args.environment,
        log_dir="logs",
        log_level=args.log_level,
        json_logs=args.json_logs,
        enable_colors=not args.json_logs,
        disable_console=not args.enable_console_logging,
    )
    logger = get_logger(__name__)

    # Load configuration
    try:
        config = SimulationConfig.from_centralized_config(environment=args.environment, profile=args.profile)
        logger.info(
            "configuration_loaded",
            environment=args.environment,
            profile=args.profile or "none",
        )

        # Apply in-memory database settings if requested
        if args.in_memory:
            config.database.use_in_memory_db = True
            config.database.in_memory_db_memory_limit_mb = args.memory_limit if args.memory_limit else 1000
            config.database.persist_db_on_completion = not args.no_persist
            logger.info(
                "in_memory_db_configured",
                memory_limit_mb=config.database.in_memory_db_memory_limit_mb,
                persist=config.database.persist_db_on_completion,
            )
            if args.no_persist:
                logger.warning("in_memory_db_no_persist", message="Database will not be persisted to disk")

        # Apply seed override if provided
        if args.seed is not None:
            try:
                config.seed = args.seed
                logger.info("seed_configured", seed=args.seed, deterministic=True)
            except Exception as e:
                logger.warning(
                    "seed_configuration_failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
    except Exception as e:
        logger.error(
            "configuration_load_failed",
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        sys.exit(1)

    # Run simulation
    logger.info(
        "simulation_starting",
        num_steps=args.steps,
        output_dir=output_dir,
        system_agents=config.population.system_agents,
        independent_agents=config.population.independent_agents,
        control_agents=config.population.control_agents,
    )

    try:
        start_time = time.time()

        if args.perf_profile:
            logger.info("profiling_enabled", profiler="cProfile")
            # Run with profiling
            profiler = cProfile.Profile()
            profiler.enable()

            environment = run_profiled_simulation(args.steps, config, output_dir, args.enable_console_logging)

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

            logger.info(
                "profiling_results_saved",
                profile_text=profile_path,
                profile_binary=profile_binary_path,
            )

            # Open with snakeviz by default unless disabled
            if not args.no_snakeviz:
                try:
                    logger.info("launching_snakeviz", profile_file=profile_binary_path)
                    import subprocess

                    subprocess.Popen(["snakeviz", profile_binary_path])
                except (ImportError, FileNotFoundError):
                    logger.warning(
                        "snakeviz_not_found",
                        message="Install with 'pip install snakeviz'",
                        command=f"snakeviz {profile_binary_path}",
                    )
        else:
            # Run without profiling
            environment = run_simulation(
                num_steps=args.steps,
                config=config,
                path=output_dir,
                save_config=True,
                disable_console_logging=not args.enable_console_logging,
            )

        elapsed_time = time.time() - start_time

        # Log completion
        logger.info(
            "simulation_completed",
            duration_seconds=round(elapsed_time, 2),
            final_agent_count=len(environment.agents),
            output_dir=output_dir,
        )

        # Validate database integrity (runs by default unless skipped)
        should_validate = not args.skip_validation and config.database.enable_validation
        if should_validate:
            if environment.db and hasattr(environment.db, 'db_path'):
                try:
                    from farm.database.validation import validate_simulation_database
                    logger.info("database_validation_starting", database_path=environment.db.db_path)
                    print("🔍 Running database validation...", flush=True)
                    
                    validation_report = validate_simulation_database(
                        environment.db.db_path,
                        simulation_id=environment.simulation_id,
                        include_integrity=config.database.validation_include_integrity,
                        include_statistical=config.database.validation_include_statistical
                    )
                    
                    if validation_report.has_warnings():
                        logger.warning(
                            "database_validation_warnings",
                            warnings=validation_report.warning_count,
                            details=validation_report.get_summary()
                        )
                        print(f"⚠️  Database validation found {validation_report.warning_count} warnings", flush=True)
                    
                    if validation_report.has_errors():
                        logger.error(
                            "database_validation_errors",
                            errors=validation_report.error_count,
                            details=validation_report.get_summary()
                        )
                        print(f"❌ Database validation found {validation_report.error_count} errors", flush=True)
                    else:
                        logger.info(
                            "database_validation_passed",
                            checks_performed=validation_report.total_checks,
                            duration_seconds=round(validation_report.end_time - validation_report.start_time, 2)
                        )
                        print(f"✅ Database validation passed ({validation_report.total_checks} checks)", flush=True)
                except Exception as e:
                    logger.warning(
                        "database_validation_failed",
                        error=str(e),
                        message="Validation could not complete but simulation succeeded"
                    )
                    print(f"⚠️  Database validation failed: {str(e)}", flush=True)
            else:
                logger.info(
                    "database_validation_skipped",
                    reason="No database found or database path not available"
                )
        else:
            if args.skip_validation:
                logger.info("database_validation_skipped", reason="Validation disabled by user")
                print("⏭️  Database validation skipped (disabled by user)", flush=True)
            elif not config.database.enable_validation:
                logger.info("database_validation_skipped", reason="Validation disabled in configuration")
                print("⏭️  Database validation skipped (disabled in configuration)", flush=True)

        # Check for empty simulation
        if len(environment.agents) == 0:
            logger.warning(
                "simulation_no_agents",
                message="No agents were created or all agents died during simulation",
            )
        else:
            # Count agent types
            agent_types = {}
            for agent in environment.agent_objects:
                agent_type = agent.__class__.__name__
                agent_types[agent_type] = agent_types.get(agent_type, 0) + 1

            logger.info("simulation_agent_distribution", agent_types=agent_types)

        # Print summary for CI/console (keeping print for backward compatibility)
        print("\n=== SIMULATION COMPLETED SUCCESSFULLY ===", flush=True)
        print(f"Simulation completed in {elapsed_time:.2f} seconds", flush=True)
        print(f"Final agent count: {len(environment.agents)}", flush=True)
        print(f"Results saved to {output_dir}", flush=True)

    except Exception as e:
        logger.error(
            "simulation_failed",
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
