"""
Simulation Runner for AgentFarm Multi-Agent Simulations

This module provides the main simulation execution engine for AgentFarm.
It handles the complete lifecycle of multi-agent simulations, from initialization
through execution to finalization and analysis.

The simulation runner orchestrates the interaction between agents and their
environment, managing the simulation loop, agent lifecycle, data collection,
and result analysis. It supports both single-run simulations and batch processing.

Key Components:
    - setup_logging: Configure logging system for simulation runs
    - create_initial_agents: Generate initial agent population
    - run_simulation: Execute complete simulation with progress tracking
    - run_simulation_batch: Run multiple simulations with parameter variations
    - Simulation result collection and analysis

Features:
    - Configurable agent populations and types
    - Progress tracking with tqdm
    - Comprehensive logging and metrics collection
    - Database integration for result persistence
    - Support for various termination conditions
    - Batch processing capabilities for parameter studies

Usage:
    # Run a single simulation
    config = SimulationConfig(...)
    results = run_simulation(config)

    # Run multiple simulations with different parameters
    configs = [SimulationConfig(...), SimulationConfig(...)]
    batch_results = run_simulation_batch(configs)
"""

import json
import os
import random
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from farm.config import SimulationConfig
from farm.core.agent import AgentFactory, AgentServices
from farm.core.environment import Environment
from farm.utils.identity import Identity
from farm.utils.logging import configure_logging, get_logger, log_simulation, log_step

# Shared Identity instance for efficiency
_shared_identity = Identity()

logger = get_logger(__name__)


def create_services_from_environment(environment: Environment) -> AgentServices:
    """
    Create AgentServices from environment, handling optional services gracefully.
    
    Args:
        environment: The environment to extract services from
        
    Returns:
        AgentServices container with all available services
    """
    from farm.core.services.implementations import (
        EnvironmentLoggingService,
        EnvironmentMetricsService,
        EnvironmentTimeService,
        EnvironmentValidationService,
        EnvironmentAgentLifecycleService,
    )
    
    return AgentServices(
        spatial_service=environment.spatial_service,
        time_service=EnvironmentTimeService(environment),
        metrics_service=EnvironmentMetricsService(environment),
        logging_service=EnvironmentLoggingService(environment),
        validation_service=EnvironmentValidationService(environment),
        lifecycle_service=EnvironmentAgentLifecycleService(environment),
    )


def create_initial_agents(
    environment: Environment,
    num_system_agents: int,
    num_independent_agents: int,
    num_control_agents: int,
) -> List[Tuple[float, float]]:
    """
    Create initial population of agents.

    Parameters
    ----------
    environment : Environment
        Simulation environment
    num_system_agents : int
        Number of system agents to create
    num_independent_agents : int
        Number of independent agents to create

    Returns
    -------
    List[Tuple[float, float]]
        List of initial agent positions
    """
    positions = []
    logger.info(
        "creating_initial_agents",
        num_system_agents=num_system_agents,
        num_independent_agents=num_independent_agents,
        num_control_agents=num_control_agents,
    )

    # Create services from environment
    services = create_services_from_environment(environment)
    
    # Create factory
    factory = AgentFactory(services)

    # Create agent component configuration from simulation config
    from farm.core.agent.config.component_configs import AgentComponentConfig
    agent_config = None
    if environment.config is not None:
        agent_config = AgentComponentConfig.from_simulation_config(environment.config)

    # Use a seeded random number generator for deterministic agent creation
    if hasattr(environment, "seed_value") and environment.seed_value is not None:
        rng = random.Random(environment.seed_value)
    else:
        rng = random

    # Helper function to generate deterministic positions
    def get_random_position():
        return (
            int(rng.uniform(0, environment.width)),
            int(rng.uniform(0, environment.height)),
        )

    # Get initial resource level with fallback for when config is None
    initial_resource_level = 5.0  # Default fallback value (matches config default)
    if environment.config is not None and hasattr(environment.config, 'agent_behavior'):
        initial_resource_level = environment.config.agent_behavior.initial_resource_level

    # Create system agents with learning behavior
    for _ in range(num_system_agents):
        position = get_random_position()
        agent = factory.create_learning_agent(
            agent_id=environment.get_next_agent_id(),
            position=position,
            initial_resources=int(initial_resource_level),
            config=agent_config,
            environment=environment,
        )
        environment.add_agent(agent, flush_immediately=True)
        positions.append(position)

    # Create independent agents with learning behavior
    for _ in range(num_independent_agents):
        position = get_random_position()
        agent = factory.create_learning_agent(
            agent_id=environment.get_next_agent_id(),
            position=position,
            initial_resources=int(initial_resource_level),
            config=agent_config,
            environment=environment,
        )
        environment.add_agent(agent, flush_immediately=True)
        positions.append(position)

    # Create control agents with learning behavior
    for _ in range(num_control_agents):
        position = get_random_position()
        agent = factory.create_learning_agent(
            agent_id=environment.get_next_agent_id(),
            position=position,
            initial_resources=int(initial_resource_level),
            config=agent_config,
            environment=environment,
        )
        environment.add_agent(agent, flush_immediately=True)
        positions.append(position)

    logger.info("initial_agents_complete", total_agents=len(environment.agents))

    return positions


def init_random_seeds(seed=None):
    """
    Initialize all random number generators with a seed for deterministic behavior.

    Parameters
    ----------
    seed : int, optional
        The seed value to use, by default None
    """
    if seed is not None:
        # Set the Python random module seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch seeds if available
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Uncomment for full determinism (may affect performance)
                # torch.backends.cudnn.deterministic = True
                # torch.backends.cudnn.benchmark = False
        except ImportError:
            logger.info("pytorch_unavailable", message="Skipping torch seed initialization")

        logger.info("random_seeds_initialized", seed=seed, deterministic=True)


def run_simulation(
    num_steps: int,
    config: SimulationConfig,
    path: Optional[str] = None,
    save_config: bool = True,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
    identity: Optional[Identity] = None,
    disable_console_logging: bool = False,
) -> Environment:
    """
    Run the main simulation loop.

    Parameters
    ----------
    num_steps : int
        Number of simulation steps to run
    config : SimulationConfig
        Configuration for the simulation
    path : Optional[str], optional
        Path where to save simulation data, by default None
    save_config : bool, optional
        Whether to save configuration to disk, by default True
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
    simulation_id : Optional[str], optional
        Unique ID for this simulation run. If None, one will be generated.
    identity : Optional[Identity], optional
        Identity service instance for ID generation. If None, uses shared instance.
    disable_console_logging : bool, optional
        Whether to disable console logging during simulation (useful with tqdm), by default False

    Returns
    -------
    Environment
        The simulation environment after completion
    """
    # Generate simulation_id if not provided
    if simulation_id is None:
        identity_service = identity if identity is not None else _shared_identity
        simulation_id = str(identity_service.simulation_id())

    # Set seed for reproducibility if provided
    if seed is not None:
        # Store seed in config for future reference
        config.seed = seed

        # Initialize all random seeds
        init_random_seeds(seed)

    # Configure logging with console control if requested
    if disable_console_logging:
        configure_logging(disable_console=True)

    # Log simulation start
    logger.info(
        "simulation_starting",
        simulation_id=simulation_id,
        seed=seed,
        num_steps=num_steps,
        environment_size=(config.environment.width, config.environment.height),
    )

    # Initialize start_time here so it's available in both code paths
    start_time = datetime.now()

    try:
        # Set up database path (None if path is None)
        # Include simulation_id in filename to prevent conflicts between multiple simulations
        db_path = f"{path}/simulation_{simulation_id}.db" if path is not None else None

        # Ensure the database directory exists
        if db_path is not None:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Handle in-memory database configuration
        if config.database.use_in_memory_db:
            logger.info(
                "using_in_memory_database",
                memory_limit_mb=config.database.in_memory_db_memory_limit_mb,
            )
            from farm.database.database import InMemorySimulationDatabase

            # Create environment with in-memory database
            environment = Environment(
                width=config.environment.width,
                height=config.environment.height,
                resource_distribution={
                    "type": "random",
                    "amount": config.resources.initial_resources,
                },
                db_path="",  # Will be ignored for in-memory DB
                config=config,
                simulation_id=simulation_id,
                seed=seed,  # Pass seed to Environment
            )

            # Replace the default database with in-memory database
            if hasattr(environment, "db") and environment.db is not None:
                environment.db.close()

            # Initialize in-memory database with optional memory limit
            environment.db = InMemorySimulationDatabase(
                memory_limit_mb=config.database.in_memory_db_memory_limit_mb,
                simulation_id=simulation_id,
            )

            # Create simulation record for in-memory database
            environment.db.add_simulation_record(
                simulation_id=simulation_id,
                start_time=datetime.now(),
                status="running",
                parameters=config.to_dict(),
            )
            
            # CRITICAL: Ensure simulation record is committed before creating agents
            # This prevents foreign key constraint violations when agents are created
            environment.db.logger.flush_all_buffers()
            logger.info("simulation_record_committed_to_database")

        else:
            # Clean up any existing database file for disk-based DB
            if db_path is not None and os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except PermissionError:
                    # If can't remove, create unique filename
                    original_db_path = db_path  # Store original path before modification
                    base, ext = os.path.splitext(db_path)
                    db_path = f"{base}_{int(time.time())}{ext}"
                    logger.warning(
                        "database_path_changed",
                        original_path=original_db_path,
                        new_path=db_path,
                    )

            # Create parent directory if it doesn't exist
            if db_path is not None:
                os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Note: Database tables and simulation record are already created in setup_db

            # Create environment with disk-based database
            environment = Environment(
                width=config.environment.width,
                height=config.environment.height,
                resource_distribution={
                    "type": "random",
                    "amount": config.resources.initial_resources,
                },
                db_path=db_path if db_path is not None else "simulation.db",
                config=config,
                simulation_id=simulation_id,
                seed=seed,  # Pass seed to Environment
            )

        # Save configuration if requested and path is provided
        if save_config and path is not None:
            config_path = f"{path}/config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, indent=4)
            logger.info("configuration_saved", config_path=config_path)

        # Ensure simulation record exists and is committed before creating agents
        if environment.db is not None:
            # Check if simulation record exists, create if not
            try:
                environment.db.save_configuration(config.to_dict())
            except Exception as e:
                # If save_configuration fails due to missing simulation record,
                # create the record first and try again
                if "FOREIGN KEY constraint failed" in str(e):
                    environment.db.add_simulation_record(
                        simulation_id=simulation_id,
                        start_time=datetime.now(),
                        status="running",
                        parameters=config.to_dict(),
                    )
                    environment.db.save_configuration(config.to_dict())
                else:
                    raise
            
            # CRITICAL: Ensure simulation record is committed before creating agents
            # This prevents foreign key constraint violations when agents are created
            environment.db.logger.flush_all_buffers()
            logger.info("simulation_record_committed_to_database")

        # Create initial agents
        create_initial_agents(
            environment,
            config.population.system_agents,
            config.population.independent_agents,
            config.population.control_agents,
        )

        # Ensure all initial agents are committed to database before simulation starts
        if environment.db is not None:
            environment.db.logger.flush_all_buffers()
            logger.info("initial_agents_committed_to_database")

        # Main simulation loop
        # Disable tqdm progress bar in CI environments to avoid output interference
        disable_tqdm = (
            os.environ.get("CI", "").lower() in ("true", "1") or os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
        )

        for step in tqdm(
            range(num_steps),
            desc="Simulation progress",
            unit="step",
            disable=disable_tqdm,
        ):
            logger.debug("step_starting", step=step, total_steps=num_steps)

            # Time the step processing
            step_start_time = time.time()

            # Process agents in batches
            alive_agents = [agent for agent in environment.agent_objects if agent.alive]

            # Stop if no agents are alive
            if len(alive_agents) < 1:
                logger.info(
                    "simulation_stopped_early",
                    step=step,
                    total_steps=num_steps,
                    reason="no_agents_remaining",
                )
                break

            # Get batch size from config, with fallback to default
            perf_cfg = getattr(config, "performance", None)
            batch_size = getattr(perf_cfg, "agent_processing_batch_size", 32)

            # Process batches without a nested progress bar
            batch_ranges = list(range(0, len(alive_agents), batch_size))
            for i in batch_ranges:
                batch = alive_agents[i : i + batch_size]

                for agent in batch:
                    agent.act()

            # Ensure all database operations are flushed before environment update
            # This prevents timing mismatches between metrics calculation and agent database logging
            if environment.db is not None:
                environment.db.logger.flush_all_buffers()
            
            # Update environment once per step
            environment.update()

            # Check for slow steps
            step_duration = time.time() - step_start_time

            # Warn if step takes > 1 second
            if step_duration > 1.0:
                logger.warning(
                    "slow_step_detected",
                    step=step,
                    duration_seconds=round(step_duration, 3),
                    agents_count=len(environment.agents),
                    resources_count=len(environment.resources),
                    threshold_seconds=1.0,
                )

            # Also log step timing at DEBUG level for performance analysis
            elif step % 10 == 0:  # Every 10 steps
                logger.debug(
                    "step_timing",
                    step=step,
                    duration_ms=round(step_duration * 1000, 2),
                    agents_count=len(environment.agents),
                )

        # Ensure final state is saved
        environment.update()

        # Force final flush of database buffers
        if environment.db:
            environment.db.logger.flush_all_buffers()

            # Persist in-memory database to disk if configured and db_path is provided
            if config.database.persist_db_on_completion and db_path is not None:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

                    logger.info("persisting_in_memory_database", db_path=db_path)
                    # Persist with selected tables or all tables
                    from farm.database.database import InMemorySimulationDatabase

                    if isinstance(environment.db, InMemorySimulationDatabase):
                        stats = environment.db.persist_to_disk(
                            db_path=db_path,
                            tables=config.database.in_memory_tables_to_persist,
                            show_progress=True,
                        )
                        logger.info(
                            "database_persisted",
                            rows_copied=stats["rows_copied"],
                            duration_seconds=round(stats["duration"], 2),
                            db_path=db_path,
                        )
                    else:
                        logger.warning(
                            "database_persistence_skipped",
                            reason="not_in_memory_database",
                            db_type=type(environment.db).__name__,
                        )
                except Exception as e:
                    logger.error(
                        "database_persistence_failed",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        exc_info=True,
                    )

            # Close database connection
            environment.db.close()

    except Exception as e:
        logger.error(
            "simulation_failed",
            simulation_id=simulation_id,
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        if "environment" in locals():
            try:
                environment.cleanup()
            except Exception as cleanup_error:
                logger.error(
                    "cleanup_error",
                    error_type=type(cleanup_error).__name__,
                    error_message=str(cleanup_error),
                )
        raise
    finally:
        if "environment" in locals():
            try:
                if hasattr(environment, "db") and environment.db:
                    environment.db.close()
                environment.cleanup()
            except Exception as e:
                logger.error(
                    "final_cleanup_error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )

    # Calculate summary statistics
    elapsed_time = datetime.now() - start_time
    total_duration = elapsed_time.total_seconds()
    avg_step_time_ms = (total_duration * 1000) / max(environment.time, 1)

    # Count births and deaths from database or metrics
    birth_count = 0
    death_count = 0
    reproduction_count = 0

    if (
        environment.metrics_tracker is not None
        and hasattr(environment.metrics_tracker, "cumulative_metrics")
        and environment.metrics_tracker.cumulative_metrics is not None
    ):
        birth_count = getattr(environment.metrics_tracker.cumulative_metrics, "total_births", 0)
        death_count = getattr(environment.metrics_tracker.cumulative_metrics, "total_deaths", 0)
        reproduction_count = getattr(environment.metrics_tracker.cumulative_metrics, "total_reproduction_successes", 0)

    # Calculate initial population for logging
    def calculate_initial_population(config):
        """Calculate the initial population from configuration."""
        if config is None:
            return 0
        return config.population.system_agents + config.population.independent_agents

    logger.info(
        "simulation_completed",
        simulation_id=simulation_id,
        total_steps=environment.time,
        max_steps_configured=num_steps,
        final_population=len(environment.agents),
        initial_population=calculate_initial_population(config),
        total_births=birth_count,
        total_deaths=death_count,
        reproduction_events=reproduction_count,
        final_resources=environment.cached_total_resources,
        resource_nodes=len(environment.resources),
        duration_seconds=round(total_duration, 2),
        avg_step_time_ms=round(avg_step_time_ms, 2),
        steps_per_second=round(environment.time / max(total_duration, 0.001), 2),
        database_path=environment.db.db_path if environment.db else None,
        terminated_early=environment.time < num_steps,
        termination_reason=(
            "resources_depleted"
            if environment.cached_total_resources == 0
            else "agents_extinct"
            if len(environment.agents) == 0
            else "completed"
        ),
    )

    # Validate database (runs by default if enabled in config)
    # Skip validation for in-memory databases as they will be validated after persistence
    if (config.database.enable_validation and 
        hasattr(environment, 'db') and environment.db and 
        hasattr(environment.db, 'db_path') and 
        environment.db.db_path != ":memory:"):
        try:
            from farm.database.validation import validate_simulation_database
            logger.info("simulation_database_validation_starting", database_path=environment.db.db_path)
            
            validation_report = validate_simulation_database(
                environment.db.db_path,
                simulation_id=simulation_id,
                include_integrity=config.database.validation_include_integrity,
                include_statistical=config.database.validation_include_statistical
            )
            
            if not validation_report.is_clean():
                logger.warning(
                    "simulation_validation_issues",
                    warnings=validation_report.warning_count,
                    errors=validation_report.error_count,
                    duration_seconds=round(validation_report.end_time - validation_report.start_time, 2)
                )
            else:
                logger.info(
                    "simulation_database_validation_passed",
                    checks_performed=validation_report.total_checks,
                    duration_seconds=round(validation_report.end_time - validation_report.start_time, 2)
                )
        except Exception as e:
            logger.warning("simulation_validation_error", error=str(e))
    else:
        if not config.database.enable_validation:
            logger.info("simulation_database_validation_skipped", reason="Validation disabled in configuration")
        else:
            logger.info("simulation_database_validation_skipped", reason="No database available")

    return environment


def main():
    """
    Main entry point for running a simulation directly.
    """
    # Load configuration
    config = SimulationConfig.from_centralized_config()

    # Run simulation
    run_simulation(num_steps=1000, config=config, save_config=True, path="simulations")


if __name__ == "__main__":
    main()
