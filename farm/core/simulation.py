import json
import logging
import os
import random
import time
from datetime import datetime
from typing import List, Optional, Tuple

from tqdm import tqdm

from farm.agents import IndependentAgent, SystemAgent
from farm.agents.control_agent import ControlAgent
from farm.core.config import SimulationConfig
from farm.core.environment import Environment
from farm.utils.short_id import generate_simulation_id


def setup_logging(log_dir: str = "logs") -> None:
    """
    Configure the logging system for the simulation.

    Parameters
    ----------
    log_dir : str
        Directory to store log files
    """
    # Create absolute path for log directory
    log_dir = os.path.abspath(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"simulation.log")

    # Add more detailed logging format and ensure file is writable
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w")],
        force=True,  # This ensures logging config is reset
    )

    # Test log file creation
    logging.info(f"Logging initialized. Log file: {log_file}")


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
    logging.info(
        f"Creating initial agents - System: {num_system_agents}, Independent: {num_independent_agents}, Control: {num_control_agents}"
    )

    # Create system agents
    for _ in range(num_system_agents):
        position = (
            random.uniform(0, environment.width),
            random.uniform(0, environment.height),
        )
        agent = SystemAgent(
            agent_id=environment.get_next_agent_id(),
            position=position,
            resource_level=1,
            environment=environment,
            generation=0,
        )
        environment.add_agent(agent)
        positions.append(position)

    logging.info(f"Created {num_system_agents} SystemAgents")

    # Create independent agents
    for _ in range(num_independent_agents):
        position = (
            random.uniform(0, environment.width),
            random.uniform(0, environment.height),
        )
        agent = IndependentAgent(
            agent_id=environment.get_next_agent_id(),
            position=position,
            resource_level=1,
            environment=environment,
            generation=0,
        )
        environment.add_agent(agent)
        positions.append(position)

    logging.info(f"Created {num_independent_agents} IndependentAgents")

    # Create control agents
    for _ in range(num_control_agents):
        position = (
            random.uniform(0, environment.width),
            random.uniform(0, environment.height),
        )
        agent = ControlAgent(
            agent_id=environment.get_next_agent_id(),
            position=position,
            resource_level=1,
            environment=environment,
            generation=0,
        )
        environment.add_agent(agent)
        positions.append(position)

    logging.info(f"Created {num_control_agents} ControlAgents")
    logging.info(f"Total initial agents: {len(environment.agents)}")

    return positions


def run_simulation(
    num_steps: int,
    config: SimulationConfig,
    path: Optional[str] = None,
    save_config: bool = True,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
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

    Returns
    -------
    Environment
        The simulation environment after completion
    """
    # Generate simulation_id if not provided
    if simulation_id is None:
        simulation_id = generate_simulation_id()

    # Setup logging
    setup_logging()
    logging.info(f"Starting simulation with ID: {simulation_id}")
    logging.info(f"Configuration: {config}")

    # Initialize start_time here so it's available in both code paths
    start_time = datetime.now()

    try:
        # Set up database path (None if path is None)
        db_path = f"{path}/simulation.db" if path is not None else None

        # Ensure the database directory exists
        if db_path is not None:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Handle in-memory database configuration
        if config.use_in_memory_db:
            logging.info("Using in-memory database for improved performance")
            from farm.database.database import InMemorySimulationDatabase

            # Create environment with in-memory database
            environment = Environment(
                width=config.width,
                height=config.height,
                resource_distribution={
                    "type": "random",
                    "amount": config.initial_resources,
                },
                db_path=None,  # Will be ignored for in-memory DB
                config=config,
                simulation_id=simulation_id,
            )

            # Replace the default database with in-memory database
            if hasattr(environment, "db") and environment.db is not None:
                environment.db.close()

            # Initialize in-memory database with optional memory limit
            environment.db = InMemorySimulationDatabase(
                memory_limit_mb=config.in_memory_db_memory_limit_mb,
                simulation_id=simulation_id,
            )

            # Create a simulation record in the in-memory database
            from farm.database.models import Simulation

            # Add simulation record to the in-memory database
            environment.db.add_simulation_record(
                simulation_id=simulation_id,
                start_time=datetime.now(),
                status="running",
                parameters=config.to_dict(),
            )

        else:
            # Clean up any existing database file for disk-based DB
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except PermissionError:
                    # If can't remove, create unique filename
                    base, ext = os.path.splitext(db_path)
                    db_path = f"{base}_{int(time.time())}{ext}"
                    logging.warning(f"Using alternative database path: {db_path}")

            # Initialize the database schema before creating the Environment
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            from farm.database.models import Base, Simulation

            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Create database engine and initialize tables using SQLAlchemy
            engine = create_engine(
                f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(engine)

            # Create a session and add the simulation record
            Session = sessionmaker(bind=engine)
            session = Session()
            sim_record = Simulation(
                simulation_id=simulation_id,
                start_time=datetime.now(),
                status="running",
                parameters=config.to_dict(),
                simulation_db_path=db_path,
            )
            session.add(sim_record)
            session.commit()
            session.close()

            # Create environment with disk-based database
            environment = Environment(
                width=config.width,
                height=config.height,
                resource_distribution={
                    "type": "random",
                    "amount": config.initial_resources,
                },
                db_path=db_path,
                config=config,
                simulation_id=simulation_id,
            )

        # Set seed if provided
        config.seed = seed

        # Save configuration if requested and path is provided
        if save_config and path is not None:
            config_path = f"{path}/config.json"
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=4)
            logging.info(f"Saved configuration to {config_path}")

        if environment.db is not None:
            environment.db.save_configuration(config.to_dict())

        # Create initial agents
        create_initial_agents(
            environment,
            config.system_agents,
            config.independent_agents,
            config.control_agents,
        )

        # Main simulation loop
        for step in tqdm(range(num_steps), desc="Simulation progress", unit="step"):
            logging.info(f"Starting step {step}/{num_steps}")

            # Process agents in batches
            alive_agents = [agent for agent in environment.agents if agent.alive]

            # Stop if no agents are alive
            if len(alive_agents) < 1:
                logging.info(
                    f"Simulation stopped early at step {step} - no agents remaining"
                )
                break

            batch_size = 32  # Adjust based on your needs

            # Process batches without a nested progress bar
            batch_ranges = list(range(0, len(alive_agents), batch_size))
            for i in batch_ranges:
                batch = alive_agents[i : i + batch_size]

                for agent in batch:
                    agent.act()

            # Update environment once per step
            environment.update()

        # Ensure final state is saved
        environment.update()

        # Force final flush of database buffers
        if environment.db:
            environment.db.logger.flush_all_buffers()

            # Persist in-memory database to disk if configured and db_path is provided
            if config.persist_db_on_completion and db_path is not None:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(
                        os.path.dirname(os.path.abspath(db_path)), exist_ok=True
                    )

                    logging.info(f"Persisting in-memory database to {db_path}")
                    # Persist with selected tables or all tables
                    stats = environment.db.persist_to_disk(
                        db_path=db_path,
                        tables=config.in_memory_tables_to_persist,
                        show_progress=True,
                    )
                    logging.info(
                        f"Database persistence completed: {stats['rows_copied']} rows in {stats['duration']:.2f} seconds"
                    )
                except Exception as e:
                    logging.error(f"Failed to persist in-memory database: {e}")

            # Close database connection
            environment.db.close()

    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}", exc_info=True)
        if "environment" in locals():
            try:
                environment.cleanup()
            except Exception as cleanup_error:
                logging.error(f"Cleanup error: {cleanup_error}")
        raise
    finally:
        if "environment" in locals():
            try:
                if hasattr(environment, "db") and environment.db:
                    environment.db.close()
                environment.cleanup()
            except Exception as e:
                logging.error(f"Final cleanup error: {e}")

    elapsed_time = datetime.now() - start_time
    logging.info(f"Simulation completed in {elapsed_time.total_seconds():.2f} seconds")
    return environment


def main():
    """
    Main entry point for running a simulation directly.
    """
    # Load configuration
    config = SimulationConfig.from_yaml("config.yaml")

    # Run simulation
    run_simulation(num_steps=1000, config=config, save_config=True, path="simulations")


if __name__ == "__main__":
    main()
