import json
import logging
import os
import random
import time
from datetime import datetime
from typing import List, Optional, Tuple

from farm.agents import IndependentAgent, SystemAgent
from farm.agents.control_agent import ControlAgent
from farm.core.config import SimulationConfig
from farm.core.environment import Environment


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

    #! removed timestamp from log file name
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"simulation.log")

    # Add more detailed logging format and ensure file is writable
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
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

    return positions


def run_simulation(
    num_steps: int,
    config: SimulationConfig,
    path: Optional[str] = None,
    save_config: bool = True,
) -> Environment:
    """
    Run the main simulation loop.
    """
    # Setup logging
    setup_logging()
    logging.info("Starting simulation")
    logging.info(f"Configuration: {config}")

    try:
        # Clean up any existing database file
        db_path = f"{path}/simulation.db"
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                # If can't remove, create unique filename
                base, ext = os.path.splitext(db_path)
                db_path = f"{base}_{int(time.time())}{ext}"
                logging.warning(f"Using alternative database path: {db_path}")

        # Create environment with clean database
        environment = Environment(
            width=config.width,
            height=config.height,
            resource_distribution={
                "type": "random",
                "amount": config.initial_resources,
            },
            db_path=db_path,
            config=config,
        )

        # Save configuration if requested
        if save_config:
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
        start_time = datetime.now()
        for step in range(num_steps):
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

            for i in range(0, len(alive_agents), batch_size):
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
