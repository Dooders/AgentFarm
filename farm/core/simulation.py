import logging
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

import yaml

from agents import IndependentAgent, SystemAgent
from core.config import SimulationConfig
from core.environment import Environment


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
    environment: Environment, num_system_agents: int, num_independent_agents: int
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
            resource_level=10,
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
            resource_level=10,
            environment=environment,
            generation=0,
        )
        environment.add_agent(agent)
        positions.append(position)

    return positions


def run_simulation(
    num_steps: int,
    config: SimulationConfig,
    db_path: Optional[str] = None,
    save_config: bool = False,
) -> Environment:
    """
    Run the main simulation loop.

    Parameters
    ----------
    num_steps : int
        Number of simulation steps to run
    config : SimulationConfig
        Configuration object containing simulation parameters
    db_path : Optional[str]
        Path to database file for storing results
    save_config : bool
        If True, saves the configuration to a timestamped YAML file
    """
    # Setup logging
    setup_logging()
    logging.info("Starting simulation")
    logging.info(f"Configuration: {config}")

    try:
        # Create environment
        environment = Environment(
            width=config.width,
            height=config.height,
            resource_distribution={
                "type": "random",
                "amount": config.initial_resources,
            },
            db_path=db_path or "simulation_results.db",
            config=config,
        )

        # Save configuration if requested
        if save_config and environment.db is not None:
            environment.db.save_configuration(config.to_dict())

        # Create initial agents
        create_initial_agents(
            environment, config.system_agents, config.independent_agents
        )

        # Main simulation loop
        start_time = datetime.now()
        for step in range(num_steps):
            logging.info(f"Starting step {step}/{num_steps}")

            # Process agents in batches
            alive_agents = [agent for agent in environment.agents if agent.alive]
            batch_size = 32  # Adjust based on your needs

            for i in range(0, len(alive_agents), batch_size):
                batch = alive_agents[i : i + batch_size]

                for agent in batch:
                    agent.act()

                # Process reproduction in parallel
                for agent in batch:
                    agent.reproduce()

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
            environment.cleanup()
        raise
    finally:
        if "environment" in locals():
            environment.cleanup()

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
    run_simulation(num_steps=1000, config=config, save_config=True)


if __name__ == "__main__":
    main()
