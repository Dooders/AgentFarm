"""Test script for the ExperimentDatabase class.

This script demonstrates how to use the ExperimentDatabase class to manage
multiple simulations in a single database file.
"""

import logging
import os
import random
import time
from typing import Dict, List, Tuple

from farm.database.experiment_database import ExperimentDatabase, SimulationContext

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_agent_states(num_agents: int, step: int) -> List[Tuple]:
    """Generate random agent states for testing.

    Parameters
    ----------
    num_agents : int
        Number of agents to generate
    step : int
        Current step number

    Returns
    -------
    List[Tuple]
        List of agent state tuples
    """
    states = []
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        position_x = random.uniform(0, 100)
        position_y = random.uniform(0, 100)
        resource_level = random.uniform(10, 100)
        current_health = random.uniform(50, 100)
        starting_health = 100
        starvation_counter = 10
        is_defending = random.choice([0, 1])
        total_reward = random.uniform(0, 1000)
        age = step

        states.append(
            (
                agent_id,
                position_x,
                position_y,
                resource_level,
                current_health,
                starting_health,
                starvation_counter,
                is_defending,
                total_reward,
                age,
            )
        )

    return states


def generate_resource_states(num_resources: int) -> List[Tuple]:
    """Generate random resource states for testing.

    Parameters
    ----------
    num_resources : int
        Number of resources to generate

    Returns
    -------
    List[Tuple]
        List of resource state tuples
    """
    states = []
    for i in range(num_resources):
        resource_id = i
        amount = random.uniform(10, 100)
        position_x = random.uniform(0, 100)
        position_y = random.uniform(0, 100)

        states.append((resource_id, amount, position_x, position_y))

    return states


def generate_step_metrics(step: int, num_agents: int) -> Dict:
    """Generate random step metrics for testing.

    Parameters
    ----------
    step : int
        Current step number
    num_agents : int
        Number of agents in the simulation

    Returns
    -------
    Dict
        Dictionary of step metrics
    """
    return {
        "total_agents": num_agents,
        "system_agents": int(num_agents * 0.2),
        "independent_agents": int(num_agents * 0.7),
        "control_agents": int(num_agents * 0.1),
        "total_resources": random.uniform(1000, 5000),
        "average_agent_resources": random.uniform(50, 200),
        "births": random.randint(0, 5),
        "deaths": random.randint(0, 3),
        "current_max_generation": step // 10 + 1,
        "resource_efficiency": random.uniform(0.5, 0.9),
        "resource_distribution_entropy": random.uniform(0.1, 0.9),
        "average_agent_health": random.uniform(50, 100),
        "average_agent_age": step // 2,
        "average_reward": random.uniform(100, 500),
        "combat_encounters": random.randint(0, 10),
        "successful_attacks": random.randint(0, 5),
        "resources_shared": random.uniform(0, 100),
        "resources_shared_this_step": random.uniform(0, 10),
        "combat_encounters_this_step": random.randint(0, 2),
        "successful_attacks_this_step": random.randint(0, 1),
        "genetic_diversity": random.uniform(0.1, 0.9),
        "dominant_genome_ratio": random.uniform(0.1, 0.9),
        "resources_consumed": random.uniform(10, 100),
    }


def log_agent_actions(
    sim_context: SimulationContext, num_actions: int, step: int
) -> None:
    """Log random agent actions for testing.

    Parameters
    ----------
    sim_context : SimulationContext
        Simulation context
    num_actions : int
        Number of actions to log
    step : int
        Current step number
    """
    action_types = ["move", "eat", "attack", "defend", "share"]

    for _ in range(num_actions):
        agent_id = f"agent_{random.randint(0, 9)}"
        action_type = random.choice(action_types)

        # Sometimes include a target
        target_id = None
        if action_type in ["attack", "share"]:
            target_id = f"agent_{random.randint(0, 9)}"

        resources_before = random.uniform(20, 100)
        resources_after = resources_before + random.uniform(-20, 20)

        sim_context.log_agent_action(
            step_number=step,
            agent_id=agent_id,
            action_type=action_type,
            action_target_id=target_id,
            resources_before=resources_before,
            resources_after=resources_after,
            reward=random.uniform(-10, 10),
            details={"success": random.choice([True, False])},
        )


def run_simulation(
    sim_context: SimulationContext,
    num_steps: int,
    num_agents: int,
    num_resources: int,
    sim_id: int,
) -> None:
    """Run a test simulation.

    Parameters
    ----------
    sim_context : SimulationContext
        Simulation context
    num_steps : int
        Number of steps to run
    num_agents : int
        Number of agents to create
    num_resources : int
        Number of resources to create
    sim_id : int
        Simulation ID number (used for step offset)
    """
    # Create agents
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        sim_context.log_agent(
            agent_id=agent_id,
            birth_time=0,
            agent_type=random.choice(["system", "independent", "control"]),
            position=(random.uniform(0, 100), random.uniform(0, 100)),
            initial_resources=random.uniform(50, 100),
            starting_health=100,
            starvation_counter=10,
            genome_id=f"genome_{random.randint(0, 5)}",
            generation=0,
            action_weights={
                "move": 0.3,
                "eat": 0.3,
                "attack": 0.2,
                "defend": 0.1,
                "share": 0.1,
            },
        )

    # Run steps
    for step in range(num_steps):
        # Use unique step numbers for each simulation by adding an offset
        # This ensures there are no primary key conflicts
        unique_step = step + (sim_id * 1000)

        # Generate random data
        agent_states = generate_agent_states(num_agents, step)
        resource_states = generate_resource_states(num_resources)
        metrics = generate_step_metrics(step, num_agents)

        # Log step data with the unique step number
        sim_context.log_step(unique_step, agent_states, resource_states, metrics)

        # Log some random actions
        log_agent_actions(sim_context, 5, unique_step)

        # Log a reproduction event occasionally
        if random.random() < 0.2:
            parent_id = f"agent_{random.randint(0, num_agents - 1)}"
            success = random.random() < 0.7

            offspring_id = None
            offspring_gen = None
            offspring_resources = None
            failure_reason = None

            if success:
                offspring_id = f"agent_{num_agents}"
                num_agents += 1
                offspring_gen = step // 10 + 1
                offspring_resources = random.uniform(20, 40)
            else:
                failure_reason = "insufficient_resources"

            sim_context.log_reproduction_event(
                step_number=unique_step,
                parent_id=parent_id,
                offspring_id=offspring_id,
                success=success,
                parent_resources_before=random.uniform(60, 100),
                parent_resources_after=random.uniform(20, 60),
                offspring_initial_resources=offspring_resources,
                failure_reason=failure_reason,
                parent_generation=step // 10,
                offspring_generation=offspring_gen,
                parent_position=(random.uniform(0, 100), random.uniform(0, 100)),
            )

        # Log a health incident occasionally
        if random.random() < 0.15:
            agent_id = f"agent_{random.randint(0, num_agents - 1)}"
            health_before = random.uniform(40, 100)
            health_change = random.uniform(-30, 10)

            sim_context.log_health_incident(
                step_number=unique_step,
                agent_id=agent_id,
                health_before=health_before,
                health_after=max(0, health_before + health_change),
                cause=random.choice(["starvation", "combat", "disease"]),
                details={"severity": "medium" if health_change < -10 else "low"},
            )


def main():
    """Run the experiment database test."""
    logger.info("Starting experiment database test")

    # Clean up any existing test database
    db_path = "test_experiment.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Removed existing database at {db_path}")

    # Create the experiment database
    experiment_id = f"test_experiment_{int(time.time())}"
    logger.info(f"Creating experiment database with ID: {experiment_id}")
    try:
        experiment_db = ExperimentDatabase(db_path, experiment_id)
        logger.info("Successfully created experiment database")
    except Exception as e:
        logger.error(f"Failed to create experiment database: {e}")
        raise

    # Run multiple simulations
    for sim_id in range(3):
        # Create a simulation context
        simulation_id = f"simulation_{sim_id}"
        parameters = {
            "num_agents": 10,
            "num_resources": 20,
            "num_steps": 5,
            "seed": 42 + sim_id,
        }

        logger.info(f"Running simulation {simulation_id}...")
        try:
            sim_context = experiment_db.create_simulation_context(
                simulation_id, parameters
            )
            logger.info(f"Successfully created simulation context for {simulation_id}")
        except Exception as e:
            logger.error(f"Failed to create simulation context: {e}")
            continue

        # Run the simulation
        try:
            run_simulation(
                sim_context,
                num_steps=parameters["num_steps"],
                num_agents=parameters["num_agents"],
                num_resources=parameters["num_resources"],
                sim_id=sim_id,  # Pass sim_id for step offset
            )
            logger.info(f"Successfully ran simulation {simulation_id}")
        except Exception as e:
            logger.error(f"Error during simulation run: {e}")
            continue

        # Update simulation status
        try:
            experiment_db.update_simulation_status(
                simulation_id,
                "completed",
                results_summary={
                    "total_agents_final": 10 + random.randint(0, 5),
                    "average_reward_final": random.uniform(200, 800),
                },
            )
            logger.info(f"Updated status for simulation {simulation_id}")
        except Exception as e:
            logger.error(f"Failed to update simulation status: {e}")

    # Update experiment status
    try:
        experiment_db.update_experiment_status(
            "completed", results_summary={"num_simulations": 3, "average_runtime": 0.5}
        )
        logger.info("Updated experiment status to completed")
    except Exception as e:
        logger.error(f"Failed to update experiment status: {e}")

    # Get simulation IDs
    try:
        sim_ids = experiment_db.get_simulation_ids()
        logger.info(f"Simulations in experiment: {sim_ids}")
    except Exception as e:
        logger.error(f"Failed to get simulation IDs: {e}")

    # Close the database
    try:
        experiment_db.close()
        logger.info("Successfully closed the database")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    logger.info(f"Test completed. Database saved to {db_path}")


if __name__ == "__main__":
    main()
