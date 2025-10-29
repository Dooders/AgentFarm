#!/usr/bin/env python3
"""
Run an ant colony simulation with biologically accurate parameters.

This script uses the ant colony preset configuration to run a simulation
with realistic ant metabolism, behavior, and colony dynamics.
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from farm.core.simulation import run_simulation
from farm.config.config import SimulationConfig
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def load_ant_colony_config(config_path: str = "ant_colony_preset.json") -> SimulationConfig:
    """Load the ant colony configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create SimulationConfig from the loaded data
        sim_config = SimulationConfig.from_dict(config_data)
        
        logger.info("Ant colony configuration loaded successfully")
        logger.info(f"  - Simulation steps: {sim_config.simulation_steps}")
        logger.info(f"  - Environment size: {sim_config.environment.width}x{sim_config.environment.height}")
        logger.info(f"  - Population: {sim_config.population.system_agents} system + {sim_config.population.independent_agents} independent")
        logger.info(f"  - Base consumption rate: {sim_config.agent_behavior.base_consumption_rate}")
        logger.info(f"  - Starvation threshold: {sim_config.agent_behavior.starvation_threshold}")
        logger.info(f"  - Initial resources: {sim_config.agent_behavior.initial_resource_level}")
        
        return sim_config
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please ensure ant_colony_preset.json exists in the project root")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def main():
    """Run the ant colony simulation."""
    logger.info("Starting Ant Colony Simulation")
    logger.info("=" * 50)
    
    # Load configuration
    config = load_ant_colony_config()
    
    # Generate simulation ID
    import uuid
    simulation_id = f"ant_colony_{uuid.uuid4().hex[:8]}"
    
    # Run simulation
    try:
        logger.info(f"Starting simulation with ID: {simulation_id}")
        
        environment = run_simulation(
            num_steps=config.simulation_steps,
            config=config,
            path=f"simulations/ant_colony_{simulation_id}.db",
            save_config=True,
            seed=config.seed,
            simulation_id=simulation_id,
            disable_console_logging=False
        )
        
        logger.info("Simulation completed successfully!")
        logger.info(f"Database saved to: simulations/ant_colony_{simulation_id}.db")
        
        # Print final statistics
        if hasattr(environment, 'metrics_tracker'):
            metrics = environment.metrics_tracker
            logger.info("Final Statistics:")
            logger.info(f"  - Total agents: {len(environment.agents)}")
            logger.info(f"  - Total births: {metrics.cumulative_metrics.total_births}")
            logger.info(f"  - Total deaths: {metrics.cumulative_metrics.total_deaths}")
            logger.info(f"  - Total combat encounters: {metrics.cumulative_metrics.total_combat_encounters}")
            logger.info(f"  - Total successful attacks: {metrics.cumulative_metrics.total_successful_attacks}")
            logger.info(f"  - Total resources shared: {metrics.cumulative_metrics.total_resources_shared:.2f}")
            logger.info(f"  - Total reproduction attempts: {metrics.cumulative_metrics.total_reproduction_attempts}")
            logger.info(f"  - Total reproduction successes: {metrics.cumulative_metrics.total_reproduction_successes}")
            logger.info(f"  - Total resource consumption: {metrics.cumulative_metrics.total_resource_consumption:.2f}")
            
            # Calculate average agent resources
            if len(environment.agents) > 0:
                total_resources = sum(agent.resource_level for agent in environment._agent_objects.values())
                avg_resources = total_resources / len(environment.agents)
                logger.info(f"  - Average agent resources: {avg_resources:.2f}")
            else:
                logger.info("  - Average agent resources: N/A (no agents)")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
