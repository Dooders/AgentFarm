#!/usr/bin/env python3
"""
Native Hydra entry point for Agent Farm simulation.

This script uses Hydra's @hydra.main() decorator for full Hydra functionality,
including native multi-run and sweep support.

Usage:
    # Single run
    python run_simulation_hydra.py environment=production profile=benchmark
    
    # Multi-run (grid search)
    python run_simulation_hydra.py -m \
        learning.learning_rate=0.0001,0.0005,0.001 \
        learning.gamma=0.9,0.99
    
    # Sweep from config file
    python run_simulation_hydra.py --config-path=conf/sweeps --config-name=learning_rate_sweep -m
    
    # Override during sweep
    python run_simulation_hydra.py --config-path=conf/sweeps --config-name=learning_rate_sweep -m \
        simulation_steps=200
"""

import os
import sys
import time
import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.utils.logging import configure_logging, get_logger

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tianshou")

# CRITICAL: Ensure PYTHONHASHSEED is set for deterministic simulations
if "PYTHONHASHSEED" not in os.environ or os.environ["PYTHONHASHSEED"] != "0":
    print("??  PYTHONHASHSEED not set - restarting with PYTHONHASHSEED=0 for determinism...", flush=True)
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point using Hydra's decorator.
    
    This function is called by Hydra, which handles:
    - Config loading and composition
    - Multi-run coordination
    - Output directory management
    - Config saving
    
    Args:
        cfg: OmegaConf DictConfig from Hydra
    """
    # Determine output directory from Hydra's output
    # Hydra creates outputs/YYYY-MM-DD/HH-MM-SS/ directory
    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Get environment and profile from config
    environment = cfg.get("environment", "development")
    profile = cfg.get("profile", None)
    
    # Configure logging
    configure_logging(
        environment=environment,
        log_dir=os.path.join(output_dir, "logs"),
        log_level="INFO",
        json_logs=False,
        enable_colors=True,
        disable_console=False,
    )
    logger = get_logger(__name__)
    
    # Convert Hydra config to SimulationConfig
    try:
        config = SimulationConfig.from_hydra(cfg)
        logger.info(
            "configuration_loaded_with_hydra",
            environment=environment,
            profile=profile or "none",
            output_dir=output_dir,
        )
    except Exception as e:
        logger.error(
            "configuration_load_failed",
            error_type=type(e).__name__,
            error_message=str(e),
            exc_info=True,
        )
        sys.exit(1)
    
    # Determine simulation steps
    num_steps = config.simulation_steps
    
    logger.info(
        "simulation_starting",
        num_steps=num_steps,
        output_dir=output_dir,
        system_agents=config.population.system_agents,
        independent_agents=config.population.independent_agents,
        control_agents=config.population.control_agents,
        environment_width=config.environment.width,
        environment_height=config.environment.height,
    )
    
    # Run simulation
    try:
        start_time = time.time()
        
        environment = run_simulation(
            num_steps=num_steps,
            config=config,
            path=output_dir,
            save_config=True,
            disable_console_logging=False,
        )
        
        elapsed_time = time.time() - start_time
        
        # Log completion
        logger.info(
            "simulation_completed",
            duration_seconds=round(elapsed_time, 2),
            final_agent_count=len(environment.agents),
            output_dir=output_dir,
        )
        
        # Print summary
        print(f"\n=== SIMULATION COMPLETED ===", flush=True)
        print(f"Duration: {elapsed_time:.2f} seconds", flush=True)
        print(f"Final agents: {len(environment.agents)}", flush=True)
        print(f"Output: {output_dir}", flush=True)
        
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
