#!/usr/bin/env python3
"""FastMCP Server for AgentFarm Simulation System"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastmcp import FastMCP

# Add farm to path
sys.path.insert(0, str(Path(__file__).parent))

from farm.core.simulation import run_simulation
from farm.core.config import SimulationConfig
from farm.database.database import SimulationDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastmcp")

# Initialize server
mcp = FastMCP("AgentFarm Simulation Server")

# Global state
simulations = {}

@mcp.tool()
def create_simulation(config: dict, output_path: Optional[str] = None) -> str:
    """Create and run a simulation with specified parameters."""
    sim_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Create config
        sim_config = SimulationConfig()
        for key, value in config.items():
            if hasattr(sim_config, key):
                setattr(sim_config, key, value)
        
        # Set paths
        if not output_path:
            output_path = f"simulations/sim_{sim_id}"
        os.makedirs(output_path, exist_ok=True)
        db_path = os.path.join(output_path, "simulation.db")
        
        # Run simulation
        logger.info(f"Starting simulation {sim_id}")
        environment = run_simulation(
            num_steps=sim_config.simulation_steps,
            config=sim_config,
            path=db_path,
            simulation_id=sim_id
        )
        
        # Store info
        simulations[sim_id] = {
            "db_path": db_path,
            "config": config,
            "status": "completed"
        }
        
        return json.dumps({
            "simulation_id": sim_id,
            "status": "completed",
            "output_path": output_path,
            "db_path": db_path
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return f"Failed: {str(e)}"

@mcp.tool() 
def list_simulations() -> str:
    """List all available simulations."""
    return json.dumps({
        "active_simulations": len(simulations),
        "simulations": list(simulations.keys())
    }, indent=2)

@mcp.tool()
def get_simulation_status(simulation_id: str) -> str:
    """Get simulation status and results."""
    if simulation_id not in simulations:
        return f"Simulation {simulation_id} not found"
    
    sim_info = simulations[simulation_id]
    try:
        db = SimulationDatabase(sim_info["db_path"])
        
        # Get final state
        final_step = None
        for step in range(1000, -1, -1):
            try:
                final_state = db.simulation_results(step)
                final_step = step
                break
            except:
                continue
        
        if final_step is not None:
            final_state = db.simulation_results(final_step)
            result = {
                "simulation_id": simulation_id,
                "status": sim_info["status"],
                "final_step": final_step,
                "total_agents": len(final_state.agent_states),
                "total_resources": sum(r[1] for r in final_state.resource_states)
            }
        else:
            result = {"simulation_id": simulation_id, "status": "no_data"}
        
        db.close()
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return f"Error getting status: {str(e)}"

if __name__ == "__main__":
    mcp.run()
