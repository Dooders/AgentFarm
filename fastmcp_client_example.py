#!/usr/bin/env python3
"""
Example FastMCP Client for AgentFarm Simulation System

This demonstrates how to interact with the FastMCP AgentFarm server.
"""

import asyncio
import json
import logging
from pathlib import Path

# For testing, we'll import the server directly
# In production, you'd connect via MCP protocol
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import the FastMCP server
from fastmcp_simulation_server import mcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastmcp-client")

async def main():
    """Example client demonstrating FastMCP server usage."""
    
    logger.info("=== AgentFarm FastMCP Client Example ===")
    
    # Example 1: Create a simple simulation
    logger.info("\n1. Creating a simple simulation...")
    
    config = {
        "width": 80,
        "height": 80,
        "system_agents": 12,
        "independent_agents": 8,
        "control_agents": 5,
        "simulation_steps": 400,
        "initial_resources": 20,
        "seed": 42
    }
    
    # Note: In a real MCP client, you'd call this through the MCP protocol
    # This is a direct function call for demonstration
    simulation_result = mcp._tools["create_simulation"].func(config, "examples/demo_sim")
    logger.info(f"Simulation result: {simulation_result}")
    
    # Parse the result to get simulation ID
    try:
        result_data = json.loads(simulation_result)
        sim_id = result_data.get("simulation_id")
        sim_path = result_data.get("db_path")
    except:
        logger.error("Failed to parse simulation result")
        return
    
    # Example 2: Get simulation status
    logger.info(f"\n2. Getting status for simulation {sim_id}...")
    
    status_result = mcp._tools["get_simulation_status"].func(sim_id)
    logger.info(f"Status result: {status_result}")
    
    # Example 3: Get detailed summary
    if sim_path:
        logger.info(f"\n3. Getting detailed summary...")
        
        summary_result = mcp._tools["get_simulation_summary"].func(sim_path)
        logger.info(f"Summary result: {summary_result}")
    
    # Example 4: Export simulation data
    if sim_path:
        logger.info(f"\n4. Exporting simulation data...")
        
        export_result = mcp._tools["export_simulation_data"].func(
            sim_path, 
            "examples/demo_export", 
            "csv", 
            ["agents", "steps"]
        )
        logger.info(f"Export result: {export_result}")
    
    # Example 5: List all simulations
    logger.info("\n5. Listing all simulations...")
    
    list_result = mcp._tools["list_simulations"].func()
    logger.info(f"List result: {list_result}")
    
    # Example 6: Create an experiment
    logger.info("\n6. Creating an experiment...")
    
    base_config = {
        "width": 60,
        "height": 60,
        "simulation_steps": 200,
        "initial_resources": 15
    }
    
    variations = [
        {"system_agents": 15, "independent_agents": 5, "control_agents": 5},
        {"system_agents": 8, "independent_agents": 12, "control_agents": 5},
        {"system_agents": 5, "independent_agents": 5, "control_agents": 15}
    ]
    
    experiment_result = mcp._tools["create_experiment"].func(
        name="agent_ratio_test",
        base_config=base_config,
        variations=variations,
        num_iterations=3,
        description="Testing different agent ratios"
    )
    logger.info(f"Experiment result: {experiment_result}")
    
    logger.info("\n=== FastMCP Client Example Complete ===")

if __name__ == "__main__":
    asyncio.run(main())