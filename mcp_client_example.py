#!/usr/bin/env python3
"""
Example MCP Client for AgentFarm Simulation System

This script demonstrates how an LLM agent would interact with the 
AgentFarm MCP server to create, run, and analyze simulations.
"""

import asyncio
import json
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-client-example")

async def main():
    """Example client demonstrating MCP server usage."""
    
    # Connect to the MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_simulation_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            logger.info(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            # Example 1: Create a simple simulation
            logger.info("\n=== Creating a Simple Simulation ===")
            result = await session.call_tool(
                "create_simulation",
                {
                    "config": {
                        "width": 100,
                        "height": 100,
                        "system_agents": 15,
                        "independent_agents": 10,
                        "control_agents": 5,
                        "simulation_steps": 500,
                        "initial_resources": 25,
                        "seed": 42
                    },
                    "output_path": "examples/simple_sim"
                }
            )
            
            for content in result.content:
                logger.info(f"Simulation Result: {content.text}")
            
            # Example 2: Create an experiment with parameter variations
            logger.info("\n=== Creating an Experiment ===")
            result = await session.call_tool(
                "create_experiment",
                {
                    "name": "agent_ratio_study",
                    "description": "Study the effect of different agent type ratios on population dynamics",
                    "base_config": {
                        "width": 100,
                        "height": 100,
                        "simulation_steps": 1000,
                        "initial_resources": 20
                    },
                    "variations": [
                        {"system_agents": 20, "independent_agents": 5, "control_agents": 5},
                        {"system_agents": 10, "independent_agents": 15, "control_agents": 5},
                        {"system_agents": 5, "independent_agents": 5, "control_agents": 20},
                        {"system_agents": 10, "independent_agents": 10, "control_agents": 10}
                    ],
                    "num_iterations": 4,
                    "steps_per_iteration": 800
                }
            )
            
            experiment_id = None
            for content in result.content:
                logger.info(f"Experiment Creation: {content.text}")
                # Extract experiment ID (in a real implementation, this would be parsed properly)
                if "exp_" in content.text:
                    import re
                    match = re.search(r"exp_\d+_\d+_\d+", content.text)
                    if match:
                        experiment_id = match.group()
            
            # Example 3: Run the experiment
            if experiment_id:
                logger.info(f"\n=== Running Experiment {experiment_id} ===")
                result = await session.call_tool(
                    "run_experiment",
                    {
                        "experiment_id": experiment_id,
                        "run_analysis": True
                    }
                )
                
                for content in result.content:
                    logger.info(f"Experiment Results: {content.text}")
            
            # Example 4: List all simulations
            logger.info("\n=== Listing Available Simulations ===")
            result = await session.call_tool("list_simulations", {})
            
            for content in result.content:
                logger.info(f"Available Simulations: {content.text}")
            
            # Example 5: Analyze a simulation
            logger.info("\n=== Running Analysis ===")
            result = await session.call_tool(
                "analyze_simulation",
                {
                    "simulation_path": "examples/simple_sim/simulation.db",
                    "analysis_types": ["dominance", "advantage"],
                    "output_path": "examples/simple_sim_analysis"
                }
            )
            
            for content in result.content:
                logger.info(f"Analysis Results: {content.text}")
            
            # Example 6: Export simulation data
            logger.info("\n=== Exporting Simulation Data ===")
            result = await session.call_tool(
                "export_simulation_data",
                {
                    "simulation_path": "examples/simple_sim/simulation.db",
                    "format": "csv",
                    "data_types": ["agents", "actions", "steps"],
                    "output_path": "examples/simple_sim_export"
                }
            )
            
            for content in result.content:
                logger.info(f"Export Results: {content.text}")
            
            logger.info("\n=== MCP Client Example Complete ===")

if __name__ == "__main__":
    asyncio.run(main())