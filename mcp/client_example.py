#!/usr/bin/env python3
"""
Production FastMCP Client Example for AgentFarm

Demonstrates how to interact with the production AgentFarm MCP server
using proper error handling and structured workflows.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-client")

class AgentFarmMCPClient:
    """Production client for AgentFarm MCP server interactions."""
    
    def __init__(self, server_path: str = None):
        """Initialize the MCP client.
        
        Args:
            server_path: Path to the MCP server script
        """
        self.server_path = server_path or str(Path(__file__).parent / "main.py")
        self.session = None
    
    async def connect(self):
        """Connect to the MCP server."""
        try:
            # For demonstration, we'll import the server directly
            # In production, you'd use proper MCP client connection
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from mcp.server import AgentFarmMCPServer
            from mcp.config import MCPServerConfig
            
            config = MCPServerConfig()
            self.server = AgentFarmMCPServer(config)
            logger.info("✅ Connected to AgentFarm MCP server")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to server: {e}")
            raise
    
    def call_tool(self, tool_name: str, **kwargs) -> str:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
        
        Returns:
            Tool result as string
        """
        if not hasattr(self, 'server'):
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        if tool_name not in self.server.mcp._tools:
            raise ValueError(f"Tool {tool_name} not available")
        
        try:
            tool_func = self.server.mcp._tools[tool_name].func
            result = tool_func(**kwargs)
            logger.info(f"✅ Tool {tool_name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Tool {tool_name} failed: {e}")
            raise
    
    def parse_json_result(self, result: str) -> Optional[Dict[str, Any]]:
        """Parse JSON result from tool call.
        
        Args:
            result: Result string from tool call
        
        Returns:
            Parsed JSON data or None if parsing fails
        """
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"Result not valid JSON: {result[:100]}...")
            return None

async def run_basic_simulation_demo():
    """Demonstrate basic simulation creation and analysis."""
    logger.info("🎯 Basic Simulation Demo")
    logger.info("-" * 30)
    
    client = AgentFarmMCPClient()
    await client.connect()
    
    # 1. Create a simple simulation
    logger.info("1️⃣ Creating simulation...")
    
    config = {
        "width": 60,
        "height": 60,
        "system_agents": 8,
        "independent_agents": 8,
        "control_agents": 4,
        "simulation_steps": 100,  # Short for demo
        "initial_resources": 15,
        "seed": 42
    }
    
    sim_result = client.call_tool("create_simulation", config=config)
    logger.info(f"Simulation result: {sim_result[:200]}...")
    
    # Parse simulation ID
    sim_data = client.parse_json_result(sim_result)
    if not sim_data or "simulation_id" not in sim_data:
        logger.error("Failed to get simulation ID")
        return
    
    sim_id = sim_data["simulation_id"]
    db_path = sim_data.get("db_path")
    
    # 2. Get simulation status
    logger.info("2️⃣ Getting simulation status...")
    
    status_result = client.call_tool("get_simulation_status", simulation_id=sim_id)
    logger.info(f"Status: {status_result[:200]}...")
    
    # 3. Get detailed summary
    if db_path:
        logger.info("3️⃣ Getting detailed summary...")
        
        summary_result = client.call_tool("get_simulation_summary", simulation_path=db_path)
        logger.info(f"Summary: {summary_result[:200]}...")
    
    # 4. List all simulations
    logger.info("4️⃣ Listing simulations...")
    
    list_result = client.call_tool("list_simulations")
    logger.info(f"Simulations: {list_result[:200]}...")
    
    # 5. Export data
    if db_path:
        logger.info("5️⃣ Exporting simulation data...")
        
        export_result = client.call_tool(
            "export_simulation_data",
            simulation_path=db_path,
            output_path="demo_exports",
            format_type="csv",
            data_types=["agents", "steps"]
        )
        logger.info(f"Export: {export_result[:200]}...")
    
    logger.info("✅ Basic simulation demo completed!")

async def run_experiment_demo():
    """Demonstrate experiment creation and execution."""
    logger.info("\n🧪 Experiment Demo")
    logger.info("-" * 20)
    
    client = AgentFarmMCPClient()
    await client.connect()
    
    # 1. Create experiment
    logger.info("1️⃣ Creating experiment...")
    
    base_config = {
        "width": 50,
        "height": 50,
        "simulation_steps": 50,  # Very short for demo
        "initial_resources": 10
    }
    
    variations = [
        {"system_agents": 10, "independent_agents": 5, "control_agents": 5},
        {"system_agents": 5, "independent_agents": 10, "control_agents": 5},
        {"system_agents": 5, "independent_agents": 5, "control_agents": 10}
    ]
    
    exp_result = client.call_tool(
        "create_experiment",
        name="agent_ratio_demo",
        base_config=base_config,
        variations=variations,
        num_iterations=3,
        description="Demo experiment testing agent ratios"
    )
    logger.info(f"Experiment created: {exp_result[:200]}...")
    
    # Parse experiment ID
    exp_data = client.parse_json_result(exp_result)
    if not exp_data or "experiment_id" not in exp_data:
        logger.error("Failed to get experiment ID")
        return
    
    exp_id = exp_data["experiment_id"]
    
    # 2. Run experiment
    logger.info("2️⃣ Running experiment...")
    
    run_result = client.call_tool("run_experiment", experiment_id=exp_id)
    logger.info(f"Experiment run: {run_result[:200]}...")
    
    logger.info("✅ Experiment demo completed!")

async def run_research_workflow_demo():
    """Demonstrate a complete research workflow."""
    logger.info("\n🔬 Research Workflow Demo")
    logger.info("-" * 30)
    
    client = AgentFarmMCPClient()
    await client.connect()
    
    # 1. Create research project
    logger.info("1️⃣ Creating research project...")
    
    project_result = client.call_tool(
        "create_research_project",
        name="cooperation_emergence_demo",
        description="Investigating emergence of cooperative behavior",
        tags=["cooperation", "emergence", "demo"]
    )
    logger.info(f"Project created: {project_result[:200]}...")
    
    # 2. Run a quick simulation for the research
    logger.info("2️⃣ Running research simulation...")
    
    research_config = {
        "width": 40,
        "height": 40,
        "system_agents": 12,
        "independent_agents": 6,
        "control_agents": 3,
        "simulation_steps": 75,
        "initial_resources": 20,
        "share_range": 30.0,
        "cooperation_memory": 50
    }
    
    sim_result = client.call_tool("create_simulation", config=research_config)
    sim_data = client.parse_json_result(sim_result)
    
    if sim_data and "db_path" in sim_data:
        # 3. Analyze the research simulation
        logger.info("3️⃣ Analyzing research results...")
        
        analysis_result = client.call_tool(
            "analyze_simulation",
            simulation_path=sim_data["db_path"],
            analysis_types=["dominance"]
        )
        logger.info(f"Analysis: {analysis_result[:200]}...")
    
    logger.info("✅ Research workflow demo completed!")

async def main():
    """Main demo function."""
    logger.info("🚀 AgentFarm FastMCP Client Demo")
    logger.info("=" * 50)
    
    try:
        # Run demos
        await run_basic_simulation_demo()
        await run_experiment_demo()
        await run_research_workflow_demo()
        
        logger.info("\n🎉 All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Check if AgentFarm is available
    try:
        import farm
        asyncio.run(main())
    except ImportError:
        logger.error("❌ AgentFarm package not found")
        logger.info("Ensure you're running from the AgentFarm root directory")
        sys.exit(1)