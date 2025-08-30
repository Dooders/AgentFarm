#!/usr/bin/env python3
"""
Production Demo for AgentFarm MCP Server

Demonstrates the complete production MCP server capabilities
with realistic research workflows.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .server import AgentFarmMCPServer
from .config import MCPServerConfig
from .utils import extract_simulation_id, extract_experiment_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-demo")

class ProductionDemo:
    """Production demo showcasing MCP server capabilities."""
    
    def __init__(self):
        """Initialize demo with production configuration."""
        self.config = MCPServerConfig()
        self.config.log_level = "INFO"
        self.server = None
    
    async def setup(self):
        """Setup demo environment."""
        logger.info("🔧 Setting up production demo environment...")
        
        try:
            self.server = AgentFarmMCPServer(self.config)
            logger.info("✅ MCP server initialized")
            
            # Verify server health
            server_info = self.server.get_server_info()
            logger.info(f"   Server: {server_info['server_name']}")
            logger.info(f"   Tools: {server_info['tools_available']}")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
    
    def call_tool(self, tool_name: str, **kwargs) -> str:
        """Call server tool with error handling."""
        try:
            tool_func = self.server.mcp._tools[tool_name].func
            result = tool_func(**kwargs)
            logger.info(f"✅ {tool_name} completed")
            return result
        except Exception as e:
            logger.error(f"❌ {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})
    
    async def demo_basic_simulation(self):
        """Demo: Basic simulation creation and analysis."""
        logger.info("\n🎯 Demo 1: Basic Simulation")
        logger.info("-" * 30)
        
        # Create simulation with balanced configuration
        config = {
            "width": 80,
            "height": 80,
            "system_agents": 12,
            "independent_agents": 12,
            "control_agents": 6,
            "simulation_steps": 300,
            "initial_resources": 20,
            "seed": 12345
        }
        
        logger.info("   Creating balanced agent simulation...")
        result = self.call_tool("create_simulation", config=config, output_path="demo/basic_sim")
        
        # Extract simulation info
        sim_data = json.loads(result) if result.startswith("{") else None
        if sim_data and "simulation_id" in sim_data:
            sim_id = sim_data["simulation_id"]
            db_path = sim_data["db_path"]
            
            logger.info(f"   ✅ Simulation {sim_id} completed")
            logger.info(f"   📊 Final agents: {sim_data['final_metrics']['total_agents']}")
            
            # Get detailed summary
            logger.info("   Getting detailed summary...")
            summary = self.call_tool("get_simulation_summary", simulation_path=db_path)
            
            # Export data
            logger.info("   Exporting simulation data...")
            export_result = self.call_tool(
                "export_simulation_data",
                simulation_path=db_path,
                output_path="demo/basic_sim_export",
                format_type="csv"
            )
            
        else:
            logger.error(f"   ❌ Simulation creation failed: {result}")
    
    async def demo_parameter_study(self):
        """Demo: Multi-parameter experiment."""
        logger.info("\n🧪 Demo 2: Parameter Study Experiment")
        logger.info("-" * 40)
        
        # Create experiment studying agent composition effects
        base_config = {
            "width": 70,
            "height": 70,
            "simulation_steps": 200,  # Short for demo
            "initial_resources": 18,
            "seed": 54321
        }
        
        # Test different agent compositions
        variations = [
            {"system_agents": 15, "independent_agents": 5, "control_agents": 5},   # System-heavy
            {"system_agents": 8, "independent_agents": 12, "control_agents": 5},  # Independent-heavy
            {"system_agents": 7, "independent_agents": 7, "control_agents": 11},  # Control-heavy
            {"system_agents": 8, "independent_agents": 8, "control_agents": 8}    # Balanced
        ]
        
        logger.info("   Creating parameter study experiment...")
        exp_result = self.call_tool(
            "create_experiment",
            name="agent_composition_study",
            base_config=base_config,
            variations=variations,
            num_iterations=4,
            description="Study of agent composition effects on population dynamics"
        )
        
        # Extract experiment info
        exp_data = json.loads(exp_result) if exp_result.startswith("{") else None
        if exp_data and "experiment_id" in exp_data:
            exp_id = exp_data["experiment_id"]
            
            logger.info(f"   ✅ Experiment {exp_id} created")
            logger.info(f"   📋 Variations: {exp_data['variations_count']}")
            
            # Run the experiment
            logger.info("   Running experiment...")
            run_result = self.call_tool("run_experiment", experiment_id=exp_id, run_analysis=True)
            
            run_data = json.loads(run_result) if run_result.startswith("{") else None
            if run_data and run_data.get("status") == "completed":
                logger.info(f"   ✅ Experiment completed")
                logger.info(f"   📁 Output: {run_data['output_dir']}")
                
                # Run batch analysis
                logger.info("   Running batch analysis...")
                analysis_result = self.call_tool(
                    "batch_analyze",
                    experiment_path=run_data["output_dir"],
                    analysis_modules=["dominance", "advantage"]
                )
                
            else:
                logger.error(f"   ❌ Experiment run failed: {run_result}")
        else:
            logger.error(f"   ❌ Experiment creation failed: {exp_result}")
    
    async def demo_research_project(self):
        """Demo: Complete research project workflow."""
        logger.info("\n🔬 Demo 3: Research Project Workflow")
        logger.info("-" * 40)
        
        # Create research project
        logger.info("   Creating research project...")
        project_result = self.call_tool(
            "create_research_project",
            name="cooperation_emergence_study",
            description="Comprehensive study of cooperative behavior emergence in multi-agent systems",
            tags=["cooperation", "emergence", "social_behavior", "competition"]
        )
        
        project_data = json.loads(project_result) if project_result.startswith("{") else None
        if project_data:
            logger.info(f"   ✅ Research project created")
            logger.info(f"   📁 Path: {project_data['project_path']}")
            
            # Create focused simulation for research
            research_config = {
                "width": 100,
                "height": 100,
                "system_agents": 18,
                "independent_agents": 12,
                "control_agents": 6,
                "simulation_steps": 400,
                "initial_resources": 25,
                "share_range": 35.0,
                "cooperation_memory": 100,
                "seed": 98765
            }
            
            logger.info("   Running research simulation...")
            sim_result = self.call_tool(
                "create_simulation", 
                config=research_config,
                output_path="demo/research_sim"
            )
            
            sim_data = json.loads(sim_result) if sim_result.startswith("{") else None
            if sim_data and "db_path" in sim_data:
                logger.info("   ✅ Research simulation completed")
                
                # Comprehensive analysis
                logger.info("   Running comprehensive analysis...")
                analysis_result = self.call_tool(
                    "analyze_simulation",
                    simulation_path=sim_data["db_path"],
                    analysis_types=["dominance", "advantage"],
                    output_path="demo/research_analysis"
                )
                
        else:
            logger.error(f"   ❌ Research project creation failed: {project_result}")
    
    async def demo_data_management(self):
        """Demo: Data management and export capabilities."""
        logger.info("\n📊 Demo 4: Data Management")
        logger.info("-" * 30)
        
        # List all simulations
        logger.info("   Listing all simulations...")
        list_result = self.call_tool("list_simulations")
        
        list_data = json.loads(list_result) if list_result.startswith("{") else None
        if list_data:
            total_sims = list_data.get("total_found", 0)
            logger.info(f"   📋 Found {total_sims} simulations")
            
            # Show breakdown by type
            sim_types = {}
            for sim in list_data.get("simulations", []):
                sim_type = sim.get("type", "unknown")
                sim_types[sim_type] = sim_types.get(sim_type, 0) + 1
            
            for sim_type, count in sim_types.items():
                logger.info(f"      {sim_type}: {count}")
        
        logger.info("   ✅ Data management demo completed")
    
    async def run_all_demos(self):
        """Run all demonstration workflows."""
        logger.info("🚀 AgentFarm MCP Server - Production Demo")
        logger.info("=" * 50)
        
        await self.setup()
        
        demos = [
            self.demo_basic_simulation,
            self.demo_parameter_study,
            self.demo_research_project,
            self.demo_data_management
        ]
        
        for demo in demos:
            try:
                await demo()
            except Exception as e:
                logger.error(f"❌ Demo failed: {e}")
        
        logger.info("\n🎉 Production demo completed!")
        logger.info("\n📖 Next steps:")
        logger.info("   - Start production server: python -m mcp.main")
        logger.info("   - Connect LLM client to server")
        logger.info("   - Use tools for simulation research")

async def main():
    """Main demo entry point."""
    demo = ProductionDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)