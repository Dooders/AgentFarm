#!/usr/bin/env python3
"""
Complete Research Workflow Example using AgentFarm MCP Server

This script demonstrates a full research workflow using the MCP server:
1. Create a research project
2. Design and run experiments
3. Analyze results
4. Compare outcomes
5. Export findings

This example investigates how different agent type ratios affect 
population dynamics and cooperative behavior emergence.
"""

import asyncio
import json
import logging
import yaml
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-workflow")

class AgentFarmResearcher:
    """Helper class for conducting research using the MCP server."""
    
    def __init__(self, session: ClientSession):
        self.session = session
        self.results = {}
    
    async def create_research_project(self, name: str, description: str, tags: list = None):
        """Create a new research project."""
        logger.info(f"Creating research project: {name}")
        
        result = await self.session.call_tool("create_research_project", {
            "name": name,
            "description": description,
            "tags": tags or []
        })
        
        return result
    
    async def run_parameter_study(self, study_name: str, base_config: dict, variations: list):
        """Run a parameter study experiment."""
        logger.info(f"Starting parameter study: {study_name}")
        
        # Create experiment
        exp_result = await self.session.call_tool("create_experiment", {
            "name": study_name,
            "description": f"Parameter study: {study_name}",
            "base_config": base_config,
            "variations": variations,
            "num_iterations": len(variations),
            "steps_per_iteration": base_config.get("simulation_steps", 1000)
        })
        
        # Extract experiment ID (simplified extraction)
        experiment_id = None
        for content in exp_result.content:
            if "exp_" in content.text:
                import re
                match = re.search(r"exp_\d+_\d+_\d+", content.text)
                if match:
                    experiment_id = match.group()
                    break
        
        if not experiment_id:
            raise ValueError("Failed to extract experiment ID")
        
        # Run experiment
        run_result = await self.session.call_tool("run_experiment", {
            "experiment_id": experiment_id,
            "run_analysis": True
        })
        
        self.results[study_name] = {
            "experiment_id": experiment_id,
            "creation_result": exp_result,
            "run_result": run_result
        }
        
        return experiment_id, run_result
    
    async def analyze_experiment(self, experiment_path: str, analysis_name: str):
        """Run comprehensive analysis on an experiment."""
        logger.info(f"Analyzing experiment: {analysis_name}")
        
        analysis_result = await self.session.call_tool("analyze_simulation", {
            "simulation_path": experiment_path,
            "analysis_types": ["dominance", "advantage", "social_behavior"],
            "output_path": f"analysis/{analysis_name}"
        })
        
        self.results[f"{analysis_name}_analysis"] = analysis_result
        return analysis_result
    
    async def export_findings(self, simulation_paths: list, export_name: str):
        """Export simulation data for external analysis."""
        logger.info(f"Exporting findings: {export_name}")
        
        export_results = []
        for i, sim_path in enumerate(simulation_paths):
            result = await self.session.call_tool("export_simulation_data", {
                "simulation_path": sim_path,
                "format": "csv",
                "data_types": ["all"],
                "output_path": f"exports/{export_name}/simulation_{i+1}"
            })
            export_results.append(result)
        
        self.results[f"{export_name}_exports"] = export_results
        return export_results

async def cooperative_behavior_study():
    """
    Complete research study: How do different agent compositions affect cooperative behavior?
    
    Research Questions:
    1. Do system agents promote more cooperation than independent agents?
    2. How does agent ratio affect resource sharing patterns?
    3. What role do control agents play in stabilizing populations?
    """
    
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_simulation_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            researcher = AgentFarmResearcher(session)
            
            # Step 1: Create research project
            await researcher.create_research_project(
                name="cooperative_behavior_emergence",
                description="Investigating emergence of cooperative behavior in multi-agent systems with different agent type compositions",
                tags=["cooperation", "emergence", "agent_types", "social_behavior"]
            )
            
            # Step 2: Design base configuration
            base_config = {
                "width": 120,
                "height": 120,
                "simulation_steps": 2000,
                "initial_resources": 25,
                "resource_regen_rate": 0.12,
                "max_resource_amount": 35,
                "share_range": 35.0,
                "cooperation_memory": 200,
                "seed": 12345
            }
            
            # Step 3: Define experimental conditions
            conditions = [
                # High cooperation potential (many system agents)
                {"system_agents": 25, "independent_agents": 5, "control_agents": 5, "condition": "high_cooperation"},
                
                # Balanced composition
                {"system_agents": 12, "independent_agents": 12, "control_agents": 12, "condition": "balanced"},
                
                # High competition (many independent agents)
                {"system_agents": 5, "independent_agents": 25, "control_agents": 5, "condition": "high_competition"},
                
                # Control-dominated (stable environment)
                {"system_agents": 8, "independent_agents": 8, "control_agents": 20, "condition": "control_dominated"},
                
                # Mixed populations
                {"system_agents": 15, "independent_agents": 18, "control_agents": 7, "condition": "system_dominant"},
                {"system_agents": 7, "independent_agents": 15, "control_agents": 18, "condition": "mixed_competitive"}
            ]
            
            # Step 4: Run experiments
            experiment_paths = []
            
            for i, condition in enumerate(conditions):
                condition_name = condition.pop("condition")
                logger.info(f"\n=== Running Condition {i+1}: {condition_name} ===")
                
                experiment_id, run_result = await researcher.run_parameter_study(
                    study_name=f"cooperation_study_{condition_name}",
                    base_config={**base_config, **condition},
                    variations=[condition]  # Single condition per experiment
                )
                
                # Extract experiment path from results
                for content in run_result.content:
                    if "Output directory:" in content.text:
                        import re
                        match = re.search(r"Output directory: (.+)", content.text)
                        if match:
                            experiment_paths.append(match.group(1).strip())
            
            # Step 5: Comprehensive analysis
            logger.info("\n=== Running Comprehensive Analysis ===")
            
            analysis_results = []
            for path in experiment_paths:
                if path:  # Only analyze if we have a valid path
                    result = await researcher.analyze_experiment(path, f"analysis_{Path(path).name}")
                    analysis_results.append(result)
            
            # Step 6: Batch analysis across all experiments
            if experiment_paths:
                logger.info("\n=== Running Batch Analysis ===")
                
                # Find common parent directory
                parent_dir = str(Path(experiment_paths[0]).parent)
                
                batch_result = await session.call_tool("batch_analyze", {
                    "experiment_path": parent_dir,
                    "analysis_modules": ["dominance", "advantage", "social_behavior"],
                    "save_to_db": True,
                    "output_path": f"{parent_dir}/consolidated_analysis"
                })
                
                researcher.results["batch_analysis"] = batch_result
            
            # Step 7: Export key findings
            logger.info("\n=== Exporting Research Data ===")
            
            # Export data from each experiment for external analysis
            if experiment_paths:
                simulation_dbs = []
                for exp_path in experiment_paths:
                    db_path = Path(exp_path) / "iteration_1" / "simulation.db"
                    if db_path.exists():
                        simulation_dbs.append(str(db_path))
                
                if simulation_dbs:
                    await researcher.export_findings(
                        simulation_dbs,
                        "cooperation_behavior_study"
                    )
            
            # Step 8: Generate research summary
            logger.info("\n=== Research Study Complete ===")
            
            summary = {
                "study_name": "Cooperative Behavior Emergence Study",
                "conditions_tested": len(conditions),
                "experiments_completed": len(experiment_paths),
                "analysis_modules_run": ["dominance", "advantage", "social_behavior"],
                "key_findings": {
                    "note": "Detailed findings available in analysis outputs",
                    "experiments": experiment_paths,
                    "analysis_outputs": [f"analysis_{Path(p).name}" for p in experiment_paths if p]
                }
            }
            
            # Save research summary
            summary_file = "research_summary_cooperative_behavior.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Research summary saved to: {summary_file}")
            logger.info(f"Summary:\n{json.dumps(summary, indent=2)}")
            
            return researcher.results

async def quick_simulation_demo():
    """Quick demonstration of basic simulation capabilities."""
    
    server_params = StdioServerParameters(
        command="python", 
        args=["mcp_simulation_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            logger.info("=== Quick Simulation Demo ===")
            
            # Create a simple simulation
            result = await session.call_tool("create_simulation", {
                "config": {
                    "width": 60,
                    "height": 60,
                    "system_agents": 8,
                    "independent_agents": 8,
                    "control_agents": 4,
                    "simulation_steps": 300,
                    "initial_resources": 15,
                    "seed": 999
                },
                "output_path": "demo/quick_sim"
            })
            
            logger.info("Simulation Result:")
            for content in result.content:
                logger.info(content.text)
            
            # Get simulation summary
            summary_result = await session.call_tool("get_simulation_summary", {
                "simulation_path": "demo/quick_sim/simulation.db"
            })
            
            logger.info("Simulation Summary:")
            for content in summary_result.content:
                logger.info(content.text)
            
            # Quick analysis
            analysis_result = await session.call_tool("analyze_simulation", {
                "simulation_path": "demo/quick_sim/simulation.db",
                "analysis_types": ["dominance"],
                "output_path": "demo/quick_analysis"
            })
            
            logger.info("Analysis Result:")
            for content in analysis_result.content:
                logger.info(content.text)

async def main():
    """Main function - choose which demo to run."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        await quick_simulation_demo()
    else:
        await cooperative_behavior_study()

if __name__ == "__main__":
    asyncio.run(main())