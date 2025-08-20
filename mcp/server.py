"""
Production-ready FastMCP Server for AgentFarm Simulation System.

This module implements the main MCP server with all simulation tools
organized for production use.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastmcp import FastMCP

# Add farm package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.core.simulation import run_simulation
from farm.core.config import SimulationConfig
from farm.controllers.experiment_controller import ExperimentController
from farm.database.database import SimulationDatabase
from farm.analysis.dominance.analyze import process_dominance_data
from farm.analysis.advantage.analyze import analyze_advantages
from farm.research.research import ResearchProject

from .config import MCPServerConfig
from .utils import generate_unique_id, safe_json_loads, format_error_response

# Configure logger
logger = logging.getLogger(__name__)

class AgentFarmMCPServer:
    """Production-ready FastMCP server for AgentFarm simulations."""
    
    def __init__(self, config: Optional[MCPServerConfig] = None):
        """Initialize the MCP server with configuration."""
        self.config = config or MCPServerConfig.from_env()
        self.mcp = FastMCP(self.config.server_name)
        
        # State tracking
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        self.active_experiments: Dict[str, ExperimentController] = {}
        
        # Register all tools
        self._register_tools()
        
        logger.info(f"AgentFarm MCP Server initialized: {self.config.server_name}")
    
    def _register_tools(self) -> None:
        """Register all MCP tools with the server."""
        
        @self.mcp.tool()
        def create_simulation(config: dict, output_path: Optional[str] = None) -> str:
            """Create and run a single simulation with specified parameters.
            
            Args:
                config: Simulation configuration parameters
                output_path: Path to save simulation results
            
            Returns:
                JSON string with simulation results and metadata
            """
            return self._create_simulation(config, output_path)
        
        @self.mcp.tool()
        def create_experiment(
            name: str,
            base_config: dict,
            variations: Optional[List[dict]] = None,
            num_iterations: int = 10,
            steps_per_iteration: int = 1000,
            description: str = ""
        ) -> str:
            """Create a multi-iteration experiment with parameter variations."""
            return self._create_experiment(
                name, base_config, variations, num_iterations, 
                steps_per_iteration, description
            )
        
        @self.mcp.tool()
        def run_experiment(experiment_id: str, run_analysis: bool = True) -> str:
            """Execute a created experiment."""
            return self._run_experiment(experiment_id, run_analysis)
        
        @self.mcp.tool()
        def list_simulations(search_path: Optional[str] = None) -> str:
            """List all available simulations and experiments."""
            return self._list_simulations(search_path)
        
        @self.mcp.tool()
        def get_simulation_status(simulation_id: str) -> str:
            """Get simulation status and key metrics."""
            return self._get_simulation_status(simulation_id)
        
        @self.mcp.tool()
        def get_simulation_summary(simulation_path: str, include_charts: bool = False) -> str:
            """Get detailed summary of simulation results."""
            return self._get_simulation_summary(simulation_path, include_charts)
        
        @self.mcp.tool()
        def analyze_simulation(
            simulation_path: str,
            analysis_types: List[str] = None,
            output_path: Optional[str] = None
        ) -> str:
            """Run comprehensive analysis on simulation results."""
            if analysis_types is None:
                analysis_types = self.config.default_analysis_types
            return self._analyze_simulation(simulation_path, analysis_types, output_path)
        
        @self.mcp.tool()
        def export_simulation_data(
            simulation_path: str,
            output_path: str,
            format_type: str = "csv",
            data_types: List[str] = None
        ) -> str:
            """Export simulation data in specified format."""
            if data_types is None:
                data_types = self.config.default_export_types
            return self._export_simulation_data(simulation_path, output_path, format_type, data_types)
        
        @self.mcp.tool()
        def batch_analyze(
            experiment_path: str,
            analysis_modules: List[str] = None,
            save_to_db: bool = True,
            output_path: Optional[str] = None
        ) -> str:
            """Run batch analysis across multiple simulations."""
            if analysis_modules is None:
                analysis_modules = self.config.default_analysis_types
            return self._batch_analyze(experiment_path, analysis_modules, save_to_db, output_path)
        
        @self.mcp.tool()
        def create_research_project(
            name: str,
            description: str,
            base_path: Optional[str] = None,
            tags: Optional[List[str]] = None
        ) -> str:
            """Create a structured research project."""
            if base_path is None:
                base_path = str(self.config.get_path("research"))
            return self._create_research_project(name, description, base_path, tags)
    
    def _create_simulation(self, config: dict, output_path: Optional[str] = None) -> str:
        """Implementation of create_simulation tool."""
        sim_id = generate_unique_id("sim")
        
        try:
            # Create simulation configuration
            sim_config = SimulationConfig()
            
            # Apply user configuration with validation
            for key, value in config.items():
                if hasattr(sim_config, key):
                    setattr(sim_config, key, value)
                else:
                    logger.warning(f"Unknown config parameter ignored: {key}")
            
            # Setup output directory
            if not output_path:
                output_path = str(self.config.get_path("simulations") / f"sim_{sim_id}")
            
            os.makedirs(output_path, exist_ok=True)
            db_path = os.path.join(output_path, "simulation.db")
            
            # Run the simulation
            logger.info(f"Starting simulation {sim_id} with {sim_config.simulation_steps} steps")
            environment = run_simulation(
                num_steps=sim_config.simulation_steps,
                config=sim_config,
                path=db_path,
                simulation_id=sim_id
            )
            
            # Store simulation info
            self.active_simulations[sim_id] = {
                "db_path": db_path,
                "config": config,
                "created_at": datetime.now().isoformat(),
                "status": "completed",
                "output_path": output_path
            }
            
            # Get final metrics
            db = SimulationDatabase(db_path)
            try:
                final_step = sim_config.simulation_steps - 1
                final_state = db.simulation_results(final_step)
                
                # Count agent types
                agent_types = {}
                for agent_state in final_state.agent_states:
                    agent_type = agent_state[2]
                    agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
                
                result = {
                    "simulation_id": sim_id,
                    "status": "completed",
                    "output_path": output_path,
                    "db_path": db_path,
                    "final_metrics": {
                        "total_agents": len(final_state.agent_states),
                        "agent_types": agent_types,
                        "total_resources": sum(r[1] for r in final_state.resource_states),
                        "steps_completed": sim_config.simulation_steps
                    },
                    "configuration": config
                }
                
            finally:
                db.close()
            
            logger.info(f"Simulation {sim_id} completed successfully")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error in create_simulation: {str(e)}", exc_info=True)
            return format_error_response("create_simulation", str(e))
    
    def _create_experiment(
        self, name: str, base_config: dict, variations: Optional[List[dict]], 
        num_iterations: int, steps_per_iteration: int, description: str
    ) -> str:
        """Implementation of create_experiment tool."""
        try:
            # Create base configuration
            sim_config = SimulationConfig()
            for key, value in base_config.items():
                if hasattr(sim_config, key):
                    setattr(sim_config, key, value)
            
            # Create experiment controller
            experiment_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = self.config.get_path("experiments") / experiment_name
            
            experiment = ExperimentController(
                name=name,
                description=description,
                base_config=sim_config,
                output_dir=output_dir
            )
            
            # Store experiment metadata
            experiment._variations = variations or []
            experiment._num_iterations = num_iterations
            experiment._steps_per_iteration = steps_per_iteration
            
            # Generate experiment ID and store
            exp_id = generate_unique_id("exp")
            self.active_experiments[exp_id] = experiment
            
            result = {
                "experiment_id": exp_id,
                "name": name,
                "description": description,
                "output_dir": str(output_dir),
                "num_iterations": num_iterations,
                "steps_per_iteration": steps_per_iteration,
                "variations_count": len(variations) if variations else 0,
                "status": "created",
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"Experiment {exp_id} created: {name}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}", exc_info=True)
            return format_error_response("create_experiment", str(e))
    
    def _run_experiment(self, experiment_id: str, run_analysis: bool) -> str:
        """Implementation of run_experiment tool."""
        try:
            if experiment_id not in self.active_experiments:
                available = list(self.active_experiments.keys())
                return format_error_response(
                    "run_experiment", 
                    f"Experiment {experiment_id} not found. Available: {available}"
                )
            
            experiment = self.active_experiments[experiment_id]
            
            # Get experiment parameters
            variations = getattr(experiment, '_variations', [])
            num_iterations = getattr(experiment, '_num_iterations', 10)
            steps_per_iteration = getattr(experiment, '_steps_per_iteration', 1000)
            
            logger.info(f"Running experiment {experiment_id}: {experiment.name}")
            
            # Execute the experiment
            experiment.run_experiment(
                num_iterations=num_iterations,
                variations=variations,
                num_steps=steps_per_iteration,
                run_analysis=run_analysis
            )
            
            # Get results
            state = experiment.get_state()
            
            result = {
                "experiment_id": experiment_id,
                "status": "completed",
                "output_dir": str(experiment.output_dir),
                "iterations_completed": state.get("current_iteration", 0),
                "total_iterations": state.get("total_iterations", 0),
                "completed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Experiment {experiment_id} completed successfully")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error running experiment: {str(e)}", exc_info=True)
            return format_error_response("run_experiment", str(e))
    
    def _list_simulations(self, search_path: Optional[str] = None) -> str:
        """Implementation of list_simulations tool."""
        try:
            if search_path is None:
                search_path = str(self.config.get_path("simulations"))
            
            simulations = []
            
            # Active simulations
            for sim_id, sim_info in self.active_simulations.items():
                simulations.append({
                    "id": sim_id,
                    "type": "active",
                    "created_at": sim_info["created_at"],
                    "status": sim_info["status"],
                    "db_path": sim_info["db_path"],
                    "output_path": sim_info.get("output_path")
                })
            
            # File-based simulations
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith(".db") and "simulation" in file:
                            file_path = os.path.join(root, file)
                            stat = os.stat(file_path)
                            simulations.append({
                                "id": os.path.splitext(file)[0],
                                "type": "file",
                                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                "size": stat.st_size,
                                "db_path": file_path
                            })
            
            # Experiment directories
            experiments_path = self.config.get_path("experiments")
            if experiments_path.exists():
                for root, dirs, files in os.walk(experiments_path):
                    if "iteration_1" in dirs:
                        simulations.append({
                            "id": os.path.basename(root),
                            "type": "experiment",
                            "path": root,
                            "iterations": len([d for d in dirs if d.startswith("iteration_")])
                        })
            
            result = {
                "total_found": len(simulations),
                "active_simulations": len(self.active_simulations),
                "active_experiments": len(self.active_experiments),
                "simulations": simulations,
                "search_path": search_path
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error listing simulations: {str(e)}")
            return format_error_response("list_simulations", str(e))
    
    def _get_simulation_status(self, simulation_id: str) -> str:
        """Implementation of get_simulation_status tool."""
        try:
            if simulation_id not in self.active_simulations:
                return format_error_response(
                    "get_simulation_status",
                    f"Simulation {simulation_id} not found in active simulations"
                )
            
            sim_info = self.active_simulations[simulation_id]
            db_path = sim_info["db_path"]
            
            if not os.path.exists(db_path):
                return format_error_response(
                    "get_simulation_status",
                    f"Database not found: {db_path}"
                )
            
            db = SimulationDatabase(db_path)
            
            try:
                # Find the last available step
                final_step = None
                for step in range(2000, -1, -1):
                    try:
                        final_state = db.simulation_results(step)
                        final_step = step
                        break
                    except:
                        continue
                
                if final_step is not None:
                    final_state = db.simulation_results(final_step)
                    
                    # Count agent types
                    agent_types = {}
                    for agent_state in final_state.agent_states:
                        agent_type = agent_state[2]
                        agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
                    
                    result = {
                        "simulation_id": simulation_id,
                        "status": sim_info["status"],
                        "created_at": sim_info["created_at"],
                        "output_path": sim_info.get("output_path"),
                        "final_step": final_step,
                        "final_metrics": {
                            "total_agents": len(final_state.agent_states),
                            "agent_types": agent_types,
                            "total_resources": sum(r[1] for r in final_state.resource_states),
                            "simulation_state": final_state.simulation_state
                        }
                    }
                else:
                    result = {
                        "simulation_id": simulation_id,
                        "status": "no_data_found",
                        "message": "No simulation data found in database"
                    }
                    
            finally:
                db.close()
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error getting simulation status: {str(e)}")
            return format_error_response("get_simulation_status", str(e))
    
    def _get_simulation_summary(self, simulation_path: str, include_charts: bool) -> str:
        """Implementation of get_simulation_summary tool."""
        try:
            if not os.path.exists(simulation_path):
                return format_error_response(
                    "get_simulation_summary",
                    f"Simulation not found: {simulation_path}"
                )
            
            db = SimulationDatabase(simulation_path)
            
            try:
                # Get configuration
                config = db.get_configuration()
                
                # Find final step with data
                final_step = None
                for step in range(2000, -1, -1):
                    try:
                        final_state = db.simulation_results(step)
                        final_step = step
                        break
                    except:
                        continue
                
                if final_step is None:
                    return json.dumps({"error": "No simulation data found in database"})
                
                final_state = db.simulation_results(final_step)
                
                # Calculate comprehensive statistics
                agent_types = {}
                agent_health = []
                agent_resources = []
                
                for agent_state in final_state.agent_states:
                    agent_type = agent_state[2]
                    health = agent_state[6]
                    resources = agent_state[5]
                    
                    agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
                    agent_health.append(health)
                    agent_resources.append(resources)
                
                summary = {
                    "simulation_path": simulation_path,
                    "steps_completed": final_step + 1,
                    "final_state": {
                        "total_agents": len(final_state.agent_states),
                        "agent_types": agent_types,
                        "total_resources": sum(r[1] for r in final_state.resource_states),
                        "avg_agent_health": sum(agent_health) / len(agent_health) if agent_health else 0,
                        "avg_agent_resources": sum(agent_resources) / len(agent_resources) if agent_resources else 0
                    },
                    "configuration_summary": {
                        "environment_size": f"{config.get('width', 'unknown')}x{config.get('height', 'unknown')}",
                        "initial_agents": {
                            "system": config.get("system_agents", 0),
                            "independent": config.get("independent_agents", 0),
                            "control": config.get("control_agents", 0)
                        }
                    },
                    "simulation_metrics": final_state.simulation_state
                }
                
                return json.dumps(summary, indent=2, default=str)
                
            finally:
                db.close()
            
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return format_error_response("get_simulation_summary", str(e))
    
    def _analyze_simulation(self, simulation_path: str, analysis_types: List[str], output_path: Optional[str] = None) -> str:
        """Implementation of analyze_simulation tool."""
        try:
            if not os.path.exists(simulation_path):
                return format_error_response(
                    "analyze_simulation",
                    f"Path not found: {simulation_path}"
                )
            
            # Setup output directory
            if not output_path:
                if simulation_path.endswith(".db"):
                    output_path = f"{os.path.splitext(simulation_path)[0]}_analysis"
                else:
                    output_path = str(self.config.get_path("analysis") / f"analysis_{generate_unique_id('ana')}")
            
            os.makedirs(output_path, exist_ok=True)
            results = {}
            
            # Run requested analyses
            for analysis_type in analysis_types:
                logger.info(f"Running {analysis_type} analysis...")
                
                try:
                    if analysis_type == "dominance":
                        if simulation_path.endswith(".db"):
                            results[analysis_type] = {"status": "skipped", "reason": "Single DB dominance analysis requires experiment directory"}
                        else:
                            dominance_df = process_dominance_data(simulation_path, save_to_db=self.config.save_analysis_to_db)
                            results[analysis_type] = {
                                "status": "completed",
                                "simulations_analyzed": len(dominance_df) if dominance_df is not None else 0,
                                "output_path": output_path
                            }
                    
                    elif analysis_type == "advantage":
                        advantage_df = analyze_advantages(simulation_path, save_to_db=self.config.save_analysis_to_db)
                        results[analysis_type] = {
                            "status": "completed",
                            "simulations_analyzed": len(advantage_df) if advantage_df is not None else 0,
                            "output_path": output_path
                        }
                    
                    else:
                        results[analysis_type] = {"status": "not_implemented", "message": f"Analysis type {analysis_type} not yet implemented"}
                        
                except Exception as e:
                    logger.error(f"Error in {analysis_type} analysis: {str(e)}")
                    results[analysis_type] = {"status": "error", "message": str(e)}
            
            final_result = {
                "analysis_completed": True,
                "target_path": simulation_path,
                "output_path": output_path,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(final_result, indent=2)
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return format_error_response("analyze_simulation", str(e))
    
    def _export_simulation_data(self, simulation_path: str, output_path: str, format_type: str, data_types: List[str]) -> str:
        """Implementation of export_simulation_data tool."""
        try:
            if not os.path.exists(simulation_path):
                return format_error_response(
                    "export_simulation_data",
                    f"Simulation not found: {simulation_path}"
                )
            
            os.makedirs(output_path, exist_ok=True)
            
            db = SimulationDatabase(simulation_path)
            exported_files = []
            
            try:
                # Export requested data types
                if "all" in data_types or "agents" in data_types:
                    agents_file = os.path.join(output_path, f"agents.{format_type}")
                    db.export_data(agents_file, format_type, ["agents"])
                    exported_files.append(agents_file)
                
                if "all" in data_types or "actions" in data_types:
                    actions_file = os.path.join(output_path, f"actions.{format_type}")
                    db.export_data(actions_file, format_type, ["actions"])
                    exported_files.append(actions_file)
                
                if "all" in data_types or "steps" in data_types:
                    steps_file = os.path.join(output_path, f"steps.{format_type}")
                    db.export_data(steps_file, format_type, ["steps"])
                    exported_files.append(steps_file)
                
                if "all" in data_types or "states" in data_types:
                    states_file = os.path.join(output_path, f"states.{format_type}")
                    db.export_data(states_file, format_type, ["states"])
                    exported_files.append(states_file)
                    
            finally:
                db.close()
            
            result = {
                "export_completed": True,
                "simulation_path": simulation_path,
                "output_path": output_path,
                "format": format_type,
                "exported_files": exported_files,
                "total_files": len(exported_files),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Export completed: {len(exported_files)} files exported")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return format_error_response("export_simulation_data", str(e))
    
    def _batch_analyze(self, experiment_path: str, analysis_modules: List[str], save_to_db: bool, output_path: Optional[str] = None) -> str:
        """Implementation of batch_analyze tool."""
        try:
            if not os.path.exists(experiment_path):
                return format_error_response(
                    "batch_analyze",
                    f"Experiment path not found: {experiment_path}"
                )
            
            if not output_path:
                output_path = str(self.config.get_path("analysis") / f"batch_{generate_unique_id('batch')}")
            
            os.makedirs(output_path, exist_ok=True)
            results = {}
            
            # Run each analysis module
            for module in analysis_modules:
                logger.info(f"Running {module} batch analysis...")
                
                try:
                    if module == "dominance":
                        df = process_dominance_data(experiment_path, save_to_db=save_to_db)
                        if df is not None:
                            output_file = os.path.join(output_path, "dominance_analysis.csv")
                            df.to_csv(output_file, index=False)
                            results[module] = {
                                "status": "success",
                                "simulations_analyzed": len(df),
                                "output_file": output_file
                            }
                        else:
                            results[module] = {"status": "no_data"}
                    
                    elif module == "advantage":
                        df = analyze_advantages(experiment_path, save_to_db=save_to_db)
                        if df is not None:
                            output_file = os.path.join(output_path, "advantage_analysis.csv")
                            df.to_csv(output_file, index=False)
                            results[module] = {
                                "status": "success",
                                "simulations_analyzed": len(df),
                                "output_file": output_file
                            }
                        else:
                            results[module] = {"status": "no_data"}
                    
                    else:
                        results[module] = {"status": "not_implemented", "message": f"Module {module} not yet implemented"}
                    
                except Exception as e:
                    logger.error(f"Error in {module} analysis: {str(e)}")
                    results[module] = {"status": "error", "error": str(e)}
            
            # Save consolidated results
            results_file = os.path.join(output_path, "batch_analysis_summary.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            final_result = {
                "batch_analysis_completed": True,
                "experiment_path": experiment_path,
                "output_directory": output_path,
                "modules_run": analysis_modules,
                "results_summary": results,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Batch analysis completed for {experiment_path}")
            return json.dumps(final_result, indent=2)
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            return format_error_response("batch_analyze", str(e))
    
    def _create_research_project(self, name: str, description: str, base_path: str, tags: Optional[List[str]]) -> str:
        """Implementation of create_research_project tool."""
        try:
            project = ResearchProject(
                name=name,
                description=description,
                base_path=base_path,
                tags=tags or []
            )
            
            result = {
                "project_name": name,
                "project_path": str(project.path),
                "description": description,
                "tags": tags or [],
                "base_path": base_path,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"Research project created: {name}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error creating research project: {str(e)}")
            return format_error_response("create_research_project", str(e))
    
    def run(self, **kwargs) -> None:
        """Start the FastMCP server."""
        logger.info(f"Starting {self.config.server_name}")
        logger.info(f"Base directory: {self.config.base_dir}")
        logger.info("Server ready for LLM agent connections...")
        
        self.mcp.run(**kwargs)
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and statistics."""
        return {
            "server_name": self.config.server_name,
            "version": "1.0.0",
            "active_simulations": len(self.active_simulations),
            "active_experiments": len(self.active_experiments),
            "tools_available": len(self.mcp._tools),
            "config": self.config.to_dict()
        }