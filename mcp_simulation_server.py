#!/usr/bin/env python3
"""
MCP Server for AgentFarm Simulation System

This Model Context Protocol (MCP) server provides an interface for LLM agents to:
- Create and configure simulations
- Run experiments with parameter variations
- Analyze simulation results
- Export and compare data

The server exposes the core AgentFarm functionality through standardized MCP tools.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from mcp import types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Add the farm package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from farm.core.simulation import run_simulation
from farm.core.config import SimulationConfig
from farm.controllers.experiment_controller import ExperimentController
from farm.database.database import SimulationDatabase
from farm.analysis.dominance.analyze import process_dominance_data
from farm.analysis.advantage.analyze import analyze_advantages
from farm.analysis.genesis.analyze import analyze_genesis_across_simulations
from farm.analysis.social_behavior.analyze import analyze_social_behaviors_across_simulations
from farm.research.research import ResearchProject

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-simulation-server")

# Global state
server = Server("agentfarm-simulation")
active_simulations: Dict[str, Dict[str, Any]] = {}
active_experiments: Dict[str, ExperimentController] = {}

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available simulation tools."""
    return [
        types.Tool(
            name="create_simulation",
            description="Create and run a single simulation with specified parameters",
            inputSchema={
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "description": "Simulation configuration parameters",
                        "properties": {
                            "width": {"type": "integer", "default": 100},
                            "height": {"type": "integer", "default": 100},
                            "system_agents": {"type": "integer", "default": 10},
                            "independent_agents": {"type": "integer", "default": 10},
                            "control_agents": {"type": "integer", "default": 10},
                            "simulation_steps": {"type": "integer", "default": 1000},
                            "initial_resources": {"type": "integer", "default": 20},
                            "seed": {"type": "integer", "default": None}
                        }
                    },
                    "output_path": {"type": "string", "description": "Path to save simulation results"}
                },
                "required": ["config"]
            }
        ),
        
        types.Tool(
            name="create_experiment",
            description="Create a research experiment with multiple simulation iterations",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Experiment name"},
                    "description": {"type": "string", "description": "Experiment description"},
                    "base_config": {
                        "type": "object",
                        "description": "Base configuration for all iterations"
                    },
                    "variations": {
                        "type": "array",
                        "description": "List of parameter variations for each iteration",
                        "items": {"type": "object"}
                    },
                    "num_iterations": {"type": "integer", "default": 10},
                    "steps_per_iteration": {"type": "integer", "default": 1000}
                },
                "required": ["name", "base_config"]
            }
        ),
        
        types.Tool(
            name="run_experiment",
            description="Execute a created experiment",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "ID of experiment to run"},
                    "run_analysis": {"type": "boolean", "default": True, "description": "Whether to run analysis after completion"}
                },
                "required": ["experiment_id"]
            }
        ),
        
        types.Tool(
            name="get_simulation_status",
            description="Get the current status and results of a simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulation_id": {"type": "string", "description": "ID of simulation to check"}
                },
                "required": ["simulation_id"]
            }
        ),
        
        types.Tool(
            name="analyze_simulation",
            description="Run comprehensive analysis on a completed simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulation_path": {"type": "string", "description": "Path to simulation database or directory"},
                    "analysis_types": {
                        "type": "array",
                        "description": "Types of analysis to run",
                        "items": {
                            "type": "string",
                            "enum": ["dominance", "advantage", "genesis", "social_behavior", "all"]
                        },
                        "default": ["all"]
                    },
                    "output_path": {"type": "string", "description": "Path to save analysis results"}
                },
                "required": ["simulation_path"]
            }
        ),
        
        types.Tool(
            name="compare_simulations",
            description="Compare results between two or more simulations",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulation_paths": {
                        "type": "array",
                        "description": "Paths to simulation databases to compare",
                        "items": {"type": "string"},
                        "minItems": 2
                    },
                    "metrics": {
                        "type": "array",
                        "description": "Specific metrics to compare",
                        "items": {"type": "string"},
                        "default": ["population_dynamics", "resource_efficiency", "survival_rates"]
                    },
                    "output_path": {"type": "string", "description": "Path to save comparison results"}
                },
                "required": ["simulation_paths"]
            }
        ),
        
        types.Tool(
            name="export_simulation_data",
            description="Export simulation data in various formats",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulation_path": {"type": "string", "description": "Path to simulation database"},
                    "format": {
                        "type": "string",
                        "enum": ["csv", "json", "parquet"],
                        "default": "csv",
                        "description": "Export format"
                    },
                    "data_types": {
                        "type": "array",
                        "description": "Types of data to export",
                        "items": {
                            "type": "string",
                            "enum": ["agents", "actions", "states", "resources", "steps", "all"]
                        },
                        "default": ["all"]
                    },
                    "output_path": {"type": "string", "description": "Path to save exported data"}
                },
                "required": ["simulation_path", "output_path"]
            }
        ),
        
        types.Tool(
            name="list_simulations",
            description="List all available simulations and their metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_path": {"type": "string", "default": "simulations", "description": "Directory to search for simulations"}
                }
            }
        ),
        
        types.Tool(
            name="get_simulation_summary",
            description="Get a summary of simulation results and key metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulation_path": {"type": "string", "description": "Path to simulation database"},
                    "include_charts": {"type": "boolean", "default": False, "description": "Whether to generate summary charts"}
                },
                "required": ["simulation_path"]
            }
        ),
        
        types.Tool(
            name="create_research_project",
            description="Create a structured research project with multiple experiments",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Research project name"},
                    "description": {"type": "string", "description": "Research description and goals"},
                    "base_path": {"type": "string", "default": "research", "description": "Base directory for research"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorizing research"
                    }
                },
                "required": ["name", "description"]
            }
        ),
        
        types.Tool(
            name="batch_analyze",
            description="Run batch analysis across multiple simulations or experiments",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_path": {"type": "string", "description": "Path to experiment directory"},
                    "analysis_modules": {
                        "type": "array",
                        "description": "Analysis modules to run",
                        "items": {
                            "type": "string",
                            "enum": ["dominance", "advantage", "genesis", "social_behavior"]
                        },
                        "default": ["dominance", "advantage"]
                    },
                    "save_to_db": {"type": "boolean", "default": True, "description": "Whether to save results to database"},
                    "output_path": {"type": "string", "description": "Path to save consolidated analysis"}
                },
                "required": ["experiment_path"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls from the LLM agent."""
    try:
        if name == "create_simulation":
            return await _handle_create_simulation(arguments)
        elif name == "create_experiment":
            return await _handle_create_experiment(arguments)
        elif name == "run_experiment":
            return await _handle_run_experiment(arguments)
        elif name == "get_simulation_status":
            return await _handle_get_simulation_status(arguments)
        elif name == "analyze_simulation":
            return await _handle_analyze_simulation(arguments)
        elif name == "compare_simulations":
            return await _handle_compare_simulations(arguments)
        elif name == "export_simulation_data":
            return await _handle_export_simulation_data(arguments)
        elif name == "list_simulations":
            return await _handle_list_simulations(arguments)
        elif name == "get_simulation_summary":
            return await _handle_get_simulation_summary(arguments)
        elif name == "create_research_project":
            return await _handle_create_research_project(arguments)
        elif name == "batch_analyze":
            return await _handle_batch_analyze(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error handling tool {name}: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
            )
        ]

async def _handle_create_simulation(arguments: dict) -> list[types.TextContent]:
    """Create and run a single simulation."""
    config_data = arguments.get("config", {})
    output_path = arguments.get("output_path")
    
    # Generate unique simulation ID
    sim_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    try:
        # Create config
        config = SimulationConfig()
        
        # Update config with provided parameters
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Set output path
        if not output_path:
            output_path = f"simulations/sim_{sim_id}"
        
        os.makedirs(output_path, exist_ok=True)
        db_path = os.path.join(output_path, "simulation.db")
        
        # Run simulation
        logger.info(f"Starting simulation {sim_id} with {config.simulation_steps} steps")
        environment = run_simulation(
            num_steps=config.simulation_steps,
            config=config,
            path=db_path,
            simulation_id=sim_id
        )
        
        # Store simulation info
        active_simulations[sim_id] = {
            "db_path": db_path,
            "config": config_data,
            "created_at": datetime.now().isoformat(),
            "status": "completed",
            "environment": environment
        }
        
        # Get basic metrics
        db = SimulationDatabase(db_path)
        final_step = config.simulation_steps - 1
        final_state = db.simulation_results(final_step)
        
        db.close()
        
        result = {
            "simulation_id": sim_id,
            "status": "completed",
            "output_path": output_path,
            "db_path": db_path,
            "final_metrics": {
                "total_agents": len(final_state.agent_states),
                "total_resources": sum(r[1] for r in final_state.resource_states),
                "steps_completed": config.simulation_steps
            }
        }
        
        return [
            types.TextContent(
                type="text",
                text=f"Simulation {sim_id} completed successfully!\n\n" +
                     f"Results:\n{json.dumps(result, indent=2)}"
            )
        ]
        
    except Exception as e:
        logger.error(f"Error in create_simulation: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to create simulation: {str(e)}"
            )
        ]

async def _handle_create_experiment(arguments: dict) -> list[types.TextContent]:
    """Create a multi-iteration experiment."""
    name = arguments["name"]
    description = arguments.get("description", "")
    base_config_data = arguments["base_config"]
    variations = arguments.get("variations", [])
    num_iterations = arguments.get("num_iterations", 10)
    steps_per_iteration = arguments.get("steps_per_iteration", 1000)
    
    try:
        # Create base config
        base_config = SimulationConfig()
        for key, value in base_config_data.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
        
        # Create experiment controller
        output_dir = Path(f"experiments/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        experiment = ExperimentController(
            name=name,
            description=description,
            base_config=base_config,
            output_dir=output_dir
        )
        
        # Store experiment
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        active_experiments[exp_id] = experiment
        
        result = {
            "experiment_id": exp_id,
            "name": name,
            "description": description,
            "output_dir": str(output_dir),
            "num_iterations": num_iterations,
            "steps_per_iteration": steps_per_iteration,
            "variations_count": len(variations),
            "status": "created"
        }
        
        return [
            types.TextContent(
                type="text",
                text=f"Experiment '{name}' created successfully!\n\n" +
                     f"Details:\n{json.dumps(result, indent=2)}\n\n" +
                     f"Use run_experiment with experiment_id '{exp_id}' to execute it."
            )
        ]
        
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to create experiment: {str(e)}"
            )
        ]

async def _handle_run_experiment(arguments: dict) -> list[types.TextContent]:
    """Run a created experiment."""
    exp_id = arguments["experiment_id"]
    run_analysis = arguments.get("run_analysis", True)
    
    try:
        if exp_id not in active_experiments:
            return [
                types.TextContent(
                    type="text",
                    text=f"Experiment {exp_id} not found. Available experiments: {list(active_experiments.keys())}"
                )
            ]
        
        experiment = active_experiments[exp_id]
        
        # Get variations from experiment metadata if available
        variations = getattr(experiment, '_variations', None)
        num_iterations = getattr(experiment, '_num_iterations', 10)
        steps_per_iteration = getattr(experiment, '_steps_per_iteration', 1000)
        
        logger.info(f"Running experiment {exp_id}: {experiment.name}")
        
        # Run the experiment
        experiment.run_experiment(
            num_iterations=num_iterations,
            variations=variations,
            num_steps=steps_per_iteration,
            run_analysis=run_analysis
        )
        
        # Get experiment state
        state = experiment.get_state()
        
        result = {
            "experiment_id": exp_id,
            "status": "completed",
            "output_dir": str(experiment.output_dir),
            "iterations_completed": state.get("current_iteration", 0),
            "total_iterations": state.get("total_iterations", 0)
        }
        
        return [
            types.TextContent(
                type="text",
                text=f"Experiment {exp_id} completed successfully!\n\n" +
                     f"Results:\n{json.dumps(result, indent=2)}\n\n" +
                     f"Output directory: {experiment.output_dir}"
            )
        ]
        
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to run experiment: {str(e)}"
            )
        ]

async def _handle_get_simulation_status(arguments: dict) -> list[types.TextContent]:
    """Get simulation status and basic metrics."""
    sim_id = arguments["simulation_id"]
    
    try:
        if sim_id not in active_simulations:
            return [
                types.TextContent(
                    type="text",
                    text=f"Simulation {sim_id} not found. Available simulations: {list(active_simulations.keys())}"
                )
            ]
        
        sim_info = active_simulations[sim_id]
        db_path = sim_info["db_path"]
        
        if not os.path.exists(db_path):
            return [
                types.TextContent(
                    type="text",
                    text=f"Simulation database not found at {db_path}"
                )
            ]
        
        # Get basic metrics from database
        db = SimulationDatabase(db_path)
        
        # Get final step metrics
        try:
            # Find the last available step
            final_step = None
            for step in range(2000, -1, -1):  # Check backwards from a high number
                try:
                    final_state = db.simulation_results(step)
                    final_step = step
                    break
                except:
                    continue
            
            if final_step is not None:
                final_state = db.simulation_results(final_step)
                
                result = {
                    "simulation_id": sim_id,
                    "status": sim_info["status"],
                    "created_at": sim_info["created_at"],
                    "db_path": db_path,
                    "final_step": final_step,
                    "final_metrics": {
                        "total_agents": len(final_state.agent_states),
                        "agent_types": {},
                        "total_resources": sum(r[1] for r in final_state.resource_states),
                        "simulation_state": final_state.simulation_state
                    }
                }
                
                # Count agent types
                for agent_state in final_state.agent_states:
                    agent_type = agent_state[2]  # agent_type is at index 2
                    result["final_metrics"]["agent_types"][agent_type] = result["final_metrics"]["agent_types"].get(agent_type, 0) + 1
            else:
                result = {
                    "simulation_id": sim_id,
                    "status": "no_data_found",
                    "message": "No simulation data found in database"
                }
                
        except Exception as db_error:
            result = {
                "simulation_id": sim_id,
                "status": "error",
                "error": str(db_error)
            }
        finally:
            db.close()
        
        return [
            types.TextContent(
                type="text",
                text=f"Simulation Status:\n{json.dumps(result, indent=2)}"
            )
        ]
        
    except Exception as e:
        logger.error(f"Error getting simulation status: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to get simulation status: {str(e)}"
            )
        ]

async def _handle_analyze_simulation(arguments: dict) -> list[types.TextContent]:
    """Run comprehensive analysis on a simulation."""
    simulation_path = arguments["simulation_path"]
    analysis_types = arguments.get("analysis_types", ["all"])
    output_path = arguments.get("output_path")
    
    try:
        if not os.path.exists(simulation_path):
            return [
                types.TextContent(
                    type="text",
                    text=f"Simulation path not found: {simulation_path}"
                )
            ]
        
        # Determine if it's a single simulation or experiment directory
        if simulation_path.endswith(".db"):
            # Single simulation database
            db_path = simulation_path
            if not output_path:
                output_path = f"{os.path.splitext(simulation_path)[0]}_analysis"
        else:
            # Experiment directory
            if not output_path:
                output_path = os.path.join(simulation_path, "analysis")
        
        os.makedirs(output_path, exist_ok=True)
        
        results = {}
        
        # Run requested analyses
        if "all" in analysis_types or "dominance" in analysis_types:
            logger.info("Running dominance analysis...")
            try:
                if simulation_path.endswith(".db"):
                    # Single simulation - need to adapt for single DB
                    results["dominance"] = "Single simulation dominance analysis not implemented yet"
                else:
                    dominance_df = process_dominance_data(simulation_path, save_to_db=True)
                    results["dominance"] = {
                        "summary": f"Analyzed {len(dominance_df)} simulations",
                        "data_shape": dominance_df.shape if dominance_df is not None else None
                    }
            except Exception as e:
                results["dominance"] = f"Error: {str(e)}"
        
        if "all" in analysis_types or "advantage" in analysis_types:
            logger.info("Running advantage analysis...")
            try:
                advantage_df = analyze_advantages(simulation_path, save_to_db=True)
                results["advantage"] = {
                    "summary": f"Analyzed {len(advantage_df)} simulations",
                    "data_shape": advantage_df.shape if advantage_df is not None else None
                }
            except Exception as e:
                results["advantage"] = f"Error: {str(e)}"
        
        if "all" in analysis_types or "genesis" in analysis_types:
            logger.info("Running genesis analysis...")
            try:
                genesis_results = analyze_genesis_across_simulations(simulation_path)
                results["genesis"] = {
                    "summary": "Genesis analysis completed",
                    "simulations_analyzed": len(genesis_results.get("simulation_data", []))
                }
            except Exception as e:
                results["genesis"] = f"Error: {str(e)}"
        
        if "all" in analysis_types or "social_behavior" in analysis_types:
            logger.info("Running social behavior analysis...")
            try:
                social_results = analyze_social_behaviors_across_simulations(simulation_path, output_path)
                results["social_behavior"] = {
                    "summary": "Social behavior analysis completed",
                    "output_saved": output_path
                }
            except Exception as e:
                results["social_behavior"] = f"Error: {str(e)}"
        
        return [
            types.TextContent(
                type="text",
                text=f"Analysis completed for: {simulation_path}\n\n" +
                     f"Results:\n{json.dumps(results, indent=2)}\n\n" +
                     f"Output saved to: {output_path}"
            )
        ]
        
    except Exception as e:
        logger.error(f"Error in analyze_simulation: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to analyze simulation: {str(e)}"
            )
        ]

async def _handle_compare_simulations(arguments: dict) -> list[types.TextContent]:
    """Compare multiple simulations."""
    simulation_paths = arguments["simulation_paths"]
    metrics = arguments.get("metrics", ["population_dynamics", "resource_efficiency", "survival_rates"])
    output_path = arguments.get("output_path")
    
    try:
        if not output_path:
            output_path = f"comparisons/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(output_path, exist_ok=True)
        
        # Load data from each simulation
        simulation_data = {}
        
        for i, sim_path in enumerate(simulation_paths):
            if not os.path.exists(sim_path):
                logger.warning(f"Simulation path not found: {sim_path}")
                continue
                
            try:
                db = SimulationDatabase(sim_path)
                
                # Get summary metrics
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
                    simulation_data[f"sim_{i+1}"] = {
                        "path": sim_path,
                        "final_step": final_step,
                        "total_agents": len(final_state.agent_states),
                        "total_resources": sum(r[1] for r in final_state.resource_states),
                        "simulation_state": final_state.simulation_state
                    }
                
                db.close()
                
            except Exception as e:
                logger.error(f"Error loading simulation {sim_path}: {str(e)}")
                simulation_data[f"sim_{i+1}"] = {"error": str(e)}
        
        # Generate comparison report
        comparison_file = os.path.join(output_path, "comparison_report.json")
        with open(comparison_file, "w") as f:
            json.dump(simulation_data, f, indent=2)
        
        return [
            types.TextContent(
                type="text",
                text=f"Simulation comparison completed!\n\n" +
                     f"Compared {len(simulation_data)} simulations\n" +
                     f"Results saved to: {comparison_file}\n\n" +
                     f"Summary:\n{json.dumps(simulation_data, indent=2)}"
            )
        ]
        
    except Exception as e:
        logger.error(f"Error comparing simulations: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to compare simulations: {str(e)}"
            )
        ]

async def _handle_export_simulation_data(arguments: dict) -> list[types.TextContent]:
    """Export simulation data in specified format."""
    simulation_path = arguments["simulation_path"]
    format_type = arguments.get("format", "csv")
    data_types = arguments.get("data_types", ["all"])
    output_path = arguments["output_path"]
    
    try:
        if not os.path.exists(simulation_path):
            return [
                types.TextContent(
                    type="text",
                    text=f"Simulation path not found: {simulation_path}"
                )
            ]
        
        os.makedirs(output_path, exist_ok=True)
        
        db = SimulationDatabase(simulation_path)
        
        # Export requested data types
        exported_files = []
        
        try:
            if "all" in data_types or "agents" in data_types:
                agents_file = os.path.join(output_path, f"agents.{format_type}")
                db.export_data(agents_file, format_type, ["agents"])
                exported_files.append(agents_file)
            
            if "all" in data_types or "actions" in data_types:
                actions_file = os.path.join(output_path, f"actions.{format_type}")
                db.export_data(actions_file, format_type, ["actions"])
                exported_files.append(actions_file)
            
            if "all" in data_types or "states" in data_types:
                states_file = os.path.join(output_path, f"states.{format_type}")
                db.export_data(states_file, format_type, ["states"])
                exported_files.append(states_file)
            
            if "all" in data_types or "steps" in data_types:
                steps_file = os.path.join(output_path, f"steps.{format_type}")
                db.export_data(steps_file, format_type, ["steps"])
                exported_files.append(steps_file)
                
        finally:
            db.close()
        
        return [
            types.TextContent(
                type="text",
                text=f"Data export completed!\n\n" +
                     f"Exported files:\n" + "\n".join(f"- {f}" for f in exported_files) +
                     f"\n\nTotal files exported: {len(exported_files)}"
            )
        ]
        
    except Exception as e:
        logger.error(f"Error exporting simulation data: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to export simulation data: {str(e)}"
            )
        ]

async def _handle_list_simulations(arguments: dict) -> list[types.TextContent]:
    """List available simulations."""
    search_path = arguments.get("search_path", "simulations")
    
    try:
        if not os.path.exists(search_path):
            os.makedirs(search_path, exist_ok=True)
            return [
                types.TextContent(
                    type="text",
                    text=f"No simulations found. Created directory: {search_path}"
                )
            ]
        
        # Find simulation databases
        simulations = []
        
        # Check active simulations first
        for sim_id, sim_info in active_simulations.items():
            simulations.append({
                "id": sim_id,
                "type": "active",
                "created_at": sim_info["created_at"],
                "status": sim_info["status"],
                "db_path": sim_info["db_path"]
            })
        
        # Search for simulation files
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
        
        # Search for experiment directories
        for root, dirs, files in os.walk("experiments"):
            if "iteration_1" in dirs:  # Experiment directory
                simulations.append({
                    "id": os.path.basename(root),
                    "type": "experiment",
                    "path": root,
                    "iterations": len([d for d in dirs if d.startswith("iteration_")])
                })
        
        return [
            types.TextContent(
                type="text",
                text=f"Found {len(simulations)} simulations:\n\n" +
                     json.dumps(simulations, indent=2)
            )
        ]
        
    except Exception as e:
        logger.error(f"Error listing simulations: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to list simulations: {str(e)}"
            )
        ]

async def _handle_get_simulation_summary(arguments: dict) -> list[types.TextContent]:
    """Get a detailed summary of simulation results."""
    simulation_path = arguments["simulation_path"]
    include_charts = arguments.get("include_charts", False)
    
    try:
        if not os.path.exists(simulation_path):
            return [
                types.TextContent(
                    type="text",
                    text=f"Simulation path not found: {simulation_path}"
                )
            ]
        
        db = SimulationDatabase(simulation_path)
        
        try:
            # Get simulation metadata
            config = db.get_configuration()
            
            # Find final step
            final_step = None
            for step in range(2000, -1, -1):
                try:
                    final_state = db.simulation_results(step)
                    final_step = step
                    break
                except:
                    continue
            
            if final_step is None:
                return [
                    types.TextContent(
                        type="text",
                        text="No simulation data found in database"
                    )
                ]
            
            final_state = db.simulation_results(final_step)
            
            # Calculate summary statistics
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
                "configuration": config,
                "final_state": {
                    "total_agents": len(final_state.agent_states),
                    "agent_types": agent_types,
                    "total_resources": sum(r[1] for r in final_state.resource_states),
                    "average_agent_health": sum(agent_health) / len(agent_health) if agent_health else 0,
                    "average_agent_resources": sum(agent_resources) / len(agent_resources) if agent_resources else 0,
                    "simulation_metrics": final_state.simulation_state
                }
            }
            
            return [
                types.TextContent(
                    type="text",
                    text=f"Simulation Summary:\n{json.dumps(summary, indent=2, default=str)}"
                )
            ]
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error getting simulation summary: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to get simulation summary: {str(e)}"
            )
        ]

async def _handle_create_research_project(arguments: dict) -> list[types.TextContent]:
    """Create a structured research project."""
    name = arguments["name"]
    description = arguments["description"]
    base_path = arguments.get("base_path", "research")
    tags = arguments.get("tags", [])
    
    try:
        # Create research project
        project = ResearchProject(
            name=name,
            description=description,
            base_path=base_path,
            tags=tags
        )
        
        result = {
            "project_name": name,
            "project_path": str(project.path),
            "description": description,
            "tags": tags,
            "created_at": datetime.now().isoformat()
        }
        
        return [
            types.TextContent(
                type="text",
                text=f"Research project '{name}' created successfully!\n\n" +
                     f"Details:\n{json.dumps(result, indent=2)}\n\n" +
                     f"Project directory: {project.path}"
            )
        ]
        
    except Exception as e:
        logger.error(f"Error creating research project: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to create research project: {str(e)}"
            )
        ]

async def _handle_batch_analyze(arguments: dict) -> list[types.TextContent]:
    """Run batch analysis across multiple simulations."""
    experiment_path = arguments["experiment_path"]
    analysis_modules = arguments.get("analysis_modules", ["dominance", "advantage"])
    save_to_db = arguments.get("save_to_db", True)
    output_path = arguments.get("output_path")
    
    try:
        if not os.path.exists(experiment_path):
            return [
                types.TextContent(
                    type="text",
                    text=f"Experiment path not found: {experiment_path}"
                )
            ]
        
        if not output_path:
            output_path = os.path.join(experiment_path, "batch_analysis")
        
        os.makedirs(output_path, exist_ok=True)
        
        results = {}
        
        # Run each requested analysis module
        for module in analysis_modules:
            logger.info(f"Running {module} analysis...")
            
            try:
                if module == "dominance":
                    df = process_dominance_data(experiment_path, save_to_db=save_to_db)
                    if df is not None:
                        results[module] = {
                            "status": "success",
                            "simulations_analyzed": len(df),
                            "output_file": os.path.join(output_path, f"dominance_analysis.csv")
                        }
                        df.to_csv(results[module]["output_file"], index=False)
                    else:
                        results[module] = {"status": "no_data"}
                
                elif module == "advantage":
                    df = analyze_advantages(experiment_path, save_to_db=save_to_db)
                    if df is not None:
                        results[module] = {
                            "status": "success",
                            "simulations_analyzed": len(df),
                            "output_file": os.path.join(output_path, f"advantage_analysis.csv")
                        }
                        df.to_csv(results[module]["output_file"], index=False)
                    else:
                        results[module] = {"status": "no_data"}
                
                elif module == "genesis":
                    genesis_results = analyze_genesis_across_simulations(experiment_path)
                    results[module] = {
                        "status": "success",
                        "simulations_analyzed": len(genesis_results.get("simulation_data", [])),
                        "output_file": os.path.join(output_path, f"genesis_analysis.json")
                    }
                    with open(results[module]["output_file"], "w") as f:
                        json.dump(genesis_results, f, indent=2, default=str)
                
                elif module == "social_behavior":
                    social_results = analyze_social_behaviors_across_simulations(experiment_path, output_path)
                    results[module] = {
                        "status": "success",
                        "output_path": output_path
                    }
                
            except Exception as e:
                logger.error(f"Error in {module} analysis: {str(e)}")
                results[module] = {"status": "error", "error": str(e)}
        
        # Save consolidated results
        results_file = os.path.join(output_path, "batch_analysis_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        return [
            types.TextContent(
                type="text",
                text=f"Batch analysis completed!\n\n" +
                     f"Results:\n{json.dumps(results, indent=2)}\n\n" +
                     f"Output directory: {output_path}"
            )
        ]
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=f"Failed to run batch analysis: {str(e)}"
            )
        ]

async def main():
    """Main entry point for the MCP server."""
    # Configure the server
    server_info = types.Implementation(
        name="agentfarm-simulation",
        version="1.0.0"
    )
    
    logger.info("Starting AgentFarm MCP Server...")
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="agentfarm-simulation",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())