#!/usr/bin/env python3
"""
FastMCP Server for AgentFarm Simulation System

This FastMCP server provides an interface for LLM agents to:
- Create and configure simulations
- Run experiments with parameter variations
- Analyze simulation results
- Export and compare data

The server exposes the core AgentFarm functionality through FastMCP tools.
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastmcp import FastMCP

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
logger = logging.getLogger("fastmcp-simulation-server")

# Initialize FastMCP server
mcp = FastMCP("AgentFarm Simulation Server")

# Global state
active_simulations: Dict[str, Dict[str, Any]] = {}
active_experiments: Dict[str, ExperimentController] = {}

@mcp.tool()
def create_simulation(
    config: dict,
    output_path: Optional[str] = None
) -> str:
    """Create and run a single simulation with specified parameters.
    
    Args:
        config: Simulation configuration parameters including:
            - width, height: Environment dimensions (default: 100)
            - system_agents, independent_agents, control_agents: Agent counts (default: 10 each)
            - simulation_steps: Number of steps to run (default: 1000)
            - initial_resources: Starting resource amount (default: 20)
            - seed: Random seed for reproducibility (optional)
        output_path: Path to save simulation results (optional)
    
    Returns:
        JSON string with simulation results and metadata
    """
    global active_simulations
    
    # Generate unique simulation ID
    sim_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    try:
        # Create config
        sim_config = SimulationConfig()
        
        # Update config with provided parameters
        for key, value in config.items():
            if hasattr(sim_config, key):
                setattr(sim_config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Set output path
        if not output_path:
            output_path = f"simulations/sim_{sim_id}"
        
        os.makedirs(output_path, exist_ok=True)
        db_path = os.path.join(output_path, "simulation.db")
        
        # Run simulation
        logger.info(f"Starting simulation {sim_id} with {sim_config.simulation_steps} steps")
        environment = run_simulation(
            num_steps=sim_config.simulation_steps,
            config=sim_config,
            path=db_path,
            simulation_id=sim_id
        )
        
        # Store simulation info
        active_simulations[sim_id] = {
            "db_path": db_path,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "status": "completed",
            "environment": environment
        }
        
        # Get basic metrics
        db = SimulationDatabase(db_path)
        try:
            final_step = sim_config.simulation_steps - 1
            final_state = db.simulation_results(final_step)
            
            result = {
                "simulation_id": sim_id,
                "status": "completed",
                "output_path": output_path,
                "db_path": db_path,
                "final_metrics": {
                    "total_agents": len(final_state.agent_states),
                    "total_resources": sum(r[1] for r in final_state.resource_states),
                    "steps_completed": sim_config.simulation_steps
                }
            }
        finally:
            db.close()
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in create_simulation: {str(e)}", exc_info=True)
        return f"Failed to create simulation: {str(e)}"

@mcp.tool()
def create_experiment(
    name: str,
    base_config: dict,
    description: str = "",
    variations: Optional[List[dict]] = None,
    num_iterations: int = 10,
    steps_per_iteration: int = 1000
) -> str:
    """Create a multi-iteration experiment with parameter variations.
    
    Args:
        name: Experiment name
        base_config: Base configuration for all iterations
        description: Experiment description
        variations: List of parameter variations for each iteration
        num_iterations: Number of iterations to run
        steps_per_iteration: Steps per iteration
    
    Returns:
        JSON string with experiment details and ID
    """
    global active_experiments
    
    try:
        # Create base config
        sim_config = SimulationConfig()
        for key, value in base_config.items():
            if hasattr(sim_config, key):
                setattr(sim_config, key, value)
        
        # Create experiment controller
        output_dir = Path(f"experiments/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        experiment = ExperimentController(
            name=name,
            description=description,
            base_config=sim_config,
            output_dir=output_dir
        )
        
        # Store experiment metadata
        experiment._variations = variations
        experiment._num_iterations = num_iterations
        experiment._steps_per_iteration = steps_per_iteration
        
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
            "variations_count": len(variations) if variations else 0,
            "status": "created"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}", exc_info=True)
        return f"Failed to create experiment: {str(e)}"

@mcp.tool()
def run_experiment(experiment_id: str, run_analysis: bool = True) -> str:
    """Execute a created experiment.
    
    Args:
        experiment_id: ID of experiment to run
        run_analysis: Whether to run analysis after completion
    
    Returns:
        JSON string with experiment results
    """
    global active_experiments
    
    try:
        if experiment_id not in active_experiments:
            available_experiments = list(active_experiments.keys())
            return f"Experiment {experiment_id} not found. Available experiments: {available_experiments}"
        
        experiment = active_experiments[experiment_id]
        
        # Get experiment parameters
        variations = getattr(experiment, '_variations', None)
        num_iterations = getattr(experiment, '_num_iterations', 10)
        steps_per_iteration = getattr(experiment, '_steps_per_iteration', 1000)
        
        logger.info(f"Running experiment {experiment_id}: {experiment.name}")
        
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
            "experiment_id": experiment_id,
            "status": "completed",
            "output_dir": str(experiment.output_dir),
            "iterations_completed": state.get("current_iteration", 0),
            "total_iterations": state.get("total_iterations", 0)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}", exc_info=True)
        return f"Failed to run experiment: {str(e)}"

@mcp.tool()
def get_simulation_status(simulation_id: str) -> str:
    """Get the current status and results of a simulation.
    
    Args:
        simulation_id: ID of simulation to check
    
    Returns:
        JSON string with simulation status and metrics
    """
    global active_simulations
    
    try:
        if simulation_id not in active_simulations:
            available_simulations = list(active_simulations.keys())
            return f"Simulation {simulation_id} not found. Available simulations: {available_simulations}"
        
        sim_info = active_simulations[simulation_id]
        db_path = sim_info["db_path"]
        
        if not os.path.exists(db_path):
            return f"Simulation database not found at {db_path}"
        
        # Get basic metrics from database
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
                    agent_type = agent_state[2]  # agent_type is at index 2
                    agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
                
                result = {
                    "simulation_id": simulation_id,
                    "status": sim_info["status"],
                    "created_at": sim_info["created_at"],
                    "db_path": db_path,
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
        logger.error(f"Error getting simulation status: {str(e)}", exc_info=True)
        return f"Failed to get simulation status: {str(e)}"

@mcp.tool()
def analyze_simulation(
    simulation_path: str,
    analysis_types: List[str] = ["all"],
    output_path: Optional[str] = None
) -> str:
    """Run comprehensive analysis on a completed simulation.
    
    Args:
        simulation_path: Path to simulation database or directory
        analysis_types: Types of analysis to run (dominance, advantage, genesis, social_behavior, all)
        output_path: Path to save analysis results
    
    Returns:
        JSON string with analysis results summary
    """
    try:
        if not os.path.exists(simulation_path):
            return f"Simulation path not found: {simulation_path}"
        
        # Determine output path
        if simulation_path.endswith(".db"):
            if not output_path:
                output_path = f"{os.path.splitext(simulation_path)[0]}_analysis"
        else:
            if not output_path:
                output_path = os.path.join(simulation_path, "analysis")
        
        os.makedirs(output_path, exist_ok=True)
        
        results = {}
        
        # Run requested analyses
        if "all" in analysis_types or "dominance" in analysis_types:
            logger.info("Running dominance analysis...")
            try:
                if simulation_path.endswith(".db"):
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
        
        final_result = {
            "analysis_completed_for": simulation_path,
            "output_path": output_path,
            "results": results
        }
        
        return json.dumps(final_result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in analyze_simulation: {str(e)}", exc_info=True)
        return f"Failed to analyze simulation: {str(e)}"

@mcp.tool()
def compare_simulations(
    simulation_paths: List[str],
    metrics: List[str] = ["population_dynamics", "resource_efficiency", "survival_rates"],
    output_path: Optional[str] = None
) -> str:
    """Compare results between multiple simulations.
    
    Args:
        simulation_paths: Paths to simulation databases to compare
        metrics: Specific metrics to compare
        output_path: Path to save comparison results
    
    Returns:
        JSON string with comparison results
    """
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
        
        result = {
            "comparison_completed": True,
            "simulations_compared": len(simulation_data),
            "output_path": comparison_file,
            "summary": simulation_data
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error comparing simulations: {str(e)}", exc_info=True)
        return f"Failed to compare simulations: {str(e)}"

@mcp.tool()
def export_simulation_data(
    simulation_path: str,
    output_path: str,
    format_type: str = "csv",
    data_types: List[str] = ["all"]
) -> str:
    """Export simulation data in specified format.
    
    Args:
        simulation_path: Path to simulation database
        output_path: Path to save exported data
        format_type: Export format (csv, json, parquet)
        data_types: Types of data to export (agents, actions, states, resources, steps, all)
    
    Returns:
        JSON string with export results
    """
    try:
        if not os.path.exists(simulation_path):
            return f"Simulation path not found: {simulation_path}"
        
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
        
        result = {
            "export_completed": True,
            "exported_files": exported_files,
            "total_files": len(exported_files)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error exporting simulation data: {str(e)}", exc_info=True)
        return f"Failed to export simulation data: {str(e)}"

@mcp.tool()
def list_simulations(search_path: str = "simulations") -> str:
    """List all available simulations and their metadata.
    
    Args:
        search_path: Directory to search for simulations
    
    Returns:
        JSON string with simulation list
    """
    try:
        if not os.path.exists(search_path):
            os.makedirs(search_path, exist_ok=True)
            return f"No simulations found. Created directory: {search_path}"
        
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
        if os.path.exists("experiments"):
            for root, dirs, files in os.walk("experiments"):
                if "iteration_1" in dirs:  # Experiment directory
                    simulations.append({
                        "id": os.path.basename(root),
                        "type": "experiment",
                        "path": root,
                        "iterations": len([d for d in dirs if d.startswith("iteration_")])
                    })
        
        result = {
            "simulations_found": len(simulations),
            "simulations": simulations
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error listing simulations: {str(e)}", exc_info=True)
        return f"Failed to list simulations: {str(e)}"

@mcp.tool()
def get_simulation_summary(
    simulation_path: str,
    include_charts: bool = False
) -> str:
    """Get a detailed summary of simulation results.
    
    Args:
        simulation_path: Path to simulation database
        include_charts: Whether to generate summary charts
    
    Returns:
        JSON string with detailed simulation summary
    """
    try:
        if not os.path.exists(simulation_path):
            return f"Simulation path not found: {simulation_path}"
        
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
                return "No simulation data found in database"
            
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
            
            return json.dumps(summary, indent=2, default=str)
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error getting simulation summary: {str(e)}", exc_info=True)
        return f"Failed to get simulation summary: {str(e)}"

@mcp.tool()
def create_research_project(
    name: str,
    description: str,
    base_path: str = "research",
    tags: Optional[List[str]] = None
) -> str:
    """Create a structured research project with multiple experiments.
    
    Args:
        name: Research project name
        description: Research description and goals
        base_path: Base directory for research
        tags: Tags for categorizing research
    
    Returns:
        JSON string with project details
    """
    try:
        # Create research project
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
            "created_at": datetime.now().isoformat()
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error creating research project: {str(e)}", exc_info=True)
        return f"Failed to create research project: {str(e)}"

@mcp.tool()
def batch_analyze(
    experiment_path: str,
    analysis_modules: List[str] = ["dominance", "advantage"],
    save_to_db: bool = True,
    output_path: Optional[str] = None
) -> str:
    """Run batch analysis across multiple simulations or experiments.
    
    Args:
        experiment_path: Path to experiment directory
        analysis_modules: Analysis modules to run (dominance, advantage, genesis, social_behavior)
        save_to_db: Whether to save results to database
        output_path: Path to save consolidated analysis
    
    Returns:
        JSON string with batch analysis results
    """
    try:
        if not os.path.exists(experiment_path):
            return f"Experiment path not found: {experiment_path}"
        
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
        
        final_result = {
            "batch_analysis_completed": True,
            "output_directory": output_path,
            "results": results
        }
        
        return json.dumps(final_result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}", exc_info=True)
        return f"Failed to run batch analysis: {str(e)}"

if __name__ == "__main__":
    # Run the FastMCP server
    mcp.run()