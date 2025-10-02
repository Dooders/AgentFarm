"""MCP Server for AgentFarm API integration.

This module provides an MCP (Model Context Protocol) server that exposes
the unified AgentFarm API as tools that agentic systems can use to control
simulations and experiments.
"""

import json
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import Tool, TextContent

from farm.api.unified_controller import AgentFarmController
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


class AgentFarmMCPServer:
    """MCP Server for AgentFarm API integration."""
    
    def __init__(self, controller: AgentFarmController):
        """Initialize the MCP server.
        
        Args:
            controller: AgentFarmController instance
        """
        self.controller = controller
        self.server = Server("agentfarm")
        self._register_tools()
        logger.info("Initialized AgentFarm MCP Server")
    
    def _register_tools(self):
        """Register MCP tools for agentic system interaction."""
        
        # === SESSION MANAGEMENT TOOLS ===
        
        @self.server.call_tool()
        async def create_session(name: str, description: str = "") -> List[TextContent]:
            """Create a new research session.
            
            Args:
                name: Name of the session
                description: Optional description of the session
                
            Returns:
                JSON with session_id and confirmation message
            """
            try:
                session_id = self.controller.create_session(name, description)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "session_id": session_id,
                    "message": f"Created session '{name}' with ID: {session_id}"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to create session: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def list_sessions() -> List[TextContent]:
            """List all available sessions.
            
            Returns:
                JSON with list of sessions and their information
            """
            try:
                sessions = self.controller.list_sessions()
                sessions_data = [session.to_dict() for session in sessions]
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "sessions": sessions_data,
                    "count": len(sessions_data),
                    "message": f"Found {len(sessions_data)} sessions"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to list sessions: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def get_session_info(session_id: str) -> List[TextContent]:
            """Get information about a specific session.
            
            Args:
                session_id: Session identifier
                
            Returns:
                JSON with session information
            """
            try:
                session = self.controller.get_session(session_id)
                if not session:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Session not found",
                        "message": f"Session {session_id} not found"
                    }, indent=2))]
                
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "session": session.to_dict(),
                    "message": f"Retrieved session information for {session.name}"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to get session info: {e}"
                }, indent=2))]
        
        # === SIMULATION CONTROL TOOLS ===
        
        @self.server.call_tool()
        async def create_simulation(session_id: str, config: Dict[str, Any]) -> List[TextContent]:
            """Create a new simulation in a session.
            
            Args:
                session_id: Session identifier
                config: Simulation configuration dictionary
                
            Returns:
                JSON with simulation_id and confirmation message
            """
            try:
                simulation_id = self.controller.create_simulation(session_id, config)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "simulation_id": simulation_id,
                    "message": f"Created simulation with ID: {simulation_id}"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to create simulation: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def start_simulation(session_id: str, simulation_id: str) -> List[TextContent]:
            """Start a simulation.
            
            Args:
                session_id: Session identifier
                simulation_id: Simulation identifier
                
            Returns:
                JSON with simulation status
            """
            try:
                status = self.controller.start_simulation(session_id, simulation_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "status": status.to_dict(),
                    "message": f"Simulation {simulation_id} started successfully"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to start simulation: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def get_simulation_status(session_id: str, simulation_id: str) -> List[TextContent]:
            """Get current status of a simulation.
            
            Args:
                session_id: Session identifier
                simulation_id: Simulation identifier
                
            Returns:
                JSON with current simulation status
            """
            try:
                status = self.controller.get_simulation_status(session_id, simulation_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "status": status.to_dict(),
                    "message": f"Simulation is {status.status.value} at step {status.current_step}/{status.total_steps} ({status.progress_percentage:.1f}%)"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to get simulation status: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def pause_simulation(session_id: str, simulation_id: str) -> List[TextContent]:
            """Pause a running simulation.
            
            Args:
                session_id: Session identifier
                simulation_id: Simulation identifier
                
            Returns:
                JSON with updated simulation status
            """
            try:
                status = self.controller.pause_simulation(session_id, simulation_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "status": status.to_dict(),
                    "message": f"Simulation {simulation_id} paused"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to pause simulation: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def resume_simulation(session_id: str, simulation_id: str) -> List[TextContent]:
            """Resume a paused simulation.
            
            Args:
                session_id: Session identifier
                simulation_id: Simulation identifier
                
            Returns:
                JSON with updated simulation status
            """
            try:
                status = self.controller.resume_simulation(session_id, simulation_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "status": status.to_dict(),
                    "message": f"Simulation {simulation_id} resumed"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to resume simulation: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def stop_simulation(session_id: str, simulation_id: str) -> List[TextContent]:
            """Stop a simulation.
            
            Args:
                session_id: Session identifier
                simulation_id: Simulation identifier
                
            Returns:
                JSON with final simulation status
            """
            try:
                status = self.controller.stop_simulation(session_id, simulation_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "status": status.to_dict(),
                    "message": f"Simulation {simulation_id} stopped"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to stop simulation: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def get_simulation_results(session_id: str, simulation_id: str) -> List[TextContent]:
            """Get results from a completed simulation.
            
            Args:
                session_id: Session identifier
                simulation_id: Simulation identifier
                
            Returns:
                JSON with simulation results
            """
            try:
                results = self.controller.get_simulation_results(session_id, simulation_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "results": results.to_dict(),
                    "message": f"Retrieved results for simulation {simulation_id}"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to get simulation results: {e}"
                }, indent=2))]
        
        # === EXPERIMENT CONTROL TOOLS ===
        
        @self.server.call_tool()
        async def create_experiment(session_id: str, config: Dict[str, Any]) -> List[TextContent]:
            """Create a new experiment in a session.
            
            Args:
                session_id: Session identifier
                config: Experiment configuration dictionary
                
            Returns:
                JSON with experiment_id and confirmation message
            """
            try:
                experiment_id = self.controller.create_experiment(session_id, config)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "experiment_id": experiment_id,
                    "message": f"Created experiment with ID: {experiment_id}"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to create experiment: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def start_experiment(session_id: str, experiment_id: str) -> List[TextContent]:
            """Start an experiment.
            
            Args:
                session_id: Session identifier
                experiment_id: Experiment identifier
                
            Returns:
                JSON with experiment status
            """
            try:
                status = self.controller.start_experiment(session_id, experiment_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "status": status.to_dict(),
                    "message": f"Experiment {experiment_id} started successfully"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to start experiment: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def get_experiment_status(session_id: str, experiment_id: str) -> List[TextContent]:
            """Get current status of an experiment.
            
            Args:
                session_id: Session identifier
                experiment_id: Experiment identifier
                
            Returns:
                JSON with current experiment status
            """
            try:
                status = self.controller.get_experiment_status(session_id, experiment_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "status": status.to_dict(),
                    "message": f"Experiment is {status.status.value} at iteration {status.current_iteration}/{status.total_iterations} ({status.progress_percentage:.1f}%)"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to get experiment status: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def get_experiment_results(session_id: str, experiment_id: str) -> List[TextContent]:
            """Get results from a completed experiment.
            
            Args:
                session_id: Session identifier
                experiment_id: Experiment identifier
                
            Returns:
                JSON with experiment results
            """
            try:
                results = self.controller.get_experiment_results(session_id, experiment_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "results": results.to_dict(),
                    "message": f"Retrieved results for experiment {experiment_id}"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to get experiment results: {e}"
                }, indent=2))]
        
        # === CONFIGURATION TOOLS ===
        
        @self.server.call_tool()
        async def get_available_configs() -> List[TextContent]:
            """Get available configuration templates.
            
            Returns:
                JSON with list of available configuration templates
            """
            try:
                configs = self.controller.get_available_configs()
                configs_data = [config.to_dict() for config in configs]
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "configs": configs_data,
                    "count": len(configs_data),
                    "message": f"Found {len(configs_data)} available configurations"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to get available configs: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def validate_config(config: Dict[str, Any]) -> List[TextContent]:
            """Validate a configuration.
            
            Args:
                config: Configuration dictionary to validate
                
            Returns:
                JSON with validation results
            """
            try:
                result = self.controller.validate_config(config)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "validation": result.to_dict(),
                    "message": "Configuration is valid" if result.is_valid else "Configuration has errors"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to validate config: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def create_config_from_template(template_name: str, overrides: Optional[Dict[str, Any]] = None) -> List[TextContent]:
            """Create a configuration from a template.
            
            Args:
                template_name: Name of the template to use
                overrides: Optional parameter overrides
                
            Returns:
                JSON with generated configuration
            """
            try:
                config = self.controller.create_config_from_template(template_name, overrides)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "config": config,
                    "message": f"Created configuration from template '{template_name}'"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to create config from template: {e}"
                }, indent=2))]
        
        # === ANALYSIS TOOLS ===
        
        @self.server.call_tool()
        async def analyze_simulation(session_id: str, simulation_id: str) -> List[TextContent]:
            """Analyze simulation results.
            
            Args:
                session_id: Session identifier
                simulation_id: Simulation identifier
                
            Returns:
                JSON with analysis results
            """
            try:
                results = self.controller.analyze_simulation(session_id, simulation_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "analysis": results.to_dict(),
                    "message": f"Analysis completed for simulation {simulation_id}"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to analyze simulation: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def compare_simulations(session_id: str, simulation_ids: List[str]) -> List[TextContent]:
            """Compare multiple simulations.
            
            Args:
                session_id: Session identifier
                simulation_ids: List of simulation identifiers to compare
                
            Returns:
                JSON with comparison results
            """
            try:
                results = self.controller.compare_simulations(session_id, simulation_ids)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "comparison": results.to_dict(),
                    "message": f"Comparison completed for {len(simulation_ids)} simulations"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to compare simulations: {e}"
                }, indent=2))]
        
        # === UTILITY TOOLS ===
        
        @self.server.call_tool()
        async def list_simulations(session_id: str) -> List[TextContent]:
            """List simulations in a session.
            
            Args:
                session_id: Session identifier
                
            Returns:
                JSON with list of simulation identifiers
            """
            try:
                simulations = self.controller.list_simulations(session_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "simulations": simulations,
                    "count": len(simulations),
                    "message": f"Found {len(simulations)} simulations in session"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to list simulations: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def list_experiments(session_id: str) -> List[TextContent]:
            """List experiments in a session.
            
            Args:
                session_id: Session identifier
                
            Returns:
                JSON with list of experiment identifiers
            """
            try:
                experiments = self.controller.list_experiments(session_id)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "experiments": experiments,
                    "count": len(experiments),
                    "message": f"Found {len(experiments)} experiments in session"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to list experiments: {e}"
                }, indent=2))]
        
        @self.server.call_tool()
        async def get_session_stats(session_id: str) -> List[TextContent]:
            """Get statistics for a session.
            
            Args:
                session_id: Session identifier
                
            Returns:
                JSON with session statistics
            """
            try:
                stats = self.controller.get_session_stats(session_id)
                if not stats:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": "Session not found",
                        "message": f"Session {session_id} not found"
                    }, indent=2))]
                
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "stats": stats,
                    "message": f"Retrieved statistics for session {session_id}"
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to get session stats: {e}"
                }, indent=2))]
    
    def get_server(self) -> Server:
        """Get the MCP server instance.
        
        Returns:
            MCP Server instance
        """
        return self.server
