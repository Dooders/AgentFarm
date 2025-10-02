"""Unified AgentFarm Controller for agentic system interaction.

This module provides the main AgentFarmController class that serves as the
unified API for agentic systems to control simulations and experiments through MCP.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from farm.api.config_templates import ConfigTemplateManager
from farm.api.models import (
    AnalysisResults,
    ComparisonResults,
    ConfigTemplate,
    Event,
    ExperimentResults,
    ExperimentStatus,
    ExperimentStatusInfo,
    SessionInfo,
    SimulationResults,
    SimulationStatus,
    SimulationStatusInfo,
    ValidationResult,
)
from farm.api.session_manager import SessionManager
from farm.api.unified_adapter import UnifiedAdapter
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


class AgentFarmController:
    """Unified API for agentic systems to control simulations and experiments.

    This controller provides a clean, intuitive interface that abstracts away
    the complexity of the underlying simulation and experiment systems while
    providing comprehensive functionality for research and experimentation.
    """

    def __init__(self, workspace_path: Optional[str] = None):
        """Initialize the unified controller.

        Args:
            workspace_path: Base path for workspace storage. If None, uses default.
        """
        self.workspace_path = Path(workspace_path or "agentfarm_workspace")
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.session_manager = SessionManager(str(self.workspace_path / "sessions"))
        self.config_manager = ConfigTemplateManager()

        # Active adapters per session
        self._adapters: Dict[str, UnifiedAdapter] = {}

        logger.info(
            f"Initialized AgentFarmController with workspace: {self.workspace_path}"
        )

    def _get_adapter(self, session_id: str) -> UnifiedAdapter:
        """Get or create adapter for a session.

        Args:
            session_id: Session identifier

        Returns:
            UnifiedAdapter instance for the session
        """
        if session_id not in self._adapters:
            session_path = self.session_manager.get_session_path(session_id)
            if not session_path:
                raise ValueError(f"Session {session_id} not found")

            self._adapters[session_id] = UnifiedAdapter(session_path)
            logger.info(f"Created adapter for session: {session_id}")

        return self._adapters[session_id]

    # === SESSION MANAGEMENT ===

    def create_session(self, name: str, description: str = "") -> str:
        """Create a new session and return session_id.

        Args:
            name: Name of the session
            description: Optional description

        Returns:
            session_id: Unique identifier for the session
        """
        session_id = self.session_manager.create_session(name, description)
        logger.info(f"Created session: {name} ({session_id})")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get information about an existing session.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo if found, None otherwise
        """
        return self.session_manager.get_session(session_id)

    def list_sessions(self) -> List[SessionInfo]:
        """List all available sessions.

        Returns:
            List of SessionInfo objects
        """
        return self.session_manager.list_sessions()

    def delete_session(self, session_id: str, delete_files: bool = False) -> bool:
        """Delete a session and clean up resources.

        Args:
            session_id: Session identifier
            delete_files: Whether to delete associated files

        Returns:
            True if deleted successfully, False otherwise
        """
        # Clean up adapter if exists
        if session_id in self._adapters:
            self._adapters[session_id].cleanup()
            del self._adapters[session_id]

        return self.session_manager.delete_session(session_id, delete_files)

    # === SIMULATION CONTROL ===

    def create_simulation(self, session_id: str, config: Dict[str, Any]) -> str:
        """Create a new simulation and return simulation_id.

        Args:
            session_id: Session identifier
            config: Simulation configuration

        Returns:
            simulation_id: Unique identifier for the simulation
        """
        # Validate session exists
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Get adapter and create simulation
        adapter = self._get_adapter(session_id)
        simulation_id = adapter.create_simulation(config)

        # Add to session
        self.session_manager.add_simulation_to_session(session_id, simulation_id)

        logger.info(f"Created simulation {simulation_id} in session {session_id}")
        return simulation_id

    def start_simulation(self, session_id: str, simulation_id: str) -> SimulationStatusInfo:
        """Start a simulation.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        adapter = self._get_adapter(session_id)
        return adapter.start_simulation(simulation_id)

    def pause_simulation(self, session_id: str, simulation_id: str) -> SimulationStatusInfo:
        """Pause a running simulation.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        adapter = self._get_adapter(session_id)
        return adapter.pause_simulation(simulation_id)

    def resume_simulation(
        self, session_id: str, simulation_id: str
    ) -> SimulationStatusInfo:
        """Resume a paused simulation.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        adapter = self._get_adapter(session_id)
        return adapter.resume_simulation(simulation_id)

    def stop_simulation(self, session_id: str, simulation_id: str) -> SimulationStatusInfo:
        """Stop a simulation.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        adapter = self._get_adapter(session_id)
        return adapter.stop_simulation(simulation_id)

    def get_simulation_status(
        self, session_id: str, simulation_id: str
    ) -> SimulationStatusInfo:
        """Get current status of a simulation.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        adapter = self._get_adapter(session_id)
        return adapter.get_simulation_status(simulation_id)

    def get_simulation_results(
        self, session_id: str, simulation_id: str
    ) -> SimulationResults:
        """Get results from a completed simulation.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier

        Returns:
            Simulation results
        """
        adapter = self._get_adapter(session_id)
        return adapter.get_simulation_results(simulation_id)

    # === EXPERIMENT CONTROL ===

    def create_experiment(self, session_id: str, config: Dict[str, Any]) -> str:
        """Create a new experiment and return experiment_id.

        Args:
            session_id: Session identifier
            config: Experiment configuration

        Returns:
            experiment_id: Unique identifier for the experiment
        """
        # Validate session exists
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Get adapter and create experiment
        adapter = self._get_adapter(session_id)
        experiment_id = adapter.create_experiment(config)

        # Add to session
        self.session_manager.add_experiment_to_session(session_id, experiment_id)

        logger.info(f"Created experiment {experiment_id} in session {session_id}")
        return experiment_id

    def start_experiment(self, session_id: str, experiment_id: str) -> ExperimentStatusInfo:
        """Start an experiment.

        Args:
            session_id: Session identifier
            experiment_id: Experiment identifier

        Returns:
            Current experiment status
        """
        adapter = self._get_adapter(session_id)
        return adapter.start_experiment(experiment_id)

    def get_experiment_status(
        self, session_id: str, experiment_id: str
    ) -> ExperimentStatusInfo:
        """Get current status of an experiment.

        Args:
            session_id: Session identifier
            experiment_id: Experiment identifier

        Returns:
            Current experiment status
        """
        adapter = self._get_adapter(session_id)
        return adapter.get_experiment_status(experiment_id)

    def get_experiment_results(
        self, session_id: str, experiment_id: str
    ) -> ExperimentResults:
        """Get results from a completed experiment.

        Args:
            session_id: Session identifier
            experiment_id: Experiment identifier

        Returns:
            Experiment results
        """
        adapter = self._get_adapter(session_id)
        return adapter.get_experiment_results(experiment_id)

    # === CONFIGURATION MANAGEMENT ===

    def get_available_configs(self) -> List[ConfigTemplate]:
        """Get available configuration templates.

        Returns:
            List of available configuration templates
        """
        return self.config_manager.list_templates()

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a configuration.

        Args:
            config: Configuration to validate

        Returns:
            Validation result with status and messages
        """
        return self.config_manager.validate_config(config)

    def create_config_from_template(
        self, template_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a configuration from a template.

        Args:
            template_name: Name of the template to use
            overrides: Optional parameter overrides

        Returns:
            Configuration dictionary
        """
        config = self.config_manager.create_config_from_template(
            template_name, overrides
        )
        if not config:
            raise ValueError(f"Template {template_name} not found")
        return config

    # === ANALYSIS & VISUALIZATION ===

    def analyze_simulation(
        self, session_id: str, simulation_id: str
    ) -> AnalysisResults:
        """Run analysis on simulation results.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier

        Returns:
            Analysis results
        """
        adapter = self._get_adapter(session_id)
        return adapter.analyze_simulation(simulation_id)

    def compare_simulations(
        self, session_id: str, simulation_ids: List[str]
    ) -> ComparisonResults:
        """Compare multiple simulations.

        Args:
            session_id: Session identifier
            simulation_ids: List of simulation identifiers to compare

        Returns:
            Comparison results
        """
        adapter = self._get_adapter(session_id)
        return adapter.compare_simulations(simulation_ids)

    def generate_visualization(
        self, session_id: str, simulation_id: str, viz_type: str
    ) -> str:
        """Generate visualization and return file path.

        Args:
            session_id: Session identifier
            simulation_id: Simulation identifier
            viz_type: Type of visualization to generate

        Returns:
            Path to generated visualization file
        """
        # For now, this is a placeholder that would integrate with existing visualization
        # TODO: Implement actual visualization generation
        session_path = self.session_manager.get_session_path(session_id)
        if not session_path:
            raise ValueError(f"Session {session_id} not found")

        viz_path = session_path / "visualizations" / f"{simulation_id}_{viz_type}.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)

        # Placeholder - would generate actual visualization
        viz_path.touch()

        logger.info(f"Generated visualization: {viz_path}")
        return str(viz_path)

    # === MONITORING & EVENTS ===

    def subscribe_to_events(
        self,
        session_id: str,
        event_types: List[str],
        simulation_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> str:
        """Subscribe to simulation/experiment events.

        Args:
            session_id: Session identifier
            event_types: List of event types to subscribe to
            simulation_id: Optional simulation filter
            experiment_id: Optional experiment filter

        Returns:
            subscription_id: Unique identifier for the subscription
        """
        adapter = self._get_adapter(session_id)
        return adapter.subscribe_to_events(event_types, simulation_id, experiment_id)

    def get_event_history(self, session_id: str, subscription_id: str) -> List[Event]:
        """Get event history for a subscription.

        Args:
            session_id: Session identifier
            subscription_id: Subscription identifier

        Returns:
            List of matching events
        """
        adapter = self._get_adapter(session_id)
        return adapter.get_event_history(subscription_id)

    # === UTILITY METHODS ===

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics, None if not found
        """
        return self.session_manager.get_session_stats(session_id)

    def list_simulations(self, session_id: str) -> List[str]:
        """List simulations in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of simulation identifiers
        """
        session = self.get_session(session_id)
        if not session:
            return []
        return session.simulations

    def list_experiments(self, session_id: str) -> List[str]:
        """List experiments in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of experiment identifiers
        """
        session = self.get_session(session_id)
        if not session:
            return []
        return session.experiments

    def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up AgentFarmController")

        # Clean up all adapters
        for session_id, adapter in self._adapters.items():
            try:
                adapter.cleanup()
                logger.info(f"Cleaned up adapter for session: {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up adapter for session {session_id}: {e}")

        self._adapters.clear()
        logger.info("AgentFarmController cleanup complete")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and ensure cleanup."""
        self.cleanup()
        return False
