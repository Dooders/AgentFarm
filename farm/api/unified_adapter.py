"""Unified adapter for existing simulation and experiment controllers.

This module provides a unified interface that adapts the existing
SimulationController and ExperimentController classes to work with
the new unified API.
"""

import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from farm.analysis.comparative_analysis import compare_simulations
from farm.api.config_templates import ConfigTemplateManager
from farm.api.experiment_controller import ExperimentController
from farm.api.models import (
    AnalysisResults,
    ComparisonResults,
    Event,
    EventSubscription,
    ExperimentResults,
    ExperimentStatus,
    ExperimentStatusInfo,
    SimulationResults,
    SimulationStatus,
    SimulationStatusInfo,
)
from farm.api.simulation_controller import SimulationController
from farm.core.analysis import SimulationAnalyzer
from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


class UnifiedAdapter:
    """Adapter that unifies existing controllers for the new API."""

    def __init__(self, session_path: Path):
        """Initialize the unified adapter.

        Args:
            session_path: Path to the session directory
        """
        self.session_path = session_path
        self.config_manager = ConfigTemplateManager()

        # Active simulations and experiments
        self._simulations: Dict[str, Dict[str, Any]] = {}
        self._experiments: Dict[str, Dict[str, Any]] = {}

        # Event system
        self._event_subscriptions: Dict[str, EventSubscription] = {}
        self._event_history: List[Event] = []
        self._event_lock = threading.Lock()

    def create_simulation(self, config: Dict[str, Any]) -> str:
        """Create a new simulation.

        Args:
            config: Simulation configuration

        Returns:
            simulation_id: Unique identifier for the simulation
        """
        simulation_id = str(uuid.uuid4())

        # Convert config to SimulationConfig
        sim_config = self.config_manager.convert_to_simulation_config(config)
        if not sim_config:
            raise ValueError("Invalid simulation configuration")

        # Create simulation directory
        sim_dir = self.session_path / "simulations" / simulation_id
        sim_dir.mkdir(parents=True, exist_ok=True)

        # Create database path
        db_path = sim_dir / "simulation.db"

        # Create simulation controller
        controller = SimulationController(sim_config, str(db_path))

        # Store simulation info
        self._simulations[simulation_id] = {
            "controller": controller,
            "config": config,
            "sim_config": sim_config,
            "db_path": str(db_path),
            "directory": sim_dir,
            "status": SimulationStatus.CREATED,
            "created_at": datetime.now(),
            "start_time": None,
            "end_time": None,
            "current_step": 0,
            "total_steps": sim_config.simulation_steps,
            "error_message": None,
        }

        # Emit event
        self._emit_event("simulation_created", simulation_id=simulation_id, data=config)

        logger.info(f"Created simulation: {simulation_id}")
        return simulation_id

    def start_simulation(self, simulation_id: str) -> SimulationStatus:
        """Start a simulation.

        Args:
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        if simulation_id not in self._simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_info = self._simulations[simulation_id]
        controller = sim_info["controller"]

        try:
            # Initialize simulation
            controller.initialize_simulation()

            # Register callbacks for monitoring
            def on_step(step_num):
                sim_info["current_step"] = step_num
                self._emit_event(
                    "simulation_step",
                    simulation_id=simulation_id,
                    data={"step": step_num, "total": sim_info["total_steps"]},
                )

            def on_status(status):
                status_map = {
                    "initialized": SimulationStatus.CREATED,
                    "started": SimulationStatus.RUNNING,
                    "resumed": SimulationStatus.RUNNING,
                    "paused": SimulationStatus.PAUSED,
                    "stopped": SimulationStatus.STOPPED,
                    "completed": SimulationStatus.COMPLETED,
                    "error": SimulationStatus.ERROR,
                }
                sim_info["status"] = status_map.get(str(status).lower(), SimulationStatus.ERROR)
                self._emit_event(
                    "simulation_status_change",
                    simulation_id=simulation_id,
                    data={"status": status},
                )

            controller.register_step_callback("monitor", on_step)
            controller.register_status_callback("monitor", on_status)

            # Start simulation
            controller.start()
            sim_info["status"] = SimulationStatus.RUNNING
            sim_info["start_time"] = datetime.now()

            self._emit_event("simulation_started", simulation_id=simulation_id)

        except Exception as e:
            sim_info["status"] = SimulationStatus.ERROR
            sim_info["error_message"] = str(e)
            self._emit_event(
                "simulation_error", simulation_id=simulation_id, data={"error": str(e)}
            )
            raise

        return self.get_simulation_status(simulation_id)

    def pause_simulation(self, simulation_id: str) -> SimulationStatus:
        """Pause a simulation.

        Args:
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        if simulation_id not in self._simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_info = self._simulations[simulation_id]
        controller = sim_info["controller"]

        controller.pause()
        sim_info["status"] = SimulationStatus.PAUSED

        self._emit_event("simulation_paused", simulation_id=simulation_id)

        return self.get_simulation_status(simulation_id)

    def resume_simulation(self, simulation_id: str) -> SimulationStatus:
        """Resume a paused simulation.

        Args:
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        if simulation_id not in self._simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_info = self._simulations[simulation_id]
        controller = sim_info["controller"]

        controller.start()  # This resumes if paused
        sim_info["status"] = SimulationStatus.RUNNING

        self._emit_event("simulation_resumed", simulation_id=simulation_id)

        return self.get_simulation_status(simulation_id)

    def stop_simulation(self, simulation_id: str) -> SimulationStatus:
        """Stop a simulation.

        Args:
            simulation_id: Simulation identifier

        Returns:
            Current simulation status
        """
        if simulation_id not in self._simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_info = self._simulations[simulation_id]
        controller = sim_info["controller"]

        controller.stop()
        sim_info["status"] = SimulationStatus.STOPPED
        sim_info["end_time"] = datetime.now()

        self._emit_event("simulation_stopped", simulation_id=simulation_id)

        return self.get_simulation_status(simulation_id)

    def get_simulation_status(self, simulation_id: str) -> SimulationStatusInfo:
        """Get simulation status.

        Args:
            simulation_id: Simulation identifier

        Returns:
            SimulationStatus object
        """
        if simulation_id not in self._simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_info = self._simulations[simulation_id]
        controller = sim_info["controller"]

        # Get current state from controller
        state = controller.get_state()

        # Calculate progress
        progress = (
            (sim_info["current_step"] / sim_info["total_steps"]) * 100
            if sim_info["total_steps"] > 0
            else 0
        )

        return SimulationStatusInfo(
            simulation_id=simulation_id,
            status=sim_info["status"],
            current_step=sim_info["current_step"],
            total_steps=sim_info["total_steps"],
            progress_percentage=progress,
            start_time=sim_info["start_time"],
            end_time=sim_info["end_time"],
            error_message=sim_info["error_message"],
            metadata={
                "agent_count": state.get("agent_count", 0),
                "resource_count": state.get("resource_count", 0),
            },
        )

    def get_simulation_results(self, simulation_id: str) -> SimulationResults:
        """Get simulation results.

        Args:
            simulation_id: Simulation identifier

        Returns:
            SimulationResults object
        """
        if simulation_id not in self._simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_info = self._simulations[simulation_id]
        controller = sim_info["controller"]

        # Get final state
        state = controller.get_state()

        # Collect data files
        data_files = []
        sim_dir = sim_info["directory"]
        if sim_dir.exists():
            for file_path in sim_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in [
                    ".db",
                    ".csv",
                    ".json",
                    ".png",
                    ".html",
                ]:
                    data_files.append(str(file_path))

        return SimulationResults(
            simulation_id=simulation_id,
            status=sim_info["status"],
            total_steps=sim_info["total_steps"],
            final_agent_count=state.get("agent_count", 0),
            final_resource_count=state.get("resource_count", 0),
            metrics={
                "duration_seconds": (
                    (sim_info["end_time"] - sim_info["start_time"]).total_seconds()
                    if sim_info["start_time"] and sim_info["end_time"]
                    else 0
                ),
                "steps_per_second": sim_info["current_step"]
                / (
                    (sim_info["end_time"] - sim_info["start_time"]).total_seconds()
                    if sim_info["start_time"] and sim_info["end_time"]
                    else 1
                ),
            },
            data_files=data_files,
            analysis_available=len(data_files) > 0,
        )

    def create_experiment(self, config: Dict[str, Any]) -> str:
        """Create a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            experiment_id: Unique identifier for the experiment
        """
        experiment_id = str(uuid.uuid4())

        # Create experiment directory
        exp_dir = self.session_path / "experiments" / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Convert base config to SimulationConfig
        base_config = self.config_manager.convert_to_simulation_config(
            config.get("base_config", {})
        )
        if not base_config:
            raise ValueError("Invalid experiment base configuration")

        # Create experiment controller
        controller = ExperimentController(
            name=config.get("name", f"Experiment {experiment_id}"),
            description=config.get("description", ""),
            base_config=base_config,
            output_dir=exp_dir,
        )

        # Store experiment info
        self._experiments[experiment_id] = {
            "controller": controller,
            "config": config,
            "base_config": base_config,
            "directory": exp_dir,
            "status": ExperimentStatus.CREATED,
            "created_at": datetime.now(),
            "start_time": None,
            "end_time": None,
            "current_iteration": 0,
            "total_iterations": config.get("iterations", 1),
            "error_message": None,
        }

        # Emit event
        self._emit_event("experiment_created", experiment_id=experiment_id, data=config)

        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id

    def start_experiment(self, experiment_id: str) -> ExperimentStatus:
        """Start an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Current experiment status
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp_info = self._experiments[experiment_id]
        controller = exp_info["controller"]

        try:
            # Get variations from config
            variations = exp_info["config"].get("variations", [])
            num_steps = exp_info["config"].get("base_config", {}).get("steps", 1000)

            # Start experiment in background thread
            def run_experiment():
                try:
                    exp_info["status"] = ExperimentStatus.RUNNING
                    exp_info["start_time"] = datetime.now()

                    self._emit_event("experiment_started", experiment_id=experiment_id)

                    controller.run_experiment(
                        num_iterations=exp_info["total_iterations"],
                        variations=variations,
                        num_steps=num_steps,
                        run_analysis=True,
                    )

                    exp_info["status"] = ExperimentStatus.COMPLETED
                    exp_info["end_time"] = datetime.now()

                    self._emit_event(
                        "experiment_completed", experiment_id=experiment_id
                    )

                except Exception as e:
                    exp_info["status"] = ExperimentStatus.ERROR
                    exp_info["error_message"] = str(e)
                    self._emit_event(
                        "experiment_error",
                        experiment_id=experiment_id,
                        data={"error": str(e)},
                    )

            # Start experiment thread
            thread = threading.Thread(target=run_experiment)
            thread.daemon = True
            thread.start()

        except Exception as e:
            exp_info["status"] = ExperimentStatus.ERROR
            exp_info["error_message"] = str(e)
            self._emit_event(
                "experiment_error", experiment_id=experiment_id, data={"error": str(e)}
            )
            raise

        return self.get_experiment_status(experiment_id)

    def get_experiment_status(self, experiment_id: str) -> ExperimentStatusInfo:
        """Get experiment status.

        Args:
            experiment_id: Experiment identifier

        Returns:
            ExperimentStatus object
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp_info = self._experiments[experiment_id]
        controller = exp_info["controller"]

        # Get current state from controller
        state = controller.get_state()

        # Calculate progress
        progress = (
            (state["current_iteration"] / exp_info["total_iterations"]) * 100
            if exp_info["total_iterations"] > 0
            else 0
        )

        return ExperimentStatusInfo(
            experiment_id=experiment_id,
            status=exp_info["status"],
            current_iteration=state["current_iteration"],
            total_iterations=exp_info["total_iterations"],
            progress_percentage=progress,
            start_time=exp_info["start_time"],
            end_time=exp_info["end_time"],
            error_message=exp_info["error_message"],
        )

    def get_experiment_results(self, experiment_id: str) -> ExperimentResults:
        """Get experiment results.

        Args:
            experiment_id: Experiment identifier

        Returns:
            ExperimentResults object
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp_info = self._experiments[experiment_id]
        controller = exp_info["controller"]

        # Get state from controller
        state = controller.get_state()

        # Collect data files
        data_files = []
        exp_dir = exp_info["directory"]
        if exp_dir.exists():
            for file_path in exp_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in [
                    ".db",
                    ".csv",
                    ".json",
                    ".png",
                    ".html",
                ]:
                    data_files.append(str(file_path))

        return ExperimentResults(
            experiment_id=experiment_id,
            status=exp_info["status"],
            total_iterations=exp_info["total_iterations"],
            completed_iterations=state["current_iteration"],
            results_summary={
                "duration_seconds": (
                    (exp_info["end_time"] - exp_info["start_time"]).total_seconds()
                    if exp_info["start_time"] and exp_info["end_time"]
                    else 0
                )
            },
            data_files=data_files,
            analysis_available=len(data_files) > 0,
        )

    def analyze_simulation(self, simulation_id: str) -> AnalysisResults:
        """Analyze simulation results.

        Args:
            simulation_id: Simulation identifier

        Returns:
            AnalysisResults object
        """
        if simulation_id not in self._simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_info = self._simulations[simulation_id]
        db_path = sim_info["db_path"]

        if not os.path.exists(db_path):
            raise ValueError(f"Simulation database not found: {db_path}")

        # Create analyzer
        analyzer = SimulationAnalyzer(db_path=db_path)

        # Generate analysis
        analysis_dir = sim_info["directory"] / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        report_path = analysis_dir / "analysis_report.html"
        analyzer.generate_report(output_file=str(report_path))

        # Collect output files
        output_files = []
        if analysis_dir.exists():
            for file_path in analysis_dir.rglob("*"):
                if file_path.is_file():
                    output_files.append(str(file_path))

        return AnalysisResults(
            analysis_id=str(uuid.uuid4()),
            analysis_type="simulation_analysis",
            summary={"report_generated": True},
            output_files=output_files,
            metadata={"simulation_id": simulation_id},
        )

    def compare_simulations(self, simulation_ids: List[str]) -> ComparisonResults:
        """Compare multiple simulations.

        Args:
            simulation_ids: List of simulation identifiers

        Returns:
            ComparisonResults object
        """
        # Validate all simulations exist
        for sim_id in simulation_ids:
            if sim_id not in self._simulations:
                raise ValueError(f"Simulation {sim_id} not found")

        # Create comparison directory
        comparison_id = str(uuid.uuid4())
        comparison_dir = self.session_path / "comparisons" / comparison_id
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Run comparative analysis
        search_path = str(self.session_path / "simulations")
        compare_simulations(search_path=search_path, analysis_path=str(comparison_dir))

        # Collect output files
        output_files = []
        if comparison_dir.exists():
            for file_path in comparison_dir.rglob("*"):
                if file_path.is_file():
                    output_files.append(str(file_path))

        return ComparisonResults(
            comparison_id=comparison_id,
            simulation_ids=simulation_ids,
            comparison_summary={"comparison_completed": True},
            output_files=output_files,
            metadata={"created_at": datetime.now().isoformat()},
        )

    def subscribe_to_events(
        self,
        event_types: List[str],
        simulation_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> str:
        """Subscribe to events.

        Args:
            event_types: List of event types to subscribe to
            simulation_id: Optional simulation filter
            experiment_id: Optional experiment filter

        Returns:
            subscription_id: Unique identifier for the subscription
        """
        subscription_id = str(uuid.uuid4())

        subscription = EventSubscription(
            subscription_id=subscription_id,
            session_id=str(self.session_path),
            event_types=event_types,
            simulation_id=simulation_id,
            experiment_id=experiment_id,
        )

        self._event_subscriptions[subscription_id] = subscription

        logger.info(f"Created event subscription: {subscription_id}")
        return subscription_id

    def get_event_history(self, subscription_id: str) -> List[Event]:
        """Get event history for a subscription.

        Args:
            subscription_id: Subscription identifier

        Returns:
            List of matching events
        """
        if subscription_id not in self._event_subscriptions:
            raise ValueError(f"Subscription {subscription_id} not found")

        subscription = self._event_subscriptions[subscription_id]

        # Filter events based on subscription criteria
        matching_events = []
        for event in self._event_history:
            # Check event type
            if event.event_type not in subscription.event_types:
                continue

            # Check simulation filter
            if (
                subscription.simulation_id
                and event.simulation_id != subscription.simulation_id
            ):
                continue

            # Check experiment filter
            if (
                subscription.experiment_id
                and event.experiment_id != subscription.experiment_id
            ):
                continue

            matching_events.append(event)

        return matching_events

    def _emit_event(
        self,
        event_type: str,
        simulation_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        data: Optional[Dict] = None,
    ):
        """Emit an event.

        Args:
            event_type: Type of event
            simulation_id: Optional simulation identifier
            experiment_id: Optional experiment identifier
            data: Optional event data
        """
        with self._event_lock:
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(),
                session_id=str(self.session_path),
                simulation_id=simulation_id,
                experiment_id=experiment_id,
                data=data or {},
                message=f"Event: {event_type}",
            )

            self._event_history.append(event)

            # Keep only last 1000 events to prevent memory issues
            if len(self._event_history) > 1000:
                self._event_history = self._event_history[-1000:]

    def cleanup(self):
        """Clean up resources."""
        # Clean up simulations
        for sim_info in self._simulations.values():
            try:
                sim_info["controller"].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up simulation: {e}")

        # Clean up experiments
        for exp_info in self._experiments.values():
            try:
                exp_info["controller"].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up experiment: {e}")

        self._simulations.clear()
        self._experiments.clear()
        self._event_subscriptions.clear()
        self._event_history.clear()
