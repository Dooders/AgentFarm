"""Simulation controller for managing simulation execution and state.

This module provides a centralized controller for:
- Starting/stopping/pausing simulations
- Managing simulation state and configuration
- Coordinating between GUI, database, and simulation components
- Handling simulation events and updates
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional

from farm.core.config import SimulationConfig
from farm.core.environment import Environment
from farm.database.database import SimulationDatabase
from farm.database.models import Simulation

logger = logging.getLogger(__name__)


class SimulationController:
    """Controls and manages simulation execution.

    This controller provides centralized control over simulation execution including:
    - Starting/stopping/pausing simulations
    - Managing simulation state and configuration
    - Coordinating between GUI, database, and simulation components
    - Handling simulation events and updates

    Example usage:
        ```python
        # Initialize configuration and controller
        config = SimulationConfig.from_yaml("config.yaml")
        controller = SimulationController(config, "simulations/sim.db")

        # Register callbacks for monitoring
        def on_step(step_num):
            print(f"Completed step {step_num}")

        def on_status(status):
            print(f"Simulation status changed to: {status}")

        controller.register_step_callback("progress", on_step)
        controller.register_status_callback("status", on_status)

        # Run simulation
        try:
            controller.initialize_simulation()
            controller.start()

            # Get state while running
            while controller.is_running:
                state = controller.get_state()
                print(f"Current step: {state['current_step']}")
                time.sleep(1)

            # Can also control execution:
            controller.pause()  # Pause execution
            controller.start()  # Resume
            controller.stop()   # Stop execution

        finally:
            # Clean up resources
            controller.cleanup()
        ```

    The controller runs the simulation in a background thread and provides thread-safe
    access to simulation state and control. Callbacks can be registered to monitor
    simulation progress and status changes.
    """

    def __init__(self, config: SimulationConfig, db_path: str):
        """Initialize simulation controller.

        Args:
            config: Simulation configuration
            db_path: Path to simulation database
        """
        self.config = config
        self.db_path = db_path
        self.db = SimulationDatabase(db_path)

        # Simulation state
        self.environment: Optional[Environment] = None
        self.current_step = 0
        self.is_running = False
        self.is_paused = False
        self._stop_requested = False

        # Callbacks
        self.step_callbacks: Dict[str, Callable] = {}
        self.status_callbacks: Dict[str, Callable] = {}

        # Threading
        self._simulation_thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()

    def initialize_simulation(self) -> None:
        """Initialize a new simulation environment.

        Creates a new simulation record in the database and initializes the environment
        with the configured parameters. This must be called before starting the simulation.

        Raises:
            RuntimeError: If initialization fails
            DatabaseError: If database operations fail

        Example:
            ```python
            controller = SimulationController(config, "sim.db")
            controller.initialize_simulation()
            controller.start()  # Start after initialization
            ```
        """
        try:
            # Create simulation record
            simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.db.add_simulation_record(
                simulation_id=simulation_id,
                start_time=datetime.now(),
                status="initialized",
                parameters=self.config.to_dict(),
            )

            # Initialize environment
            self.environment = Environment(
                width=self.config.width,
                height=self.config.height,
                resource_distribution={
                    "type": "random",
                    "amount": self.config.initial_resources,
                },
                db_path=self.db_path,
                config=self.config,
                simulation_id=simulation_id,
            )
            self.current_step = 0
            self.is_running = False
            self.is_paused = False
            self._stop_requested = False

            logger.info("Simulation initialized successfully")
            self._notify_status_change("initialized")

        except Exception as e:
            logger.error(f"Error initializing simulation: {e}")
            raise

    def start(self) -> None:
        """Start or resume simulation execution.

        If the simulation is paused, resumes execution. If not started, begins execution
        in a new background thread. The simulation will run until completion or until
        stop() is called.

        Thread Safety:
            Safe to call from any thread. The simulation runs in a background thread.

        Example:
            ```python
            controller.start()  # Start execution
            controller.pause()  # Pause execution
            controller.start()  # Resume execution
            ```
        """
        if self._simulation_thread and self._simulation_thread.is_alive():
            if self.is_paused:
                self.is_paused = False
                logger.info("Resuming simulation")
                self._notify_status_change("resumed")
            return

        self.is_running = True
        self.is_paused = False
        self._stop_requested = False

        # Start simulation in separate thread
        self._simulation_thread = threading.Thread(target=self._run_simulation)
        self._simulation_thread.start()

        logger.info("Simulation started")
        self._notify_status_change("started")

    def pause(self) -> None:
        """Pause simulation execution.

        Temporarily halts simulation execution. The simulation can be resumed by calling
        start(). State is preserved while paused.

        Thread Safety:
            Safe to call from any thread. The simulation will pause at the next step.

        Example:
            ```python
            controller.pause()  # Pause execution
            # ... do something while paused ...
            controller.start()  # Resume execution
            ```
        """
        self.is_paused = True
        logger.info("Simulation paused")
        self._notify_status_change("paused")

    def stop(self) -> None:
        """Stop simulation execution.

        Permanently stops simulation execution. The simulation cannot be resumed after
        stopping - a new simulation must be initialized.

        Thread Safety:
            Safe to call from any thread. Blocks until simulation thread terminates.

        Example:
            ```python
            controller.stop()  # Stop execution
            controller.cleanup()  # Clean up after stopping
            ```
        """
        self._stop_requested = True
        self.is_running = False
        self.is_paused = False

        if self._simulation_thread:
            self._simulation_thread.join()

        logger.info("Simulation stopped")
        self._notify_status_change("stopped")

    def step(self) -> None:
        """Execute a single simulation step.

        Advances the simulation by one step, updating all agents and the environment.
        Notifies step callbacks after completion.

        Thread Safety:
            Thread-safe but should typically only be called from the simulation thread.

        Raises:
            RuntimeError: If simulation is not initialized
            SimulationError: If step execution fails

        Example:
            ```python
            controller.initialize_simulation()
            controller.step()  # Execute one step
            state = controller.get_state()  # Get updated state
            ```
        """
        if not self.environment:
            raise RuntimeError("Simulation not initialized")

        try:
            with self._thread_lock:
                # Execute environment step
                self.environment.step(
                    None
                )  # Environment step doesn't require action parameter
                self.current_step += 1

                # Notify step callbacks
                self._notify_step_complete()

        except Exception as e:
            logger.error(f"Error executing simulation step: {e}")
            self.stop()
            raise

    def _run_simulation(self) -> None:
        """Main simulation loop."""
        try:
            while self.is_running and not self._stop_requested:
                if self.is_paused:
                    time.sleep(0.1)  # Reduce CPU usage while paused
                    continue

                if self.current_step >= self.config.simulation_steps:
                    logger.info("Simulation completed")
                    self.stop()
                    self._notify_status_change("completed")
                    break

                self.step()

        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            self.stop()
            self._notify_status_change("error")
            raise

        finally:
            # Ensure database is updated with final state
            if self.environment:
                self.environment.cleanup()  # Use cleanup instead of save_state

    def register_step_callback(self, name: str, callback: Callable) -> None:
        """Register callback for step completion.

        The callback will be called after each simulation step with the current step
        number as an argument.

        Args:
            name: Unique identifier for this callback
            callback: Function taking a single int argument (step number)

        Thread Safety:
            Safe to call from any thread. Callbacks may be called from simulation thread.

        Example:
            ```python
            def on_step(step_num):
                print(f"Completed step {step_num}")
            controller.register_step_callback("progress", on_step)
            ```
        """
        self.step_callbacks[name] = callback

    def register_status_callback(self, name: str, callback: Callable) -> None:
        """Register callback for simulation status changes.

        The callback will be called whenever simulation status changes with the new
        status as an argument.

        Args:
            name: Unique identifier for this callback
            callback: Function taking a single str argument (new status)

        Thread Safety:
            Safe to call from any thread. Callbacks may be called from simulation thread.

        Status Values:
            - "initialized": Simulation is ready to start
            - "started": Simulation has started/resumed
            - "paused": Simulation is temporarily paused
            - "stopped": Simulation has been stopped
            - "completed": Simulation finished all steps
            - "error": An error occurred during simulation

        Example:
            ```python
            def on_status(status):
                print(f"Simulation status: {status}")
            controller.register_status_callback("status_monitor", on_status)
            ```
        """
        self.status_callbacks[name] = callback

    def _notify_step_complete(self) -> None:
        """Notify all registered step callbacks."""
        for callback in self.step_callbacks.values():
            try:
                callback(self.current_step)
            except Exception as e:
                logger.error(f"Error in step callback: {e}")

    def _notify_status_change(self, status: str) -> None:
        """Notify all registered status callbacks.

        Args:
            status: New simulation status
        """
        for callback in self.status_callbacks.values():
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def get_state(self) -> Dict:
        """Get current simulation state.

        Returns a dictionary containing the current state of the simulation including
        step count, running status, and environment statistics.

        Thread Safety:
            Safe to call from any thread.

        Returns:
            Dict with keys:
                - current_step: Current simulation step number
                - total_steps: Total steps to run
                - is_running: Whether simulation is running
                - is_paused: Whether simulation is paused
                - status: Current status string
                - agent_count: Number of active agents
                - resource_count: Number of resources

        Example:
            ```python
            state = controller.get_state()
            print(f"Step {state['current_step']} of {state['total_steps']}")
            print(f"Agents: {state['agent_count']}")
            ```
        """
        return {
            "current_step": self.current_step,
            "total_steps": self.config.simulation_steps,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "status": "running" if self.is_running else "stopped",
            "agent_count": len(self.environment.agents) if self.environment else 0,
            "resource_count": (
                len(self.environment.resources) if self.environment else 0
            ),
        }

    def cleanup(self) -> None:
        """Clean up simulation resources.

        Stops the simulation if running and cleans up all resources including database
        connections and environment state. Should be called when simulation is no longer
        needed.

        Thread Safety:
            Safe to call from any thread. Blocks until cleanup is complete.

        Raises:
            Exception: If cleanup fails

        Example:
            ```python
            try:
                controller.start()
                # ... run simulation ...
            finally:
                controller.cleanup()  # Always clean up
            ```
        """
        try:
            if self.is_running:
                self.stop()

            if self.environment:
                self.environment.cleanup()

            if self.db:
                self.db.close()

            logger.info("Simulation cleanup complete")

        except Exception as e:
            logger.error(f"Error during simulation cleanup: {e}")
            raise

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and ensure cleanup."""
        self.cleanup()
        # Don't suppress exceptions
        return False
