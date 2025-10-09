"""Analysis controller for managing analysis execution and state.

This module provides a centralized controller for:
- Starting/stopping/pausing analysis jobs
- Managing analysis state and configuration
- Coordinating between API, database, and analysis components
- Handling analysis events and progress updates
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Any, List

from farm.analysis.service import AnalysisRequest, AnalysisResult, AnalysisService
from farm.core.services import IConfigService
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisController:
    """Controls and manages analysis execution.

    This controller provides centralized control over analysis execution including:
    - Starting/stopping/pausing analysis jobs
    - Managing analysis state and progress
    - Coordinating between API, service layer, and analysis modules
    - Handling analysis events and updates

    Example usage:
        ```python
        from farm.core.services import EnvConfigService
        
        # Initialize controller
        config_service = EnvConfigService()
        controller = AnalysisController(config_service)

        # Register callbacks for monitoring
        def on_progress(message, percent):
            print(f"Progress: {message} ({percent*100:.1f}%)")

        def on_status(status):
            print(f"Analysis status changed to: {status}")

        controller.register_progress_callback("progress", on_progress)
        controller.register_status_callback("status", on_status)

        # Create analysis request
        request = AnalysisRequest(
            module_name="genesis",
            experiment_path=Path("results/experiment_1"),
            output_path=Path("results/analysis/genesis"),
            group="all"
        )

        # Run analysis
        try:
            controller.initialize_analysis(request)
            controller.start()

            # Get state while running
            while controller.is_running:
                state = controller.get_state()
                print(f"Status: {state['status']}")
                time.sleep(1)

            # Get results
            result = controller.get_result()
            if result and result.success:
                print(f"Analysis complete! Results in {result.output_path}")
            else:
                print(f"Analysis failed: {result.error if result else 'Unknown error'}")

        finally:
            controller.cleanup()
        ```

    The controller runs the analysis in a background thread and provides thread-safe
    access to analysis state and control. Callbacks can be registered to monitor
    analysis progress and status changes.
    """

    def __init__(
        self,
        config_service: IConfigService,
        cache_dir: Optional[Path] = None
    ):
        """Initialize analysis controller.

        Args:
            config_service: Configuration service for analysis modules
            cache_dir: Optional directory for caching results
        """
        self.service = AnalysisService(
            config_service=config_service,
            cache_dir=cache_dir,
            auto_register=True
        )

        # Analysis state
        self.request: Optional[AnalysisRequest] = None
        self.result: Optional[AnalysisResult] = None
        self.analysis_id: Optional[str] = None
        self.is_running = False
        self.is_paused = False
        self._stop_requested = False

        # Progress tracking
        self.current_progress = 0.0
        self.current_message = ""

        # Callbacks
        self.progress_callbacks: Dict[str, Callable] = {}
        self.status_callbacks: Dict[str, Callable] = {}

        # Threading
        self._analysis_thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()

    def initialize_analysis(self, request: AnalysisRequest) -> None:
        """Initialize a new analysis job.

        Validates the request and prepares for execution. This must be called
        before starting the analysis.

        Args:
            request: Analysis request with module name, paths, and parameters

        Raises:
            ConfigurationError: If request is invalid
            ModuleNotFoundError: If analysis module doesn't exist

        Example:
            ```python
            request = AnalysisRequest(
                module_name="dominance",
                experiment_path=Path("results/sim_001"),
                output_path=Path("results/analysis/dominance")
            )
            controller.initialize_analysis(request)
            controller.start()
            ```
        """
        try:
            # Validate request
            self.service.validate_request(request)

            # Set up progress callback
            request.progress_callback = self._progress_handler

            # Generate analysis ID
            self.analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Store request
            self.request = request
            self.result = None
            self.current_progress = 0.0
            self.current_message = "Initialized"
            self.is_running = False
            self.is_paused = False
            self._stop_requested = False

            logger.info(
                "analysis_initialized",
                analysis_id=self.analysis_id,
                module_name=request.module_name,
                group=request.group
            )
            self._notify_status_change("initialized")

        except Exception as e:
            logger.error(
                "analysis_initialization_failed",
                analysis_id=self.analysis_id if self.analysis_id else "unknown",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def start(self) -> None:
        """Start or resume analysis execution.

        If the analysis is paused, resumes execution. If not started, begins execution
        in a new background thread. The analysis will run until completion or until
        stop() is called.

        Thread Safety:
            Safe to call from any thread. The analysis runs in a background thread.

        Example:
            ```python
            controller.start()  # Start execution
            controller.pause()  # Pause execution
            controller.start()  # Resume execution
            ```
        """
        if not self.request:
            raise RuntimeError("Analysis not initialized. Call initialize_analysis() first.")

        if self._analysis_thread and self._analysis_thread.is_alive():
            if self.is_paused:
                self.is_paused = False
                logger.info(
                    "analysis_resumed",
                    analysis_id=self.analysis_id,
                    progress=self.current_progress
                )
                self._notify_status_change("resumed")
            return

        self.is_running = True
        self.is_paused = False
        self._stop_requested = False

        # Start analysis in separate thread
        self._analysis_thread = threading.Thread(target=self._run_analysis)
        self._analysis_thread.start()

        logger.info(
            "analysis_started",
            analysis_id=self.analysis_id,
            module_name=self.request.module_name
        )
        self._notify_status_change("started")

    def pause(self) -> None:
        """Pause analysis execution.

        Temporarily halts analysis execution. The analysis can be resumed by calling
        start(). State is preserved while paused.

        Note: The pause will take effect at the next progress callback. Some analysis
        functions may not support pausing mid-execution.

        Thread Safety:
            Safe to call from any thread.

        Example:
            ```python
            controller.pause()  # Pause execution
            # ... do something while paused ...
            controller.start()  # Resume execution
            ```
        """
        self.is_paused = True
        logger.info(
            "analysis_paused",
            analysis_id=self.analysis_id,
            progress=self.current_progress
        )
        self._notify_status_change("paused")

    def stop(self) -> None:
        """Stop analysis execution.

        Permanently stops analysis execution. The analysis cannot be resumed after
        stopping - a new analysis must be initialized.

        Thread Safety:
            Safe to call from any thread. Blocks until analysis thread terminates.

        Example:
            ```python
            controller.stop()  # Stop execution
            controller.cleanup()  # Clean up after stopping
            ```
        """
        self._stop_requested = True
        self.is_running = False
        self.is_paused = False

        if self._analysis_thread and threading.current_thread() is not self._analysis_thread:
            self._analysis_thread.join(timeout=5.0)
            if self._analysis_thread.is_alive():
                logger.warning(
                    "analysis_thread_timeout",
                    analysis_id=self.analysis_id,
                    progress=self.current_progress,
                    message="Analysis thread did not terminate within timeout.",
                )

        logger.info(
            "analysis_stopped",
            analysis_id=self.analysis_id,
            progress=self.current_progress
        )
        self._notify_status_change("stopped")

    def _run_analysis(self) -> None:
        """Main analysis execution loop (runs in background thread)."""
        try:
            if not self.request:
                raise RuntimeError("No analysis request available")

            # Run the analysis (no lock held to avoid deadlock with _progress_handler)
            result = self.service.run(self.request)
            
            # Update result with lock
            with self._thread_lock:
                self.result = result

            if result.success:
                self.current_progress = 1.0
                self.current_message = "Analysis complete"
                logger.info(
                    "analysis_completed",
                    analysis_id=self.analysis_id,
                    module_name=self.request.module_name,
                    execution_time=result.execution_time,
                    output_path=str(result.output_path)
                )
                self._notify_status_change("completed")
            else:
                logger.error(
                    "analysis_failed",
                    analysis_id=self.analysis_id,
                    module_name=self.request.module_name,
                    error=result.error
                )
                self._notify_status_change("error")

            self.is_running = False

        except Exception as e:
            logger.error(
                "analysis_loop_error",
                analysis_id=self.analysis_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self.is_running = False
            self._notify_status_change("error")
            raise

    def _progress_handler(self, message: str, progress: float) -> None:
        """Handle progress updates from analysis service.

        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
        """
        # Check if stop was requested
        if self._stop_requested:
            raise InterruptedError("Analysis stop requested")

        # Wait if paused
        while self.is_paused and not self._stop_requested:
            time.sleep(0.1)

        # Update state
        with self._thread_lock:
            self.current_progress = progress
            self.current_message = message

        # Notify callbacks
        self._notify_progress(message, progress)

    def register_progress_callback(self, name: str, callback: Callable) -> None:
        """Register callback for progress updates.

        The callback will be called during analysis execution with progress updates.

        Args:
            name: Unique identifier for this callback
            callback: Function taking (message: str, progress: float) arguments

        Thread Safety:
            Safe to call from any thread. Callbacks may be called from analysis thread.

        Example:
            ```python
            def on_progress(message, progress):
                print(f"{message}: {progress*100:.1f}%")
            controller.register_progress_callback("console", on_progress)
            ```
        """
        self.progress_callbacks[name] = callback

    def register_status_callback(self, name: str, callback: Callable) -> None:
        """Register callback for analysis status changes.

        The callback will be called whenever analysis status changes with the new
        status as an argument.

        Args:
            name: Unique identifier for this callback
            callback: Function taking a single str argument (new status)

        Thread Safety:
            Safe to call from any thread. Callbacks may be called from analysis thread.

        Status Values:
            - "initialized": Analysis is ready to start
            - "started": Analysis has started/resumed
            - "paused": Analysis is temporarily paused
            - "stopped": Analysis has been stopped
            - "completed": Analysis finished successfully
            - "error": An error occurred during analysis

        Example:
            ```python
            def on_status(status):
                print(f"Analysis status: {status}")
            controller.register_status_callback("status_monitor", on_status)
            ```
        """
        self.status_callbacks[name] = callback

    def unregister_progress_callback(self, name: str) -> None:
        """Remove a registered progress callback.

        Args:
            name: Name of callback to remove
        """
        self.progress_callbacks.pop(name, None)

    def unregister_status_callback(self, name: str) -> None:
        """Remove a registered status callback.

        Args:
            name: Name of callback to remove
        """
        self.status_callbacks.pop(name, None)

    def _notify_progress(self, message: str, progress: float) -> None:
        """Notify all registered progress callbacks.

        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
        """
        for callback in self.progress_callbacks.values():
            try:
                callback(message, progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def _notify_status_change(self, status: str) -> None:
        """Notify all registered status callbacks.

        Args:
            status: New analysis status
        """
        for callback in self.status_callbacks.values():
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Get current analysis state.

        Returns a dictionary containing the current state of the analysis including
        progress, status, and configuration.

        Thread Safety:
            Safe to call from any thread.

        Returns:
            Dict with keys:
                - analysis_id: Unique analysis identifier
                - module_name: Analysis module being run
                - is_running: Whether analysis is running
                - is_paused: Whether analysis is paused
                - status: Current status string
                - progress: Current progress (0.0 to 1.0)
                - message: Current progress message
                - cache_hit: Whether result was from cache (if completed)
                - execution_time: Time taken (if completed)

        Example:
            ```python
            state = controller.get_state()
            print(f"Progress: {state['progress']*100:.1f}%")
            print(f"Status: {state['status']}")
            ```
        """
        state = {
            "analysis_id": self.analysis_id,
            "module_name": self.request.module_name if self.request else None,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "progress": self.current_progress,
            "message": self.current_message,
        }

        # Determine status
        if self.result and self.result.success:
            state["status"] = "completed"
            state["cache_hit"] = self.result.cache_hit
            state["execution_time"] = self.result.execution_time
        elif self.result and not self.result.success:
            state["status"] = "error"
            state["error"] = self.result.error
        elif self.is_paused:
            state["status"] = "paused"
        elif self.is_running:
            state["status"] = "running"
        else:
            state["status"] = "stopped"

        return state

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for analysis to complete.

        This is the recommended way to wait for an analysis to finish, rather than
        polling is_running or accessing private thread members.

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Returns:
            True if analysis completed within timeout, False otherwise

        Thread Safety:
            Safe to call from any thread except the analysis thread itself.

        Example:
            ```python
            controller.start()
            if controller.wait_for_completion(timeout=300):
                result = controller.get_result()
                print("Analysis completed!")
            else:
                print("Analysis timed out")
            ```
        """
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=timeout)
            return not self._analysis_thread.is_alive()
        return True

    def get_result(self) -> Optional[AnalysisResult]:
        """Get analysis result.

        Returns the result object after analysis completes. Returns None if
        analysis hasn't completed yet.

        Thread Safety:
            Safe to call from any thread.

        Returns:
            AnalysisResult if complete, None otherwise

        Example:
            ```python
            controller.start()
            controller.wait_for_completion()
            result = controller.get_result()
            if result and result.success:
                print(f"Analysis saved to {result.output_path}")
                print(f"Processed {len(result.dataframe)} rows")
            ```
        """
        return self.result

    def list_available_modules(self) -> List[Dict[str, str]]:
        """List all available analysis modules.

        Returns:
            List of module info dictionaries with name and description

        Example:
            ```python
            modules = controller.list_available_modules()
            for module in modules:
                print(f"{module['name']}: {module['description']}")
            ```
        """
        return self.service.list_modules()

    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """Get detailed information about an analysis module.

        Args:
            module_name: Name of the module

        Returns:
            Module information including available function groups

        Example:
            ```python
            info = controller.get_module_info("dominance")
            print(f"Function groups: {info['function_groups']}")
            print(f"Functions: {info['functions']}")
            ```
        """
        return self.service.get_module_info(module_name)

    def clear_cache(self) -> int:
        """Clear the analysis results cache.

        Returns:
            Number of cache entries cleared

        Example:
            ```python
            cleared = controller.clear_cache()
            print(f"Cleared {cleared} cached results")
            ```
        """
        return self.service.clear_cache()

    def cleanup(self) -> None:
        """Clean up analysis resources.

        Stops the analysis if running and cleans up all resources. Should be called
        when controller is no longer needed.

        Thread Safety:
            Safe to call from any thread. Blocks until cleanup is complete.

        Example:
            ```python
            try:
                controller.start()
                # ... run analysis ...
            finally:
                controller.cleanup()  # Always clean up
            ```
        """
        try:
            if self.is_running:
                self.stop()

            logger.info("Analysis controller cleanup complete")

        except Exception as e:
            logger.error(f"Error during analysis cleanup: {e}")
            raise

    def __del__(self):
        """Ensure cleanup on deletion.
        
        Note: __del__ is not guaranteed to be called in Python, and the context
        manager pattern (__enter__/__exit__) is the preferred cleanup method.
        This method is provided as a safety net for cases where the context
        manager is not used, but should not be relied upon for critical cleanup.
        """
        try:
            self.cleanup()
        except Exception:
            pass  # Suppress exceptions in __del__ to avoid issues during interpreter shutdown

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and ensure cleanup."""
        self.cleanup()
        # Don't suppress exceptions
        return False
