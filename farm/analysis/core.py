"""
Core implementation classes for the analysis module system.

This module provides concrete base classes that implement the protocols
defined in protocols.py. These classes provide default functionality that
can be extended by specific analysis modules.
"""

from typing import List, Dict, Any, Optional, Callable, Iterator
from pathlib import Path
from enum import Enum
import pandas as pd
from farm.utils.logging_config import get_logger

from farm.analysis.protocols import (
    DataProcessor,
    DataValidator,
    AnalysisFunction,
    AnalysisModule as AnalysisModuleProtocol,
)
from farm.analysis.common.context import AnalysisContext
from farm.analysis.exceptions import AnalysisFunctionError, ConfigurationError, DataProcessingError
from farm.analysis.validation import CompositeValidator


logger = get_logger(__name__)


class ErrorHandlingMode(Enum):
    """Error handling modes for analysis functions."""

    CONTINUE = "continue"  # Continue on error (default)
    FAIL_FAST = "fail_fast"  # Stop on first error
    COLLECT = "collect"  # Continue and collect all errors


class BaseAnalysisModule:
    """Base implementation of an analysis module.

    Provides common functionality for registering and running analysis functions.
    Subclasses should override register_functions() and get_data_processor().
    """

    def __init__(self, name: str, description: str):
        """Initialize the analysis module.

        Args:
            name: Unique identifier for this module
            description: Human-readable description
        """
        self._name = name
        self._description = description
        self._functions: Dict[str, AnalysisFunction] = {}
        self._groups: Dict[str, List[AnalysisFunction]] = {}
        self._registered = False
        self._validator: Optional[DataValidator] = None
        self._error_mode: ErrorHandlingMode = ErrorHandlingMode.CONTINUE

    @property
    def name(self) -> str:
        """Module name."""
        return self._name

    @property
    def description(self) -> str:
        """Module description."""
        return self._description

    def register_functions(self) -> None:
        """Register all analysis functions for this module.

        Subclasses must override this to populate self._functions and self._groups.
        Example:
            self._functions = {
                "plot_distribution": plot_distribution_func,
                "compute_stats": compute_stats_func,
            }
            self._groups = {
                "all": list(self._functions.values()),
                "plots": [plot_distribution_func],
                "metrics": [compute_stats_func],
            }
        """
        raise NotImplementedError("Subclasses must implement register_functions()")

    def get_data_processor(self) -> DataProcessor:
        """Get the data processor for this module.

        Returns:
            DataProcessor instance that transforms raw data for this analysis

        Raises:
            NotImplementedError: If subclass doesn't implement this
        """
        raise NotImplementedError("Subclasses must implement get_data_processor()")

    def get_validator(self) -> Optional[DataValidator]:
        """Get the data validator for this module.

        Returns:
            DataValidator instance or None if no validation needed
        """
        return self._validator

    def set_validator(self, validator: DataValidator) -> None:
        """Set the data validator for this module.

        Args:
            validator: DataValidator instance
        """
        self._validator = validator

    def set_error_mode(self, mode: ErrorHandlingMode) -> None:
        """Set the error handling mode for analysis functions.

        Args:
            mode: Error handling mode (CONTINUE, FAIL_FAST, or COLLECT)
        """
        self._error_mode = mode

    def get_error_mode(self) -> ErrorHandlingMode:
        """Get the current error handling mode.

        Returns:
            Current error handling mode
        """
        return self._error_mode

    def get_analysis_functions(self, group: str = "all") -> List[AnalysisFunction]:
        """Get analysis functions by group.

        Args:
            group: Function group name

        Returns:
            List of analysis functions in the group
        """
        self._ensure_registered()
        return self._groups.get(group, [])

    def get_function_groups(self) -> List[str]:
        """Get available function group names.

        Returns:
            List of group names
        """
        self._ensure_registered()
        return list(self._groups.keys())

    def get_function_groups_dict(self) -> Dict[str, List[AnalysisFunction]]:
        """Get available function groups as dictionary.

        Returns:
            Dictionary mapping group names to lists of functions
        """
        self._ensure_registered()
        return self._groups.copy()

    def get_function(self, name: str) -> Optional[AnalysisFunction]:
        """Get a specific analysis function by name.

        Args:
            name: Function name

        Returns:
            Analysis function or None if not found
        """
        self._ensure_registered()
        return self._functions.get(name)

    def get_function_names(self) -> List[str]:
        """Get all available function names.

        Returns:
            List of function names
        """
        self._ensure_registered()
        return list(self._functions.keys())

    def get_functions(self) -> List[AnalysisFunction]:
        """Get all analysis functions.

        Returns:
            List of all analysis functions
        """
        self._ensure_registered()
        return list(self._functions.values())

    def supports_database(self) -> bool:
        """Whether this module uses database for intermediate storage.

        Returns:
            False by default, override in subclass if needed
        """
        return False

    def get_db_filename(self) -> Optional[str]:
        """Get database filename if module uses database.

        Returns:
            Database filename or None
        """
        return None

    def get_db_loader(self) -> Optional[Callable]:
        """Get database loader function if module uses database.

        Returns:
            Database loader function or None
        """
        return None

    def run_analysis(
        self,
        experiment_path: Path,
        output_path: Path,
        group: str = "all",
        processor_kwargs: Optional[Dict[str, Any]] = None,
        analysis_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        error_mode: Optional[ErrorHandlingMode] = None,
    ) -> tuple[Path, Optional[pd.DataFrame]]:
        """Run complete analysis workflow.

        Args:
            experiment_path: Path to experiment data
            output_path: Path to save results
            group: Function group to run
            processor_kwargs: Arguments for data processor
            analysis_kwargs: Arguments for specific analysis functions
            progress_callback: Optional progress callback
            error_mode: Override error handling mode for this run

        Returns:
            Tuple of (output_path, processed_dataframe, list_of_errors)
        """
        self._ensure_registered()

        # Ensure paths are Path objects
        experiment_path = Path(experiment_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        processor_kwargs = processor_kwargs or {}
        analysis_kwargs = analysis_kwargs or {}

        # Create analysis context
        ctx = AnalysisContext(
            output_path=output_path,
            progress_callback=progress_callback,
            metadata={"experiment_path": str(experiment_path)},
        )

        ctx.report_progress("Starting data processing", 0.1)

        # Process data
        try:
            data_processor = self.get_data_processor()

            # Handle database if module supports it
            if self.supports_database():
                db_filename = self.get_db_filename()
                if db_filename:
                    db_path = output_path / db_filename
                    db_uri = f"sqlite:///{db_path}"
                    processor_kwargs["save_to_db"] = True
                    processor_kwargs["db_path"] = db_uri
                    ctx.logger.info(f"Using database: {db_uri}")

            df = data_processor.process(experiment_path, **processor_kwargs)

            # Load from database if needed
            if df is None and self.supports_database():
                db_loader = self.get_db_loader()
                if db_loader:
                    ctx.logger.info("Loading data from database...")
                    df = db_loader(db_uri)

        except Exception as e:
            raise DataProcessingError(f"Data processing failed: {e}", step="processing") from e

        # Validate data if validator is set (even if empty, to catch insufficient data)
        validator = self.get_validator()
        if validator:
            try:
                ctx.logger.info("Validating data...")
                validator.validate(df)
            except Exception as e:
                ctx.logger.error(f"Data validation failed: {e}")
                raise

        # Check for empty data after validation (validation should handle this)
        is_empty = False
        if df is None:
            is_empty = True
        elif isinstance(df, pd.DataFrame):
            is_empty = df.empty
        elif isinstance(df, dict):
            # For dict structures (like spatial data), check if all DataFrames are empty
            is_empty = all(pd.DataFrame(v).empty for v in df.values() if isinstance(v, (pd.DataFrame, list)))
        else:
            # For other types, assume not empty
            is_empty = False

        if is_empty:
            ctx.logger.warning("No data produced by processor")
            return output_path, None, []

        # Report progress with appropriate record count
        if isinstance(df, pd.DataFrame):
            record_count = len(df)
        elif isinstance(df, dict):
            record_count = sum(len(pd.DataFrame(v)) for v in df.values() if isinstance(v, (pd.DataFrame, list)))
        else:
            record_count = 1  # Fallback for other types

        ctx.report_progress(f"Processed {record_count} records", 0.3)

        # Log summary statistics
        ctx.logger.info(f"Analyzed {record_count} records")
        if isinstance(df, pd.DataFrame):
            ctx.logger.info(f"Columns: {list(df.columns)}")
        elif isinstance(df, dict):
            for key, value in df.items():
                if isinstance(value, pd.DataFrame):
                    ctx.logger.info(f"DataFrame '{key}': {len(value)} rows, columns: {list(value.columns)}")
                else:
                    ctx.logger.info(f"Data '{key}': {type(value)}")

        # Get functions to run
        functions = self.get_analysis_functions(group)
        if not functions:
            ctx.logger.warning(f"No functions found for group '{group}'")
            return output_path, df, []

        ctx.report_progress(f"Running {len(functions)} analysis functions", 0.4)

        # Determine error handling mode
        active_error_mode = error_mode if error_mode is not None else self._error_mode
        errors_collected: List[AnalysisFunctionError] = []

        # Run each analysis function
        for i, func in enumerate(functions):
            func_name = func.__name__ if hasattr(func, "__name__") else str(func)

            try:
                ctx.logger.info(f"Running {func_name}...")

                # Get function-specific kwargs
                func_kwargs = analysis_kwargs.get(func_name, {})

                # Call function with standardized signature
                func(df=df, ctx=ctx, **func_kwargs)

                progress = 0.4 + (0.5 * (i + 1) / len(functions))
                ctx.report_progress(f"Completed {func_name}", progress)

            except Exception as e:
                error = AnalysisFunctionError(func_name, e)
                ctx.logger.error(f"Error in {func_name}: {e}", exc_info=True)

                # Handle error based on mode
                if active_error_mode == ErrorHandlingMode.FAIL_FAST:
                    ctx.logger.error("Stopping analysis due to FAIL_FAST mode")
                    raise error
                elif active_error_mode == ErrorHandlingMode.COLLECT:
                    errors_collected.append(error)
                    ctx.logger.warning("Collected error, continuing with remaining functions")
                else:  # CONTINUE mode (default)
                    ctx.logger.warning(f"Skipping {func_name}, continuing with remaining functions")
                    continue

        ctx.report_progress("Analysis complete", 1.0)
        ctx.logger.info(f"Results saved to: {output_path}")

        if errors_collected:
            ctx.logger.warning(f"Analysis completed with {len(errors_collected)} error(s)")

        return output_path, df

    def _ensure_registered(self) -> None:
        """Ensure functions are registered (lazy registration)."""
        if not self._registered:
            self.register_functions()
            self._registered = True

    def get_info(self) -> Dict[str, Any]:
        """Get module information.

        Returns:
            Dictionary with module metadata
        """
        self._ensure_registered()
        return {
            "name": self.name,
            "description": self.description,
            "function_groups": self.get_function_groups_dict(),
            "functions": self.get_function_names(),
            "supports_database": self.supports_database(),
            "db_filename": self.get_db_filename(),
        }


class SimpleDataProcessor:
    """Simple data processor that applies a processing function."""

    def __init__(self, process_func: Callable[..., pd.DataFrame]):
        """Initialize with a processing function.

        Args:
            process_func: Function that processes data
        """
        self.process_func = process_func

    def process(self, data: Any, **kwargs) -> pd.DataFrame:
        """Process data using the configured function.

        Args:
            data: Input data (path, DataFrame, etc.)
            **kwargs: Additional arguments

        Returns:
            Processed DataFrame
        """
        return self.process_func(data, **kwargs)


class ChainedDataProcessor:
    """Data processor that chains multiple processors together."""

    def __init__(self, processors: List[DataProcessor]):
        """Initialize with list of processors.

        Args:
            processors: List of processors to chain
        """
        self.processors = processors

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process data through all processors in sequence.

        Args:
            data: Input DataFrame
            **kwargs: Additional arguments

        Returns:
            Processed DataFrame
        """
        result = data
        for processor in self.processors:
            result = processor.process(result, **kwargs)
        return result


def make_analysis_function(func: Callable, name: Optional[str] = None) -> AnalysisFunction:
    """Wrap a function to match the AnalysisFunction protocol.

    Args:
        func: Function to wrap
        name: Optional custom name for the function

    Returns:
        Wrapped function matching AnalysisFunction protocol
    """

    def wrapped(df: pd.DataFrame, ctx: AnalysisContext, **kwargs) -> Any:
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Determine what arguments to pass based on function signature
        call_kwargs = {}

        # Check each parameter and provide appropriate value
        for param_name in params:
            if param_name == "ctx" or param_name == "context":
                call_kwargs[param_name] = ctx
            elif param_name == "output_path":
                call_kwargs[param_name] = str(ctx.output_path)
            elif param_name == "experiment_path":
                # Get experiment path from context metadata
                call_kwargs[param_name] = ctx.metadata.get("experiment_path", str(ctx.output_path.parent))
            elif param_name in ["df", "data", "dataframe"]:
                call_kwargs[param_name] = df
            elif param_name in ["patterns_df", "spatial_data", "segmentation_data", "efficiency_data"]:
                # These are computed dataframes/dicts that should be derived from df
                # For now, pass the main df and let the function handle it
                call_kwargs[param_name] = df
            # Skip **kwargs parameter as it's handled separately

        # Add any remaining kwargs
        for k, v in kwargs.items():
            if k not in call_kwargs:
                call_kwargs[k] = v

        return func(**call_kwargs)

    if name:
        wrapped.__name__ = name
        wrapped.name = name
    elif hasattr(func, "__name__"):
        wrapped.__name__ = func.__name__
        wrapped.name = func.__name__

    return wrapped
