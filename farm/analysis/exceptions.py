"""
Custom exceptions for the analysis module system.

Provides specific exception types for better error handling and debugging.
"""

from typing import List, Set, Optional, Dict


class AnalysisError(Exception):
    """Base exception for all analysis-related errors."""
    pass


class DataValidationError(AnalysisError):
    """Raised when data doesn't meet required schema or constraints."""

    def __init__(
        self,
        message: str,
        missing_columns: Optional[Set[str]] = None,
        invalid_columns: Optional[Dict[str, str]] = None
    ):
        super().__init__(message)
        self.missing_columns = missing_columns or set()
        self.invalid_columns = invalid_columns or {}

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.missing_columns:
            parts.append(f"Missing columns: {sorted(self.missing_columns)}")
        if self.invalid_columns:
            parts.append(f"Invalid columns: {self.invalid_columns}")
        return "\n".join(parts)


class ModuleNotFoundError(AnalysisError):
    """Raised when requested analysis module doesn't exist."""

    def __init__(self, module_name: str, available_modules: List[str]):
        self.module_name = module_name
        self.available_modules = available_modules
        super().__init__(
            f"Analysis module '{module_name}' not found. "
            f"Available modules: {', '.join(sorted(available_modules))}"
        )


class DataLoaderError(AnalysisError):
    """Raised when data loading fails."""
    pass


class DataProcessingError(AnalysisError):
    """Raised when data processing fails."""

    def __init__(self, message: str, step: Optional[str] = None):
        self.step = step
        if step:
            message = f"[{step}] {message}"
        super().__init__(message)


class AnalysisFunctionError(AnalysisError):
    """Raised when an analysis function fails."""

    def __init__(self, function_name: str, original_error: Exception):
        self.function_name = function_name
        self.original_error = original_error
        super().__init__(
            f"Analysis function '{function_name}' failed: {original_error}"
        )


class ConfigurationError(AnalysisError):
    """Raised when module or analysis configuration is invalid."""
    pass


class InsufficientDataError(AnalysisError):
    """Raised when there's not enough data to perform analysis."""

    def __init__(self, message: str, required_rows: Optional[int] = None, actual_rows: Optional[int] = None):
        self.required_rows = required_rows
        self.actual_rows = actual_rows
        if required_rows is not None and actual_rows is not None:
            message = f"{message} (required: {required_rows}, actual: {actual_rows})"
        super().__init__(message)


class VisualizationError(AnalysisError):
    """Raised when chart/plot creation fails."""
    pass


class DatabaseError(AnalysisError):
    """Raised when database operations fail."""

    def __init__(self, message: str, db_path: Optional[str] = None):
        self.db_path = db_path
        if db_path:
            message = f"{message} (database: {db_path})"
        super().__init__(message)
