"""
Configuration-related exceptions for the hierarchical configuration system.

This module defines custom exceptions for handling configuration errors,
validation failures, and migration issues.
"""

from typing import Any, Optional


class ConfigurationError(Exception):
    """Base exception for configuration-related errors.
    
    This is the base class for all configuration-related exceptions.
    It provides a common interface for error handling and logging.
    """
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """Initialize configuration error.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ValidationException(ConfigurationError):
    """Exception raised when configuration validation fails.
    
    This exception is raised when a configuration value fails validation
    against the defined schema or constraints.
    """
    
    def __init__(
        self, 
        field: str, 
        value: Any, 
        message: str,
        expected_type: Optional[type] = None,
        constraints: Optional[dict] = None
    ):
        """Initialize validation exception.
        
        Args:
            field: Name of the field that failed validation
            value: The value that failed validation
            message: Human-readable error message
            expected_type: Expected type for the field (if applicable)
            constraints: Validation constraints that were violated
        """
        self.field = field
        self.value = value
        self.expected_type = expected_type
        self.constraints = constraints or {}
        
        details = {
            "field": field,
            "value": value,
            "expected_type": expected_type.__name__ if expected_type else None,
            "constraints": constraints
        }
        
        super().__init__(
            f"Validation error for '{field}': {message}",
            details=details
        )


class ConfigurationMigrationError(ConfigurationError):
    """Exception raised when configuration migration fails.
    
    This exception is raised when there are issues during the migration
    of configuration from one version to another.
    """
    
    def __init__(
        self, 
        from_version: str, 
        to_version: str, 
        message: str,
        migration_step: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize configuration migration error.
        
        Args:
            from_version: Source version of the configuration
            to_version: Target version of the configuration
            message: Human-readable error message
            migration_step: Specific migration step that failed (if applicable)
            original_error: Original exception that caused the migration failure
        """
        self.from_version = from_version
        self.to_version = to_version
        self.migration_step = migration_step
        self.original_error = original_error
        
        details = {
            "from_version": from_version,
            "to_version": to_version,
            "migration_step": migration_step,
            "original_error": str(original_error) if original_error else None
        }
        
        super().__init__(
            f"Migration error from {from_version} to {to_version}: {message}",
            details=details
        )


class ConfigurationLoadError(ConfigurationError):
    """Exception raised when configuration loading fails.
    
    This exception is raised when there are issues loading configuration
    from files, environment variables, or other sources.
    """
    
    def __init__(
        self, 
        source: str, 
        message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize configuration load error.
        
        Args:
            source: Source of the configuration (file, env var, etc.)
            message: Human-readable error message
            file_path: Path to the configuration file (if applicable)
            line_number: Line number where the error occurred (if applicable)
            original_error: Original exception that caused the load failure
        """
        self.source = source
        self.file_path = file_path
        self.line_number = line_number
        self.original_error = original_error
        
        details = {
            "source": source,
            "file_path": file_path,
            "line_number": line_number,
            "original_error": str(original_error) if original_error else None
        }
        
        super().__init__(
            f"Configuration load error from {source}: {message}",
            details=details
        )


class ConfigurationSaveError(ConfigurationError):
    """Exception raised when configuration saving fails.
    
    This exception is raised when there are issues saving configuration
    to files or other destinations.
    """
    
    def __init__(
        self, 
        destination: str, 
        message: str,
        file_path: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize configuration save error.
        
        Args:
            destination: Destination of the configuration (file, etc.)
            message: Human-readable error message
            file_path: Path to the configuration file (if applicable)
            original_error: Original exception that caused the save failure
        """
        self.destination = destination
        self.file_path = file_path
        self.original_error = original_error
        
        details = {
            "destination": destination,
            "file_path": file_path,
            "original_error": str(original_error) if original_error else None
        }
        
        super().__init__(
            f"Configuration save error to {destination}: {message}",
            details=details
        )