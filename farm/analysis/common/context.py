"""
Shared analysis context passed to analysis functions.

Standardizing the context object allows analysis functions to access
common execution details without requiring ad-hoc parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
from pathlib import Path
from farm.utils.logging_config import get_logger


@dataclass
class AnalysisContext:
    """
    Runtime context for analysis functions.

    Attributes:
        output_path: Directory path where outputs (plots, files) should be written
        config: Optional configuration dictionary for analysis-time options
        services: Optional dependency map (e.g., writers, loggers) for advanced cases
        logger: Logger instance for this analysis session
        progress_callback: Optional callback for progress updates
        metadata: Additional metadata about the analysis run
    """

    output_path: Path
    config: Dict[str, Any] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    progress_callback: Optional[Callable[[str, float], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure output_path is a Path object."""
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def get_output_file(self, filename: str, subdir: Optional[str] = None) -> Path:
        """Get full path for an output file.
        
        Args:
            filename: Name of the output file
            subdir: Optional subdirectory within output_path
            
        Returns:
            Full path to the output file
        """
        if subdir:
            path = self.output_path / subdir
            path.mkdir(parents=True, exist_ok=True)
            return path / filename
        return self.output_path / filename
    
    def report_progress(self, message: str, progress: float = 0.0) -> None:
        """Report progress if callback is set.
        
        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
        """
        if self.progress_callback:
            self.progress_callback(message, progress)
        self.logger.info(f"Progress ({progress:.1%}): {message}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

