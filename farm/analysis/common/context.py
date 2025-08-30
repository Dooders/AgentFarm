"""
Shared analysis context passed to analysis functions.

Standardizing the context object allows analysis functions to access
common execution details without requiring ad-hoc parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AnalysisContext:
    """
    Runtime context for analysis functions.

    Attributes:
        output_path: Directory path where outputs (plots, files) should be written
        config: Optional configuration dictionary for analysis-time options
        services: Optional dependency map (e.g., writers, loggers) for advanced cases
    """

    output_path: str = ""
    config: Optional[Dict[str, Any]] = None
    services: Dict[str, Any] = field(default_factory=dict)

