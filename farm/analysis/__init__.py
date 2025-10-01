"""
Modern Analysis Module System

A protocol-based, type-safe framework for creating and running analysis modules
on simulation data.

Quick Start:
    >>> from farm.analysis.service import AnalysisService, AnalysisRequest
    >>> from farm.core.services import EnvConfigService
    >>> from pathlib import Path
    >>> 
    >>> # Initialize service
    >>> config_service = EnvConfigService()
    >>> service = AnalysisService(config_service)
    >>> 
    >>> # Create and run analysis
    >>> request = AnalysisRequest(
    ...     module_name="dominance",
    ...     experiment_path=Path("data/experiment"),
    ...     output_path=Path("results")
    ... )
    >>> result = service.run(request)
    >>> 
    >>> if result.success:
    ...     print(f"Analysis complete in {result.execution_time:.2f}s")

Features:
    - Protocol-based architecture for type safety
    - Comprehensive data validation
    - Smart caching for repeated analyses
    - Progress tracking with callbacks
    - Batch processing support
    - Rich error messages
    - Complete test coverage

Documentation:
    - README.md - Complete user guide
    - ARCHITECTURE.md - System architecture
    - examples/analysis_example.py - Working examples
    - tests/analysis/ - Test suite

Core Components:
    - protocols: Type-safe protocol definitions
    - core: Base implementations
    - validation: Data validators
    - exceptions: Custom exception types
    - registry: Module discovery and management
    - service: High-level API
"""

from farm.analysis.protocols import (
    AnalysisModule,
    DataLoader,
    DataProcessor,
    DataValidator,
    AnalysisFunction,
    Analyzer,
    Visualizer,
)

from farm.analysis.core import (
    BaseAnalysisModule,
    SimpleDataProcessor,
    ChainedDataProcessor,
    make_analysis_function,
)

from farm.analysis.validation import (
    ColumnValidator,
    DataQualityValidator,
    CompositeValidator,
    validate_numeric_columns,
    validate_simulation_data,
)

from farm.analysis.exceptions import (
    AnalysisError,
    DataValidationError,
    ModuleNotFoundError,
    DataLoaderError,
    DataProcessingError,
    AnalysisFunctionError,
    ConfigurationError,
    InsufficientDataError,
    VisualizationError,
    DatabaseError,
)

from farm.analysis.registry import (
    ModuleRegistry,
    registry,
    register_modules,
    get_module,
    get_module_names,
    list_modules,
)

from farm.analysis.service import (
    AnalysisService,
    AnalysisRequest,
    AnalysisResult,
    AnalysisCache,
)

from farm.analysis.common.context import (
    AnalysisContext,
)

# Version info
__version__ = "2.0.0"
__author__ = "AgentFarm Team"
__all__ = [
    # Protocols
    "AnalysisModule",
    "DataLoader",
    "DataProcessor",
    "DataValidator",
    "AnalysisFunction",
    "Analyzer",
    "Visualizer",
    # Core
    "BaseAnalysisModule",
    "SimpleDataProcessor",
    "ChainedDataProcessor",
    "make_analysis_function",
    # Validation
    "ColumnValidator",
    "DataQualityValidator",
    "CompositeValidator",
    "validate_numeric_columns",
    "validate_simulation_data",
    # Exceptions
    "AnalysisError",
    "DataValidationError",
    "ModuleNotFoundError",
    "DataLoaderError",
    "DataProcessingError",
    "AnalysisFunctionError",
    "ConfigurationError",
    "InsufficientDataError",
    "VisualizationError",
    "DatabaseError",
    # Registry
    "ModuleRegistry",
    "registry",
    "register_modules",
    "get_module",
    "get_module_names",
    "list_modules",
    # Service
    "AnalysisService",
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisCache",
    # Context
    "AnalysisContext",
]


def get_info() -> dict:
    """Get information about the analysis system.
    
    Returns:
        Dictionary with system information
    """
    return {
        "version": __version__,
        "modules": get_module_names(),
        "features": [
            "Protocol-based architecture",
            "Comprehensive validation",
            "Smart caching",
            "Progress tracking",
            "Batch processing",
            "Type safety",
            "Rich error messages",
        ],
        "documentation": [
            "README.md",
            "ARCHITECTURE.md",
            "examples/analysis_example.py",
        ],
    }


def show_info():
    """Print information about the analysis system."""
    info = get_info()
    
    print("=" * 60)
    print(f"Analysis Module System v{info['version']}")
    print("=" * 60)
    
    print("\n📦 Registered Modules:")
    for module in info['modules']:
        print(f"  - {module}")
    
    print("\n✨ Features:")
    for feature in info['features']:
        print(f"  - {feature}")
    
    print("\n📚 Documentation:")
    for doc in info['documentation']:
        print(f"  - {doc}")
    
    print("\n" + "=" * 60)
