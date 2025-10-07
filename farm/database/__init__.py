"""Database module for simulation state persistence and analysis.

This module provides database implementations for storing and analyzing simulation data,
with optimized SQLite pragma settings for different workloads.

The module follows the Dependency Inversion Principle using protocols defined in
farm.core.interfaces to enable loose coupling and easier testing.

Classes
-------
SimulationDatabase : Main database interface for disk-based storage (implements DatabaseProtocol)
InMemorySimulationDatabase : High-performance in-memory database (implements DatabaseProtocol)
ShardedSimulationDatabase : Database implementation for large-scale simulations
ExperimentDatabase : Database for managing multiple simulations in a single file
SimulationContext : Context for a specific simulation within an experiment

Protocols
---------
DatabaseProtocol : Protocol for database operations (from farm.core.interfaces)
DataLoggerProtocol : Protocol for data logging operations (from farm.core.interfaces)
RepositoryProtocol : Protocol for repository operations (from farm.core.interfaces)

Functions
---------
get_pragma_profile : Get pragma settings for a specific profile
get_pragma_info : Get information about a specific pragma
analyze_pragma_value : Analyze the performance implications of a pragma value

Pragma Profiles
--------------
- performance: Maximum performance for write-heavy workloads
- safety: Maximum data safety, reduced performance
- balanced: Good balance of performance and data safety
- memory: Optimized for low memory usage
"""

from farm.core.interfaces import (
    DatabaseProtocol,
    DataLoggerProtocol,
    RepositoryProtocol,
)
from .database import (
    SimulationDatabase,
    InMemorySimulationDatabase,
    ShardedSimulationDatabase,
    AsyncDataLogger,
)
from .experiment_database import (
    ExperimentDatabase,
    SimulationContext,
    ExperimentDataLogger,
)
from .pragma_docs import (
    get_pragma_profile,
    get_pragma_info,
    analyze_pragma_value,
    PRAGMA_PROFILES,
    PRAGMA_INFO,
)

__all__ = [
    # Concrete implementations
    "SimulationDatabase",
    "InMemorySimulationDatabase",
    "ShardedSimulationDatabase",
    "AsyncDataLogger",
    "ExperimentDatabase",
    "SimulationContext",
    "ExperimentDataLogger",
    # Protocols for type hints and testing
    "DatabaseProtocol",
    "DataLoggerProtocol",
    "RepositoryProtocol",
    # Pragma utilities
    "get_pragma_profile",
    "get_pragma_info",
    "analyze_pragma_value",
    "PRAGMA_PROFILES",
    "PRAGMA_INFO",
]
