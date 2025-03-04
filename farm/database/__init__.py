"""Database module for simulation state persistence and analysis.

This module provides database implementations for storing and analyzing simulation data,
with optimized SQLite pragma settings for different workloads.

Classes
-------
SimulationDatabase : Main database interface for disk-based storage
InMemorySimulationDatabase : High-performance in-memory database
ShardedSimulationDatabase : Database implementation for large-scale simulations

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

from .database import (
    SimulationDatabase,
    InMemorySimulationDatabase,
    ShardedSimulationDatabase,
    AsyncDataLogger,
)
from .pragma_docs import (
    get_pragma_profile,
    get_pragma_info,
    analyze_pragma_value,
    PRAGMA_PROFILES,
    PRAGMA_INFO,
)

__all__ = [
    "SimulationDatabase",
    "InMemorySimulationDatabase",
    "ShardedSimulationDatabase",
    "AsyncDataLogger",
    "get_pragma_profile",
    "get_pragma_info",
    "analyze_pragma_value",
    "PRAGMA_PROFILES",
    "PRAGMA_INFO",
]
