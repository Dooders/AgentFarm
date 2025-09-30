"""Database module for simulation state persistence and analysis.

This module provides a SQLAlchemy-based database implementation for storing and
analyzing simulation data. It handles all database operations including state
logging, configuration management, and data analysis.

Key Components
-------------
- SimulationDatabase : Main database interface class
- DataLogger : Handles buffered data logging operations
- DataRetriever : Handles data querying operations
- SQLAlchemy Models : ORM models for different data types
    - Agent : Agent entity data
    - AgentState : Time-series agent state data
    - ResourceState : Resource position and amount tracking
    - SimulationStep : Per-step simulation metrics
    - AgentAction : Agent behavior logging
    - HealthIncident : Health-related events
    - SimulationConfig : Configuration management

Features
--------
- Efficient batch operations through DataLogger
- Transaction safety with rollback support
- Comprehensive error handling
- Multi-format data export (CSV, Excel, JSON, Parquet)
- Configuration management
- Performance optimized queries through DataRetriever

Usage Example
------------
>>> db = SimulationDatabase("simulation.db")
>>> db.save_configuration(config_dict)
>>> db.logger.log_step(step_number, agent_states, resource_states, metrics)
>>> db.export_data("results.csv")
>>> db.close()

Notes
-----
- Uses SQLite as the backend database
- Implements foreign key constraints
- Includes indexes for performance
- Supports concurrent access through session management
"""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

from farm.database.data_retrieval import DataRetriever
from farm.database.session_manager import SessionManager

from .data_logging import DataLogger, DataLoggingConfig, ShardedDataLogger
from .models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    Base,
    HealthIncident,
    InteractionModel,
    ReproductionEventModel,
    ResourceModel,
    Simulation,
    SimulationConfig,
    SimulationStepModel,
)
from .utilities import (
    create_database_schema,
    execute_with_retry,
    format_agent_state,
    format_position,
    safe_json_loads,
    validate_export_format,
)

logger = logging.getLogger(__name__)


class SimulationDatabase:
    """Database interface for simulation state persistence and analysis.

    This class provides a high-level interface for storing and retrieving simulation
    data using SQLAlchemy ORM. It handles all database operations including state
    logging, configuration management, and data analysis with transaction safety
    and efficient batch operations.

    Features
    --------
    - Buffered batch operations via DataLogger
    - Data querying via DataRetriever
    - Transaction safety with automatic rollback
    - Comprehensive error handling
    - Multi-format data export
    - Thread-safe session management

    Attributes
    ----------
    db_path : str
        Path to the SQLite database file
    engine : sqlalchemy.engine.Engine
        SQLAlchemy database engine instance
    Session : sqlalchemy.orm.scoped_session
        Thread-local session factory
    logger : DataLogger
        Handles buffered data logging operations
    query : DataRetriever
        Handles data querying operations

    Methods
    -------
    update_agent_death(agent_id, death_time, cause)
        Update agent record with death information
    update_agent_state(agent_id, step_number, state_data)
        Update agent state in the database
    export_data(filepath, format, data_types, ...)
        Export simulation data to various file formats
    save_configuration(config)
        Save simulation configuration to database
    get_configuration()
        Retrieve current simulation configuration
    cleanup()
        Clean up database resources
    close()
        Close database connections

    Example
    -------
    >>> db = SimulationDatabase("simulation.db")
    >>> db.save_configuration({"agents": 100, "resources": 1000})
    >>> db.logger.log_step(1, agent_states, resource_states, metrics)
    >>> db.export_data("results.csv")
    >>> db.close()

    Notes
    -----
    - Uses SQLite as the backend database
    - Implements foreign key constraints
    - Creates required tables automatically
    - Handles concurrent access through thread-local sessions
    - Buffers operations through DataLogger for better performance
    """

    def _get_database_url(self, db_path: str) -> str:
        """Get the database URL for SQLAlchemy engine.

        This method can be overridden by subclasses to customize the database URL.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file

        Returns
        -------
        str
            Database URL for SQLAlchemy
        """
        return f"sqlite:///{db_path}"

    def __init__(self, db_path: str, config=None, simulation_id=None) -> None:
        """Initialize a new SimulationDatabase instance with SQLAlchemy.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file
        config : SimulationConfig, optional
            Configuration object with database settings
        simulation_id : str, optional
            Unique identifier for this simulation

        Notes
        -----
        - Enables foreign key support for SQLite
        - Creates session factory with thread-local scope
        - Initializes tables and indexes
        - Sets up batch operation buffers
        """
        self.db_path = db_path
        self.config = config
        self.simulation_id = simulation_id

        # Configure pragma settings from config
        pragma_profile = "balanced"
        cache_size_mb = 200
        synchronous_mode = "NORMAL"
        journal_mode = "WAL"
        custom_pragmas = {}

        if config:
            pragma_profile = getattr(config, "db_pragma_profile", pragma_profile)
            cache_size_mb = getattr(config, "db_cache_size_mb", cache_size_mb)
            synchronous_mode = getattr(config, "db_synchronous_mode", synchronous_mode)
            journal_mode = getattr(config, "db_journal_mode", journal_mode)
            custom_pragmas = getattr(config, "db_custom_pragmas", {})

        # Store pragma settings for reference
        self.pragma_profile = pragma_profile
        self.cache_size_mb = cache_size_mb
        self.synchronous_mode = synchronous_mode
        self.journal_mode = journal_mode
        self.custom_pragmas = custom_pragmas

        # Store config values for reference (prefer nested config.database)
        db_cfg = getattr(config, "database", None) if config else None

        def _get_db_setting(name, default):
            if db_cfg is None:
                # Backward-compatible fallback to top-level config
                return getattr(config, name, default) if config else default
            if isinstance(db_cfg, dict):
                return db_cfg.get(name, getattr(config, name, default) if config else default)
            return getattr(db_cfg, name, getattr(config, name, default) if config else default)

        self.pool_size = _get_db_setting("connection_pool_size", 10)
        self.pool_recycle = _get_db_setting("connection_pool_recycle", 3600)
        self.connection_timeout = _get_db_setting("connection_timeout", 30)
        
        # Create engine with connect_args using config values
        self.engine = create_engine(
            self._get_database_url(db_path),
            # Larger pool size for concurrent operations
            pool_size=self.pool_size,
            # Longer timeout before connections are recycled
            pool_recycle=self.pool_recycle,
            # Enable connection pooling
            poolclass=QueuePool,
            # Optimize for write-heavy workloads
            connect_args={
                "timeout": self.connection_timeout,  # Longer timeout for busy database
                "check_same_thread": False,  # Allow cross-thread usage
            },
        )

        # Apply pragma settings directly for the initial connection
        conn = self.engine.raw_connection()
        cursor = conn.cursor()

        # Apply profile-specific settings
        self._direct_apply_pragma_profile(cursor, pragma_profile)

        # Apply any specific overrides
        if synchronous_mode not in ["NORMAL", "OFF", "FULL"]:
            cursor.execute(f"PRAGMA synchronous={synchronous_mode}")

        if journal_mode not in [
            "WAL",
            "MEMORY",
            "DELETE",
            "TRUNCATE",
            "PERSIST",
            "OFF",
        ]:
            cursor.execute(f"PRAGMA journal_mode={journal_mode}")

        # Apply custom pragmas
        for pragma, value in custom_pragmas.items():
            cursor.execute(f"PRAGMA {pragma}={value}")

        cursor.close()
        conn.close()

        # Also set up the event listener for future connections
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()

            # Apply the selected pragma profile
            self._direct_apply_pragma_profile(cursor, pragma_profile)

            # Apply any specific overrides
            if synchronous_mode not in ["NORMAL", "OFF", "FULL"]:
                cursor.execute(f"PRAGMA synchronous={synchronous_mode}")

            if journal_mode not in [
                "WAL",
                "MEMORY",
                "DELETE",
                "TRUNCATE",
                "PERSIST",
                "OFF",
            ]:
                cursor.execute(f"PRAGMA journal_mode={journal_mode}")

            # Apply custom pragmas
            for pragma, value in custom_pragmas.items():
                cursor.execute(f"PRAGMA {pragma}={value}")

            cursor.close()

        # Create session factory
        session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.Session = scoped_session(session_factory)

        # Create tables and indexes
        self._create_tables()

        # Create SessionManager for DataRetriever
        self.session_manager = SessionManager()
        self.session_manager.engine = self.engine
        self.session_manager.Session = self.Session

        # Initialize data logger with simulation_id and config values (prefer nested config.database)
        db_cfg_for_logger = getattr(config, 'database', None) if config else None
        if isinstance(db_cfg_for_logger, dict):
            log_buffer_size = db_cfg_for_logger.get('log_buffer_size', getattr(config, 'log_buffer_size', 1000) if config else 1000)
            commit_interval = db_cfg_for_logger.get('commit_interval_seconds', getattr(config, 'commit_interval_seconds', 30) if config else 30)
        elif db_cfg_for_logger is not None:
            log_buffer_size = getattr(db_cfg_for_logger, 'log_buffer_size', getattr(config, 'log_buffer_size', 1000) if config else 1000)
            commit_interval = getattr(db_cfg_for_logger, 'commit_interval_seconds', getattr(config, 'commit_interval_seconds', 30) if config else 30)
        else:
            log_buffer_size = getattr(config, 'log_buffer_size', 1000) if config else 1000
            commit_interval = getattr(config, 'commit_interval_seconds', 30) if config else 30
        self.logger = DataLogger(
            self,
            simulation_id=self.simulation_id,
            config=DataLoggingConfig(buffer_size=log_buffer_size, commit_interval=commit_interval),
        )
        self.query = DataRetriever(self.session_manager)

    def _direct_apply_pragma_profile(self, cursor, profile):
        """Apply a specific pragma profile directly to a cursor.

        Parameters
        ----------
        cursor : Cursor
            SQLite cursor
        profile : str
            Profile name: "balanced", "performance", "safety", or "memory"
        """
        # Convert cache size from MB to KB (negative value for KB)
        cache_size_kb = -1 * self.cache_size_mb * 1024

        # Common settings
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory

        # Apply profile-specific settings
        if profile == "performance":
            # Maximum performance settings
            cursor.execute("PRAGMA synchronous=OFF")
            cursor.execute("PRAGMA journal_mode=MEMORY")
            cursor.execute(f"PRAGMA cache_size={cache_size_kb}")
            cursor.execute("PRAGMA page_size=8192")  # Larger pages for fewer I/O ops
            cursor.execute("PRAGMA mmap_size=1073741824")  # 1GB memory-mapped I/O
            cursor.execute("PRAGMA automatic_index=OFF")  # Disable automatic indexing
            cursor.execute("PRAGMA busy_timeout=60000")  # 60-second timeout
        elif profile == "safety":
            # Data safety settings
            cursor.execute("PRAGMA synchronous=FULL")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(
                f"PRAGMA cache_size={min(cache_size_kb, -102400)}"
            )  # Max 100MB for safety
            cursor.execute("PRAGMA page_size=4096")  # Default page size
            cursor.execute("PRAGMA busy_timeout=30000")  # 30-second timeout
        elif profile == "memory":
            # Memory-optimized settings
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute(
                "PRAGMA journal_mode=MEMORY"
            )  # Explicitly set to MEMORY for memory profile

            # Always limit to 50MB max for memory profile, regardless of the configured size
            cursor.execute(
                "PRAGMA cache_size=-51200"
            )  # Fixed at 50MB max for memory profile

            cursor.execute("PRAGMA page_size=4096")  # Default page size
            cursor.execute("PRAGMA busy_timeout=15000")  # 15-second timeout
        else:  # balanced (default)
            # Balanced settings
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA cache_size={cache_size_kb}")
            cursor.execute("PRAGMA page_size=4096")  # Default page size
            cursor.execute("PRAGMA busy_timeout=30000")  # 30-second timeout

        # Run optimizer
        cursor.execute("PRAGMA optimize")

    def _apply_pragma_profile(
        self,
        cursor,
        profile="balanced",
        cache_size_mb=200,
        synchronous_mode="NORMAL",
        journal_mode="WAL",
    ):
        """Apply a specific pragma profile to a database connection.

        Parameters
        ----------
        cursor : Cursor
            SQLite cursor
        profile : str
            Profile name: "balanced", "performance", "safety", or "memory"
        cache_size_mb : int
            Cache size in MB
        synchronous_mode : str
            Override for synchronous mode
        journal_mode : str
            Override for journal mode

        Notes
        -----
        SQLite in-memory databases (:memory:) always use journal_mode=MEMORY
        regardless of the setting. This is a limitation of SQLite itself.

        When using the "performance" profile with in-memory databases, the synchronous
        mode may not be set to OFF as expected. To ensure synchronous=OFF, explicitly
        set synchronous_mode="OFF" in the configuration.
        """
        # Just delegate to the direct method
        self._direct_apply_pragma_profile(cursor, profile)

        # Apply overrides - this section is now simplified
        if synchronous_mode in ["OFF", "NORMAL", "FULL"]:
            cursor.execute(f"PRAGMA synchronous={synchronous_mode}")

        # For journal_mode, apply the override unless it's for an in-memory database
        is_memory_db = self.db_path == ":memory:"
        if not is_memory_db and journal_mode in [
            "WAL",
            "MEMORY",
            "DELETE",
            "TRUNCATE",
            "PERSIST",
            "OFF",
        ]:
            cursor.execute(f"PRAGMA journal_mode={journal_mode}")

    def get_current_pragmas(self):
        """Get current pragma settings for the database.

        Returns
        -------
        Dict[str, Any]
            Dictionary of current pragma settings
        """
        conn = self.engine.raw_connection()
        cursor = conn.cursor()

        pragmas = {}
        for pragma in [
            "synchronous",
            "journal_mode",
            "cache_size",
            "temp_store",
            "mmap_size",
            "page_size",
            "busy_timeout",
            "foreign_keys",
            "automatic_index",
        ]:
            try:
                cursor.execute(f"PRAGMA {pragma}")
                result = cursor.fetchone()
                if result is not None:
                    pragmas[pragma] = result[0]
            except Exception as e:
                logger.warning(f"Error getting pragma {pragma}: {e}")

        cursor.close()
        conn.close()

        return pragmas

    def analyze_pragma_performance(self):
        """Analyze current pragma settings for performance implications.

        Returns
        -------
        Dict[str, Dict]
            Performance analysis for each pragma
        """
        pragmas = self.get_current_pragmas()
        analysis = {}

        # Analyze synchronous mode
        sync_mode = pragmas.get("synchronous")
        if sync_mode == 0:  # OFF
            analysis["synchronous"] = {
                "performance": "Excellent",
                "safety": "Poor",
                "recommendation": "Only use for non-critical data or testing",
            }
        elif sync_mode == 1:  # NORMAL
            analysis["synchronous"] = {
                "performance": "Good",
                "safety": "Moderate",
                "recommendation": "Good balance for most workloads",
            }
        elif sync_mode == 2:  # FULL
            analysis["synchronous"] = {
                "performance": "Poor",
                "safety": "Excellent",
                "recommendation": "Consider NORMAL for better performance",
            }

        # Analyze journal mode
        journal_mode = pragmas.get("journal_mode", "").upper()
        if journal_mode == "MEMORY":
            analysis["journal_mode"] = {
                "performance": "Excellent",
                "safety": "Poor",
                "recommendation": "Only use for non-critical data or testing",
            }
        elif journal_mode == "WAL":
            analysis["journal_mode"] = {
                "performance": "Good",
                "safety": "Good",
                "recommendation": "Good balance for most workloads",
            }
        elif journal_mode == "DELETE":
            analysis["journal_mode"] = {
                "performance": "Poor",
                "safety": "Moderate",
                "recommendation": "Consider WAL for better performance",
            }

        # Analyze cache size
        cache_size = pragmas.get("cache_size", 0)
        cache_size_mb = (
            abs(cache_size) / 1024 if cache_size < 0 else cache_size / 1024 / 1024
        )
        if cache_size_mb < 50:
            analysis["cache_size"] = {
                "performance": "Poor",
                "memory_usage": "Excellent",
                "recommendation": "Consider increasing for better performance",
            }
        elif cache_size_mb < 200:
            analysis["cache_size"] = {
                "performance": "Good",
                "memory_usage": "Good",
                "recommendation": "Good balance for most workloads",
            }
        else:
            analysis["cache_size"] = {
                "performance": "Excellent",
                "memory_usage": "Poor",
                "recommendation": "Monitor memory usage with large cache",
            }

        return analysis

    def adjust_pragmas_for_workload(self, workload_type):
        """Adjust pragma settings based on current workload.

        Parameters
        ----------
        workload_type : str
            Type of workload: "read_heavy", "write_heavy", "balanced"
        """
        conn = self.engine.raw_connection()
        cursor = conn.cursor()

        if workload_type == "write_heavy":
            cursor.execute("PRAGMA synchronous=OFF")
            cursor.execute("PRAGMA journal_mode=MEMORY")
        elif workload_type == "read_heavy":
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA mmap_size=1073741824")  # 1GB memory-mapped I/O
        else:  # balanced
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA journal_mode=WAL")

        cursor.close()
        conn.close()

    def _execute_in_transaction(self, func: Callable) -> Any:
        """Execute database operations within a transaction with error handling.

        Parameters
        ----------
        func : callable
            Function that takes a session argument and performs database operations

        Returns
        -------
        Any
            Result of the executed function

        Raises
        ------
        IntegrityError
            If there's a database constraint violation
        OperationalError
            If there's a database connection issue
        ProgrammingError
            If there's a SQL syntax error
        SQLAlchemyError
            For other database-related errors
        """
        session = self.Session()
        try:
            # Disable autoflush temporarily for bulk operations
            session.autoflush = False
            result = execute_with_retry(session, lambda: func(session))
            session.autoflush = True
            return result
        finally:
            self.Session.remove()

    def close(self) -> None:
        """Close the database connection and clean up resources."""
        try:
            if hasattr(self, "engine") and self.engine:
                self.engine.dispose()
                logger.debug("Database engine disposed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
        finally:
            try:
                # Additional cleanup if needed
                pass
            except Exception as e:
                logger.error(f"Final cleanup error: {e}")

    def get_table_row_count(self, table_name: str) -> int:
        """Get the number of rows in a specified table.

        Parameters
        ----------
        table_name : str
            Name of the table to count rows from

        Returns
        -------
        int
            Number of rows in the table
        """

        def _count(session):
            try:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar()
            except Exception as e:
                logger.error(f"Error counting rows in table {table_name}: {e}")
                return 0

        return self._execute_in_transaction(_count)

    def export_data(
        self,
        filepath: str,
        format: str = "csv",
        data_types: Optional[List[str]] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        include_metadata: bool = True,
    ) -> None:
        """Export simulation data to various file formats with filtering options.

        Exports selected simulation data to a file in the specified format. Supports
        filtering by data type and time range, with optional metadata inclusion.

        Parameters
        ----------
        filepath : str
            Path where the export file will be saved
        format : str, optional
            Export format, one of: 'csv', 'excel', 'json', 'parquet'
            Default is 'csv'
        data_types : List[str], optional
            List of data types to export, can include:
            'metrics', 'agents', 'resources', 'actions'
            If None, exports all types
        start_step : int, optional
            Starting step number for data range
        end_step : int, optional
            Ending step number for data range
        include_metadata : bool, optional
            Whether to include simulation metadata, by default True

        Raises
        ------
        ValueError
            If format is unsupported or filepath is invalid
        SQLAlchemyError
            If there's an error retrieving data
        IOError
            If there's an error writing to the file
        """
        if not validate_export_format(format):
            raise ValueError(f"Unsupported export format: {format}")

        def _query(session):
            data = {}

            # Build step number filter
            step_filter = []
            if start_step is not None:
                step_filter.append(SimulationStepModel.step_number >= start_step)
            if end_step is not None:
                step_filter.append(SimulationStepModel.step_number <= end_step)

            # Default to all data types if none specified
            export_types = data_types or ["metrics", "agents", "resources", "actions"]

            # Collect requested data
            if "metrics" in export_types:
                metrics_query = session.query(SimulationStepModel)
                if step_filter:
                    metrics_query = metrics_query.filter(*step_filter)
                data["metrics"] = pd.read_sql(
                    metrics_query.statement, session.bind, index_col="step_number"
                )

            if "agents" in export_types:
                agents_query = (
                    session.query(AgentStateModel)
                    .join(AgentModel)
                    .options(joinedload(AgentStateModel.agent))
                )
                if step_filter:
                    agents_query = agents_query.filter(*step_filter)
                data["agents"] = pd.read_sql(agents_query.statement, session.bind)

            if "resources" in export_types:
                resources_query = session.query(ResourceModel)
                if step_filter:
                    resources_query = resources_query.filter(*step_filter)
                data["resources"] = pd.read_sql(resources_query.statement, session.bind)

            if "actions" in export_types:
                actions_query = session.query(ActionModel)
                if step_filter:
                    actions_query = actions_query.filter(*step_filter)
                data["actions"] = pd.read_sql(actions_query.statement, session.bind)

            # Add metadata if requested
            if include_metadata:
                config = self.get_configuration()
                data["metadata"] = {
                    "config": config,
                    "export_timestamp": datetime.now().isoformat(),
                    "data_range": {"start_step": start_step, "end_step": end_step},
                    "exported_types": export_types,
                }

            # Export based on format
            if format == "csv":
                # Create separate CSV files for each data type
                base_path = filepath.rsplit(".", 1)[0]
                for data_type, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(f"{base_path}_{data_type}.csv", index=False, encoding="utf-8")
                    elif data_type == "metadata":
                        with open(f"{base_path}_metadata.json", "w", encoding="utf-8") as f:
                            json.dump(df, f, indent=2)

            elif format == "excel":
                # Save all data types as separate sheets in one Excel file
                with pd.ExcelWriter(filepath) as writer:
                    for data_type, df in data.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_excel(writer, sheet_name=data_type, index=False)
                        elif data_type == "metadata":
                            pd.DataFrame([df]).to_excel(
                                writer, sheet_name="metadata", index=False
                            )

            elif format == "json":
                # Convert all data to JSON format
                json_data = {
                    k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                    for k, v in data.items()
                }
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)

            elif format == "parquet":
                # Save as parquet format (good for large datasets)
                if len(data) == 1:
                    # Single dataframe case
                    next(iter(data.values())).to_parquet(filepath)
                else:
                    # Multiple dataframes case
                    base_path = filepath.rsplit(".", 1)[0]
                    for data_type, df in data.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_parquet(f"{base_path}_{data_type}.parquet")

            else:
                raise ValueError(f"Unsupported export format: {format}")

        self._execute_in_transaction(_query)

    def update_agent_death(
        self, agent_id: str, death_time: int, cause: str = "starvation"
    ):
        """Update agent record with death information.

        Parameters
        ----------
        agent_id : str
            ID of the agent that died
        death_time : int
            Time step when death occurred
        cause : str, optional
            Cause of death, defaults to "starvation"
        """

        def _update(session):
            # Update agent record with death time
            agent = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == agent_id)
                .first()
            )

            if agent:
                agent.death_time = death_time

                # Log health incident for death
                health_incident = HealthIncident(
                    step_number=death_time,
                    agent_id=agent_id,
                    health_before=0,  # Death implies health reached 0
                    health_after=0,
                    cause=cause,
                    details=json.dumps({"final_state": "dead"}),
                )
                session.add(health_incident)

        self._execute_in_transaction(_update)

    def _load_agent_stats(self, conn, agent_id) -> Dict:
        """Load current agent statistics from database."""

        def _query(session):
            # Get latest state for the agent using lowercase table names
            latest_state = (
                session.query(
                    AgentStateModel.current_health,
                    AgentStateModel.resource_level,
                    AgentStateModel.total_reward,
                    AgentStateModel.age,
                    AgentStateModel.is_defending,
                    AgentStateModel.position_x,
                    AgentStateModel.position_y,
                    AgentStateModel.step_number,
                )
                .filter(AgentStateModel.agent_id == agent_id)
                .order_by(AgentStateModel.step_number.desc())
                .first()
            )

            if latest_state:
                position = (latest_state[5], latest_state[6])
                return {
                    "current_health": latest_state[0],
                    "resource_level": latest_state[1],
                    "total_reward": latest_state[2],
                    "age": latest_state[3],
                    "is_defending": latest_state[4],
                    "current_position": format_position(position),
                }

            return {
                "current_health": 0,
                "resource_level": 0,
                "total_reward": 0,
                "age": 0,
                "is_defending": False,
                "current_position": format_position((0.0, 0.0)),
            }

        return self._execute_in_transaction(_query)

    def update_agent_state(self, agent_id: str, step_number: int, state_data: Dict):
        """Update agent state in the database.

        Parameters
        ----------
        agent_id : str
            ID of the agent to update
        step_number : int
            Current simulation step
        state_data : Dict
            Dictionary containing state data:
            - current_health: float
            - starting_health: float
            - resource_level: float
            - position: Tuple[float, float]
            - is_defending: bool
            - total_reward: float
            - starvation_counter: int
        """

        def _update(session):
            # Get the agent to access its properties
            agent = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == agent_id)
                .first()
            )
            if not agent:
                logger.error(f"Agent {agent_id} not found")
                return

            formatted_state = format_agent_state(
                agent_id, step_number, state_data, simulation_id=self.simulation_id
            )
            agent_state = AgentStateModel(**formatted_state)
            session.add(agent_state)

        self._execute_in_transaction(_update)

    def _create_tables(self):
        """Create the required database schema.

        Creates all tables defined in the SQLAlchemy models if they don't exist.
        Also creates necessary indexes for performance optimization.

        Raises
        ------
        SQLAlchemyError
            If there's an error creating the tables
        OperationalError
            If there's a database connection issue
        """

        def _create(session):
            create_database_schema(self.engine, Base)

        self._execute_in_transaction(_create)

    def update_notes(self, notes_data: Dict):
        """Update simulation notes in the database.

        Parameters
        ----------
        notes_data : Dict
            Dictionary of notes to update, where:
            - keys are note identifiers
            - values are note content objects

        Raises
        ------
        SQLAlchemyError
            If there's an error updating the notes
        """

        def _update(session):
            # Use merge instead of update to handle both inserts and updates
            for key, value in notes_data.items():
                session.merge(value)

        self._execute_in_transaction(_update)

    def cleanup(self):
        """Clean up database and GUI resources.

        Performs cleanup operations including:
        - Flushing all data buffers
        - Closing database connections
        - Disposing of the engine
        - Removing session

        This method should be called before application shutdown.

        Raises
        ------
        Exception
            If cleanup operations fail
        """
        try:
            # Flush any pending changes
            self.logger.flush_all_buffers()

            # Close database connections
            self.Session.remove()
            self.engine.dispose()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
        finally:
            # Ensure critical resources are released
            if hasattr(self, "Session"):
                self.Session.remove()

    def get_configuration(self) -> Dict:
        """Retrieve the simulation configuration from the database."""

        def _query(session):
            config = (
                session.query(SimulationConfig)
                .order_by(SimulationConfig.timestamp.desc())
                .first()
            )

            if config and config.config_data:
                return safe_json_loads(config.config_data) or {}
            return {}

        return self._execute_in_transaction(_query)

    def save_configuration(self, config: Dict) -> None:
        """Save simulation configuration to the database."""

        def _insert(session):
            config_obj = SimulationConfig(
                simulation_id=self.simulation_id,
                timestamp=int(time.time()),
                config_data=json.dumps(config),
            )
            session.add(config_obj)

        self._execute_in_transaction(_insert)

    def add_simulation_record(
        self, simulation_id: str, start_time: datetime, status: str, parameters: Dict
    ) -> None:
        """Add a simulation record to the database.

        Parameters
        ----------
        simulation_id : str
            Unique identifier for the simulation
        start_time : datetime
            Start time of the simulation
        status : str
            Current status of the simulation (e.g., "running", "completed")
        parameters : Dict
            Configuration parameters for the simulation
        """

        def _insert(session):
            sim_record = Simulation(
                simulation_id=simulation_id,
                start_time=start_time,
                status=status,
                parameters=parameters,
                simulation_db_path=(
                    self.db_path if hasattr(self, "db_path") else ":memory:"
                ),
            )
            session.add(sim_record)

        self._execute_in_transaction(_insert)

    def log_reproduction_event(
        self,
        step_number: int,
        parent_id: str,
        success: bool,
        parent_resources_before: float,
        parent_resources_after: float,
        offspring_id: Optional[str] = None,
        offspring_initial_resources: Optional[float] = None,
        failure_reason: Optional[str] = None,
        parent_position: Optional[tuple[float, float]] = None,
        parent_generation: Optional[int] = None,
        offspring_generation: Optional[int] = None,
    ) -> None:
        """Log a reproduction event to the database.

        Parameters
        ----------
        step_number : int
            Current simulation step
        parent_id : str
            ID of parent agent attempting reproduction
        success : bool
            Whether reproduction was successful
        parent_resources_before : float
            Parent's resources before reproduction
        parent_resources_after : float
            Parent's resources after reproduction
        offspring_id : str, optional
            ID of created offspring (if successful)
        offspring_initial_resources : float, optional
            Resources given to offspring (if successful)
        failure_reason : str, optional
            Reason for failure (if unsuccessful)
        parent_position : tuple[float, float], optional
            Position where reproduction occurred
        parent_generation : int, optional
            Parent's generation number
        offspring_generation : int, optional
            Offspring's generation number (if successful)
        """

        def _log(session):
            # Log reproduction event to ReproductionEventModel
            event = ReproductionEventModel(
                step_number=step_number,
                parent_id=parent_id,
                offspring_id=offspring_id,
                success=success,
                parent_resources_before=parent_resources_before,
                parent_resources_after=parent_resources_after,
                offspring_initial_resources=offspring_initial_resources,
                failure_reason=failure_reason,
                parent_generation=parent_generation,
                offspring_generation=offspring_generation,
                parent_position_x=parent_position[0] if parent_position else None,
                parent_position_y=parent_position[1] if parent_position else None,
                timestamp=datetime.now(),
            )
            session.add(event)

            # Also log as interaction for comprehensive tracking (minimal data to avoid duplication)
            interaction_type = "reproduce" if success else "reproduce_failed"
            target_type = "agent" if offspring_id and success else "position"
            target_id = offspring_id if offspring_id and success else (
                f"{parent_position[0]},{parent_position[1]}" if parent_position else "unknown"
            )

            # Only store essential relationship data - full details already in ReproductionEventModel
            interaction_details = {}
            if not success and failure_reason:
                interaction_details["failure_reason"] = failure_reason

            interaction = InteractionModel(
                step_number=step_number,
                source_type="agent",
                source_id=parent_id,
                target_type=target_type,
                target_id=target_id,
                interaction_type=interaction_type,
                action_type="reproduce",
                details=interaction_details,  # Minimal details to avoid duplication
            )
            session.add(interaction)

        self._execute_in_transaction(_log)


class AsyncDataLogger:
    """Asynchronous data logger for non-critical simulation data.

    This class provides a non-blocking interface for logging data that
    doesn't need to be immediately persisted, improving simulation performance.
    """

    def __init__(self, database, flush_interval=5.0):
        """Initialize the asynchronous logger.

        Parameters
        ----------
        database : SimulationDatabase
            Database instance to use for persistence
        flush_interval : float
            How often to flush data to database (seconds)
        """
        self.database = database
        self.flush_interval = flush_interval
        self.queue = queue.Queue()
        self.running = False
        self.worker_thread = None

    def start(self):
        """Start the background worker thread."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the background worker and flush remaining data."""
        if not self.running:
            return

        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)
            self._flush_queue()

    def log(self, log_type, data):
        """Queue data for asynchronous logging.

        Parameters
        ----------
        log_type : str
            Type of data being logged (e.g., 'action', 'health')
        data : dict
            Data to be logged
        """
        self.queue.put((log_type, data))

    def _worker_loop(self):
        """Background worker that periodically flushes data."""
        while self.running:
            time.sleep(self.flush_interval)
            self._flush_queue()

    def _flush_queue(self):
        """Process all queued log entries."""
        if self.queue.empty():
            return

        # Group log entries by type for batch processing
        batched_data = {
            "actions": [],
            "health": [],
            "reproduction": [],
            # Add other types as needed
        }

        # Collect all available items without blocking
        try:
            while True:
                log_type, data = self.queue.get_nowait()
                if log_type in batched_data:
                    batched_data[log_type].append(data)
                self.queue.task_done()
        except queue.Empty:
            pass

        # Process batched data
        session = self.database.Session()
        try:
            # Process actions
            if batched_data["actions"]:
                actions = [ActionModel(**data) for data in batched_data["actions"]]
                session.bulk_save_objects(actions)

            # Process health incidents
            if batched_data["health"]:
                incidents = [HealthIncident(**data) for data in batched_data["health"]]
                session.bulk_save_objects(incidents)

            # Process reproduction events
            if batched_data["reproduction"]:
                events = [
                    ReproductionEventModel(**data)
                    for data in batched_data["reproduction"]
                ]
                session.bulk_save_objects(events)

            # Commit all changes at once
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error in async data logger: {e}")
        finally:
            session.close()


class InMemorySimulationDatabase(SimulationDatabase):
    """In-memory SQLite database implementation for performance-critical simulations.

    This class provides a high-performance database implementation using SQLite's
    in-memory mode. It is designed for simulations that prioritize execution speed
    over persistence, with optional periodic snapshots to disk.

    Features
    --------
    - Ultra-fast in-memory storage
    - No disk I/O overhead during simulation
    - Optional memory usage monitoring
    - Ability to persist final state to disk
    - Compatible with all SimulationDatabase methods

    Notes
    -----
    Memory usage should be carefully monitored, as the database can grow quickly
    with large or long-running simulations. Use the memory_limit_mb parameter
    to enable automatic monitoring and warnings.

    For best performance with in-memory databases:
    - Set pragma_profile="memory"
    - Set journal_mode="MEMORY"
    - Set synchronous_mode="OFF"

    To explicitly set synchronous=OFF, use the db_synchronous_mode="OFF" configuration.
    """

    def _get_database_url(self, db_path: str) -> str:
        """Return the in-memory database URL."""
        return "sqlite:///:memory:"

    def __init__(self, memory_limit_mb=None, config=None, simulation_id=None):
        """Initialize an in-memory database.

        Parameters
        ----------
        memory_limit_mb : int, optional
            Memory usage limit in MB for warnings, by default None
        config : SimulationConfig, optional
            Configuration object with database settings
        simulation_id : str, optional
            Unique identifier for this simulation
        """
        # Store memory monitoring parameters
        self.memory_limit_mb = memory_limit_mb
        self.memory_warning_threshold = 0.7  # 70% of limit
        self.memory_critical_threshold = 0.9  # 90% of limit
        self.memory_usage_samples = []

        # Override memory limit if specified in config
        if memory_limit_mb is None and config and hasattr(
            config, "in_memory_db_memory_limit_mb"
        ):
            self.memory_limit_mb = config.in_memory_db_memory_limit_mb

        # Initialize parent class - it will use our overridden _get_database_url method
        super().__init__(db_path=":memory:", config=config, simulation_id=simulation_id)

        # Start memory monitoring if a limit is set
        if self.memory_limit_mb:
            self._start_memory_monitoring()

    def _start_memory_monitoring(self):
        """Start background thread for memory usage monitoring."""
        import os
        import threading
        import time

        import psutil

        def monitor_memory():
            process = psutil.Process(os.getpid())
            while True:
                try:
                    # Get memory usage in MB
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)

                    # Store in rolling window (last 5 samples)
                    self.memory_usage_samples.append(memory_mb)
                    if len(self.memory_usage_samples) > 5:
                        self.memory_usage_samples.pop(0)

                    # Check against thresholds
                    if self.memory_limit_mb:
                        usage_ratio = memory_mb / self.memory_limit_mb
                        if usage_ratio > self.memory_critical_threshold:
                            logger.critical(
                                f"CRITICAL: Memory usage at {usage_ratio:.1%} of limit "
                                f"({memory_mb:.1f}MB/{self.memory_limit_mb}MB). "
                                f"Consider persisting to disk immediately."
                            )
                        elif usage_ratio > self.memory_warning_threshold:
                            logger.warning(
                                f"WARNING: Memory usage at {usage_ratio:.1%} of limit "
                                f"({memory_mb:.1f}MB/{self.memory_limit_mb}MB)."
                            )
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")

                # Check every 5 seconds
                time.sleep(5)

        # Start monitoring thread
        thread = threading.Thread(target=monitor_memory, daemon=True)
        thread.start()

    def get_memory_usage(self):
        """Get current memory usage statistics.

        Returns
        -------
        dict
            Dictionary with memory usage information:
            - current_mb: Current memory usage in MB
            - limit_mb: Memory limit in MB (if set)
            - usage_percent: Percentage of limit used
            - trend: Recent trend (increasing/stable/decreasing)
        """
        if not self.memory_usage_samples:
            return {
                "current_mb": 0,
                "limit_mb": self.memory_limit_mb,
                "usage_percent": 0,
                "trend": "unknown",
            }

        current = self.memory_usage_samples[-1]

        # Determine trend
        trend = "stable"
        if len(self.memory_usage_samples) >= 3:
            recent = self.memory_usage_samples[-3:]
            if recent[2] > recent[0] * 1.05:  # 5% increase
                trend = "increasing"
            elif recent[2] < recent[0] * 0.95:  # 5% decrease
                trend = "decreasing"

        return {
            "current_mb": current,
            "limit_mb": self.memory_limit_mb,
            "usage_percent": (
                (current / self.memory_limit_mb * 100) if self.memory_limit_mb else None
            ),
            "trend": trend,
        }

    def persist_to_disk(self, db_path, tables=None, show_progress=True):
        """Save the in-memory database to disk with selective table persistence.

        Parameters
        ----------
        db_path : str
            Path where the database should be saved
        tables : List[str], optional
            Specific tables to persist (if None, persist all)
        show_progress : bool, optional
            Whether to show progress information during persistence

        Returns
        -------
        dict
            Statistics about the persistence operation
        """
        # Flush all pending changes
        self.logger.flush_all_buffers()

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # Create a new disk-based database
        disk_engine = create_engine(f"sqlite:///{db_path}")

        # Copy schema
        Base.metadata.create_all(disk_engine)

        # Copy data
        source_conn = self.engine.raw_connection()
        dest_conn = disk_engine.raw_connection()

        source_cursor = source_conn.cursor()
        dest_cursor = dest_conn.cursor()

        # Get all tables
        source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [
            table_name
            for (table_name,) in source_cursor.fetchall()
            if not table_name.startswith("sqlite_")
        ]

        # Filter tables if specified
        if tables:
            tables_to_copy = [t for t in all_tables if t in tables]
        else:
            tables_to_copy = all_tables

        if show_progress:
            logger.info(f"Persisting in-memory database to {db_path}")
            logger.info(f"Tables to copy: {', '.join(tables_to_copy)}")

        # Statistics for return
        stats = {
            "tables_copied": 0,
            "rows_copied": 0,
            "tables_skipped": len(all_tables) - len(tables_to_copy),
            "start_time": time.time(),
        }

        # Begin transaction on destination
        dest_cursor.execute("BEGIN TRANSACTION")

        try:
            # Copy each table's data
            for i, table_name in enumerate(tables_to_copy):
                if show_progress:
                    logger.info(
                        f"Copying table {i+1}/{len(tables_to_copy)}: {table_name}"
                    )

                # Get data from source
                source_cursor.execute(f"SELECT * FROM {table_name}")
                rows = source_cursor.fetchall()

                if not rows:
                    if show_progress:
                        logger.info(f"  Table {table_name} is empty, skipping")
                    continue

                # Get column names
                source_cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [info[1] for info in source_cursor.fetchall()]
                placeholders = ", ".join(["?" for _ in columns])
                columns_str = ", ".join(columns)

                # Use more efficient bulk insert for large tables
                if len(rows) > 1000:
                    # SQLite has a limit on the number of parameters in a query
                    # So we need to batch the inserts
                    batch_size = 1000  # Adjust based on your needs
                    for j in range(0, len(rows), batch_size):
                        batch = rows[j : j + batch_size]
                        dest_cursor.executemany(
                            f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})",
                            batch,
                        )
                        if show_progress and len(rows) > 10000 and j % 10000 == 0:
                            logger.info(
                                f"  Progress: {j}/{len(rows)} rows ({j/len(rows)*100:.1f}%)"
                            )
                else:
                    # For smaller tables, insert one by one
                    for row in rows:
                        dest_cursor.execute(
                            f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})",
                            row,
                        )

                stats["tables_copied"] += 1
                stats["rows_copied"] += len(rows)

                if show_progress:
                    logger.info(f"  Copied {len(rows)} rows from {table_name}")

            # Commit transaction
            dest_cursor.execute("COMMIT")

            stats["end_time"] = time.time()
            stats["duration"] = stats["end_time"] - stats["start_time"]

            if show_progress:
                logger.info(
                    f"Database persistence completed in {stats['duration']:.2f} seconds"
                )
                logger.info(
                    f"Copied {stats['rows_copied']} rows across {stats['tables_copied']} tables"
                )

            return stats

        except Exception as e:
            dest_cursor.execute("ROLLBACK")
            logger.error(f"Error during database persistence: {e}")
            raise
        finally:
            source_conn.close()
            dest_conn.close()


class ShardedSimulationDatabase:
    #! Look into this kmore
    """Database implementation that shards data across multiple SQLite files.

    This class distributes simulation data across multiple database files
    based on data type and time periods, improving write performance for
    large simulations.
    """

    def __init__(self, base_path, shard_size=1000, simulation_id=None):
        """Initialize a sharded database.

        Parameters
        ----------
        base_path : str
            Base path for database files
        shard_size : int
            Number of steps per time shard
        simulation_id : str, optional
            Unique identifier for this simulation, by default None
        """
        self.base_path = base_path
        self.shard_size = shard_size
        self.current_step = 0
        self.simulation_id = simulation_id

        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        # Initialize metadata database
        self.metadata_db = SimulationDatabase(
            os.path.join(base_path, "metadata.db"), simulation_id=simulation_id
        )

        # Initialize shard databases
        self.shards = {}
        self._init_shard(0)

        # Create logger that routes to appropriate shards
        self.logger = ShardedDataLogger(self, simulation_id=simulation_id)

    def _get_shard_path(self, shard_id: int, data_type: str) -> str:
        """Get the file path for a specific shard and data type.

        Parameters
        ----------
        shard_id : int
            Shard identifier
        data_type : str
            Type of data (agents, resources, actions, metrics)

        Returns
        -------
        str
            File path for the shard database
        """
        return os.path.join(self.base_path, f"shard_{shard_id}_{data_type}.db")

    def _init_shard(self, shard_id):
        """Initialize databases for a new shard."""
        if shard_id in self.shards:
            return

        self.shards[shard_id] = {
            "agents": SimulationDatabase(
                self._get_shard_path(shard_id, "agents"),
                simulation_id=self.simulation_id,
            ),
            "resources": SimulationDatabase(
                self._get_shard_path(shard_id, "resources"),
                simulation_id=self.simulation_id,
            ),
            "actions": SimulationDatabase(
                self._get_shard_path(shard_id, "actions"),
                simulation_id=self.simulation_id,
            ),
            "metrics": SimulationDatabase(
                self._get_shard_path(shard_id, "metrics"),
                simulation_id=self.simulation_id,
            ),
        }
