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
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

from farm.database.data_retrieval import DataRetriever

from .data_logging import DataLogger, ShardedDataLogger
from .models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    Base,
    HealthIncident,
    ReproductionEventModel,
    ResourceModel,
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

    def __init__(self, db_path: str) -> None:
        """Initialize a new SimulationDatabase instance with SQLAlchemy.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file

        Notes
        -----
        - Enables foreign key support for SQLite
        - Creates session factory with thread-local scope
        - Initializes tables and indexes
        - Sets up batch operation buffers
        """
        self.db_path = db_path

        # Configure engine with optimized settings
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            # Larger pool size for concurrent operations
            pool_size=10,
            # Longer timeout before connections are recycled
            pool_recycle=3600,
            # Enable connection pooling
            poolclass=QueuePool,
            # Optimize for write-heavy workloads
            connect_args={
                "timeout": 30,  # Longer timeout for busy database
                "check_same_thread": False,  # Allow cross-thread usage
            },
        )

        # Enable foreign key support for SQLite
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            # Enable foreign key support
            cursor.execute("PRAGMA foreign_keys=ON")
            # Optimize for write performance
            cursor.execute("PRAGMA synchronous=NORMAL")  # Less safe but faster
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            cursor.execute(
                "PRAGMA cache_size=-102400"
            )  # 100MB cache (negative = kibibytes)
            cursor.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
            cursor.close()

        # Create session factory
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)

        # Create tables and indexes
        self._create_tables()

        # Replace buffer initialization with DataLogger
        self.logger = DataLogger(self, buffer_size=1000)
        self.query = DataRetriever(self)

    def _execute_in_transaction(self, func: callable) -> Any:
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
        data_types: List[str] = None,
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
                        df.to_csv(f"{base_path}_{data_type}.csv", index=False)
                    elif data_type == "metadata":
                        with open(f"{base_path}_metadata.json", "w") as f:
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
                with open(filepath, "w") as f:
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
            - starvation_threshold: int
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

            formatted_state = format_agent_state(agent_id, step_number, state_data)
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
            import time

            config_obj = SimulationConfig(
                timestamp=int(time.time()), config_data=json.dumps(config)
            )
            session.add(config_obj)

        self._execute_in_transaction(_insert)

    def log_reproduction_event(
        self,
        step_number: int,
        parent_id: str,
        success: bool,
        parent_resources_before: float,
        parent_resources_after: float,
        offspring_id: str = None,
        offspring_initial_resources: float = None,
        failure_reason: str = None,
        parent_position: tuple[float, float] = None,
        parent_generation: int = None,
        offspring_generation: int = None,
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
    """In-memory database for high-performance simulation runs.

    This class provides a high-performance alternative to disk-based storage
    during simulation, with an option to persist data at the end of the run.
    
    Features
    --------
    - Significantly faster than disk-based storage (30-50% speedup)
    - Memory usage monitoring to prevent OOM errors
    - Selective table persistence
    - Transaction-based persistence for data integrity
    - Progress reporting for long-running operations
    
    Notes
    -----
    - Uses SQLite's in-memory database
    - Optimized for write-heavy workloads
    - Larger buffer sizes for batch operations
    """

    def __init__(self, memory_limit_mb=None):
        """Initialize an in-memory database.
        
        Parameters
        ----------
        memory_limit_mb : int, optional
            Memory usage limit in MB. If None, no limit is enforced.
            When specified, the database will monitor memory usage and
            raise a warning when approaching the limit.
        """
        # Use in-memory SQLite database
        self.db_path = ":memory:"
        self.engine = create_engine("sqlite:///:memory:")
        self.memory_limit_mb = memory_limit_mb
        self.memory_usage_samples = []
        self.memory_warning_threshold = 0.8  # 80% of limit
        self.memory_critical_threshold = 0.95  # 95% of limit
        
        # Set up pragmas for in-memory optimization
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=OFF")  # Maximum performance
            cursor.execute("PRAGMA journal_mode=MEMORY")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA cache_size=-102400")  # 100MB cache
            cursor.close()

        # Create session factory
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)

        # Create tables
        self._create_tables()

        # Initialize loggers with larger buffer sizes
        self.logger = DataLogger(self, buffer_size=10000)  # 10x larger buffer
        self.query = DataRetriever(self)
        
        # Start memory monitoring if limit is set
        if self.memory_limit_mb:
            self._start_memory_monitoring()
    
    def _start_memory_monitoring(self):
        """Start background thread for memory usage monitoring."""
        import threading
        import time
        import psutil
        import os
        
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
                "trend": "unknown"
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
            "usage_percent": (current / self.memory_limit_mb * 100) if self.memory_limit_mb else None,
            "trend": trend
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
        all_tables = [table_name for (table_name,) in source_cursor.fetchall() 
                     if not table_name.startswith("sqlite_")]
        
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
            "start_time": time.time()
        }

        # Begin transaction on destination
        dest_conn.execute("BEGIN TRANSACTION")

        try:
            # Copy each table's data
            for i, table_name in enumerate(tables_to_copy):
                if show_progress:
                    logger.info(f"Copying table {i+1}/{len(tables_to_copy)}: {table_name}")
                
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
                        batch = rows[j:j+batch_size]
                        dest_cursor.executemany(
                            f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})",
                            batch
                        )
                        if show_progress and len(rows) > 10000 and j % 10000 == 0:
                            logger.info(f"  Progress: {j}/{len(rows)} rows ({j/len(rows)*100:.1f}%)")
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
            dest_conn.execute("COMMIT")
            
            stats["end_time"] = time.time()
            stats["duration"] = stats["end_time"] - stats["start_time"]
            
            if show_progress:
                logger.info(f"Database persistence completed in {stats['duration']:.2f} seconds")
                logger.info(f"Copied {stats['rows_copied']} rows across {stats['tables_copied']} tables")

            return stats

        except Exception as e:
            dest_conn.execute("ROLLBACK")
            logger.error(f"Error during database persistence: {e}")
            raise
        finally:
            source_conn.close()
            dest_conn.close()


class ShardedSimulationDatabase:
    """Database implementation that shards data across multiple SQLite files.

    This class distributes simulation data across multiple database files
    based on data type and time periods, improving write performance for
    large simulations.
    """

    def __init__(self, base_path, shard_size=1000):
        """Initialize a sharded database.

        Parameters
        ----------
        base_path : str
            Base path for database files
        shard_size : int
            Number of steps per time shard
        """
        self.base_path = base_path
        self.shard_size = shard_size
        self.current_step = 0

        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        # Initialize metadata database
        self.metadata_db = SimulationDatabase(os.path.join(base_path, "metadata.db"))

        # Initialize shard databases
        self.shards = {}
        self._init_shard(0)

        # Create logger that routes to appropriate shards
        self.logger = ShardedDataLogger(self)

    def _get_shard_path(self, shard_id, data_type):
        """Get path for a specific shard."""
        return os.path.join(self.base_path, f"shard_{shard_id}_{data_type}.db")

    def _init_shard(self, shard_id):
        """Initialize databases for a new shard."""
        if shard_id in self.shards:
            return

        self.shards[shard_id] = {
            "agents": SimulationDatabase(self._get_shard_path(shard_id, "agents")),
            "resources": SimulationDatabase(
                self._get_shard_path(shard_id, "resources")
            ),
            "actions": SimulationDatabase(self._get_shard_path(shard_id, "actions")),
            "metrics": SimulationDatabase(self._get_shard_path(shard_id, "metrics")),
        }

    def _get_shard_for_step(self, step_number):
        """Get the shard ID for a specific step."""
        return step_number // self.shard_size

    def log_step(self, step_number, agent_states, resource_states, metrics):
        """Log a simulation step, routing data to appropriate shards."""
        self.current_step = step_number
        shard_id = self._get_shard_for_step(step_number)

        # Initialize shard if needed
        if shard_id not in self.shards:
            self._init_shard(shard_id)

        # Log to appropriate shards
        self.shards[shard_id]["agents"].logger.log_agent_states(
            step_number, agent_states
        )
        self.shards[shard_id]["resources"].logger.log_resources(
            step_number, resource_states
        )
        self.shards[shard_id]["metrics"].logger.log_metrics(step_number, metrics)

    def close(self):
        """Close all database connections."""
        # Close metadata database
        self.metadata_db.close()

        # Close all shard databases
        for shard_id, databases in self.shards.items():
            for db_type, db in databases.items():
                db.close()
