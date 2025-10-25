"""
Simulation data loader for file-based comparison.

This module provides functionality to load simulation data from file paths,
including SQLite databases, configuration files, and log files.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from dataclasses import dataclass

from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationData:
    """Container for all simulation data loaded from files."""
    
    simulation_path: Path
    config: Dict[str, Any]
    database_metrics: Dict[str, Any]
    log_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate loaded data after initialization."""
        if not self.config:
            logger.warning(f"No configuration data loaded from {self.simulation_path}")
        if not self.database_metrics:
            logger.warning(f"No database metrics loaded from {self.simulation_path}")
        if not self.log_metrics:
            logger.warning(f"No log metrics loaded from {self.simulation_path}")


class SimulationLoader:
    """Loads simulation data from file paths for comparison."""
    
    def __init__(self, simulation_path: Union[str, Path]):
        """Initialize loader with simulation directory path.
        
        Args:
            simulation_path: Path to simulation directory containing:
                - simulation.db (SQLite database)
                - config.json (simulation configuration)
                - *.log (log files)
        """
        self.simulation_path = Path(simulation_path)
        self.db_path = self.simulation_path / "simulation.db"
        self.config_path = self.simulation_path / "config.json"
        self.log_paths = list(self.simulation_path.glob("*.log"))
        
        # Validate simulation directory structure
        self._validate_simulation_directory()
    
    def _validate_simulation_directory(self) -> None:
        """Validate that the simulation directory contains required files."""
        if not self.simulation_path.exists():
            raise FileNotFoundError(f"Simulation directory does not exist: {self.simulation_path}")
        
        if not self.simulation_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.simulation_path}")
        
        missing_files = []
        if not self.db_path.exists():
            missing_files.append("simulation.db")
        if not self.config_path.exists():
            missing_files.append("config.json")
        
        if missing_files:
            logger.warning(f"Missing files in {self.simulation_path}: {missing_files}")
        
        if not self.log_paths:
            logger.warning(f"No log files found in {self.simulation_path}")
    
    def load_simulation_data(self) -> SimulationData:
        """Load all simulation data from files.
        
        Returns:
            SimulationData object containing all loaded data
        """
        logger.info(f"Loading simulation data from {self.simulation_path}")
        
        # Load configuration
        config = self.load_config()
        
        # Load database metrics
        database_metrics = self.load_database_metrics()
        
        # Load log metrics
        log_metrics = self.load_log_metrics()
        
        # Create metadata
        metadata = self._create_metadata()
        
        return SimulationData(
            simulation_path=self.simulation_path,
            config=config,
            database_metrics=database_metrics,
            log_metrics=log_metrics,
            metadata=metadata
        )
    
    def load_config(self) -> Dict[str, Any]:
        """Load and parse configuration file.
        
        Returns:
            Dictionary containing configuration data
        """
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON config file {self.config_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_path}: {e}")
            return {}
    
    def load_database_metrics(self) -> Dict[str, Any]:
        """Load metrics from SQLite database.
        
        Returns:
            Dictionary containing database metrics and metadata
        """
        if not self.db_path.exists():
            logger.warning(f"Database file not found: {self.db_path}")
            return {}
        
        try:
            metrics = {}
            
            # Connect to database
            with sqlite3.connect(self.db_path) as conn:
                # Get basic database info
                metrics['database_info'] = self._get_database_info(conn)
                
                # Get table information
                metrics['tables'] = self._get_table_info(conn)
                
                # Load simulation metadata if available
                metrics['simulation_metadata'] = self._load_simulation_metadata(conn)
                
                # Load step metrics if available
                metrics['step_metrics'] = self._load_step_metrics(conn)
                
                # Load agent data if available
                metrics['agent_data'] = self._load_agent_data(conn)
                
                # Load action data if available
                metrics['action_data'] = self._load_action_data(conn)
            
            logger.info(f"Loaded database metrics from {self.db_path}")
            return metrics
            
        except sqlite3.Error as e:
            logger.error(f"Database error loading {self.db_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load database metrics from {self.db_path}: {e}")
            return {}
    
    def load_log_metrics(self) -> Dict[str, Any]:
        """Parse log files for performance metrics.
        
        Returns:
            Dictionary containing log-based metrics
        """
        if not self.log_paths:
            logger.warning(f"No log files found in {self.simulation_path}")
            return {}
        
        try:
            metrics = {
                'log_files': [str(path) for path in self.log_paths],
                'performance_metrics': {},
                'error_metrics': {},
                'summary_stats': {}
            }
            
            # Parse each log file
            for log_path in self.log_paths:
                log_data = self._parse_log_file(log_path)
                if log_data:
                    # Merge log data
                    for key, value in log_data.items():
                        if key in metrics:
                            if isinstance(metrics[key], dict) and isinstance(value, dict):
                                metrics[key].update(value)
                            elif isinstance(metrics[key], list) and isinstance(value, list):
                                metrics[key].extend(value)
                        else:
                            metrics[key] = value
            
            logger.info(f"Loaded log metrics from {len(self.log_paths)} files")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load log metrics: {e}")
            return {}
    
    def _get_database_info(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get basic database information."""
        try:
            cursor = conn.cursor()
            
            # Get database version
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            size_result = cursor.fetchone()
            size = size_result[0] if size_result else 0
            
            return {
                'sqlite_version': version,
                'database_size_bytes': size,
                'database_size_mb': round(size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.warning(f"Could not get database info: {e}")
            return {}
    
    def _get_table_info(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get information about database tables."""
        try:
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            table_info = {}
            for table in tables:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                
                table_info[table] = {
                    'row_count': row_count,
                    'columns': columns
                }
            
            return table_info
        except Exception as e:
            logger.warning(f"Could not get table info: {e}")
            return {}
    
    def _load_simulation_metadata(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Load simulation metadata from database."""
        try:
            cursor = conn.cursor()
            
            # Try to find simulation metadata table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%simulation%'")
            sim_tables = [row[0] for row in cursor.fetchall()]
            
            metadata = {}
            for table in sim_tables:
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                    columns = [description[0] for description in cursor.description]
                    row = cursor.fetchone()
                    
                    if row:
                        metadata[table] = dict(zip(columns, row))
                except Exception as e:
                    logger.debug(f"Could not load metadata from {table}: {e}")
            
            return metadata
        except Exception as e:
            logger.warning(f"Could not load simulation metadata: {e}")
            return {}
    
    def _load_step_metrics(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Load step-by-step metrics from database."""
        try:
            cursor = conn.cursor()
            
            # Look for metrics tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE '%metric%' OR name LIKE '%step%')")
            metric_tables = [row[0] for row in cursor.fetchall()]
            
            step_metrics = {}
            for table in metric_tables:
                try:
                    # Get sample of data
                    cursor.execute(f"SELECT * FROM {table} LIMIT 100")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    if rows:
                        df = pd.DataFrame(rows, columns=columns)
                        step_metrics[table] = {
                            'sample_data': df.to_dict('records'),
                            'columns': columns,
                            'row_count': len(rows)
                        }
                except Exception as e:
                    logger.debug(f"Could not load step metrics from {table}: {e}")
            
            return step_metrics
        except Exception as e:
            logger.warning(f"Could not load step metrics: {e}")
            return {}
    
    def _load_agent_data(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Load agent-related data from database."""
        try:
            cursor = conn.cursor()
            
            # Look for agent tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%agent%'")
            agent_tables = [row[0] for row in cursor.fetchall()]
            
            agent_data = {}
            for table in agent_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        # Get sample data
                        cursor.execute(f"SELECT * FROM {table} LIMIT 10")
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        agent_data[table] = {
                            'total_count': count,
                            'columns': columns,
                            'sample_data': [dict(zip(columns, row)) for row in rows[:5]]
                        }
                except Exception as e:
                    logger.debug(f"Could not load agent data from {table}: {e}")
            
            return agent_data
        except Exception as e:
            logger.warning(f"Could not load agent data: {e}")
            return {}
    
    def _load_action_data(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Load action-related data from database."""
        try:
            cursor = conn.cursor()
            
            # Look for action tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%action%'")
            action_tables = [row[0] for row in cursor.fetchall()]
            
            action_data = {}
            for table in action_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        # Get action frequency
                        cursor.execute(f"SELECT action_type, COUNT(*) as frequency FROM {table} GROUP BY action_type ORDER BY frequency DESC LIMIT 10")
                        action_freq = cursor.fetchall()
                        
                        action_data[table] = {
                            'total_count': count,
                            'action_frequencies': [{'action_type': row[0], 'frequency': row[1]} for row in action_freq]
                        }
                except Exception as e:
                    logger.debug(f"Could not load action data from {table}: {e}")
            
            return action_data
        except Exception as e:
            logger.warning(f"Could not load action data: {e}")
            return {}
    
    def _parse_log_file(self, log_path: Path) -> Dict[str, Any]:
        """Parse a single log file for metrics."""
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            log_data = {
                'file_name': log_path.name,
                'total_lines': len(lines),
                'performance_metrics': {},
                'error_metrics': {},
                'warnings': 0,
                'errors': 0
            }
            
            # Simple parsing for common log patterns
            for line in lines:
                line_lower = line.lower()
                
                # Count warnings and errors
                if 'warning' in line_lower:
                    log_data['warnings'] += 1
                if 'error' in line_lower:
                    log_data['errors'] += 1
                
                # Look for performance metrics (basic patterns)
                if 'time' in line_lower and ('ms' in line or 'seconds' in line_lower):
                    # Extract timing information
                    import re
                    time_match = re.search(r'(\d+(?:\.\d+)?)\s*(ms|seconds?|s)', line)
                    if time_match:
                        value = float(time_match.group(1))
                        unit = time_match.group(2)
                        if 'ms' in unit:
                            value = value / 1000  # Convert to seconds
                        log_data['performance_metrics'][f'timing_{len(log_data["performance_metrics"])}'] = value
            
            return log_data
            
        except Exception as e:
            logger.warning(f"Could not parse log file {log_path}: {e}")
            return {}
    
    def _create_metadata(self) -> Dict[str, Any]:
        """Create metadata about the loaded simulation."""
        return {
            'simulation_path': str(self.simulation_path),
            'database_exists': self.db_path.exists(),
            'config_exists': self.config_path.exists(),
            'log_files_count': len(self.log_paths),
            'log_files': [str(path) for path in self.log_paths]
        }