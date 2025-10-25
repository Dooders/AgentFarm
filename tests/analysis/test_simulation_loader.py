"""
Tests for SimulationLoader class.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from farm.analysis.comparative.simulation_loader import SimulationLoader, SimulationData


class TestSimulationLoader:
    """Test cases for SimulationLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.simulation_dir = self.temp_dir / "test_simulation"
        self.simulation_dir.mkdir()
        
        # Create test files
        self.db_path = self.simulation_dir / "simulation.db"
        self.config_path = self.simulation_dir / "config.json"
        self.log_path = self.simulation_dir / "simulation.log"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_database(self):
        """Create a test SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create test tables
            cursor.execute("""
                CREATE TABLE simulation_metadata (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    created_at TEXT,
                    parameters TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE step_metrics (
                    step INTEGER,
                    agent_count INTEGER,
                    action_count INTEGER,
                    timestamp TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE agents (
                    id INTEGER PRIMARY KEY,
                    type TEXT,
                    position_x REAL,
                    position_y REAL
                )
            """)
            
            # Insert test data
            cursor.execute("""
                INSERT INTO simulation_metadata (name, created_at, parameters)
                VALUES ('test_simulation', '2024-01-01T00:00:00Z', '{"test": true}')
            """)
            
            for i in range(10):
                cursor.execute("""
                    INSERT INTO step_metrics (step, agent_count, action_count, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (i, 100 + i, 50 + i, f'2024-01-01T00:00:{i:02d}Z'))
            
            for i in range(5):
                cursor.execute("""
                    INSERT INTO agents (type, position_x, position_y)
                    VALUES (?, ?, ?)
                """, (f'agent_type_{i}', float(i), float(i * 2)))
            
            conn.commit()
    
    def create_test_config(self):
        """Create a test configuration file."""
        config = {
            "simulation": {
                "name": "test_simulation",
                "duration": 1000,
                "agents": 100
            },
            "environment": {
                "width": 1000,
                "height": 1000
            },
            "parameters": {
                "learning_rate": 0.01,
                "exploration_rate": 0.1
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def create_test_log(self):
        """Create a test log file."""
        log_content = """2024-01-01T00:00:00Z INFO: Simulation started
2024-01-01T00:00:01Z INFO: Execution time: 1.5 seconds
2024-01-01T00:00:02Z INFO: Memory usage: 128 MB
2024-01-01T00:00:03Z INFO: Throughput: 100 ops/sec
2024-01-01T00:00:04Z WARNING: Low memory warning
2024-01-01T00:00:05Z ERROR: Database connection failed
2024-01-01T00:00:06Z INFO: Iteration 1 completed
2024-01-01T00:00:07Z INFO: Agents: 100
2024-01-01T00:00:08Z INFO: Step 1 completed
"""
        
        with open(self.log_path, 'w') as f:
            f.write(log_content)
    
    def test_init_valid_directory(self):
        """Test initialization with valid simulation directory."""
        self.create_test_database()
        self.create_test_config()
        self.create_test_log()
        
        loader = SimulationLoader(self.simulation_dir)
        
        assert loader.simulation_path == self.simulation_dir
        assert loader.db_path == self.db_path
        assert loader.config_path == self.config_path
        assert len(loader.log_paths) == 1
        assert loader.log_paths[0] == self.log_path
    
    def test_init_missing_files(self):
        """Test initialization with missing files (should warn but not fail)."""
        # Create only the directory
        loader = SimulationLoader(self.simulation_dir)
        
        assert loader.simulation_path == self.simulation_dir
        assert not loader.db_path.exists()
        assert not loader.config_path.exists()
        assert len(loader.log_paths) == 0
    
    def test_init_nonexistent_directory(self):
        """Test initialization with non-existent directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            SimulationLoader(nonexistent_dir)
    
    def test_init_file_not_directory(self):
        """Test initialization with file instead of directory."""
        file_path = self.temp_dir / "not_a_directory.txt"
        file_path.write_text("test")
        
        with pytest.raises(NotADirectoryError):
            SimulationLoader(file_path)
    
    def test_load_config_success(self):
        """Test successful config loading."""
        self.create_test_config()
        loader = SimulationLoader(self.simulation_dir)
        
        config = loader.load_config()
        
        assert config["simulation"]["name"] == "test_simulation"
        assert config["simulation"]["duration"] == 1000
        assert config["environment"]["width"] == 1000
    
    def test_load_config_missing_file(self):
        """Test config loading with missing file."""
        loader = SimulationLoader(self.simulation_dir)
        
        config = loader.load_config()
        
        assert config == {}
    
    def test_load_config_invalid_json(self):
        """Test config loading with invalid JSON."""
        with open(self.config_path, 'w') as f:
            f.write("invalid json content")
        
        loader = SimulationLoader(self.simulation_dir)
        
        config = loader.load_config()
        
        assert config == {}
    
    def test_load_database_metrics_success(self):
        """Test successful database metrics loading."""
        self.create_test_database()
        loader = SimulationLoader(self.simulation_dir)
        
        metrics = loader.load_database_metrics()
        
        assert "database_info" in metrics
        assert "tables" in metrics
        assert "simulation_metadata" in metrics
        assert "step_metrics" in metrics
        assert "agent_data" in metrics
        
        # Check database info
        assert "sqlite_version" in metrics["database_info"]
        assert "database_size_bytes" in metrics["database_info"]
        
        # Check tables
        assert "simulation_metadata" in metrics["tables"]
        assert "step_metrics" in metrics["tables"]
        assert "agents" in metrics["tables"]
        
        # Check step metrics
        step_metrics = metrics["step_metrics"]["step_metrics"]
        assert step_metrics["row_count"] == 10
        assert len(step_metrics["sample_data"]) == 10
    
    def test_load_database_metrics_missing_file(self):
        """Test database metrics loading with missing file."""
        loader = SimulationLoader(self.simulation_dir)
        
        metrics = loader.load_database_metrics()
        
        assert metrics == {}
    
    def test_load_log_metrics_success(self):
        """Test successful log metrics loading."""
        self.create_test_log()
        loader = SimulationLoader(self.simulation_dir)
        
        metrics = loader.load_log_metrics()
        
        assert "log_files" in metrics
        assert "performance_metrics" in metrics
        assert "error_metrics" in metrics
        assert "summary_stats" in metrics
        
        assert len(metrics["log_files"]) == 1
        assert "simulation.log" in metrics["log_files"]
    
    def test_load_log_metrics_no_files(self):
        """Test log metrics loading with no log files."""
        loader = SimulationLoader(self.simulation_dir)
        
        metrics = loader.load_log_metrics()
        
        assert metrics == {}
    
    def test_load_simulation_data_complete(self):
        """Test loading complete simulation data."""
        self.create_test_database()
        self.create_test_config()
        self.create_test_log()
        
        loader = SimulationLoader(self.simulation_dir)
        simulation_data = loader.load_simulation_data()
        
        assert isinstance(simulation_data, SimulationData)
        assert simulation_data.simulation_path == self.simulation_dir
        assert simulation_data.config["simulation"]["name"] == "test_simulation"
        assert "database_info" in simulation_data.database_metrics
        assert "log_files" in simulation_data.log_metrics
        assert "simulation_path" in simulation_data.metadata
    
    def test_load_simulation_data_partial(self):
        """Test loading simulation data with only some files."""
        self.create_test_config()
        # No database or log files
        
        loader = SimulationLoader(self.simulation_dir)
        simulation_data = loader.load_simulation_data()
        
        assert isinstance(simulation_data, SimulationData)
        assert simulation_data.config["simulation"]["name"] == "test_simulation"
        assert simulation_data.database_metrics == {}
        assert simulation_data.log_metrics == {}
    
    def test_create_metadata(self):
        """Test metadata creation."""
        self.create_test_database()
        self.create_test_config()
        self.create_test_log()
        
        loader = SimulationLoader(self.simulation_dir)
        metadata = loader._create_metadata()
        
        assert metadata["simulation_path"] == str(self.simulation_dir)
        assert metadata["database_exists"] is True
        assert metadata["config_exists"] is True
        assert metadata["log_files_count"] == 1
        assert "simulation.log" in metadata["log_files"]
    
    def test_get_database_info(self):
        """Test database info extraction."""
        self.create_test_database()
        loader = SimulationLoader(self.simulation_dir)
        
        with sqlite3.connect(self.db_path) as conn:
            info = loader._get_database_info(conn)
        
        assert "sqlite_version" in info
        assert "database_size_bytes" in info
        assert "database_size_mb" in info
        assert info["database_size_bytes"] > 0
    
    def test_get_table_info(self):
        """Test table info extraction."""
        self.create_test_database()
        loader = SimulationLoader(self.simulation_dir)
        
        with sqlite3.connect(self.db_path) as conn:
            table_info = loader._get_table_info(conn)
        
        assert "simulation_metadata" in table_info
        assert "step_metrics" in table_info
        assert "agents" in table_info
        
        assert table_info["simulation_metadata"]["row_count"] == 1
        assert table_info["step_metrics"]["row_count"] == 10
        assert table_info["agents"]["row_count"] == 5
    
    def test_parse_log_file(self):
        """Test log file parsing."""
        self.create_test_log()
        loader = SimulationLoader(self.simulation_dir)
        
        log_data = loader._parse_log_file(self.log_path)
        
        assert log_data["file_name"] == "simulation.log"
        assert log_data["total_lines"] == 8
        assert log_data["warnings"] == 1
        assert log_data["errors"] == 1
        assert "performance_metrics" in log_data