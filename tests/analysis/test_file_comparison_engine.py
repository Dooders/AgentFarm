"""
Tests for FileComparisonEngine class.
"""

import tempfile
import sqlite3
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from farm.analysis.comparative.file_comparison_engine import FileComparisonEngine
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult,
    ComparisonSummary,
    ConfigComparisonResult,
    DatabaseComparisonResult,
    LogComparisonResult,
    MetricsComparisonResult
)


class TestFileComparisonEngine:
    """Test cases for FileComparisonEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        
        # Create test simulation directories
        self.sim1_dir = self.temp_dir / "simulation1"
        self.sim2_dir = self.temp_dir / "simulation2"
        self.sim1_dir.mkdir()
        self.sim2_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_simulation(self, sim_dir: Path, name: str, duration: int = 1000, agents: int = 100):
        """Create a test simulation directory with all required files."""
        # Create database
        db_path = sim_dir / "simulation.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE simulation_metadata (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    duration INTEGER,
                    agent_count INTEGER
                )
            """)
            cursor.execute("""
                INSERT INTO simulation_metadata (name, duration, agent_count)
                VALUES (?, ?, ?)
            """, (name, duration, agents))
            conn.commit()
        
        # Create config
        config_path = sim_dir / "config.json"
        config = {
            "simulation": {
                "name": name,
                "duration": duration,
                "agents": agents
            },
            "environment": {
                "width": 1000,
                "height": 1000
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Create log file
        log_path = sim_dir / "simulation.log"
        with open(log_path, 'w') as f:
            f.write(f"""2024-01-01T00:00:00Z INFO: Simulation {name} started
2024-01-01T00:00:01Z INFO: Execution time: {duration/1000} seconds
2024-01-01T00:00:02Z INFO: Agents: {agents}
2024-01-01T00:00:03Z INFO: Simulation completed
""")
        
        return sim_dir
    
    def test_init_with_output_dir(self):
        """Test initialization with custom output directory."""
        engine = FileComparisonEngine(self.output_dir)
        
        assert engine.output_dir == self.output_dir
        assert self.output_dir.exists()
    
    def test_init_without_output_dir(self):
        """Test initialization without output directory."""
        engine = FileComparisonEngine()
        
        assert engine.output_dir.exists()
        assert engine.output_dir.is_dir()
    
    @patch('farm.analysis.comparative.file_comparison_engine.SimulationLoader')
    def test_compare_simulations_success(self, mock_loader_class):
        """Test successful simulation comparison."""
        # Create test simulations
        self.create_test_simulation(self.sim1_dir, "sim1", 1000, 100)
        self.create_test_simulation(self.sim2_dir, "sim2", 2000, 150)
        
        # Mock simulation data
        mock_sim1_data = MagicMock()
        mock_sim1_data.config = {"simulation": {"name": "sim1", "duration": 1000}}
        mock_sim1_data.metadata = {"database_exists": True, "log_files": ["simulation.log"]}
        mock_sim1_data.simulation_path = self.sim1_dir
        
        mock_sim2_data = MagicMock()
        mock_sim2_data.config = {"simulation": {"name": "sim2", "duration": 2000}}
        mock_sim2_data.metadata = {"database_exists": True, "log_files": ["simulation.log"]}
        mock_sim2_data.simulation_path = self.sim2_dir
        
        # Mock loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load_simulation_data.side_effect = [mock_sim1_data, mock_sim2_data]
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine.compare_simulations(self.sim1_dir, self.sim2_dir)
        
        assert isinstance(result, SimulationComparisonResult)
        assert result.simulation1_path == self.sim1_dir
        assert result.simulation2_path == self.sim2_dir
        assert isinstance(result.comparison_summary, ComparisonSummary)
        assert isinstance(result.config_comparison, ConfigComparisonResult)
        assert isinstance(result.database_comparison, DatabaseComparisonResult)
        assert isinstance(result.log_comparison, LogComparisonResult)
        assert isinstance(result.metrics_comparison, MetricsComparisonResult)
    
    def test_compare_simulations_real_data(self):
        """Test simulation comparison with real data."""
        # Create test simulations
        self.create_test_simulation(self.sim1_dir, "sim1", 1000, 100)
        self.create_test_simulation(self.sim2_dir, "sim2", 2000, 150)
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine.compare_simulations(self.sim1_dir, self.sim2_dir)
        
        assert isinstance(result, SimulationComparisonResult)
        assert result.simulation1_path == self.sim1_dir
        assert result.simulation2_path == self.sim2_dir
        
        # Should have some differences
        assert result.comparison_summary.total_differences > 0
    
    def test_compare_simulations_without_logs(self):
        """Test simulation comparison without log analysis."""
        self.create_test_simulation(self.sim1_dir, "sim1", 1000, 100)
        self.create_test_simulation(self.sim2_dir, "sim2", 2000, 150)
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine.compare_simulations(
            self.sim1_dir, 
            self.sim2_dir, 
            include_logs=False
        )
        
        assert isinstance(result, SimulationComparisonResult)
        # Log comparison should be empty
        assert result.log_comparison.performance_differences == {}
        assert result.log_comparison.error_differences == {}
    
    def test_compare_simulations_without_metrics(self):
        """Test simulation comparison without metrics analysis."""
        self.create_test_simulation(self.sim1_dir, "sim1", 1000, 100)
        self.create_test_simulation(self.sim2_dir, "sim2", 2000, 150)
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine.compare_simulations(
            self.sim1_dir, 
            self.sim2_dir, 
            include_metrics=False
        )
        
        assert isinstance(result, SimulationComparisonResult)
        # Metrics comparison should be empty
        assert result.metrics_comparison.metric_differences == {}
        assert result.metrics_comparison.performance_comparison == {}
    
    def test_load_simulation_data_success(self):
        """Test successful simulation data loading."""
        self.create_test_simulation(self.sim1_dir, "sim1", 1000, 100)
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine._load_simulation_data(self.sim1_dir)
        
        assert result is not None
        assert hasattr(result, 'config')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'simulation_path')
    
    def test_load_simulation_data_failure(self):
        """Test simulation data loading failure."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        engine = FileComparisonEngine(self.output_dir)
        
        with pytest.raises(Exception):
            engine._load_simulation_data(nonexistent_dir)
    
    def test_compare_configurations(self):
        """Test configuration comparison."""
        mock_sim1_data = MagicMock()
        mock_sim1_data.config = {"simulation": {"name": "sim1", "duration": 1000}}
        
        mock_sim2_data = MagicMock()
        mock_sim2_data.config = {"simulation": {"name": "sim2", "duration": 2000}}
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine._compare_configurations(mock_sim1_data, mock_sim2_data)
        
        assert isinstance(result, ConfigComparisonResult)
        assert 'changed' in result.differences or 'added' in result.differences or 'removed' in result.differences
    
    def test_compare_databases(self):
        """Test database comparison."""
        # Create test databases
        self.create_test_simulation(self.sim1_dir, "sim1", 1000, 100)
        self.create_test_simulation(self.sim2_dir, "sim2", 2000, 150)
        
        mock_sim1_data = MagicMock()
        mock_sim1_data.metadata = {"database_exists": True}
        mock_sim1_data.simulation_path = self.sim1_dir
        
        mock_sim2_data = MagicMock()
        mock_sim2_data.metadata = {"database_exists": True}
        mock_sim2_data.simulation_path = self.sim2_dir
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine._compare_databases(mock_sim1_data, mock_sim2_data)
        
        assert isinstance(result, DatabaseComparisonResult)
        assert 'metric_differences' in result.metric_differences
    
    def test_compare_databases_no_database(self):
        """Test database comparison when one simulation has no database."""
        mock_sim1_data = MagicMock()
        mock_sim1_data.metadata = {"database_exists": False}
        
        mock_sim2_data = MagicMock()
        mock_sim2_data.metadata = {"database_exists": True}
        mock_sim2_data.simulation_path = self.sim2_dir
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine._compare_databases(mock_sim1_data, mock_sim2_data)
        
        assert isinstance(result, DatabaseComparisonResult)
        assert result.schema_differences == {}
        assert result.data_differences == {}
    
    def test_compare_logs(self):
        """Test log comparison."""
        mock_sim1_data = MagicMock()
        mock_sim1_data.metadata = {"log_files": ["simulation.log"]}
        
        mock_sim2_data = MagicMock()
        mock_sim2_data.metadata = {"log_files": ["simulation.log"]}
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine._compare_logs(mock_sim1_data, mock_sim2_data)
        
        assert isinstance(result, LogComparisonResult)
        assert 'sim1_log_files' in result.summary
        assert 'sim2_log_files' in result.summary
    
    def test_compare_logs_no_logs(self):
        """Test log comparison when no log files exist."""
        mock_sim1_data = MagicMock()
        mock_sim1_data.metadata = {"log_files": []}
        
        mock_sim2_data = MagicMock()
        mock_sim2_data.metadata = {"log_files": []}
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine._compare_logs(mock_sim1_data, mock_sim2_data)
        
        assert isinstance(result, LogComparisonResult)
        assert result.performance_differences == {}
        assert result.error_differences == {}
    
    @patch('farm.analysis.comparative.file_comparison_engine.MetricsLoader')
    def test_compare_metrics(self, mock_metrics_loader_class):
        """Test metrics comparison."""
        # Mock metrics data
        mock_metrics1 = MagicMock()
        mock_metrics1.metrics = {"metric1": 100, "metric2": 200}
        
        mock_metrics2 = MagicMock()
        mock_metrics2.metrics = {"metric1": 150, "metric2": 250}
        
        # Mock metrics loader instances
        mock_loader1 = MagicMock()
        mock_loader2 = MagicMock()
        mock_loader1.load_comprehensive_metrics.return_value = mock_metrics1
        mock_loader1.load_basic_metrics.return_value = mock_metrics1
        mock_loader2.load_comprehensive_metrics.return_value = mock_metrics2
        mock_loader2.load_basic_metrics.return_value = mock_metrics2
        
        # Configure the class to return different instances
        mock_metrics_loader_class.side_effect = [mock_loader1, mock_loader2]
        
        mock_sim1_data = MagicMock()
        mock_sim1_data.simulation_path = self.sim1_dir
        
        mock_sim2_data = MagicMock()
        mock_sim2_data.simulation_path = self.sim2_dir
        
        engine = FileComparisonEngine(self.output_dir)
        result = engine._compare_metrics(mock_sim1_data, mock_sim2_data)
        
        assert isinstance(result, MetricsComparisonResult)
        assert 'metric1' in result.metric_differences
        assert 'metric2' in result.metric_differences
    
    def test_compare_performance_metrics(self):
        """Test performance metrics comparison."""
        engine = FileComparisonEngine(self.output_dir)
        
        sim1_perf = {
            'aggregated_metrics': {
                'execution_time': 1.5,
                'memory_usage': 128,
                'throughput': 100
            }
        }
        
        sim2_perf = {
            'aggregated_metrics': {
                'execution_time': 2.0,
                'memory_usage': 256,
                'throughput': 150
            }
        }
        
        result = engine._compare_performance_metrics(sim1_perf, sim2_perf)
        
        assert 'execution_time' in result
        assert 'memory_usage' in result
        assert 'throughput' in result
        
        assert result['execution_time']['sim1_value'] == 1.5
        assert result['execution_time']['sim2_value'] == 2.0
        assert result['execution_time']['difference'] == 0.5
    
    def test_compare_error_metrics(self):
        """Test error metrics comparison."""
        engine = FileComparisonEngine(self.output_dir)
        
        sim1_errors = {
            'total_error_counts': {
                'error': 5,
                'warning': 10
            }
        }
        
        sim2_errors = {
            'total_error_counts': {
                'error': 3,
                'warning': 15
            }
        }
        
        result = engine._compare_error_metrics(sim1_errors, sim2_errors)
        
        assert 'error' in result
        assert 'warning' in result
        
        assert result['error']['sim1_count'] == 5
        assert result['error']['sim2_count'] == 3
        assert result['error']['difference'] == -2
    
    def test_compare_metric_values(self):
        """Test metric values comparison."""
        engine = FileComparisonEngine(self.output_dir)
        
        metrics1 = {"metric1": 100, "metric2": 200, "metric3": "text"}
        metrics2 = {"metric1": 150, "metric2": 180, "metric3": "different"}
        
        result = engine._compare_metric_values(metrics1, metrics2)
        
        assert 'metric1' in result
        assert 'metric2' in result
        assert 'metric3' in result
        
        assert result['metric1']['sim1_value'] == 100
        assert result['metric1']['sim2_value'] == 150
        assert result['metric1']['difference'] == 50
        
        assert result['metric3']['type'] == 'non_numeric'
    
    def test_compare_metric_performance(self):
        """Test metric performance comparison."""
        engine = FileComparisonEngine(self.output_dir)
        
        metrics1 = {
            "execution_time": 1.0,
            "duration": 2.0,
            "throughput": 100,
            "other_metric": 50
        }
        
        metrics2 = {
            "execution_time": 2.0,
            "duration": 1.0,
            "throughput": 200,
            "other_metric": 60
        }
        
        result = engine._compare_metric_performance(metrics1, metrics2)
        
        assert 'execution_time' in result
        assert 'duration' in result
        assert 'throughput' in result
        assert 'other_metric' not in result  # Not performance-related
        
        assert result['execution_time']['ratio'] == 2.0
        assert result['execution_time']['faster'] == 'sim2'
        assert result['duration']['ratio'] == 0.5
        assert result['duration']['faster'] == 'sim1'
    
    def test_create_comparison_summary(self):
        """Test comparison summary creation."""
        engine = FileComparisonEngine(self.output_dir)
        
        config_comp = ConfigComparisonResult()
        config_comp.differences = {"added": [1, 2], "removed": [3]}
        
        database_comp = DatabaseComparisonResult()
        database_comp.schema_differences = {"table1": "diff1"}
        database_comp.data_differences = {"table2": "diff2"}
        
        log_comp = LogComparisonResult()
        log_comp.performance_differences = {"perf1": "diff1"}
        log_comp.error_differences = {"error1": "diff1"}
        
        metrics_comp = MetricsComparisonResult()
        metrics_comp.metric_differences = {"metric1": "diff1", "metric2": "diff2"}
        
        summary = engine._create_comparison_summary(
            config_comp, database_comp, log_comp, metrics_comp
        )
        
        assert isinstance(summary, ComparisonSummary)
        assert summary.config_differences == 3  # 2 added + 1 removed
        assert summary.database_differences == 2  # 1 schema + 1 data
        assert summary.log_differences == 2  # 1 perf + 1 error
        assert summary.metrics_differences == 2  # 2 metrics
        assert summary.total_differences == 9
    
    def test_create_metadata(self):
        """Test metadata creation."""
        engine = FileComparisonEngine(self.output_dir)
        
        mock_sim1_data = MagicMock()
        mock_sim1_data.metadata = {"sim1": "data1"}
        
        mock_sim2_data = MagicMock()
        mock_sim2_data.metadata = {"sim2": "data2"}
        
        metadata = engine._create_metadata(mock_sim1_data, mock_sim2_data)
        
        assert 'sim1_metadata' in metadata
        assert 'sim2_metadata' in metadata
        assert 'comparison_engine' in metadata
        assert 'output_directory' in metadata
        
        assert metadata['sim1_metadata'] == {"sim1": "data1"}
        assert metadata['sim2_metadata'] == {"sim2": "data2"}
        assert metadata['comparison_engine'] == 'FileComparisonEngine'