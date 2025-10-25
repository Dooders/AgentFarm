"""
Tests for MetricsLoader class.
"""

import tempfile
import sqlite3
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from farm.analysis.comparative.metrics_loader import MetricsLoader, MetricsData


class TestMetricsLoader:
    """Test cases for MetricsLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.simulation_dir = self.temp_dir / "test_simulation"
        self.simulation_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_database(self):
        """Create a test database."""
        db_path = self.simulation_dir / "simulation.db"
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
                VALUES ('test_simulation', 1000, 100)
            """)
            conn.commit()
        return db_path
    
    def create_test_config(self):
        """Create a test configuration file."""
        config_path = self.simulation_dir / "config.json"
        config = {
            "simulation": {
                "name": "test_simulation",
                "duration": 1000,
                "agents": 100
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        return config_path
    
    def test_init_valid_directory(self):
        """Test initialization with valid simulation directory."""
        self.create_test_database()
        self.create_test_config()
        
        loader = MetricsLoader(self.simulation_dir)
        
        assert loader.simulation_path == self.simulation_dir
        assert loader.analysis_service is not None
    
    def test_init_nonexistent_directory(self):
        """Test initialization with non-existent directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            MetricsLoader(nonexistent_dir)
    
    def test_init_file_not_directory(self):
        """Test initialization with file instead of directory."""
        file_path = self.temp_dir / "not_a_directory.txt"
        file_path.write_text("test")
        
        with pytest.raises(NotADirectoryError):
            MetricsLoader(file_path)
    
    @patch('farm.analysis.comparative.metrics_loader.AnalysisService')
    def test_load_comprehensive_metrics_success(self, mock_analysis_service):
        """Test successful comprehensive metrics loading."""
        self.create_test_database()
        self.create_test_config()
        
        # Mock analysis service
        mock_service = MagicMock()
        mock_analysis_service.return_value = mock_service
        
        # Mock analysis request and result
        mock_request = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {'test_data': 123}
        mock_result.metrics = {'test_metric': 456}
        mock_result.metadata = {'test_meta': 'value'}
        
        mock_service.run_analysis.return_value = mock_result
        
        loader = MetricsLoader(self.simulation_dir)
        result = loader.load_comprehensive_metrics(['test_module'])
        
        assert isinstance(result, MetricsData)
        assert result.simulation_path == self.simulation_dir
        assert 'test_module.test_metric' in result.metrics
        assert result.metrics['test_module.test_metric'] == 456
        assert 'test_module' in result.analysis_results
    
    @patch('farm.analysis.comparative.metrics_loader.AnalysisService')
    def test_load_comprehensive_metrics_failure(self, mock_analysis_service):
        """Test comprehensive metrics loading with analysis service failure."""
        self.create_test_database()
        self.create_test_config()
        
        # Mock analysis service to raise exception
        mock_service = MagicMock()
        mock_analysis_service.return_value = mock_service
        mock_service.run_analysis.side_effect = Exception("Analysis failed")
        
        loader = MetricsLoader(self.simulation_dir)
        result = loader.load_comprehensive_metrics(['test_module'])
        
        assert isinstance(result, MetricsData)
        assert 'test_module' in result.analysis_results
        assert 'error' in result.analysis_results['test_module']
    
    def test_load_basic_metrics(self):
        """Test basic metrics loading."""
        self.create_test_database()
        self.create_test_config()
        
        loader = MetricsLoader(self.simulation_dir)
        result = loader.load_basic_metrics()
        
        assert isinstance(result, MetricsData)
        assert result.simulation_path == self.simulation_dir
        assert 'database.table_count' in result.metrics
        assert 'database.total_rows' in result.metrics
        assert result.metadata['metrics_type'] == 'basic'
    
    def test_get_available_analysis_modules(self):
        """Test getting available analysis modules."""
        loader = MetricsLoader(self.simulation_dir)
        modules = loader._get_available_analysis_modules()
        
        assert isinstance(modules, list)
        assert len(modules) > 0
        # Should include some common modules
        assert any(module in modules for module in ['actions', 'agents', 'temporal'])
    
    @patch('farm.analysis.comparative.metrics_loader.AnalysisService')
    def test_run_analysis_module_success(self, mock_analysis_service):
        """Test running a single analysis module successfully."""
        self.create_test_database()
        self.create_test_config()
        
        # Mock analysis service
        mock_service = MagicMock()
        mock_analysis_service.return_value = mock_service
        
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {'data': 123}
        mock_result.metrics = {'metric': 456}
        mock_result.metadata = {'meta': 'value'}
        
        mock_service.run_analysis.return_value = mock_result
        
        loader = MetricsLoader(self.simulation_dir)
        result = loader._run_analysis_module('test_module')
        
        assert result is not None
        assert result['success'] is True
        assert result['data'] == {'data': 123}
        assert result['metrics'] == {'metric': 456}
        assert result['metadata'] == {'meta': 'value'}
    
    @patch('farm.analysis.comparative.metrics_loader.AnalysisService')
    def test_run_analysis_module_failure(self, mock_analysis_service):
        """Test running analysis module with failure."""
        self.create_test_database()
        self.create_test_config()
        
        # Mock analysis service to raise exception
        mock_service = MagicMock()
        mock_analysis_service.return_value = mock_service
        mock_service.run_analysis.side_effect = Exception("Analysis failed")
        
        loader = MetricsLoader(self.simulation_dir)
        result = loader._run_analysis_module('test_module')
        
        assert result is None
    
    def test_extract_metrics_from_results(self):
        """Test extracting metrics from analysis results."""
        loader = MetricsLoader(self.simulation_dir)
        
        analysis_results = {
            'module1': {
                'success': True,
                'metrics': {'metric1': 123, 'metric2': 456},
                'data': {'data1': 789}
            },
            'module2': {
                'success': True,
                'metrics': {'metric3': 321},
                'data': {'data2': 654}
            }
        }
        
        metrics = loader._extract_metrics_from_results(analysis_results)
        
        assert 'module1.metric1' in metrics
        assert 'module1.metric2' in metrics
        assert 'module2.metric3' in metrics
        assert 'module1.data.data1' in metrics
        assert 'module2.data.data2' in metrics
        
        assert metrics['module1.metric1'] == 123
        assert metrics['module1.metric2'] == 456
        assert metrics['module2.metric3'] == 321
        assert metrics['module1.data.data1'] == 789
        assert metrics['module2.data.data2'] == 654
    
    def test_create_metrics_metadata(self):
        """Test creating metrics metadata."""
        loader = MetricsLoader(self.simulation_dir)
        
        analysis_modules = ['module1', 'module2', 'module3']
        analysis_results = {
            'module1': {'success': True, 'metrics': {'m1': 1}},
            'module2': {'success': False, 'error': 'Failed'},
            'module3': {'success': True, 'metrics': {'m3': 3}}
        }
        
        metadata = loader._create_metrics_metadata(analysis_modules, analysis_results)
        
        assert metadata['simulation_path'] == str(self.simulation_dir)
        assert metadata['requested_modules'] == analysis_modules
        assert metadata['successful_modules'] == ['module1', 'module3']
        assert metadata['failed_modules'] == ['module2']
        assert metadata['total_metrics'] == 2  # m1 and m3
        assert metadata['modules_run'] == 3
    
    def test_extract_basic_simulation_metrics(self):
        """Test extracting basic simulation metrics."""
        self.create_test_database()
        
        loader = MetricsLoader(self.simulation_dir)
        metrics = loader._extract_basic_simulation_metrics()
        
        assert 'database.table_count' in metrics
        assert 'database.total_rows' in metrics
        assert 'database.simulation_metadata.row_count' in metrics
        
        assert metrics['database.table_count'] == 1
        assert metrics['database.total_rows'] == 1
        assert metrics['database.simulation_metadata.row_count'] == 1
    
    def test_extract_database_metrics(self):
        """Test extracting database metrics."""
        self.create_test_database()
        
        loader = MetricsLoader(self.simulation_dir)
        metrics = loader._extract_database_metrics()
        
        assert 'database.simulation_metadata.count' in metrics
        assert 'database.simulation_metadata.duration.avg' in metrics
        assert 'database.simulation_metadata.agent_count.avg' in metrics
        
        assert metrics['database.simulation_metadata.count'] == 1
        assert metrics['database.simulation_metadata.duration.avg'] == 1000
        assert metrics['database.simulation_metadata.agent_count.avg'] == 100
    
    def test_extract_file_metrics(self):
        """Test extracting file metrics."""
        self.create_test_database()
        self.create_test_config()
        
        # Create additional files
        (self.simulation_dir / "test.log").write_text("log content")
        (self.simulation_dir / "data.csv").write_text("csv content")
        
        loader = MetricsLoader(self.simulation_dir)
        metrics = loader._extract_file_metrics()
        
        assert 'files.simulation.db.size_mb' in metrics
        assert 'files.config.json.size_mb' in metrics
        assert 'files.test.log.size_mb' in metrics
        assert 'files.data.csv.size_mb' in metrics
        
        assert 'files.extension.db.count' in metrics
        assert 'files.extension.json.count' in metrics
        assert 'files.extension.log.count' in metrics
        assert 'files.extension.csv.count' in metrics
        
        assert metrics['files.extension.db.count'] == 1
        assert metrics['files.extension.json.count'] == 1
        assert metrics['files.extension.log.count'] == 1
        assert metrics['files.extension.csv.count'] == 1
    
    def test_load_basic_metrics_no_database(self):
        """Test basic metrics loading without database."""
        self.create_test_config()
        
        loader = MetricsLoader(self.simulation_dir)
        result = loader.load_basic_metrics()
        
        assert isinstance(result, MetricsData)
        assert 'database.table_count' not in result.metrics
        assert 'files.config.json.size_mb' in result.metrics
    
    def test_load_basic_metrics_no_files(self):
        """Test basic metrics loading with no files."""
        loader = MetricsLoader(self.simulation_dir)
        result = loader.load_basic_metrics()
        
        assert isinstance(result, MetricsData)
        assert len(result.metrics) == 0
        assert result.metadata['total_metrics'] == 0