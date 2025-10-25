"""
Tests for VisualizationEngine.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from farm.analysis.comparative.visualization_engine import VisualizationEngine
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, ConfigComparisonResult,
    DatabaseComparisonResult, LogComparisonResult, MetricsComparisonResult
)


class TestVisualizationEngine:
    """Test cases for VisualizationEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "visualizations"
        self.engine = VisualizationEngine(output_dir=self.output_dir)
        
        # Create mock comparison result
        self.mock_result = self._create_mock_comparison_result()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_comparison_result(self):
        """Create a mock comparison result for testing."""
        summary = ComparisonSummary(
            total_differences=10,
            severity="medium",
            config_differences=2,
            database_differences=3,
            log_differences=2,
            metrics_differences=3,
            comparison_time=datetime(2024, 1, 1, 0, 0, 0)
        )
        
        config_comparison = ConfigComparisonResult(
            differences={"param1": "value1", "param2": ["item1", "item2"]}
        )
        
        database_comparison = DatabaseComparisonResult(
            schema_differences={"table1": {"column1": "type_change"}},
            data_differences={"table1": {"difference": 5}},
            metric_differences={"metric1": {"db1_value": 10, "db2_value": 15, "difference": 5}}
        )
        
        log_comparison = LogComparisonResult(
            performance_differences={"cpu_usage": {"sim1_value": 50, "sim2_value": 60, "difference": 10}},
            error_differences={"error1": {"sim1_count": 2, "sim2_count": 5, "difference": 3}}
        )
        
        metrics_comparison = MetricsComparisonResult(
            metric_differences={
                "throughput": {"sim1_value": 100, "sim2_value": 120, "difference": 20, "percentage_change": 20.0}
            },
            performance_comparison={"latency": {"ratio": 0.8, "faster": "sim2"}}
        )
        
        return SimulationComparisonResult(
            simulation1_path=Path("sim1"),
            simulation2_path=Path("sim2"),
            comparison_summary=summary,
            config_comparison=config_comparison,
            database_comparison=database_comparison,
            log_comparison=log_comparison,
            metrics_comparison=metrics_comparison,
            metadata={"test": True}
        )
    
    def test_init(self):
        """Test VisualizationEngine initialization."""
        assert self.engine.output_dir == self.output_dir
        assert self.output_dir.exists()
    
    def test_init_custom_style(self):
        """Test VisualizationEngine initialization with custom style."""
        # Use a valid matplotlib style
        engine = VisualizationEngine(output_dir=self.output_dir, style="ggplot")
        assert engine.output_dir == self.output_dir
    
    def test_create_comparison_dashboard(self):
        """Test creating comparison dashboard."""
        dashboard_files = self.engine.create_comparison_dashboard(self.mock_result)
        
        assert isinstance(dashboard_files, dict)
        assert 'summary_chart' in dashboard_files
        assert 'metrics_comparison' in dashboard_files
        assert 'database_analysis' in dashboard_files
        assert 'performance_comparison' in dashboard_files
        assert 'error_analysis' in dashboard_files
        assert 'heatmap' in dashboard_files
        assert 'dashboard' in dashboard_files
        
        # Check that files were created
        for name, file_path in dashboard_files.items():
            assert Path(file_path).exists()
    
    def test_create_summary_chart(self):
        """Test creating summary chart."""
        file_path = self.engine.create_summary_chart(self.mock_result)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.png')
    
    def test_create_summary_chart_no_differences(self):
        """Test creating summary chart with no differences."""
        # Create result with no differences
        summary = ComparisonSummary(
            total_differences=0,
            severity="low",
            config_differences=0,
            database_differences=0,
            log_differences=0,
            metrics_differences=0,
            comparison_time="2024-01-01T00:00:00"
        )
        
        result = SimulationComparisonResult(
            simulation1_path=Path("sim1"),
            simulation2_path=Path("sim2"),
            comparison_summary=summary,
            config_comparison=ConfigComparisonResult(differences={}),
            database_comparison=DatabaseComparisonResult(schema_differences={}, data_differences={}, metric_differences={}),
            log_comparison=LogComparisonResult(performance_differences={}, error_differences={}),
            metrics_comparison=MetricsComparisonResult(metric_differences={}, performance_comparison={}),
            metadata={}
        )
        
        file_path = self.engine.create_summary_chart(result)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
    
    def test_create_metrics_comparison_chart(self):
        """Test creating metrics comparison chart."""
        file_path = self.engine.create_metrics_comparison_chart(self.mock_result)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.png')
    
    def test_create_metrics_comparison_chart_no_metrics(self):
        """Test creating metrics comparison chart with no metrics."""
        result = SimulationComparisonResult(
            simulation1_path=Path("sim1"),
            simulation2_path=Path("sim2"),
            comparison_summary=ComparisonSummary(
                total_differences=0,
                severity="low",
                config_differences=0,
                database_differences=0,
                log_differences=0,
                metrics_differences=0,
                comparison_time=datetime(2024, 1, 1, 0, 0, 0)
            ),
            config_comparison=ConfigComparisonResult(differences={}),
            database_comparison=DatabaseComparisonResult(schema_differences={}, data_differences={}, metric_differences={}),
            log_comparison=LogComparisonResult(performance_differences={}, error_differences={}),
            metrics_comparison=MetricsComparisonResult(metric_differences={}, performance_comparison={}),
            metadata={}
        )
        
        file_path = self.engine.create_metrics_comparison_chart(result)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
    
    def test_create_database_analysis_chart(self):
        """Test creating database analysis chart."""
        file_path = self.engine.create_database_analysis_chart(self.mock_result)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.png')
    
    def test_create_performance_comparison_chart(self):
        """Test creating performance comparison chart."""
        file_path = self.engine.create_performance_comparison_chart(self.mock_result)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.png')
    
    def test_create_error_analysis_chart(self):
        """Test creating error analysis chart."""
        file_path = self.engine.create_error_analysis_chart(self.mock_result)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.png')
    
    def test_create_differences_heatmap(self):
        """Test creating differences heatmap."""
        file_path = self.engine.create_differences_heatmap(self.mock_result)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.png')
    
    def test_create_combined_dashboard(self):
        """Test creating combined dashboard."""
        chart_files = {
            'summary_chart': 'summary.png',
            'metrics_comparison': 'metrics.png',
            'database_analysis': 'database.png',
            'performance_comparison': 'performance.png',
            'error_analysis': 'error.png',
            'heatmap': 'heatmap.png'
        }
        
        file_path = self.engine.create_combined_dashboard(self.mock_result, chart_files)
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.png')
    
    @patch('importlib.import_module')
    def test_create_interactive_plot(self, mock_import):
        """Test creating interactive plot."""
        # Mock plotly imports
        mock_go = Mock()
        mock_pyo = Mock()
        mock_subplots = Mock()
        
        mock_import.side_effect = [
            Mock(go=mock_go, offline=mock_pyo),
            Mock(make_subplots=mock_subplots)
        ]
        
        file_path = self.engine.create_interactive_plot(self.mock_result)
        
        # Should return empty string when plotly is not available
        assert file_path == ""
    
    def test_export_data_for_external_tools(self):
        """Test exporting data for external tools."""
        export_files = self.engine.export_data_for_external_tools(self.mock_result)
        
        assert isinstance(export_files, dict)
        assert 'summary_json' in export_files
        
        # Check that files were created
        for name, file_path in export_files.items():
            assert Path(file_path).exists()
    
    def test_export_data_no_metrics(self):
        """Test exporting data with no metrics."""
        result = SimulationComparisonResult(
            simulation1_path=Path("sim1"),
            simulation2_path=Path("sim2"),
            comparison_summary=ComparisonSummary(
                total_differences=0,
                severity="low",
                config_differences=0,
                database_differences=0,
                log_differences=0,
                metrics_differences=0,
                comparison_time=datetime(2024, 1, 1, 0, 0, 0)
            ),
            config_comparison=ConfigComparisonResult(differences={}),
            database_comparison=DatabaseComparisonResult(schema_differences={}, data_differences={}, metric_differences={}),
            log_comparison=LogComparisonResult(performance_differences={}, error_differences={}),
            metrics_comparison=MetricsComparisonResult(metric_differences={}, performance_comparison={}),
            metadata={}
        )
        
        export_files = self.engine.export_data_for_external_tools(result)
        
        assert isinstance(export_files, dict)
        assert 'summary_json' in export_files
        # Should not have metrics_csv when no metrics
        assert 'metrics_csv' not in export_files
    
    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        new_output_dir = Path(self.temp_dir) / "new_viz_dir"
        engine = VisualizationEngine(output_dir=new_output_dir)
        
        assert new_output_dir.exists()
        assert engine.output_dir == new_output_dir
    
    def test_matplotlib_configuration(self):
        """Test that matplotlib is properly configured."""
        # Check that matplotlib settings are applied
        import matplotlib.pyplot as plt
        
        # The engine should have set some rcParams
        assert plt.rcParams['figure.dpi'] == 300
        assert plt.rcParams['savefig.dpi'] == 300
    
    def test_file_naming_with_timestamps(self):
        """Test that generated files have timestamp-based names."""
        file_path = self.engine.create_summary_chart(self.mock_result)
        
        # Should contain timestamp pattern
        assert '2024' in file_path or '2025' in file_path  # Current year
        assert file_path.endswith('.png')
    
    def test_error_handling_in_chart_creation(self):
        """Test error handling in chart creation methods."""
        # Test with invalid data that might cause errors
        result = SimulationComparisonResult(
            simulation1_path=Path("sim1"),
            simulation2_path=Path("sim2"),
            comparison_summary=ComparisonSummary(
                total_differences=0,
                severity="low",
                config_differences=0,
                database_differences=0,
                log_differences=0,
                metrics_differences=0,
                comparison_time=datetime(2024, 1, 1, 0, 0, 0)
            ),
            config_comparison=ConfigComparisonResult(differences={}),
            database_comparison=DatabaseComparisonResult(schema_differences={}, data_differences={}, metric_differences={}),
            log_comparison=LogComparisonResult(performance_differences={}, error_differences={}),
            metrics_comparison=MetricsComparisonResult(metric_differences={}, performance_comparison={}),
            metadata={}
        )
        
        # Should not raise exceptions even with empty data
        file_path = self.engine.create_summary_chart(result)
        assert isinstance(file_path, str)
        assert Path(file_path).exists()