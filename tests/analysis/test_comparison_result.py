"""
Tests for comparison result data structures.
"""

from datetime import datetime
from pathlib import Path

import pytest

from farm.analysis.comparative.comparison_result import (
    ComparisonSummary,
    ConfigComparisonResult,
    DatabaseComparisonResult,
    LogComparisonResult,
    MetricsComparisonResult,
    SimulationComparisonResult
)


class TestComparisonSummary:
    """Test cases for ComparisonSummary."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        summary = ComparisonSummary()
        
        assert summary.total_differences == 0
        assert summary.config_differences == 0
        assert summary.database_differences == 0
        assert summary.log_differences == 0
        assert summary.metrics_differences == 0
        assert summary.severity == "low"
        assert isinstance(summary.comparison_time, datetime)
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        summary = ComparisonSummary(
            config_differences=5,
            database_differences=10,
            log_differences=3,
            metrics_differences=2
        )
        
        assert summary.config_differences == 5
        assert summary.database_differences == 10
        assert summary.log_differences == 3
        assert summary.metrics_differences == 2
        assert summary.total_differences == 20
        assert summary.severity == "medium"
    
    def test_post_init_calculates_total(self):
        """Test that __post_init__ calculates total differences."""
        summary = ComparisonSummary(
            config_differences=1,
            database_differences=2,
            log_differences=3,
            metrics_differences=4
        )
        
        assert summary.total_differences == 10
    
    def test_severity_high(self):
        """Test high severity calculation."""
        summary = ComparisonSummary(
            config_differences=50,
            database_differences=50,
            log_differences=50,
            metrics_differences=50
        )
        
        assert summary.severity == "high"
    
    def test_severity_medium(self):
        """Test medium severity calculation."""
        summary = ComparisonSummary(
            config_differences=5,
            database_differences=5,
            log_differences=5,
            metrics_differences=5
        )
        
        assert summary.severity == "medium"
    
    def test_severity_low(self):
        """Test low severity calculation."""
        summary = ComparisonSummary(
            config_differences=1,
            database_differences=1,
            log_differences=1,
            metrics_differences=1
        )
        
        assert summary.severity == "low"


class TestConfigComparisonResult:
    """Test cases for ConfigComparisonResult."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        result = ConfigComparisonResult()
        
        assert result.differences == {}
        assert result.summary == {}
        assert result.significant_changes == []
        assert result.formatted_output == ""


class TestDatabaseComparisonResult:
    """Test cases for DatabaseComparisonResult."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        result = DatabaseComparisonResult()
        
        assert result.schema_differences == {}
        assert result.data_differences == {}
        assert result.metric_differences == {}
        assert result.summary == {}


class TestLogComparisonResult:
    """Test cases for LogComparisonResult."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        result = LogComparisonResult()
        
        assert result.performance_differences == {}
        assert result.error_differences == {}
        assert result.summary == {}


class TestMetricsComparisonResult:
    """Test cases for MetricsComparisonResult."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        result = MetricsComparisonResult()
        
        assert result.metric_differences == {}
        assert result.performance_comparison == {}
        assert result.summary == {}


class TestSimulationComparisonResult:
    """Test cases for SimulationComparisonResult."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sim1_path = Path("/path/to/sim1")
        self.sim2_path = Path("/path/to/sim2")
        
        self.summary = ComparisonSummary(
            config_differences=5,
            database_differences=10,
            log_differences=3,
            metrics_differences=2
        )
        
        self.config_comp = ConfigComparisonResult()
        self.database_comp = DatabaseComparisonResult()
        self.log_comp = LogComparisonResult()
        self.metrics_comp = MetricsComparisonResult()
    
    def test_init(self):
        """Test initialization."""
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        assert result.simulation1_path == self.sim1_path
        assert result.simulation2_path == self.sim2_path
        assert result.comparison_summary == self.summary
        assert result.config_comparison == self.config_comp
        assert result.database_comparison == self.database_comp
        assert result.log_comparison == self.log_comp
        assert result.metrics_comparison == self.metrics_comp
        assert result.metadata == {}
    
    def test_post_init_updates_summary(self):
        """Test that __post_init__ updates the summary."""
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=ComparisonSummary(),  # Empty summary
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # Summary should be updated by __post_init__
        assert result.comparison_summary.total_differences == 0
    
    def test_count_config_differences(self):
        """Test counting configuration differences."""
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # Test with empty differences
        count = result._count_config_differences()
        assert count == 0
        
        # Test with populated differences
        result.config_comparison.differences = {
            'added': [1, 2, 3],
            'removed': [4, 5],
            'changed': [6, 7, 8, 9]
        }
        count = result._count_config_differences()
        assert count == 9  # 3 + 2 + 4
    
    def test_count_database_differences(self):
        """Test counting database differences."""
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # Test with empty differences
        count = result._count_database_differences()
        assert count == 0
        
        # Test with populated differences
        result.database_comparison.schema_differences = {'table1': 'diff1', 'table2': 'diff2'}
        result.database_comparison.data_differences = {'table1': 'diff1', 'table2': 'diff2', 'table3': 'diff3'}
        result.database_comparison.metric_differences = {'metric1': 'diff1'}
        count = result._count_database_differences()
        assert count == 6  # 2 + 3 + 1
    
    def test_count_log_differences(self):
        """Test counting log differences."""
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # Test with empty differences
        count = result._count_log_differences()
        assert count == 0
        
        # Test with populated differences
        result.log_comparison.performance_differences = {'perf1': 'diff1', 'perf2': 'diff2'}
        result.log_comparison.error_differences = {'error1': 'diff1', 'error2': 'diff2', 'error3': 'diff3'}
        count = result._count_log_differences()
        assert count == 5  # 2 + 3
    
    def test_count_metrics_differences(self):
        """Test counting metrics differences."""
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # Test with empty differences
        count = result._count_metrics_differences()
        assert count == 0
        
        # Test with populated differences
        result.metrics_comparison.metric_differences = {
            'metric1': 'diff1',
            'metric2': 'diff2',
            'metric3': 'diff3'
        }
        count = result._count_metrics_differences()
        assert count == 3
    
    def test_get_summary_text(self):
        """Test getting summary text."""
        # Create a result with some actual differences
        config_comp = ConfigComparisonResult()
        config_comp.differences = {"added": [1, 2, 3, 4, 5]}  # 5 differences
        
        database_comp = DatabaseComparisonResult()
        database_comp.schema_differences = {"table1": "diff1", "table2": "diff2", "table3": "diff3", "table4": "diff4", "table5": "diff5"}  # 5 differences
        database_comp.data_differences = {"table1": "diff1", "table2": "diff2", "table3": "diff3", "table4": "diff4", "table5": "diff5"}  # 5 differences
        
        log_comp = LogComparisonResult()
        log_comp.performance_differences = {"perf1": "diff1", "perf2": "diff2", "perf3": "diff3"}  # 3 differences
        
        metrics_comp = MetricsComparisonResult()
        metrics_comp.metric_differences = {"metric1": "diff1", "metric2": "diff2"}  # 2 differences
        
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=ComparisonSummary(),  # Empty summary
            config_comparison=config_comp,
            database_comparison=database_comp,
            log_comparison=log_comp,
            metrics_comparison=metrics_comp
        )
        
        text = result.get_summary_text()
        
        assert "Simulation Comparison Summary" in text
        assert str(self.sim1_path) in text
        assert str(self.sim2_path) in text
        assert "Configuration: 5" in text
        assert "Database: 10" in text
        assert "Logs: 3" in text
        assert "Metrics: 2" in text
        assert "Total: 20" in text
        assert "Severity: MEDIUM" in text
    
    def test_get_detailed_report(self):
        """Test getting detailed report."""
        # Set up result with some differences
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # Add some differences
        result.config_comparison.differences = {"added": ["item1"]}
        result.config_comparison.formatted_output = "Config differences found"
        
        result.database_comparison.schema_differences = {"table1": ["change1"]}
        result.database_comparison.data_differences = {"table1": {"difference": 5}}
        
        result.log_comparison.performance_differences = {"perf1": "diff1"}
        result.log_comparison.error_differences = {"error1": 5}
        
        result.metrics_comparison.metric_differences = {
            "metric1": {"db1_value": 100, "db2_value": 150}
        }
        
        report = result.get_detailed_report()
        
        assert "Simulation Comparison Summary" in report
        assert "Configuration Differences:" in report
        assert "Database Differences:" in report
        assert "Log Differences:" in report
        assert "Metrics Differences:" in report
        assert "Config differences found" in report
        assert "table1: 1 changes" in report
        assert "table1: 5 row difference" in report
        assert "perf1: diff1" in report
        assert "error1: 5" in report
        assert "metric1: 100 -> 150" in report
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        # Create a result with some actual differences
        config_comp = ConfigComparisonResult()
        config_comp.differences = {"added": [1, 2, 3, 4, 5]}  # 5 differences
        
        database_comp = DatabaseComparisonResult()
        database_comp.schema_differences = {"table1": "diff1", "table2": "diff2", "table3": "diff3", "table4": "diff4", "table5": "diff5"}  # 5 differences
        database_comp.data_differences = {"table1": "diff1", "table2": "diff2", "table3": "diff3", "table4": "diff4", "table5": "diff5"}  # 5 differences
        
        log_comp = LogComparisonResult()
        log_comp.performance_differences = {"perf1": "diff1", "perf2": "diff2", "perf3": "diff3"}  # 3 differences
        
        metrics_comp = MetricsComparisonResult()
        metrics_comp.metric_differences = {"metric1": "diff1", "metric2": "diff2"}  # 2 differences
        
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=ComparisonSummary(),  # Empty summary
            config_comparison=config_comp,
            database_comparison=database_comp,
            log_comparison=log_comp,
            metrics_comparison=metrics_comp,
            metadata={"test": "value"}
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data['simulation1_path'] == str(self.sim1_path)
        assert data['simulation2_path'] == str(self.sim2_path)
        assert 'comparison_summary' in data
        assert 'config_comparison' in data
        assert 'database_comparison' in data
        assert 'log_comparison' in data
        assert 'metrics_comparison' in data
        assert data['metadata'] == {"test": "value"}
        
        # Check summary data
        summary_data = data['comparison_summary']
        assert summary_data['total_differences'] == 20
        assert summary_data['config_differences'] == 5
        assert summary_data['database_differences'] == 10
        assert summary_data['log_differences'] == 3
        assert summary_data['metrics_differences'] == 2
        assert summary_data['severity'] == "medium"
        assert 'comparison_time' in summary_data
    
    def test_save_to_file_json(self):
        """Test saving to JSON file."""
        import tempfile
        import json
        
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result.save_to_file(temp_path, "json")
            
            assert temp_path.exists()
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert data['simulation1_path'] == str(self.sim1_path)
            assert data['simulation2_path'] == str(self.sim2_path)
        
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_save_to_file_txt(self):
        """Test saving to text file."""
        import tempfile
        
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result.save_to_file(temp_path, "txt")
            
            assert temp_path.exists()
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            assert "Simulation Comparison Summary" in content
            assert str(self.sim1_path) in content
            assert str(self.sim2_path) in content
        
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_save_to_file_unsupported_format(self):
        """Test saving with unsupported format."""
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=self.summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        with pytest.raises(ValueError, match="Unsupported format"):
            result.save_to_file("test.xyz", "xyz")
    
    def test_is_significantly_different_high_severity(self):
        """Test significant difference detection with high severity."""
        high_severity_summary = ComparisonSummary(
            config_differences=50,
            database_differences=50,
            log_differences=50,
            metrics_differences=50
        )
        
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=high_severity_summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # The summary should already have high severity due to total_differences > 100
        assert result.comparison_summary.severity == "high"
        assert result.is_significantly_different(0.5) is True
        assert result.is_significantly_different(0.9) is True
    
    def test_is_significantly_different_medium_severity(self):
        """Test significant difference detection with medium severity."""
        medium_severity_summary = ComparisonSummary(
            config_differences=5,
            database_differences=5,
            log_differences=5,
            metrics_differences=5
        )
        
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=medium_severity_summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # The summary should already have medium severity due to total_differences > 10
        assert result.comparison_summary.severity == "medium"
        assert result.is_significantly_different(0.3) is True
        assert result.is_significantly_different(0.7) is False
    
    def test_is_significantly_different_low_severity(self):
        """Test significant difference detection with low severity."""
        low_severity_summary = ComparisonSummary(
            config_differences=1,
            database_differences=1,
            log_differences=1,
            metrics_differences=1
        )
        
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=low_severity_summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        # The summary should already have low severity due to total_differences <= 10
        assert result.comparison_summary.severity == "low"
        assert result.is_significantly_different(0.05) is True
        assert result.is_significantly_different(0.2) is False
    
    def test_is_significantly_different_no_differences(self):
        """Test significant difference detection with no differences."""
        no_differences_summary = ComparisonSummary()
        
        result = SimulationComparisonResult(
            simulation1_path=self.sim1_path,
            simulation2_path=self.sim2_path,
            comparison_summary=no_differences_summary,
            config_comparison=self.config_comp,
            database_comparison=self.database_comp,
            log_comparison=self.log_comp,
            metrics_comparison=self.metrics_comp
        )
        
        assert result.is_significantly_different(0.1) is False
        assert result.is_significantly_different(0.5) is False