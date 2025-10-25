"""
Tests for ReportGenerator.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime
import json
import yaml

from farm.analysis.comparative.report_generator import ReportGenerator
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, ConfigComparisonResult,
    DatabaseComparisonResult, LogComparisonResult, MetricsComparisonResult
)
from farm.analysis.comparative.statistical_analyzer import StatisticalAnalysisResult


class TestReportGenerator:
    """Test cases for ReportGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "reports"
        self.generator = ReportGenerator(output_dir=self.output_dir)
        
        # Create mock comparison result
        self.mock_result = self._create_mock_comparison_result()
        self.mock_statistical_analysis = self._create_mock_statistical_analysis()
    
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
            comparison_time="2024-01-01T00:00:00"
        )
        
        config_comparison = ConfigComparisonResult(
            differences={"param1": "value1", "param2": ["item1", "item2"]}
        )
        
        database_comparison = DatabaseComparisonResult(
            schema_differences={"table1": {"column1": "type_change"}},
            data_differences={"table1": {"difference": 5}},
            metric_differences={"db_metric1": {"db1_value": 10, "db2_value": 15, "difference": 5}}
        )
        
        log_comparison = LogComparisonResult(
            performance_differences={"cpu_usage": {"sim1_value": 50, "sim2_value": 60, "difference": 10}},
            error_differences={"error1": {"sim1_count": 2, "sim2_count": 5, "difference": 3}}
        )
        
        metrics_comparison = MetricsComparisonResult(
            metric_differences={"throughput": {"sim1_value": 100, "sim2_value": 120, "difference": 20, "percentage_change": 20.0}},
            performance_comparison={"response_time": {"ratio": 0.8, "faster": "sim2"}}
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
    
    def _create_mock_statistical_analysis(self):
        """Create a mock statistical analysis result for testing."""
        return StatisticalAnalysisResult(
            correlation_analysis={"test_correlation": {"correlation": 0.8, "significant": True}},
            significance_tests={"test_significance": {"significant": True, "p_value": 0.01}},
            trend_analysis={"overall_trend": {"trend": "positive", "strength": 0.7}},
            anomaly_detection={"summary": {"total_anomalies": 2, "high_severity": 1}},
            summary={"analysis_quality": "High quality", "significant_correlations": 1}
        )
    
    def test_init(self):
        """Test ReportGenerator initialization."""
        assert self.generator.output_dir == self.output_dir
        assert self.output_dir.exists()
    
    def test_init_with_template_dir(self):
        """Test ReportGenerator initialization with template directory."""
        template_dir = Path(self.temp_dir) / "templates"
        generator = ReportGenerator(output_dir=self.output_dir, template_dir=template_dir)
        
        assert generator.output_dir == self.output_dir
        assert generator.template_dir == template_dir
    
    def test_generate_comprehensive_report_html(self):
        """Test generating comprehensive HTML report."""
        report_files = self.generator.generate_comprehensive_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis,
            report_format="html"
        )
        
        assert isinstance(report_files, dict)
        assert 'html' in report_files
        assert Path(report_files['html']).exists()
        assert report_files['html'].endswith('.html')
    
    def test_generate_comprehensive_report_text(self):
        """Test generating comprehensive text report."""
        report_files = self.generator.generate_comprehensive_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis,
            report_format="text"
        )
        
        assert isinstance(report_files, dict)
        assert 'text' in report_files
        assert Path(report_files['text']).exists()
        assert report_files['text'].endswith('.txt')
    
    def test_generate_comprehensive_report_json(self):
        """Test generating comprehensive JSON report."""
        report_files = self.generator.generate_comprehensive_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis,
            report_format="json"
        )
        
        assert isinstance(report_files, dict)
        assert 'json' in report_files
        assert Path(report_files['json']).exists()
        assert report_files['json'].endswith('.json')
        
        # Verify JSON content
        with open(report_files['json'], 'r') as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert 'comparison_result' in data
        assert 'statistical_analysis' in data
    
    def test_generate_comprehensive_report_yaml(self):
        """Test generating comprehensive YAML report."""
        report_files = self.generator.generate_comprehensive_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis,
            report_format="yaml"
        )
        
        assert isinstance(report_files, dict)
        assert 'yaml' in report_files
        assert Path(report_files['yaml']).exists()
        assert report_files['yaml'].endswith('.yaml')
        
        # Verify YAML content
        with open(report_files['yaml'], 'r') as f:
            data = yaml.safe_load(f)
        
        assert 'metadata' in data
        assert 'comparison_result' in data
        assert 'statistical_analysis' in data
    
    def test_generate_comprehensive_report_all_formats(self):
        """Test generating all report formats."""
        report_files = self.generator.generate_comprehensive_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis,
            report_format="all"
        )
        
        assert isinstance(report_files, dict)
        assert 'html' in report_files
        assert 'text' in report_files
        assert 'json' in report_files
        assert 'yaml' in report_files
        
        # Check that all files exist
        for format_type, file_path in report_files.items():
            assert Path(file_path).exists()
    
    def test_generate_html_report(self):
        """Test generating HTML report."""
        file_path = self.generator.generate_html_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.html')
        
        # Verify HTML content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '<html' in content
        assert 'Simulation Comparison Report' in content
        assert 'sim1' in content
        assert 'sim2' in content
    
    def test_generate_text_report(self):
        """Test generating text report."""
        file_path = self.generator.generate_text_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.txt')
        
        # Verify text content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'SIMULATION COMPARISON REPORT' in content
        assert 'sim1' in content
        assert 'sim2' in content
        assert 'EXECUTIVE SUMMARY' in content
    
    def test_generate_json_report(self):
        """Test generating JSON report."""
        file_path = self.generator.generate_json_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.json')
        
        # Verify JSON content
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert 'comparison_result' in data
        assert 'statistical_analysis' in data
        assert data['metadata']['report_type'] == 'simulation_comparison'
    
    def test_generate_yaml_report(self):
        """Test generating YAML report."""
        file_path = self.generator.generate_yaml_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        assert file_path.endswith('.yaml')
        
        # Verify YAML content
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert 'metadata' in data
        assert 'comparison_result' in data
        assert 'statistical_analysis' in data
        assert data['metadata']['report_type'] == 'simulation_comparison'
    
    @patch('importlib.import_module')
    def test_generate_pdf_report(self, mock_import):
        """Test generating PDF report."""
        # Mock reportlab imports
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_spacer = Mock()
        mock_table = Mock()
        mock_image = Mock()
        mock_styles = Mock()
        
        mock_import.side_effect = [
            Mock(SimpleDocTemplate=mock_doc, Paragraph=mock_paragraph, Spacer=mock_spacer, Table=mock_table, Image=mock_image),
            Mock(getSampleStyleSheet=lambda: mock_styles),
            Mock(letter=Mock(), A4=Mock()),
            Mock(colors=Mock())
        ]
        
        file_path = self.generator.generate_pdf_report(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(file_path, str)
        # Should fall back to text report when reportlab is not available
        assert file_path.endswith('.txt')
    
    def test_generate_pdf_report_fallback(self):
        """Test PDF report generation fallback when reportlab is not available."""
        with patch('importlib.import_module', side_effect=ImportError):
            file_path = self.generator.generate_pdf_report(
                result=self.mock_result,
                statistical_analysis=self.mock_statistical_analysis
            )
            
            # Should fall back to text report
            assert isinstance(file_path, str)
            assert file_path.endswith('.txt')
    
    def test_create_html_content(self):
        """Test HTML content creation."""
        content = self.generator._create_html_content(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(content, str)
        assert '<html' in content
        assert 'Simulation Comparison Report' in content
        assert 'sim1' in content
        assert 'sim2' in content
        assert 'Executive Summary' in content
    
    def test_create_text_content(self):
        """Test text content creation."""
        content = self.generator._create_text_content(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(content, str)
        assert 'SIMULATION COMPARISON REPORT' in content
        assert 'sim1' in content
        assert 'sim2' in content
        assert 'EXECUTIVE SUMMARY' in content
    
    def test_create_summary_text(self):
        """Test summary text creation."""
        summary_text = self.generator._create_summary_text(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(summary_text, str)
        assert 'sim1' in summary_text
        assert 'sim2' in summary_text
        assert '10' in summary_text  # total_differences
        assert 'MEDIUM' in summary_text  # severity
    
    def test_create_config_section_text(self):
        """Test configuration section text creation."""
        config_text = self.generator._create_config_section_text(self.mock_result)
        
        assert isinstance(config_text, str)
        assert 'CONFIGURATION COMPARISON' in config_text
        assert 'param1' in config_text
        assert 'param2' in config_text
    
    def test_create_database_section_text(self):
        """Test database section text creation."""
        db_text = self.generator._create_database_section_text(self.mock_result)
        
        assert isinstance(db_text, str)
        assert 'DATABASE COMPARISON' in db_text
        assert 'table1' in db_text
        assert 'db_metric1' in db_text
    
    def test_create_log_section_text(self):
        """Test log section text creation."""
        log_text = self.generator._create_log_section_text(self.mock_result)
        
        assert isinstance(log_text, str)
        assert 'LOG ANALYSIS' in log_text
        assert 'cpu_usage' in log_text
        assert 'error1' in log_text
    
    def test_create_metrics_section_text(self):
        """Test metrics section text creation."""
        metrics_text = self.generator._create_metrics_section_text(self.mock_result)
        
        assert isinstance(metrics_text, str)
        assert 'METRICS COMPARISON' in metrics_text
        assert 'throughput' in metrics_text
        assert 'response_time' in metrics_text
    
    def test_create_statistical_section_text(self):
        """Test statistical section text creation."""
        stats_text = self.generator._create_statistical_section_text(self.mock_statistical_analysis)
        
        assert isinstance(stats_text, str)
        assert 'STATISTICAL ANALYSIS' in stats_text
        assert 'High quality' in stats_text
    
    def test_create_html_sections(self):
        """Test HTML sections creation."""
        sections = self.generator._create_html_sections(
            result=self.mock_result,
            statistical_analysis=self.mock_statistical_analysis
        )
        
        assert isinstance(sections, str)
        assert 'Configuration Comparison' in sections
        assert 'Database Comparison' in sections
        assert 'Log Analysis' in sections
        assert 'Metrics Comparison' in sections
        assert 'Statistical Analysis' in sections
    
    def test_create_differences_html(self):
        """Test differences HTML creation."""
        differences = {"key1": "value1", "key2": ["item1", "item2"]}
        html = self.generator._create_differences_html(differences, "Test")
        
        assert isinstance(html, str)
        assert 'key1' in html
        assert 'key2' in html
        assert 'differences-list' in html
    
    def test_create_differences_html_empty(self):
        """Test differences HTML creation with empty differences."""
        differences = {}
        html = self.generator._create_differences_html(differences, "Test")
        
        assert isinstance(html, str)
        assert 'No test differences found' in html
        assert 'no-differences' in html
    
    def test_create_correlation_html(self):
        """Test correlation HTML creation."""
        correlation_analysis = {
            "test_correlation": {
                "label": "Test Correlation",
                "correlation": 0.8,
                "p_value": 0.01,
                "significant": True
            }
        }
        html = self.generator._create_correlation_html(correlation_analysis)
        
        assert isinstance(html, str)
        assert 'Test Correlation' in html
        assert '0.8' in html
        assert 'Significant' in html
    
    def test_create_significance_html(self):
        """Test significance HTML creation."""
        significance_tests = {
            "test_significance": {
                "test_name": "Test Significance",
                "significant": True,
                "p_value": 0.01
            }
        }
        html = self.generator._create_significance_html(significance_tests)
        
        assert isinstance(html, str)
        assert 'Test Significance' in html
        assert 'Significant' in html
    
    def test_create_trend_html(self):
        """Test trend HTML creation."""
        trend_analysis = {
            "overall_trend": {
                "trend": "positive",
                "trend_strength": 0.7,
                "interpretation": "Strong positive trend"
            }
        }
        html = self.generator._create_trend_html(trend_analysis)
        
        assert isinstance(html, str)
        assert 'positive' in html
        assert '0.7' in html
        assert 'Strong positive trend' in html
    
    def test_create_anomaly_html(self):
        """Test anomaly HTML creation."""
        anomaly_detection = {
            "summary": {
                "total_anomalies": 5,
                "high_severity": 2,
                "medium_severity": 2,
                "low_severity": 1
            }
        }
        html = self.generator._create_anomaly_html(anomaly_detection)
        
        assert isinstance(html, str)
        assert '5' in html
        assert '2' in html
        assert '1' in html
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_result = SimulationComparisonResult(
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
        
        # Should not raise exceptions
        report_files = self.generator.generate_comprehensive_report(
            result=empty_result,
            report_format="text"
        )
        
        assert isinstance(report_files, dict)
        assert 'text' in report_files
        assert Path(report_files['text']).exists()
    
    def test_no_statistical_analysis(self):
        """Test report generation without statistical analysis."""
        report_files = self.generator.generate_comprehensive_report(
            result=self.mock_result,
            statistical_analysis=None,
            report_format="html"
        )
        
        assert isinstance(report_files, dict)
        assert 'html' in report_files
        assert Path(report_files['html']).exists()
    
    def test_file_naming_with_timestamps(self):
        """Test that generated files have timestamp-based names."""
        report_files = self.generator.generate_comprehensive_report(
            result=self.mock_result,
            report_format="text"
        )
        
        file_path = report_files['text']
        # Should contain timestamp pattern
        assert '2024' in file_path or '2025' in file_path  # Current year
        assert file_path.endswith('.txt')
    
    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        new_output_dir = Path(self.temp_dir) / "new_reports_dir"
        generator = ReportGenerator(output_dir=new_output_dir)
        
        assert new_output_dir.exists()
        assert generator.output_dir == new_output_dir