"""
Tests for reporting system.

This module contains unit tests for the comprehensive reporting
and documentation system.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
from pathlib import Path

from farm.analysis.comparative.reporting_system import (
    ReportingSystem,
    ReportConfig,
    ReportSection,
    AnalysisReport
)


class TestReportingSystem:
    """Test cases for ReportingSystem."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ReportConfig(
            output_dir="test_reports",
            format="html",
            include_charts=True,
            include_raw_data=False,
            title="Test Report",
            author="Test Author",
            version="1.0.0"
        )
        
        with patch('farm.analysis.comparative.reporting_system.MATPLOTLIB_AVAILABLE', True), \
             patch('farm.analysis.comparative.reporting_system.JINJA2_AVAILABLE', True), \
             patch('farm.analysis.comparative.reporting_system.PLOTLY_AVAILABLE', True):
            
            self.reporting_system = ReportingSystem(self.config)
    
    def test_initialization(self):
        """Test reporting system initialization."""
        assert self.reporting_system.config == self.config
        assert self.reporting_system.output_dir == Path("test_reports")
        assert self.reporting_system.output_dir.exists()
        assert hasattr(self.reporting_system, 'template_env')
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Create mock orchestration result
        orchestration_result = Mock()
        orchestration_result.success = True
        orchestration_result.total_duration = 120.0
        orchestration_result.phase_results = [
            Mock(
                phase_name="basic_analysis",
                success=True,
                duration=30.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error=None
            ),
            Mock(
                phase_name="advanced_analysis",
                success=True,
                duration=60.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error=None
            )
        ]
        orchestration_result.errors = []
        orchestration_result.warnings = []
        orchestration_result.output_paths = {"basic_analysis": "/path/to/basic.json"}
        orchestration_result.summary = {"total_phases": 2}
        
        analysis_data = {
            "ml_results": {"clustering": "completed"},
            "performance_summary": {"total_duration": 120.0}
        }
        
        # Mock the individual section generation methods
        with patch.object(self.reporting_system, '_generate_executive_summary') as mock_exec, \
             patch.object(self.reporting_system, '_generate_analysis_overview') as mock_overview, \
             patch.object(self.reporting_system, '_generate_phase_results_section') as mock_phases, \
             patch.object(self.reporting_system, '_generate_performance_analysis') as mock_perf, \
             patch.object(self.reporting_system, '_generate_detailed_analysis') as mock_detailed, \
             patch.object(self.reporting_system, '_generate_recommendations') as mock_recommendations, \
             patch.object(self.reporting_system, '_generate_appendix') as mock_appendix, \
             patch.object(self.reporting_system, '_export_report') as mock_export:
            
            # Setup mock section returns
            mock_exec.return_value = ReportSection("Executive Summary", "Content")
            mock_overview.return_value = ReportSection("Analysis Overview", "Content")
            mock_phases.return_value = ReportSection("Phase Results", "Content")
            mock_perf.return_value = ReportSection("Performance Analysis", "Content")
            mock_detailed.return_value = ReportSection("Detailed Analysis", "Content")
            mock_recommendations.return_value = ReportSection("Recommendations", "Content")
            mock_appendix.return_value = ReportSection("Appendix", "Content")
            
            # Generate report
            report = self.reporting_system.generate_comprehensive_report(
                orchestration_result, analysis_data
            )
            
            # Verify report structure
            assert isinstance(report, AnalysisReport)
            assert report.title == "Test Report"
            assert report.author == "Test Author"
            assert report.version == "1.0.0"
            assert len(report.sections) == 7  # All sections included
            # AnalysisReport doesn't have a success attribute
            
            # Verify all section generation methods were called
            mock_exec.assert_called_once()
            mock_overview.assert_called_once()
            mock_phases.assert_called_once()
            mock_perf.assert_called_once()
            mock_detailed.assert_called_once()
            mock_recommendations.assert_called_once()
            mock_appendix.assert_called_once()
            mock_export.assert_called_once()
    
    def test_generate_executive_summary(self):
        """Test executive summary generation."""
        summary = {
            "success": True,
            "total_duration": 120.0,
            "phases_completed": 5,
            "errors": [],
            "warnings": ["Warning 1"]
        }
        
        phase_results = [
            Mock(phase_name="phase1", success=True, duration=30.0),
            Mock(phase_name="phase2", success=False, duration=20.0, error="Failed")
        ]
        
        section = self.reporting_system._generate_executive_summary(summary, phase_results)
        
        assert isinstance(section, ReportSection)
        assert section.title == "Executive Summary"
        assert "Successfully completed" in section.content
        assert "120.0" in section.content
        assert "<strong>Phases Completed:</strong> 5" in section.content
        assert "<strong>Errors:</strong> 0" in section.content
        assert "<strong>Warnings:</strong> 1" in section.content
    
    def test_generate_analysis_overview(self):
        """Test analysis overview generation."""
        summary = {
            "success": True,
            "total_duration": 120.0,
            "phases_completed": 3,
            "summary": {
                "performance_metrics": {
                    "total_simulations_analyzed": 10,
                    "analysis_throughput": 0.083
                }
            }
        }
        
        phase_results = [
            Mock(
                phase_name="basic_analysis",
                success=True,
                duration=30.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error=None
            ),
            Mock(
                phase_name="advanced_analysis",
                success=False,
                duration=20.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error="Analysis failed"
            )
        ]
        
        section = self.reporting_system._generate_analysis_overview(summary, phase_results)
        
        assert isinstance(section, ReportSection)
        assert section.title == "Analysis Overview"
        assert "Phase Execution Summary" in section.content
        assert "basic_analysis" in section.content
        assert "advanced_analysis" in section.content
        assert "✅ Success" in section.content
        assert "❌ Failed" in section.content
        assert "Total Simulations Analyzed:" in section.content
    
    def test_generate_phase_results_section(self):
        """Test phase results section generation."""
        phase_results = [
            Mock(
                phase_name="phase1",
                success=True,
                duration=30.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                result={"status": "completed"},
                error=None,
                metadata={"execution_mode": "parallel"}
            ),
            Mock(
                phase_name="phase2",
                success=False,
                duration=20.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                result=None,
                error="Phase failed",
                metadata={}
            )
        ]
        
        section = self.reporting_system._generate_phase_results_section(phase_results)
        
        assert isinstance(section, ReportSection)
        assert section.title == "Phase Results"
        assert "Phase1" in section.content
        assert "Phase2" in section.content
        assert "Success" in section.content
        assert "Failed" in section.content
        assert "Phase failed" in section.content
    
    def test_generate_performance_analysis(self):
        """Test performance analysis section generation."""
        summary = {
            "total_duration": 120.0,
            "phases_completed": 3
        }
        
        phase_results = [
            Mock(success=True, duration=30.0, phase_name="phase1", start_time=datetime.now()),
            Mock(success=True, duration=60.0, phase_name="phase2", start_time=datetime.now()),
            Mock(success=False, duration=30.0, phase_name="phase3", start_time=datetime.now())
        ]
        
        section = self.reporting_system._generate_performance_analysis(summary, phase_results)
        
        assert isinstance(section, ReportSection)
        assert section.title == "Performance Analysis"
        assert "<strong>Total Analysis Time:</strong> 120.00 seconds" in section.content
        assert "<strong>Average Phase Duration:</strong> 40.00 seconds" in section.content
        assert "<strong>Success Rate:</strong> 66.7%" in section.content
    
    def test_generate_detailed_analysis(self):
        """Test detailed analysis section generation."""
        analysis_data = {
            "ml_analysis": {
                "clustering": {"n_clusters": 3},
                "anomaly_detection": {"anomalies_found": 5}
            },
            "performance_metrics": {
                "cpu_usage": 75.5,
                "memory_usage": 60.2
            }
        }
        
        section = self.reporting_system._generate_detailed_analysis(analysis_data)
        
        assert isinstance(section, ReportSection)
        assert section.title == "Detailed Analysis"
        assert "Ml Analysis" in section.content
        assert "Performance Metrics" in section.content
        assert "'n_clusters': 3" in section.content
        assert "'anomalies_found': 5" in section.content
        assert "<strong>cpu_usage:</strong> 75.5000" in section.content
    
    def test_generate_recommendations(self):
        """Test recommendations section generation."""
        summary = {
            "errors": ["Error 1", "Error 2"],
            "warnings": ["Warning 1"],
            "total_duration": 7200  # 2 hours
        }
        
        phase_results = [
            Mock(phase_name="ml_analysis", duration=1200),  # 20 minutes
            Mock(phase_name="clustering", duration=400)    # 6.7 minutes
        ]
        
        section = self.reporting_system._generate_recommendations(summary, phase_results)
        
        assert isinstance(section, ReportSection)
        assert section.title == "Recommendations"
        assert "Address the identified errors" in section.content
        assert "Review warnings" in section.content
        assert "optimizing analysis performance" in section.content
        assert "optimizing ML analysis parameters" in section.content
    
    def test_generate_appendix(self):
        """Test appendix section generation."""
        analysis_data = {"test": "data"}
        
        section = self.reporting_system._generate_appendix(analysis_data)
        
        assert isinstance(section, ReportSection)
        assert section.title == "Appendix"
        assert "Configuration Details" in section.content
        assert "System Information" in section.content
        assert "Generated on:" in section.content
        assert "Report version: 1.0.0" in section.content
    
    def test_summarize_result(self):
        """Test result summarization."""
        # Test with dict
        result_dict = {"key": "value"}
        summary = self.reporting_system._summarize_result(result_dict)
        assert "Dictionary with 1 keys" in summary
        
        # Test with list
        result_list = [1, 2, 3]
        summary = self.reporting_system._summarize_result(result_list)
        assert "Collection with 3 items" in summary
        
        # Test with object
        result_obj = Mock()
        result_obj.__dict__ = {"attr1": "value1", "attr2": "value2"}
        summary = self.reporting_system._summarize_result(result_obj)
        assert "Result object with 2 attributes" in summary
        
        # Test with primitive
        result_primitive = "test"
        summary = self.reporting_system._summarize_result(result_primitive)
        assert "str" in summary
    
    def test_export_html_report(self):
        """Test HTML report export."""
        report = AnalysisReport(
            title="Test Report",
            author="Test Author",
            version="1.0.0",
            generated_at=datetime.now(),
            summary={"total_phases": 2},
            sections=[
                ReportSection("Section 1", "Content 1"),
                ReportSection("Section 2", "Content 2")
            ]
        )
        
        with patch('builtins.open', mock_open()) as mock_file:
            self.reporting_system._export_html_report(report, "test_report")
            
            # Verify file was opened for writing
            mock_file.assert_called_once()
            call_args = mock_file.call_args
            assert "test_report.html" in str(call_args[0][0])
            assert call_args[0][1] == 'w'
            assert call_args[1]['encoding'] == 'utf-8'
    
    def test_export_json_report(self):
        """Test JSON report export."""
        report = AnalysisReport(
            title="Test Report",
            author="Test Author",
            version="1.0.0",
            generated_at=datetime.now(),
            summary={"total_phases": 2},
            sections=[
                ReportSection("Section 1", "Content 1"),
                ReportSection("Section 2", "Content 2")
            ]
        )
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            
            self.reporting_system._export_json_report(report, "test_report")
            
            # Verify file was opened and JSON was dumped
            mock_file.assert_called_once()
            mock_json_dump.assert_called_once()
            
            # Check JSON dump arguments
            call_args = mock_json_dump.call_args
            assert call_args[1]['indent'] == 2
            assert call_args[1]['ensure_ascii'] is False
    
    def test_export_markdown_report(self):
        """Test Markdown report export."""
        report = AnalysisReport(
            title="Test Report",
            author="Test Author",
            version="1.0.0",
            generated_at=datetime.now(),
            summary={"total_phases": 2},
            sections=[
                ReportSection("Section 1", "<h2>Content 1</h2><p>Text</p>"),
                ReportSection("Section 2", "<h3>Content 2</h3><ul><li>Item</li></ul>")
            ]
        )
        
        with patch('builtins.open', mock_open()) as mock_file:
            self.reporting_system._export_markdown_report(report, "test_report")
            
            # Verify file was opened for writing
            mock_file.assert_called_once()
            call_args = mock_file.call_args
            assert "test_report.md" in str(call_args[0][0])
            assert call_args[0][1] == 'w'
            assert call_args[1]['encoding'] == 'utf-8'
    
    def test_get_report_summary(self):
        """Test report summary generation."""
        # Create some mock report files
        report_dir = self.reporting_system.output_dir
        report_file1 = report_dir / "analysis_report_20230101_120000.html"
        report_file2 = report_dir / "analysis_report_20230101_130000.html"
        
        # Mock file stats
        with patch.object(Path, 'glob') as mock_glob:
            
            mock_glob.return_value = [report_file1, report_file2]
            
            # Create mock files with stat method
            mock_file1 = Mock()
            mock_file1.name = "analysis_report_20230101_120000.html"
            mock_file1.stat.return_value.st_mtime = 1672574400  # 2023-01-01 12:00:00
            mock_file1.stat.return_value.st_size = 1024 * 1024  # 1MB
            
            mock_file2 = Mock()
            mock_file2.name = "analysis_report_20230101_130000.html"
            mock_file2.stat.return_value.st_mtime = 1672578000  # 2023-01-01 13:00:00
            mock_file2.stat.return_value.st_size = 2 * 1024 * 1024  # 2MB
            
            # Mock the glob to return our mock files
            mock_glob.return_value = [mock_file1, mock_file2]
            
            summary = self.reporting_system.get_report_summary()
            
            assert summary["total_reports"] == 2
            assert len(summary["reports"]) == 2
            assert summary["reports"][0]["filename"] == "analysis_report_20230101_120000.html"
            assert summary["reports"][0]["size_mb"] == 1.0
            assert summary["output_directory"] == str(report_dir)


class TestReportConfig:
    """Test cases for ReportConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()
        
        assert config.output_dir == "analysis_reports"
        assert config.format == "html"
        assert config.include_charts is True
        assert config.include_raw_data is False
        assert config.title == "Simulation Analysis Report"
        assert config.author == "Analysis System"
        assert config.version == "1.0"
        assert config.include_executive_summary is True
        assert config.include_detailed_analysis is True
        assert config.include_recommendations is True
        assert config.include_appendix is True
        assert config.theme == "default"
        assert config.color_scheme == "viridis"
        assert config.font_family == "Arial, sans-serif"
        assert config.template_dir is None
        assert config.custom_template is None
        assert config.export_formats == ["html", "json"]
        assert config.compress_output is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReportConfig(
            output_dir="custom_reports",
            format="pdf",
            include_charts=False,
            include_raw_data=True,
            title="Custom Report",
            author="Custom Author",
            version="2.0",
            include_executive_summary=False,
            include_detailed_analysis=False,
            include_recommendations=False,
            include_appendix=False,
            theme="dark",
            color_scheme="plasma",
            font_family="Times New Roman, serif",
            template_dir="custom_templates",
            custom_template="custom.html",
            export_formats=["pdf", "markdown"],
            compress_output=True
        )
        
        assert config.output_dir == "custom_reports"
        assert config.format == "pdf"
        assert config.include_charts is False
        assert config.include_raw_data is True
        assert config.title == "Custom Report"
        assert config.author == "Custom Author"
        assert config.version == "2.0"
        assert config.include_executive_summary is False
        assert config.include_detailed_analysis is False
        assert config.include_recommendations is False
        assert config.include_appendix is False
        assert config.theme == "dark"
        assert config.color_scheme == "plasma"
        assert config.font_family == "Times New Roman, serif"
        assert config.template_dir == "custom_templates"
        assert config.custom_template == "custom.html"
        assert config.export_formats == ["pdf", "markdown"]
        assert config.compress_output is True


class TestReportSection:
    """Test cases for ReportSection."""
    
    def test_report_section_creation(self):
        """Test report section creation."""
        section = ReportSection(
            title="Test Section",
            content="Test content",
            charts=[{"type": "bar", "data": "test"}],
            tables=[{"headers": ["A", "B"], "rows": [["1", "2"]]}],
            metadata={"key": "value"}
        )
        
        assert section.title == "Test Section"
        assert section.content == "Test content"
        assert len(section.charts) == 1
        assert len(section.tables) == 1
        assert section.metadata == {"key": "value"}
    
    def test_report_section_minimal(self):
        """Test minimal report section creation."""
        section = ReportSection(title="Minimal Section", content="Content")
        
        assert section.title == "Minimal Section"
        assert section.content == "Content"
        assert section.charts == []
        assert section.tables == []
        assert section.metadata == {}


class TestAnalysisReport:
    """Test cases for AnalysisReport."""
    
    def test_analysis_report_creation(self):
        """Test analysis report creation."""
        sections = [
            ReportSection("Section 1", "Content 1"),
            ReportSection("Section 2", "Content 2")
        ]
        
        report = AnalysisReport(
            title="Test Report",
            author="Test Author",
            version="1.0.0",
            generated_at=datetime.now(),
            summary={"total_phases": 2},
            sections=sections,
            raw_data={"test": "data"},
            metadata={"key": "value"}
        )
        
        assert report.title == "Test Report"
        assert report.author == "Test Author"
        assert report.version == "1.0.0"
        assert isinstance(report.generated_at, datetime)
        assert report.summary == {"total_phases": 2}
        assert len(report.sections) == 2
        assert report.raw_data == {"test": "data"}
        assert report.metadata == {"key": "value"}
    
    def test_analysis_report_minimal(self):
        """Test minimal analysis report creation."""
        report = AnalysisReport(
            title="Minimal Report",
            author="Minimal Author",
            version="0.1.0",
            generated_at=datetime.now(),
            summary={},
            sections=[]
        )
        
        assert report.title == "Minimal Report"
        assert report.author == "Minimal Author"
        assert report.version == "0.1.0"
        assert isinstance(report.generated_at, datetime)
        assert report.summary == {}
        assert report.sections == []
        assert report.raw_data is None
        assert report.metadata == {}