"""
Tests for CLI interface.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

from farm.analysis.comparative.cli_interface import (
    create_argument_parser, run_comparison, generate_visualizations,
    perform_statistical_analysis, generate_reports, print_summary,
    validate_arguments
)
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, ConfigComparisonResult,
    DatabaseComparisonResult, LogComparisonResult, MetricsComparisonResult
)


class TestCLIInterface:
    """Test cases for CLI interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sim1_dir = Path(self.temp_dir) / "sim1"
        self.sim2_dir = Path(self.temp_dir) / "sim2"
        
        # Create simulation directories
        self.sim1_dir.mkdir(parents=True, exist_ok=True)
        self.sim2_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some test files
        (self.sim1_dir / "config.json").write_text('{"param1": "value1"}')
        (self.sim2_dir / "config.json").write_text('{"param1": "value2"}')
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_argument_parser(self):
        """Test argument parser creation."""
        parser = create_argument_parser()
        
        assert parser is not None
        assert parser.prog in ['simulation comparison', '__main__.py']
        
        # Test parsing valid arguments
        args = parser.parse_args([str(self.sim1_dir), str(self.sim2_dir)])
        assert args.simulation1 == str(self.sim1_dir)
        assert args.simulation2 == str(self.sim2_dir)
        assert args.output_dir == 'comparison_results'
        assert args.report == ['text']
        assert args.visualize is False
        assert args.statistical is False
    
    def test_create_argument_parser_with_options(self):
        """Test argument parser with various options."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            str(self.sim1_dir), str(self.sim2_dir),
            '--output-dir', 'custom_output',
            '--visualize', '--statistical',
            '--report', 'html', 'pdf',
            '--no-logs', '--no-metrics',
            '--analysis-modules', 'module1', 'module2',
            '--significance-level', '0.01',
            '--verbose'
        ])
        
        assert args.output_dir == 'custom_output'
        assert args.visualize is True
        assert args.statistical is True
        assert args.report == ['html', 'pdf']
        assert args.no_logs is True
        assert args.no_metrics is True
        assert args.analysis_modules == ['module1', 'module2']
        assert args.significance_level == 0.01
        assert args.verbose is True
    
    def test_validate_arguments_valid(self):
        """Test argument validation with valid arguments."""
        args = Mock()
        args.simulation1 = str(self.sim1_dir)
        args.simulation2 = str(self.sim2_dir)
        args.significance_level = 0.05
        args.output_dir = 'test_output'
        
        # Should not raise any exceptions
        validate_arguments(args)
    
    def test_validate_arguments_invalid_paths(self):
        """Test argument validation with invalid paths."""
        args = Mock()
        args.simulation1 = '/nonexistent/path'
        args.simulation2 = str(self.sim2_dir)
        args.significance_level = 0.05
        args.output_dir = 'test_output'
        
        with pytest.raises(FileNotFoundError):
            validate_arguments(args)
    
    def test_validate_arguments_invalid_significance_level(self):
        """Test argument validation with invalid significance level."""
        args = Mock()
        args.simulation1 = str(self.sim1_dir)
        args.simulation2 = str(self.sim2_dir)
        args.significance_level = 1.5  # Invalid
        args.output_dir = 'test_output'
        
        with pytest.raises(ValueError):
            validate_arguments(args)
    
    @patch('farm.analysis.comparative.cli_interface.FileComparisonEngine')
    def test_run_comparison(self, mock_engine_class):
        """Test running comparison."""
        # Mock the comparison engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # Mock the comparison result
        mock_result = Mock()
        mock_result.comparison_summary.total_differences = 5
        mock_result.comparison_summary.severity = "medium"
        mock_engine.compare_simulations.return_value = mock_result
        
        args = Mock()
        args.simulation1 = str(self.sim1_dir)
        args.simulation2 = str(self.sim2_dir)
        args.output_dir = 'test_output'
        args.no_logs = False
        args.no_metrics = False
        args.analysis_modules = None
        args.quiet = False
        
        result = run_comparison(args)
        
        assert result == mock_result
        mock_engine.compare_simulations.assert_called_once()
    
    @patch('farm.analysis.comparative.cli_interface.VisualizationEngine')
    def test_generate_visualizations(self, mock_viz_class):
        """Test generating visualizations."""
        # Mock the visualization engine
        mock_viz_engine = Mock()
        mock_viz_class.return_value = mock_viz_engine
        
        # Mock visualization files
        mock_viz_engine.create_comparison_dashboard.return_value = {
            'summary_chart': 'summary.png',
            'metrics_comparison': 'metrics.png'
        }
        mock_viz_engine.create_interactive_plot.return_value = 'interactive.html'
        mock_viz_engine.export_data_for_external_tools.return_value = {
            'summary_json': 'summary.json'
        }
        
        mock_result = Mock()
        args = Mock()
        args.output_dir = 'test_output'
        args.viz_style = 'default'
        args.interactive = True
        args.quiet = False
        
        visualization_files = generate_visualizations(mock_result, args)
        
        assert isinstance(visualization_files, dict)
        assert 'summary_chart' in visualization_files
        assert 'metrics_comparison' in visualization_files
        assert 'interactive' in visualization_files
        assert 'summary_json' in visualization_files
    
    @patch('farm.analysis.comparative.cli_interface.StatisticalAnalyzer')
    def test_perform_statistical_analysis(self, mock_analyzer_class):
        """Test performing statistical analysis."""
        # Mock the statistical analyzer
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock analysis result
        mock_analysis_result = Mock()
        mock_analyzer.analyze_comparison.return_value = mock_analysis_result
        
        mock_result = Mock()
        args = Mock()
        args.output_dir = 'test_output'
        args.significance_level = 0.05
        args.quiet = False
        
        analysis_result = perform_statistical_analysis(mock_result, args)
        
        assert analysis_result == mock_analysis_result
        mock_analyzer.analyze_comparison.assert_called_once_with(mock_result)
    
    @patch('farm.analysis.comparative.cli_interface.ReportGenerator')
    def test_generate_reports(self, mock_report_class):
        """Test generating reports."""
        # Mock the report generator
        mock_report_generator = Mock()
        mock_report_class.return_value = mock_report_generator
        
        # Mock report files
        mock_report_generator.generate_comprehensive_report.return_value = {
            'html': 'report.html',
            'text': 'report.txt'
        }
        
        mock_result = Mock()
        mock_statistical_analysis = Mock()
        mock_visualization_files = {'chart1': 'chart.png'}
        
        args = Mock()
        args.output_dir = 'test_output'
        args.report = ['html', 'text']
        args.template_dir = None
        args.quiet = False
        
        generate_reports(mock_result, mock_statistical_analysis, mock_visualization_files, args)
        
        # Should be called once for each report format
        assert mock_report_generator.generate_comprehensive_report.call_count == 2
    
    def test_print_summary(self):
        """Test printing summary."""
        # Create mock result
        mock_result = Mock()
        mock_result.simulation1_path = Path("sim1")
        mock_result.simulation2_path = Path("sim2")
        mock_result.comparison_summary.comparison_time = "2024-01-01T00:00:00"
        mock_result.comparison_summary.total_differences = 5
        mock_result.comparison_summary.severity = "medium"
        mock_result.comparison_summary.config_differences = 1
        mock_result.comparison_summary.database_differences = 2
        mock_result.comparison_summary.log_differences = 1
        mock_result.comparison_summary.metrics_differences = 1
        
        args = Mock()
        args.output_dir = 'test_output'
        args.visualize = True
        args.statistical = True
        args.verbose = False
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            print_summary(mock_result, args)
        
        output = captured_output.getvalue()
        assert 'COMPARISON SUMMARY' in output
        assert 'sim1' in output
        assert 'sim2' in output
        assert '5' in output
        assert 'MEDIUM' in output
    
    def test_print_summary_verbose(self):
        """Test printing summary with verbose output."""
        # Create mock result with detailed differences
        mock_result = Mock()
        mock_result.simulation1_path = Path("sim1")
        mock_result.simulation2_path = Path("sim2")
        mock_result.comparison_summary.comparison_time = "2024-01-01T00:00:00"
        mock_result.comparison_summary.total_differences = 5
        mock_result.comparison_summary.severity = "medium"
        mock_result.comparison_summary.config_differences = 1
        mock_result.comparison_summary.database_differences = 2
        mock_result.comparison_summary.log_differences = 1
        mock_result.comparison_summary.metrics_differences = 1
        
        # Mock detailed differences
        mock_result.config_comparison.differences = {"param1": "value1"}
        mock_result.database_comparison.schema_differences = {"table1": {"col1": "change"}}
        mock_result.database_comparison.data_differences = {"table1": {"difference": 5}}
        mock_result.database_comparison.metric_differences = {"metric1": {"db1_value": 10, "db2_value": 15}}
        mock_result.log_comparison.performance_differences = {"cpu": {"sim1_value": 50, "sim2_value": 60}}
        mock_result.log_comparison.error_differences = {"error1": {"sim1_count": 2, "sim2_count": 5}}
        mock_result.metrics_comparison.metric_differences = {"throughput": {"sim1_value": 100, "sim2_value": 120, "percentage_change": 20.0}}
        mock_result.metrics_comparison.performance_comparison = {"latency": {"ratio": 0.8, "faster": "sim2"}}
        
        args = Mock()
        args.output_dir = 'test_output'
        args.visualize = True
        args.statistical = True
        args.verbose = True
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            print_summary(mock_result, args)
        
        output = captured_output.getvalue()
        assert 'DETAILED DIFFERENCES' in output
        assert 'Configuration Differences' in output
        assert 'Database Schema Differences' in output
        assert 'Log Performance Differences' in output
        assert 'Metrics Differences' in output
    
    def test_main_function(self):
        """Test main function execution."""
        # Mock all the functions called by main
        with patch('farm.analysis.comparative.cli_interface.run_comparison') as mock_run, \
             patch('farm.analysis.comparative.cli_interface.generate_visualizations') as mock_viz, \
             patch('farm.analysis.comparative.cli_interface.perform_statistical_analysis') as mock_stats, \
             patch('farm.analysis.comparative.cli_interface.generate_reports') as mock_reports, \
             patch('farm.analysis.comparative.cli_interface.print_summary') as mock_print, \
             patch('sys.argv', ['cli_interface.py', str(self.sim1_dir), str(self.sim2_dir)]):
            
            from farm.analysis.comparative.cli_interface import main
            
            # Mock return values
            mock_result = Mock()
            mock_run.return_value = mock_result
            mock_viz.return_value = {}
            mock_stats.return_value = Mock()
            
            # Should not raise exceptions
            main()
            
            # Verify functions were called
            mock_run.assert_called_once()
            # Visualization generation is optional based on args
            # mock_viz.assert_called_once()
            # Statistical analysis is optional based on args
            # mock_stats.assert_called_once()
            mock_reports.assert_called_once()
            mock_print.assert_called_once()
    
    def test_main_function_with_error(self):
        """Test main function with error handling."""
        with patch('farm.analysis.comparative.cli_interface.run_comparison', side_effect=Exception("Test error")), \
             patch('sys.argv', ['cli_interface.py', str(self.sim1_dir), str(self.sim2_dir)]):
            
            from farm.analysis.comparative.cli_interface import main
            
            # Should exit with code 1
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1
    
    def test_argument_parser_help(self):
        """Test argument parser help text."""
        parser = create_argument_parser()
        
        # Test that help text contains expected information
        help_text = parser.format_help()
        assert 'Compare two simulations' in help_text
        assert 'simulation1' in help_text
        assert 'simulation2' in help_text
        assert '--output-dir' in help_text
        assert '--visualize' in help_text
        assert '--statistical' in help_text
    
    def test_argument_parser_examples(self):
        """Test argument parser examples."""
        parser = create_argument_parser()
        
        # Test that examples are included
        help_text = parser.format_help()
        assert 'Examples:' in help_text
        assert 'python -m farm.analysis.comparative.cli_interface' in help_text
    
    def test_output_directory_creation(self):
        """Test that output directory is created during validation."""
        args = Mock()
        args.simulation1 = str(self.sim1_dir)
        args.simulation2 = str(self.sim2_dir)
        args.significance_level = 0.05
        args.output_dir = str(Path(self.temp_dir) / "new_output_dir")
        
        # Directory should not exist initially
        assert not Path(args.output_dir).exists()
        
        # Validation should create the directory
        validate_arguments(args)
        
        # Directory should now exist
        assert Path(args.output_dir).exists()
    
    def test_file_type_validation(self):
        """Test file type validation."""
        # Create a file instead of directory
        file_path = Path(self.temp_dir) / "not_a_directory"
        file_path.write_text("test")
        
        args = Mock()
        args.simulation1 = str(file_path)
        args.simulation2 = str(self.sim2_dir)
        args.significance_level = 0.05
        args.output_dir = 'test_output'
        
        with pytest.raises(NotADirectoryError):
            validate_arguments(args)