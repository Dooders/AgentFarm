"""
Report generator for simulation comparison results.

This module provides comprehensive report generation capabilities for
simulation comparison data, including HTML, PDF, and text reports.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import asdict

from farm.analysis.comparative.comparison_result import SimulationComparisonResult
from farm.analysis.comparative.statistical_analyzer import StatisticalAnalysisResult
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generator for comprehensive simulation comparison reports."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None, 
                 template_dir: Optional[Union[str, Path]] = None):
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save generated reports
            template_dir: Directory containing report templates
        """
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.template_dir = Path(template_dir) if template_dir else None
        
        logger.info(f"ReportGenerator initialized with output directory: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    result: SimulationComparisonResult,
                                    statistical_analysis: Optional[StatisticalAnalysisResult] = None,
                                    visualization_files: Optional[Dict[str, str]] = None,
                                    report_format: str = "html") -> Dict[str, str]:
        """Generate a comprehensive report in multiple formats.
        
        Args:
            result: Simulation comparison result
            statistical_analysis: Optional statistical analysis results
            visualization_files: Optional dictionary of visualization file paths
            report_format: Format to generate ('html', 'pdf', 'text', 'all')
            
        Returns:
            Dictionary mapping report types to file paths
        """
        logger.info(f"Generating comprehensive report in {report_format} format")
        
        report_files = {}
        
        if report_format in ["html", "all"]:
            report_files['html'] = self.generate_html_report(result, statistical_analysis, visualization_files)
        
        if report_format in ["pdf", "all"]:
            report_files['pdf'] = self.generate_pdf_report(result, statistical_analysis, visualization_files)
        
        if report_format in ["text", "all"]:
            report_files['text'] = self.generate_text_report(result, statistical_analysis)
        
        if report_format in ["json", "all"]:
            report_files['json'] = self.generate_json_report(result, statistical_analysis)
        
        if report_format in ["yaml", "all"]:
            report_files['yaml'] = self.generate_yaml_report(result, statistical_analysis)
        
        return report_files
    
    def generate_html_report(self, 
                           result: SimulationComparisonResult,
                           statistical_analysis: Optional[StatisticalAnalysisResult] = None,
                           visualization_files: Optional[Dict[str, str]] = None) -> str:
        """Generate an HTML report."""
        logger.info("Generating HTML report")
        
        # Generate HTML content
        html_content = self._create_html_content(result, statistical_analysis, visualization_files)
        
        # Save HTML file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"comparison_report_{timestamp}.html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {filename}")
        return str(filename)
    
    def generate_pdf_report(self, 
                          result: SimulationComparisonResult,
                          statistical_analysis: Optional[StatisticalAnalysisResult] = None,
                          visualization_files: Optional[Dict[str, str]] = None) -> str:
        """Generate a PDF report."""
        logger.info("Generating PDF report")
        
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            # Create PDF document
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.output_dir / f"comparison_report_{timestamp}.pdf"
            
            doc = SimpleDocTemplate(str(filename), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("Simulation Comparison Report", title_style))
            story.append(Spacer(1, 12))
            
            # Summary section
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            summary_text = self._create_summary_text(result, statistical_analysis)
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Detailed sections
            story.append(Paragraph("Detailed Analysis", styles['Heading2']))
            
            # Configuration comparison
            story.append(Paragraph("Configuration Comparison", styles['Heading3']))
            config_text = self._create_config_section_text(result)
            story.append(Paragraph(config_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Database comparison
            story.append(Paragraph("Database Comparison", styles['Heading3']))
            db_text = self._create_database_section_text(result)
            story.append(Paragraph(db_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Log comparison
            story.append(Paragraph("Log Analysis", styles['Heading3']))
            log_text = self._create_log_section_text(result)
            story.append(Paragraph(log_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Metrics comparison
            story.append(Paragraph("Metrics Comparison", styles['Heading3']))
            metrics_text = self._create_metrics_section_text(result)
            story.append(Paragraph(metrics_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Statistical analysis
            if statistical_analysis:
                story.append(Paragraph("Statistical Analysis", styles['Heading3']))
                stats_text = self._create_statistical_section_text(statistical_analysis)
                story.append(Paragraph(stats_text, styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report saved to {filename}")
            return str(filename)
            
        except ImportError:
            logger.warning("ReportLab not available, falling back to text report")
            return self.generate_text_report(result, statistical_analysis)
    
    def generate_text_report(self, 
                           result: SimulationComparisonResult,
                           statistical_analysis: Optional[StatisticalAnalysisResult] = None) -> str:
        """Generate a text report."""
        logger.info("Generating text report")
        
        # Generate text content
        text_content = self._create_text_content(result, statistical_analysis)
        
        # Save text file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"comparison_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        logger.info(f"Text report saved to {filename}")
        return str(filename)
    
    def generate_json_report(self, 
                           result: SimulationComparisonResult,
                           statistical_analysis: Optional[StatisticalAnalysisResult] = None) -> str:
        """Generate a JSON report."""
        logger.info("Generating JSON report")
        
        # Prepare data for JSON export
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'simulation_comparison',
                'version': '1.0'
            },
            'comparison_result': asdict(result),
            'statistical_analysis': asdict(statistical_analysis) if statistical_analysis else None
        }
        
        # Save JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"comparison_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {filename}")
        return str(filename)
    
    def generate_yaml_report(self, 
                           result: SimulationComparisonResult,
                           statistical_analysis: Optional[StatisticalAnalysisResult] = None) -> str:
        """Generate a YAML report."""
        logger.info("Generating YAML report")
        
        # Prepare data for YAML export
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'simulation_comparison',
                'version': '1.0'
            },
            'comparison_result': self._convert_paths_to_strings(asdict(result)),
            'statistical_analysis': self._convert_paths_to_strings(asdict(statistical_analysis)) if statistical_analysis else None
        }
        
        # Save YAML file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"comparison_report_{timestamp}.yaml"
        
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(report_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"YAML report saved to {filename}")
        return str(filename)
    
    def _convert_paths_to_strings(self, data):
        """Convert Path objects to strings for YAML serialization."""
        if isinstance(data, dict):
            return {key: self._convert_paths_to_strings(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_paths_to_strings(item) for item in data]
        elif isinstance(data, Path):
            return str(data)
        else:
            return data
    
    def _create_html_content(self, 
                           result: SimulationComparisonResult,
                           statistical_analysis: Optional[StatisticalAnalysisResult] = None,
                           visualization_files: Optional[Dict[str, str]] = None) -> str:
        """Create HTML content for the report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulation Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #333;
            margin: 0;
        }}
        .header .subtitle {{
            color: #666;
            margin-top: 10px;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .summary h2 {{
            color: #333;
            margin-top: 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .summary-item {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .summary-item h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .summary-item .value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .section h3 {{
            color: #555;
            margin-top: 25px;
        }}
        .differences-list {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .differences-list ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .differences-list li {{
            margin-bottom: 5px;
        }}
        .no-differences {{
            color: #28a745;
            font-style: italic;
        }}
        .severity-high {{
            color: #dc3545;
            font-weight: bold;
        }}
        .severity-medium {{
            color: #ffc107;
            font-weight: bold;
        }}
        .severity-low {{
            color: #28a745;
            font-weight: bold;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Simulation Comparison Report</h1>
            <div class="subtitle">
                {result.simulation1_path.name} vs {result.simulation2_path.name}
            </div>
            <div class="subtitle">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>Total Differences</h3>
                    <div class="value severity-{result.comparison_summary.severity}">
                        {result.comparison_summary.total_differences}
                    </div>
                </div>
                <div class="summary-item">
                    <h3>Severity</h3>
                    <div class="value severity-{result.comparison_summary.severity}">
                        {result.comparison_summary.severity.upper()}
                    </div>
                </div>
                <div class="summary-item">
                    <h3>Configuration</h3>
                    <div class="value">{result.comparison_summary.config_differences}</div>
                </div>
                <div class="summary-item">
                    <h3>Database</h3>
                    <div class="value">{result.comparison_summary.database_differences}</div>
                </div>
                <div class="summary-item">
                    <h3>Logs</h3>
                    <div class="value">{result.comparison_summary.log_differences}</div>
                </div>
                <div class="summary-item">
                    <h3>Metrics</h3>
                    <div class="value">{result.comparison_summary.metrics_differences}</div>
                </div>
            </div>
        </div>
        
        {self._create_html_sections(result, statistical_analysis, visualization_files)}
        
        <div class="footer">
            <p>Report generated by Farm Simulation Analysis System</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _create_html_sections(self, 
                            result: SimulationComparisonResult,
                            statistical_analysis: Optional[StatisticalAnalysisResult] = None,
                            visualization_files: Optional[Dict[str, str]] = None) -> str:
        """Create HTML sections for detailed analysis."""
        sections = []
        
        # Configuration section
        sections.append(self._create_config_html_section(result))
        
        # Database section
        sections.append(self._create_database_html_section(result))
        
        # Log section
        sections.append(self._create_log_html_section(result))
        
        # Metrics section
        sections.append(self._create_metrics_html_section(result))
        
        # Statistical analysis section
        if statistical_analysis:
            sections.append(self._create_statistical_html_section(statistical_analysis))
        
        # Visualizations section
        if visualization_files:
            sections.append(self._create_visualizations_html_section(visualization_files))
        
        return "\n".join(sections)
    
    def _create_config_html_section(self, result: SimulationComparisonResult) -> str:
        """Create HTML section for configuration comparison."""
        if not result.config_comparison.differences:
            differences_html = '<div class="no-differences">No configuration differences found</div>'
        else:
            differences_list = []
            for key, value in result.config_comparison.differences.items():
                if isinstance(value, list):
                    differences_list.append(f"<li><strong>{key}:</strong> {len(value)} differences</li>")
                elif isinstance(value, dict):
                    differences_list.append(f"<li><strong>{key}:</strong> {len(value)} differences</li>")
                else:
                    differences_list.append(f"<li><strong>{key}:</strong> {value}</li>")
            
            differences_html = f'<div class="differences-list"><ul>{"".join(differences_list)}</ul></div>'
        
        return f"""
        <div class="section">
            <h2>Configuration Comparison</h2>
            <h3>Differences Found</h3>
            {differences_html}
        </div>
        """
    
    def _create_database_html_section(self, result: SimulationComparisonResult) -> str:
        """Create HTML section for database comparison."""
        schema_html = self._create_differences_html(result.database_comparison.schema_differences, "Schema")
        data_html = self._create_differences_html(result.database_comparison.data_differences, "Data")
        metrics_html = self._create_differences_html(result.database_comparison.metric_differences, "Metrics")
        
        return f"""
        <div class="section">
            <h2>Database Comparison</h2>
            <h3>Schema Differences</h3>
            {schema_html}
            <h3>Data Differences</h3>
            {data_html}
            <h3>Metric Differences</h3>
            {metrics_html}
        </div>
        """
    
    def _create_log_html_section(self, result: SimulationComparisonResult) -> str:
        """Create HTML section for log comparison."""
        performance_html = self._create_differences_html(result.log_comparison.performance_differences, "Performance")
        error_html = self._create_differences_html(result.log_comparison.error_differences, "Errors")
        
        return f"""
        <div class="section">
            <h2>Log Analysis</h2>
            <h3>Performance Differences</h3>
            {performance_html}
            <h3>Error Differences</h3>
            {error_html}
        </div>
        """
    
    def _create_metrics_html_section(self, result: SimulationComparisonResult) -> str:
        """Create HTML section for metrics comparison."""
        metric_html = self._create_differences_html(result.metrics_comparison.metric_differences, "Metrics")
        performance_html = self._create_differences_html(result.metrics_comparison.performance_comparison, "Performance")
        
        return f"""
        <div class="section">
            <h2>Metrics Comparison</h2>
            <h3>Metric Differences</h3>
            {metric_html}
            <h3>Performance Comparison</h3>
            {performance_html}
        </div>
        """
    
    def _create_statistical_html_section(self, statistical_analysis: StatisticalAnalysisResult) -> str:
        """Create HTML section for statistical analysis."""
        # Correlation analysis
        correlation_html = self._create_correlation_html(statistical_analysis.correlation_analysis)
        
        # Significance tests
        significance_html = self._create_significance_html(statistical_analysis.significance_tests)
        
        # Trend analysis
        trend_html = self._create_trend_html(statistical_analysis.trend_analysis)
        
        # Anomaly detection
        anomaly_html = self._create_anomaly_html(statistical_analysis.anomaly_detection)
        
        return f"""
        <div class="section">
            <h2>Statistical Analysis</h2>
            <h3>Correlation Analysis</h3>
            {correlation_html}
            <h3>Significance Tests</h3>
            {significance_html}
            <h3>Trend Analysis</h3>
            {trend_html}
            <h3>Anomaly Detection</h3>
            {anomaly_html}
        </div>
        """
    
    def _create_visualizations_html_section(self, visualization_files: Dict[str, str]) -> str:
        """Create HTML section for visualizations."""
        viz_html = []
        for name, file_path in visualization_files.items():
            if file_path and Path(file_path).exists():
                viz_html.append(f"""
                <div class="visualization">
                    <h3>{name.replace('_', ' ').title()}</h3>
                    <img src="{file_path}" alt="{name}">
                </div>
                """)
        
        return f"""
        <div class="section">
            <h2>Visualizations</h2>
            {"".join(viz_html)}
        </div>
        """
    
    def _create_differences_html(self, differences: Dict[str, Any], title: str) -> str:
        """Create HTML for differences display."""
        if not differences:
            return f'<div class="no-differences">No {title.lower()} differences found</div>'
        
        differences_list = []
        for key, value in differences.items():
            if isinstance(value, dict):
                if 'sim1_value' in value and 'sim2_value' in value:
                    # Metric comparison
                    diff = value.get('difference', 0)
                    pct_change = value.get('percentage_change', 0)
                    differences_list.append(f"""
                    <li><strong>{key}:</strong> 
                        Sim1: {value['sim1_value']}, Sim2: {value['sim2_value']}, 
                        Diff: {diff}, Change: {pct_change:.1f}%
                    </li>
                    """)
                elif 'db1_value' in value and 'db2_value' in value:
                    # Database comparison
                    diff = value.get('difference', 0)
                    differences_list.append(f"""
                    <li><strong>{key}:</strong> 
                        DB1: {value['db1_value']}, DB2: {value['db2_value']}, 
                        Diff: {diff}
                    </li>
                    """)
                else:
                    differences_list.append(f"<li><strong>{key}:</strong> {len(value)} items</li>")
            elif isinstance(value, list):
                differences_list.append(f"<li><strong>{key}:</strong> {len(value)} items</li>")
            else:
                differences_list.append(f"<li><strong>{key}:</strong> {value}</li>")
        
        return f'<div class="differences-list"><ul>{"".join(differences_list)}</ul></div>'
    
    def _create_correlation_html(self, correlation_analysis: Dict[str, Any]) -> str:
        """Create HTML for correlation analysis."""
        if not correlation_analysis:
            return '<div class="no-differences">No correlation analysis available</div>'
        
        correlations = []
        for key, corr in correlation_analysis.items():
            if isinstance(corr, dict) and 'correlation' in corr:
                significance = "Significant" if corr.get('significant', False) else "Not significant"
                correlations.append(f"""
                <li><strong>{corr.get('label', key)}:</strong> 
                    Correlation: {corr['correlation']:.3f}, 
                    P-value: {corr.get('p_value', 'N/A')}, 
                    {significance}
                </li>
                """)
        
        if correlations:
            return f'<div class="differences-list"><ul>{"".join(correlations)}</ul></div>'
        else:
            return '<div class="no-differences">No significant correlations found</div>'
    
    def _create_significance_html(self, significance_tests: Dict[str, Any]) -> str:
        """Create HTML for significance tests."""
        if not significance_tests:
            return '<div class="no-differences">No significance tests available</div>'
        
        tests = []
        for key, test in significance_tests.items():
            if isinstance(test, dict) and 'test_name' in test:
                significant = "Significant" if test.get('significant', False) else "Not significant"
                tests.append(f"""
                <li><strong>{test['test_name']}:</strong> 
                    {significant}, 
                    Statistic: {test.get('statistic', 'N/A')}, 
                    P-value: {test.get('p_value', 'N/A')}
                </li>
                """)
        
        if tests:
            return f'<div class="differences-list"><ul>{"".join(tests)}</ul></div>'
        else:
            return '<div class="no-differences">No significant test results</div>'
    
    def _create_trend_html(self, trend_analysis: Dict[str, Any]) -> str:
        """Create HTML for trend analysis."""
        if not trend_analysis:
            return '<div class="no-differences">No trend analysis available</div>'
        
        overall_trend = trend_analysis.get('overall_trend', {})
        if overall_trend:
            trend = overall_trend.get('trend', 'unknown')
            strength = overall_trend.get('trend_strength', 0)
            interpretation = overall_trend.get('interpretation', 'No interpretation available')
            
            return f"""
            <div class="differences-list">
                <ul>
                    <li><strong>Overall Trend:</strong> {trend.title()}</li>
                    <li><strong>Trend Strength:</strong> {strength:.2f}</li>
                    <li><strong>Interpretation:</strong> {interpretation}</li>
                </ul>
            </div>
            """
        else:
            return '<div class="no-differences">No trend information available</div>'
    
    def _create_anomaly_html(self, anomaly_detection: Dict[str, Any]) -> str:
        """Create HTML for anomaly detection."""
        if not anomaly_detection:
            return '<div class="no-differences">No anomaly detection available</div>'
        
        summary = anomaly_detection.get('summary', {})
        if summary:
            total = summary.get('total_anomalies', 0)
            high = summary.get('high_severity', 0)
            medium = summary.get('medium_severity', 0)
            
            return f"""
            <div class="differences-list">
                <ul>
                    <li><strong>Total Anomalies:</strong> {total}</li>
                    <li><strong>High Severity:</strong> {high}</li>
                    <li><strong>Medium Severity:</strong> {medium}</li>
                    <li><strong>Low Severity:</strong> {total - high - medium}</li>
                </ul>
            </div>
            """
        else:
            return '<div class="no-differences">No anomaly information available</div>'
    
    def _create_text_content(self, 
                           result: SimulationComparisonResult,
                           statistical_analysis: Optional[StatisticalAnalysisResult] = None) -> str:
        """Create text content for the report."""
        content = []
        
        # Header
        content.append("=" * 80)
        content.append("SIMULATION COMPARISON REPORT")
        content.append("=" * 80)
        content.append(f"Simulation 1: {result.simulation1_path}")
        content.append(f"Simulation 2: {result.simulation2_path}")
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Summary
        content.append("EXECUTIVE SUMMARY")
        content.append("-" * 40)
        content.append(f"Total Differences: {result.comparison_summary.total_differences}")
        content.append(f"Severity: {result.comparison_summary.severity.upper()}")
        content.append(f"Configuration Differences: {result.comparison_summary.config_differences}")
        content.append(f"Database Differences: {result.comparison_summary.database_differences}")
        content.append(f"Log Differences: {result.comparison_summary.log_differences}")
        content.append(f"Metrics Differences: {result.comparison_summary.metrics_differences}")
        content.append("")
        
        # Detailed sections
        content.append(self._create_config_section_text(result))
        content.append(self._create_database_section_text(result))
        content.append(self._create_log_section_text(result))
        content.append(self._create_metrics_section_text(result))
        
        if statistical_analysis:
            content.append(self._create_statistical_section_text(statistical_analysis))
        
        return "\n".join(content)
    
    def _create_summary_text(self, 
                           result: SimulationComparisonResult,
                           statistical_analysis: Optional[StatisticalAnalysisResult] = None) -> str:
        """Create summary text for the report."""
        summary_parts = []
        
        summary_parts.append(f"This report compares two simulations: {result.simulation1_path.name} and {result.simulation2_path.name}.")
        summary_parts.append(f"A total of {result.comparison_summary.total_differences} differences were found across all categories.")
        summary_parts.append(f"The overall severity of differences is classified as {result.comparison_summary.severity.upper()}.")
        
        if result.comparison_summary.config_differences > 0:
            summary_parts.append(f"Configuration differences: {result.comparison_summary.config_differences}")
        
        if result.comparison_summary.database_differences > 0:
            summary_parts.append(f"Database differences: {result.comparison_summary.database_differences}")
        
        if result.comparison_summary.log_differences > 0:
            summary_parts.append(f"Log differences: {result.comparison_summary.log_differences}")
        
        if result.comparison_summary.metrics_differences > 0:
            summary_parts.append(f"Metrics differences: {result.comparison_summary.metrics_differences}")
        
        if statistical_analysis:
            summary = statistical_analysis.summary
            if summary.get('total_anomalies', 0) > 0:
                summary_parts.append(f"Statistical analysis detected {summary['total_anomalies']} anomalies.")
            
            trend = summary.get('overall_trend', 'unknown')
            if trend != 'unknown':
                summary_parts.append(f"Overall trend: {trend}")
        
        return " ".join(summary_parts) + "."
    
    def _create_config_section_text(self, result: SimulationComparisonResult) -> str:
        """Create configuration section text."""
        content = []
        content.append("CONFIGURATION COMPARISON")
        content.append("-" * 40)
        
        if not result.config_comparison.differences:
            content.append("No configuration differences found.")
        else:
            content.append("Configuration differences found:")
            for key, value in result.config_comparison.differences.items():
                if isinstance(value, list):
                    content.append(f"  {key}: {len(value)} differences")
                elif isinstance(value, dict):
                    content.append(f"  {key}: {len(value)} differences")
                else:
                    content.append(f"  {key}: {value}")
        
        content.append("")
        return "\n".join(content)
    
    def _create_database_section_text(self, result: SimulationComparisonResult) -> str:
        """Create database section text."""
        content = []
        content.append("DATABASE COMPARISON")
        content.append("-" * 40)
        
        # Schema differences
        content.append("Schema Differences:")
        if not result.database_comparison.schema_differences:
            content.append("  No schema differences found.")
        else:
            for table, changes in result.database_comparison.schema_differences.items():
                if isinstance(changes, dict):
                    content.append(f"  {table}: {len(changes)} changes")
                elif isinstance(changes, list):
                    content.append(f"  {table}: {len(changes)} changes")
                else:
                    content.append(f"  {table}: 1 change")
        
        # Data differences
        content.append("\nData Differences:")
        if not result.database_comparison.data_differences:
            content.append("  No data differences found.")
        else:
            for table, info in result.database_comparison.data_differences.items():
                if isinstance(info, dict) and 'difference' in info:
                    content.append(f"  {table}: {info['difference']} row difference")
                else:
                    content.append(f"  {table}: differences found")
        
        # Metric differences
        content.append("\nMetric Differences:")
        if not result.database_comparison.metric_differences:
            content.append("  No metric differences found.")
        else:
            for metric, diff in result.database_comparison.metric_differences.items():
                if isinstance(diff, dict) and 'db1_value' in diff and 'db2_value' in diff:
                    content.append(f"  {metric}: DB1={diff['db1_value']}, DB2={diff['db2_value']}")
                else:
                    content.append(f"  {metric}: differences found")
        
        content.append("")
        return "\n".join(content)
    
    def _create_log_section_text(self, result: SimulationComparisonResult) -> str:
        """Create log section text."""
        content = []
        content.append("LOG ANALYSIS")
        content.append("-" * 40)
        
        # Performance differences
        content.append("Performance Differences:")
        if not result.log_comparison.performance_differences:
            content.append("  No performance differences found.")
        else:
            for metric, diff in result.log_comparison.performance_differences.items():
                if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                    content.append(f"  {metric}: Sim1={diff['sim1_value']}, Sim2={diff['sim2_value']}")
                else:
                    content.append(f"  {metric}: differences found")
        
        # Error differences
        content.append("\nError Differences:")
        if not result.log_comparison.error_differences:
            content.append("  No error differences found.")
        else:
            for error_type, diff in result.log_comparison.error_differences.items():
                if isinstance(diff, dict) and 'sim1_count' in diff and 'sim2_count' in diff:
                    content.append(f"  {error_type}: Sim1={diff['sim1_count']}, Sim2={diff['sim2_count']}")
                else:
                    content.append(f"  {error_type}: differences found")
        
        content.append("")
        return "\n".join(content)
    
    def _create_metrics_section_text(self, result: SimulationComparisonResult) -> str:
        """Create metrics section text."""
        content = []
        content.append("METRICS COMPARISON")
        content.append("-" * 40)
        
        # Metric differences
        content.append("Metric Differences:")
        if not result.metrics_comparison.metric_differences:
            content.append("  No metric differences found.")
        else:
            for metric, diff in result.metrics_comparison.metric_differences.items():
                if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                    pct_change = diff.get('percentage_change', 0)
                    content.append(f"  {metric}: Sim1={diff['sim1_value']}, Sim2={diff['sim2_value']}, Change={pct_change:.1f}%")
                else:
                    content.append(f"  {metric}: differences found")
        
        # Performance comparison
        content.append("\nPerformance Comparison:")
        if not result.metrics_comparison.performance_comparison:
            content.append("  No performance comparison available.")
        else:
            for metric, comp in result.metrics_comparison.performance_comparison.items():
                if isinstance(comp, dict) and 'ratio' in comp:
                    content.append(f"  {metric}: Ratio={comp['ratio']:.2f}, Faster={comp.get('faster', 'equal')}")
                else:
                    content.append(f"  {metric}: comparison available")
        
        content.append("")
        return "\n".join(content)
    
    def _create_statistical_section_text(self, statistical_analysis: StatisticalAnalysisResult) -> str:
        """Create statistical analysis section text."""
        content = []
        content.append("STATISTICAL ANALYSIS")
        content.append("-" * 40)
        
        # Summary
        summary = statistical_analysis.summary
        content.append(f"Analysis Quality: {summary.get('analysis_quality', 'Unknown')}")
        content.append(f"Significant Correlations: {summary.get('significant_correlations', 0)}")
        content.append(f"Significant Tests: {summary.get('significant_tests', 0)}")
        content.append(f"Overall Trend: {summary.get('overall_trend', 'Unknown')}")
        content.append(f"Total Anomalies: {summary.get('total_anomalies', 0)}")
        
        content.append("")
        return "\n".join(content)