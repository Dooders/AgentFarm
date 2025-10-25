"""
Comprehensive reporting and documentation system for simulation analysis.

This module provides automated report generation, documentation creation,
and result summarization capabilities.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import base64
from io import BytesIO

# Optional imports for enhanced reporting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Output settings
    output_dir: Union[str, Path] = "analysis_reports"
    format: str = "html"  # html, pdf, json, markdown
    include_charts: bool = True
    include_raw_data: bool = False
    
    # Report content
    title: str = "Simulation Analysis Report"
    author: str = "Analysis System"
    version: str = "1.0"
    include_executive_summary: bool = True
    include_detailed_analysis: bool = True
    include_recommendations: bool = True
    include_appendix: bool = True
    
    # Styling
    theme: str = "default"  # default, dark, light
    color_scheme: str = "viridis"
    font_family: str = "Arial, sans-serif"
    
    # Template settings
    template_dir: Optional[Union[str, Path]] = None
    custom_template: Optional[str] = None
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["html", "json"])
    compress_output: bool = False


@dataclass
class ReportSection:
    """A section of the analysis report."""
    
    title: str
    content: str
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    
    title: str
    author: str
    version: str
    generated_at: datetime
    summary: Dict[str, Any]
    sections: List[ReportSection]
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportingSystem:
    """Comprehensive reporting system for simulation analysis."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize the reporting system."""
        self.config = config or ReportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize template engine if available
        self.template_env = None
        if JINJA2_AVAILABLE and self.config.template_dir:
            template_dir = Path(self.config.template_dir)
            if template_dir.exists():
                self.template_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        logger.info("ReportingSystem initialized")
    
    def generate_comprehensive_report(self, 
                                    orchestration_result: Any,
                                    analysis_data: Dict[str, Any],
                                    config: Optional[Dict[str, Any]] = None) -> AnalysisReport:
        """Generate a comprehensive analysis report."""
        logger.info("Generating comprehensive analysis report")
        
        # Extract data from orchestration result
        summary = self._extract_summary_data(orchestration_result)
        phase_results = getattr(orchestration_result, 'phase_results', [])
        
        # Generate report sections
        sections = []
        
        # Executive Summary
        if self.config.include_executive_summary:
            sections.append(self._generate_executive_summary(summary, phase_results))
        
        # Analysis Overview
        sections.append(self._generate_analysis_overview(summary, phase_results))
        
        # Phase Results
        sections.append(self._generate_phase_results_section(phase_results))
        
        # Performance Analysis
        sections.append(self._generate_performance_analysis(summary, phase_results))
        
        # Detailed Analysis
        if self.config.include_detailed_analysis:
            sections.append(self._generate_detailed_analysis(analysis_data))
        
        # Recommendations
        if self.config.include_recommendations:
            sections.append(self._generate_recommendations(summary, phase_results))
        
        # Appendix
        if self.config.include_appendix:
            sections.append(self._generate_appendix(analysis_data))
        
        # Create report
        report = AnalysisReport(
            title=self.config.title,
            author=self.config.author,
            version=self.config.version,
            generated_at=datetime.now(),
            summary=summary,
            sections=sections,
            raw_data=analysis_data if self.config.include_raw_data else None,
            metadata={
                "config": self.config.__dict__,
                "generation_time": datetime.now().isoformat()
            }
        )
        
        # Export report
        self._export_report(report)
        
        logger.info("Comprehensive report generated successfully")
        return report
    
    def _extract_summary_data(self, orchestration_result: Any) -> Dict[str, Any]:
        """Extract summary data from orchestration result."""
        return {
            "success": getattr(orchestration_result, 'success', False),
            "total_duration": getattr(orchestration_result, 'total_duration', 0),
            "phases_completed": len(getattr(orchestration_result, 'phase_results', [])),
            "errors": getattr(orchestration_result, 'errors', []),
            "warnings": getattr(orchestration_result, 'warnings', []),
            "output_paths": getattr(orchestration_result, 'output_paths', {}),
            "summary": getattr(orchestration_result, 'summary', {})
        }
    
    def _generate_executive_summary(self, summary: Dict[str, Any], phase_results: List[Any]) -> ReportSection:
        """Generate executive summary section."""
        content = f"""
        <h2>Executive Summary</h2>
        <p>This report presents a comprehensive analysis of simulation comparison results generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
        
        <h3>Key Findings</h3>
        <ul>
            <li><strong>Analysis Status:</strong> {'Successfully completed' if summary['success'] else 'Failed'}</li>
            <li><strong>Total Duration:</strong> {summary['total_duration']:.2f} seconds</li>
            <li><strong>Phases Completed:</strong> {summary['phases_completed']}</li>
            <li><strong>Errors:</strong> {len(summary['errors'])}</li>
            <li><strong>Warnings:</strong> {len(summary['warnings'])}</li>
        </ul>
        
        <h3>Analysis Overview</h3>
        <p>The analysis included multiple phases of examination, from basic comparison metrics to advanced machine learning-based pattern recognition and anomaly detection.</p>
        """
        
        # Add charts if available
        charts = []
        if self.config.include_charts and MATPLOTLIB_AVAILABLE:
            charts.append(self._create_phase_duration_chart(phase_results))
            charts.append(self._create_success_rate_chart(phase_results))
        
        return ReportSection(
            title="Executive Summary",
            content=content,
            charts=charts,
            metadata={"section_type": "executive_summary"}
        )
    
    def _generate_analysis_overview(self, summary: Dict[str, Any], phase_results: List[Any]) -> ReportSection:
        """Generate analysis overview section."""
        successful_phases = [p for p in phase_results if getattr(p, 'success', False)]
        failed_phases = [p for p in phase_results if not getattr(p, 'success', False)]
        
        content = f"""
        <h2>Analysis Overview</h2>
        
        <h3>Phase Execution Summary</h3>
        <table class="analysis-table">
            <tr>
                <th>Phase</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Details</th>
            </tr>
        """
        
        for phase in phase_results:
            phase_name = getattr(phase, 'phase_name', 'Unknown')
            success = getattr(phase, 'success', False)
            duration = getattr(phase, 'duration', 0)
            error = getattr(phase, 'error', '')
            
            status = "✅ Success" if success else "❌ Failed"
            details = error if error else "Completed successfully"
            
            content += f"""
            <tr>
                <td>{phase_name}</td>
                <td>{status}</td>
                <td>{duration:.2f}</td>
                <td>{details}</td>
            </tr>
            """
        
        content += "</table>"
        
        # Add performance metrics
        if summary.get('summary', {}).get('performance_metrics'):
            metrics = summary['summary']['performance_metrics']
            content += f"""
            <h3>Performance Metrics</h3>
            <ul>
                <li><strong>Total Simulations Analyzed:</strong> {metrics.get('total_simulations_analyzed', 'N/A')}</li>
                <li><strong>Analysis Throughput:</strong> {metrics.get('analysis_throughput', 0):.2f} simulations/second</li>
            </ul>
            """
        
        return ReportSection(
            title="Analysis Overview",
            content=content,
            metadata={"section_type": "overview"}
        )
    
    def _generate_phase_results_section(self, phase_results: List[Any]) -> ReportSection:
        """Generate detailed phase results section."""
        content = "<h2>Detailed Phase Results</h2>"
        
        for phase in phase_results:
            phase_name = getattr(phase, 'phase_name', 'Unknown')
            success = getattr(phase, 'success', False)
            duration = getattr(phase, 'duration', 0)
            result = getattr(phase, 'result', None)
            error = getattr(phase, 'error', '')
            metadata = getattr(phase, 'metadata', {})
            
            content += f"""
            <h3>{phase_name.replace('_', ' ').title()}</h3>
            <div class="phase-result">
                <p><strong>Status:</strong> {'Success' if success else 'Failed'}</p>
                <p><strong>Duration:</strong> {duration:.2f} seconds</p>
                <p><strong>Execution Mode:</strong> {metadata.get('execution_mode', 'Unknown')}</p>
            """
            
            if error:
                content += f"<p><strong>Error:</strong> {error}</p>"
            
            if result and success:
                content += f"<p><strong>Result Summary:</strong> {self._summarize_result(result)}</p>"
            
            content += "</div>"
        
        return ReportSection(
            title="Phase Results",
            content=content,
            metadata={"section_type": "phase_results"}
        )
    
    def _generate_performance_analysis(self, summary: Dict[str, Any], phase_results: List[Any]) -> ReportSection:
        """Generate performance analysis section."""
        content = "<h2>Performance Analysis</h2>"
        
        # Calculate performance metrics
        total_duration = sum(getattr(p, 'duration', 0) for p in phase_results)
        avg_duration = total_duration / len(phase_results) if phase_results else 0
        successful_phases = [p for p in phase_results if getattr(p, 'success', False)]
        success_rate = len(successful_phases) / len(phase_results) if phase_results else 0
        
        content += f"""
        <h3>Performance Metrics</h3>
        <ul>
            <li><strong>Total Analysis Time:</strong> {total_duration:.2f} seconds</li>
            <li><strong>Average Phase Duration:</strong> {avg_duration:.2f} seconds</li>
            <li><strong>Success Rate:</strong> {success_rate:.1%}</li>
            <li><strong>Phases Completed:</strong> {len(successful_phases)}/{len(phase_results)}</li>
        </ul>
        """
        
        # Add performance charts
        charts = []
        if self.config.include_charts and MATPLOTLIB_AVAILABLE:
            charts.append(self._create_performance_timeline_chart(phase_results))
            charts.append(self._create_phase_efficiency_chart(phase_results))
        
        return ReportSection(
            title="Performance Analysis",
            content=content,
            charts=charts,
            metadata={"section_type": "performance"}
        )
    
    def _generate_detailed_analysis(self, analysis_data: Dict[str, Any]) -> ReportSection:
        """Generate detailed analysis section."""
        content = "<h2>Detailed Analysis</h2>"
        
        # Process different types of analysis data
        for analysis_type, data in analysis_data.items():
            content += f"<h3>{analysis_type.replace('_', ' ').title()}</h3>"
            
            if isinstance(data, dict):
                content += "<ul>"
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        content += f"<li><strong>{key}:</strong> {value:.4f}</li>"
                    else:
                        content += f"<li><strong>{key}:</strong> {str(value)}</li>"
                content += "</ul>"
            else:
                content += f"<p>{str(data)}</p>"
        
        return ReportSection(
            title="Detailed Analysis",
            content=content,
            metadata={"section_type": "detailed_analysis"}
        )
    
    def _generate_recommendations(self, summary: Dict[str, Any], phase_results: List[Any]) -> ReportSection:
        """Generate recommendations section."""
        content = "<h2>Recommendations</h2>"
        
        recommendations = []
        
        # Analyze errors and warnings
        if summary.get('errors'):
            recommendations.append("Address the identified errors to improve analysis reliability.")
        
        if summary.get('warnings'):
            recommendations.append("Review warnings to optimize analysis performance.")
        
        # Analyze performance
        failed_phases = [p for p in phase_results if not getattr(p, 'success', False)]
        if failed_phases:
            recommendations.append("Investigate failed phases to ensure complete analysis coverage.")
        
        # Analyze duration
        total_duration = summary.get('total_duration', 0)
        if total_duration > 3600:  # More than 1 hour
            recommendations.append("Consider optimizing analysis performance for large-scale processing.")
        
        # Add specific recommendations based on analysis results
        recommendations.extend(self._generate_specific_recommendations(phase_results))
        
        content += "<ul>"
        for i, rec in enumerate(recommendations, 1):
            content += f"<li>{rec}</li>"
        content += "</ul>"
        
        return ReportSection(
            title="Recommendations",
            content=content,
            metadata={"section_type": "recommendations"}
        )
    
    def _generate_specific_recommendations(self, phase_results: List[Any]) -> List[str]:
        """Generate specific recommendations based on phase results."""
        recommendations = []
        
        for phase in phase_results:
            phase_name = getattr(phase, 'phase_name', '')
            duration = getattr(phase, 'duration', 0)
            
            if phase_name == 'ml_analysis' and duration > 600:  # More than 10 minutes
                recommendations.append("Consider optimizing ML analysis parameters for faster processing.")
            
            if phase_name == 'clustering' and duration > 300:  # More than 5 minutes
                recommendations.append("Review clustering parameters to improve performance.")
        
        return recommendations
    
    def _generate_appendix(self, analysis_data: Dict[str, Any]) -> ReportSection:
        """Generate appendix section."""
        content = "<h2>Appendix</h2>"
        
        content += "<h3>Configuration Details</h3>"
        content += f"<p>Report generated with configuration: {json.dumps(self.config.__dict__, indent=2)}</p>"
        
        content += "<h3>System Information</h3>"
        content += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        content += f"<p>Report version: {self.config.version}</p>"
        
        if self.config.include_raw_data and analysis_data:
            content += "<h3>Raw Analysis Data</h3>"
            content += f"<pre>{json.dumps(analysis_data, indent=2)}</pre>"
        
        return ReportSection(
            title="Appendix",
            content=content,
            metadata={"section_type": "appendix"}
        )
    
    def _summarize_result(self, result: Any) -> str:
        """Summarize a phase result."""
        if hasattr(result, '__dict__'):
            return f"Result object with {len(result.__dict__)} attributes"
        elif isinstance(result, (list, tuple)):
            return f"Collection with {len(result)} items"
        elif isinstance(result, dict):
            return f"Dictionary with {len(result)} keys"
        else:
            return str(type(result).__name__)
    
    def _create_phase_duration_chart(self, phase_results: List[Any]) -> Dict[str, Any]:
        """Create phase duration chart."""
        if not MATPLOTLIB_AVAILABLE:
            return {}
        
        phases = [getattr(p, 'phase_name', 'Unknown') for p in phase_results]
        durations = [getattr(p, 'duration', 0) for p in phase_results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(phases, durations)
        plt.title('Phase Duration Analysis')
        plt.xlabel('Phase')
        plt.ylabel('Duration (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "image",
            "title": "Phase Duration Analysis",
            "data": f"data:image/png;base64,{image_base64}",
            "width": 800,
            "height": 400
        }
    
    def _create_success_rate_chart(self, phase_results: List[Any]) -> Dict[str, Any]:
        """Create success rate chart."""
        if not MATPLOTLIB_AVAILABLE:
            return {}
        
        successful = len([p for p in phase_results if getattr(p, 'success', False)])
        failed = len(phase_results) - successful
        
        plt.figure(figsize=(8, 8))
        plt.pie([successful, failed], labels=['Successful', 'Failed'], autopct='%1.1f%%')
        plt.title('Phase Success Rate')
        
        # Convert to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "image",
            "title": "Phase Success Rate",
            "data": f"data:image/png;base64,{image_base64}",
            "width": 400,
            "height": 400
        }
    
    def _create_performance_timeline_chart(self, phase_results: List[Any]) -> Dict[str, Any]:
        """Create performance timeline chart."""
        if not MATPLOTLIB_AVAILABLE:
            return {}
        
        phases = [getattr(p, 'phase_name', 'Unknown') for p in phase_results]
        start_times = [getattr(p, 'start_time', datetime.now()) for p in phase_results]
        durations = [getattr(p, 'duration', 0) for p in phase_results]
        
        # Convert to relative times
        if start_times:
            base_time = min(start_times)
            relative_times = [(t - base_time).total_seconds() for t in start_times]
        else:
            relative_times = []
        
        plt.figure(figsize=(12, 6))
        plt.barh(phases, durations, left=relative_times)
        plt.title('Performance Timeline')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Phase')
        plt.tight_layout()
        
        # Convert to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "image",
            "title": "Performance Timeline",
            "data": f"data:image/png;base64,{image_base64}",
            "width": 1000,
            "height": 500
        }
    
    def _create_phase_efficiency_chart(self, phase_results: List[Any]) -> Dict[str, Any]:
        """Create phase efficiency chart."""
        if not MATPLOTLIB_AVAILABLE:
            return {}
        
        phases = [getattr(p, 'phase_name', 'Unknown') for p in phase_results]
        durations = [getattr(p, 'duration', 0) for p in phase_results]
        successes = [getattr(p, 'success', False) for p in phase_results]
        
        # Calculate efficiency (success rate * speed factor)
        max_duration = max(durations) if durations else 1
        efficiencies = [
            (1 if success else 0) * (1 - duration / max_duration) 
            for success, duration in zip(successes, durations)
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(phases, efficiencies)
        plt.title('Phase Efficiency Analysis')
        plt.xlabel('Phase')
        plt.ylabel('Efficiency Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "image",
            "title": "Phase Efficiency Analysis",
            "data": f"data:image/png;base64,{image_base64}",
            "width": 800,
            "height": 400
        }
    
    def _export_report(self, report: AnalysisReport):
        """Export report in configured formats."""
        timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")
        base_filename = f"analysis_report_{timestamp}"
        
        for format_type in self.config.export_formats:
            if format_type == "html":
                self._export_html_report(report, base_filename)
            elif format_type == "json":
                self._export_json_report(report, base_filename)
            elif format_type == "markdown":
                self._export_markdown_report(report, base_filename)
            elif format_type == "pdf":
                self._export_pdf_report(report, base_filename)
    
    def _export_html_report(self, report: AnalysisReport, base_filename: str):
        """Export report as HTML."""
        html_content = self._generate_html_content(report)
        
        filename = f"{base_filename}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported to {filepath}")
    
    def _generate_html_content(self, report: AnalysisReport) -> str:
        """Generate HTML content for the report."""
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ report.title }}</title>
            <style>
                body { font-family: {{ config.font_family }}; margin: 40px; line-height: 1.6; }
                .header { border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
                .section { margin-bottom: 40px; }
                .analysis-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .analysis-table th, .analysis-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                .analysis-table th { background-color: #f2f2f2; }
                .phase-result { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .chart { text-align: center; margin: 20px 0; }
                .chart img { max-width: 100%; height: auto; }
                .recommendations ul { background-color: #e8f4f8; padding: 20px; border-radius: 5px; }
                .metadata { font-size: 0.9em; color: #666; margin-top: 30px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report.title }}</h1>
                <p><strong>Author:</strong> {{ report.author }}</p>
                <p><strong>Version:</strong> {{ report.version }}</p>
                <p><strong>Generated:</strong> {{ report.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            {% for section in report.sections %}
            <div class="section">
                {{ section.content|safe }}
                
                {% if section.charts %}
                {% for chart in section.charts %}
                <div class="chart">
                    <h4>{{ chart.title }}</h4>
                    <img src="{{ chart.data }}" alt="{{ chart.title }}" width="{{ chart.width }}" height="{{ chart.height }}">
                </div>
                {% endfor %}
                {% endif %}
            </div>
            {% endfor %}
            
            <div class="metadata">
                <p>Report generated by Simulation Analysis System v{{ report.version }}</p>
                <p>Generated at: {{ report.generated_at.isoformat() }}</p>
            </div>
        </body>
        </html>
        """
        
        if JINJA2_AVAILABLE and self.template_env:
            template = self.template_env.get_template("report_template.html")
            return template.render(report=report, config=self.config)
        else:
            # Simple template rendering
            return html_template.replace("{{ report.title }}", report.title) \
                              .replace("{{ report.author }}", report.author) \
                              .replace("{{ report.version }}", report.version) \
                              .replace("{{ report.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}", report.generated_at.strftime('%Y-%m-%d %H:%M:%S'))
    
    def _export_json_report(self, report: AnalysisReport, base_filename: str):
        """Export report as JSON."""
        filename = f"{base_filename}.json"
        filepath = self.output_dir / filename
        
        # Convert report to serializable format
        report_dict = {
            "title": report.title,
            "author": report.author,
            "version": report.version,
            "generated_at": report.generated_at.isoformat(),
            "summary": report.summary,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "charts": section.charts,
                    "tables": section.tables,
                    "metadata": section.metadata
                } for section in report.sections
            ],
            "raw_data": report.raw_data,
            "metadata": report.metadata
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report exported to {filepath}")
    
    def _export_markdown_report(self, report: AnalysisReport, base_filename: str):
        """Export report as Markdown."""
        filename = f"{base_filename}.md"
        filepath = self.output_dir / filename
        
        markdown_content = f"""# {report.title}

**Author:** {report.author}  
**Version:** {report.version}  
**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

---

"""
        
        for section in report.sections:
            markdown_content += f"## {section.title}\n\n"
            # Convert HTML to Markdown (basic conversion)
            content = section.content.replace('<h2>', '## ').replace('</h2>', '')
            content = content.replace('<h3>', '### ').replace('</h3>', '')
            content = content.replace('<p>', '').replace('</p>', '\n\n')
            content = content.replace('<ul>', '').replace('</ul>', '')
            content = content.replace('<li>', '- ').replace('</li>', '')
            content = content.replace('<strong>', '**').replace('</strong>', '**')
            markdown_content += content + "\n\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report exported to {filepath}")
    
    def _export_pdf_report(self, report: AnalysisReport, base_filename: str):
        """Export report as PDF (placeholder)."""
        # PDF export would require additional libraries like weasyprint or reportlab
        logger.warning("PDF export not implemented - requires additional dependencies")
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary of available reports."""
        reports = list(self.output_dir.glob("analysis_report_*.html"))
        
        return {
            "total_reports": len(reports),
            "reports": [
                {
                    "filename": report.name,
                    "created": datetime.fromtimestamp(report.stat().st_mtime).isoformat(),
                    "size_mb": report.stat().st_size / (1024 * 1024)
                } for report in reports
            ],
            "output_directory": str(self.output_dir)
        }