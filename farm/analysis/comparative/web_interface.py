"""
Web interface for simulation comparison.

This module provides a web-based interface for interactive simulation
comparison and analysis using Flask.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from farm.analysis.comparative.file_comparison_engine import FileComparisonEngine
from farm.analysis.comparative.visualization_engine import VisualizationEngine
from farm.analysis.comparative.statistical_analyzer import StatisticalAnalyzer
from farm.analysis.comparative.report_generator import ReportGenerator
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class SimulationComparisonWebApp:
    """Web application for simulation comparison."""
    
    def __init__(self, 
                 upload_dir: str = "uploads",
                 results_dir: str = "web_results",
                 host: str = "0.0.0.0",
                 port: int = 5000,
                 debug: bool = False):
        """Initialize the web application.
        
        Args:
            upload_dir: Directory for uploaded files
            results_dir: Directory for comparison results
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for web interface. Install with: pip install flask")
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'simulation-comparison-secret-key'
        self.app.config['UPLOAD_FOLDER'] = upload_dir
        self.app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
        
        self.upload_dir = Path(upload_dir)
        self.results_dir = Path(results_dir)
        self.host = host
        self.port = port
        self.debug = debug
        
        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.comparison_engine = FileComparisonEngine(output_dir=self.results_dir)
        self.viz_engine = VisualizationEngine(output_dir=self.results_dir / "visualizations")
        self.statistical_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator(output_dir=self.results_dir / "reports")
        
        # Store active comparisons
        self.active_comparisons: Dict[str, Dict[str, Any]] = {}
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Web interface initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return render_template('index.html')
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_simulation():
            """Upload simulation files."""
            try:
                if 'simulation1' not in request.files or 'simulation2' not in request.files:
                    return jsonify({'error': 'Both simulation files are required'}), 400
                
                sim1_file = request.files['simulation1']
                sim2_file = request.files['simulation2']
                
                if sim1_file.filename == '' or sim2_file.filename == '':
                    return jsonify({'error': 'No files selected'}), 400
                
                # Create unique directories for this comparison
                comparison_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                sim1_dir = self.upload_dir / comparison_id / "simulation1"
                sim2_dir = self.upload_dir / comparison_id / "simulation2"
                sim1_dir.mkdir(parents=True, exist_ok=True)
                sim2_dir.mkdir(parents=True, exist_ok=True)
                
                # Save uploaded files
                sim1_file.save(sim1_dir / secure_filename(sim1_file.filename))
                sim2_file.save(sim2_dir / secure_filename(sim2_file.filename))
                
                return jsonify({
                    'success': True,
                    'comparison_id': comparison_id,
                    'sim1_path': str(sim1_dir),
                    'sim2_path': str(sim2_dir)
                })
                
            except Exception as e:
                logger.error(f"Error uploading files: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/compare', methods=['POST'])
        def compare_simulations():
            """Compare two simulations."""
            try:
                data = request.get_json()
                sim1_path = data.get('sim1_path')
                sim2_path = data.get('sim2_path')
                comparison_id = data.get('comparison_id')
                options = data.get('options', {})
                
                if not sim1_path or not sim2_path:
                    return jsonify({'error': 'Simulation paths are required'}), 400
                
                # Run comparison
                result = self.comparison_engine.compare_simulations(
                    sim1_path=sim1_path,
                    sim2_path=sim2_path,
                    include_logs=options.get('include_logs', True),
                    include_metrics=options.get('include_metrics', True),
                    analysis_modules=options.get('analysis_modules')
                )
                
                # Store result
                self.active_comparisons[comparison_id] = {
                    'result': result,
                    'sim1_path': sim1_path,
                    'sim2_path': sim2_path,
                    'created_at': datetime.now().isoformat()
                }
                
                # Generate visualizations
                visualization_files = self.viz_engine.create_comparison_dashboard(result)
                
                # Perform statistical analysis
                statistical_analysis = self.statistical_analyzer.analyze_comparison(result)
                
                return jsonify({
                    'success': True,
                    'comparison_id': comparison_id,
                    'summary': {
                        'total_differences': result.comparison_summary.total_differences,
                        'severity': result.comparison_summary.severity,
                        'config_differences': result.comparison_summary.config_differences,
                        'database_differences': result.comparison_summary.database_differences,
                        'log_differences': result.comparison_summary.log_differences,
                        'metrics_differences': result.comparison_summary.metrics_differences
                    },
                    'visualization_files': visualization_files,
                    'statistical_analysis': {
                        'correlation_analysis': statistical_analysis.correlation_analysis,
                        'significance_tests': statistical_analysis.significance_tests,
                        'trend_analysis': statistical_analysis.trend_analysis,
                        'anomaly_detection': statistical_analysis.anomaly_detection,
                        'summary': statistical_analysis.summary
                    }
                })
                
            except Exception as e:
                logger.error(f"Error comparing simulations: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/comparison/<comparison_id>')
        def get_comparison(comparison_id):
            """Get comparison results."""
            if comparison_id not in self.active_comparisons:
                return jsonify({'error': 'Comparison not found'}), 404
            
            comparison = self.active_comparisons[comparison_id]
            result = comparison['result']
            
            return jsonify({
                'comparison_id': comparison_id,
                'sim1_path': comparison['sim1_path'],
                'sim2_path': comparison['sim2_path'],
                'created_at': comparison['created_at'],
                'summary': {
                    'total_differences': result.comparison_summary.total_differences,
                    'severity': result.comparison_summary.severity,
                    'config_differences': result.comparison_summary.config_differences,
                    'database_differences': result.comparison_summary.database_differences,
                    'log_differences': result.comparison_summary.log_differences,
                    'metrics_differences': result.comparison_summary.metrics_differences
                },
                'config_comparison': {
                    'differences': result.config_comparison.differences
                },
                'database_comparison': {
                    'schema_differences': result.database_comparison.schema_differences,
                    'data_differences': result.database_comparison.data_differences,
                    'metric_differences': result.database_comparison.metric_differences
                },
                'log_comparison': {
                    'performance_differences': result.log_comparison.performance_differences,
                    'error_differences': result.log_comparison.error_differences
                },
                'metrics_comparison': {
                    'metric_differences': result.metrics_comparison.metric_differences,
                    'performance_comparison': result.metrics_comparison.performance_comparison
                }
            })
        
        @self.app.route('/api/comparisons')
        def list_comparisons():
            """List all active comparisons."""
            comparisons = []
            for comp_id, comp_data in self.active_comparisons.items():
                result = comp_data['result']
                comparisons.append({
                    'comparison_id': comp_id,
                    'sim1_path': comp_data['sim1_path'],
                    'sim2_path': comp_data['sim2_path'],
                    'created_at': comp_data['created_at'],
                    'total_differences': result.comparison_summary.total_differences,
                    'severity': result.comparison_summary.severity
                })
            
            return jsonify({'comparisons': comparisons})
        
        @self.app.route('/api/visualize/<comparison_id>')
        def generate_visualizations(comparison_id):
            """Generate visualizations for a comparison."""
            if comparison_id not in self.active_comparisons:
                return jsonify({'error': 'Comparison not found'}), 404
            
            result = self.active_comparisons[comparison_id]['result']
            visualization_files = self.viz_engine.create_comparison_dashboard(result)
            
            return jsonify({
                'success': True,
                'visualization_files': visualization_files
            })
        
        @self.app.route('/api/report/<comparison_id>')
        def generate_report(comparison_id):
            """Generate report for a comparison."""
            if comparison_id not in self.active_comparisons:
                return jsonify({'error': 'Comparison not found'}), 404
            
            result = self.active_comparisons[comparison_id]['result']
            report_format = request.args.get('format', 'html')
            
            # Perform statistical analysis
            statistical_analysis = self.statistical_analyzer.analyze_comparison(result)
            
            # Generate report
            report_files = self.report_generator.generate_comprehensive_report(
                result=result,
                statistical_analysis=statistical_analysis,
                report_format=report_format
            )
            
            return jsonify({
                'success': True,
                'report_files': report_files
            })
        
        @self.app.route('/api/download/<path:filename>')
        def download_file(filename):
            """Download a file."""
            try:
                return send_from_directory(self.results_dir, filename, as_attachment=True)
            except FileNotFoundError:
                return jsonify({'error': 'File not found'}), 404
        
        @self.app.route('/api/visualizations/<path:filename>')
        def serve_visualization(filename):
            """Serve visualization files."""
            try:
                return send_from_directory(self.viz_engine.output_dir, filename)
            except FileNotFoundError:
                return jsonify({'error': 'Visualization not found'}), 404
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'active_comparisons': len(self.active_comparisons)
            })
    
    def run(self):
        """Run the web application."""
        logger.info(f"Starting web interface on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)


def create_web_app(upload_dir: str = "uploads",
                  results_dir: str = "web_results",
                  host: str = "0.0.0.0",
                  port: int = 5000,
                  debug: bool = False) -> SimulationComparisonWebApp:
    """Create and configure the web application.
    
    Args:
        upload_dir: Directory for uploaded files
        results_dir: Directory for comparison results
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        
    Returns:
        Configured web application
    """
    return SimulationComparisonWebApp(
        upload_dir=upload_dir,
        results_dir=results_dir,
        host=host,
        port=port,
        debug=debug
    )


def main():
    """Main entry point for web interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start simulation comparison web interface")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--upload-dir', default='uploads', help='Upload directory')
    parser.add_argument('--results-dir', default='web_results', help='Results directory')
    
    args = parser.parse_args()
    
    try:
        app = create_web_app(
            upload_dir=args.upload_dir,
            results_dir=args.results_dir,
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        app.run()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install Flask to use the web interface:")
        print("pip install flask")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting web interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()