"""
Command-line interface for simulation comparison.

This module provides a user-friendly CLI for running simulation comparisons
and generating reports.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from farm.analysis.comparative.file_comparison_engine import FileComparisonEngine
from farm.analysis.comparative.visualization_engine import VisualizationEngine
from farm.analysis.comparative.statistical_analyzer import StatisticalAnalyzer
from farm.analysis.comparative.report_generator import ReportGenerator
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Run the comparison
        result = run_comparison(args)
        
        # Generate visualizations if requested
        visualization_files = {}
        if args.visualize:
            visualization_files = generate_visualizations(result, args)
        
        # Perform statistical analysis if requested
        statistical_analysis = None
        if args.statistical:
            statistical_analysis = perform_statistical_analysis(result, args)
        
        # Generate reports
        generate_reports(result, statistical_analysis, visualization_files, args)
        
        # Print summary
        print_summary(result, args)
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare two simulations and generate comprehensive reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python -m farm.analysis.comparative.cli_interface sim1/ sim2/
  
  # Full analysis with visualizations and reports
  python -m farm.analysis.comparative.cli_interface sim1/ sim2/ --visualize --statistical --report html
  
  # Compare specific aspects only
  python -m farm.analysis.comparative.cli_interface sim1/ sim2/ --no-logs --no-metrics
  
  # Custom output directory
  python -m farm.analysis.comparative.cli_interface sim1/ sim2/ --output-dir results/
        """
    )
    
    # Required arguments
    parser.add_argument(
        'simulation1',
        type=str,
        help='Path to first simulation directory'
    )
    parser.add_argument(
        'simulation2',
        type=str,
        help='Path to second simulation directory'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='comparison_results',
        help='Output directory for results (default: comparison_results)'
    )
    
    # Comparison options
    parser.add_argument(
        '--no-logs',
        action='store_true',
        help='Skip log analysis'
    )
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Skip metrics analysis'
    )
    parser.add_argument(
        '--analysis-modules',
        nargs='+',
        help='Specific analysis modules to run for metrics'
    )
    
    # Visualization options
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--viz-style',
        choices=['default', 'seaborn', 'ggplot', 'bmh', 'fivethirtyeight'],
        default='default',
        help='Matplotlib style for visualizations (default: default)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Generate interactive plots (requires plotly)'
    )
    
    # Statistical analysis options
    parser.add_argument(
        '--statistical', '-s',
        action='store_true',
        help='Perform statistical analysis'
    )
    parser.add_argument(
        '--significance-level',
        type=float,
        default=0.05,
        help='Significance level for statistical tests (default: 0.05)'
    )
    
    # Report options
    parser.add_argument(
        '--report', '-r',
        nargs='+',
        choices=['html', 'pdf', 'text', 'json', 'yaml', 'all'],
        default=['text'],
        help='Report formats to generate (default: text)'
    )
    parser.add_argument(
        '--template-dir',
        type=str,
        help='Directory containing custom report templates'
    )
    
    # Output options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--verbose', '-V',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def run_comparison(args) -> 'SimulationComparisonResult':
    """Run the simulation comparison."""
    if not args.quiet:
        print("Starting simulation comparison...")
        print(f"Simulation 1: {args.simulation1}")
        print(f"Simulation 2: {args.simulation2}")
        print()
    
    # Initialize comparison engine
    engine = FileComparisonEngine(output_dir=args.output_dir)
    
    # Run comparison
    result = engine.compare_simulations(
        sim1_path=args.simulation1,
        sim2_path=args.simulation2,
        analysis_modules=args.analysis_modules,
        include_logs=not args.no_logs,
        include_metrics=not args.no_metrics
    )
    
    if not args.quiet:
        print(f"Comparison completed: {result.comparison_summary.total_differences} differences found")
        print(f"Severity: {result.comparison_summary.severity.upper()}")
        print()
    
    return result


def generate_visualizations(result: 'SimulationComparisonResult', args) -> dict:
    """Generate visualizations."""
    if not args.quiet:
        print("Generating visualizations...")
    
    # Initialize visualization engine
    viz_engine = VisualizationEngine(
        output_dir=Path(args.output_dir) / "visualizations",
        style=args.viz_style
    )
    
    # Generate dashboard
    visualization_files = viz_engine.create_comparison_dashboard(result)
    
    # Generate interactive plot if requested
    if args.interactive:
        interactive_file = viz_engine.create_interactive_plot(result)
        if interactive_file:
            visualization_files['interactive'] = interactive_file
    
    # Export data for external tools
    export_files = viz_engine.export_data_for_external_tools(result)
    visualization_files.update(export_files)
    
    if not args.quiet:
        print(f"Visualizations saved to {viz_engine.output_dir}")
        print()
    
    return visualization_files


def perform_statistical_analysis(result: 'SimulationComparisonResult', args) -> 'StatisticalAnalysisResult':
    """Perform statistical analysis."""
    if not args.quiet:
        print("Performing statistical analysis...")
    
    # Initialize statistical analyzer
    analyzer = StatisticalAnalyzer(significance_level=args.significance_level)
    
    # Run analysis
    analysis_result = analyzer.analyze_comparison(result)
    
    # Export results
    export_path = Path(args.output_dir) / "statistical_analysis.json"
    analyzer.export_analysis_results(analysis_result, export_path)
    
    if not args.quiet:
        print(f"Statistical analysis completed")
        print(f"Results saved to {export_path}")
        print()
    
    return analysis_result


def generate_reports(result: 'SimulationComparisonResult', 
                    statistical_analysis: Optional['StatisticalAnalysisResult'],
                    visualization_files: dict,
                    args) -> None:
    """Generate reports."""
    if not args.quiet:
        print("Generating reports...")
    
    # Initialize report generator
    report_generator = ReportGenerator(
        output_dir=Path(args.output_dir) / "reports",
        template_dir=args.template_dir
    )
    
    # Generate reports
    report_formats = args.report
    if 'all' in report_formats:
        report_formats = ['html', 'pdf', 'text', 'json', 'yaml']
    
    report_files = {}
    for format_type in report_formats:
        if not args.quiet:
            print(f"  Generating {format_type.upper()} report...")
        
        files = report_generator.generate_comprehensive_report(
            result=result,
            statistical_analysis=statistical_analysis,
            visualization_files=visualization_files,
            report_format=format_type
        )
        report_files.update(files)
    
    if not args.quiet:
        print(f"Reports saved to {report_generator.output_dir}")
        print()


def print_summary(result: 'SimulationComparisonResult', args) -> None:
    """Print a summary of the comparison results."""
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Simulation 1: {result.simulation1_path}")
    print(f"Simulation 2: {result.simulation2_path}")
    print(f"Comparison Time: {result.comparison_summary.comparison_time}")
    print()
    
    print("DIFFERENCES FOUND:")
    print(f"  Total: {result.comparison_summary.total_differences}")
    print(f"  Severity: {result.comparison_summary.severity.upper()}")
    print(f"  Configuration: {result.comparison_summary.config_differences}")
    print(f"  Database: {result.comparison_summary.database_differences}")
    print(f"  Logs: {result.comparison_summary.log_differences}")
    print(f"  Metrics: {result.comparison_summary.metrics_differences}")
    print()
    
    # Print detailed differences if verbose
    if args.verbose:
        print_detailed_differences(result)
    
    print("OUTPUT FILES:")
    print(f"  Results directory: {args.output_dir}")
    if args.visualize:
        print(f"  Visualizations: {args.output_dir}/visualizations/")
    if args.statistical:
        print(f"  Statistical analysis: {args.output_dir}/statistical_analysis.json")
    print(f"  Reports: {args.output_dir}/reports/")
    print()


def print_detailed_differences(result: 'SimulationComparisonResult') -> None:
    """Print detailed information about differences."""
    print("DETAILED DIFFERENCES:")
    print("-" * 40)
    
    # Configuration differences
    if result.config_comparison.differences:
        print("Configuration Differences:")
        for key, value in result.config_comparison.differences.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} differences")
            elif isinstance(value, dict):
                print(f"  {key}: {len(value)} differences")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Database differences
    if result.database_comparison.schema_differences:
        print("Database Schema Differences:")
        for table, changes in result.database_comparison.schema_differences.items():
            if isinstance(changes, dict):
                print(f"  {table}: {len(changes)} changes")
            elif isinstance(changes, list):
                print(f"  {table}: {len(changes)} changes")
            else:
                print(f"  {table}: 1 change")
        print()
    
    if result.database_comparison.data_differences:
        print("Database Data Differences:")
        for table, info in result.database_comparison.data_differences.items():
            if isinstance(info, dict) and 'difference' in info:
                print(f"  {table}: {info['difference']} row difference")
            else:
                print(f"  {table}: differences found")
        print()
    
    # Log differences
    if result.log_comparison.performance_differences:
        print("Log Performance Differences:")
        for metric, diff in result.log_comparison.performance_differences.items():
            if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                print(f"  {metric}: Sim1={diff['sim1_value']}, Sim2={diff['sim2_value']}")
            else:
                print(f"  {metric}: differences found")
        print()
    
    if result.log_comparison.error_differences:
        print("Log Error Differences:")
        for error_type, diff in result.log_comparison.error_differences.items():
            if isinstance(diff, dict) and 'sim1_count' in diff and 'sim2_count' in diff:
                print(f"  {error_type}: Sim1={diff['sim1_count']}, Sim2={diff['sim2_count']}")
            else:
                print(f"  {error_type}: differences found")
        print()
    
    # Metrics differences
    if result.metrics_comparison.metric_differences:
        print("Metrics Differences:")
        for metric, diff in result.metrics_comparison.metric_differences.items():
            if isinstance(diff, dict) and 'sim1_value' in diff and 'sim2_value' in diff:
                pct_change = diff.get('percentage_change', 0)
                print(f"  {metric}: Sim1={diff['sim1_value']}, Sim2={diff['sim2_value']}, Change={pct_change:.1f}%")
            else:
                print(f"  {metric}: differences found")
        print()
    
    if result.metrics_comparison.performance_comparison:
        print("Performance Comparison:")
        for metric, comp in result.metrics_comparison.performance_comparison.items():
            if isinstance(comp, dict) and 'ratio' in comp:
                print(f"  {metric}: Ratio={comp['ratio']:.2f}, Faster={comp.get('faster', 'equal')}")
            else:
                print(f"  {metric}: comparison available")
        print()


def validate_arguments(args) -> None:
    """Validate command line arguments."""
    # Check if simulation directories exist
    sim1_path = Path(args.simulation1)
    sim2_path = Path(args.simulation2)
    
    if not sim1_path.exists():
        raise FileNotFoundError(f"Simulation 1 directory not found: {args.simulation1}")
    
    if not sim2_path.exists():
        raise FileNotFoundError(f"Simulation 2 directory not found: {args.simulation2}")
    
    if not sim1_path.is_dir():
        raise NotADirectoryError(f"Simulation 1 path is not a directory: {args.simulation1}")
    
    if not sim2_path.is_dir():
        raise NotADirectoryError(f"Simulation 2 path is not a directory: {args.simulation2}")
    
    # Check significance level
    if not 0 < args.significance_level < 1:
        raise ValueError(f"Significance level must be between 0 and 1, got: {args.significance_level}")
    
    # Check output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()