"""
File-based simulation comparison engine.

This module provides the main orchestrator for comparing two simulations
based on their file paths, integrating all comparison components.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile

from farm.analysis.comparative.simulation_loader import SimulationLoader
from farm.analysis.comparative.config_comparison import ConfigComparison
from farm.analysis.comparative.log_parser import LogParser
from farm.analysis.comparative.database_comparison import DatabaseComparison
from farm.analysis.comparative.metrics_loader import MetricsLoader
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult,
    ComparisonSummary,
    ConfigComparisonResult,
    DatabaseComparisonResult,
    LogComparisonResult,
    MetricsComparisonResult
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class FileComparisonEngine:
    """Main engine for file-based simulation comparison."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize comparison engine.
        
        Args:
            output_dir: Directory for temporary files and outputs.
                       If None, uses system temp directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileComparisonEngine initialized with output directory: {self.output_dir}")
    
    def compare_simulations(self, 
                          sim1_path: Union[str, Path], 
                          sim2_path: Union[str, Path],
                          analysis_modules: Optional[List[str]] = None,
                          include_logs: bool = True,
                          include_metrics: bool = True) -> SimulationComparisonResult:
        """Compare two simulations comprehensively.
        
        Args:
            sim1_path: Path to first simulation directory
            sim2_path: Path to second simulation directory
            analysis_modules: List of analysis modules to run for metrics
            include_logs: Whether to include log file comparison
            include_metrics: Whether to include comprehensive metrics comparison
        
        Returns:
            SimulationComparisonResult containing all comparison data
        """
        logger.info(f"Starting comparison: {sim1_path} vs {sim2_path}")
        
        # Load simulation data
        sim1_data = self._load_simulation_data(sim1_path)
        sim2_data = self._load_simulation_data(sim2_path)
        
        # Perform comparisons
        config_comparison = self._compare_configurations(sim1_data, sim2_data)
        database_comparison = self._compare_databases(sim1_data, sim2_data)
        log_comparison = self._compare_logs(sim1_data, sim2_data) if include_logs else LogComparisonResult()
        metrics_comparison = self._compare_metrics(sim1_data, sim2_data, analysis_modules) if include_metrics else MetricsComparisonResult()
        
        # Create summary
        summary = self._create_comparison_summary(
            config_comparison, database_comparison, log_comparison, metrics_comparison
        )
        
        # Create result
        result = SimulationComparisonResult(
            simulation1_path=Path(sim1_path),
            simulation2_path=Path(sim2_path),
            comparison_summary=summary,
            config_comparison=config_comparison,
            database_comparison=database_comparison,
            log_comparison=log_comparison,
            metrics_comparison=metrics_comparison,
            metadata=self._create_metadata(sim1_data, sim2_data)
        )
        
        logger.info(f"Comparison completed: {summary.total_differences} total differences found")
        return result
    
    def _load_simulation_data(self, sim_path: Union[str, Path]) -> Any:
        """Load simulation data from path."""
        try:
            loader = SimulationLoader(sim_path)
            return loader.load_simulation_data()
        except Exception as e:
            logger.error(f"Failed to load simulation data from {sim_path}: {e}")
            raise
    
    def _compare_configurations(self, sim1_data: Any, sim2_data: Any) -> ConfigComparisonResult:
        """Compare simulation configurations."""
        logger.debug("Comparing configurations")
        
        try:
            comparison = ConfigComparison()
            differences = comparison.compare_configurations(
                sim1_data.config, sim2_data.config
            )
            
            # Get significant changes
            significant_changes = comparison.get_significant_changes(differences)
            
            # Format output
            formatted_output = comparison.format_config_differences(differences)
            
            return ConfigComparisonResult(
                differences=differences.get('differences', {}),
                summary=differences.get('summary', {}),
                significant_changes=significant_changes.get('differences', {}).keys() if significant_changes.get('differences') else [],
                formatted_output=formatted_output
            )
            
        except Exception as e:
            logger.error(f"Error comparing configurations: {e}")
            return ConfigComparisonResult(
                differences={'error': str(e)},
                summary={'error': str(e)},
                significant_changes=[],
                formatted_output=f"Error comparing configurations: {e}"
            )
    
    def _compare_databases(self, sim1_data: Any, sim2_data: Any) -> DatabaseComparisonResult:
        """Compare simulation databases."""
        logger.debug("Comparing databases")
        
        try:
            # Check if both simulations have databases
            if not sim1_data.metadata.get('database_exists', False):
                logger.warning("Simulation 1 has no database")
                return DatabaseComparisonResult()
            
            if not sim2_data.metadata.get('database_exists', False):
                logger.warning("Simulation 2 has no database")
                return DatabaseComparisonResult()
            
            # Get database paths
            db1_path = sim1_data.simulation_path / "simulation.db"
            db2_path = sim2_data.simulation_path / "simulation.db"
            
            # Compare databases
            db_comparison = DatabaseComparison(db1_path, db2_path)
            result = db_comparison.compare_databases()
            
            return DatabaseComparisonResult(
                schema_differences=result.schema_differences,
                data_differences=result.data_differences,
                metric_differences=result.metric_differences,
                summary=result.summary
            )
            
        except Exception as e:
            logger.error(f"Error comparing databases: {e}")
            return DatabaseComparisonResult(
                schema_differences={'error': str(e)},
                data_differences={'error': str(e)},
                metric_differences={'error': str(e)},
                summary={'error': str(e)}
            )
    
    def _compare_logs(self, sim1_data: Any, sim2_data: Any) -> LogComparisonResult:
        """Compare simulation log files."""
        logger.debug("Comparing log files")
        
        try:
            parser = LogParser()
            
            # Parse logs from both simulations
            sim1_logs = [Path(p) for p in sim1_data.metadata.get('log_files', [])]
            sim2_logs = [Path(p) for p in sim2_data.metadata.get('log_files', [])]
            
            if not sim1_logs and not sim2_logs:
                logger.warning("No log files found in either simulation")
                return LogComparisonResult()
            
            # Parse performance metrics
            sim1_perf = parser.parse_performance_metrics(sim1_logs) if sim1_logs else {}
            sim2_perf = parser.parse_performance_metrics(sim2_logs) if sim2_logs else {}
            
            # Parse error metrics
            sim1_errors = parser.parse_error_metrics(sim1_logs) if sim1_logs else {}
            sim2_errors = parser.parse_error_metrics(sim2_logs) if sim2_logs else {}
            
            # Compare performance metrics
            performance_differences = self._compare_performance_metrics(sim1_perf, sim2_perf)
            
            # Compare error metrics
            error_differences = self._compare_error_metrics(sim1_errors, sim2_errors)
            
            # Create summary
            summary = {
                'sim1_log_files': len(sim1_logs),
                'sim2_log_files': len(sim2_logs),
                'performance_differences': len(performance_differences),
                'error_differences': len(error_differences)
            }
            
            return LogComparisonResult(
                performance_differences=performance_differences,
                error_differences=error_differences,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error comparing logs: {e}")
            return LogComparisonResult(
                performance_differences={'error': str(e)},
                error_differences={'error': str(e)},
                summary={'error': str(e)}
            )
    
    def _compare_metrics(self, sim1_data: Any, sim2_data: Any, analysis_modules: Optional[List[str]] = None) -> MetricsComparisonResult:
        """Compare simulation metrics."""
        logger.debug("Comparing metrics")
        
        try:
            # Load metrics from both simulations
            metrics_loader1 = MetricsLoader(sim1_data.simulation_path)
            metrics_loader2 = MetricsLoader(sim2_data.simulation_path)
            
            # Try comprehensive metrics first, fall back to basic if needed
            try:
                sim1_metrics = metrics_loader1.load_comprehensive_metrics(analysis_modules)
                sim2_metrics = metrics_loader2.load_comprehensive_metrics(analysis_modules)
            except Exception as e:
                logger.warning(f"Comprehensive metrics failed, using basic metrics: {e}")
                sim1_metrics = metrics_loader1.load_basic_metrics()
                sim2_metrics = metrics_loader2.load_basic_metrics()
            
            # Compare metrics
            metric_differences = self._compare_metric_values(sim1_metrics.metrics, sim2_metrics.metrics)
            
            # Compare performance
            performance_comparison = self._compare_metric_performance(sim1_metrics.metrics, sim2_metrics.metrics)
            
            # Create summary
            summary = {
                'sim1_metrics_count': len(sim1_metrics.metrics),
                'sim2_metrics_count': len(sim2_metrics.metrics),
                'common_metrics': len(set(sim1_metrics.metrics.keys()) & set(sim2_metrics.metrics.keys())),
                'differences_found': len(metric_differences)
            }
            
            return MetricsComparisonResult(
                metric_differences=metric_differences,
                performance_comparison=performance_comparison,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error comparing metrics: {e}")
            return MetricsComparisonResult(
                metric_differences={'error': str(e)},
                performance_comparison={'error': str(e)},
                summary={'error': str(e)}
            )
    
    def _compare_performance_metrics(self, sim1_perf: Dict[str, Any], sim2_perf: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance metrics between simulations."""
        differences = {}
        
        # Compare aggregated metrics
        sim1_agg = sim1_perf.get('aggregated_metrics', {})
        sim2_agg = sim2_perf.get('aggregated_metrics', {})
        
        for metric in set(sim1_agg.keys()) | set(sim2_agg.keys()):
            val1 = sim1_agg.get(metric)
            val2 = sim2_agg.get(metric)
            
            if val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
                    differences[metric] = {
                        'sim1_value': val1,
                        'sim2_value': val2,
                        'difference': diff,
                        'percentage_change': pct_change
                    }
                else:
                    differences[metric] = {
                        'sim1_value': val1,
                        'sim2_value': val2,
                        'type': 'non_numeric'
                    }
        
        return differences
    
    def _compare_error_metrics(self, sim1_errors: Dict[str, Any], sim2_errors: Dict[str, Any]) -> Dict[str, Any]:
        """Compare error metrics between simulations."""
        differences = {}
        
        # Compare error counts
        sim1_counts = sim1_errors.get('total_error_counts', {})
        sim2_counts = sim2_errors.get('total_error_counts', {})
        
        for error_type in set(sim1_counts.keys()) | set(sim2_counts.keys()):
            count1 = sim1_counts.get(error_type, 0)
            count2 = sim2_counts.get(error_type, 0)
            
            if count1 != count2:
                differences[error_type] = {
                    'sim1_count': count1,
                    'sim2_count': count2,
                    'difference': count2 - count1
                }
        
        return differences
    
    def _compare_metric_values(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metric values between simulations."""
        differences = {}
        
        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            val1 = metrics1.get(metric)
            val2 = metrics2.get(metric)
            
            if val1 != val2:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
                    differences[metric] = {
                        'sim1_value': val1,
                        'sim2_value': val2,
                        'difference': diff,
                        'percentage_change': pct_change
                    }
                else:
                    differences[metric] = {
                        'sim1_value': val1,
                        'sim2_value': val2,
                        'type': 'non_numeric'
                    }
        
        return differences
    
    def _compare_metric_performance(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance characteristics of metrics."""
        performance = {}
        
        # Find performance-related metrics
        perf_metrics = [k for k in metrics1.keys() if any(p in k.lower() for p in ['time', 'duration', 'speed', 'throughput', 'rate'])]
        
        for metric in perf_metrics:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 > 0 and val2 > 0:
                        ratio = val2 / val1
                        performance[metric] = {
                            'sim1_value': val1,
                            'sim2_value': val2,
                            'ratio': ratio,
                            'faster': 'sim2' if ratio > 1 else 'sim1' if ratio < 1 else 'equal'
                        }
        
        return performance
    
    def _create_comparison_summary(self, 
                                 config_comp: ConfigComparisonResult,
                                 database_comp: DatabaseComparisonResult,
                                 log_comp: LogComparisonResult,
                                 metrics_comp: MetricsComparisonResult) -> ComparisonSummary:
        """Create overall comparison summary."""
        return ComparisonSummary(
            config_differences=len(config_comp.differences),
            database_differences=len(database_comp.schema_differences) + len(database_comp.data_differences),
            log_differences=len(log_comp.performance_differences) + len(log_comp.error_differences),
            metrics_differences=len(metrics_comp.metric_differences)
        )
    
    def _create_metadata(self, sim1_data: Any, sim2_data: Any) -> Dict[str, Any]:
        """Create metadata about the comparison process."""
        return {
            'sim1_metadata': sim1_data.metadata,
            'sim2_metadata': sim2_data.metadata,
            'comparison_engine': 'FileComparisonEngine',
            'output_directory': str(self.output_dir)
        }