"""
Metrics loader using AnalysisService for comprehensive simulation analysis.

This module provides functionality to load comprehensive metrics from
simulation data using the existing AnalysisService framework.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from farm.analysis.service import AnalysisService, AnalysisRequest
from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsData:
    """Container for loaded metrics data."""
    
    simulation_path: Path
    metrics: Dict[str, Any]
    analysis_results: Dict[str, Any]
    metadata: Dict[str, Any]


class MetricsLoader:
    """Loads comprehensive metrics using AnalysisService."""
    
    def __init__(self, simulation_path: Union[str, Path]):
        """Initialize metrics loader.
        
        Args:
            simulation_path: Path to simulation directory
        """
        self.simulation_path = Path(simulation_path)
        
        # Try to initialize AnalysisService, but don't fail if it's not available
        try:
            self.analysis_service = AnalysisService()
        except Exception as e:
            logger.warning(f"Could not initialize AnalysisService: {e}")
            self.analysis_service = None
        
        # Validate simulation directory
        if not self.simulation_path.exists():
            raise FileNotFoundError(f"Simulation directory not found: {self.simulation_path}")
        
        if not self.simulation_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.simulation_path}")
    
    def load_comprehensive_metrics(self, analysis_modules: Optional[List[str]] = None) -> MetricsData:
        """Load comprehensive metrics using AnalysisService.
        
        Args:
            analysis_modules: List of specific analysis modules to run.
                            If None, runs all available modules.
        
        Returns:
            MetricsData containing all loaded metrics
        """
        logger.info(f"Loading comprehensive metrics from {self.simulation_path}")
        
        # If AnalysisService is not available, fall back to basic metrics
        if self.analysis_service is None:
            logger.warning("AnalysisService not available, falling back to basic metrics")
            return self.load_basic_metrics()
        
        # Determine which modules to run
        if analysis_modules is None:
            analysis_modules = self._get_available_analysis_modules()
        
        # Run analysis modules
        analysis_results = {}
        for module_name in analysis_modules:
            try:
                result = self._run_analysis_module(module_name)
                if result:
                    analysis_results[module_name] = result
            except Exception as e:
                logger.warning(f"Failed to run analysis module {module_name}: {e}")
                analysis_results[module_name] = {'error': str(e)}
        
        # Extract metrics from analysis results
        metrics = self._extract_metrics_from_results(analysis_results)
        
        # Create metadata
        metadata = self._create_metrics_metadata(analysis_modules, analysis_results)
        
        return MetricsData(
            simulation_path=self.simulation_path,
            metrics=metrics,
            analysis_results=analysis_results,
            metadata=metadata
        )
    
    def _get_available_analysis_modules(self) -> List[str]:
        """Get list of available analysis modules."""
        # Common analysis modules that are likely to be available
        common_modules = [
            'actions',
            'advantage',
            'agents',
            'combat',
            'dominance',
            'genesis',
            'learning',
            'population',
            'resources',
            'social_behavior',
            'spatial',
            'temporal'
        ]
        
        # Try to determine which modules are actually available
        available_modules = []
        for module in common_modules:
            try:
                # Try to import the module to check if it exists
                module_path = f"farm.analysis.{module}"
                __import__(module_path)
                available_modules.append(module)
            except ImportError:
                logger.debug(f"Analysis module {module} not available")
                continue
        
        if not available_modules:
            logger.warning("No analysis modules found, using basic modules")
            available_modules = ['actions', 'agents', 'temporal']
        
        return available_modules
    
    def _run_analysis_module(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Run a specific analysis module."""
        if self.analysis_service is None:
            logger.warning(f"AnalysisService not available, cannot run module {module_name}")
            return None
            
        try:
            # Create analysis request - try different parameter combinations
            try:
                request = AnalysisRequest(
                    experiment_path=str(self.simulation_path),
                    analysis_module=module_name,
                    output_path=str(self.simulation_path / "analysis_output"),
                    force_recompute=False
                )
            except TypeError:
                # Try with different parameter names
                try:
                    request = AnalysisRequest(
                        experiment_path=str(self.simulation_path),
                        module_name=module_name,
                        output_path=str(self.simulation_path / "analysis_output"),
                        force_recompute=False
                    )
                except TypeError:
                    # Try with minimal parameters
                    request = AnalysisRequest(
                        experiment_path=str(self.simulation_path),
                        output_path=str(self.simulation_path / "analysis_output")
                    )
            
            # Run analysis
            result = self.analysis_service.run_analysis(request)
            
            if result and result.success:
                return {
                    'success': True,
                    'data': result.data,
                    'metrics': result.metrics,
                    'metadata': result.metadata
                }
            else:
                logger.warning(f"Analysis module {module_name} did not succeed")
                return None
                
        except Exception as e:
            logger.error(f"Error running analysis module {module_name}: {e}")
            return None
    
    def _extract_metrics_from_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from analysis results."""
        metrics = {}
        
        for module_name, result in analysis_results.items():
            if result and 'metrics' in result:
                module_metrics = result['metrics']
                
                # Flatten metrics with module prefix
                for metric_name, metric_value in module_metrics.items():
                    full_name = f"{module_name}.{metric_name}"
                    metrics[full_name] = metric_value
            
            # Also extract data-based metrics
            if result and 'data' in result:
                data = result['data']
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            full_name = f"{module_name}.data.{key}"
                            metrics[full_name] = value
        
        return metrics
    
    def _create_metrics_metadata(self, analysis_modules: List[str], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata about the metrics loading process."""
        successful_modules = [name for name, result in analysis_results.items() 
                            if result and result.get('success', False)]
        failed_modules = [name for name, result in analysis_results.items() 
                        if not result or not result.get('success', False)]
        
        return {
            'simulation_path': str(self.simulation_path),
            'requested_modules': analysis_modules,
            'successful_modules': successful_modules,
            'failed_modules': failed_modules,
            'total_metrics': len(self._extract_metrics_from_results(analysis_results)),
            'modules_run': len(analysis_results)
        }
    
    def load_basic_metrics(self) -> MetricsData:
        """Load basic metrics without using AnalysisService.
        
        This is a fallback method when AnalysisService is not available
        or when we want to avoid the overhead of full analysis.
        """
        logger.info(f"Loading basic metrics from {self.simulation_path}")
        
        # Basic metrics that can be extracted without AnalysisService
        metrics = {}
        
        # Try to extract basic simulation info
        try:
            metrics.update(self._extract_basic_simulation_metrics())
        except Exception as e:
            logger.warning(f"Could not extract basic simulation metrics: {e}")
        
        # Try to extract database metrics
        try:
            metrics.update(self._extract_database_metrics())
        except Exception as e:
            logger.warning(f"Could not extract database metrics: {e}")
        
        # Try to extract file-based metrics
        try:
            metrics.update(self._extract_file_metrics())
        except Exception as e:
            logger.warning(f"Could not extract file metrics: {e}")
        
        metadata = {
            'simulation_path': str(self.simulation_path),
            'metrics_type': 'basic',
            'total_metrics': len(metrics)
        }
        
        return MetricsData(
            simulation_path=self.simulation_path,
            metrics=metrics,
            analysis_results={},
            metadata=metadata
        )
    
    def _extract_basic_simulation_metrics(self) -> Dict[str, Any]:
        """Extract basic simulation metrics from files."""
        metrics = {}
        
        # Check for simulation.db
        db_path = self.simulation_path / "simulation.db"
        if db_path.exists():
            import sqlite3
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get basic database info
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    table_count = cursor.fetchone()[0]
                    metrics['database.table_count'] = table_count
                    
                    # Get total row count across all tables
                    total_rows = 0
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    for table in tables:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            count = cursor.fetchone()[0]
                            total_rows += count
                            metrics[f'database.{table}.row_count'] = count
                        except Exception:
                            continue
                    
                    metrics['database.total_rows'] = total_rows
                    
            except Exception as e:
                logger.debug(f"Could not extract basic database metrics: {e}")
        
        return metrics
    
    def _extract_database_metrics(self) -> Dict[str, Any]:
        """Extract metrics from database tables."""
        metrics = {}
        
        db_path = self.simulation_path / "simulation.db"
        if not db_path.exists():
            return metrics
        
        try:
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Look for common simulation tables
                common_tables = ['simulation_metadata', 'step_metrics', 'agents', 'actions']
                
                for table in common_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        metrics[f'database.{table}.count'] = count
                        
                        # Get sample of data for analysis
                        if count > 0:
                            cursor.execute(f"SELECT * FROM {table} LIMIT 10")
                            columns = [desc[0] for desc in cursor.description]
                            
                            # Look for numeric columns
                            for col in columns:
                                try:
                                    cursor.execute(f"""
                                        SELECT 
                                            AVG({col}) as avg_val,
                                            MIN({col}) as min_val,
                                            MAX({col}) as max_val
                                        FROM {table}
                                        WHERE {col} IS NOT NULL
                                    """)
                                    result = cursor.fetchone()
                                    if result and result[0] is not None:
                                        metrics[f'database.{table}.{col}.avg'] = result[0]
                                        metrics[f'database.{table}.{col}.min'] = result[1]
                                        metrics[f'database.{table}.{col}.max'] = result[2]
                                except Exception:
                                    continue
                                    
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"Could not extract database metrics: {e}")
        
        return metrics
    
    def _extract_file_metrics(self) -> Dict[str, Any]:
        """Extract metrics from file system."""
        metrics = {}
        
        # File sizes
        for file_path in self.simulation_path.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                metrics[f'files.{file_path.name}.size_mb'] = size_mb
        
        # Count files by type
        file_types = {}
        for file_path in self.simulation_path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        for ext, count in file_types.items():
            metrics[f'files.extension{ext}.count'] = count
        
        return metrics