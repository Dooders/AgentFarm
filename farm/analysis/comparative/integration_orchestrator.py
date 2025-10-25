"""
Integration orchestrator for simulation comparison analysis.

This module provides a high-level orchestrator that coordinates all phases
of the simulation comparison analysis system.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import json

from farm.utils.logging import get_logger

# Import all analysis phases
from farm.analysis.comparative.statistical_analyzer import StatisticalAnalyzer
from farm.analysis.comparative.ml_analyzer import MLAnalyzer
from farm.analysis.comparative.anomaly_detector import AdvancedAnomalyDetector
from farm.analysis.comparative.clustering_analyzer import ClusteringAnalyzer
from farm.analysis.comparative.trend_predictor import TrendPredictor
from farm.analysis.comparative.similarity_analyzer import SimilarityAnalyzer
from farm.analysis.comparative.ml_visualizer import MLVisualizer
from farm.analysis.comparative.comparison_result import SimulationComparisonResult

logger = get_logger(__name__)


@dataclass
class AnalysisPhaseConfig:
    """Configuration for individual analysis phases."""
    
    enabled: bool = True
    priority: int = 1  # 1 = highest priority
    timeout: Optional[float] = None  # seconds
    parallel: bool = False
    config: Optional[Dict[str, Any]] = None


@dataclass
class OrchestrationConfig:
    """Configuration for the analysis orchestration."""
    
    max_workers: int = 4
    timeout: Optional[float] = None  # seconds
    phases: Dict[str, AnalysisPhaseConfig] = field(default_factory=dict)
    output_dir: Union[str, Path] = "analysis_results"
    cache_results: bool = True
    parallel_execution: bool = True
    progress_callback: Optional[Callable] = None


@dataclass
class AnalysisPhaseResult:
    """Result of a single analysis phase."""
    
    phase_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    duration: float
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Complete result of analysis orchestration."""
    
    success: bool
    start_time: datetime
    end_time: datetime
    total_duration: float
    phase_results: List[AnalysisPhaseResult]
    summary: Dict[str, Any]
    output_paths: Dict[str, str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class IntegrationOrchestrator:
    """Orchestrates all phases of simulation comparison analysis."""
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        """Initialize the integration orchestrator."""
        self.config = config or OrchestrationConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self._initialize_analyzers()
        
        # Setup default phase configurations
        self._setup_default_phases()
        
        logger.info("IntegrationOrchestrator initialized")
    
    def _initialize_analyzers(self):
        """Initialize all analysis components."""
        try:
            self.statistical_analyzer = StatisticalAnalyzer()
            self.ml_analyzer = MLAnalyzer()
            self.anomaly_detector = AdvancedAnomalyDetector()
            self.clustering_analyzer = ClusteringAnalyzer()
            self.trend_predictor = TrendPredictor()
            self.similarity_analyzer = SimilarityAnalyzer()
            self.ml_visualizer = MLVisualizer()
            
            logger.info("All analyzers initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing analyzers: {e}")
            raise
    
    def _setup_default_phases(self):
        """Setup default phase configurations."""
        if not self.config.phases:
            self.config.phases = {
                "statistical_analysis": AnalysisPhaseConfig(
                    enabled=True, priority=1, timeout=300, parallel=False
                ),
                "ml_analysis": AnalysisPhaseConfig(
                    enabled=True, priority=2, timeout=1200, parallel=True
                ),
                "anomaly_detection": AnalysisPhaseConfig(
                    enabled=True, priority=3, timeout=300, parallel=True
                ),
                "clustering": AnalysisPhaseConfig(
                    enabled=True, priority=4, timeout=600, parallel=True
                ),
                "trend_prediction": AnalysisPhaseConfig(
                    enabled=True, priority=5, timeout=600, parallel=True
                ),
                "similarity_analysis": AnalysisPhaseConfig(
                    enabled=True, priority=6, timeout=300, parallel=True
                ),
                "visualization": AnalysisPhaseConfig(
                    enabled=True, priority=7, timeout=300, parallel=False
                )
            }
    
    async def analyze_simulations(self, 
                                simulation_pairs: List[tuple],
                                analysis_config: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """Perform comprehensive analysis on simulation pairs."""
        start_time = datetime.now()
        logger.info(f"Starting comprehensive analysis of {len(simulation_pairs)} simulation pairs")
        
        try:
            # Phase 1: Statistical Analysis
            statistical_results = await self._run_phase(
                "statistical_analysis",
                self._run_statistical_analysis,
                simulation_pairs,
                analysis_config
            )
            
            if not statistical_results.success:
                return self._create_failure_result(start_time, "Statistical analysis failed", statistical_results.error)
            
            # Phase 2: ML Analysis (can run in parallel with other phases)
            ml_tasks = []
            if self.config.phases.get("ml_analysis", AnalysisPhaseConfig()).enabled:
                ml_tasks.append(self._run_phase(
                    "ml_analysis",
                    self._run_ml_analysis,
                    statistical_results.result,
                    analysis_config
                ))
            
            if self.config.phases.get("anomaly_detection", AnalysisPhaseConfig()).enabled:
                ml_tasks.append(self._run_phase(
                    "anomaly_detection",
                    self._run_anomaly_detection,
                    statistical_results.result,
                    analysis_config
                ))
            
            if self.config.phases.get("clustering", AnalysisPhaseConfig()).enabled:
                ml_tasks.append(self._run_phase(
                    "clustering",
                    self._run_clustering,
                    statistical_results.result,
                    analysis_config
                ))
            
            if self.config.phases.get("trend_prediction", AnalysisPhaseConfig()).enabled:
                ml_tasks.append(self._run_phase(
                    "trend_prediction",
                    self._run_trend_prediction,
                    statistical_results.result,
                    analysis_config
                ))
            
            if self.config.phases.get("similarity_analysis", AnalysisPhaseConfig()).enabled:
                ml_tasks.append(self._run_phase(
                    "similarity_analysis",
                    self._run_similarity_analysis,
                    statistical_results.result,
                    analysis_config
                ))
            
            # Execute ML phases in parallel
            ml_results = {}
            if ml_tasks:
                if self.config.parallel_execution:
                    ml_phase_results = await asyncio.gather(*ml_tasks, return_exceptions=True)
                    for result in ml_phase_results:
                        if isinstance(result, AnalysisPhaseResult):
                            ml_results[result.phase_name] = result
                        else:
                            logger.error(f"ML phase failed with exception: {result}")
                else:
                    for task in ml_tasks:
                        result = await task
                        ml_results[result.phase_name] = result
            
            # Phase 3: Visualization
            visualization_results = None
            if self.config.phases.get("visualization", AnalysisPhaseConfig()).enabled:
                visualization_results = await self._run_phase(
                    "visualization",
                    self._run_visualization,
                    {
                        "statistical_results": statistical_results.result,
                        "ml_results": ml_results
                    },
                    analysis_config
                )
            
            # Compile final results
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            phase_results = [statistical_results]
            phase_results.extend(ml_results.values())
            if visualization_results:
                phase_results.append(visualization_results)
            
            # Generate summary
            summary = self._generate_summary(phase_results, total_duration)
            
            # Generate output paths
            output_paths = self._generate_output_paths(phase_results)
            
            # Collect errors and warnings
            errors = [r.error for r in phase_results if r.error]
            warnings = [r.metadata.get('warnings', []) for r in phase_results if r.metadata.get('warnings')]
            warnings = [w for warning_list in warnings for w in warning_list]
            
            result = OrchestrationResult(
                success=True,
                start_time=start_time,
                end_time=end_time,
                total_duration=total_duration,
                phase_results=phase_results,
                summary=summary,
                output_paths=output_paths,
                errors=errors,
                warnings=warnings
            )
            
            # Save results
            if self.config.cache_results:
                self._save_orchestration_result(result)
            
            logger.info(f"Analysis completed successfully in {total_duration:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed with error: {e}")
            return self._create_failure_result(start_time, "Analysis failed", str(e))
    
    async def _run_phase(self, 
                        phase_name: str, 
                        phase_func: Callable, 
                        *args, 
                        **kwargs) -> AnalysisPhaseResult:
        """Run a single analysis phase."""
        phase_config = self.config.phases.get(phase_name, AnalysisPhaseConfig())
        
        if not phase_config.enabled:
            return AnalysisPhaseResult(
                phase_name=phase_name,
                success=True,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=0.0,
                result=None,
                metadata={"skipped": True}
            )
        
        start_time = datetime.now()
        logger.info(f"Starting phase: {phase_name}")
        
        try:
            if phase_config.parallel and self.config.parallel_execution:
                # Run in thread pool for CPU-bound tasks
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await loop.run_in_executor(
                        executor, 
                        self._run_phase_sync, 
                        phase_func, 
                        *args, 
                        **kwargs
                    )
            else:
                result = await phase_func(*args, **kwargs)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Phase {phase_name} completed in {duration:.2f} seconds")
            
            return AnalysisPhaseResult(
                phase_name=phase_name,
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                result=result,
                metadata={"execution_mode": "parallel" if phase_config.parallel else "sequential"}
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Phase {phase_name} failed: {e}")
            
            return AnalysisPhaseResult(
                phase_name=phase_name,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                error=str(e)
            )
    
    def _run_phase_sync(self, phase_func: Callable, *args, **kwargs):
        """Run a phase function synchronously."""
        return phase_func(*args, **kwargs)
    
    async def _run_statistical_analysis(self, simulation_pairs: List[tuple], config: Optional[Dict[str, Any]] = None):
        """Run statistical analysis phase."""
        results = []
        for sim1_path, sim2_path in simulation_pairs:
            # Create a mock comparison result for statistical analysis
            # In a real implementation, this would load and compare the simulations
            mock_result = self._create_mock_comparison_result(sim1_path, sim2_path)
            analysis_result = self.statistical_analyzer.analyze_comparison(mock_result)
            results.append(analysis_result)
        return results
    
    async def _run_ml_analysis(self, statistical_results: List[Any], config: Optional[Dict[str, Any]] = None):
        """Run ML analysis phase."""
        # Convert statistical results to comparison results for ML analysis
        comparison_results = self._convert_to_comparison_results(statistical_results)
        return self.ml_analyzer.analyze_simulation_data(comparison_results, config)
    
    async def _run_anomaly_detection(self, statistical_results: List[Any], config: Optional[Dict[str, Any]] = None):
        """Run anomaly detection phase."""
        comparison_results = self._convert_to_comparison_results(statistical_results)
        return self.anomaly_detector.detect_anomalies(comparison_results)
    
    async def _run_clustering(self, statistical_results: List[Any], config: Optional[Dict[str, Any]] = None):
        """Run clustering analysis phase."""
        comparison_results = self._convert_to_comparison_results(statistical_results)
        return self.clustering_analyzer.cluster_simulations(comparison_results)
    
    async def _run_trend_prediction(self, statistical_results: List[Any], config: Optional[Dict[str, Any]] = None):
        """Run trend prediction phase."""
        comparison_results = self._convert_to_comparison_results(statistical_results)
        return self.trend_predictor.predict_trends(comparison_results)
    
    async def _run_similarity_analysis(self, statistical_results: List[Any], config: Optional[Dict[str, Any]] = None):
        """Run similarity analysis phase."""
        comparison_results = self._convert_to_comparison_results(statistical_results)
        return self.similarity_analyzer.analyze_similarity(comparison_results)
    
    async def _run_visualization(self, all_results: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Run visualization phase."""
        ml_results = all_results.get("ml_results", {})
        statistical_results = all_results.get("statistical_results", [])
        
        # Create ML dashboard
        dashboard_files = self.ml_visualizer.create_ml_dashboard(
            ml_results, statistical_results, "Comprehensive Analysis Dashboard"
        )
        
        return dashboard_files
    
    def _generate_summary(self, phase_results: List[AnalysisPhaseResult], total_duration: float) -> Dict[str, Any]:
        """Generate analysis summary."""
        successful_phases = [r for r in phase_results if r.success]
        failed_phases = [r for r in phase_results if not r.success]
        
        return {
            "total_phases": len(phase_results),
            "successful_phases": len(successful_phases),
            "failed_phases": len(failed_phases),
            "total_duration": total_duration,
            "average_phase_duration": sum(r.duration for r in phase_results) / len(phase_results) if phase_results else 0,
            "phase_breakdown": {
                r.phase_name: {
                    "success": r.success,
                    "duration": r.duration,
                    "error": r.error
                } for r in phase_results
            },
            "performance_metrics": {
                "total_simulations_analyzed": len(phase_results[0].result) if phase_results and phase_results[0].result else 0,
                "analysis_throughput": len(phase_results[0].result) / total_duration if phase_results and phase_results[0].result and total_duration > 0 else 0
            }
        }
    
    def _generate_output_paths(self, phase_results: List[AnalysisPhaseResult]) -> Dict[str, str]:
        """Generate output file paths."""
        output_paths = {}
        
        for result in phase_results:
            if result.success and result.result:
                timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
                filename = f"{result.phase_name}_{timestamp}.json"
                output_paths[result.phase_name] = str(self.output_dir / filename)
        
        return output_paths
    
    def _save_orchestration_result(self, result: OrchestrationResult):
        """Save orchestration result to file."""
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"orchestration_result_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert result to serializable format
        result_dict = {
            "success": result.success,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "total_duration": result.total_duration,
            "summary": result.summary,
            "output_paths": result.output_paths,
            "errors": result.errors,
            "warnings": result.warnings,
            "phase_results": [
                {
                    "phase_name": pr.phase_name,
                    "success": pr.success,
                    "start_time": pr.start_time.isoformat(),
                    "end_time": pr.end_time.isoformat(),
                    "duration": pr.duration,
                    "error": pr.error,
                    "metadata": pr.metadata
                } for pr in result.phase_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Orchestration result saved to {filepath}")
    
    def _create_failure_result(self, start_time: datetime, message: str, error: str) -> OrchestrationResult:
        """Create a failure result."""
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        return OrchestrationResult(
            success=False,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            phase_results=[],
            summary={"error": message},
            output_paths={},
            errors=[error]
        )
    
    def _create_mock_comparison_result(self, sim1_path: Path, sim2_path: Path) -> SimulationComparisonResult:
        """Create a mock comparison result for testing."""
        from farm.analysis.comparative.comparison_result import (
            MetricsComparisonResult, DatabaseComparisonResult, 
            LogComparisonResult, ConfigComparisonResult
        )
        
        # Create mock comparison results
        metrics_comp = MetricsComparisonResult(
            simulation1_metrics={"total_steps": 1000},
            simulation2_metrics={"total_steps": 1050},
            differences={"total_steps": 50},
            summary={"status": "completed"}
        )
        
        db_comp = DatabaseComparisonResult(
            simulation1_tables={"entities": 100},
            simulation2_tables={"entities": 105},
            differences={"entities": 5},
            summary={"status": "completed"}
        )
        
        log_comp = LogComparisonResult(
            simulation1_logs={"errors": 0},
            simulation2_logs={"errors": 1},
            differences={"errors": 1},
            summary={"status": "completed"}
        )
        
        config_comp = ConfigComparisonResult(
            simulation1_config={"param1": "value1"},
            simulation2_config={"param1": "value2"},
            differences={"param1": "different"},
            summary={"status": "completed"}
        )
        
        return SimulationComparisonResult(
            simulation1_path=sim1_path,
            simulation2_path=sim2_path,
            metrics_comparison=metrics_comp,
            database_comparison=db_comp,
            log_comparison=log_comp,
            config_comparison=config_comp,
            summary={"overall_status": "completed"}
        )
    
    def _convert_to_comparison_results(self, statistical_results: List[Any]) -> List[SimulationComparisonResult]:
        """Convert statistical results to comparison results for ML analysis."""
        # This is a simplified conversion - in practice, you'd extract the actual comparison data
        comparison_results = []
        for i, result in enumerate(statistical_results):
            sim1_path = Path(f"/tmp/sim1_{i}")
            sim2_path = Path(f"/tmp/sim2_{i}")
            comparison_result = self._create_mock_comparison_result(sim1_path, sim2_path)
            comparison_results.append(comparison_result)
        return comparison_results
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis system status."""
        return {
            "analyzers_initialized": {
                "statistical_analyzer": hasattr(self, 'statistical_analyzer'),
                "ml_analyzer": hasattr(self, 'ml_analyzer'),
                "anomaly_detector": hasattr(self, 'anomaly_detector'),
                "clustering_analyzer": hasattr(self, 'clustering_analyzer'),
                "trend_predictor": hasattr(self, 'trend_predictor'),
                "similarity_analyzer": hasattr(self, 'similarity_analyzer'),
                "ml_visualizer": hasattr(self, 'ml_visualizer')
            },
            "configuration": {
                "max_workers": self.config.max_workers,
                "parallel_execution": self.config.parallel_execution,
                "output_dir": str(self.output_dir),
                "enabled_phases": [name for name, config in self.config.phases.items() if config.enabled]
            },
            "output_directory": str(self.output_dir),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis system status."""
        return {
            "analyzers_initialized": {
                "basic_analyzer": hasattr(self, 'basic_analyzer'),
                "advanced_analyzer": hasattr(self, 'advanced_analyzer'),
                "ml_analyzer": hasattr(self, 'ml_analyzer'),
                "anomaly_detector": hasattr(self, 'anomaly_detector'),
                "clustering_analyzer": hasattr(self, 'clustering_analyzer'),
                "trend_predictor": hasattr(self, 'trend_predictor'),
                "similarity_analyzer": hasattr(self, 'similarity_analyzer'),
                "ml_visualizer": hasattr(self, 'ml_visualizer')
            },
            "configuration": {
                "max_workers": self.config.max_workers,
                "parallel_execution": self.config.parallel_execution,
                "output_dir": str(self.output_dir),
                "enabled_phases": [name for name, config in self.config.phases.items() if config.enabled]
            },
            "output_directory": str(self.output_dir),
            "timestamp": datetime.now().isoformat()
        }