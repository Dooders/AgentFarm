"""
Automated insight generation and explanation system.

This module provides intelligent insight generation that automatically identifies
patterns, anomalies, and meaningful findings in simulation analysis results.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import asyncio
from enum import Enum
import statistics
import math

# Optional imports for advanced analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult
from farm.analysis.comparative.comparison_result import SimulationComparisonResult

logger = get_logger(__name__)


class InsightType(Enum):
    """Types of insights that can be generated."""
    PERFORMANCE_PATTERN = "performance_pattern"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    CORRELATION_FINDING = "correlation_finding"
    CLUSTERING_RESULT = "clustering_result"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    QUALITY_ASSESSMENT = "quality_assessment"
    PREDICTIVE_INSIGHT = "predictive_insight"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"


class InsightSeverity(Enum):
    """Severity levels for insights."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Insight:
    """A generated insight with metadata."""
    
    id: str
    type: InsightType
    title: str
    description: str
    severity: InsightSeverity
    confidence: float
    data_points: List[Any] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightGenerationConfig:
    """Configuration for insight generation."""
    
    enable_performance_analysis: bool = True
    enable_anomaly_detection: bool = True
    enable_trend_analysis: bool = True
    enable_clustering: bool = True
    enable_correlation_analysis: bool = True
    enable_optimization_suggestions: bool = True
    enable_quality_assessment: bool = True
    enable_predictive_insights: bool = True
    
    # Thresholds
    anomaly_threshold: float = 0.1
    correlation_threshold: float = 0.7
    performance_threshold: float = 0.2
    quality_threshold: float = 0.8
    
    # Clustering parameters
    min_cluster_size: int = 2
    max_clusters: int = 10
    clustering_algorithm: str = "dbscan"  # dbscan, kmeans
    
    # Output settings
    max_insights: int = 50
    min_confidence: float = 0.5
    include_visualizations: bool = True


class AutomatedInsightGenerator:
    """Automated insight generation and explanation system."""
    
    def __init__(self, config: Optional[InsightGenerationConfig] = None):
        """Initialize the insight generator."""
        self.config = config or InsightGenerationConfig()
        self.insights: List[Insight] = []
        self.analysis_data: Optional[Dict[str, Any]] = None
        
        logger.info("AutomatedInsightGenerator initialized")
    
    async def generate_insights(self, 
                              analysis_result: OrchestrationResult,
                              simulation_data: Optional[Dict[str, Any]] = None) -> List[Insight]:
        """Generate insights from analysis results."""
        logger.info("Starting automated insight generation")
        
        self.analysis_data = self._extract_analysis_data(analysis_result, simulation_data)
        self.insights = []
        
        # Generate different types of insights
        if self.config.enable_performance_analysis:
            await self._generate_performance_insights()
        
        if self.config.enable_anomaly_detection:
            await self._generate_anomaly_insights()
        
        if self.config.enable_trend_analysis:
            await self._generate_trend_insights()
        
        if self.config.enable_clustering:
            await self._generate_clustering_insights()
        
        if self.config.enable_correlation_analysis:
            await self._generate_correlation_insights()
        
        if self.config.enable_optimization_suggestions:
            await self._generate_optimization_insights()
        
        if self.config.enable_quality_assessment:
            await self._generate_quality_insights()
        
        if self.config.enable_predictive_insights:
            await self._generate_predictive_insights()
        
        # Filter and rank insights
        self.insights = self._filter_and_rank_insights()
        
        logger.info(f"Generated {len(self.insights)} insights")
        return self.insights
    
    def _extract_analysis_data(self, 
                              analysis_result: OrchestrationResult,
                              simulation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data from analysis results."""
        data = {
            "analysis_success": analysis_result.success,
            "total_duration": analysis_result.total_duration,
            "phase_results": analysis_result.phase_results,
            "errors": analysis_result.errors,
            "warnings": analysis_result.warnings,
            "summary": analysis_result.summary,
            "simulation_data": simulation_data or {}
        }
        
        # Extract metrics from phase results
        metrics = {}
        for phase_result in analysis_result.phase_results:
            if hasattr(phase_result, 'metrics'):
                metrics[phase_result.phase_name] = phase_result.metrics
        
        data["metrics"] = metrics
        return data
    
    async def _generate_performance_insights(self):
        """Generate performance-related insights."""
        if not self.analysis_data:
            return
        
        # Analyze execution time patterns
        duration = self.analysis_data.get("total_duration", 0)
        if duration > 0:
            # Check if duration is unusually long
            if duration > 300:  # 5 minutes
                insight = Insight(
                    id=f"perf_duration_{len(self.insights)}",
                    type=InsightType.PERFORMANCE_PATTERN,
                    title="Long Analysis Duration",
                    description=f"Analysis took {duration:.2f} seconds, which is longer than typical runs.",
                    severity=InsightSeverity.MEDIUM,
                    confidence=0.8,
                    metrics={"duration": duration, "threshold": 300},
                    recommendations=[
                        "Consider optimizing analysis algorithms",
                        "Use parallel processing for large datasets",
                        "Implement caching for repeated calculations"
                    ],
                    tags=["performance", "duration", "optimization"]
                )
                self.insights.append(insight)
            
            # Check for performance consistency
            if duration < 10:  # Very fast
                insight = Insight(
                    id=f"perf_fast_{len(self.insights)}",
                    type=InsightType.PERFORMANCE_PATTERN,
                    title="Very Fast Analysis",
                    description=f"Analysis completed in {duration:.2f} seconds, which is unusually fast.",
                    severity=InsightSeverity.LOW,
                    confidence=0.7,
                    metrics={"duration": duration},
                    recommendations=[
                        "Verify that all analysis phases completed successfully",
                        "Check if any phases were skipped"
                    ],
                    tags=["performance", "duration", "verification"]
                )
                self.insights.append(insight)
        
        # Analyze phase performance
        phase_results = self.analysis_data.get("phase_results", [])
        if len(phase_results) > 1:
            phase_durations = []
            for phase in phase_results:
                if hasattr(phase, 'duration'):
                    phase_durations.append(phase.duration)
            
            if phase_durations:
                avg_duration = statistics.mean(phase_durations)
                std_duration = statistics.stdev(phase_durations) if len(phase_durations) > 1 else 0
                
                # Check for phase performance variance
                if std_duration > avg_duration * 0.5:  # High variance
                    insight = Insight(
                        id=f"perf_variance_{len(self.insights)}",
                        type=InsightType.PERFORMANCE_PATTERN,
                        title="High Phase Performance Variance",
                        description=f"Analysis phases show high performance variance (std: {std_duration:.2f}s, mean: {avg_duration:.2f}s).",
                        severity=InsightSeverity.MEDIUM,
                        confidence=0.8,
                        metrics={
                            "mean_duration": avg_duration,
                            "std_duration": std_duration,
                            "coefficient_of_variation": std_duration / avg_duration if avg_duration > 0 else 0
                        },
                        recommendations=[
                            "Investigate phases with unusually long durations",
                            "Consider load balancing across phases",
                            "Profile individual phase performance"
                        ],
                        tags=["performance", "variance", "phases"]
                    )
                    self.insights.append(insight)
    
    async def _generate_anomaly_insights(self):
        """Generate anomaly detection insights."""
        if not self.analysis_data:
            return
        
        # Check for analysis errors
        errors = self.analysis_data.get("errors", [])
        if errors:
            insight = Insight(
                id=f"anomaly_errors_{len(self.insights)}",
                type=InsightType.ANOMALY_DETECTION,
                title="Analysis Errors Detected",
                description=f"Found {len(errors)} errors during analysis execution.",
                severity=InsightSeverity.HIGH,
                confidence=0.9,
                data_points=errors,
                recommendations=[
                    "Review and fix the identified errors",
                    "Check input data quality and format",
                    "Verify analysis configuration",
                    "Consider running analysis in debug mode"
                ],
                tags=["anomaly", "errors", "quality"]
            )
            self.insights.append(insight)
        
        # Check for warnings
        warnings = self.analysis_data.get("warnings", [])
        if warnings:
            insight = Insight(
                id=f"anomaly_warnings_{len(self.insights)}",
                type=InsightType.ANOMALY_DETECTION,
                title="Analysis Warnings Generated",
                description=f"Generated {len(warnings)} warnings during analysis.",
                severity=InsightSeverity.MEDIUM,
                confidence=0.8,
                data_points=warnings,
                recommendations=[
                    "Review warnings for potential issues",
                    "Consider addressing warnings to improve analysis quality",
                    "Monitor warnings for patterns"
                ],
                tags=["anomaly", "warnings", "quality"]
            )
            self.insights.append(insight)
        
        # Analyze simulation data for anomalies if available
        simulation_data = self.analysis_data.get("simulation_data", {})
        if simulation_data and SKLEARN_AVAILABLE:
            await self._detect_data_anomalies(simulation_data)
    
    async def _detect_data_anomalies(self, simulation_data: Dict[str, Any]):
        """Detect anomalies in simulation data using ML techniques."""
        try:
            # Extract numerical data for anomaly detection
            numerical_data = self._extract_numerical_data(simulation_data)
            if not numerical_data or len(numerical_data) < 10:
                return
            
            # Convert to numpy array for sklearn
            data_array = np.array(numerical_data)
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=self.config.anomaly_threshold, random_state=42)
            anomaly_labels = iso_forest.fit_predict(data_array)
            
            # Count anomalies
            num_anomalies = sum(1 for label in anomaly_labels if label == -1)
            anomaly_ratio = num_anomalies / len(anomaly_labels)
            
            if anomaly_ratio > 0.1:  # More than 10% anomalies
                insight = Insight(
                    id=f"anomaly_data_{len(self.insights)}",
                    type=InsightType.ANOMALY_DETECTION,
                    title="Data Anomalies Detected",
                    description=f"Found {num_anomalies} anomalous data points ({anomaly_ratio:.1%} of total data).",
                    severity=InsightSeverity.HIGH if anomaly_ratio > 0.2 else InsightSeverity.MEDIUM,
                    confidence=0.8,
                    metrics={
                        "anomaly_count": num_anomalies,
                        "anomaly_ratio": anomaly_ratio,
                        "total_data_points": len(anomaly_labels)
                    },
                    recommendations=[
                        "Investigate anomalous data points",
                        "Check data collection process",
                        "Consider data cleaning or filtering",
                        "Verify data source integrity"
                    ],
                    tags=["anomaly", "data", "quality", "ml"]
                )
                self.insights.append(insight)
        
        except Exception as e:
            logger.warning(f"Error in anomaly detection: {e}")
    
    def _extract_numerical_data(self, simulation_data: Dict[str, Any]) -> List[List[float]]:
        """Extract numerical data from simulation data for analysis."""
        numerical_data = []
        
        # This is a simplified extraction - in practice, you'd need to know the data structure
        for key, value in simulation_data.items():
            if isinstance(value, (int, float)):
                numerical_data.append([float(value)])
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                numerical_data.append([float(x) for x in value])
            elif isinstance(value, dict):
                # Recursively extract from nested dictionaries
                nested_data = self._extract_numerical_data(value)
                numerical_data.extend(nested_data)
        
        return numerical_data
    
    async def _generate_trend_insights(self):
        """Generate trend analysis insights."""
        if not self.analysis_data:
            return
        
        # Analyze phase completion trends
        phase_results = self.analysis_data.get("phase_results", [])
        if len(phase_results) > 2:
            phase_names = [phase.phase_name for phase in phase_results]
            
            # Check for sequential phase completion
            if len(set(phase_names)) == len(phase_names):  # All phases unique
                insight = Insight(
                    id=f"trend_sequential_{len(self.insights)}",
                    type=InsightType.TREND_ANALYSIS,
                    title="Sequential Phase Execution",
                    description=f"Analysis executed {len(phase_results)} phases sequentially: {', '.join(phase_names)}.",
                    severity=InsightSeverity.LOW,
                    confidence=0.9,
                    metrics={"phase_count": len(phase_results), "phases": phase_names},
                    recommendations=[
                        "Consider parallel execution for independent phases",
                        "Optimize phase dependencies for better performance"
                    ],
                    tags=["trend", "phases", "execution"]
                )
                self.insights.append(insight)
        
        # Analyze error/warning trends
        errors = self.analysis_data.get("errors", [])
        warnings = self.analysis_data.get("warnings", [])
        
        if errors and warnings:
            error_warning_ratio = len(errors) / len(warnings) if warnings else 0
            
            if error_warning_ratio > 0.5:  # More errors than warnings
                insight = Insight(
                    id=f"trend_error_heavy_{len(self.insights)}",
                    type=InsightType.TREND_ANALYSIS,
                    title="Error-Heavy Analysis",
                    description=f"Analysis shows {len(errors)} errors vs {len(warnings)} warnings (ratio: {error_warning_ratio:.2f}).",
                    severity=InsightSeverity.HIGH,
                    confidence=0.8,
                    metrics={"error_count": len(errors), "warning_count": len(warnings), "ratio": error_warning_ratio},
                    recommendations=[
                        "Focus on error resolution before warning cleanup",
                        "Review analysis configuration and input data",
                        "Consider running in debug mode for detailed error information"
                    ],
                    tags=["trend", "errors", "warnings", "quality"]
                )
                self.insights.append(insight)
    
    async def _generate_clustering_insights(self):
        """Generate clustering analysis insights."""
        if not self.analysis_data or not SKLEARN_AVAILABLE:
            return
        
        # Extract data for clustering
        simulation_data = self.analysis_data.get("simulation_data", {})
        numerical_data = self._extract_numerical_data(simulation_data)
        
        if not numerical_data or len(numerical_data) < 10:
            return
        
        try:
            data_array = np.array(numerical_data)
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_array)
            
            # Try different clustering algorithms
            if self.config.clustering_algorithm == "dbscan":
                clusterer = DBSCAN(min_samples=self.config.min_cluster_size)
            else:  # kmeans
                n_clusters = min(self.config.max_clusters, len(data_array) // 2)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            
            cluster_labels = clusterer.fit_predict(data_scaled)
            
            # Analyze clustering results
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise points
            n_noise = list(cluster_labels).count(-1)
            
            if n_clusters > 1:
                # Calculate silhouette score
                if len(set(cluster_labels)) > 1 and -1 not in unique_labels:
                    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
                else:
                    silhouette_avg = 0.0
                
                insight = Insight(
                    id=f"clustering_{len(self.insights)}",
                    type=InsightType.CLUSTERING_RESULT,
                    title="Data Clustering Identified",
                    description=f"Found {n_clusters} distinct clusters in the data with {n_noise} noise points.",
                    severity=InsightSeverity.MEDIUM,
                    confidence=0.7,
                    metrics={
                        "n_clusters": n_clusters,
                        "n_noise": n_noise,
                        "silhouette_score": silhouette_avg,
                        "algorithm": self.config.clustering_algorithm
                    },
                    recommendations=[
                        "Analyze cluster characteristics to understand data patterns",
                        "Consider cluster-based analysis approaches",
                        "Investigate noise points for potential anomalies"
                    ],
                    tags=["clustering", "patterns", "ml"]
                )
                self.insights.append(insight)
        
        except Exception as e:
            logger.warning(f"Error in clustering analysis: {e}")
    
    async def _generate_correlation_insights(self):
        """Generate correlation analysis insights."""
        if not self.analysis_data or not PANDAS_AVAILABLE:
            return
        
        # Extract metrics for correlation analysis
        metrics = self.analysis_data.get("metrics", {})
        if not metrics:
            return
        
        try:
            # Create a DataFrame from metrics
            df_data = {}
            for phase_name, phase_metrics in metrics.items():
                if isinstance(phase_metrics, dict):
                    for metric_name, metric_value in phase_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            key = f"{phase_name}_{metric_name}"
                            df_data[key] = [metric_value]
            
            if len(df_data) < 2:
                return
            
            df = pd.DataFrame(df_data)
            
            # Calculate correlations
            correlation_matrix = df.corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > self.config.correlation_threshold:
                        strong_correlations.append({
                            "metric1": correlation_matrix.columns[i],
                            "metric2": correlation_matrix.columns[j],
                            "correlation": corr_value
                        })
            
            if strong_correlations:
                insight = Insight(
                    id=f"correlation_{len(self.insights)}",
                    type=InsightType.CORRELATION_FINDING,
                    title="Strong Correlations Found",
                    description=f"Found {len(strong_correlations)} strong correlations between metrics.",
                    severity=InsightSeverity.MEDIUM,
                    confidence=0.8,
                    data_points=strong_correlations,
                    recommendations=[
                        "Investigate correlated metrics for causal relationships",
                        "Consider using correlated metrics as predictors",
                        "Look for redundant metrics that can be removed"
                    ],
                    tags=["correlation", "metrics", "relationships"]
                )
                self.insights.append(insight)
        
        except Exception as e:
            logger.warning(f"Error in correlation analysis: {e}")
    
    async def _generate_optimization_insights(self):
        """Generate optimization opportunity insights."""
        if not self.analysis_data:
            return
        
        # Check for optimization opportunities
        duration = self.analysis_data.get("total_duration", 0)
        phase_results = self.analysis_data.get("phase_results", [])
        
        if duration > 60 and len(phase_results) > 1:  # Long analysis with multiple phases
            insight = Insight(
                id=f"optimization_parallel_{len(self.insights)}",
                type=InsightType.OPTIMIZATION_OPPORTUNITY,
                title="Parallel Processing Opportunity",
                description=f"Analysis with {len(phase_results)} phases took {duration:.2f}s. Consider parallel execution for independent phases.",
                severity=InsightSeverity.MEDIUM,
                confidence=0.8,
                metrics={"duration": duration, "phase_count": len(phase_results)},
                recommendations=[
                    "Identify independent phases for parallel execution",
                    "Implement parallel processing framework",
                    "Profile individual phase performance",
                    "Consider distributed processing for large datasets"
                ],
                tags=["optimization", "parallel", "performance"]
            )
            self.insights.append(insight)
        
        # Check for memory optimization opportunities
        if self.analysis_data.get("warnings"):
            memory_warnings = [w for w in self.analysis_data["warnings"] if "memory" in w.lower()]
            if memory_warnings:
                insight = Insight(
                    id=f"optimization_memory_{len(self.insights)}",
                    type=InsightType.OPTIMIZATION_OPPORTUNITY,
                    title="Memory Optimization Opportunity",
                    description=f"Found {len(memory_warnings)} memory-related warnings. Consider memory optimization strategies.",
                    severity=InsightSeverity.MEDIUM,
                    confidence=0.7,
                    data_points=memory_warnings,
                    recommendations=[
                        "Implement memory-efficient data structures",
                        "Use streaming processing for large datasets",
                        "Consider data compression techniques",
                        "Monitor memory usage patterns"
                    ],
                    tags=["optimization", "memory", "performance"]
                )
                self.insights.append(insight)
    
    async def _generate_quality_insights(self):
        """Generate quality assessment insights."""
        if not self.analysis_data:
            return
        
        # Calculate quality score
        quality_score = self._calculate_quality_score()
        
        if quality_score < self.config.quality_threshold:
            insight = Insight(
                id=f"quality_low_{len(self.insights)}",
                type=InsightType.QUALITY_ASSESSMENT,
                title="Low Analysis Quality",
                description=f"Analysis quality score is {quality_score:.2f}, below the threshold of {self.config.quality_threshold}.",
                severity=InsightSeverity.HIGH,
                confidence=0.8,
                metrics={"quality_score": quality_score, "threshold": self.config.quality_threshold},
                recommendations=[
                    "Review and fix analysis errors",
                    "Improve input data quality",
                    "Verify analysis configuration",
                    "Consider additional validation steps"
                ],
                tags=["quality", "assessment", "improvement"]
            )
            self.insights.append(insight)
        elif quality_score > 0.9:
            insight = Insight(
                id=f"quality_high_{len(self.insights)}",
                type=InsightType.QUALITY_ASSESSMENT,
                title="High Analysis Quality",
                description=f"Analysis quality score is {quality_score:.2f}, indicating excellent quality.",
                severity=InsightSeverity.LOW,
                confidence=0.9,
                metrics={"quality_score": quality_score},
                recommendations=[
                    "Maintain current analysis practices",
                    "Consider this as a baseline for future analyses",
                    "Document successful analysis patterns"
                ],
                tags=["quality", "assessment", "excellence"]
            )
            self.insights.append(insight)
    
    def _calculate_quality_score(self) -> float:
        """Calculate a quality score for the analysis."""
        if not self.analysis_data:
            return 0.0
        
        score = 1.0
        
        # Deduct for errors
        errors = self.analysis_data.get("errors", [])
        if errors:
            score -= min(0.5, len(errors) * 0.1)
        
        # Deduct for warnings
        warnings = self.analysis_data.get("warnings", [])
        if warnings:
            score -= min(0.3, len(warnings) * 0.05)
        
        # Check if analysis was successful
        if not self.analysis_data.get("analysis_success", False):
            score -= 0.4
        
        # Check for reasonable duration
        duration = self.analysis_data.get("total_duration", 0)
        if duration > 600:  # Very long analysis
            score -= 0.1
        
        return max(0.0, score)
    
    async def _generate_predictive_insights(self):
        """Generate predictive insights."""
        if not self.analysis_data:
            return
        
        # Simple predictive insights based on current data
        duration = self.analysis_data.get("total_duration", 0)
        phase_count = len(self.analysis_data.get("phase_results", []))
        
        if duration > 0 and phase_count > 0:
            avg_phase_duration = duration / phase_count
            
            # Predict future analysis duration
            if phase_count < 10:  # Limited data
                predicted_duration = duration * 1.2  # Conservative estimate
                
                insight = Insight(
                    id=f"predictive_duration_{len(self.insights)}",
                    type=InsightType.PREDICTIVE_INSIGHT,
                    title="Duration Prediction",
                    description=f"Based on current performance, future analyses with similar complexity may take approximately {predicted_duration:.2f} seconds.",
                    severity=InsightSeverity.LOW,
                    confidence=0.6,
                    metrics={
                        "current_duration": duration,
                        "predicted_duration": predicted_duration,
                        "avg_phase_duration": avg_phase_duration
                    },
                    recommendations=[
                        "Use this prediction for resource planning",
                        "Monitor actual vs predicted performance",
                        "Update predictions as more data becomes available"
                    ],
                    tags=["predictive", "duration", "planning"]
                )
                self.insights.append(insight)
    
    def _filter_and_rank_insights(self) -> List[Insight]:
        """Filter and rank insights by relevance and importance."""
        # Filter by confidence and max count
        filtered_insights = [
            insight for insight in self.insights
            if insight.confidence >= self.config.min_confidence
        ]
        
        # Sort by severity and confidence
        severity_order = {
            InsightSeverity.CRITICAL: 4,
            InsightSeverity.HIGH: 3,
            InsightSeverity.MEDIUM: 2,
            InsightSeverity.LOW: 1
        }
        
        filtered_insights.sort(
            key=lambda x: (severity_order.get(x.severity, 0), x.confidence),
            reverse=True
        )
        
        # Limit to max insights
        return filtered_insights[:self.config.max_insights]
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get a summary of generated insights."""
        if not self.insights:
            return {"total_insights": 0}
        
        # Count by type and severity
        type_counts = {}
        severity_counts = {}
        
        for insight in self.insights:
            type_counts[insight.type.value] = type_counts.get(insight.type.value, 0) + 1
            severity_counts[insight.severity.value] = severity_counts.get(insight.severity.value, 0) + 1
        
        return {
            "total_insights": len(self.insights),
            "by_type": type_counts,
            "by_severity": severity_counts,
            "avg_confidence": statistics.mean([i.confidence for i in self.insights]),
            "high_severity_count": severity_counts.get("high", 0) + severity_counts.get("critical", 0)
        }
    
    def export_insights(self, format: str = "json", file_path: Optional[Union[str, Path]] = None) -> Union[str, Path]:
        """Export insights to various formats."""
        if not self.insights:
            return "No insights to export"
        
        if format == "json":
            data = [insight.__dict__ for insight in self.insights]
            json_str = json.dumps(data, indent=2, default=str)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(json_str)
                return Path(file_path)
            else:
                return json_str
        
        elif format == "markdown":
            md_content = "# Analysis Insights\n\n"
            md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_content += f"Total insights: {len(self.insights)}\n\n"
            
            # Group by severity
            for severity in [InsightSeverity.CRITICAL, InsightSeverity.HIGH, InsightSeverity.MEDIUM, InsightSeverity.LOW]:
                severity_insights = [i for i in self.insights if i.severity == severity]
                if severity_insights:
                    md_content += f"## {severity.value.title()} Severity ({len(severity_insights)} insights)\n\n"
                    for insight in severity_insights:
                        md_content += f"### {insight.title}\n"
                        md_content += f"**Type**: {insight.type.value}\n"
                        md_content += f"**Confidence**: {insight.confidence:.2f}\n"
                        md_content += f"**Description**: {insight.description}\n\n"
                        if insight.recommendations:
                            md_content += "**Recommendations**:\n"
                            for rec in insight.recommendations:
                                md_content += f"- {rec}\n"
                            md_content += "\n"
                        md_content += "---\n\n"
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(md_content)
                return Path(file_path)
            else:
                return md_content
        
        else:
            raise ValueError(f"Unsupported format: {format}")