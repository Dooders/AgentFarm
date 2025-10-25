"""
Advanced anomaly detection module for simulation comparison.

This module provides sophisticated anomaly detection capabilities using
multiple ML algorithms and statistical methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json

from farm.analysis.comparative.comparison_result import SimulationComparisonResult
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import ML libraries
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Anomaly detection will be limited.")

try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Some statistical methods will be limited.")


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    contamination: float = 0.1
    n_neighbors: int = 20
    nu: float = 0.1
    kernel: str = 'rbf'
    gamma: str = 'scale'
    min_samples: int = 5
    eps: float = 0.5
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    enable_ensemble: bool = True
    ensemble_threshold: float = 0.5


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    anomalies: List[Dict[str, Any]]
    anomaly_scores: List[float]
    anomaly_types: List[str]
    severity_levels: List[str]
    confidence_scores: List[float]
    recommendations: List[str]
    detection_methods: List[str]
    summary: Dict[str, Any]


class AdvancedAnomalyDetector:
    """Advanced anomaly detector for simulation comparison data."""
    
    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        """Initialize anomaly detector.
        
        Args:
            config: Configuration for anomaly detection
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for advanced anomaly detection")
        
        self.config = config or AnomalyDetectionConfig()
        self._initialize_detectors()
        
        logger.info("AdvancedAnomalyDetector initialized")
    
    def _initialize_detectors(self):
        """Initialize anomaly detection models."""
        self.detectors = {
            'isolation_forest': IsolationForest(
                contamination=self.config.contamination,
                random_state=42
            ),
            'local_outlier_factor': LocalOutlierFactor(
                n_neighbors=self.config.n_neighbors,
                contamination=self.config.contamination
            ),
            'one_class_svm': OneClassSVM(
                nu=self.config.nu,
                kernel=self.config.kernel,
                gamma=self.config.gamma
            ),
            'dbscan': DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples
            )
        }
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
    
    def detect_anomalies(self, 
                        results: List[SimulationComparisonResult],
                        method: str = 'ensemble',
                        features: Optional[List[str]] = None) -> AnomalyResult:
        """Detect anomalies in simulation comparison results.
        
        Args:
            results: List of simulation comparison results
            method: Detection method ('ensemble', 'isolation_forest', 'lof', 'svm', 'dbscan', 'statistical')
            features: List of features to use for detection
            
        Returns:
            Anomaly detection result
        """
        logger.info(f"Detecting anomalies using method: {method}")
        
        if len(results) < 4:
            logger.warning("Insufficient data for anomaly detection")
            return self._create_empty_result()
        
        # Extract features
        features_df = self._extract_features(results, features)
        
        if features_df.empty:
            logger.warning("No features extracted for anomaly detection")
            return self._create_empty_result()
        
        # Detect anomalies based on method
        if method == 'ensemble':
            return self._ensemble_detection(features_df, results)
        elif method == 'statistical':
            return self._statistical_detection(features_df, results)
        else:
            return self._single_method_detection(features_df, results, method)
    
    def _extract_features(self, 
                         results: List[SimulationComparisonResult],
                         features: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract features for anomaly detection."""
        feature_data = []
        
        for i, result in enumerate(results):
            row = {'simulation_id': i}
            
            # Basic comparison features
            row.update({
                'total_differences': result.comparison_summary.total_differences,
                'config_differences': result.comparison_summary.config_differences,
                'database_differences': result.comparison_summary.database_differences,
                'log_differences': result.comparison_summary.log_differences,
                'metrics_differences': result.comparison_summary.metrics_differences,
                'severity_numeric': self._severity_to_numeric(result.comparison_summary.severity)
            })
            
            # Metrics features
            if result.metrics_comparison.metric_differences:
                for metric, diff in result.metrics_comparison.metric_differences.items():
                    if isinstance(diff, dict):
                        row[f'metric_{metric}_change'] = diff.get('percentage_change', 0)
                        row[f'metric_{metric}_abs_change'] = abs(diff.get('percentage_change', 0))
            
            # Performance features
            if result.metrics_comparison.performance_comparison:
                for metric, comp in result.metrics_comparison.performance_comparison.items():
                    if isinstance(comp, dict):
                        ratio = comp.get('ratio', 1.0)
                        row[f'perf_{metric}_ratio'] = ratio
                        row[f'perf_{metric}_improvement'] = max(0, ratio - 1.0)
                        row[f'perf_{metric}_degradation'] = max(0, 1.0 - ratio)
            
            # Error features
            if result.log_comparison.error_differences:
                total_errors = 0
                error_increase = 0
                error_decrease = 0
                
                for error_type, diff in result.log_comparison.error_differences.items():
                    if isinstance(diff, dict):
                        error_diff = diff.get('difference', 0)
                        total_errors += abs(error_diff)
                        if error_diff > 0:
                            error_increase += error_diff
                        else:
                            error_decrease += abs(error_diff)
                
                row.update({
                    'total_error_changes': total_errors,
                    'error_increase': error_increase,
                    'error_decrease': error_decrease,
                    'error_net_change': error_increase - error_decrease
                })
            
            # Database features
            if result.database_comparison.schema_differences:
                row['schema_differences'] = len(result.database_comparison.schema_differences)
            
            if result.database_comparison.data_differences:
                row['data_differences'] = len(result.database_comparison.data_differences)
            
            # Log features
            if result.log_comparison.performance_differences:
                total_perf_diff = 0
                for perf_type, diff in result.log_comparison.performance_differences.items():
                    if isinstance(diff, dict):
                        total_perf_diff += abs(diff.get('difference', 0))
                row['log_perf_differences'] = total_perf_diff
            
            feature_data.append(row)
        
        features_df = pd.DataFrame(feature_data)
        
        # Select specific features if requested
        if features:
            available_features = [f for f in features if f in features_df.columns]
            if available_features:
                features_df = features_df[['simulation_id'] + available_features]
        
        # Remove non-numeric columns except simulation_id
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'simulation_id' not in numeric_cols:
            numeric_cols.append('simulation_id')
        
        return features_df[numeric_cols]
    
    def _severity_to_numeric(self, severity: str) -> int:
        """Convert severity string to numeric value."""
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return severity_map.get(severity.lower(), 0)
    
    def _ensemble_detection(self, features_df: pd.DataFrame, 
                          results: List[SimulationComparisonResult]) -> AnomalyResult:
        """Perform ensemble anomaly detection."""
        logger.debug("Performing ensemble anomaly detection")
        
        # Prepare features
        feature_cols = [col for col in features_df.columns if col != 'simulation_id']
        X = features_df[feature_cols].values
        
        # Scale features
        scaler = self.scalers['robust']  # Use robust scaler for anomaly detection
        X_scaled = scaler.fit_transform(X)
        
        # Get predictions from all detectors
        detector_predictions = {}
        detector_scores = {}
        
        for name, detector in self.detectors.items():
            try:
                if name == 'dbscan':
                    # DBSCAN doesn't have predict method, use fit_predict
                    labels = detector.fit_predict(X_scaled)
                    predictions = (labels == -1).astype(int)
                    scores = np.zeros(len(X_scaled))  # DBSCAN doesn't provide scores
                else:
                    predictions = detector.fit_predict(X_scaled)
                    predictions = (predictions == -1).astype(int)
                    
                    # Get anomaly scores
                    if hasattr(detector, 'decision_function'):
                        scores = detector.decision_function(X_scaled)
                    elif hasattr(detector, 'score_samples'):
                        scores = detector.score_samples(X_scaled)
                    else:
                        scores = np.zeros(len(X_scaled))
                
                detector_predictions[name] = predictions
                detector_scores[name] = scores
                
            except Exception as e:
                logger.warning(f"Error with detector {name}: {e}")
                continue
        
        if not detector_predictions:
            return self._create_empty_result()
        
        # Combine predictions using voting
        prediction_matrix = np.array(list(detector_predictions.values()))
        ensemble_predictions = np.mean(prediction_matrix, axis=0) >= self.config.ensemble_threshold
        
        # Calculate ensemble scores
        score_matrix = np.array(list(detector_scores.values()))
        ensemble_scores = np.mean(score_matrix, axis=0)
        
        # Normalize scores to 0-1 range
        if np.max(ensemble_scores) > np.min(ensemble_scores):
            ensemble_scores = (ensemble_scores - np.min(ensemble_scores)) / (np.max(ensemble_scores) - np.min(ensemble_scores))
        else:
            ensemble_scores = np.zeros_like(ensemble_scores)
        
        # Create anomaly results
        anomalies = []
        anomaly_scores = []
        anomaly_types = []
        severity_levels = []
        confidence_scores = []
        recommendations = []
        detection_methods = []
        
        for i, is_anomaly in enumerate(ensemble_predictions):
            if is_anomaly:
                sim_id = features_df.iloc[i]['simulation_id']
                score = float(ensemble_scores[i])
                
                # Determine anomaly type
                anomaly_type = self._classify_anomaly_type(features_df.iloc[i])
                
                # Determine severity
                severity = self._determine_severity(score, features_df.iloc[i])
                
                # Generate recommendation
                recommendation = self._generate_recommendation(anomaly_type, severity, features_df.iloc[i])
                
                # Calculate confidence based on agreement between detectors
                agreement = np.mean(prediction_matrix[:, i])
                confidence = float(agreement)
                
                anomalies.append({
                    'simulation_id': int(sim_id),
                    'features': features_df.iloc[i].to_dict(),
                    'detection_methods': list(detector_predictions.keys())
                })
                
                anomaly_scores.append(score)
                anomaly_types.append(anomaly_type)
                severity_levels.append(severity)
                confidence_scores.append(confidence)
                recommendations.append(recommendation)
                detection_methods.append('ensemble')
        
        # Sort by score
        sorted_indices = np.argsort(anomaly_scores)[::-1]
        anomalies = [anomalies[i] for i in sorted_indices]
        anomaly_scores = [anomaly_scores[i] for i in sorted_indices]
        anomaly_types = [anomaly_types[i] for i in sorted_indices]
        severity_levels = [severity_levels[i] for i in sorted_indices]
        confidence_scores = [confidence_scores[i] for i in sorted_indices]
        recommendations = [recommendations[i] for i in sorted_indices]
        detection_methods = [detection_methods[i] for i in sorted_indices]
        
        return AnomalyResult(
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            anomaly_types=anomaly_types,
            severity_levels=severity_levels,
            confidence_scores=confidence_scores,
            recommendations=recommendations,
            detection_methods=detection_methods,
            summary=self._generate_anomaly_summary(anomalies, anomaly_scores, anomaly_types)
        )
    
    def _single_method_detection(self, features_df: pd.DataFrame,
                               results: List[SimulationComparisonResult],
                               method: str) -> AnomalyResult:
        """Perform single method anomaly detection."""
        logger.debug(f"Performing {method} anomaly detection")
        
        if method not in self.detectors:
            logger.error(f"Unknown detection method: {method}")
            return self._create_empty_result()
        
        # Prepare features
        feature_cols = [col for col in features_df.columns if col != 'simulation_id']
        X = features_df[feature_cols].values
        
        # Scale features
        scaler = self.scalers['robust']
        X_scaled = scaler.fit_transform(X)
        
        # Detect anomalies
        detector = self.detectors[method]
        
        try:
            if method == 'dbscan':
                labels = detector.fit_predict(X_scaled)
                predictions = (labels == -1).astype(int)
                scores = np.zeros(len(X_scaled))
            else:
                predictions = detector.fit_predict(X_scaled)
                predictions = (predictions == -1).astype(int)
                
                if hasattr(detector, 'decision_function'):
                    scores = detector.decision_function(X_scaled)
                elif hasattr(detector, 'score_samples'):
                    scores = detector.score_samples(X_scaled)
                else:
                    scores = np.zeros(len(X_scaled))
            
            # Normalize scores
            if np.max(scores) > np.min(scores):
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                scores = np.zeros_like(scores)
            
            # Create results
            anomalies = []
            anomaly_scores = []
            anomaly_types = []
            severity_levels = []
            confidence_scores = []
            recommendations = []
            detection_methods = []
            
            for i, is_anomaly in enumerate(predictions):
                if is_anomaly:
                    sim_id = features_df.iloc[i]['simulation_id']
                    score = float(scores[i])
                    
                    anomaly_type = self._classify_anomaly_type(features_df.iloc[i])
                    severity = self._determine_severity(score, features_df.iloc[i])
                    recommendation = self._generate_recommendation(anomaly_type, severity, features_df.iloc[i])
                    
                    anomalies.append({
                        'simulation_id': int(sim_id),
                        'features': features_df.iloc[i].to_dict()
                    })
                    
                    anomaly_scores.append(score)
                    anomaly_types.append(anomaly_type)
                    severity_levels.append(severity)
                    confidence_scores.append(1.0)  # Single method confidence
                    recommendations.append(recommendation)
                    detection_methods.append(method)
            
            return AnomalyResult(
                anomalies=anomalies,
                anomaly_scores=anomaly_scores,
                anomaly_types=anomaly_types,
                severity_levels=severity_levels,
                confidence_scores=confidence_scores,
                recommendations=recommendations,
                detection_methods=detection_methods,
                summary=self._generate_anomaly_summary(anomalies, anomaly_scores, anomaly_types)
            )
            
        except Exception as e:
            logger.error(f"Error in {method} detection: {e}")
            return self._create_empty_result()
    
    def _statistical_detection(self, features_df: pd.DataFrame,
                             results: List[SimulationComparisonResult]) -> AnomalyResult:
        """Perform statistical anomaly detection."""
        logger.debug("Performing statistical anomaly detection")
        
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available for statistical detection")
            return self._create_empty_result()
        
        feature_cols = [col for col in features_df.columns if col != 'simulation_id']
        X = features_df[feature_cols].values
        
        anomalies = []
        anomaly_scores = []
        anomaly_types = []
        severity_levels = []
        confidence_scores = []
        recommendations = []
        detection_methods = []
        
        # Z-score based detection
        for i, row in enumerate(features_df.itertuples()):
            is_anomaly = False
            max_z_score = 0
            anomaly_reasons = []
            
            for col in feature_cols:
                values = features_df[col].values
                if len(values) > 2 and np.std(values) > 0:
                    z_score = abs(stats.zscore(values)[i])
                    if z_score > self.config.z_score_threshold:
                        is_anomaly = True
                        max_z_score = max(max_z_score, z_score)
                        anomaly_reasons.append(f"{col}_zscore_{z_score:.2f}")
            
            # IQR based detection
            for col in feature_cols:
                values = features_df[col].values
                if len(values) > 4:
                    Q1 = np.percentile(values, 25)
                    Q3 = np.percentile(values, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.iqr_multiplier * IQR
                    upper_bound = Q3 + self.config.iqr_multiplier * IQR
                    
                    value = values[i]
                    if value < lower_bound or value > upper_bound:
                        is_anomaly = True
                        outlier_score = abs(value - np.median(values)) / (IQR + 1e-8)
                        max_z_score = max(max_z_score, outlier_score)
                        anomaly_reasons.append(f"{col}_iqr_outlier")
            
            if is_anomaly:
                sim_id = features_df.iloc[i]['simulation_id']
                score = min(1.0, max_z_score / 5.0)  # Normalize to 0-1
                
                anomaly_type = self._classify_anomaly_type(features_df.iloc[i])
                severity = self._determine_severity(score, features_df.iloc[i])
                recommendation = self._generate_recommendation(anomaly_type, severity, features_df.iloc[i])
                
                anomalies.append({
                    'simulation_id': int(sim_id),
                    'features': features_df.iloc[i].to_dict(),
                    'anomaly_reasons': anomaly_reasons
                })
                
                anomaly_scores.append(score)
                anomaly_types.append(anomaly_type)
                severity_levels.append(severity)
                confidence_scores.append(0.8)  # Statistical confidence
                recommendations.append(recommendation)
                detection_methods.append('statistical')
        
        return AnomalyResult(
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            anomaly_types=anomaly_types,
            severity_levels=severity_levels,
            confidence_scores=confidence_scores,
            recommendations=recommendations,
            detection_methods=detection_methods,
            summary=self._generate_anomaly_summary(anomalies, anomaly_scores, anomaly_types)
        )
    
    def _classify_anomaly_type(self, features: pd.Series) -> str:
        """Classify the type of anomaly based on features."""
        # High difference anomaly
        if features.get('total_differences', 0) > 50:
            return 'high_differences'
        
        # Performance anomaly
        perf_ratios = [v for k, v in features.items() if k.startswith('perf_') and k.endswith('_ratio')]
        if perf_ratios and any(ratio > 2.0 or ratio < 0.5 for ratio in perf_ratios):
            return 'performance_anomaly'
        
        # Error anomaly
        if features.get('error_increase', 0) > 10:
            return 'error_anomaly'
        
        # Metrics anomaly
        metric_changes = [abs(v) for k, v in features.items() if k.startswith('metric_') and k.endswith('_change')]
        if metric_changes and any(change > 100 for change in metric_changes):
            return 'metrics_anomaly'
        
        # Database anomaly
        if features.get('database_differences', 0) > 20:
            return 'database_anomaly'
        
        return 'general_anomaly'
    
    def _determine_severity(self, score: float, features: pd.Series) -> str:
        """Determine anomaly severity based on score and features."""
        if score > 0.8 or features.get('total_differences', 0) > 100:
            return 'critical'
        elif score > 0.6 or features.get('total_differences', 0) > 50:
            return 'high'
        elif score > 0.4 or features.get('total_differences', 0) > 20:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendation(self, anomaly_type: str, severity: str, features: pd.Series) -> str:
        """Generate recommendation based on anomaly type and severity."""
        recommendations = {
            'high_differences': "Review simulation configuration and ensure proper setup",
            'performance_anomaly': "Investigate performance bottlenecks and optimization opportunities",
            'error_anomaly': "Check error logs and fix underlying issues",
            'metrics_anomaly': "Validate metric calculations and data sources",
            'database_anomaly': "Review database schema and data integrity",
            'general_anomaly': "Perform comprehensive analysis of simulation differences"
        }
        
        base_recommendation = recommendations.get(anomaly_type, recommendations['general_anomaly'])
        
        if severity == 'critical':
            return f"URGENT: {base_recommendation}. Immediate attention required."
        elif severity == 'high':
            return f"HIGH PRIORITY: {base_recommendation}. Address within 24 hours."
        elif severity == 'medium':
            return f"MEDIUM PRIORITY: {base_recommendation}. Address within a week."
        else:
            return f"LOW PRIORITY: {base_recommendation}. Monitor and address when convenient."
    
    def _generate_pair_recommendation(self, sim1_id: int, sim2_id: int, similarity: float, 
                                    features1: List[str], features2: List[str]) -> str:
        """Generate recommendation for simulation pair based on similarity."""
        if similarity >= 0.95:
            return f"Simulations {sim1_id} and {sim2_id} are very similar (similarity: {similarity:.2f}). Consider consolidating or investigating why they are so similar."
        elif similarity >= 0.85:
            return f"Simulations {sim1_id} and {sim2_id} are highly similar (similarity: {similarity:.2f}). Review differences in features: {', '.join(set(features1 + features2))}."
        elif similarity >= 0.75:
            return f"Simulations {sim1_id} and {sim2_id} show moderate similarity (similarity: {similarity:.2f}). Compare key differences in features: {', '.join(set(features1 + features2))}."
        else:
            return f"Simulations {sim1_id} and {sim2_id} are quite different (similarity: {similarity:.2f}). This may indicate significant changes or different simulation conditions."
    
    def _generate_anomaly_summary(self, anomalies: List[Dict], scores: List[float], 
                                types: List[str]) -> Dict[str, Any]:
        """Generate summary of anomaly detection results."""
        if not anomalies:
            return {
                'total_anomalies': 0,
                'anomaly_rate': 0.0,
                'severity_distribution': {},
                'type_distribution': {},
                'average_confidence': 0.0,
                'top_anomalies': []
            }
        
        # Severity distribution
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Type distribution
        type_counts = {}
        for anomaly_type in types:
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
        
        # Top anomalies by score
        top_anomalies = []
        if scores:
            sorted_indices = np.argsort(scores)[::-1]
            for i in sorted_indices[:5]:  # Top 5
                if i < len(anomalies):
                    top_anomalies.append({
                        'simulation_id': anomalies[i].get('simulation_id'),
                        'score': scores[i],
                        'type': types[i] if i < len(types) else 'unknown'
                    })
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / (len(anomalies) + 1),  # Approximate
            'severity_distribution': severity_counts,
            'type_distribution': type_counts,
            'average_confidence': np.mean(scores) if scores else 0.0,
            'top_anomalies': top_anomalies
        }
    
    def _create_empty_result(self) -> AnomalyResult:
        """Create empty anomaly detection result."""
        return AnomalyResult(
            anomalies=[],
            anomaly_scores=[],
            anomaly_types=[],
            severity_levels=[],
            confidence_scores=[],
            recommendations=[],
            detection_methods=[],
            summary={
                'total_anomalies': 0,
                'anomaly_rate': 0.0,
                'severity_distribution': {},
                'type_distribution': {},
                'average_confidence': 0.0,
                'top_anomalies': []
            }
        )
    
    def export_anomaly_results(self, result: AnomalyResult, 
                             output_path: Union[str, Path]) -> str:
        """Export anomaly detection results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'anomalies': result.anomalies,
            'anomaly_scores': result.anomaly_scores,
            'anomaly_types': result.anomaly_types,
            'severity_levels': result.severity_levels,
            'confidence_scores': result.confidence_scores,
            'recommendations': result.recommendations,
            'detection_methods': result.detection_methods,
            'summary': result.summary,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Anomaly detection results exported to {output_path}")
        return str(output_path)