"""
Machine Learning-based analysis engine for simulation comparison.

This module provides advanced ML capabilities for pattern recognition,
anomaly detection, clustering, and predictive analysis of simulation data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import pickle

from farm.analysis.comparative.comparison_result import SimulationComparisonResult
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import ML libraries with fallbacks
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML features will be limited.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Gradient boosting features will be limited.")

try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Some advanced features will be limited.")


@dataclass
class MLAnalysisResult:
    """Result of ML analysis."""
    pattern_analysis: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    clustering_results: Dict[str, Any]
    similarity_analysis: Dict[str, Any]
    predictions: Dict[str, Any]
    feature_importance: Dict[str, Any]
    summary: Dict[str, Any]


@dataclass
class PatternRecognitionResult:
    """Result of pattern recognition analysis."""
    patterns: List[Dict[str, Any]]
    pattern_types: List[str]
    confidence_scores: List[float]
    pattern_descriptions: List[str]


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""
    anomalies: List[Dict[str, Any]]
    anomaly_scores: List[float]
    anomaly_types: List[str]
    severity_levels: List[str]
    recommendations: List[str]


@dataclass
class ClusteringResult:
    """Result of clustering analysis."""
    clusters: List[Dict[str, Any]]
    cluster_labels: List[int]
    cluster_centers: List[List[float]]
    silhouette_score: float
    cluster_characteristics: List[Dict[str, Any]]


class MLAnalyzer:
    """Machine Learning analyzer for simulation comparison data."""
    
    def __init__(self, 
                 scaler_type: str = "standard",
                 random_state: int = 42,
                 model_cache_dir: Optional[Union[str, Path]] = None):
        """Initialize ML analyzer.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'minmax', 'none')
            random_state: Random state for reproducibility
            model_cache_dir: Directory to cache trained models
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ML analysis")
        
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else None
        
        if self.model_cache_dir:
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"MLAnalyzer initialized with scaler: {scaler_type}")
    
    def _initialize_models(self):
        """Initialize ML models."""
        self.anomaly_models = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=self.random_state),
            'local_outlier_factor': LocalOutlierFactor(n_neighbors=20, contamination=0.1),
            'one_class_svm': OneClassSVM(nu=0.1, kernel='rbf')
        }
        
        self.clustering_models = {
            'kmeans': KMeans(n_clusters=3, random_state=self.random_state),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'agglomerative': AgglomerativeClustering(n_clusters=3)
        }
        
        self.regression_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
        
        if XGBOOST_AVAILABLE:
            self.regression_models['xgboost'] = xgb.XGBRegressor(random_state=self.random_state)
    
    def analyze_simulation_data(self, 
                              results: List[SimulationComparisonResult],
                              analysis_config: Optional[Dict[str, Any]] = None) -> MLAnalysisResult:
        """Perform comprehensive ML analysis on simulation comparison results.
        
        Args:
            results: List of simulation comparison results
            analysis_config: Configuration for analysis parameters
            
        Returns:
            ML analysis result
        """
        logger.info(f"Starting ML analysis on {len(results)} simulation comparisons")
        
        # Extract features from results
        features_df = self._extract_features(results)
        
        if features_df.empty:
            logger.warning("No features extracted from results")
            return self._create_empty_result()
        
        # Scale features
        if self.scaler:
            features_scaled = self.scaler.fit_transform(features_df)
        else:
            features_scaled = features_df.values
        
        # Perform different types of analysis
        pattern_analysis = self._analyze_patterns(features_df, features_scaled)
        anomaly_detection = self._detect_anomalies(features_df, features_scaled)
        clustering_results = self._perform_clustering(features_df, features_scaled)
        similarity_analysis = self._analyze_similarity(features_df, features_scaled)
        predictions = self._make_predictions(features_df, features_scaled)
        feature_importance = self._analyze_feature_importance(features_df, features_scaled)
        
        # Generate summary
        summary = self._generate_ml_summary(
            pattern_analysis, anomaly_detection, clustering_results,
            similarity_analysis, predictions, feature_importance
        )
        
        return MLAnalysisResult(
            pattern_analysis=pattern_analysis,
            anomaly_detection=anomaly_detection,
            clustering_results=clustering_results,
            similarity_analysis=similarity_analysis,
            predictions=predictions,
            feature_importance=feature_importance,
            summary=summary
        )
    
    def _extract_features(self, results: List[SimulationComparisonResult]) -> pd.DataFrame:
        """Extract features from simulation comparison results."""
        features = []
        
        for i, result in enumerate(results):
            feature_row = {
                'simulation_id': i,
                'total_differences': result.comparison_summary.total_differences,
                'config_differences': result.comparison_summary.config_differences,
                'database_differences': result.comparison_summary.database_differences,
                'log_differences': result.comparison_summary.log_differences,
                'metrics_differences': result.comparison_summary.metrics_differences,
                'severity_numeric': self._severity_to_numeric(result.comparison_summary.severity)
            }
            
            # Extract metrics features
            if result.metrics_comparison.metric_differences:
                for metric, diff in result.metrics_comparison.metric_differences.items():
                    if isinstance(diff, dict) and 'percentage_change' in diff:
                        feature_row[f'metric_{metric}_change'] = diff.get('percentage_change', 0)
                        feature_row[f'metric_{metric}_abs_change'] = abs(diff.get('percentage_change', 0))
            
            # Extract performance features
            if result.metrics_comparison.performance_comparison:
                for metric, comp in result.metrics_comparison.performance_comparison.items():
                    if isinstance(comp, dict) and 'ratio' in comp:
                        feature_row[f'perf_{metric}_ratio'] = comp.get('ratio', 1.0)
                        feature_row[f'perf_{metric}_improvement'] = max(0, comp.get('ratio', 1.0) - 1.0)
            
            # Extract error features
            if result.log_comparison.error_differences:
                total_errors = 0
                error_increase = 0
                for error_type, diff in result.log_comparison.error_differences.items():
                    if isinstance(diff, dict) and 'difference' in diff:
                        error_diff = diff.get('difference', 0)
                        total_errors += abs(error_diff)
                        if error_diff > 0:
                            error_increase += error_diff
                
                feature_row['total_error_changes'] = total_errors
                feature_row['error_increase'] = error_increase
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _severity_to_numeric(self, severity: str) -> int:
        """Convert severity string to numeric value."""
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return severity_map.get(severity.lower(), 0)
    
    def _analyze_patterns(self, features_df: pd.DataFrame, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Analyze patterns in the simulation data."""
        logger.debug("Analyzing patterns in simulation data")
        
        patterns = []
        
        # Correlation patterns
        correlation_matrix = features_df.corr()
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        # Trend patterns
        trend_patterns = []
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col == 'simulation_id':
                continue
                
            values = features_df[col].values
            if len(values) > 2:
                # Calculate trend using linear regression
                x = np.arange(len(values)).reshape(-1, 1)
                y = values
                
                try:
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(x, y)
                    slope = lr.coef_[0]
                    r2 = lr.score(x, y)
                    
                    if abs(slope) > 0.1 and r2 > 0.3:  # Significant trend
                        trend_patterns.append({
                            'feature': col,
                            'trend': 'increasing' if slope > 0 else 'decreasing',
                            'slope': slope,
                            'r2_score': r2,
                            'strength': 'strong' if r2 > 0.7 else 'moderate'
                        })
                except Exception as e:
                    logger.warning(f"Error analyzing trend for {col}: {e}")
        
        # Distribution patterns
        distribution_patterns = []
        for col in numeric_columns:
            if col == 'simulation_id':
                continue
                
            values = features_df[col].values
            if len(values) > 3:
                # Check for normal distribution
                try:
                    if SCIPY_AVAILABLE:
                        from scipy import stats
                        stat, p_value = stats.normaltest(values)
                        is_normal = p_value > 0.05
                    else:
                        is_normal = False
                    
                    # Calculate skewness and kurtosis
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    skewness = np.mean(((values - mean_val) / std_val) ** 3)
                    kurtosis = np.mean(((values - mean_val) / std_val) ** 4) - 3
                    
                    distribution_patterns.append({
                        'feature': col,
                        'is_normal': is_normal,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'distribution_type': self._classify_distribution(skewness, kurtosis)
                    })
                except Exception as e:
                    logger.warning(f"Error analyzing distribution for {col}: {e}")
        
        return {
            'correlation_patterns': high_correlations,
            'trend_patterns': trend_patterns,
            'distribution_patterns': distribution_patterns,
            'total_patterns': len(high_correlations) + len(trend_patterns) + len(distribution_patterns)
        }
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution based on skewness and kurtosis."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return 'normal'
        elif skewness > 0.5:
            return 'right_skewed'
        elif skewness < -0.5:
            return 'left_skewed'
        elif kurtosis > 0.5:
            return 'heavy_tailed'
        elif kurtosis < -0.5:
            return 'light_tailed'
        else:
            return 'unknown'
    
    def _detect_anomalies(self, features_df: pd.DataFrame, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in simulation data."""
        logger.debug("Detecting anomalies in simulation data")
        
        if len(features_df) < 4:  # Need minimum samples for anomaly detection
            return {'anomalies': [], 'anomaly_scores': [], 'total_anomalies': 0}
        
        anomalies = []
        anomaly_scores = []
        
        # Use multiple anomaly detection methods
        for method_name, model in self.anomaly_models.items():
            try:
                if method_name == 'local_outlier_factor':
                    scores = model.fit_predict(features_scaled)
                    outlier_mask = scores == -1
                else:
                    scores = model.fit_predict(features_scaled)
                    outlier_mask = scores == -1
                
                # Get anomaly scores
                if hasattr(model, 'decision_function'):
                    anomaly_scores_method = model.decision_function(features_scaled)
                elif hasattr(model, 'score_samples'):
                    anomaly_scores_method = model.score_samples(features_scaled)
                else:
                    anomaly_scores_method = np.zeros(len(features_scaled))
                
                # Normalize scores to 0-1 range
                if len(anomaly_scores_method) > 0:
                    min_score = np.min(anomaly_scores_method)
                    max_score = np.max(anomaly_scores_method)
                    if max_score > min_score:
                        normalized_scores = (anomaly_scores_method - min_score) / (max_score - min_score)
                    else:
                        normalized_scores = np.zeros_like(anomaly_scores_method)
                else:
                    normalized_scores = np.zeros(len(features_scaled))
                
                # Identify anomalies
                for i, is_outlier in enumerate(outlier_mask):
                    if is_outlier:
                        anomalies.append({
                            'simulation_id': features_df.iloc[i]['simulation_id'],
                            'method': method_name,
                            'score': float(normalized_scores[i]),
                            'features': features_df.iloc[i].to_dict()
                        })
                        anomaly_scores.append(float(normalized_scores[i]))
                
            except Exception as e:
                logger.warning(f"Error in anomaly detection with {method_name}: {e}")
                continue
        
        # Remove duplicates and rank by score
        unique_anomalies = {}
        for anomaly in anomalies:
            sim_id = anomaly['simulation_id']
            if sim_id not in unique_anomalies or anomaly['score'] > unique_anomalies[sim_id]['score']:
                unique_anomalies[sim_id] = anomaly
        
        final_anomalies = list(unique_anomalies.values())
        final_anomalies.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'anomalies': final_anomalies,
            'anomaly_scores': sorted(anomaly_scores, reverse=True),
            'total_anomalies': len(final_anomalies),
            'anomaly_rate': len(final_anomalies) / len(features_df) if len(features_df) > 0 else 0
        }
    
    def _perform_clustering(self, features_df: pd.DataFrame, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Perform clustering analysis on simulation data."""
        logger.debug("Performing clustering analysis")
        
        if len(features_df) < 3:  # Need minimum samples for clustering
            return {'clusters': [], 'cluster_labels': [], 'silhouette_score': 0.0}
        
        clustering_results = {}
        
        for method_name, model in self.clustering_models.items():
            try:
                if method_name == 'dbscan':
                    cluster_labels = model.fit_predict(features_scaled)
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                else:
                    cluster_labels = model.fit_predict(features_scaled)
                    n_clusters = len(set(cluster_labels))
                
                if n_clusters < 2:
                    continue
                
                # Calculate silhouette score
                try:
                    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                except:
                    silhouette_avg = 0.0
                
                # Analyze cluster characteristics
                clusters = []
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_data = features_df[cluster_mask]
                    
                    cluster_characteristics = {
                        'cluster_id': cluster_id,
                        'size': len(cluster_data),
                        'percentage': len(cluster_data) / len(features_df) * 100,
                        'simulation_ids': cluster_data['simulation_id'].tolist()
                    }
                    
                    # Calculate cluster statistics for numeric features
                    numeric_features = cluster_data.select_dtypes(include=[np.number]).columns
                    for feature in numeric_features:
                        if feature != 'simulation_id':
                            values = cluster_data[feature].values
                            cluster_characteristics[f'{feature}_mean'] = float(np.mean(values))
                            cluster_characteristics[f'{feature}_std'] = float(np.std(values))
                            cluster_characteristics[f'{feature}_min'] = float(np.min(values))
                            cluster_characteristics[f'{feature}_max'] = float(np.max(values))
                    
                    clusters.append(cluster_characteristics)
                
                clustering_results[method_name] = {
                    'cluster_labels': cluster_labels.tolist(),
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette_avg,
                    'clusters': clusters
                }
                
            except Exception as e:
                logger.warning(f"Error in clustering with {method_name}: {e}")
                continue
        
        # Select best clustering result
        best_method = None
        best_score = -1
        
        for method, result in clustering_results.items():
            if result['silhouette_score'] > best_score:
                best_score = result['silhouette_score']
                best_method = method
        
        if best_method:
            return clustering_results[best_method]
        else:
            return {'clusters': [], 'cluster_labels': [], 'silhouette_score': 0.0}
    
    def _analyze_similarity(self, features_df: pd.DataFrame, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Analyze similarity between simulations."""
        logger.debug("Analyzing similarity between simulations")
        
        if len(features_df) < 2:
            return {'similarity_matrix': [], 'similar_pairs': [], 'average_similarity': 0.0}
        
        # Calculate pairwise similarity using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(features_scaled)
        
        # Find similar pairs (above threshold)
        similar_pairs = []
        threshold = 0.8  # Similarity threshold
        
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    similar_pairs.append({
                        'sim1_id': int(features_df.iloc[i]['simulation_id']),
                        'sim2_id': int(features_df.iloc[j]['simulation_id']),
                        'similarity': float(similarity),
                        'strength': 'high' if similarity > 0.9 else 'medium'
                    })
        
        # Sort by similarity
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Calculate average similarity
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        average_similarity = np.mean(similarity_matrix[mask])
        
        return {
            'similarity_matrix': similarity_matrix.tolist(),
            'similar_pairs': similar_pairs,
            'average_similarity': float(average_similarity),
            'total_similar_pairs': len(similar_pairs)
        }
    
    def _make_predictions(self, features_df: pd.DataFrame, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Make predictions based on simulation data."""
        logger.debug("Making predictions from simulation data")
        
        if len(features_df) < 4:  # Need minimum samples for prediction
            return {'predictions': [], 'model_performance': {}}
        
        predictions = {}
        model_performance = {}
        
        # Prepare target variables
        target_vars = ['total_differences', 'severity_numeric']
        
        for target in target_vars:
            if target not in features_df.columns:
                continue
            
            # Prepare features (exclude target and simulation_id)
            feature_cols = [col for col in features_df.columns 
                          if col not in [target, 'simulation_id'] and features_df[col].dtype in ['int64', 'float64']]
            
            if len(feature_cols) < 1:
                continue
            
            X = features_df[feature_cols].values
            y = features_df[target].values
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Train models
            for model_name, model in self.regression_models.items():
                try:
                    # Split data for validation
                    if len(X_scaled) > 3:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=0.3, random_state=self.random_state
                        )
                    else:
                        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate performance metrics
                    mse = np.mean((y_test - y_pred) ** 2)
                    r2 = model.score(X_test, y_test) if hasattr(model, 'score') else 0.0
                    
                    model_performance[f'{target}_{model_name}'] = {
                        'mse': float(mse),
                        'r2_score': float(r2),
                        'feature_importance': self._get_feature_importance(model, feature_cols)
                    }
                    
                    # Store predictions
                    predictions[f'{target}_{model_name}'] = {
                        'actual': y_test.tolist(),
                        'predicted': y_pred.tolist(),
                        'residuals': (y_test - y_pred).tolist()
                    }
                    
                except Exception as e:
                    logger.warning(f"Error training {model_name} for {target}: {e}")
                    continue
        
        return {
            'predictions': predictions,
            'model_performance': model_performance
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                return {}
            
            return dict(zip(feature_names, importances.tolist()))
        except:
            return {}
    
    def _analyze_feature_importance(self, features_df: pd.DataFrame, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance across different models."""
        logger.debug("Analyzing feature importance")
        
        if len(features_df) < 3:
            return {'feature_importance': {}, 'top_features': []}
        
        # Use Random Forest for feature importance
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Prepare data
            feature_cols = [col for col in features_df.columns 
                          if col not in ['simulation_id'] and features_df[col].dtype in ['int64', 'float64']]
            
            if len(feature_cols) < 2:
                return {'feature_importance': {}, 'top_features': []}
            
            X = features_df[feature_cols].values
            y = features_df['total_differences'].values
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf.fit(X, y)
            
            # Get feature importance
            importances = rf.feature_importances_
            feature_importance = dict(zip(feature_cols, importances.tolist()))
            
            # Get top features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [{'feature': name, 'importance': importance} 
                          for name, importance in sorted_features[:10]]
            
            return {
                'feature_importance': feature_importance,
                'top_features': top_features,
                'total_features': len(feature_cols)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing feature importance: {e}")
            return {'feature_importance': {}, 'top_features': []}
    
    def _generate_ml_summary(self, pattern_analysis: Dict, anomaly_detection: Dict,
                           clustering_results: Dict, similarity_analysis: Dict,
                           predictions: Dict, feature_importance: Dict) -> Dict[str, Any]:
        """Generate summary of ML analysis results."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_patterns_found': pattern_analysis.get('total_patterns', 0),
            'anomalies_detected': anomaly_detection.get('total_anomalies', 0),
            'anomaly_rate': anomaly_detection.get('anomaly_rate', 0.0),
            'clusters_found': clustering_results.get('n_clusters', 0),
            'clustering_quality': clustering_results.get('silhouette_score', 0.0),
            'similar_pairs': similarity_analysis.get('total_similar_pairs', 0),
            'average_similarity': similarity_analysis.get('average_similarity', 0.0),
            'models_trained': len(predictions.get('model_performance', {})),
            'top_features': len(feature_importance.get('top_features', [])),
            'analysis_quality': self._assess_analysis_quality(
                pattern_analysis, anomaly_detection, clustering_results, similarity_analysis
            )
        }
    
    def _assess_analysis_quality(self, pattern_analysis: Dict, anomaly_detection: Dict,
                                clustering_results: Dict, similarity_analysis: Dict) -> str:
        """Assess the quality of ML analysis."""
        quality_score = 0
        
        # Pattern analysis quality
        if pattern_analysis.get('total_patterns', 0) > 0:
            quality_score += 1
        
        # Anomaly detection quality
        if anomaly_detection.get('total_anomalies', 0) > 0:
            quality_score += 1
        
        # Clustering quality
        if clustering_results.get('silhouette_score', 0) > 0.5:
            quality_score += 1
        
        # Similarity analysis quality
        if similarity_analysis.get('total_similar_pairs', 0) > 0:
            quality_score += 1
        
        if quality_score >= 3:
            return 'High'
        elif quality_score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _create_empty_result(self) -> MLAnalysisResult:
        """Create empty result when no data is available."""
        return MLAnalysisResult(
            pattern_analysis={},
            anomaly_detection={},
            clustering_results={},
            similarity_analysis={},
            predictions={},
            feature_importance={},
            summary={'analysis_quality': 'Low', 'total_patterns_found': 0}
        )
    
    def save_models(self, filepath: Union[str, Path]) -> None:
        """Save trained models to file."""
        if not self.model_cache_dir:
            logger.warning("No model cache directory specified")
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'scaler': self.scaler,
            'anomaly_models': self.anomaly_models,
            'clustering_models': self.clustering_models,
            'regression_models': self.regression_models,
            'random_state': self.random_state,
            'scaler_type': self.scaler_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: Union[str, Path]) -> None:
        """Load trained models from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data.get('scaler')
            self.anomaly_models = model_data.get('anomaly_models', {})
            self.clustering_models = model_data.get('clustering_models', {})
            self.regression_models = model_data.get('regression_models', {})
            self.random_state = model_data.get('random_state', 42)
            self.scaler_type = model_data.get('scaler_type', 'standard')
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def export_analysis_results(self, result: MLAnalysisResult, 
                              output_path: Union[str, Path]) -> str:
        """Export ML analysis results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {
            'pattern_analysis': result.pattern_analysis,
            'anomaly_detection': result.anomaly_detection,
            'clustering_results': result.clustering_results,
            'similarity_analysis': result.similarity_analysis,
            'predictions': result.predictions,
            'feature_importance': result.feature_importance,
            'summary': result.summary
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"ML analysis results exported to {output_path}")
        return str(output_path)