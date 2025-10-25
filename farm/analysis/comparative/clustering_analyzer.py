"""
Clustering analysis module for simulation comparison.

This module provides advanced clustering capabilities to group similar
simulations and identify patterns in simulation behavior.
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
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Clustering will be limited.")

try:
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Hierarchical clustering will be limited.")


@dataclass
class ClusteringConfig:
    """Configuration for clustering analysis."""
    max_clusters: int = 10
    min_clusters: int = 2
    eps: float = 0.5
    min_samples: int = 5
    linkage_method: str = 'ward'
    distance_metric: str = 'euclidean'
    n_init: int = 10
    random_state: int = 42
    enable_dimensionality_reduction: bool = True
    n_components: int = 2


@dataclass
class ClusterResult:
    """Result of clustering analysis."""
    clusters: List[Dict[str, Any]]
    cluster_labels: List[int]
    cluster_centers: List[List[float]]
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    optimal_clusters: int
    cluster_characteristics: List[Dict[str, Any]]
    cluster_visualization: Dict[str, Any]
    summary: Dict[str, Any]


class ClusteringAnalyzer:
    """Advanced clustering analyzer for simulation comparison data."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """Initialize clustering analyzer.
        
        Args:
            config: Configuration for clustering analysis
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for clustering analysis")
        
        self.config = config or ClusteringConfig()
        self._initialize_clustering_models()
        
        logger.info("ClusteringAnalyzer initialized")
    
    def _initialize_clustering_models(self):
        """Initialize clustering models."""
        self.clustering_models = {
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'agglomerative': AgglomerativeClustering,
            'spectral': SpectralClustering,
            'gaussian_mixture': GaussianMixture
        }
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Initialize dimensionality reduction models
        self.dim_reduction_models = {
            'pca': PCA,
            'ica': FastICA,
            'tsne': TSNE
        }
    
    def cluster_simulations(self, 
                          results: List[SimulationComparisonResult],
                          method: str = 'auto',
                          features: Optional[List[str]] = None,
                          n_clusters: Optional[int] = None) -> ClusterResult:
        """Cluster simulation comparison results.
        
        Args:
            results: List of simulation comparison results
            method: Clustering method ('auto', 'kmeans', 'dbscan', 'agglomerative', 'spectral', 'gaussian_mixture')
            features: List of features to use for clustering
            n_clusters: Number of clusters (if applicable)
            
        Returns:
            Clustering result
        """
        logger.info(f"Clustering simulations using method: {method}")
        
        if len(results) < 3:
            logger.warning("Insufficient data for clustering")
            return self._create_empty_result()
        
        # Extract features
        features_df = self._extract_features(results, features)
        
        if features_df.empty:
            logger.warning("No features extracted for clustering")
            return self._create_empty_result()
        
        # Prepare data
        feature_cols = [col for col in features_df.columns if col != 'simulation_id']
        X = features_df[feature_cols].values
        
        # Scale features
        scaler = self.scalers['standard']
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimensionality reduction if enabled
        if self.config.enable_dimensionality_reduction and X_scaled.shape[1] > self.config.n_components:
            X_reduced = self._apply_dimensionality_reduction(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None and method != 'dbscan':
            n_clusters = self._find_optimal_clusters(X_reduced)
        
        # Perform clustering
        if method == 'auto':
            cluster_result = self._auto_clustering(X_reduced, features_df, n_clusters)
        else:
            cluster_result = self._single_method_clustering(X_reduced, features_df, method, n_clusters)
        
        return cluster_result
    
    def _extract_features(self, 
                         results: List[SimulationComparisonResult],
                         features: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract features for clustering."""
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
                metric_changes = []
                for metric, diff in result.metrics_comparison.metric_differences.items():
                    if isinstance(diff, dict):
                        change = diff.get('percentage_change', 0)
                        metric_changes.append(abs(change))
                        row[f'metric_{metric}_change'] = change
                
                if metric_changes:
                    row['avg_metric_change'] = np.mean(metric_changes)
                    row['max_metric_change'] = np.max(metric_changes)
                    row['metric_change_std'] = np.std(metric_changes)
            
            # Performance features
            if result.metrics_comparison.performance_comparison:
                perf_ratios = []
                for metric, comp in result.metrics_comparison.performance_comparison.items():
                    if isinstance(comp, dict):
                        ratio = comp.get('ratio', 1.0)
                        perf_ratios.append(ratio)
                        row[f'perf_{metric}_ratio'] = ratio
                
                if perf_ratios:
                    row['avg_perf_ratio'] = np.mean(perf_ratios)
                    row['perf_ratio_std'] = np.std(perf_ratios)
                    row['perf_improvement'] = sum(max(0, r - 1) for r in perf_ratios)
                    row['perf_degradation'] = sum(max(0, 1 - r) for r in perf_ratios)
            
            # Error features
            if result.log_comparison.error_differences:
                error_changes = []
                for error_type, diff in result.log_comparison.error_differences.items():
                    if isinstance(diff, dict):
                        change = diff.get('difference', 0)
                        error_changes.append(abs(change))
                
                if error_changes:
                    row['total_error_changes'] = sum(error_changes)
                    row['avg_error_change'] = np.mean(error_changes)
                    row['max_error_change'] = np.max(error_changes)
            
            # Database features
            if result.database_comparison.schema_differences:
                row['schema_differences'] = len(result.database_comparison.schema_differences)
            
            if result.database_comparison.data_differences:
                row['data_differences'] = len(result.database_comparison.data_differences)
            
            # Log performance features
            if result.log_comparison.performance_differences:
                log_perf_changes = []
                for perf_type, diff in result.log_comparison.performance_differences.items():
                    if isinstance(diff, dict):
                        change = diff.get('difference', 0)
                        log_perf_changes.append(abs(change))
                
                if log_perf_changes:
                    row['log_perf_changes'] = sum(log_perf_changes)
                    row['avg_log_perf_change'] = np.mean(log_perf_changes)
            
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
    
    def _apply_dimensionality_reduction(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to features."""
        logger.debug("Applying dimensionality reduction")
        
        try:
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=self.config.n_components, random_state=self.config.random_state)
            X_reduced = pca.fit_transform(X)
            
            # Log explained variance ratio
            explained_variance = pca.explained_variance_ratio_
            logger.debug(f"PCA explained variance ratio: {explained_variance}")
            
            return X_reduced
            
        except Exception as e:
            logger.warning(f"Error in dimensionality reduction: {e}")
            return X
    
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        """Find optimal number of clusters using multiple methods."""
        logger.debug("Finding optimal number of clusters")
        
        if len(X) < 4:
            return 2
        
        max_clusters = min(self.config.max_clusters, len(X) - 1)
        min_clusters = max(self.config.min_clusters, 2)
        
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                # K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init=self.config.n_init)
                cluster_labels = kmeans.fit_predict(X)
                
                # Calculate metrics
                silhouette_avg = silhouette_score(X, cluster_labels)
                calinski_avg = calinski_harabasz_score(X, cluster_labels)
                davies_bouldin_avg = davies_bouldin_score(X, cluster_labels)
                
                silhouette_scores.append(silhouette_avg)
                calinski_scores.append(calinski_avg)
                davies_bouldin_scores.append(davies_bouldin_avg)
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for {n_clusters} clusters: {e}")
                continue
        
        if not silhouette_scores:
            return min_clusters
        
        # Find optimal number of clusters based on silhouette score
        optimal_clusters = min_clusters + np.argmax(silhouette_scores)
        
        logger.debug(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
    def _auto_clustering(self, X: np.ndarray, features_df: pd.DataFrame, n_clusters: int) -> ClusterResult:
        """Perform automatic clustering using the best method."""
        logger.debug("Performing automatic clustering")
        
        best_result = None
        best_score = -1
        best_method = None
        
        # Try different clustering methods
        methods_to_try = ['kmeans', 'agglomerative', 'spectral', 'gaussian_mixture']
        
        for method in methods_to_try:
            try:
                result = self._single_method_clustering(X, features_df, method, n_clusters)
                if result.silhouette_score > best_score:
                    best_score = result.silhouette_score
                    best_result = result
                    best_method = method
            except Exception as e:
                logger.warning(f"Error with {method} clustering: {e}")
                continue
        
        if best_result is None:
            logger.warning("All clustering methods failed, using K-means")
            best_result = self._single_method_clustering(X, features_df, 'kmeans', n_clusters)
            best_method = 'kmeans'
        
        logger.info(f"Best clustering method: {best_method} with silhouette score: {best_score:.3f}")
        return best_result
    
    def _single_method_clustering(self, X: np.ndarray, features_df: pd.DataFrame, 
                                method: str, n_clusters: int) -> ClusterResult:
        """Perform clustering using a single method."""
        logger.debug(f"Performing {method} clustering")
        
        if method not in self.clustering_models:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Initialize clustering model
        if method == 'dbscan':
            model = self.clustering_models[method](
                eps=self.config.eps,
                min_samples=self.config.min_samples
            )
        elif method == 'gaussian_mixture':
            model = self.clustering_models[method](
                n_components=n_clusters,
                random_state=self.config.random_state
            )
        else:
            model = self.clustering_models[method](
                n_clusters=n_clusters,
                random_state=self.config.random_state
            )
        
        # Fit clustering model
        if method == 'gaussian_mixture':
            cluster_labels = model.fit_predict(X)
            cluster_centers = model.means_
        else:
            cluster_labels = model.fit_predict(X)
            if hasattr(model, 'cluster_centers_'):
                cluster_centers = model.cluster_centers_.tolist()
            else:
                cluster_centers = []
        
        # Calculate clustering metrics
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(X, cluster_labels)
            calinski_avg = calinski_harabasz_score(X, cluster_labels)
            davies_bouldin_avg = davies_bouldin_score(X, cluster_labels)
        else:
            silhouette_avg = 0.0
            calinski_avg = 0.0
            davies_bouldin_avg = 0.0
        
        # Analyze cluster characteristics
        clusters = self._analyze_clusters(features_df, cluster_labels, cluster_centers)
        
        # Generate cluster visualization data
        visualization_data = self._generate_cluster_visualization(X, cluster_labels, features_df)
        
        # Generate summary
        summary = self._generate_clustering_summary(
            clusters, cluster_labels, silhouette_avg, calinski_avg, davies_bouldin_avg
        )
        
        return ClusterResult(
            clusters=clusters,
            cluster_labels=cluster_labels.tolist(),
            cluster_centers=cluster_centers,
            silhouette_score=silhouette_avg,
            calinski_harabasz_score=calinski_avg,
            davies_bouldin_score=davies_bouldin_avg,
            optimal_clusters=n_clusters,
            cluster_characteristics=clusters,
            cluster_visualization=visualization_data,
            summary=summary
        )
    
    def _analyze_clusters(self, features_df: pd.DataFrame, cluster_labels: np.ndarray, 
                         cluster_centers: List[List[float]]) -> List[Dict[str, Any]]:
        """Analyze characteristics of each cluster."""
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_data = features_df[cluster_mask]
            
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features_df) * 100,
                'simulation_ids': cluster_data['simulation_id'].tolist()
            }
            
            # Calculate cluster statistics for numeric features
            numeric_features = cluster_data.select_dtypes(include=[np.number]).columns
            for feature in numeric_features:
                if feature != 'simulation_id':
                    values = cluster_data[feature].values
                    cluster_info[f'{feature}_mean'] = float(np.mean(values))
                    cluster_info[f'{feature}_std'] = float(np.std(values))
                    cluster_info[f'{feature}_min'] = float(np.min(values))
                    cluster_info[f'{feature}_max'] = float(np.max(values))
                    cluster_info[f'{feature}_median'] = float(np.median(values))
            
            # Identify cluster characteristics
            cluster_info['characteristics'] = self._identify_cluster_characteristics(cluster_data)
            
            # Calculate cluster center if available
            if cluster_id < len(cluster_centers):
                cluster_info['center'] = cluster_centers[cluster_id]
            
            clusters.append(cluster_info)
        
        return clusters
    
    def _identify_cluster_characteristics(self, cluster_data: pd.DataFrame) -> List[str]:
        """Identify key characteristics of a cluster."""
        characteristics = []
        
        # High differences cluster
        if cluster_data['total_differences'].mean() > 50:
            characteristics.append('high_differences')
        
        # Performance cluster
        perf_ratios = [col for col in cluster_data.columns if col.startswith('perf_') and col.endswith('_ratio')]
        if perf_ratios and cluster_data[perf_ratios].mean().mean() > 1.5:
            characteristics.append('high_performance')
        elif perf_ratios and cluster_data[perf_ratios].mean().mean() < 0.7:
            characteristics.append('low_performance')
        
        # Error cluster
        if 'total_error_changes' in cluster_data.columns and cluster_data['total_error_changes'].mean() > 10:
            characteristics.append('high_errors')
        
        # Metrics cluster
        metric_changes = [col for col in cluster_data.columns if col.startswith('metric_') and col.endswith('_change')]
        if metric_changes and cluster_data[metric_changes].abs().mean().mean() > 50:
            characteristics.append('high_metric_changes')
        
        # Database cluster
        if 'database_differences' in cluster_data.columns and cluster_data['database_differences'].mean() > 20:
            characteristics.append('high_database_differences')
        
        # Low differences cluster
        if cluster_data['total_differences'].mean() < 10:
            characteristics.append('low_differences')
        
        return characteristics if characteristics else ['general']
    
    def _generate_cluster_visualization(self, X: np.ndarray, cluster_labels: np.ndarray, 
                                      features_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate visualization data for clusters."""
        visualization = {}
        
        try:
            # If data is high-dimensional, apply t-SNE for visualization
            if X.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=self.config.random_state, perplexity=min(30, len(X)-1))
                X_2d = tsne.fit_transform(X)
            else:
                X_2d = X
            
            # Create visualization data
            visualization['coordinates'] = X_2d.tolist()
            visualization['cluster_labels'] = cluster_labels.tolist()
            visualization['simulation_ids'] = features_df['simulation_id'].tolist()
            
            # Calculate cluster boundaries (convex hull approximation)
            unique_labels = np.unique(cluster_labels)
            cluster_boundaries = {}
            
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_points = X_2d[cluster_mask]
                
                if len(cluster_points) > 2:
                    # Simple bounding box for cluster
                    min_x, min_y = np.min(cluster_points, axis=0)
                    max_x, max_y = np.max(cluster_points, axis=0)
                    
                    cluster_boundaries[cluster_id] = {
                        'min_x': float(min_x),
                        'min_y': float(min_y),
                        'max_x': float(max_x),
                        'max_y': float(max_y),
                        'center_x': float(np.mean(cluster_points[:, 0])),
                        'center_y': float(np.mean(cluster_points[:, 1]))
                    }
            
            visualization['cluster_boundaries'] = cluster_boundaries
            
        except Exception as e:
            logger.warning(f"Error generating cluster visualization: {e}")
            visualization = {'error': str(e)}
        
        return visualization
    
    def _generate_clustering_summary(self, clusters: List[Dict], cluster_labels: np.ndarray,
                                   silhouette_score: float, calinski_score: float, 
                                   davies_bouldin_score: float) -> Dict[str, Any]:
        """Generate summary of clustering results."""
        unique_labels = np.unique(cluster_labels)
        n_clusters = len([label for label in unique_labels if label != -1])
        
        # Cluster size distribution
        cluster_sizes = [cluster['size'] for cluster in clusters]
        
        # Cluster characteristics distribution
        all_characteristics = []
        for cluster in clusters:
            all_characteristics.extend(cluster.get('characteristics', []))
        
        characteristic_counts = {}
        for char in all_characteristics:
            characteristic_counts[char] = characteristic_counts.get(char, 0) + 1
        
        return {
            'total_clusters': n_clusters,
            'total_simulations': len(cluster_labels),
            'silhouette_score': silhouette_score,
            'calinski_harabasz_score': calinski_score,
            'davies_bouldin_score': davies_bouldin_score,
            'clustering_quality': self._assess_clustering_quality(silhouette_score, calinski_score, davies_bouldin_score),
            'cluster_size_distribution': {
                'min_size': min(cluster_sizes) if cluster_sizes else 0,
                'max_size': max(cluster_sizes) if cluster_sizes else 0,
                'avg_size': np.mean(cluster_sizes) if cluster_sizes else 0,
                'std_size': np.std(cluster_sizes) if cluster_sizes else 0
            },
            'characteristic_distribution': characteristic_counts,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _assess_clustering_quality(self, silhouette_score: float, calinski_score: float, 
                                 davies_bouldin_score: float) -> str:
        """Assess the quality of clustering results."""
        quality_score = 0
        
        # Silhouette score assessment
        if silhouette_score > 0.7:
            quality_score += 3
        elif silhouette_score > 0.5:
            quality_score += 2
        elif silhouette_score > 0.3:
            quality_score += 1
        
        # Calinski-Harabasz score assessment (higher is better)
        if calinski_score > 100:
            quality_score += 2
        elif calinski_score > 50:
            quality_score += 1
        
        # Davies-Bouldin score assessment (lower is better)
        if davies_bouldin_score < 1.0:
            quality_score += 2
        elif davies_bouldin_score < 2.0:
            quality_score += 1
        
        if quality_score >= 6:
            return 'Excellent'
        elif quality_score >= 4:
            return 'Good'
        elif quality_score >= 2:
            return 'Fair'
        else:
            return 'Poor'
    
    def _create_empty_result(self) -> ClusterResult:
        """Create empty clustering result."""
        return ClusterResult(
            clusters=[],
            cluster_labels=[],
            cluster_centers=[],
            silhouette_score=0.0,
            calinski_harabasz_score=0.0,
            davies_bouldin_score=0.0,
            optimal_clusters=0,
            cluster_characteristics=[],
            cluster_visualization={},
            summary={
                'total_clusters': 0,
                'total_simulations': 0,
                'clustering_quality': 'Poor',
                'analysis_timestamp': datetime.now().isoformat()
            }
        )
    
    def export_clustering_results(self, result: ClusterResult, 
                                output_path: Union[str, Path]) -> str:
        """Export clustering results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'clusters': result.clusters,
            'cluster_labels': result.cluster_labels,
            'cluster_centers': result.cluster_centers,
            'silhouette_score': result.silhouette_score,
            'calinski_harabasz_score': result.calinski_harabasz_score,
            'davies_bouldin_score': result.davies_bouldin_score,
            'optimal_clusters': result.optimal_clusters,
            'cluster_characteristics': result.cluster_characteristics,
            'cluster_visualization': result.cluster_visualization,
            'summary': result.summary,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Clustering results exported to {output_path}")
        return str(output_path)