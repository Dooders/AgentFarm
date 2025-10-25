"""
Similarity analysis and recommendation engine for simulation comparison.

This module provides advanced similarity analysis capabilities to identify
similar simulations and generate recommendations.
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
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Similarity analysis will be limited.")

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Some distance metrics will be limited.")


@dataclass
class SimilarityConfig:
    """Configuration for similarity analysis."""
    similarity_threshold: float = 0.8
    max_recommendations: int = 10
    distance_metrics: List[str] = None
    clustering_enabled: bool = True
    n_clusters: int = 5
    feature_weights: Optional[Dict[str, float]] = None
    random_state: int = 42


@dataclass
class SimilarityResult:
    """Result of similarity analysis."""
    similarity_matrix: List[List[float]]
    similar_pairs: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    summary: Dict[str, Any]


@dataclass
class SimilarityPair:
    """Similarity pair result."""
    sim1_id: int
    sim2_id: int
    similarity_score: float
    distance_metrics: Dict[str, float]
    common_features: List[str]
    different_features: List[str]
    similarity_type: str  # 'high', 'medium', 'low'
    recommendation: str


@dataclass
class Recommendation:
    """Recommendation result."""
    simulation_id: int
    recommended_simulations: List[Dict[str, Any]]
    recommendation_type: str  # 'similar', 'opposite', 'cluster_based'
    confidence: float
    reasoning: str


class SimilarityAnalyzer:
    """Advanced similarity analyzer for simulation comparison data."""
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        """Initialize similarity analyzer.
        
        Args:
            config: Configuration for similarity analysis
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for similarity analysis")
        
        self.config = config or SimilarityConfig()
        
        # Set default distance metrics
        if self.config.distance_metrics is None:
            self.config.distance_metrics = ['cosine', 'euclidean', 'manhattan']
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        logger.info("SimilarityAnalyzer initialized")
    
    def analyze_similarity(self, 
                         results: List[SimulationComparisonResult],
                         features: Optional[List[str]] = None) -> SimilarityResult:
        """Analyze similarity between simulation comparison results.
        
        Args:
            results: List of simulation comparison results
            features: List of features to use for similarity analysis
            
        Returns:
            Similarity analysis result
        """
        logger.info(f"Analyzing similarity between {len(results)} simulations")
        
        if len(results) < 2:
            logger.warning("Insufficient data for similarity analysis")
            return self._create_empty_result()
        
        # Extract features
        features_df = self._extract_features(results, features)
        
        if features_df.empty:
            logger.warning("No features extracted for similarity analysis")
            return self._create_empty_result()
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(features_df)
        
        # Find similar pairs
        similar_pairs = self._find_similar_pairs(features_df, similarity_matrix)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(features_df, similar_pairs, results)
        
        # Perform clustering if enabled
        clusters = []
        if self.config.clustering_enabled:
            clusters = self._perform_clustering(features_df, similarity_matrix)
        
        # Analyze feature importance
        feature_importance = self._analyze_feature_importance(features_df, similarity_matrix)
        
        # Generate summary
        summary = self._generate_similarity_summary(
            similar_pairs, recommendations, clusters, feature_importance
        )
        
        return SimilarityResult(
            similarity_matrix=similarity_matrix.tolist(),
            similar_pairs=similar_pairs,
            recommendations=recommendations,
            clusters=clusters,
            feature_importance=feature_importance,
            summary=summary
        )
    
    def _extract_features(self, 
                         results: List[SimulationComparisonResult],
                         features: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract features for similarity analysis."""
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
    
    def _calculate_similarity_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        """Calculate similarity matrix using multiple distance metrics."""
        logger.debug("Calculating similarity matrix")
        
        # Prepare features
        feature_cols = [col for col in features_df.columns if col != 'simulation_id']
        X = features_df[feature_cols].values
        
        # Apply feature weights if specified
        if self.config.feature_weights:
            for i, col in enumerate(feature_cols):
                if col in self.config.feature_weights:
                    X[:, i] *= self.config.feature_weights[col]
        
        # Scale features
        scaler = self.scalers['minmax']  # Use MinMax for similarity analysis
        X_scaled = scaler.fit_transform(X)
        
        # Calculate similarity matrix using cosine similarity
        similarity_matrix = cosine_similarity(X_scaled)
        
        # Ensure diagonal is 1.0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _find_similar_pairs(self, features_df: pd.DataFrame, 
                          similarity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Find similar pairs based on similarity matrix."""
        logger.debug("Finding similar pairs")
        
        similar_pairs = []
        n_simulations = len(features_df)
        
        for i in range(n_simulations):
            for j in range(i + 1, n_simulations):
                similarity_score = similarity_matrix[i, j]
                
                if similarity_score >= self.config.similarity_threshold:
                    sim1_id = int(features_df.iloc[i]['simulation_id'])
                    sim2_id = int(features_df.iloc[j]['simulation_id'])
                    
                    # Calculate additional distance metrics
                    distance_metrics = self._calculate_distance_metrics(
                        features_df.iloc[i], features_df.iloc[j]
                    )
                    
                    # Identify common and different features
                    common_features, different_features = self._compare_features(
                        features_df.iloc[i], features_df.iloc[j]
                    )
                    
                    # Determine similarity type
                    similarity_type = self._classify_similarity_type(similarity_score)
                    
                    # Generate recommendation
                    recommendation = self._generate_pair_recommendation(
                        sim1_id, sim2_id, similarity_score, common_features, different_features
                    )
                    
                    similar_pairs.append({
                        'sim1_id': sim1_id,
                        'sim2_id': sim2_id,
                        'similarity_score': float(similarity_score),
                        'distance_metrics': distance_metrics,
                        'common_features': common_features,
                        'different_features': different_features,
                        'similarity_type': similarity_type,
                        'recommendation': recommendation
                    })
        
        # Sort by similarity score
        similar_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_pairs
    
    def _calculate_distance_metrics(self, row1: pd.Series, row2: pd.Series) -> Dict[str, float]:
        """Calculate various distance metrics between two rows."""
        # Get numeric values (excluding simulation_id)
        values1 = row1.drop('simulation_id').values
        values2 = row2.drop('simulation_id').values
        
        metrics = {}
        
        # Cosine similarity
        if np.linalg.norm(values1) > 0 and np.linalg.norm(values2) > 0:
            cosine_sim = np.dot(values1, values2) / (np.linalg.norm(values1) * np.linalg.norm(values2))
            metrics['cosine_similarity'] = float(cosine_sim)
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(values1 - values2)
        metrics['euclidean_distance'] = float(euclidean_dist)
        
        # Manhattan distance
        manhattan_dist = np.sum(np.abs(values1 - values2))
        metrics['manhattan_distance'] = float(manhattan_dist)
        
        # Normalized distances (0-1 scale)
        max_euclidean = np.sqrt(len(values1))  # Maximum possible euclidean distance
        max_manhattan = np.sum(np.abs(values1)) + np.sum(np.abs(values2))  # Approximate max
        
        metrics['normalized_euclidean'] = float(euclidean_dist / max_euclidean) if max_euclidean > 0 else 0.0
        metrics['normalized_manhattan'] = float(manhattan_dist / max_manhattan) if max_manhattan > 0 else 0.0
        
        return metrics
    
    def _compare_features(self, row1: pd.Series, row2: pd.Series) -> Tuple[List[str], List[str]]:
        """Compare features between two rows."""
        common_features = []
        different_features = []
        
        for col in row1.index:
            if col == 'simulation_id':
                continue
            
            val1 = row1[col]
            val2 = row2[col]
            
            # Check if values are similar (within 10% tolerance)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 == 0 and val2 == 0:
                    common_features.append(col)
                elif val1 != 0 and val2 != 0:
                    relative_diff = abs(val1 - val2) / max(abs(val1), abs(val2))
                    if relative_diff < 0.1:  # 10% tolerance
                        common_features.append(col)
                    else:
                        different_features.append(col)
                else:
                    different_features.append(col)
            else:
                if val1 == val2:
                    common_features.append(col)
                else:
                    different_features.append(col)
        
        return common_features, different_features
    
    def _classify_similarity_type(self, similarity_score: float) -> str:
        """Classify similarity type based on score."""
        if similarity_score >= 0.9:
            return 'very_high'
        elif similarity_score >= 0.8:
            return 'high'
        elif similarity_score >= 0.7:
            return 'medium'
        elif similarity_score >= 0.6:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_pair_recommendation(self, sim1_id: int, sim2_id: int, 
                                    similarity_score: float, common_features: List[str],
                                    different_features: List[str]) -> str:
        """Generate recommendation for a similar pair."""
        if similarity_score >= 0.9:
            return f"Simulations {sim1_id} and {sim2_id} are very similar. Consider consolidating or investigating why they differ."
        elif similarity_score >= 0.8:
            return f"Simulations {sim1_id} and {sim2_id} are highly similar. Review common patterns: {', '.join(common_features[:3])}"
        else:
            return f"Simulations {sim1_id} and {sim2_id} show moderate similarity. Key differences in: {', '.join(different_features[:3])}"
    
    def _generate_recommendations(self, features_df: pd.DataFrame, 
                                similar_pairs: List[Dict[str, Any]],
                                results: List[SimulationComparisonResult]) -> List[Dict[str, Any]]:
        """Generate recommendations for each simulation."""
        logger.debug("Generating recommendations")
        
        recommendations = []
        n_simulations = len(features_df)
        
        for i in range(n_simulations):
            sim_id = int(features_df.iloc[i]['simulation_id'])
            
            # Find similar simulations
            similar_sims = []
            for pair in similar_pairs:
                if pair['sim1_id'] == sim_id:
                    similar_sims.append({
                        'simulation_id': pair['sim2_id'],
                        'similarity_score': pair['similarity_score'],
                        'similarity_type': pair['similarity_type'],
                        'common_features': pair['common_features'],
                        'different_features': pair['different_features']
                    })
                elif pair['sim2_id'] == sim_id:
                    similar_sims.append({
                        'simulation_id': pair['sim1_id'],
                        'similarity_score': pair['similarity_score'],
                        'similarity_type': pair['similarity_type'],
                        'common_features': pair['common_features'],
                        'different_features': pair['different_features']
                    })
            
            # Sort by similarity score
            similar_sims.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Limit recommendations
            similar_sims = similar_sims[:self.config.max_recommendations]
            
            # Determine recommendation type
            if similar_sims:
                if similar_sims[0]['similarity_score'] >= 0.9:
                    rec_type = 'very_similar'
                    confidence = 0.9
                elif similar_sims[0]['similarity_score'] >= 0.8:
                    rec_type = 'similar'
                    confidence = 0.8
                else:
                    rec_type = 'moderately_similar'
                    confidence = 0.6
            else:
                rec_type = 'unique'
                confidence = 0.5
            
            # Generate reasoning
            reasoning = self._generate_recommendation_reasoning(sim_id, similar_sims, rec_type)
            
            recommendations.append({
                'simulation_id': sim_id,
                'recommended_simulations': similar_sims,
                'recommendation_type': rec_type,
                'confidence': confidence,
                'reasoning': reasoning
            })
        
        return recommendations
    
    def _generate_recommendation_reasoning(self, sim_id: int, similar_sims: List[Dict[str, Any]], 
                                         rec_type: str) -> str:
        """Generate reasoning for recommendation."""
        if rec_type == 'unique':
            return f"Simulation {sim_id} appears to be unique with no highly similar counterparts."
        elif rec_type == 'very_similar':
            return f"Simulation {sim_id} has very similar counterparts. Consider consolidating or investigating differences."
        elif rec_type == 'similar':
            return f"Simulation {sim_id} has similar counterparts. Review common patterns and differences."
        else:
            return f"Simulation {sim_id} has moderately similar counterparts. Consider further analysis."
    
    def _perform_clustering(self, features_df: pd.DataFrame, 
                          similarity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Perform clustering based on similarity matrix."""
        logger.debug("Performing clustering for similarity analysis")
        
        try:
            # Use K-means clustering
            n_clusters = min(self.config.n_clusters, len(features_df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
            
            # Use similarity matrix as features
            cluster_labels = kmeans.fit_predict(similarity_matrix)
            
            # Analyze clusters
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_simulations = features_df[cluster_mask]['simulation_id'].tolist()
                
                if len(cluster_simulations) > 0:
                    # Calculate cluster characteristics
                    cluster_similarities = []
                    for i, sim_id in enumerate(cluster_simulations):
                        for j, other_sim_id in enumerate(cluster_simulations):
                            if i != j:
                                sim_idx = features_df[features_df['simulation_id'] == sim_id].index[0]
                                other_idx = features_df[features_df['simulation_id'] == other_sim_id].index[0]
                                cluster_similarities.append(similarity_matrix[sim_idx, other_idx])
                    
                    avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 0.0
                    
                    clusters.append({
                        'cluster_id': cluster_id,
                        'simulation_ids': cluster_simulations,
                        'size': len(cluster_simulations),
                        'average_similarity': float(avg_similarity),
                        'cohesion': 'high' if avg_similarity > 0.8 else 'medium' if avg_similarity > 0.6 else 'low'
                    })
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Error in clustering: {e}")
            return []
    
    def _analyze_feature_importance(self, features_df: pd.DataFrame, 
                                  similarity_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze feature importance for similarity."""
        logger.debug("Analyzing feature importance")
        
        feature_cols = [col for col in features_df.columns if col != 'simulation_id']
        feature_importance = {}
        
        for col in feature_cols:
            # Calculate correlation between feature and similarity
            feature_values = features_df[col].values
            
            # Calculate average similarity for each feature value
            unique_values = np.unique(feature_values)
            if len(unique_values) < 2:
                feature_importance[col] = 0.0
                continue
            
            similarities_by_value = []
            for value in unique_values:
                value_mask = feature_values == value
                if np.sum(value_mask) > 1:
                    # Calculate average similarity within this value group
                    value_indices = np.where(value_mask)[0]
                    group_similarities = []
                    for i in range(len(value_indices)):
                        for j in range(i + 1, len(value_indices)):
                            group_similarities.append(similarity_matrix[value_indices[i], value_indices[j]])
                    
                    if group_similarities:
                        similarities_by_value.append(np.mean(group_similarities))
            
            # Feature importance is the variance in similarities across different values
            if len(similarities_by_value) > 1:
                feature_importance[col] = float(np.var(similarities_by_value))
            else:
                feature_importance[col] = 0.0
        
        # Normalize importance scores
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                feature_importance = {k: v / max_importance for k, v in feature_importance.items()}
        
        return feature_importance
    
    def _generate_similarity_summary(self, similar_pairs: List[Dict[str, Any]],
                                   recommendations: List[Dict[str, Any]],
                                   clusters: List[Dict[str, Any]],
                                   feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of similarity analysis."""
        total_pairs = len(similar_pairs)
        high_similarity_pairs = len([p for p in similar_pairs if p['similarity_score'] >= 0.8])
        
        # Recommendation type distribution
        rec_type_counts = {}
        for rec in recommendations:
            rec_type = rec['recommendation_type']
            rec_type_counts[rec_type] = rec_type_counts.get(rec_type, 0) + 1
        
        # Cluster summary
        cluster_summary = {
            'total_clusters': len(clusters),
            'high_cohesion_clusters': len([c for c in clusters if c['cohesion'] == 'high']),
            'average_cluster_size': np.mean([c['size'] for c in clusters]) if clusters else 0
        }
        
        # Top features by importance
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_similar_pairs': total_pairs,
            'high_similarity_pairs': high_similarity_pairs,
            'similarity_rate': high_similarity_pairs / total_pairs if total_pairs > 0 else 0.0,
            'recommendation_type_distribution': rec_type_counts,
            'cluster_summary': cluster_summary,
            'top_important_features': [{'feature': f, 'importance': imp} for f, imp in top_features],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _create_empty_result(self) -> SimilarityResult:
        """Create empty similarity result."""
        return SimilarityResult(
            similarity_matrix=[],
            similar_pairs=[],
            recommendations=[],
            clusters=[],
            feature_importance={},
            summary={
                'total_similar_pairs': 0,
                'high_similarity_pairs': 0,
                'similarity_rate': 0.0,
                'recommendation_type_distribution': {},
                'cluster_summary': {'total_clusters': 0, 'high_cohesion_clusters': 0, 'average_cluster_size': 0},
                'top_important_features': [],
                'analysis_timestamp': datetime.now().isoformat()
            }
        )
    
    def export_similarity_results(self, result: SimilarityResult, 
                                output_path: Union[str, Path]) -> str:
        """Export similarity analysis results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'similarity_matrix': result.similarity_matrix,
            'similar_pairs': result.similar_pairs,
            'recommendations': result.recommendations,
            'clusters': result.clusters,
            'feature_importance': result.feature_importance,
            'summary': result.summary,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Similarity analysis results exported to {output_path}")
        return str(output_path)