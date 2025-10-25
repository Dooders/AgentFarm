"""
ML-specific visualization tools for simulation comparison.

This module provides advanced visualization capabilities for ML analysis results,
including clustering, anomaly detection, and trend prediction visualizations.
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

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. ML visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available. Interactive visualizations will be limited.")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Dimensionality reduction will be limited.")


@dataclass
class MLVisualizationConfig:
    """Configuration for ML visualizations."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = 'whitegrid'
    color_palette: str = 'husl'
    interactive: bool = True
    save_format: str = 'png'
    output_dir: Optional[Union[str, Path]] = None


class MLVisualizer:
    """Advanced ML visualizer for simulation comparison data."""
    
    def __init__(self, config: Optional[MLVisualizationConfig] = None):
        """Initialize ML visualizer.
        
        Args:
            config: Configuration for visualizations
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for ML visualization")
        
        self.config = config or MLVisualizationConfig()
        self.output_dir = Path(self.config.output_dir) if self.config.output_dir else Path("ml_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        plt.rcParams['figure.dpi'] = self.config.dpi
        plt.rcParams['savefig.dpi'] = self.config.dpi
        
        logger.info("MLVisualizer initialized")
    
    def create_ml_dashboard(self, 
                          ml_results: Dict[str, Any],
                          comparison_results: List[SimulationComparisonResult],
                          title: str = "ML Analysis Dashboard") -> Dict[str, str]:
        """Create comprehensive ML analysis dashboard.
        
        Args:
            ml_results: Dictionary containing ML analysis results
            comparison_results: List of simulation comparison results
            title: Dashboard title
            
        Returns:
            Dictionary of generated visualization files
        """
        logger.info("Creating ML analysis dashboard")
        
        dashboard_files = {}
        
        try:
            # Clustering visualizations
            if 'clustering' in ml_results:
                dashboard_files['clustering_plot'] = self.create_clustering_visualization(
                    ml_results['clustering'], comparison_results
                )
                dashboard_files['cluster_characteristics'] = self.create_cluster_characteristics_plot(
                    ml_results['clustering']
                )
            
            # Anomaly detection visualizations
            if 'anomaly_detection' in ml_results:
                dashboard_files['anomaly_plot'] = self.create_anomaly_visualization(
                    ml_results['anomaly_detection'], comparison_results
                )
                dashboard_files['anomaly_heatmap'] = self.create_anomaly_heatmap(
                    ml_results['anomaly_detection']
                )
            
            # Trend prediction visualizations
            if 'trend_prediction' in ml_results:
                dashboard_files['trend_plot'] = self.create_trend_visualization(
                    ml_results['trend_prediction']
                )
                dashboard_files['forecast_plot'] = self.create_forecast_visualization(
                    ml_results['trend_prediction']
                )
            
            # Similarity analysis visualizations
            if 'similarity_analysis' in ml_results:
                dashboard_files['similarity_heatmap'] = self.create_similarity_heatmap(
                    ml_results['similarity_analysis']
                )
                dashboard_files['similarity_network'] = self.create_similarity_network(
                    ml_results['similarity_analysis']
                )
            
            # Feature importance visualization
            if 'feature_importance' in ml_results:
                dashboard_files['feature_importance'] = self.create_feature_importance_plot(
                    ml_results['feature_importance']
                )
            
            # Combined overview
            dashboard_files['ml_overview'] = self.create_ml_overview_plot(
                ml_results, comparison_results, title
            )
            
            logger.info(f"ML dashboard created with {len(dashboard_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error creating ML dashboard: {e}")
        
        return dashboard_files
    
    def create_clustering_visualization(self, 
                                      clustering_result: Dict[str, Any],
                                      comparison_results: List[SimulationComparisonResult]) -> str:
        """Create clustering visualization."""
        logger.debug("Creating clustering visualization")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle('Clustering Analysis', fontsize=16, fontweight='bold')
            
            # 1. Cluster distribution
            ax1 = axes[0, 0]
            cluster_sizes = [cluster['size'] for cluster in clustering_result.get('clusters', [])]
            cluster_ids = [cluster['cluster_id'] for cluster in clustering_result.get('clusters', [])]
            
            if cluster_sizes:
                bars = ax1.bar(cluster_ids, cluster_sizes, color=plt.cm.Set3(np.linspace(0, 1, len(cluster_ids))))
                ax1.set_xlabel('Cluster ID')
                ax1.set_ylabel('Number of Simulations')
                ax1.set_title('Cluster Size Distribution')
                
                # Add value labels on bars
                for bar, size in zip(bars, cluster_sizes):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(size), ha='center', va='bottom')
            
            # 2. Silhouette score
            ax2 = axes[0, 1]
            silhouette_score = clustering_result.get('silhouette_score', 0)
            calinski_score = clustering_result.get('calinski_harabasz_score', 0)
            davies_bouldin_score = clustering_result.get('davies_bouldin_score', 0)
            
            metrics = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
            scores = [silhouette_score, calinski_score, 1 - davies_bouldin_score]  # Invert Davies-Bouldin
            
            bars = ax2.bar(metrics, scores, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax2.set_ylabel('Score')
            ax2.set_title('Clustering Quality Metrics')
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            # 3. Cluster characteristics heatmap
            ax3 = axes[1, 0]
            if clustering_result.get('clusters'):
                # Create characteristics matrix
                characteristics = []
                cluster_labels = []
                
                for cluster in clustering_result['clusters']:
                    cluster_labels.append(f"Cluster {cluster['cluster_id']}")
                    char_vector = []
                    
                    # Extract numeric characteristics
                    for key, value in cluster.items():
                        if isinstance(value, (int, float)) and key not in ['cluster_id', 'size']:
                            char_vector.append(value)
                    
                    if char_vector:
                        characteristics.append(char_vector)
                
                if characteristics:
                    im = ax3.imshow(characteristics, cmap='viridis', aspect='auto')
                    ax3.set_xticks(range(len(characteristics[0]) if characteristics else 0))
                    ax3.set_yticks(range(len(cluster_labels)))
                    ax3.set_yticklabels(cluster_labels)
                    ax3.set_title('Cluster Characteristics')
                    plt.colorbar(im, ax=ax3)
            
            # 4. Cluster visualization (2D projection)
            ax4 = axes[1, 1]
            if 'cluster_visualization' in clustering_result:
                viz_data = clustering_result['cluster_visualization']
                if 'coordinates' in viz_data and 'cluster_labels' in viz_data:
                    coords = np.array(viz_data['coordinates'])
                    labels = np.array(viz_data['cluster_labels'])
                    
                    unique_labels = np.unique(labels)
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                    
                    for i, label in enumerate(unique_labels):
                        if label == -1:  # Noise points
                            ax4.scatter(coords[labels == label, 0], coords[labels == label, 1],
                                      c='black', marker='x', s=50, alpha=0.6, label='Noise')
                        else:
                            ax4.scatter(coords[labels == label, 0], coords[labels == label, 1],
                                      c=[colors[i]], label=f'Cluster {label}', s=100, alpha=0.7)
                    
                    ax4.set_xlabel('Dimension 1')
                    ax4.set_ylabel('Dimension 2')
                    ax4.set_title('Cluster Visualization (2D Projection)')
                    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"clustering_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating clustering visualization: {e}")
            return ""
    
    def create_anomaly_visualization(self, 
                                   anomaly_result: Dict[str, Any],
                                   comparison_results: List[SimulationComparisonResult]) -> str:
        """Create anomaly detection visualization."""
        logger.debug("Creating anomaly visualization")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle('Anomaly Detection Analysis', fontsize=16, fontweight='bold')
            
            # 1. Anomaly distribution by severity
            ax1 = axes[0, 0]
            severity_counts = anomaly_result.get('summary', {}).get('severity_distribution', {})
            
            if severity_counts:
                severities = list(severity_counts.keys())
                counts = list(severity_counts.values())
                colors = ['green', 'yellow', 'orange', 'red'][:len(severities)]
                
                bars = ax1.bar(severities, counts, color=colors)
                ax1.set_xlabel('Severity Level')
                ax1.set_ylabel('Number of Anomalies')
                ax1.set_title('Anomaly Distribution by Severity')
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
            
            # 2. Anomaly scores distribution
            ax2 = axes[0, 1]
            anomaly_scores = anomaly_result.get('anomaly_scores', [])
            
            if anomaly_scores:
                ax2.hist(anomaly_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(np.mean(anomaly_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(anomaly_scores):.3f}')
                ax2.set_xlabel('Anomaly Score')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Anomaly Scores Distribution')
                ax2.legend()
            
            # 3. Anomaly types
            ax3 = axes[1, 0]
            type_counts = anomaly_result.get('summary', {}).get('type_distribution', {})
            
            if type_counts:
                types = list(type_counts.keys())
                counts = list(type_counts.values())
                
                wedges, texts, autotexts = ax3.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Anomaly Types Distribution')
            
            # 4. Anomaly timeline
            ax4 = axes[1, 1]
            anomalies = anomaly_result.get('anomalies', [])
            
            if anomalies:
                # Create timeline plot
                sim_ids = [anomaly.get('simulation_id', 0) for anomaly in anomalies]
                scores = [anomaly.get('score', 0) for anomaly in anomalies]
                
                scatter = ax4.scatter(sim_ids, scores, c=scores, cmap='Reds', s=100, alpha=0.7)
                ax4.set_xlabel('Simulation ID')
                ax4.set_ylabel('Anomaly Score')
                ax4.set_title('Anomaly Timeline')
                plt.colorbar(scatter, ax=ax4, label='Anomaly Score')
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"anomaly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating anomaly visualization: {e}")
            return ""
    
    def create_trend_visualization(self, trend_result: Dict[str, Any]) -> str:
        """Create trend prediction visualization."""
        logger.debug("Creating trend visualization")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle('Trend Prediction Analysis', fontsize=16, fontweight='bold')
            
            # 1. Trend directions
            ax1 = axes[0, 0]
            trend_analysis = trend_result.get('trend_analysis', {})
            
            directions = []
            strengths = []
            confidences = []
            
            for feature, analysis in trend_analysis.items():
                directions.append(analysis.get('trend_direction', 'stable'))
                strengths.append(analysis.get('trend_strength', 0))
                confidences.append(analysis.get('trend_confidence', 0))
            
            if directions:
                direction_counts = pd.Series(directions).value_counts()
                colors = ['green' if d == 'increasing' else 'red' if d == 'decreasing' else 'gray' 
                         for d in direction_counts.index]
                
                bars = ax1.bar(direction_counts.index, direction_counts.values, color=colors)
                ax1.set_xlabel('Trend Direction')
                ax1.set_ylabel('Number of Features')
                ax1.set_title('Trend Directions Distribution')
                
                # Add value labels
                for bar, count in zip(bars, direction_counts.values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
            
            # 2. Trend strength vs confidence
            ax2 = axes[0, 1]
            if strengths and confidences:
                scatter = ax2.scatter(strengths, confidences, c=confidences, cmap='viridis', s=100, alpha=0.7)
                ax2.set_xlabel('Trend Strength')
                ax2.set_ylabel('Trend Confidence')
                ax2.set_title('Trend Strength vs Confidence')
                plt.colorbar(scatter, ax=ax2, label='Confidence')
                
                # Add quadrant lines
                ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
                ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
            
            # 3. Forecast accuracy
            ax3 = axes[1, 0]
            forecast_accuracy = trend_result.get('forecast_accuracy', {})
            
            if forecast_accuracy:
                features = list(forecast_accuracy.keys())
                accuracies = [metrics.get('trend_accuracy', 0) for metrics in forecast_accuracy.values()]
                
                bars = ax3.bar(features, accuracies, color='lightblue')
                ax3.set_xlabel('Features')
                ax3.set_ylabel('Forecast Accuracy')
                ax3.set_title('Forecast Accuracy by Feature')
                ax3.set_ylim(0, 1)
                
                # Rotate x-axis labels
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # 4. Prediction intervals
            ax4 = axes[1, 1]
            predictions = trend_result.get('predictions', {})
            
            if predictions:
                # Plot prediction intervals for first feature
                first_feature = list(predictions.keys())[0]
                pred_data = predictions[first_feature]
                
                future_values = pred_data.get('future_values', [])
                intervals = pred_data.get('prediction_intervals', {})
                
                if future_values and intervals:
                    x = range(len(future_values))
                    ax4.plot(x, future_values, 'b-', label='Predicted', linewidth=2)
                    
                    lower = intervals.get('lower_bound', [])
                    upper = intervals.get('upper_bound', [])
                    
                    if lower and upper:
                        ax4.fill_between(x, lower, upper, alpha=0.3, color='blue', label='Confidence Interval')
                    
                    ax4.set_xlabel('Time Steps')
                    ax4.set_ylabel('Value')
                    ax4.set_title(f'Forecast for {first_feature}')
                    ax4.legend()
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating trend visualization: {e}")
            return ""
    
    def create_similarity_heatmap(self, similarity_result: Dict[str, Any]) -> str:
        """Create similarity heatmap visualization."""
        logger.debug("Creating similarity heatmap")
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('Similarity Analysis', fontsize=16, fontweight='bold')
            
            # 1. Similarity matrix heatmap
            ax1 = axes[0]
            similarity_matrix = similarity_result.get('similarity_matrix', [])
            
            if similarity_matrix:
                im = ax1.imshow(similarity_matrix, cmap='viridis', aspect='auto')
                ax1.set_xlabel('Simulation ID')
                ax1.set_ylabel('Simulation ID')
                ax1.set_title('Similarity Matrix Heatmap')
                plt.colorbar(im, ax=ax1, label='Similarity Score')
            
            # 2. Similar pairs distribution
            ax2 = axes[1]
            similar_pairs = similarity_result.get('similar_pairs', [])
            
            if similar_pairs:
                scores = [pair.get('similarity_score', 0) for pair in similar_pairs]
                types = [pair.get('similarity_type', 'unknown') for pair in similar_pairs]
                
                # Create histogram by similarity type
                type_counts = pd.Series(types).value_counts()
                colors = ['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(type_counts)]
                
                bars = ax2.bar(type_counts.index, type_counts.values, color=colors)
                ax2.set_xlabel('Similarity Type')
                ax2.set_ylabel('Number of Pairs')
                ax2.set_title('Similar Pairs Distribution')
                
                # Add value labels
                for bar, count in zip(bars, type_counts.values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"similarity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating similarity heatmap: {e}")
            return ""
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float]) -> str:
        """Create feature importance visualization."""
        logger.debug("Creating feature importance plot")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if feature_importance:
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                features = [item[0] for item in sorted_features]
                importances = [item[1] for item in sorted_features]
                
                # Create horizontal bar plot
                bars = ax.barh(features, importances, color='skyblue')
                ax.set_xlabel('Feature Importance')
                ax.set_title('Feature Importance for ML Analysis')
                
                # Add value labels
                for i, (bar, importance) in enumerate(zip(bars, importances)):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{importance:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return ""
    
    def create_ml_overview_plot(self, 
                              ml_results: Dict[str, Any],
                              comparison_results: List[SimulationComparisonResult],
                              title: str) -> str:
        """Create ML analysis overview plot."""
        logger.debug("Creating ML overview plot")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(title, fontsize=20, fontweight='bold')
            
            # 1. ML Analysis Summary
            ax1 = axes[0, 0]
            analysis_types = []
            counts = []
            
            if 'clustering' in ml_results:
                analysis_types.append('Clustering')
                counts.append(ml_results['clustering'].get('summary', {}).get('total_clusters', 0))
            
            if 'anomaly_detection' in ml_results:
                analysis_types.append('Anomalies')
                counts.append(ml_results['anomaly_detection'].get('summary', {}).get('total_anomalies', 0))
            
            if 'similarity_analysis' in ml_results:
                analysis_types.append('Similar Pairs')
                counts.append(ml_results['similarity_analysis'].get('summary', {}).get('total_similar_pairs', 0))
            
            if 'trend_prediction' in ml_results:
                analysis_types.append('Trends')
                counts.append(ml_results['trend_prediction'].get('summary', {}).get('strong_trends_detected', 0))
            
            if analysis_types:
                bars = ax1.bar(analysis_types, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
                ax1.set_ylabel('Count')
                ax1.set_title('ML Analysis Summary')
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
            
            # 2. Data Quality Metrics
            ax2 = axes[0, 1]
            quality_metrics = []
            quality_values = []
            
            if 'clustering' in ml_results:
                quality_metrics.append('Clustering\nQuality')
                quality_values.append(ml_results['clustering'].get('silhouette_score', 0))
            
            if 'anomaly_detection' in ml_results:
                quality_metrics.append('Anomaly\nDetection')
                quality_values.append(ml_results['anomaly_detection'].get('summary', {}).get('anomaly_rate', 0))
            
            if 'similarity_analysis' in ml_results:
                quality_metrics.append('Similarity\nRate')
                quality_values.append(ml_results['similarity_analysis'].get('summary', {}).get('similarity_rate', 0))
            
            if 'trend_prediction' in ml_results:
                quality_metrics.append('Forecast\nAccuracy')
                quality_values.append(ml_results['trend_prediction'].get('summary', {}).get('average_forecast_accuracy', 0))
            
            if quality_metrics:
                bars = ax2.bar(quality_metrics, quality_values, color='lightblue')
                ax2.set_ylabel('Score')
                ax2.set_title('Data Quality Metrics')
                ax2.set_ylim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, quality_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # 3. Feature Distribution
            ax3 = axes[0, 2]
            if comparison_results:
                # Extract basic statistics
                total_diffs = [r.comparison_summary.total_differences for r in comparison_results]
                config_diffs = [r.comparison_summary.config_differences for r in comparison_results]
                db_diffs = [r.comparison_summary.database_differences for r in comparison_results]
                
                ax3.hist(total_diffs, bins=10, alpha=0.7, label='Total', color='blue')
                ax3.hist(config_diffs, bins=10, alpha=0.7, label='Config', color='red')
                ax3.hist(db_diffs, bins=10, alpha=0.7, label='Database', color='green')
                ax3.set_xlabel('Number of Differences')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Difference Distribution')
                ax3.legend()
            
            # 4. ML Model Performance
            ax4 = axes[1, 0]
            if 'trend_prediction' in ml_results:
                model_performance = ml_results['trend_prediction'].get('model_performance', {})
                if model_performance:
                    models = list(model_performance.keys())
                    confidences = [perf.get('trend_confidence', 0) for perf in model_performance.values()]
                    
                    bars = ax4.bar(models, confidences, color='lightgreen')
                    ax4.set_ylabel('Confidence')
                    ax4.set_title('Model Performance')
                    ax4.set_ylim(0, 1)
                    
                    # Rotate x-axis labels
                    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            
            # 5. Recommendations Summary
            ax5 = axes[1, 1]
            recommendations = []
            
            if 'anomaly_detection' in ml_results:
                recommendations.extend(ml_results['anomaly_detection'].get('recommendations', []))
            
            if 'trend_prediction' in ml_results:
                recommendations.extend(ml_results['trend_prediction'].get('recommendations', []))
            
            if recommendations:
                # Count recommendation types
                rec_types = {}
                for rec in recommendations:
                    if 'URGENT' in rec or 'HIGH PRIORITY' in rec:
                        rec_types['High Priority'] = rec_types.get('High Priority', 0) + 1
                    elif 'MEDIUM PRIORITY' in rec:
                        rec_types['Medium Priority'] = rec_types.get('Medium Priority', 0) + 1
                    else:
                        rec_types['Low Priority'] = rec_types.get('Low Priority', 0) + 1
                
                if rec_types:
                    types = list(rec_types.keys())
                    counts = list(rec_types.values())
                    colors = ['red', 'orange', 'green'][:len(types)]
                    
                    bars = ax5.bar(types, counts, color=colors)
                    ax5.set_ylabel('Number of Recommendations')
                    ax5.set_title('Recommendations by Priority')
            
            # 6. Analysis Timeline
            ax6 = axes[1, 2]
            if comparison_results:
                # Create timeline of analysis
                timestamps = []
                for result in comparison_results:
                    if hasattr(result.comparison_summary, 'comparison_time'):
                        timestamp = result.comparison_summary.comparison_time
                        if isinstance(timestamp, str):
                            try:
                                timestamp = datetime.fromisoformat(timestamp)
                            except:
                                timestamp = datetime.now()
                        timestamps.append(timestamp)
                
                if timestamps:
                    timestamps.sort()
                    x = range(len(timestamps))
                    y = [1] * len(timestamps)  # Dummy y values
                    
                    ax6.scatter(x, y, s=100, alpha=0.7, color='blue')
                    ax6.set_xlabel('Analysis Order')
                    ax6.set_ylabel('')
                    ax6.set_title('Analysis Timeline')
                    ax6.set_yticks([])
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"ml_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating ML overview plot: {e}")
            return ""
    
    def create_interactive_ml_dashboard(self, 
                                      ml_results: Dict[str, Any],
                                      comparison_results: List[SimulationComparisonResult]) -> str:
        """Create interactive ML dashboard using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive dashboard")
            return ""
        
        logger.debug("Creating interactive ML dashboard")
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=('Clustering Analysis', 'Anomaly Detection', 'Trend Analysis',
                              'Similarity Matrix', 'Feature Importance', 'Model Performance',
                              'Recommendations', 'Data Quality', 'Analysis Summary'),
                specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"},
                       {"type": "heatmap"}, {"type": "bar"}, {"type": "bar"},
                       {"type": "bar"}, {"type": "bar"}, {"type": "table"}]]
            )
            
            # Add plots to subplots (simplified version)
            # This would be expanded with actual Plotly visualizations
            
            # Update layout
            fig.update_layout(
                title="Interactive ML Analysis Dashboard",
                height=1200,
                showlegend=True
            )
            
            # Save as HTML
            output_path = self.output_dir / f"interactive_ml_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating interactive ML dashboard: {e}")
            return ""
    
    def export_ml_visualization_data(self, 
                                   ml_results: Dict[str, Any],
                                   output_path: Union[str, Path]) -> str:
        """Export ML visualization data for external tools."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for export
        export_data = {
            'ml_results': ml_results,
            'visualization_config': {
                'figure_size': self.config.figure_size,
                'dpi': self.config.dpi,
                'style': self.config.style,
                'color_palette': self.config.color_palette
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"ML visualization data exported to {output_path}")
        return str(output_path)