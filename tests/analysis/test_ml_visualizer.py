"""
Tests for ML visualizer module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from farm.analysis.comparative.ml_visualizer import (
    MLVisualizer, MLVisualizationConfig
)
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, MetricsComparisonResult,
    DatabaseComparisonResult, LogComparisonResult
)


class TestMLVisualizer(unittest.TestCase):
    """Test cases for MLVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_results = self._create_mock_comparison_results()
        self.mock_ml_results = self._create_mock_ml_results()
    
    def _create_mock_comparison_results(self):
        """Create mock comparison results for testing."""
        results = []
        
        for i in range(5):
            # Create mock comparison summary
            summary = ComparisonSummary(
                total_differences=i * 10,
                config_differences=i * 2,
                database_differences=i * 3,
                log_differences=i * 2,
                metrics_differences=i * 3,
                severity="medium" if i % 2 == 0 else "high",
                comparison_time=datetime(2024, 1, 1, i, 0, 0)
            )
            
            # Create mock metrics comparison
            metrics_comp = MetricsComparisonResult(
                metric_differences={
                    'cpu_usage': {'percentage_change': i * 10.0},
                    'memory_usage': {'percentage_change': i * 5.0}
                },
                performance_comparison={
                    'execution_time': {'ratio': 1.0 + i * 0.1},
                    'throughput': {'ratio': 1.0 - i * 0.05}
                }
            )
            
            # Create mock database comparison
            db_comp = DatabaseComparisonResult(
                schema_differences=[f'schema_diff_{i}'],
                data_differences=[f'data_diff_{i}']
            )
            
            # Create mock log comparison
            log_comp = LogComparisonResult(
                error_differences={
                    'error_type_1': {'difference': i * 2},
                    'error_type_2': {'difference': i * 1}
                },
                performance_differences={
                    'log_perf_1': {'difference': i * 0.5}
                }
            )
            
            # Create mock config comparison
            config_comp = Mock()
            config_comp.differences = {}
            
            result = SimulationComparisonResult(
                simulation1_path=Path(f"/tmp/sim1_{i}"),
                simulation2_path=Path(f"/tmp/sim2_{i}"),
                comparison_summary=summary,
                metrics_comparison=metrics_comp,
                database_comparison=db_comp,
                log_comparison=log_comp,
                config_comparison=config_comp
            )
            
            results.append(result)
        
        return results
    
    def _create_mock_ml_results(self):
        """Create mock ML analysis results for testing."""
        return {
            'clustering': {
                'clusters': [
                    {'cluster_id': 0, 'size': 2, 'characteristics': ['high_differences']},
                    {'cluster_id': 1, 'size': 3, 'characteristics': ['low_differences']}
                ],
                'cluster_labels': [0, 0, 1, 1, 1],
                'silhouette_score': 0.7,
                'calinski_harabasz_score': 15.0,
                'davies_bouldin_score': 1.2,
                'cluster_visualization': {
                    'coordinates': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                    'cluster_labels': [0, 0, 1, 1, 1]
                }
            },
            'anomaly_detection': {
                'anomalies': [
                    {'simulation_id': 0, 'anomaly_score': 0.9, 'severity': 'high', 'type': 'high_differences'},
                    {'simulation_id': 4, 'anomaly_score': 0.8, 'severity': 'medium', 'type': 'performance_anomaly'}
                ],
                'anomaly_scores': [0.9, 0.2, 0.3, 0.4, 0.8],
                'anomaly_types': ['high_differences', 'performance_anomaly'],
                'severity_distribution': {'high': 1, 'medium': 1, 'low': 3}
            },
            'trend_prediction': {
                'predictions': {
                    'total_differences': {
                        'predictions': [50, 60, 70],
                        'confidence_intervals': [[45, 55], [55, 65], [65, 75]]
                    }
                },
                'trend_analysis': {
                    'total_differences': {
                        'direction': 'increasing',
                        'strength': 0.8,
                        'confidence': 0.9
                    }
                },
                'forecast_accuracy': {
                    'total_differences': {'mae': 0.5, 'rmse': 0.7, 'r2': 0.8}
                }
            },
            'similarity_analysis': {
                'similarity_matrix': [[1.0, 0.8, 0.6, 0.4, 0.2],
                                    [0.8, 1.0, 0.7, 0.5, 0.3],
                                    [0.6, 0.7, 1.0, 0.8, 0.4],
                                    [0.4, 0.5, 0.8, 1.0, 0.6],
                                    [0.2, 0.3, 0.4, 0.6, 1.0]],
                'similar_pairs': [
                    {'simulation_1': 0, 'simulation_2': 1, 'similarity_score': 0.8}
                ],
                'recommendations': [
                    {'simulation_id': 0, 'similar_simulations': [1], 'recommendations': ['Recommendation 1']}
                ]
            },
            'feature_importance': {
                'total_differences': 0.8,
                'config_differences': 0.6,
                'database_differences': 0.4
            }
        }
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_init(self):
        """Test MLVisualizer initialization."""
        visualizer = MLVisualizer()
        self.assertIsNotNone(visualizer.config)
        self.assertIsNotNone(visualizer.output_dir)
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_init_custom_config(self):
        """Test MLVisualizer initialization with custom config."""
        config = MLVisualizationConfig(figure_size=(16, 12), output_dir='/tmp/test_plots')
        visualizer = MLVisualizer(config)
        self.assertEqual(visualizer.config.figure_size, (16, 12))
        self.assertEqual(visualizer.config.output_dir, '/tmp/test_plots')
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', False)
    def test_init_no_matplotlib(self):
        """Test MLVisualizer initialization without matplotlib."""
        with self.assertRaises(ImportError):
            MLVisualizer()
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_ml_dashboard(self):
        """Test create_ml_dashboard method."""
        visualizer = MLVisualizer()
        
        with patch.object(visualizer, 'create_clustering_visualization') as mock_cluster, \
             patch.object(visualizer, 'create_anomaly_visualization') as mock_anomaly, \
             patch.object(visualizer, 'create_trend_visualization') as mock_trend, \
             patch.object(visualizer, 'create_similarity_heatmap') as mock_similarity, \
             patch.object(visualizer, 'create_feature_importance_plot') as mock_feature, \
             patch.object(visualizer, 'create_ml_overview_plot') as mock_overview:
            
            mock_cluster.return_value = '/tmp/clustering_plot.png'
            mock_anomaly.return_value = '/tmp/anomaly_plot.png'
            mock_trend.return_value = '/tmp/trend_plot.png'
            mock_similarity.return_value = '/tmp/similarity_heatmap.png'
            mock_feature.return_value = '/tmp/feature_importance.png'
            mock_overview.return_value = '/tmp/ml_overview.png'
            
            result = visualizer.create_ml_dashboard(self.mock_ml_results, self.mock_results)
            
            self.assertIsInstance(result, dict)
            self.assertIn('clustering_plot', result)
            self.assertIn('anomaly_plot', result)
            self.assertIn('trend_plot', result)
            self.assertIn('similarity_heatmap', result)
            self.assertIn('feature_importance', result)
            self.assertIn('ml_overview', result)
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_clustering_visualization(self):
        """Test create_clustering_visualization method."""
        visualizer = MLVisualizer()
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.scatter') as mock_scatter, \
             patch('matplotlib.pyplot.bar') as mock_bar, \
             patch('matplotlib.pyplot.imshow') as mock_imshow, \
             patch('matplotlib.pyplot.colorbar') as mock_colorbar, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
             patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
             patch('matplotlib.pyplot.legend') as mock_legend, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_scatter.return_value = Mock()
            mock_bar.return_value = Mock()
            mock_imshow.return_value = Mock()
            mock_colorbar.return_value = Mock()
            mock_title.return_value = Mock()
            mock_xlabel.return_value = Mock()
            mock_ylabel.return_value = Mock()
            mock_legend.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_clustering_visualization(
                self.mock_ml_results['clustering'], self.mock_results
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_anomaly_visualization(self):
        """Test create_anomaly_visualization method."""
        visualizer = MLVisualizer()
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.scatter') as mock_scatter, \
             patch('matplotlib.pyplot.bar') as mock_bar, \
             patch('matplotlib.pyplot.hist') as mock_hist, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
             patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
             patch('matplotlib.pyplot.legend') as mock_legend, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_scatter.return_value = Mock()
            mock_bar.return_value = Mock()
            mock_hist.return_value = Mock()
            mock_title.return_value = Mock()
            mock_xlabel.return_value = Mock()
            mock_ylabel.return_value = Mock()
            mock_legend.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_anomaly_visualization(
                self.mock_ml_results['anomaly_detection'], self.mock_results
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_trend_visualization(self):
        """Test create_trend_visualization method."""
        visualizer = MLVisualizer()
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.plot') as mock_plot, \
             patch('matplotlib.pyplot.fill_between') as mock_fill, \
             patch('matplotlib.pyplot.scatter') as mock_scatter, \
             patch('matplotlib.pyplot.bar') as mock_bar, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
             patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
             patch('matplotlib.pyplot.legend') as mock_legend, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_plot.return_value = Mock()
            mock_fill.return_value = Mock()
            mock_scatter.return_value = Mock()
            mock_bar.return_value = Mock()
            mock_title.return_value = Mock()
            mock_xlabel.return_value = Mock()
            mock_ylabel.return_value = Mock()
            mock_legend.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_trend_visualization(
                self.mock_ml_results['trend_prediction']
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_similarity_heatmap(self):
        """Test create_similarity_heatmap method."""
        visualizer = MLVisualizer()
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.imshow') as mock_imshow, \
             patch('matplotlib.pyplot.colorbar') as mock_colorbar, \
             patch('matplotlib.pyplot.hist') as mock_hist, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
             patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_imshow.return_value = Mock()
            mock_colorbar.return_value = Mock()
            mock_hist.return_value = Mock()
            mock_title.return_value = Mock()
            mock_xlabel.return_value = Mock()
            mock_ylabel.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_similarity_heatmap(
                self.mock_ml_results['similarity_analysis']
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_feature_importance_plot(self):
        """Test create_feature_importance_plot method."""
        visualizer = MLVisualizer()
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.barh') as mock_barh, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_barh.return_value = Mock()
            mock_title.return_value = Mock()
            mock_xlabel.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_feature_importance_plot(
                self.mock_ml_results['feature_importance']
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_ml_overview_plot(self):
        """Test create_ml_overview_plot method."""
        visualizer = MLVisualizer()
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.bar') as mock_bar, \
             patch('matplotlib.pyplot.pie') as mock_pie, \
             patch('matplotlib.pyplot.plot') as mock_plot, \
             patch('matplotlib.pyplot.scatter') as mock_scatter, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
             patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
             patch('matplotlib.pyplot.legend') as mock_legend, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_bar.return_value = Mock()
            mock_pie.return_value = Mock()
            mock_plot.return_value = Mock()
            mock_scatter.return_value = Mock()
            mock_title.return_value = Mock()
            mock_xlabel.return_value = Mock()
            mock_ylabel.return_value = Mock()
            mock_legend.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_ml_overview_plot(
                self.mock_ml_results, self.mock_results, "Test ML Analysis"
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_interactive_ml_dashboard(self):
        """Test create_interactive_ml_dashboard method."""
        visualizer = MLVisualizer()
        
        with patch('farm.analysis.comparative.ml_visualizer.PLOTLY_AVAILABLE', True):
            with patch('plotly.graph_objects.Figure') as mock_figure, \
                 patch('plotly.express.scatter') as mock_scatter, \
                 patch('plotly.express.imshow') as mock_imshow, \
                 patch('plotly.express.bar') as mock_bar, \
                 patch('plotly.subplots.make_subplots') as mock_subplots:
                
                mock_figure.return_value = Mock()
                mock_scatter.return_value = Mock()
                mock_imshow.return_value = Mock()
                mock_bar.return_value = Mock()
                mock_subplots.return_value = Mock()
                
                result = visualizer.create_interactive_ml_dashboard(
                    self.mock_ml_results, self.mock_results
                )
                
                self.assertIsInstance(result, str)
                self.assertTrue(result.endswith('.html'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_interactive_ml_dashboard_no_plotly(self):
        """Test create_interactive_ml_dashboard without plotly."""
        visualizer = MLVisualizer()
        
        with patch('farm.analysis.comparative.ml_visualizer.PLOTLY_AVAILABLE', False):
            result = visualizer.create_interactive_ml_dashboard(
                self.mock_ml_results, self.mock_results
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.html'))
            self.assertIn('Plotly not available', result)
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_clustering_visualization_error(self):
        """Test create_clustering_visualization with error."""
        visualizer = MLVisualizer()
        
        # Test with empty clustering result
        empty_clustering = {
            'clusters': [],
            'cluster_labels': [],
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': 0.0,
            'cluster_visualization': {}
        }
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_title.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_clustering_visualization(
                empty_clustering, self.mock_results
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_anomaly_visualization_error(self):
        """Test create_anomaly_visualization with error."""
        visualizer = MLVisualizer()
        
        # Test with empty anomaly result
        empty_anomaly = {
            'anomalies': [],
            'anomaly_scores': [],
            'anomaly_types': [],
            'severity_distribution': {}
        }
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_title.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_anomaly_visualization(
                empty_anomaly, self.mock_results
            )
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_trend_visualization_error(self):
        """Test create_trend_visualization with error."""
        visualizer = MLVisualizer()
        
        # Test with empty trend result
        empty_trend = {
            'predictions': {},
            'trend_analysis': {},
            'forecast_accuracy': {}
        }
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_title.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_trend_visualization(empty_trend)
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_similarity_heatmap_error(self):
        """Test create_similarity_heatmap with error."""
        visualizer = MLVisualizer()
        
        # Test with empty similarity result
        empty_similarity = {
            'similarity_matrix': [],
            'similar_pairs': [],
            'recommendations': []
        }
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_subplot.return_value = Mock()
            mock_title.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_similarity_heatmap(empty_similarity)
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))
    
    @patch('farm.analysis.comparative.ml_visualizer.MATPLOTLIB_AVAILABLE', True)
    def test_create_feature_importance_plot_error(self):
        """Test create_feature_importance_plot with error."""
        visualizer = MLVisualizer()
        
        # Test with empty feature importance
        empty_importance = {}
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.title') as mock_title, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            mock_figure.return_value = Mock()
            mock_title.return_value = Mock()
            mock_tight.return_value = Mock()
            mock_savefig.return_value = Mock()
            
            result = visualizer.create_feature_importance_plot(empty_importance)
            
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.png'))


if __name__ == '__main__':
    unittest.main()