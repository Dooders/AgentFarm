"""
Tests for clustering analyzer module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from farm.analysis.comparative.clustering_analyzer import (
    ClusteringAnalyzer, ClusteringConfig, ClusterResult
)
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, MetricsComparisonResult,
    DatabaseComparisonResult, LogComparisonResult
)


class TestClusteringAnalyzer(unittest.TestCase):
    """Test cases for ClusteringAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_results = self._create_mock_comparison_results()
    
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
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_init(self):
        """Test ClusteringAnalyzer initialization."""
        analyzer = ClusteringAnalyzer()
        self.assertIsNotNone(analyzer.clustering_models)
        self.assertIsNotNone(analyzer.scalers)
        self.assertIsNotNone(analyzer.dim_reduction_models)
        self.assertIsNotNone(analyzer.config)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_init_custom_config(self):
        """Test ClusteringAnalyzer initialization with custom config."""
        config = ClusteringConfig(max_clusters=5, min_clusters=2)
        analyzer = ClusteringAnalyzer(config)
        self.assertEqual(analyzer.config.max_clusters, 5)
        self.assertEqual(analyzer.config.min_clusters, 2)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', False)
    def test_init_no_sklearn(self):
        """Test ClusteringAnalyzer initialization without sklearn."""
        with self.assertRaises(ImportError):
            ClusteringAnalyzer()
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_cluster_simulations_auto(self):
        """Test cluster_simulations with auto method."""
        analyzer = ClusteringAnalyzer()
        
        with patch.object(analyzer, '_extract_features') as mock_extract, \
             patch.object(analyzer, '_apply_dimensionality_reduction') as mock_reduce, \
             patch.object(analyzer, '_find_optimal_clusters') as mock_optimal, \
             patch.object(analyzer, '_auto_clustering') as mock_auto:
            
            mock_extract.return_value = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
            mock_reduce.return_value = np.array([[1], [2], [3], [4], [5]])
            mock_optimal.return_value = 3
            mock_auto.return_value = ClusterResult(
                clusters=[], cluster_labels=[], cluster_centers=[],
                silhouette_score=0.5, calinski_harabasz_score=10.0,
                davies_bouldin_score=1.0, optimal_clusters=3,
                cluster_characteristics=[], cluster_visualization={},
                summary={}
            )
            
            result = analyzer.cluster_simulations(self.mock_results, method='auto')
            
            self.assertIsInstance(result, ClusterResult)
            mock_extract.assert_called_once()
            mock_auto.assert_called_once()
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_cluster_simulations_single_method(self):
        """Test cluster_simulations with single method."""
        analyzer = ClusteringAnalyzer()
        
        with patch.object(analyzer, '_extract_features') as mock_extract, \
             patch.object(analyzer, '_apply_dimensionality_reduction') as mock_reduce, \
             patch.object(analyzer, '_find_optimal_clusters') as mock_optimal, \
             patch.object(analyzer, '_single_method_clustering') as mock_single:
            
            mock_extract.return_value = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
            mock_reduce.return_value = np.array([[1], [2], [3], [4], [5]])
            mock_optimal.return_value = 3
            mock_single.return_value = ClusterResult(
                clusters=[], cluster_labels=[], cluster_centers=[],
                silhouette_score=0.5, calinski_harabasz_score=10.0,
                davies_bouldin_score=1.0, optimal_clusters=3,
                cluster_characteristics=[], cluster_visualization={},
                summary={}
            )
            
            result = analyzer.cluster_simulations(self.mock_results, method='kmeans', n_clusters=3)
            
            self.assertIsInstance(result, ClusterResult)
            mock_single.assert_called_once()
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_cluster_simulations_insufficient_data(self):
        """Test cluster_simulations with insufficient data."""
        analyzer = ClusteringAnalyzer()
        
        result = analyzer.cluster_simulations([], method='auto')
        
        self.assertIsInstance(result, ClusterResult)
        self.assertEqual(result.silhouette_score, 0.0)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_extract_features(self):
        """Test _extract_features method."""
        analyzer = ClusteringAnalyzer()
        
        features_df = analyzer._extract_features(self.mock_results)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIn('simulation_id', features_df.columns)
        self.assertIn('total_differences', features_df.columns)
        self.assertIn('severity_numeric', features_df.columns)
        self.assertEqual(len(features_df), 5)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_extract_features_with_specific_features(self):
        """Test _extract_features with specific features."""
        analyzer = ClusteringAnalyzer()
        
        features = ['total_differences', 'config_differences']
        features_df = analyzer._extract_features(self.mock_results, features)
        
        self.assertIn('simulation_id', features_df.columns)
        self.assertIn('total_differences', features_df.columns)
        self.assertIn('config_differences', features_df.columns)
        self.assertNotIn('database_differences', features_df.columns)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_severity_to_numeric(self):
        """Test _severity_to_numeric method."""
        analyzer = ClusteringAnalyzer()
        
        self.assertEqual(analyzer._severity_to_numeric('low'), 1)
        self.assertEqual(analyzer._severity_to_numeric('medium'), 2)
        self.assertEqual(analyzer._severity_to_numeric('high'), 3)
        self.assertEqual(analyzer._severity_to_numeric('critical'), 4)
        self.assertEqual(analyzer._severity_to_numeric('unknown'), 0)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_apply_dimensionality_reduction(self):
        """Test _apply_dimensionality_reduction method."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        result = analyzer._apply_dimensionality_reduction(X)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 2)  # Should be reduced to 2 components
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_apply_dimensionality_reduction_error(self):
        """Test _apply_dimensionality_reduction with error."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2]])  # Too few samples
        
        result = analyzer._apply_dimensionality_reduction(X)
        
        # Should return original data on error
        np.testing.assert_array_equal(result, X)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_find_optimal_clusters(self):
        """Test _find_optimal_clusters method."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        
        result = analyzer._find_optimal_clusters(X)
        
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, analyzer.config.min_clusters)
        self.assertLessEqual(result, analyzer.config.max_clusters)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_find_optimal_clusters_insufficient_data(self):
        """Test _find_optimal_clusters with insufficient data."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2]])  # Only one sample
        
        result = analyzer._find_optimal_clusters(X)
        
        self.assertEqual(result, 2)  # Should return min_clusters
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_auto_clustering(self):
        """Test _auto_clustering method."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        features_df = pd.DataFrame({'simulation_id': [0, 1, 2, 3, 4]})
        n_clusters = 3
        
        with patch.object(analyzer, '_single_method_clustering') as mock_single:
            mock_single.return_value = ClusterResult(
                clusters=[], cluster_labels=[], cluster_centers=[],
                silhouette_score=0.8, calinski_harabasz_score=10.0,
                davies_bouldin_score=1.0, optimal_clusters=3,
                cluster_characteristics=[], cluster_visualization={},
                summary={}
            )
            
            result = analyzer._auto_clustering(X, features_df, n_clusters)
            
            self.assertIsInstance(result, ClusterResult)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_auto_clustering_all_methods_fail(self):
        """Test _auto_clustering when all methods fail."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        features_df = pd.DataFrame({'simulation_id': [0, 1, 2, 3, 4]})
        n_clusters = 3
        
        with patch.object(analyzer, '_single_method_clustering') as mock_single:
            mock_single.side_effect = Exception("All methods failed")
            
            result = analyzer._auto_clustering(X, features_df, n_clusters)
            
            self.assertIsInstance(result, ClusterResult)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_single_method_clustering(self):
        """Test _single_method_clustering method."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5]
        })
        method = 'kmeans'
        n_clusters = 3
        
        with patch.object(analyzer, '_analyze_clusters') as mock_analyze, \
             patch.object(analyzer, '_generate_cluster_visualization') as mock_viz, \
             patch.object(analyzer, '_generate_clustering_summary') as mock_summary:
            
            mock_analyze.return_value = []
            mock_viz.return_value = {}
            mock_summary.return_value = {}
            
            result = analyzer._single_method_clustering(X, features_df, method, n_clusters)
            
            self.assertIsInstance(result, ClusterResult)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_single_method_clustering_unknown_method(self):
        """Test _single_method_clustering with unknown method."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        features_df = pd.DataFrame({'simulation_id': [0, 1, 2, 3, 4]})
        method = 'unknown_method'
        n_clusters = 3
        
        with self.assertRaises(ValueError):
            analyzer._single_method_clustering(X, features_df, method, n_clusters)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_clusters(self):
        """Test _analyze_clusters method."""
        analyzer = ClusteringAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        cluster_labels = np.array([0, 0, 1, 1, 2])
        cluster_centers = [[1.5, 3], [3.5, 7], [5, 10]]
        
        result = analyzer._analyze_clusters(features_df, cluster_labels, cluster_centers)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 3 clusters
        
        for cluster in result:
            self.assertIn('cluster_id', cluster)
            self.assertIn('size', cluster)
            self.assertIn('percentage', cluster)
            self.assertIn('simulation_ids', cluster)
            self.assertIn('characteristics', cluster)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_identify_cluster_characteristics(self):
        """Test _identify_cluster_characteristics method."""
        analyzer = ClusteringAnalyzer()
        
        # Test high differences cluster
        cluster_data = pd.DataFrame({'total_differences': [60, 70, 80]})
        result = analyzer._identify_cluster_characteristics(cluster_data)
        self.assertIn('high_differences', result)
        
        # Test low differences cluster
        cluster_data = pd.DataFrame({'total_differences': [5, 8, 10]})
        result = analyzer._identify_cluster_characteristics(cluster_data)
        self.assertIn('low_differences', result)
        
        # Test general cluster
        cluster_data = pd.DataFrame({'total_differences': [20, 25, 30]})
        result = analyzer._identify_cluster_characteristics(cluster_data)
        self.assertIn('general', result)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_generate_cluster_visualization(self):
        """Test _generate_cluster_visualization method."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        cluster_labels = np.array([0, 0, 1, 1, 2])
        features_df = pd.DataFrame({'simulation_id': [0, 1, 2, 3, 4]})
        
        result = analyzer._generate_cluster_visualization(X, cluster_labels, features_df)
        
        self.assertIsInstance(result, dict)
        self.assertIn('coordinates', result)
        self.assertIn('cluster_labels', result)
        self.assertIn('simulation_ids', result)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_generate_cluster_visualization_error(self):
        """Test _generate_cluster_visualization with error."""
        analyzer = ClusteringAnalyzer()
        
        X = np.array([[1, 2]])  # Too few samples
        cluster_labels = np.array([0])
        features_df = pd.DataFrame({'simulation_id': [0]})
        
        result = analyzer._generate_cluster_visualization(X, cluster_labels, features_df)
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_generate_clustering_summary(self):
        """Test _generate_clustering_summary method."""
        analyzer = ClusteringAnalyzer()
        
        clusters = [
            {'size': 2, 'characteristics': ['high_differences']},
            {'size': 3, 'characteristics': ['low_differences']}
        ]
        cluster_labels = np.array([0, 0, 1, 1, 1])
        silhouette_score = 0.7
        calinski_score = 15.0
        davies_bouldin_score = 1.2
        
        result = analyzer._generate_clustering_summary(
            clusters, cluster_labels, silhouette_score, calinski_score, davies_bouldin_score
        )
        
        self.assertIn('total_clusters', result)
        self.assertIn('total_simulations', result)
        self.assertIn('silhouette_score', result)
        self.assertIn('calinski_harabasz_score', result)
        self.assertIn('davies_bouldin_score', result)
        self.assertIn('clustering_quality', result)
        self.assertIn('cluster_size_distribution', result)
        self.assertIn('characteristic_distribution', result)
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_assess_clustering_quality(self):
        """Test _assess_clustering_quality method."""
        analyzer = ClusteringAnalyzer()
        
        # Test excellent quality
        result = analyzer._assess_clustering_quality(0.8, 120, 0.8)
        self.assertEqual(result, 'Excellent')
        
        # Test good quality
        result = analyzer._assess_clustering_quality(0.6, 80, 1.2)
        self.assertEqual(result, 'Good')
        
        # Test fair quality
        result = analyzer._assess_clustering_quality(0.4, 60, 1.8)
        self.assertEqual(result, 'Fair')
        
        # Test poor quality
        result = analyzer._assess_clustering_quality(0.2, 30, 2.5)
        self.assertEqual(result, 'Poor')
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_create_empty_result(self):
        """Test _create_empty_result method."""
        analyzer = ClusteringAnalyzer()
        
        result = analyzer._create_empty_result()
        
        self.assertIsInstance(result, ClusterResult)
        self.assertEqual(result.silhouette_score, 0.0)
        self.assertEqual(result.clustering_quality, 'Poor')
    
    @patch('farm.analysis.comparative.clustering_analyzer.SKLEARN_AVAILABLE', True)
    def test_export_clustering_results(self):
        """Test export_clustering_results method."""
        analyzer = ClusteringAnalyzer()
        
        result = ClusterResult(
            clusters=[], cluster_labels=[], cluster_centers=[],
            silhouette_score=0.5, calinski_harabasz_score=10.0,
            davies_bouldin_score=1.0, optimal_clusters=3,
            cluster_characteristics=[], cluster_visualization={},
            summary={}
        )
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            output_path = analyzer.export_clustering_results(result, '/tmp/test_export.json')
            
            self.assertEqual(output_path, '/tmp/test_export.json')
            mock_file.assert_called_once()


if __name__ == '__main__':
    unittest.main()