"""
Tests for similarity analyzer module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from farm.analysis.comparative.similarity_analyzer import (
    SimilarityAnalyzer, SimilarityConfig, SimilarityResult, SimilarityPair, Recommendation
)
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, MetricsComparisonResult,
    DatabaseComparisonResult, LogComparisonResult
)


class TestSimilarityAnalyzer(unittest.TestCase):
    """Test cases for SimilarityAnalyzer."""
    
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
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_init(self):
        """Test SimilarityAnalyzer initialization."""
        analyzer = SimilarityAnalyzer()
        self.assertIsNotNone(analyzer.scaler)
        self.assertIsNotNone(analyzer.config)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_init_custom_config(self):
        """Test SimilarityAnalyzer initialization with custom config."""
        config = SimilarityConfig(similarity_threshold=0.9, max_recommendations=10)
        analyzer = SimilarityAnalyzer(config)
        self.assertEqual(analyzer.config.similarity_threshold, 0.9)
        self.assertEqual(analyzer.config.max_recommendations, 10)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', False)
    def test_init_no_sklearn(self):
        """Test SimilarityAnalyzer initialization without sklearn."""
        with self.assertRaises(ImportError):
            SimilarityAnalyzer()
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_similarity(self):
        """Test analyze_similarity method."""
        analyzer = SimilarityAnalyzer()
        
        with patch.object(analyzer, '_extract_features') as mock_extract, \
             patch.object(analyzer, '_calculate_similarity_matrix') as mock_matrix, \
             patch.object(analyzer, '_find_similar_pairs') as mock_pairs, \
             patch.object(analyzer, '_generate_recommendations') as mock_recommend, \
             patch.object(analyzer, '_perform_similarity_clustering') as mock_cluster, \
             patch.object(analyzer, '_analyze_feature_importance') as mock_importance, \
             patch.object(analyzer, '_generate_similarity_summary') as mock_summary:
            
            mock_extract.return_value = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
            mock_matrix.return_value = np.array([[1.0, 0.8, 0.6, 0.4, 0.2],
                                                [0.8, 1.0, 0.7, 0.5, 0.3],
                                                [0.6, 0.7, 1.0, 0.8, 0.4],
                                                [0.4, 0.5, 0.8, 1.0, 0.6],
                                                [0.2, 0.3, 0.4, 0.6, 1.0]])
            mock_pairs.return_value = []
            mock_recommend.return_value = []
            mock_cluster.return_value = {}
            mock_importance.return_value = {}
            mock_summary.return_value = {}
            
            result = analyzer.analyze_similarity(self.mock_results)
            
            self.assertIsInstance(result, SimilarityResult)
            mock_extract.assert_called_once()
            mock_matrix.assert_called_once()
            mock_pairs.assert_called_once()
            mock_recommend.assert_called_once()
            mock_cluster.assert_called_once()
            mock_importance.assert_called_once()
            mock_summary.assert_called_once()
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_similarity_insufficient_data(self):
        """Test analyze_similarity with insufficient data."""
        analyzer = SimilarityAnalyzer()
        
        result = analyzer.analyze_similarity([])
        
        self.assertIsInstance(result, SimilarityResult)
        self.assertEqual(len(result.similarity_matrix), 0)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_extract_features(self):
        """Test _extract_features method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = analyzer._extract_features(self.mock_results)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIn('simulation_id', features_df.columns)
        self.assertIn('total_differences', features_df.columns)
        self.assertIn('severity_numeric', features_df.columns)
        self.assertEqual(len(features_df), 5)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_extract_features_with_specific_features(self):
        """Test _extract_features with specific features."""
        analyzer = SimilarityAnalyzer()
        
        features = ['total_differences', 'config_differences']
        features_df = analyzer._extract_features(self.mock_results, features)
        
        self.assertIn('simulation_id', features_df.columns)
        self.assertIn('total_differences', features_df.columns)
        self.assertIn('config_differences', features_df.columns)
        self.assertNotIn('database_differences', features_df.columns)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_severity_to_numeric(self):
        """Test _severity_to_numeric method."""
        analyzer = SimilarityAnalyzer()
        
        self.assertEqual(analyzer._severity_to_numeric('low'), 1)
        self.assertEqual(analyzer._severity_to_numeric('medium'), 2)
        self.assertEqual(analyzer._severity_to_numeric('high'), 3)
        self.assertEqual(analyzer._severity_to_numeric('critical'), 4)
        self.assertEqual(analyzer._severity_to_numeric('unknown'), 0)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_calculate_similarity_matrix(self):
        """Test _calculate_similarity_matrix method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        result = analyzer._calculate_similarity_matrix(features_df)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 5))
        self.assertTrue(np.allclose(result.diagonal(), 1.0))  # Diagonal should be 1.0
        self.assertTrue(np.allclose(result, result.T))  # Should be symmetric
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_calculate_similarity_matrix_different_metrics(self):
        """Test _calculate_similarity_matrix with different metrics."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        # Test cosine similarity
        result_cosine = analyzer._calculate_similarity_matrix(features_df, metric='cosine')
        self.assertIsInstance(result_cosine, np.ndarray)
        
        # Test euclidean distance
        result_euclidean = analyzer._calculate_similarity_matrix(features_df, metric='euclidean')
        self.assertIsInstance(result_euclidean, np.ndarray)
        
        # Test manhattan distance
        result_manhattan = analyzer._calculate_similarity_matrix(features_df, metric='manhattan')
        self.assertIsInstance(result_manhattan, np.ndarray)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_find_similar_pairs(self):
        """Test _find_similar_pairs method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        similarity_matrix = np.array([[1.0, 0.9, 0.6, 0.4, 0.2],
                                    [0.9, 1.0, 0.7, 0.5, 0.3],
                                    [0.6, 0.7, 1.0, 0.8, 0.4],
                                    [0.4, 0.5, 0.8, 1.0, 0.6],
                                    [0.2, 0.3, 0.4, 0.6, 1.0]])
        
        result = analyzer._find_similar_pairs(features_df, similarity_matrix)
        
        self.assertIsInstance(result, list)
        for pair in result:
            self.assertIn('simulation_1', pair)
            self.assertIn('simulation_2', pair)
            self.assertIn('similarity_score', pair)
            self.assertIn('euclidean_distance', pair)
            self.assertIn('manhattan_distance', pair)
            self.assertIn('feature_differences', pair)
            self.assertIn('recommendations', pair)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_find_similar_pairs_high_threshold(self):
        """Test _find_similar_pairs with high threshold."""
        analyzer = SimilarityAnalyzer()
        analyzer.config.similarity_threshold = 0.95
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        similarity_matrix = np.array([[1.0, 0.9, 0.6, 0.4, 0.2],
                                    [0.9, 1.0, 0.7, 0.5, 0.3],
                                    [0.6, 0.7, 1.0, 0.8, 0.4],
                                    [0.4, 0.5, 0.8, 1.0, 0.6],
                                    [0.2, 0.3, 0.4, 0.6, 1.0]])
        
        result = analyzer._find_similar_pairs(features_df, similarity_matrix)
        
        # With high threshold, should find fewer pairs
        self.assertIsInstance(result, list)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_generate_recommendations(self):
        """Test _generate_recommendations method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        similar_pairs = [
            {
                'simulation_1': 0, 'simulation_2': 1,
                'similarity_score': 0.9, 'euclidean_distance': 0.1,
                'manhattan_distance': 0.2, 'feature_differences': {},
                'recommendations': ['Recommendation 1']
            }
        ]
        
        result = analyzer._generate_recommendations(features_df, similar_pairs, self.mock_results)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)  # One for each simulation
        
        for rec in result:
            self.assertIn('simulation_id', rec)
            self.assertIn('similar_simulations', rec)
            self.assertIn('recommendations', rec)
            self.assertIn('reasoning', rec)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_generate_simulation_recommendations(self):
        """Test _generate_simulation_recommendations method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        similar_pairs = [
            {
                'simulation_1': 0, 'simulation_2': 1,
                'similarity_score': 0.9, 'euclidean_distance': 0.1,
                'manhattan_distance': 0.2, 'feature_differences': {},
                'recommendations': ['Recommendation 1']
            }
        ]
        
        result = analyzer._generate_simulation_recommendations(0, features_df, similar_pairs, self.mock_results)
        
        self.assertIn('simulation_id', result)
        self.assertIn('similar_simulations', result)
        self.assertIn('recommendations', result)
        self.assertIn('reasoning', result)
        self.assertEqual(result['simulation_id'], 0)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_perform_similarity_clustering(self):
        """Test _perform_similarity_clustering method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        similarity_matrix = np.array([[1.0, 0.9, 0.6, 0.4, 0.2],
                                    [0.9, 1.0, 0.7, 0.5, 0.3],
                                    [0.6, 0.7, 1.0, 0.8, 0.4],
                                    [0.4, 0.5, 0.8, 1.0, 0.6],
                                    [0.2, 0.3, 0.4, 0.6, 1.0]])
        
        with patch.object(analyzer, '_cluster_by_similarity') as mock_cluster:
            mock_cluster.return_value = {
                'cluster_labels': [0, 0, 1, 1, 2],
                'cluster_centers': [[1.5, 3], [3.5, 7], [5, 10]],
                'silhouette_score': 0.7
            }
            
            result = analyzer._perform_similarity_clustering(features_df, similarity_matrix)
            
            self.assertIsInstance(result, dict)
            self.assertIn('cluster_labels', result)
            self.assertIn('cluster_centers', result)
            self.assertIn('silhouette_score', result)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_cluster_by_similarity(self):
        """Test _cluster_by_similarity method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        similarity_matrix = np.array([[1.0, 0.9, 0.6, 0.4, 0.2],
                                    [0.9, 1.0, 0.7, 0.5, 0.3],
                                    [0.6, 0.7, 1.0, 0.8, 0.4],
                                    [0.4, 0.5, 0.8, 1.0, 0.6],
                                    [0.2, 0.3, 0.4, 0.6, 1.0]])
        
        result = analyzer._cluster_by_similarity(features_df, similarity_matrix)
        
        self.assertIsInstance(result, dict)
        self.assertIn('cluster_labels', result)
        self.assertIn('cluster_centers', result)
        self.assertIn('silhouette_score', result)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_feature_importance(self):
        """Test _analyze_feature_importance method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        similarity_matrix = np.array([[1.0, 0.9, 0.6, 0.4, 0.2],
                                    [0.9, 1.0, 0.7, 0.5, 0.3],
                                    [0.6, 0.7, 1.0, 0.8, 0.4],
                                    [0.4, 0.5, 0.8, 1.0, 0.6],
                                    [0.2, 0.3, 0.4, 0.6, 1.0]])
        
        result = analyzer._analyze_feature_importance(features_df, similarity_matrix)
        
        self.assertIsInstance(result, dict)
        self.assertIn('feature1', result)
        self.assertIn('feature2', result)
        
        for feature, importance in result.items():
            self.assertIsInstance(importance, float)
            self.assertGreaterEqual(importance, 0.0)
            self.assertLessEqual(importance, 1.0)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_generate_similarity_summary(self):
        """Test _generate_similarity_summary method."""
        analyzer = SimilarityAnalyzer()
        
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        similarity_matrix = np.array([[1.0, 0.9, 0.6, 0.4, 0.2],
                                    [0.9, 1.0, 0.7, 0.5, 0.3],
                                    [0.6, 0.7, 1.0, 0.8, 0.4],
                                    [0.4, 0.5, 0.8, 1.0, 0.6],
                                    [0.2, 0.3, 0.4, 0.6, 1.0]])
        
        similar_pairs = [
            {
                'simulation_1': 0, 'simulation_2': 1,
                'similarity_score': 0.9, 'euclidean_distance': 0.1,
                'manhattan_distance': 0.2, 'feature_differences': {},
                'recommendations': ['Recommendation 1']
            }
        ]
        
        recommendations = [
            {
                'simulation_id': 0, 'similar_simulations': [1],
                'recommendations': ['Recommendation 1'], 'reasoning': 'High similarity'
            }
        ]
        
        feature_importance = {'feature1': 0.8, 'feature2': 0.6}
        
        result = analyzer._generate_similarity_summary(
            features_df, similarity_matrix, similar_pairs, recommendations, feature_importance
        )
        
        self.assertIn('total_simulations', result)
        self.assertIn('similarity_threshold', result)
        self.assertIn('similar_pairs_count', result)
        self.assertIn('average_similarity', result)
        self.assertIn('similarity_distribution', result)
        self.assertIn('most_similar_pair', result)
        self.assertIn('recommendations_count', result)
        self.assertIn('feature_importance_summary', result)
        self.assertIn('clustering_quality', result)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_create_empty_result(self):
        """Test _create_empty_result method."""
        analyzer = SimilarityAnalyzer()
        
        result = analyzer._create_empty_result()
        
        self.assertIsInstance(result, SimilarityResult)
        self.assertEqual(len(result.similarity_matrix), 0)
        self.assertEqual(len(result.similar_pairs), 0)
        self.assertEqual(len(result.recommendations), 0)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_export_similarity_results(self):
        """Test export_similarity_results method."""
        analyzer = SimilarityAnalyzer()
        
        result = SimilarityResult(
            similarity_matrix=[], similar_pairs=[], recommendations=[],
            similarity_clusters={}, feature_importance={}, summary={}
        )
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            output_path = analyzer.export_similarity_results(result, '/tmp/test_export.json')
            
            self.assertEqual(output_path, '/tmp/test_export.json')
            mock_file.assert_called_once()


if __name__ == '__main__':
    unittest.main()