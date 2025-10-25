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
        self.assertIsNotNone(analyzer.scalers)
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
             patch.object(analyzer, '_perform_clustering') as mock_cluster, \
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
        
        # Test similarity matrix calculation
        result = analyzer._calculate_similarity_matrix(features_df)
        self.assertIsInstance(result, np.ndarray)
    
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
            self.assertIn('sim1_id', pair)
            self.assertIn('sim2_id', pair)
            self.assertIn('similarity_score', pair)
            self.assertIn('distance_metrics', pair)
            self.assertIn('common_features', pair)
            self.assertIn('different_features', pair)
            self.assertIn('similarity_type', pair)
            self.assertIn('recommendation', pair)
    
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
                'sim1_id': 0, 'sim2_id': 1,
                'similarity_score': 0.9, 'distance_metrics': {'euclidean_distance': 0.1, 'manhattan_distance': 0.2},
                'common_features': [], 'different_features': [],
                'similarity_type': 'high', 'recommendation': 'Recommendation 1'
            }
        ]
        
        result = analyzer._generate_recommendations(features_df, similar_pairs, self.mock_results)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)  # One for each simulation
        
        for rec in result:
            self.assertIn('simulation_id', rec)
            self.assertIn('recommended_simulations', rec)
            self.assertIn('recommendation_type', rec)
            self.assertIn('confidence', rec)
            self.assertIn('reasoning', rec)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_generate_recommendation_reasoning(self):
        """Test _generate_recommendation_reasoning method."""
        analyzer = SimilarityAnalyzer()
        
        similar_sims = [
            {
                'simulation_id': 1,
                'similarity_score': 0.9,
                'similarity_type': 'high',
                'common_features': ['feature1'],
                'different_features': ['feature2']
            }
        ]
        
        result = analyzer._generate_recommendation_reasoning(0, similar_sims, 'very_similar')
        
        self.assertIsInstance(result, str)
        self.assertIn('Simulation 0', result)
    
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
        
        result = analyzer._perform_clustering(features_df, similarity_matrix)
        
        self.assertIsInstance(result, list)
        for cluster in result:
            self.assertIn('cluster_id', cluster)
            self.assertIn('simulation_ids', cluster)
            self.assertIn('size', cluster)
            self.assertIn('average_similarity', cluster)
            self.assertIn('cohesion', cluster)
    
    @patch('farm.analysis.comparative.similarity_analyzer.SKLEARN_AVAILABLE', True)
    def test_perform_clustering(self):
        """Test _perform_clustering method."""
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
        
        result = analyzer._perform_clustering(features_df, similarity_matrix)
        
        self.assertIsInstance(result, list)
        for cluster in result:
            self.assertIn('cluster_id', cluster)
            self.assertIn('simulation_ids', cluster)
            self.assertIn('size', cluster)
            self.assertIn('average_similarity', cluster)
            self.assertIn('cohesion', cluster)
    
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
                'sim1_id': 0, 'sim2_id': 1,
                'similarity_score': 0.9, 'distance_metrics': {'euclidean_distance': 0.1, 'manhattan_distance': 0.2},
                'common_features': [], 'different_features': [],
                'similarity_type': 'high', 'recommendation': 'Recommendation 1'
            }
        ]
        
        recommendations = [
            {
                'simulation_id': 0, 'recommended_simulations': [1],
                'recommendation_type': 'similar', 'confidence': 0.8, 'reasoning': 'High similarity'
            }
        ]
        
        feature_importance = {'feature1': 0.8, 'feature2': 0.6}
        clusters = []
        
        result = analyzer._generate_similarity_summary(
            similar_pairs, recommendations, clusters, feature_importance
        )
        
        self.assertIn('total_similar_pairs', result)
        self.assertIn('high_similarity_pairs', result)
        self.assertIn('similarity_rate', result)
        self.assertIn('recommendation_type_distribution', result)
        self.assertIn('cluster_summary', result)
        self.assertIn('top_important_features', result)
        self.assertIn('analysis_timestamp', result)
    
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
            clusters=[], feature_importance={}, summary={}
        )
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            output_path = analyzer.export_similarity_results(result, '/tmp/test_export.json')
            
            self.assertEqual(output_path, '/tmp/test_export.json')
            mock_file.assert_called_once()


if __name__ == '__main__':
    unittest.main()