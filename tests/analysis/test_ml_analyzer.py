"""
Tests for ML analyzer module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from pathlib import Path

from farm.analysis.comparative.ml_analyzer import MLAnalyzer, MLAnalysisResult
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, MetricsComparisonResult,
    DatabaseComparisonResult, LogComparisonResult
)


class TestMLAnalyzer(unittest.TestCase):
    """Test cases for MLAnalyzer."""
    
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
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_init(self):
        """Test MLAnalyzer initialization."""
        analyzer = MLAnalyzer()
        self.assertIsNotNone(analyzer.scaler)
        self.assertIsNotNone(analyzer.anomaly_models)
        self.assertIsNotNone(analyzer.clustering_models)
        self.assertIsNotNone(analyzer.regression_models)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_init_custom_config(self):
        """Test MLAnalyzer initialization with custom config."""
        analyzer = MLAnalyzer(scaler_type="minmax", random_state=123)
        self.assertEqual(analyzer.scaler_type, "minmax")
        self.assertEqual(analyzer.random_state, 123)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', False)
    def test_init_no_sklearn(self):
        """Test MLAnalyzer initialization without sklearn."""
        with self.assertRaises(ImportError):
            MLAnalyzer()
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_simulation_data(self):
        """Test analyze_simulation_data method."""
        analyzer = MLAnalyzer()
        
        with patch.object(analyzer, '_extract_features') as mock_extract, \
             patch.object(analyzer, '_analyze_patterns') as mock_patterns, \
             patch.object(analyzer, '_detect_anomalies') as mock_anomalies, \
             patch.object(analyzer, '_perform_clustering') as mock_clustering, \
             patch.object(analyzer, '_analyze_similarity') as mock_similarity, \
             patch.object(analyzer, '_make_predictions') as mock_predictions, \
             patch.object(analyzer, '_analyze_feature_importance') as mock_importance, \
             patch.object(analyzer, '_generate_ml_summary') as mock_summary:
            
            # Mock return values
            mock_extract.return_value = pd.DataFrame({'feature1': [1, 2, 3]})
            mock_patterns.return_value = {'patterns': []}
            mock_anomalies.return_value = {'anomalies': []}
            mock_clustering.return_value = {'clusters': []}
            mock_similarity.return_value = {'similar_pairs': []}
            mock_predictions.return_value = {'predictions': []}
            mock_importance.return_value = {'importance': []}
            mock_summary.return_value = {'summary': 'test'}
            
            result = analyzer.analyze_simulation_data(self.mock_results)
            
            self.assertIsInstance(result, MLAnalysisResult)
            mock_extract.assert_called_once()
            mock_patterns.assert_called_once()
            mock_anomalies.assert_called_once()
            mock_clustering.assert_called_once()
            mock_similarity.assert_called_once()
            mock_predictions.assert_called_once()
            mock_importance.assert_called_once()
            mock_summary.assert_called_once()
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_simulation_data_empty(self):
        """Test analyze_simulation_data with empty results."""
        analyzer = MLAnalyzer()
        
        with patch.object(analyzer, '_extract_features') as mock_extract:
            mock_extract.return_value = pd.DataFrame()
            
            result = analyzer.analyze_simulation_data([])
            
            self.assertIsInstance(result, MLAnalysisResult)
            self.assertEqual(result.summary['total_patterns_found'], 0)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_extract_features(self):
        """Test _extract_features method."""
        analyzer = MLAnalyzer()
        
        features_df = analyzer._extract_features(self.mock_results)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIn('simulation_id', features_df.columns)
        self.assertIn('total_differences', features_df.columns)
        self.assertIn('severity_numeric', features_df.columns)
        self.assertEqual(len(features_df), 5)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_extract_features_with_specific_features(self):
        """Test _extract_features method."""
        analyzer = MLAnalyzer()
        
        features_df = analyzer._extract_features(self.mock_results)
        
        self.assertIn('simulation_id', features_df.columns)
        self.assertIn('total_differences', features_df.columns)
        self.assertIn('config_differences', features_df.columns)
        self.assertIn('database_differences', features_df.columns)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_severity_to_numeric(self):
        """Test _severity_to_numeric method."""
        analyzer = MLAnalyzer()
        
        self.assertEqual(analyzer._severity_to_numeric('low'), 1)
        self.assertEqual(analyzer._severity_to_numeric('medium'), 2)
        self.assertEqual(analyzer._severity_to_numeric('high'), 3)
        self.assertEqual(analyzer._severity_to_numeric('critical'), 4)
        self.assertEqual(analyzer._severity_to_numeric('unknown'), 0)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_patterns(self):
        """Test _analyze_patterns method."""
        analyzer = MLAnalyzer()
        
        # Create test data
        features_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [1, 1, 1, 1, 1]
        })
        
        features_scaled = np.array([[1, 2, 1], [2, 4, 1], [3, 6, 1], [4, 8, 1], [5, 10, 1]])
        
        result = analyzer._analyze_patterns(features_df, features_scaled)
        
        self.assertIn('correlation_patterns', result)
        self.assertIn('trend_patterns', result)
        self.assertIn('distribution_patterns', result)
        self.assertIn('total_patterns', result)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_detect_anomalies(self):
        """Test _detect_anomalies method."""
        analyzer = MLAnalyzer()
        
        # Create test data
        features_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'feature2': [1, 2, 3, 4, 5]
        })
        
        features_scaled = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [100, 5]])
        
        result = analyzer._detect_anomalies(features_df, features_scaled)
        
        self.assertIn('anomalies', result)
        self.assertIn('anomaly_scores', result)
        self.assertIn('total_anomalies', result)
        self.assertIn('anomaly_rate', result)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_detect_anomalies_insufficient_data(self):
        """Test _detect_anomalies with insufficient data."""
        analyzer = MLAnalyzer()
        
        features_df = pd.DataFrame({'feature1': [1, 2]})
        features_scaled = np.array([[1], [2]])
        
        result = analyzer._detect_anomalies(features_df, features_scaled)
        
        self.assertEqual(result['total_anomalies'], 0)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_perform_clustering(self):
        """Test _perform_clustering method."""
        analyzer = MLAnalyzer()
        
        # Create test data
        features_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1, 2, 3, 4, 5]
        })
        
        features_scaled = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        
        result = analyzer._perform_clustering(features_df, features_scaled)
        
        self.assertIn('clusters', result)
        self.assertIn('cluster_labels', result)
        self.assertIn('silhouette_score', result)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_perform_clustering_insufficient_data(self):
        """Test _perform_clustering with insufficient data."""
        analyzer = MLAnalyzer()
        
        features_df = pd.DataFrame({'feature1': [1, 2]})
        features_scaled = np.array([[1], [2]])
        
        result = analyzer._perform_clustering(features_df, features_scaled)
        
        self.assertEqual(result['silhouette_score'], 0.0)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_similarity(self):
        """Test _analyze_similarity method."""
        analyzer = MLAnalyzer()
        
        # Create test data with simulation_id
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1, 2, 3, 4, 5]
        })
        
        features_scaled = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        
        result = analyzer._analyze_similarity(features_df, features_scaled)
        
        self.assertIn('similarity_matrix', result)
        self.assertIn('similar_pairs', result)
        self.assertIn('average_similarity', result)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_similarity_insufficient_data(self):
        """Test _analyze_similarity with insufficient data."""
        analyzer = MLAnalyzer()
        
        features_df = pd.DataFrame({'feature1': [1]})
        features_scaled = np.array([[1]])
        
        result = analyzer._analyze_similarity(features_df, features_scaled)
        
        self.assertEqual(result['average_similarity'], 0.0)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_make_predictions(self):
        """Test _make_predictions method."""
        analyzer = MLAnalyzer()
        
        # Create test data
        features_df = pd.DataFrame({
            'total_differences': [1, 2, 3, 4, 5],
            'config_differences': [1, 2, 3, 4, 5]
        })
        
        features_scaled = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        
        result = analyzer._make_predictions(features_df, features_scaled)
        
        self.assertIn('predictions', result)
        self.assertIn('model_performance', result)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_make_predictions_insufficient_data(self):
        """Test _make_predictions with insufficient data."""
        analyzer = MLAnalyzer()
        
        features_df = pd.DataFrame({'total_differences': [1, 2]})
        features_scaled = np.array([[1], [2]])
        
        result = analyzer._make_predictions(features_df, features_scaled)
        
        self.assertEqual(len(result['predictions']), 0)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_feature_importance(self):
        """Test _analyze_feature_importance method."""
        analyzer = MLAnalyzer()
        
        # Create test data
        features_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1, 2, 3, 4, 5],
            'total_differences': [1, 2, 3, 4, 5]
        })
        
        features_scaled = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        
        result = analyzer._analyze_feature_importance(features_df, features_scaled)
        
        self.assertIn('feature_importance', result)
        self.assertIn('top_features', result)
        self.assertIn('total_features', result)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_analyze_feature_importance_insufficient_data(self):
        """Test _analyze_feature_importance with insufficient data."""
        analyzer = MLAnalyzer()
        
        features_df = pd.DataFrame({'total_differences': [1, 2]})
        features_scaled = np.array([[1], [2]])
        
        result = analyzer._analyze_feature_importance(features_df, features_scaled)
        
        self.assertEqual(len(result['feature_importance']), 0)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_classify_distribution(self):
        """Test _classify_distribution method."""
        analyzer = MLAnalyzer()
        
        # Test normal distribution
        result = analyzer._classify_distribution(0.1, 0.1)
        self.assertEqual(result, 'normal')
        
        # Test right skewed
        result = analyzer._classify_distribution(0.8, 0.1)
        self.assertEqual(result, 'right_skewed')
        
        # Test left skewed
        result = analyzer._classify_distribution(-0.8, 0.1)
        self.assertEqual(result, 'left_skewed')
        
        # Test heavy tailed
        result = analyzer._classify_distribution(0.1, 0.8)
        self.assertEqual(result, 'heavy_tailed')
        
        # Test light tailed
        result = analyzer._classify_distribution(0.1, -0.8)
        self.assertEqual(result, 'light_tailed')
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_generate_ml_summary(self):
        """Test _generate_ml_summary method."""
        analyzer = MLAnalyzer()
        
        pattern_analysis = {'total_patterns': 5}
        anomaly_detection = {'total_anomalies': 2, 'anomaly_rate': 0.1}
        clustering_results = {'n_clusters': 3, 'silhouette_score': 0.7}
        similarity_analysis = {'total_similar_pairs': 4, 'average_similarity': 0.8}
        predictions = {'model_performance': {'model1': {}}}
        feature_importance = {'top_features': [{'feature': 'f1', 'importance': 0.5}]}
        
        result = analyzer._generate_ml_summary(
            pattern_analysis, anomaly_detection, clustering_results,
            similarity_analysis, predictions, feature_importance
        )
        
        self.assertIn('total_patterns_found', result)
        self.assertIn('anomalies_detected', result)
        self.assertIn('clusters_found', result)
        self.assertIn('similar_pairs', result)
        self.assertIn('models_trained', result)
        self.assertIn('analysis_quality', result)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_assess_analysis_quality(self):
        """Test _assess_analysis_quality method."""
        analyzer = MLAnalyzer()
        
        # High quality
        result = analyzer._assess_analysis_quality(
            {'total_patterns': 5}, {'total_anomalies': 3}, 
            {'silhouette_score': 0.8}, {'total_similar_pairs': 4}
        )
        self.assertEqual(result, 'High')
        
        # Medium quality
        result = analyzer._assess_analysis_quality(
            {'total_patterns': 1}, {'total_anomalies': 0},
            {'silhouette_score': 0.3}, {'total_similar_pairs': 1}
        )
        self.assertEqual(result, 'Medium')
        
        # Low quality
        result = analyzer._assess_analysis_quality(
            {'total_patterns': 0}, {'total_anomalies': 0}, 
            {'silhouette_score': 0.2}, {'total_similar_pairs': 0}
        )
        self.assertEqual(result, 'Low')
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_create_empty_result(self):
        """Test _create_empty_result method."""
        analyzer = MLAnalyzer()
        
        result = analyzer._create_empty_result()
        
        self.assertIsInstance(result, MLAnalysisResult)
        self.assertEqual(result.summary['total_patterns_found'], 0)
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_save_models(self):
        """Test save_models method."""
        analyzer = MLAnalyzer()
        analyzer.model_cache_dir = Path('/tmp/test_models')
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            analyzer.save_models('/tmp/test_model.pkl')
            
            mock_file.assert_called_once()
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_load_models(self):
        """Test load_models method."""
        analyzer = MLAnalyzer()
        
        mock_model_data = {
            'scaler': Mock(),
            'anomaly_models': {},
            'clustering_models': {},
            'regression_models': {},
            'random_state': 42,
            'scaler_type': 'standard'
        }
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file, \
             patch('pickle.load', return_value=mock_model_data), \
             patch('pathlib.Path.exists', return_value=True):

            analyzer.load_models('/tmp/test_model.pkl')

            mock_file.assert_called_once()
    
    @patch('farm.analysis.comparative.ml_analyzer.SKLEARN_AVAILABLE', True)
    def test_export_analysis_results(self):
        """Test export_analysis_results method."""
        analyzer = MLAnalyzer()
        
        result = MLAnalysisResult(
            pattern_analysis={},
            anomaly_detection={},
            clustering_results={},
            similarity_analysis={},
            predictions={},
            feature_importance={},
            summary={}
        )
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            output_path = analyzer.export_analysis_results(result, '/tmp/test_export.json')
            
            self.assertEqual(output_path, '/tmp/test_export.json')
            mock_file.assert_called_once()


if __name__ == '__main__':
    unittest.main()