"""
Tests for trend predictor module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path, timedelta

from farm.analysis.comparative.trend_predictor import (
    TrendPredictor, TrendPredictionConfig, TrendPredictionResult, TrendAnalysis
)
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, MetricsComparisonResult,
    DatabaseComparisonResult, LogComparisonResult
)


class TestTrendPredictor(unittest.TestCase):
    """Test cases for TrendPredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_results = self._create_mock_comparison_results()
    
    def _create_mock_comparison_results(self):
        """Create mock comparison results for testing."""
        results = []
        
        for i in range(10):
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
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_init(self):
        """Test TrendPredictor initialization."""
        predictor = TrendPredictor()
        self.assertIsNotNone(predictor.regression_models)
        self.assertIsNotNone(predictor.scaler)
        self.assertIsNotNone(predictor.config)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_init_custom_config(self):
        """Test TrendPredictor initialization with custom config."""
        config = TrendPredictionConfig(forecast_horizon=20, confidence_level=0.95)
        predictor = TrendPredictor(config)
        self.assertEqual(predictor.config.forecast_horizon, 20)
        self.assertEqual(predictor.config.confidence_level, 0.95)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', False)
    def test_init_no_sklearn(self):
        """Test TrendPredictor initialization without sklearn."""
        with self.assertRaises(ImportError):
            TrendPredictor()
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_predict_trends(self):
        """Test predict_trends method."""
        predictor = TrendPredictor()
        
        with patch.object(predictor, '_extract_time_series_data') as mock_extract, \
             patch.object(predictor, '_analyze_trends') as mock_analyze, \
             patch.object(predictor, '_make_predictions') as mock_predict, \
             patch.object(predictor, '_calculate_accuracy') as mock_accuracy, \
             patch.object(predictor, '_evaluate_models') as mock_evaluate, \
             patch.object(predictor, '_generate_recommendations') as mock_recommend:
            
            mock_extract.return_value = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
            mock_analyze.return_value = {'feature1': TrendAnalysis('increasing', 0.8, 0.9, [1, 2, 3], 'linear')}
            mock_predict.return_value = {'feature1': {'predictions': [6, 7, 8], 'confidence_intervals': [[5, 7], [6, 8], [7, 9]]}}
            mock_accuracy.return_value = {'feature1': {'mae': 0.5, 'rmse': 0.7, 'r2': 0.8}}
            mock_evaluate.return_value = {'feature1': {'model_name': 'LinearRegression', 'performance': 0.8}}
            mock_recommend.return_value = {'feature1': ['Recommendation 1', 'Recommendation 2']}
            
            result = predictor.predict_trends(self.mock_results)
            
            self.assertIsInstance(result, TrendPredictionResult)
            mock_extract.assert_called_once()
            mock_analyze.assert_called_once()
            mock_predict.assert_called_once()
            mock_accuracy.assert_called_once()
            mock_evaluate.assert_called_once()
            mock_recommend.assert_called_once()
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_predict_trends_insufficient_data(self):
        """Test predict_trends with insufficient data."""
        predictor = TrendPredictor()
        
        result = predictor.predict_trends([], time_series=False)
        
        self.assertIsInstance(result, TrendPredictionResult)
        self.assertEqual(result.forecast_accuracy, {})
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_extract_time_series_data(self):
        """Test _extract_time_series_data method."""
        predictor = TrendPredictor()
        
        result = predictor._extract_time_series_data(self.mock_results)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('timestamp', result.columns)
        self.assertIn('total_differences', result.columns)
        self.assertIn('severity_numeric', result.columns)
        self.assertEqual(len(result), 10)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_extract_time_series_data_with_specific_features(self):
        """Test _extract_time_series_data with specific features."""
        predictor = TrendPredictor()
        
        features = ['total_differences', 'config_differences']
        result = predictor._extract_time_series_data(self.mock_results, features)
        
        self.assertIn('timestamp', result.columns)
        self.assertIn('total_differences', result.columns)
        self.assertIn('config_differences', result.columns)
        self.assertNotIn('database_differences', result.columns)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_severity_to_numeric(self):
        """Test _severity_to_numeric method."""
        predictor = TrendPredictor()
        
        self.assertEqual(predictor._severity_to_numeric('low'), 1)
        self.assertEqual(predictor._severity_to_numeric('medium'), 2)
        self.assertEqual(predictor._severity_to_numeric('high'), 3)
        self.assertEqual(predictor._severity_to_numeric('critical'), 4)
        self.assertEqual(predictor._severity_to_numeric('unknown'), 0)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_analyze_trends(self):
        """Test _analyze_trends method."""
        predictor = TrendPredictor()
        
        time_series_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        })
        
        result = predictor._analyze_trends(time_series_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('feature1', result)
        self.assertIn('feature2', result)
        
        for feature, trend_analysis in result.items():
            self.assertIsInstance(trend_analysis, TrendAnalysis)
            self.assertIn(trend_analysis.direction, ['increasing', 'decreasing', 'stable'])
            self.assertGreaterEqual(trend_analysis.strength, 0.0)
            self.assertLessEqual(trend_analysis.strength, 1.0)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_detect_trend(self):
        """Test _detect_trend method."""
        predictor = TrendPredictor()
        
        # Test increasing trend
        values = np.array([1, 2, 3, 4, 5])
        direction, strength = predictor._detect_trend(values)
        self.assertEqual(direction, 'increasing')
        self.assertGreater(strength, 0.5)
        
        # Test decreasing trend
        values = np.array([5, 4, 3, 2, 1])
        direction, strength = predictor._detect_trend(values)
        self.assertEqual(direction, 'decreasing')
        self.assertGreater(strength, 0.5)
        
        # Test stable trend
        values = np.array([3, 3, 3, 3, 3])
        direction, strength = predictor._detect_trend(values)
        self.assertEqual(direction, 'stable')
        self.assertLess(strength, 0.5)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_detect_seasonality(self):
        """Test _detect_seasonality method."""
        predictor = TrendPredictor()
        
        # Test seasonal data
        values = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        has_seasonality, period = predictor._detect_seasonality(values)
        self.assertTrue(has_seasonality)
        self.assertEqual(period, 3)
        
        # Test non-seasonal data
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        has_seasonality, period = predictor._detect_seasonality(values)
        self.assertFalse(has_seasonality)
        self.assertEqual(period, 0)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_detect_change_points(self):
        """Test _detect_change_points method."""
        predictor = TrendPredictor()
        
        # Test data with change point
        values = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13, 14])
        change_points = predictor._detect_change_points(values)
        
        self.assertIsInstance(change_points, list)
        self.assertGreater(len(change_points), 0)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_make_predictions(self):
        """Test _make_predictions method."""
        predictor = TrendPredictor()
        
        time_series_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        })
        
        trend_analysis = {
            'feature1': TrendAnalysis('increasing', 0.8, 0.9, [1, 2, 3], 'linear'),
            'feature2': TrendAnalysis('decreasing', 0.7, 0.8, [10, 9, 8], 'linear')
        }
        
        with patch.object(predictor, '_select_best_model') as mock_select, \
             patch.object(predictor, '_predict_feature') as mock_predict:
            
            mock_select.return_value = 'LinearRegression'
            mock_predict.return_value = {
                'predictions': [11, 12, 13],
                'confidence_intervals': [[10, 12], [11, 13], [12, 14]]
            }
            
            result = predictor._make_predictions(time_series_data, trend_analysis)
            
            self.assertIsInstance(result, dict)
            self.assertIn('feature1', result)
            self.assertIn('feature2', result)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_select_best_model(self):
        """Test _select_best_model method."""
        predictor = TrendPredictor()
        
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        with patch.object(predictor, '_evaluate_model') as mock_evaluate:
            mock_evaluate.return_value = 0.9
            
            result = predictor._select_best_model(X, y)
            
            self.assertIsInstance(result, str)
            self.assertIn(result, ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForestRegressor', 'GradientBoostingRegressor'])
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_evaluate_model(self):
        """Test _evaluate_model method."""
        predictor = TrendPredictor()
        
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        with patch.object(predictor, '_cross_validate_model') as mock_cv:
            mock_cv.return_value = {'r2': 0.9, 'mae': 0.5, 'rmse': 0.7}
            
            result = predictor._evaluate_model('LinearRegression', X, y)
            
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_cross_validate_model(self):
        """Test _cross_validate_model method."""
        predictor = TrendPredictor()
        
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        result = predictor._cross_validate_model('LinearRegression', X, y)
        
        self.assertIsInstance(result, dict)
        self.assertIn('r2', result)
        self.assertIn('mae', result)
        self.assertIn('rmse', result)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_predict_feature(self):
        """Test _predict_feature method."""
        predictor = TrendPredictor()
        
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model_name = 'LinearRegression'
        forecast_horizon = 3
        
        with patch.object(predictor, '_get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([12, 14, 16])
            mock_get_model.return_value = mock_model
            
            result = predictor._predict_feature(X, y, model_name, forecast_horizon)
            
            self.assertIsInstance(result, dict)
            self.assertIn('predictions', result)
            self.assertIn('confidence_intervals', result)
            self.assertEqual(len(result['predictions']), forecast_horizon)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_get_model(self):
        """Test _get_model method."""
        predictor = TrendPredictor()
        
        model = predictor._get_model('LinearRegression')
        self.assertIsNotNone(model)
        
        with self.assertRaises(ValueError):
            predictor._get_model('UnknownModel')
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_calculate_accuracy(self):
        """Test _calculate_accuracy method."""
        predictor = TrendPredictor()
        
        time_series_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        predictions = {
            'feature1': {
                'predictions': [11, 12, 13],
                'confidence_intervals': [[10, 12], [11, 13], [12, 14]]
            }
        }
        
        result = predictor._calculate_accuracy(time_series_data, predictions)
        
        self.assertIsInstance(result, dict)
        self.assertIn('feature1', result)
        self.assertIn('mae', result['feature1'])
        self.assertIn('rmse', result['feature1'])
        self.assertIn('r2', result['feature1'])
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_evaluate_models(self):
        """Test _evaluate_models method."""
        predictor = TrendPredictor()
        
        time_series_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        trend_analysis = {
            'feature1': TrendAnalysis('increasing', 0.8, 0.9, [1, 2, 3], 'linear')
        }
        
        result = predictor._evaluate_models(time_series_data, trend_analysis)
        
        self.assertIsInstance(result, dict)
        self.assertIn('feature1', result)
        self.assertIn('model_name', result['feature1'])
        self.assertIn('performance', result['feature1'])
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_generate_recommendations(self):
        """Test _generate_recommendations method."""
        predictor = TrendPredictor()
        
        trend_analysis = {
            'feature1': TrendAnalysis('increasing', 0.8, 0.9, [1, 2, 3], 'linear'),
            'feature2': TrendAnalysis('decreasing', 0.7, 0.8, [10, 9, 8], 'linear')
        }
        
        predictions = {
            'feature1': {'predictions': [11, 12, 13]},
            'feature2': {'predictions': [7, 6, 5]}
        }
        
        result = predictor._generate_recommendations(trend_analysis, predictions)
        
        self.assertIsInstance(result, dict)
        self.assertIn('feature1', result)
        self.assertIn('feature2', result)
        
        for feature, recommendations in result.items():
            self.assertIsInstance(recommendations, list)
            self.assertGreater(len(recommendations), 0)
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_generate_feature_recommendations(self):
        """Test _generate_feature_recommendations method."""
        predictor = TrendPredictor()
        
        # Test increasing trend
        trend_analysis = TrendAnalysis('increasing', 0.8, 0.9, [1, 2, 3], 'linear')
        predictions = {'predictions': [11, 12, 13]}
        
        result = predictor._generate_feature_recommendations('feature1', trend_analysis, predictions)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn('increasing', str(result).lower())
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_create_empty_result(self):
        """Test _create_empty_result method."""
        predictor = TrendPredictor()
        
        result = predictor._create_empty_result()
        
        self.assertIsInstance(result, TrendPredictionResult)
        self.assertEqual(result.forecast_accuracy, {})
        self.assertEqual(result.model_performance, {})
        self.assertEqual(result.recommendations, {})
    
    @patch('farm.analysis.comparative.trend_predictor.SKLEARN_AVAILABLE', True)
    def test_export_prediction_results(self):
        """Test export_prediction_results method."""
        predictor = TrendPredictor()
        
        result = TrendPredictionResult(
            predictions={}, trend_analysis={}, forecast_accuracy={},
            model_performance={}, recommendations={}, summary={}
        )
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            output_path = predictor.export_prediction_results(result, '/tmp/test_export.json')
            
            self.assertEqual(output_path, '/tmp/test_export.json')
            mock_file.assert_called_once()


if __name__ == '__main__':
    unittest.main()