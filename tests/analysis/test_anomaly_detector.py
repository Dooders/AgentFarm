"""
Tests for anomaly detector module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from farm.analysis.comparative.anomaly_detector import (
    AdvancedAnomalyDetector, AnomalyDetectionConfig, AnomalyResult
)
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, MetricsComparisonResult,
    DatabaseComparisonResult, LogComparisonResult
)


class TestAdvancedAnomalyDetector(unittest.TestCase):
    """Test cases for AdvancedAnomalyDetector."""
    
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
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_init(self):
        """Test AdvancedAnomalyDetector initialization."""
        detector = AdvancedAnomalyDetector()
        self.assertIsNotNone(detector.detectors)
        self.assertIsNotNone(detector.scalers)
        self.assertIsNotNone(detector.config)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_init_custom_config(self):
        """Test AdvancedAnomalyDetector initialization with custom config."""
        config = AnomalyDetectionConfig(contamination=0.2, n_neighbors=10)
        detector = AdvancedAnomalyDetector(config)
        self.assertEqual(detector.config.contamination, 0.2)
        self.assertEqual(detector.config.n_neighbors, 10)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', False)
    def test_init_no_sklearn(self):
        """Test AdvancedAnomalyDetector initialization without sklearn."""
        with self.assertRaises(ImportError):
            AdvancedAnomalyDetector()
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_detect_anomalies_ensemble(self):
        """Test detect_anomalies with ensemble method."""
        detector = AdvancedAnomalyDetector()
        
        with patch.object(detector, '_extract_features') as mock_extract, \
             patch.object(detector, '_ensemble_detection') as mock_ensemble:
            
            mock_extract.return_value = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
            mock_ensemble.return_value = AnomalyResult(
                anomalies=[], anomaly_scores=[], anomaly_types=[],
                severity_levels=[], confidence_scores=[], recommendations=[],
                detection_methods=[], summary={}
            )
            
            result = detector.detect_anomalies(self.mock_results, method='ensemble')
            
            self.assertIsInstance(result, AnomalyResult)
            mock_extract.assert_called_once()
            mock_ensemble.assert_called_once()
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_detect_anomalies_statistical(self):
        """Test detect_anomalies with statistical method."""
        detector = AdvancedAnomalyDetector()
        
        with patch.object(detector, '_extract_features') as mock_extract, \
             patch.object(detector, '_statistical_detection') as mock_statistical:
            
            mock_extract.return_value = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
            mock_statistical.return_value = AnomalyResult(
                anomalies=[], anomaly_scores=[], anomaly_types=[],
                severity_levels=[], confidence_scores=[], recommendations=[],
                detection_methods=[], summary={}
            )
            
            result = detector.detect_anomalies(self.mock_results, method='statistical')
            
            self.assertIsInstance(result, AnomalyResult)
            mock_statistical.assert_called_once()
    
    @patch('farm.analysis.comparative.anomaly_detection.SKLEARN_AVAILABLE', True)
    def test_detect_anomalies_single_method(self):
        """Test detect_anomalies with single method."""
        detector = AdvancedAnomalyDetector()
        
        with patch.object(detector, '_extract_features') as mock_extract, \
             patch.object(detector, '_single_method_detection') as mock_single:
            
            mock_extract.return_value = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
            mock_single.return_value = AnomalyResult(
                anomalies=[], anomaly_scores=[], anomaly_types=[],
                severity_levels=[], confidence_scores=[], recommendations=[],
                detection_methods=[], summary={}
            )
            
            result = detector.detect_anomalies(self.mock_results, method='isolation_forest')
            
            self.assertIsInstance(result, AnomalyResult)
            mock_single.assert_called_once()
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_detect_anomalies_insufficient_data(self):
        """Test detect_anomalies with insufficient data."""
        detector = AdvancedAnomalyDetector()
        
        result = detector.detect_anomalies([], method='ensemble')
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertEqual(result.summary['total_anomalies'], 0)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_extract_features(self):
        """Test _extract_features method."""
        detector = AdvancedAnomalyDetector()
        
        features_df = detector._extract_features(self.mock_results)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIn('simulation_id', features_df.columns)
        self.assertIn('total_differences', features_df.columns)
        self.assertIn('severity_numeric', features_df.columns)
        self.assertEqual(len(features_df), 5)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_extract_features_with_specific_features(self):
        """Test _extract_features with specific features."""
        detector = AdvancedAnomalyDetector()
        
        features = ['total_differences', 'config_differences']
        features_df = detector._extract_features(self.mock_results, features)
        
        self.assertIn('simulation_id', features_df.columns)
        self.assertIn('total_differences', features_df.columns)
        self.assertIn('config_differences', features_df.columns)
        self.assertNotIn('database_differences', features_df.columns)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_severity_to_numeric(self):
        """Test _severity_to_numeric method."""
        detector = AdvancedAnomalyDetector()
        
        self.assertEqual(detector._severity_to_numeric('low'), 1)
        self.assertEqual(detector._severity_to_numeric('medium'), 2)
        self.assertEqual(detector._severity_to_numeric('high'), 3)
        self.assertEqual(detector._severity_to_numeric('critical'), 4)
        self.assertEqual(detector._severity_to_numeric('unknown'), 0)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_ensemble_detection(self):
        """Test _ensemble_detection method."""
        detector = AdvancedAnomalyDetector()
        
        # Create test data
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'feature2': [1, 2, 3, 4, 5]
        })
        
        with patch.object(detector, '_classify_anomaly_type') as mock_classify, \
             patch.object(detector, '_determine_severity') as mock_severity, \
             patch.object(detector, '_generate_pair_recommendation') as mock_recommendation:
            
            mock_classify.return_value = 'general_anomaly'
            mock_severity.return_value = 'high'
            mock_recommendation.return_value = 'Test recommendation'
            
            result = detector._ensemble_detection(features_df, self.mock_results)
            
            self.assertIsInstance(result, AnomalyResult)
            self.assertIn('anomalies', result.__dict__)
            self.assertIn('anomaly_scores', result.__dict__)
            self.assertIn('anomaly_types', result.__dict__)
            self.assertIn('severity_levels', result.__dict__)
            self.assertIn('confidence_scores', result.__dict__)
            self.assertIn('recommendations', result.__dict__)
            self.assertIn('detection_methods', result.__dict__)
            self.assertIn('summary', result.__dict__)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_single_method_detection(self):
        """Test _single_method_detection method."""
        detector = AdvancedAnomalyDetector()
        
        # Create test data
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'feature2': [1, 2, 3, 4, 5]
        })
        
        with patch.object(detector, '_classify_anomaly_type') as mock_classify, \
             patch.object(detector, '_determine_severity') as mock_severity, \
             patch.object(detector, '_generate_pair_recommendation') as mock_recommendation:
            
            mock_classify.return_value = 'general_anomaly'
            mock_severity.return_value = 'high'
            mock_recommendation.return_value = 'Test recommendation'
            
            result = detector._single_method_detection(features_df, self.mock_results, 'isolation_forest')
            
            self.assertIsInstance(result, AnomalyResult)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_single_method_detection_unknown_method(self):
        """Test _single_method_detection with unknown method."""
        detector = AdvancedAnomalyDetector()
        
        features_df = pd.DataFrame({'simulation_id': [0], 'feature1': [1]})
        
        result = detector._single_method_detection(features_df, self.mock_results, 'unknown_method')
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertEqual(result.summary['total_anomalies'], 0)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    @patch('farm.analysis.comparative.anomaly_detector.SCIPY_AVAILABLE', True)
    def test_statistical_detection(self):
        """Test _statistical_detection method."""
        detector = AdvancedAnomalyDetector()
        
        # Create test data with outliers
        features_df = pd.DataFrame({
            'simulation_id': [0, 1, 2, 3, 4],
            'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'feature2': [1, 2, 3, 4, 5]
        })
        
        with patch.object(detector, '_classify_anomaly_type') as mock_classify, \
             patch.object(detector, '_determine_severity') as mock_severity, \
             patch.object(detector, '_generate_pair_recommendation') as mock_recommendation:
            
            mock_classify.return_value = 'general_anomaly'
            mock_severity.return_value = 'high'
            mock_recommendation.return_value = 'Test recommendation'
            
            result = detector._statistical_detection(features_df, self.mock_results)
            
            self.assertIsInstance(result, AnomalyResult)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    @patch('farm.analysis.comparative.anomaly_detector.SCIPY_AVAILABLE', False)
    def test_statistical_detection_no_scipy(self):
        """Test _statistical_detection without scipy."""
        detector = AdvancedAnomalyDetector()
        
        features_df = pd.DataFrame({'simulation_id': [0], 'feature1': [1]})
        
        result = detector._statistical_detection(features_df, self.mock_results)
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertEqual(result.summary['total_anomalies'], 0)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_classify_anomaly_type(self):
        """Test _classify_anomaly_type method."""
        detector = AdvancedAnomalyDetector()
        
        # Test high differences
        features = pd.Series({'total_differences': 60})
        result = detector._classify_anomaly_type(features)
        self.assertEqual(result, 'high_differences')
        
        # Test performance anomaly
        features = pd.Series({'perf_execution_time_ratio': 2.5})
        result = detector._classify_anomaly_type(features)
        self.assertEqual(result, 'performance_anomaly')
        
        # Test error anomaly
        features = pd.Series({'error_increase': 15})
        result = detector._classify_anomaly_type(features)
        self.assertEqual(result, 'error_anomaly')
        
        # Test general anomaly
        features = pd.Series({'total_differences': 5})
        result = detector._classify_anomaly_type(features)
        self.assertEqual(result, 'general_anomaly')
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_determine_severity(self):
        """Test _determine_severity method."""
        detector = AdvancedAnomalyDetector()
        
        # Test critical severity
        features = pd.Series({'total_differences': 150})
        result = detector._determine_severity(0.9, features)
        self.assertEqual(result, 'critical')
        
        # Test high severity
        features = pd.Series({'total_differences': 60})
        result = detector._determine_severity(0.7, features)
        self.assertEqual(result, 'high')
        
        # Test medium severity
        features = pd.Series({'total_differences': 30})
        result = detector._determine_severity(0.5, features)
        self.assertEqual(result, 'medium')
        
        # Test low severity
        features = pd.Series({'total_differences': 5})
        result = detector._determine_severity(0.3, features)
        self.assertEqual(result, 'low')
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_generate_pair_recommendation(self):
        """Test _generate_pair_recommendation method."""
        detector = AdvancedAnomalyDetector()
        
        # Test high similarity
        result = detector._generate_pair_recommendation(1, 2, 0.95, ['f1'], ['f2'])
        self.assertIn('very similar', result)
        
        # Test medium similarity
        result = detector._generate_pair_recommendation(1, 2, 0.85, ['f1'], ['f2'])
        self.assertIn('highly similar', result)
        
        # Test low similarity
        result = detector._generate_pair_recommendation(1, 2, 0.75, ['f1'], ['f2'])
        self.assertIn('moderate similarity', result)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_generate_anomaly_summary(self):
        """Test _generate_anomaly_summary method."""
        detector = AdvancedAnomalyDetector()
        
        anomalies = [
            {'simulation_id': 1, 'severity': 'high'},
            {'simulation_id': 2, 'severity': 'medium'}
        ]
        scores = [0.8, 0.6]
        types = ['performance_anomaly', 'error_anomaly']
        
        result = detector._generate_anomaly_summary(anomalies, scores, types)
        
        self.assertIn('total_anomalies', result)
        self.assertIn('anomaly_rate', result)
        self.assertIn('severity_distribution', result)
        self.assertIn('type_distribution', result)
        self.assertIn('average_confidence', result)
        self.assertIn('top_anomalies', result)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_generate_anomaly_summary_empty(self):
        """Test _generate_anomaly_summary with empty data."""
        detector = AdvancedAnomalyDetector()
        
        result = detector._generate_anomaly_summary([], [], [])
        
        self.assertEqual(result['total_anomalies'], 0)
        self.assertEqual(result['anomaly_rate'], 0.0)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_create_empty_result(self):
        """Test _create_empty_result method."""
        detector = AdvancedAnomalyDetector()
        
        result = detector._create_empty_result()
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertEqual(result.summary['total_anomalies'], 0)
    
    @patch('farm.analysis.comparative.anomaly_detector.SKLEARN_AVAILABLE', True)
    def test_export_anomaly_results(self):
        """Test export_anomaly_results method."""
        detector = AdvancedAnomalyDetector()
        
        result = AnomalyResult(
            anomalies=[], anomaly_scores=[], anomaly_types=[],
            severity_levels=[], confidence_scores=[], recommendations=[],
            detection_methods=[], summary={}
        )
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            output_path = detector.export_anomaly_results(result, '/tmp/test_export.json')
            
            self.assertEqual(output_path, '/tmp/test_export.json')
            mock_file.assert_called_once()


if __name__ == '__main__':
    unittest.main()