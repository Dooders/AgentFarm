"""
Tests for StatisticalAnalyzer.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from farm.analysis.comparative.statistical_analyzer import StatisticalAnalyzer, StatisticalAnalysisResult
from farm.analysis.comparative.comparison_result import (
    SimulationComparisonResult, ComparisonSummary, ConfigComparisonResult,
    DatabaseComparisonResult, LogComparisonResult, MetricsComparisonResult
)


class TestStatisticalAnalyzer:
    """Test cases for StatisticalAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer(significance_level=0.05)
        self.mock_result = self._create_mock_comparison_result()
    
    def _create_mock_comparison_result(self):
        """Create a mock comparison result for testing."""
        summary = ComparisonSummary(
            total_differences=10,
            severity="medium",
            config_differences=2,
            database_differences=3,
            log_differences=2,
            metrics_differences=3,
            comparison_time="2024-01-01T00:00:00"
        )
        
        config_comparison = ConfigComparisonResult(
            differences={"param1": "value1", "param2": ["item1", "item2"]}
        )
        
        database_comparison = DatabaseComparisonResult(
            schema_differences={"table1": {"column1": "type_change"}},
            data_differences={"table1": {"difference": 5}},
            metric_differences={
                "db_metric1": {"db1_value": 10, "db2_value": 15, "difference": 5},
                "db_metric2": {"db1_value": 20, "db2_value": 18, "difference": -2}
            }
        )
        
        log_comparison = LogComparisonResult(
            performance_differences={
                "cpu_usage": {"sim1_value": 50, "sim2_value": 60, "difference": 10},
                "memory_usage": {"sim1_value": 100, "sim2_value": 90, "difference": -10}
            },
            error_differences={
                "error1": {"sim1_count": 2, "sim2_count": 50, "difference": 48},
                "error2": {"sim1_count": 10, "sim2_count": 1, "difference": -9}
            }
        )
        
        metrics_comparison = MetricsComparisonResult(
            metric_differences={
                "throughput": {"sim1_value": 100, "sim2_value": 200, "difference": 100, "percentage_change": 100.0},
                "latency": {"sim1_value": 50, "sim2_value": 10, "difference": -40, "percentage_change": -80.0}
            },
            performance_comparison={
                "response_time": {"ratio": 0.3, "faster": "sim2"},
                "throughput": {"ratio": 3.0, "faster": "sim2"}
            }
        )
        
        return SimulationComparisonResult(
            simulation1_path=Mock(),
            simulation2_path=Mock(),
            comparison_summary=summary,
            config_comparison=config_comparison,
            database_comparison=database_comparison,
            log_comparison=log_comparison,
            metrics_comparison=metrics_comparison,
            metadata={}
        )
    
    def test_init(self):
        """Test StatisticalAnalyzer initialization."""
        assert self.analyzer.significance_level == 0.05
        
        # Test with custom significance level
        analyzer2 = StatisticalAnalyzer(significance_level=0.01)
        assert analyzer2.significance_level == 0.01
    
    def test_analyze_comparison(self):
        """Test comprehensive analysis."""
        result = self.analyzer.analyze_comparison(self.mock_result)
        
        assert isinstance(result, StatisticalAnalysisResult)
        assert 'metrics_performance' in result.correlation_analysis
        assert 'overall_significance' in result.significance_tests
        assert 'metric_trends' in result.trend_analysis
        assert 'summary' in result.anomaly_detection
        assert 'analysis_quality' in result.summary
    
    def test_extract_metrics_data(self):
        """Test extracting metrics data."""
        data = self.analyzer._extract_metrics_data(self.mock_result)
        
        assert data is not None
        assert 'throughput_sim1' in data
        assert 'throughput_sim2' in data
        assert 'throughput_diff' in data
        assert 'latency_sim1' in data
        assert 'latency_sim2' in data
        assert 'latency_diff' in data
    
    def test_extract_metrics_data_no_metrics(self):
        """Test extracting metrics data with no metrics."""
        result = SimulationComparisonResult(
            simulation1_path=Mock(),
            simulation2_path=Mock(),
            comparison_summary=ComparisonSummary(
                total_differences=0,
                severity="low",
                config_differences=0,
                database_differences=0,
                log_differences=0,
                metrics_differences=0,
                comparison_time=datetime(2024, 1, 1, 0, 0, 0)
            ),
            config_comparison=ConfigComparisonResult(differences={}),
            database_comparison=DatabaseComparisonResult(schema_differences={}, data_differences={}, metric_differences={}),
            log_comparison=LogComparisonResult(performance_differences={}, error_differences={}),
            metrics_comparison=MetricsComparisonResult(metric_differences={}, performance_comparison={}),
            metadata={}
        )
        
        data = self.analyzer._extract_metrics_data(result)
        assert data is None
    
    def test_extract_performance_data(self):
        """Test extracting performance data."""
        data = self.analyzer._extract_performance_data(self.mock_result)
        
        assert data is not None
        assert 'log_cpu_usage_sim1' in data
        assert 'log_cpu_usage_sim2' in data
        assert 'log_cpu_usage_diff' in data
        assert 'metrics_response_time_ratio' in data
        assert 'metrics_throughput_ratio' in data
    
    def test_extract_error_data(self):
        """Test extracting error data."""
        data = self.analyzer._extract_error_data(self.mock_result)
        
        assert data is not None
        assert 'error1_sim1' in data
        assert 'error1_sim2' in data
        assert 'error1_diff' in data
        assert 'error2_sim1' in data
        assert 'error2_sim2' in data
        assert 'error2_diff' in data
    
    def test_calculate_correlation(self):
        """Test correlation calculation."""
        data1 = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        data2 = {'a': 2.0, 'b': 4.0, 'c': 6.0}
        
        result = self.analyzer._calculate_correlation(data1, data2, 'Test Correlation')
        
        assert isinstance(result, dict)
        assert 'label' in result
        assert 'correlation' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert result['label'] == 'Test Correlation'
        assert result['correlation'] == 1.0  # Perfect correlation
        assert result['significant'] == True
    
    def test_calculate_correlation_insufficient_data(self):
        """Test correlation calculation with insufficient data."""
        data1 = {'a': 1.0}
        data2 = {'a': 2.0}
        
        result = self.analyzer._calculate_correlation(data1, data2, 'Test Correlation')
        
        assert result['correlation'] == 0.0
        assert result['p_value'] == 1.0
        assert result['significant'] is False
    
    def test_interpret_correlation(self):
        """Test correlation interpretation."""
        assert self.analyzer._interpret_correlation(0.05) == 'Negligible'
        assert self.analyzer._interpret_correlation(0.2) == 'Weak'
        assert self.analyzer._interpret_correlation(0.4) == 'Moderate'
        assert self.analyzer._interpret_correlation(0.6) == 'Strong'
        assert self.analyzer._interpret_correlation(0.8) == 'Very Strong'
    
    def test_test_overall_significance(self):
        """Test overall significance test."""
        result = self.analyzer._test_overall_significance(self.mock_result)
        
        assert isinstance(result, dict)
        assert 'test_name' in result
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'interpretation' in result
        assert result['test_name'] == 'Overall Differences'
        assert result['significant'] is True  # Mock has differences
    
    def test_test_metric_significance(self):
        """Test metric significance test."""
        result = self.analyzer._test_metric_significance(self.mock_result)
        
        assert isinstance(result, dict)
        assert 'test_name' in result
        assert 'significant_metrics' in result
        assert 'non_significant_metrics' in result
        assert 'total_significant' in result
        assert 'total_tested' in result
        assert result['test_name'] == 'Metric Differences'
    
    def test_test_performance_significance(self):
        """Test performance significance test."""
        result = self.analyzer._test_performance_significance(self.mock_result)
        
        assert isinstance(result, dict)
        assert 'test_name' in result
        assert 'significant_performance_changes' in result
        assert 'total_significant' in result
        assert result['test_name'] == 'Performance Differences'
    
    def test_test_error_significance(self):
        """Test error significance test."""
        result = self.analyzer._test_error_significance(self.mock_result)
        
        assert isinstance(result, dict)
        assert 'test_name' in result
        assert 'significant_error_changes' in result
        assert 'total_significant' in result
        assert result['test_name'] == 'Error Differences'
    
    def test_analyze_metric_trends(self):
        """Test metric trend analysis."""
        result = self.analyzer._analyze_metric_trends(self.mock_result)
        
        assert isinstance(result, dict)
        assert 'improving_metrics' in result
        assert 'degrading_metrics' in result
        assert 'stable_metrics' in result
        assert 'trend_summary' in result
        
        # Check that throughput is improving (20% increase)
        improving_metrics = result['improving_metrics']
        assert len(improving_metrics) > 0
        assert any(m['metric'] == 'throughput' for m in improving_metrics)
        
        # Check that latency is degrading (-20% change)
        degrading_metrics = result['degrading_metrics']
        assert len(degrading_metrics) > 0
        assert any(m['metric'] == 'latency' for m in degrading_metrics)
    
    def test_analyze_performance_trends(self):
        """Test performance trend analysis."""
        result = self.analyzer._analyze_performance_trends(self.mock_result)
        
        assert isinstance(result, dict)
        assert 'faster_metrics' in result
        assert 'slower_metrics' in result
        assert 'unchanged_metrics' in result
        assert 'performance_summary' in result
    
    def test_analyze_error_trends(self):
        """Test error trend analysis."""
        result = self.analyzer._analyze_error_trends(self.mock_result)
        
        assert isinstance(result, dict)
        assert 'increasing_errors' in result
        assert 'decreasing_errors' in result
        assert 'unchanged_errors' in result
        assert 'error_summary' in result
    
    def test_analyze_overall_trend(self):
        """Test overall trend analysis."""
        result = self.analyzer._analyze_overall_trend(self.mock_result)
        
        assert isinstance(result, dict)
        assert 'trend' in result
        assert 'trend_strength' in result
        assert 'positive_factors' in result
        assert 'negative_factors' in result
        assert 'interpretation' in result
        assert result['trend'] in ['positive', 'negative', 'neutral']
    
    def test_interpret_trend(self):
        """Test trend interpretation."""
        assert 'improvements' in self.analyzer._interpret_trend('positive', 0.8)
        assert 'degradations' in self.analyzer._interpret_trend('negative', 0.8)
        assert 'balanced' in self.analyzer._interpret_trend('neutral', 0.5)
    
    def test_detect_metric_anomalies(self):
        """Test metric anomaly detection."""
        result = self.analyzer._detect_metric_anomalies(self.mock_result)
        
        assert isinstance(result, list)
        # Should detect throughput as anomalous (>50% change)
        assert len(result) > 0
        assert any(a['metric'] == 'throughput' for a in result)
    
    def test_detect_performance_anomalies(self):
        """Test performance anomaly detection."""
        result = self.analyzer._detect_performance_anomalies(self.mock_result)
        
        assert isinstance(result, list)
        # Should detect response_time as anomalous (0.8 ratio)
        assert len(result) > 0
        assert any(a['metric'] == 'response_time' for a in result)
    
    def test_detect_error_anomalies(self):
        """Test error anomaly detection."""
        result = self.analyzer._detect_error_anomalies(self.mock_result)
        
        assert isinstance(result, list)
        # Should detect error1 as anomalous (2.5x increase)
        assert len(result) > 0
        assert any(a['error_type'] == 'error1' for a in result)
    
    def test_summarize_anomalies(self):
        """Test anomaly summarization."""
        anomalies = {
            'metric_anomalies': [
                {'severity': 'high'},
                {'severity': 'medium'},
                {'severity': 'low'}
            ],
            'performance_anomalies': [
                {'severity': 'high'},
                {'severity': 'medium'}
            ],
            'error_anomalies': [
                {'severity': 'medium'}
            ]
        }
        
        result = self.analyzer._summarize_anomalies(anomalies)
        
        assert isinstance(result, dict)
        assert 'total_anomalies' in result
        assert 'high_severity' in result
        assert 'medium_severity' in result
        assert 'low_severity' in result
        assert 'has_anomalies' in result
        assert result['total_anomalies'] == 6
        assert result['high_severity'] == 2
        assert result['medium_severity'] == 3
        assert result['low_severity'] == 1
        assert result['has_anomalies'] is True
    
    def test_assess_analysis_quality(self):
        """Test analysis quality assessment."""
        # High quality
        quality = self.analyzer._assess_analysis_quality(3, 2, {'total_anomalies': 0, 'high_severity': 0})
        assert 'High quality' in quality
        
        # Good quality
        quality = self.analyzer._assess_analysis_quality(2, 1, {'total_anomalies': 1, 'high_severity': 0})
        assert 'Good quality' in quality
        
        # Moderate quality
        quality = self.analyzer._assess_analysis_quality(1, 1, {'total_anomalies': 2, 'high_severity': 1})
        assert 'Moderate quality' in quality
        
        # Fair quality
        quality = self.analyzer._assess_analysis_quality(1, 0, {'total_anomalies': 0, 'high_severity': 0})
        assert 'Fair quality' in quality
        
        # Limited quality
        quality = self.analyzer._assess_analysis_quality(0, 0, {'total_anomalies': 0, 'high_severity': 0})
        assert 'Limited quality' in quality
    
    def test_export_analysis_results(self):
        """Test exporting analysis results."""
        analysis_result = self.analyzer.analyze_comparison(self.mock_result)
        
        # Test export
        export_path = "test_analysis.json"
        result_path = self.analyzer.export_analysis_results(analysis_result, export_path)
        
        assert result_path == export_path
        
        # Clean up
        import os
        if os.path.exists(export_path):
            os.remove(export_path)
    
    def test_t_cdf_approximation(self):
        """Test t-distribution CDF approximation."""
        # Test with large df (normal approximation)
        result = self.analyzer._t_cdf(1.96, 50)
        assert 0.9 < result < 1.0
        
        # Test with small df
        result = self.analyzer._t_cdf(2.0, 5)
        assert 0.8 < result < 1.0
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        # Create result with empty data
        empty_result = SimulationComparisonResult(
            simulation1_path=Mock(),
            simulation2_path=Mock(),
            comparison_summary=ComparisonSummary(
                total_differences=0,
                severity="low",
                config_differences=0,
                database_differences=0,
                log_differences=0,
                metrics_differences=0,
                comparison_time=datetime(2024, 1, 1, 0, 0, 0)
            ),
            config_comparison=ConfigComparisonResult(differences={}),
            database_comparison=DatabaseComparisonResult(schema_differences={}, data_differences={}, metric_differences={}),
            log_comparison=LogComparisonResult(performance_differences={}, error_differences={}),
            metrics_comparison=MetricsComparisonResult(metric_differences={}, performance_comparison={}),
            metadata={}
        )
        
        # Should not raise exceptions
        result = self.analyzer.analyze_comparison(empty_result)
        assert isinstance(result, StatisticalAnalysisResult)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data types."""
        # Create result with invalid data
        invalid_result = SimulationComparisonResult(
            simulation1_path=Mock(),
            simulation2_path=Mock(),
            comparison_summary=ComparisonSummary(
                total_differences=0,
                severity="low",
                config_differences=0,
                database_differences=0,
                log_differences=0,
                metrics_differences=0,
                comparison_time=datetime(2024, 1, 1, 0, 0, 0)
            ),
            config_comparison=ConfigComparisonResult(differences={}),
            database_comparison=DatabaseComparisonResult(schema_differences={}, data_differences={}, metric_differences={}),
            log_comparison=LogComparisonResult(performance_differences={}, error_differences={}),
            metrics_comparison=MetricsComparisonResult(
                metric_differences={
                    "invalid_metric": {"sim1_value": "not_a_number", "sim2_value": "also_not_a_number"}
                },
                performance_comparison={}
            ),
            metadata={}
        )
        
        # Should not raise exceptions
        result = self.analyzer.analyze_comparison(invalid_result)
        assert isinstance(result, StatisticalAnalysisResult)