"""
Tests for Predictive Analytics.

This module contains comprehensive tests for the predictive analytics system
that provides forecasting and prediction capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil
import numpy as np

from farm.analysis.comparative.predictive_analytics import (
    PredictiveAnalytics,
    PredictionType,
    PredictionConfidence,
    Prediction,
    TimeSeriesData,
    PredictionConfig
)
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult


class TestPredictiveAnalytics:
    """Test cases for PredictiveAnalytics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = PredictionConfig(
            enable_ml_predictions=True,
            enable_statistical_predictions=True,
            prediction_horizon=30,
            confidence_threshold=0.7,
            enable_anomaly_detection=True,
            enable_trend_analysis=True,
            enable_resource_forecasting=True,
            enable_analysis_success_prediction=True
        )
        self.analytics = PredictiveAnalytics(config=self.config)
        
        # Mock historical data
        self.mock_historical_data = [
            {
                "timestamp": datetime.now() - timedelta(days=30),
                "cpu_usage": 70.0,
                "memory_usage": 60.0,
                "disk_io": 40.0,
                "simulation_time": 1000,
                "success_rate": 0.95
            },
            {
                "timestamp": datetime.now() - timedelta(days=25),
                "cpu_usage": 75.0,
                "memory_usage": 65.0,
                "disk_io": 45.0,
                "simulation_time": 1100,
                "success_rate": 0.93
            },
            {
                "timestamp": datetime.now() - timedelta(days=20),
                "cpu_usage": 80.0,
                "memory_usage": 70.0,
                "disk_io": 50.0,
                "simulation_time": 1200,
                "success_rate": 0.91
            },
            {
                "timestamp": datetime.now() - timedelta(days=15),
                "cpu_usage": 85.0,
                "memory_usage": 75.0,
                "disk_io": 55.0,
                "simulation_time": 1300,
                "success_rate": 0.89
            },
            {
                "timestamp": datetime.now() - timedelta(days=10),
                "cpu_usage": 90.0,
                "memory_usage": 80.0,
                "disk_io": 60.0,
                "simulation_time": 1400,
                "success_rate": 0.87
            },
            {
                "timestamp": datetime.now() - timedelta(days=5),
                "cpu_usage": 95.0,
                "memory_usage": 85.0,
                "disk_io": 65.0,
                "simulation_time": 1500,
                "success_rate": 0.85
            }
        ]
        
        # Mock current data
        self.mock_current_data = {
            "cpu_usage": 95.0,
            "memory_usage": 85.0,
            "disk_io": 65.0,
            "simulation_time": 1500,
            "success_rate": 0.85
        }
        
        # Mock analysis config
        self.mock_analysis_config = {
            "simulation_type": "agent_based",
            "agent_count": 1000,
            "simulation_duration": 3600,
            "complexity_level": "high"
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test analytics initialization."""
        assert self.analytics.config == self.config
        assert self.analytics.prediction_history == []
        assert self.analytics.model_performance == {}
        assert self.analytics.anomaly_threshold == 0.1
        assert self.analytics.trend_threshold == 0.05
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        analytics = PredictiveAnalytics()
        assert analytics.config is not None
        assert analytics.config.enable_ml_predictions is True
        assert analytics.config.enable_statistical_predictions is True
    
    def test_initialization_without_sklearn(self):
        """Test initialization without sklearn."""
        with patch('farm.analysis.comparative.predictive_analytics.SKLEARN_AVAILABLE', False):
            analytics = PredictiveAnalytics()
            assert analytics.sklearn_available is False
            assert analytics.ml_models == {}
    
    def test_initialization_without_scipy(self):
        """Test initialization without scipy."""
        with patch('farm.analysis.comparative.predictive_analytics.SCIPY_AVAILABLE', False):
            analytics = PredictiveAnalytics()
            assert analytics.scipy_available is False
    
    @pytest.mark.asyncio
    async def test_predict_performance_trends(self):
        """Test predicting performance trends."""
        predictions = await self.analytics.predict_performance_trends(
            self.mock_historical_data,
            prediction_horizon=30
        )
        
        assert len(predictions) > 0
        assert all(isinstance(pred, Prediction) for pred in predictions)
        assert all(pred.prediction_type == PredictionType.PERFORMANCE_TREND for pred in predictions)
        assert all(pred.confidence > 0.0 for pred in predictions)
    
    @pytest.mark.asyncio
    async def test_predict_performance_trends_without_sklearn(self):
        """Test predicting performance trends without sklearn."""
        with patch('farm.analysis.comparative.predictive_analytics.SKLEARN_AVAILABLE', False):
            analytics = PredictiveAnalytics()
            predictions = await analytics.predict_performance_trends(
                self.mock_historical_data,
                prediction_horizon=30
            )
            
            assert len(predictions) > 0
            assert all(isinstance(pred, Prediction) for pred in predictions)
    
    @pytest.mark.asyncio
    async def test_predict_anomalies(self):
        """Test predicting anomalies."""
        predictions = await self.analytics.predict_anomalies(
            self.mock_current_data,
            self.mock_historical_data
        )
        
        assert len(predictions) > 0
        assert all(isinstance(pred, Prediction) for pred in predictions)
        assert all(pred.prediction_type == PredictionType.ANOMALY for pred in predictions)
        assert all(pred.confidence > 0.0 for pred in predictions)
    
    @pytest.mark.asyncio
    async def test_predict_anomalies_without_sklearn(self):
        """Test predicting anomalies without sklearn."""
        with patch('farm.analysis.comparative.predictive_analytics.SKLEARN_AVAILABLE', False):
            analytics = PredictiveAnalytics()
            predictions = await analytics.predict_anomalies(
                self.mock_current_data,
                self.mock_historical_data
            )
            
            assert len(predictions) > 0
            assert all(isinstance(pred, Prediction) for pred in predictions)
    
    @pytest.mark.asyncio
    async def test_predict_resource_usage(self):
        """Test predicting resource usage."""
        predictions = await self.analytics.predict_resource_usage(
            self.mock_historical_data,
            prediction_horizon=7
        )
        
        assert len(predictions) > 0
        assert all(isinstance(pred, Prediction) for pred in predictions)
        assert all(pred.prediction_type == PredictionType.RESOURCE_USAGE for pred in predictions)
        assert all(pred.confidence > 0.0 for pred in predictions)
    
    @pytest.mark.asyncio
    async def test_predict_analysis_success(self):
        """Test predicting analysis success."""
        prediction = await self.analytics.predict_analysis_success(
            self.mock_analysis_config,
            self.mock_historical_data
        )
        
        assert isinstance(prediction, Prediction)
        assert prediction.prediction_type == PredictionType.ANALYSIS_SUCCESS
        assert prediction.confidence > 0.0
        assert prediction.predicted_value is not None
    
    @pytest.mark.asyncio
    async def test_predict_analysis_success_without_sklearn(self):
        """Test predicting analysis success without sklearn."""
        with patch('farm.analysis.comparative.predictive_analytics.SKLEARN_AVAILABLE', False):
            analytics = PredictiveAnalytics()
            prediction = await analytics.predict_analysis_success(
                self.mock_analysis_config,
                self.mock_historical_data
            )
            
            assert isinstance(prediction, Prediction)
            assert prediction.prediction_type == PredictionType.ANALYSIS_SUCCESS
    
    def test_prepare_time_series_data(self):
        """Test preparing time series data."""
        time_series = self.analytics._prepare_time_series_data(
            self.mock_historical_data,
            "cpu_usage"
        )
        
        assert isinstance(time_series, TimeSeriesData)
        assert len(time_series.timestamps) > 0
        assert len(time_series.values) > 0
        assert time_series.metric_name == "cpu_usage"
    
    def test_prepare_time_series_data_invalid_metric(self):
        """Test preparing time series data with invalid metric."""
        time_series = self.analytics._prepare_time_series_data(
            self.mock_historical_data,
            "invalid_metric"
        )
        
        assert isinstance(time_series, TimeSeriesData)
        assert len(time_series.timestamps) == 0
        assert len(time_series.values) == 0
    
    def test_predict_time_series_statistical(self):
        """Test predicting time series using statistical methods."""
        time_series = self.analytics._prepare_time_series_data(
            self.mock_historical_data,
            "cpu_usage"
        )
        
        prediction = self.analytics._predict_time_series_statistical(
            time_series,
            "cpu_usage",
            PredictionType.PERFORMANCE_TREND,
            horizon=30
        )
        
        assert isinstance(prediction, Prediction)
        assert prediction.prediction_type == PredictionType.PERFORMANCE_TREND
        assert prediction.confidence > 0.0
        assert prediction.predicted_value is not None
    
    def test_predict_time_series_ml(self):
        """Test predicting time series using ML methods."""
        time_series = self.analytics._prepare_time_series_data(
            self.mock_historical_data,
            "cpu_usage"
        )
        
        prediction = self.analytics._predict_time_series_ml(
            time_series,
            "cpu_usage",
            PredictionType.PERFORMANCE_TREND,
            horizon=30
        )
        
        if prediction is not None:
            assert isinstance(prediction, Prediction)
            assert prediction.prediction_type == PredictionType.PERFORMANCE_TREND
            assert prediction.confidence > 0.0
            assert prediction.predicted_value is not None
    
    def test_predict_time_series_ml_without_sklearn(self):
        """Test predicting time series using ML methods without sklearn."""
        with patch('farm.analysis.comparative.predictive_analytics.SKLEARN_AVAILABLE', False):
            analytics = PredictiveAnalytics()
            time_series = analytics._prepare_time_series_data(
                self.mock_historical_data,
                "cpu_usage"
            )
            
            prediction = analytics._predict_time_series_ml(
                time_series,
                "cpu_usage",
                PredictionType.PERFORMANCE_TREND,
                horizon=30
            )
            
            assert prediction is None
    
    def test_detect_anomalies_statistical(self):
        """Test detecting anomalies using statistical methods."""
        anomalies = self.analytics._detect_anomalies_statistical(
            self.mock_current_data,
            self.mock_historical_data
        )
        
        assert isinstance(anomalies, list)
        assert all(isinstance(anomaly, dict) for anomaly in anomalies)
    
    def test_detect_anomalies_ml(self):
        """Test detecting anomalies using ML methods."""
        anomalies = self.analytics._detect_anomalies_ml(
            self.mock_current_data,
            self.mock_historical_data
        )
        
        assert isinstance(anomalies, list)
        assert all(isinstance(anomaly, dict) for anomaly in anomalies)
    
    def test_detect_anomalies_ml_without_sklearn(self):
        """Test detecting anomalies using ML methods without sklearn."""
        with patch('farm.analysis.comparative.predictive_analytics.SKLEARN_AVAILABLE', False):
            analytics = PredictiveAnalytics()
            anomalies = analytics._detect_anomalies_ml(
                self.mock_current_data,
                self.mock_historical_data
            )
            
            assert isinstance(anomalies, list)
            assert len(anomalies) == 0  # Should return empty list without sklearn
    
    def test_analyze_trends(self):
        """Test analyzing trends in data."""
        trends = self.analytics._analyze_trends(self.mock_historical_data)
        
        assert isinstance(trends, list)
        assert all(isinstance(trend, dict) for trend in trends)
    
    def test_analyze_trends_without_scipy(self):
        """Test analyzing trends without scipy."""
        with patch('farm.analysis.comparative.predictive_analytics.SCIPY_AVAILABLE', False):
            analytics = PredictiveAnalytics()
            trends = analytics._analyze_trends(self.mock_historical_data)
            
            assert isinstance(trends, list)
            assert all(isinstance(trend, dict) for trend in trends)
    
    def test_calculate_prediction_confidence(self):
        """Test calculating prediction confidence."""
        # Test high confidence
        confidence = self.analytics._calculate_prediction_confidence(
            "performance_trend",
            {"r_squared": 0.9, "data_points": 100}
        )
        assert confidence > 0.8
        
        # Test low confidence
        confidence = self.analytics._calculate_prediction_confidence(
            "performance_trend",
            {"r_squared": 0.3, "data_points": 10}
        )
        assert confidence < 0.5
    
    def test_evaluate_model_performance(self):
        """Test evaluating model performance."""
        # Mock model performance data
        performance_data = {
            "r_squared": 0.85,
            "mae": 5.2,
            "rmse": 7.8,
            "data_points": 100
        }
        
        performance = self.analytics._evaluate_model_performance(performance_data)
        
        assert "accuracy" in performance
        assert "reliability" in performance
        assert "overall_score" in performance
        assert performance["accuracy"] > 0.0
        assert performance["reliability"] > 0.0
        assert performance["overall_score"] > 0.0
    
    def test_get_prediction_history(self):
        """Test getting prediction history."""
        # Add some predictions to history
        self.analytics.prediction_history = [
            {"timestamp": "2023-01-01T00:00:00", "prediction_count": 5},
            {"timestamp": "2023-01-02T00:00:00", "prediction_count": 3}
        ]
        
        history = self.analytics.get_prediction_history()
        assert len(history) == 2
        assert history[0]["prediction_count"] == 5
    
    def test_clear_prediction_history(self):
        """Test clearing prediction history."""
        # Add some predictions to history
        self.analytics.prediction_history = [
            {"timestamp": "2023-01-01T00:00:00", "prediction_count": 5}
        ]
        
        # Clear history
        self.analytics.clear_prediction_history()
        assert len(self.analytics.prediction_history) == 0
    
    def test_get_prediction_statistics(self):
        """Test getting prediction statistics."""
        # Add some predictions to history
        self.analytics.prediction_history = [
            {"timestamp": "2023-01-01T00:00:00", "prediction_count": 5, "types": ["performance_trend", "anomaly"]},
            {"timestamp": "2023-01-02T00:00:00", "prediction_count": 3, "types": ["resource_usage", "analysis_success"]}
        ]
        
        stats = self.analytics.get_prediction_statistics()
        
        assert "total_predictions" in stats
        assert "predictions_per_day" in stats
        assert "most_common_type" in stats
        assert stats["total_predictions"] == 8
    
    def test_get_model_performance(self):
        """Test getting model performance."""
        # Add some model performance data
        self.analytics.model_performance = {
            "linear_regression": {"accuracy": 0.85, "reliability": 0.80},
            "random_forest": {"accuracy": 0.90, "reliability": 0.85}
        }
        
        performance = self.analytics.get_model_performance()
        
        assert len(performance) == 2
        assert "linear_regression" in performance
        assert "random_forest" in performance
    
    def test_prediction_creation(self):
        """Test Prediction creation."""
        prediction = Prediction(
            prediction_type=PredictionType.PERFORMANCE_TREND,
            metric_name="cpu_usage",
            predicted_value=95.0,
            confidence=PredictionConfidence.HIGH,
            horizon=30,
            created_at=datetime.now(),
            metadata={"r_squared": 0.85, "data_points": 100}
        )
        
        assert prediction.prediction_type == PredictionType.PERFORMANCE_TREND
        assert prediction.metric_name == "cpu_usage"
        assert prediction.predicted_value == 95.0
        assert prediction.confidence == PredictionConfidence.HIGH
        assert prediction.horizon == 30
        assert prediction.metadata == {"r_squared": 0.85, "data_points": 100}
    
    def test_time_series_data_creation(self):
        """Test TimeSeriesData creation."""
        timestamps = [datetime.now() - timedelta(days=i) for i in range(5)]
        values = [70.0, 75.0, 80.0, 85.0, 90.0]
        
        time_series = TimeSeriesData(
            metric_name="cpu_usage",
            timestamps=timestamps,
            values=values,
            unit="percentage"
        )
        
        assert time_series.metric_name == "cpu_usage"
        assert time_series.timestamps == timestamps
        assert time_series.values == values
        assert time_series.unit == "percentage"
    
    def test_prediction_config_creation(self):
        """Test PredictionConfig creation."""
        config = PredictionConfig(
            enable_ml_predictions=False,
            enable_statistical_predictions=True,
            prediction_horizon=60,
            confidence_threshold=0.8,
            enable_anomaly_detection=False,
            enable_trend_analysis=True,
            enable_resource_forecasting=False,
            enable_analysis_success_prediction=True
        )
        
        assert config.enable_ml_predictions is False
        assert config.enable_statistical_predictions is True
        assert config.prediction_horizon == 60
        assert config.confidence_threshold == 0.8
        assert config.enable_anomaly_detection is False
        assert config.enable_trend_analysis is True
        assert config.enable_resource_forecasting is False
        assert config.enable_analysis_success_prediction is True
    
    def test_prediction_type_enum(self):
        """Test PredictionType enum values."""
        assert PredictionType.PERFORMANCE_TREND == "performance_trend"
        assert PredictionType.ANOMALY == "anomaly"
        assert PredictionType.RESOURCE_USAGE == "resource_usage"
        assert PredictionType.ANALYSIS_SUCCESS == "analysis_success"
    
    def test_prediction_confidence_enum(self):
        """Test PredictionConfidence enum values."""
        assert PredictionConfidence.LOW == "low"
        assert PredictionConfidence.MEDIUM == "medium"
        assert PredictionConfidence.HIGH == "high"
        assert PredictionConfidence.VERY_HIGH == "very_high"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in prediction operations."""
        # Test with empty historical data
        predictions = await self.analytics.predict_performance_trends([])
        
        assert len(predictions) == 0
        
        # Test with invalid current data
        predictions = await self.analytics.predict_anomalies({}, [])
        
        assert len(predictions) == 0
    
    def test_data_validation(self):
        """Test data validation."""
        # Test valid data
        valid_data = [{"timestamp": datetime.now(), "cpu_usage": 80.0}]
        assert self.analytics._validate_historical_data(valid_data) is True
        
        # Test invalid data (empty list)
        assert self.analytics._validate_historical_data([]) is False
        
        # Test invalid data (missing timestamp)
        invalid_data = [{"cpu_usage": 80.0}]
        assert self.analytics._validate_historical_data(invalid_data) is False
    
    def test_metric_extraction(self):
        """Test extracting metrics from data."""
        metrics = self.analytics._extract_metrics(self.mock_historical_data[0])
        
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "disk_io" in metrics
        assert "simulation_time" in metrics
        assert "success_rate" in metrics
    
    def test_trend_calculation(self):
        """Test calculating trends."""
        values = [70.0, 75.0, 80.0, 85.0, 90.0]
        trend = self.analytics._calculate_trend(values)
        
        assert trend > 0  # Should be positive trend
        assert isinstance(trend, float)
    
    def test_anomaly_score_calculation(self):
        """Test calculating anomaly scores."""
        current_value = 95.0
        historical_values = [70.0, 75.0, 80.0, 85.0, 90.0]
        
        score = self.analytics._calculate_anomaly_score(current_value, historical_values)
        
        assert score > 0
        assert isinstance(score, float)
        assert score <= 1.0  # Should be normalized