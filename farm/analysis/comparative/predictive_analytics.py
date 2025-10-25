"""
Predictive analytics and forecasting capabilities.

This module provides advanced predictive analytics for simulation analysis,
including forecasting, trend prediction, and anomaly prediction.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from enum import Enum
import statistics
import math

# Optional imports for advanced analytics
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.cluster import KMeans
    from sklearn.anomaly import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from farm.utils.logging import get_logger
from farm.analysis.comparative.integration_orchestrator import OrchestrationResult
from farm.analysis.comparative.comparison_result import SimulationComparisonResult

logger = get_logger(__name__)


class PredictionType(Enum):
    """Types of predictions."""
    PERFORMANCE_FORECAST = "performance_forecast"
    TREND_PREDICTION = "trend_prediction"
    ANOMALY_PREDICTION = "anomaly_prediction"
    RESOURCE_USAGE_FORECAST = "resource_usage_forecast"
    ERROR_PREDICTION = "error_prediction"
    QUALITY_PREDICTION = "quality_prediction"
    DURATION_PREDICTION = "duration_prediction"
    SUCCESS_PREDICTION = "success_prediction"


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Prediction:
    """A prediction result."""
    
    id: str
    type: PredictionType
    target: str
    predicted_value: Any
    confidence: float
    confidence_level: PredictionConfidence
    prediction_horizon: int  # days, hours, or steps
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    features_used: List[str] = field(default_factory=list)
    model_info: Dict[str, Any] = field(default_factory=dict)
    uncertainty_bounds: Optional[Tuple[float, float]] = None


@dataclass
class TimeSeriesData:
    """Time series data for prediction."""
    
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionConfig:
    """Configuration for predictive analytics."""
    
    # Model settings
    enable_ml_models: bool = True
    enable_statistical_models: bool = True
    enable_ensemble_methods: bool = True
    
    # Prediction horizons
    short_term_horizon: int = 7  # days
    medium_term_horizon: int = 30  # days
    long_term_horizon: int = 90  # days
    
    # Model parameters
    min_data_points: int = 10
    max_data_points: int = 1000
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6
    low_confidence_threshold: float = 0.4
    
    # Feature engineering
    enable_feature_engineering: bool = True
    max_features: int = 50
    feature_selection_threshold: float = 0.1


class PredictiveAnalytics:
    """Predictive analytics and forecasting system."""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """Initialize the predictive analytics system."""
        self.config = config or PredictionConfig()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, List[float]] = {}
        self.prediction_history: List[Prediction] = []
        
        # Initialize models
        self._initialize_models()
        
        logger.info("PredictiveAnalytics initialized")
    
    def _initialize_models(self):
        """Initialize prediction models."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available - using basic statistical models")
            return
        
        # Initialize regression models
        self.models["linear_regression"] = LinearRegression()
        self.models["ridge_regression"] = Ridge(alpha=1.0)
        self.models["lasso_regression"] = Lasso(alpha=0.1)
        self.models["random_forest"] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models["gradient_boosting"] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.models["svr"] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        # Initialize scalers
        self.scalers["standard"] = StandardScaler()
        self.scalers["minmax"] = MinMaxScaler()
        
        logger.info("Prediction models initialized")
    
    async def predict_performance_trends(self, 
                                       historical_data: List[Dict[str, Any]],
                                       prediction_horizon: int = 30) -> List[Prediction]:
        """Predict performance trends based on historical data."""
        logger.info("Predicting performance trends")
        
        predictions = []
        
        # Extract performance metrics
        performance_metrics = self._extract_performance_metrics(historical_data)
        
        for metric_name, time_series in performance_metrics.items():
            if len(time_series.values) < self.config.min_data_points:
                continue
            
            # Create prediction
            prediction = await self._predict_time_series(
                time_series,
                metric_name,
                PredictionType.PERFORMANCE_FORECAST,
                prediction_horizon
            )
            
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    async def predict_anomalies(self, 
                              current_data: Dict[str, Any],
                              historical_data: List[Dict[str, Any]]) -> List[Prediction]:
        """Predict potential anomalies in current data."""
        logger.info("Predicting anomalies")
        
        predictions = []
        
        if not SKLEARN_AVAILABLE:
            # Use basic statistical methods
            predictions = await self._predict_anomalies_statistical(current_data, historical_data)
        else:
            # Use ML-based anomaly detection
            predictions = await self._predict_anomalies_ml(current_data, historical_data)
        
        return predictions
    
    async def predict_resource_usage(self, 
                                   current_analysis: OrchestrationResult,
                                   historical_analyses: List[OrchestrationResult]) -> List[Prediction]:
        """Predict resource usage for future analyses."""
        logger.info("Predicting resource usage")
        
        predictions = []
        
        # Extract resource usage patterns
        resource_patterns = self._extract_resource_patterns(historical_analyses)
        
        # Predict CPU usage
        if "cpu_usage" in resource_patterns:
            prediction = await self._predict_resource_metric(
                resource_patterns["cpu_usage"],
                "CPU Usage",
                PredictionType.RESOURCE_USAGE_FORECAST
            )
            if prediction:
                predictions.append(prediction)
        
        # Predict memory usage
        if "memory_usage" in resource_patterns:
            prediction = await self._predict_resource_metric(
                resource_patterns["memory_usage"],
                "Memory Usage",
                PredictionType.RESOURCE_USAGE_FORECAST
            )
            if prediction:
                predictions.append(prediction)
        
        # Predict duration
        if "duration" in resource_patterns:
            prediction = await self._predict_resource_metric(
                resource_patterns["duration"],
                "Analysis Duration",
                PredictionType.DURATION_PREDICTION
            )
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    async def predict_analysis_success(self, 
                                     analysis_config: Dict[str, Any],
                                     historical_data: List[Dict[str, Any]]) -> Prediction:
        """Predict the success probability of an analysis."""
        logger.info("Predicting analysis success")
        
        # Extract features from analysis config
        features = self._extract_analysis_features(analysis_config)
        
        # Get historical success patterns
        success_patterns = self._extract_success_patterns(historical_data)
        
        # Predict success probability
        success_probability = await self._predict_success_probability(features, success_patterns)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(success_probability)
        
        prediction = Prediction(
            id=f"success_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=PredictionType.SUCCESS_PREDICTION,
            target="Analysis Success",
            predicted_value=success_probability,
            confidence=success_probability,
            confidence_level=confidence_level,
            prediction_horizon=1,  # Immediate prediction
            features_used=list(features.keys()),
            metadata={"analysis_config": analysis_config}
        )
        
        self.prediction_history.append(prediction)
        return prediction
    
    def _extract_performance_metrics(self, historical_data: List[Dict[str, Any]]) -> Dict[str, TimeSeriesData]:
        """Extract performance metrics from historical data."""
        metrics = {}
        
        # Extract duration trends
        durations = []
        timestamps = []
        
        for data_point in historical_data:
            if "timestamp" in data_point and "duration" in data_point:
                timestamps.append(datetime.fromisoformat(data_point["timestamp"]))
                durations.append(float(data_point["duration"]))
        
        if durations:
            metrics["duration"] = TimeSeriesData(
                timestamps=timestamps,
                values=durations,
                metadata={"unit": "seconds"}
            )
        
        # Extract error rates
        error_rates = []
        error_timestamps = []
        
        for data_point in historical_data:
            if "timestamp" in data_point and "error_count" in data_point and "total_phases" in data_point:
                error_timestamps.append(datetime.fromisoformat(data_point["timestamp"]))
                error_rate = float(data_point["error_count"]) / float(data_point["total_phases"])
                error_rates.append(error_rate)
        
        if error_rates:
            metrics["error_rate"] = TimeSeriesData(
                timestamps=error_timestamps,
                values=error_rates,
                metadata={"unit": "ratio"}
            )
        
        return metrics
    
    def _extract_resource_patterns(self, historical_analyses: List[OrchestrationResult]) -> Dict[str, List[float]]:
        """Extract resource usage patterns from historical analyses."""
        patterns = {
            "cpu_usage": [],
            "memory_usage": [],
            "duration": []
        }
        
        for analysis in historical_analyses:
            # Extract duration
            patterns["duration"].append(analysis.total_duration)
            
            # Extract resource usage from metadata if available
            if hasattr(analysis, 'metadata') and analysis.metadata:
                if "cpu_usage" in analysis.metadata:
                    patterns["cpu_usage"].append(analysis.metadata["cpu_usage"])
                if "memory_usage" in analysis.metadata:
                    patterns["memory_usage"].append(analysis.metadata["memory_usage"])
        
        return patterns
    
    def _extract_analysis_features(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from analysis configuration."""
        features = {}
        
        # Basic features
        features["phase_count"] = len(analysis_config.get("phases", []))
        features["parallel_enabled"] = analysis_config.get("parallel", False)
        features["max_workers"] = analysis_config.get("max_workers", 1)
        
        # Complexity features
        if "phases" in analysis_config:
            phase_types = [phase.get("type", "unknown") for phase in analysis_config["phases"]]
            features["ml_phases"] = sum(1 for ptype in phase_types if "ml" in ptype.lower())
            features["statistical_phases"] = sum(1 for ptype in phase_types if "statistical" in ptype.lower())
        
        # Resource features
        features["memory_limit"] = analysis_config.get("memory_limit", 0)
        features["timeout"] = analysis_config.get("timeout", 0)
        
        return features
    
    def _extract_success_patterns(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract success patterns from historical data."""
        patterns = {
            "success_rates": [],
            "feature_success": {},
            "total_analyses": len(historical_data)
        }
        
        for data_point in historical_data:
            success = data_point.get("success", False)
            patterns["success_rates"].append(1.0 if success else 0.0)
            
            # Extract feature-success relationships
            for feature, value in data_point.items():
                if feature not in ["success", "timestamp"]:
                    if feature not in patterns["feature_success"]:
                        patterns["feature_success"][feature] = {"success": [], "failure": []}
                    
                    if success:
                        patterns["feature_success"][feature]["success"].append(value)
                    else:
                        patterns["feature_success"][feature]["failure"].append(value)
        
        return patterns
    
    async def _predict_time_series(self, 
                                 time_series: TimeSeriesData,
                                 metric_name: str,
                                 prediction_type: PredictionType,
                                 horizon: int) -> Optional[Prediction]:
        """Predict future values for a time series."""
        if len(time_series.values) < self.config.min_data_points:
            return None
        
        try:
            if SKLEARN_AVAILABLE and self.config.enable_ml_models:
                return await self._predict_time_series_ml(time_series, metric_name, prediction_type, horizon)
            else:
                return await self._predict_time_series_statistical(time_series, metric_name, prediction_type, horizon)
        except Exception as e:
            logger.error(f"Error predicting time series for {metric_name}: {e}")
            return None
    
    async def _predict_time_series_ml(self, 
                                    time_series: TimeSeriesData,
                                    metric_name: str,
                                    prediction_type: PredictionType,
                                    horizon: int) -> Optional[Prediction]:
        """Predict time series using ML models."""
        # Prepare data
        X, y = self._prepare_time_series_data(time_series)
        
        if len(X) < 5:  # Need minimum data for ML
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-self.config.train_test_split, random_state=42
        )
        
        # Scale features
        scaler = self.scalers["standard"]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try different models
        best_model = None
        best_score = float('-inf')
        best_predictions = None
        
        for model_name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate score
                score = r2_score(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
                    best_predictions = y_pred
                
            except Exception as e:
                logger.warning(f"Error training model {model_name}: {e}")
                continue
        
        if best_model is None:
            return None
        
        # Make future predictions
        last_features = X[-1:].reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        
        future_predictions = []
        current_features = last_features_scaled.copy()
        
        for _ in range(horizon):
            pred = self.models[best_model].predict(current_features)[0]
            future_predictions.append(pred)
            
            # Update features for next prediction (simple approach)
            current_features = np.roll(current_features, -1)
            current_features[0, -1] = pred
        
        # Calculate confidence
        confidence = max(0.0, min(1.0, best_score))
        confidence_level = self._determine_confidence_level(confidence)
        
        # Calculate uncertainty bounds
        if len(future_predictions) > 0:
            mean_pred = np.mean(future_predictions)
            std_pred = np.std(future_predictions)
            uncertainty_bounds = (mean_pred - 2*std_pred, mean_pred + 2*std_pred)
        else:
            uncertainty_bounds = None
        
        return Prediction(
            id=f"pred_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=prediction_type,
            target=metric_name,
            predicted_value=future_predictions[-1] if future_predictions else 0.0,
            confidence=confidence,
            confidence_level=confidence_level,
            prediction_horizon=horizon,
            features_used=[f"lag_{i}" for i in range(X.shape[1])],
            model_info={"model": best_model, "r2_score": best_score},
            uncertainty_bounds=uncertainty_bounds
        )
    
    async def _predict_time_series_statistical(self, 
                                             time_series: TimeSeriesData,
                                             metric_name: str,
                                             prediction_type: PredictionType,
                                             horizon: int) -> Optional[Prediction]:
        """Predict time series using statistical methods."""
        values = time_series.values
        
        if len(values) < 3:
            return None
        
        # Simple linear trend
        x = list(range(len(values)))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Predict future values
        future_x = list(range(len(values), len(values) + horizon))
        future_predictions = [slope * x_val + intercept for x_val in future_x]
        
        # Calculate confidence based on R-squared
        confidence = max(0.0, min(1.0, r_value ** 2))
        confidence_level = self._determine_confidence_level(confidence)
        
        return Prediction(
            id=f"pred_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=prediction_type,
            target=metric_name,
            predicted_value=future_predictions[-1] if future_predictions else 0.0,
            confidence=confidence,
            confidence_level=confidence_level,
            prediction_horizon=horizon,
            features_used=["linear_trend"],
            model_info={"method": "linear_regression", "r_squared": r_value ** 2, "p_value": p_value},
            uncertainty_bounds=(future_predictions[-1] - std_err, future_predictions[-1] + std_err) if future_predictions else None
        )
    
    def _prepare_time_series_data(self, time_series: TimeSeriesData, lag: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for ML models."""
        values = np.array(time_series.values)
        
        X = []
        y = []
        
        for i in range(lag, len(values)):
            X.append(values[i-lag:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    async def _predict_anomalies_ml(self, 
                                  current_data: Dict[str, Any],
                                  historical_data: List[Dict[str, Any]]) -> List[Prediction]:
        """Predict anomalies using ML methods."""
        predictions = []
        
        # Prepare historical data for training
        historical_features = []
        for data_point in historical_data:
            features = self._extract_anomaly_features(data_point)
            historical_features.append(features)
        
        if len(historical_features) < 10:
            return predictions
        
        # Train isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(historical_features)
        
        # Predict anomaly for current data
        current_features = self._extract_anomaly_features(current_data)
        anomaly_score = iso_forest.decision_function([current_features])[0]
        is_anomaly = iso_forest.predict([current_features])[0] == -1
        
        # Calculate confidence
        confidence = abs(anomaly_score)
        confidence_level = self._determine_confidence_level(confidence)
        
        prediction = Prediction(
            id=f"anomaly_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=PredictionType.ANOMALY_PREDICTION,
            target="Anomaly Detection",
            predicted_value=is_anomaly,
            confidence=confidence,
            confidence_level=confidence_level,
            prediction_horizon=1,
            features_used=list(current_features.keys()),
            model_info={"model": "isolation_forest", "anomaly_score": anomaly_score}
        )
        
        predictions.append(prediction)
        return predictions
    
    async def _predict_anomalies_statistical(self, 
                                           current_data: Dict[str, Any],
                                           historical_data: List[Dict[str, Any]]) -> List[Prediction]:
        """Predict anomalies using statistical methods."""
        predictions = []
        
        # Extract numerical features
        numerical_features = {}
        for key, value in current_data.items():
            if isinstance(value, (int, float)):
                numerical_features[key] = value
        
        for feature_name, current_value in numerical_features.items():
            # Get historical values for this feature
            historical_values = []
            for data_point in historical_data:
                if feature_name in data_point and isinstance(data_point[feature_name], (int, float)):
                    historical_values.append(data_point[feature_name])
            
            if len(historical_values) < 5:
                continue
            
            # Calculate statistical measures
            mean_val = statistics.mean(historical_values)
            std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
            
            # Check if current value is an outlier (beyond 2 standard deviations)
            z_score = abs((current_value - mean_val) / std_val) if std_val > 0 else 0
            is_anomaly = z_score > 2
            
            # Calculate confidence
            confidence = min(1.0, z_score / 3.0)  # Normalize to 0-1
            confidence_level = self._determine_confidence_level(confidence)
            
            prediction = Prediction(
                id=f"anomaly_pred_{feature_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=PredictionType.ANOMALY_PREDICTION,
                target=f"Anomaly in {feature_name}",
                predicted_value=is_anomaly,
                confidence=confidence,
                confidence_level=confidence_level,
                prediction_horizon=1,
                features_used=[feature_name],
                model_info={"method": "z_score", "z_score": z_score, "mean": mean_val, "std": std_val}
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _extract_anomaly_features(self, data_point: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for anomaly detection."""
        features = {}
        
        # Extract numerical features
        for key, value in data_point.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                features[f"{key}_mean"] = float(statistics.mean(value))
                features[f"{key}_std"] = float(statistics.stdev(value)) if len(value) > 1 else 0.0
        
        return features
    
    async def _predict_resource_metric(self, 
                                     historical_values: List[float],
                                     metric_name: str,
                                     prediction_type: PredictionType) -> Optional[Prediction]:
        """Predict a resource metric based on historical values."""
        if len(historical_values) < self.config.min_data_points:
            return None
        
        # Create time series data
        time_series = TimeSeriesData(
            timestamps=[datetime.now() - timedelta(days=i) for i in range(len(historical_values))],
            values=historical_values,
            metadata={"unit": "percentage" if "usage" in metric_name.lower() else "seconds"}
        )
        
        # Predict future values
        prediction = await self._predict_time_series(
            time_series,
            metric_name,
            prediction_type,
            horizon=7  # Predict next 7 days
        )
        
        return prediction
    
    async def _predict_success_probability(self, 
                                        features: Dict[str, Any],
                                        success_patterns: Dict[str, Any]) -> float:
        """Predict success probability based on features and patterns."""
        if not success_patterns["feature_success"]:
            return 0.5  # Default probability
        
        # Simple feature-based success prediction
        success_indicators = 0
        total_indicators = 0
        
        for feature_name, feature_value in features.items():
            if feature_name in success_patterns["feature_success"]:
                pattern = success_patterns["feature_success"][feature_name]
                
                if pattern["success"] and pattern["failure"]:
                    # Calculate success rate for this feature value
                    success_count = sum(1 for val in pattern["success"] if val == feature_value)
                    failure_count = sum(1 for val in pattern["failure"] if val == feature_value)
                    
                    if success_count + failure_count > 0:
                        success_rate = success_count / (success_count + failure_count)
                        success_indicators += success_rate
                        total_indicators += 1
        
        if total_indicators > 0:
            return success_indicators / total_indicators
        
        # Fallback to overall success rate
        if success_patterns["success_rates"]:
            return statistics.mean(success_patterns["success_rates"])
        
        return 0.5
    
    def _determine_confidence_level(self, confidence: float) -> PredictionConfidence:
        """Determine confidence level based on confidence score."""
        if confidence >= self.config.high_confidence_threshold:
            return PredictionConfidence.HIGH
        elif confidence >= self.config.medium_confidence_threshold:
            return PredictionConfidence.MEDIUM
        elif confidence >= self.config.low_confidence_threshold:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.LOW
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of prediction system."""
        if not self.prediction_history:
            return {"message": "No predictions available"}
        
        # Analyze prediction types
        type_counts = {}
        confidence_levels = {}
        
        for prediction in self.prediction_history:
            type_counts[prediction.type.value] = type_counts.get(prediction.type.value, 0) + 1
            confidence_levels[prediction.confidence_level.value] = confidence_levels.get(prediction.confidence_level.value, 0) + 1
        
        # Calculate average confidence
        avg_confidence = statistics.mean([p.confidence for p in self.prediction_history])
        
        return {
            "total_predictions": len(self.prediction_history),
            "prediction_types": type_counts,
            "confidence_levels": confidence_levels,
            "average_confidence": avg_confidence,
            "ml_models_available": SKLEARN_AVAILABLE,
            "statistical_models_available": SCIPY_AVAILABLE
        }
    
    def export_predictions(self, format: str = "json", file_path: Optional[Union[str, Path]] = None) -> Union[str, Path]:
        """Export predictions to various formats."""
        if not self.prediction_history:
            return "No predictions to export"
        
        if format == "json":
            data = [prediction.__dict__ for prediction in self.prediction_history]
            json_str = json.dumps(data, indent=2, default=str)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(json_str)
                return Path(file_path)
            else:
                return json_str
        
        elif format == "markdown":
            md_content = "# Prediction Export\n\n"
            md_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_content += f"Total predictions: {len(self.prediction_history)}\n\n"
            
            # Group by prediction type
            type_groups = {}
            for prediction in self.prediction_history:
                pred_type = prediction.type.value
                if pred_type not in type_groups:
                    type_groups[pred_type] = []
                type_groups[pred_type].append(prediction)
            
            for pred_type, predictions in type_groups.items():
                md_content += f"## {pred_type.replace('_', ' ').title()} ({len(predictions)} predictions)\n\n"
                
                for prediction in predictions:
                    md_content += f"### {prediction.target}\n"
                    md_content += f"**Predicted Value**: {prediction.predicted_value}\n"
                    md_content += f"**Confidence**: {prediction.confidence:.2f} ({prediction.confidence_level.value})\n"
                    md_content += f"**Horizon**: {prediction.prediction_horizon} steps\n"
                    md_content += f"**Created**: {prediction.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    
                    if prediction.uncertainty_bounds:
                        md_content += f"**Uncertainty Bounds**: {prediction.uncertainty_bounds[0]:.2f} - {prediction.uncertainty_bounds[1]:.2f}\n\n"
                    
                    md_content += "---\n\n"
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(md_content)
                return Path(file_path)
            else:
                return md_content
        
        else:
            raise ValueError(f"Unsupported format: {format}")