"""
Trend prediction and forecasting module for simulation comparison.

This module provides advanced forecasting capabilities to predict future
trends and patterns in simulation behavior.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from farm.analysis.comparative.comparison_result import SimulationComparisonResult
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import ML libraries
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Trend prediction will be limited.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Gradient boosting features will be limited.")

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Some signal processing features will be limited.")


@dataclass
class TrendPredictionConfig:
    """Configuration for trend prediction."""
    forecast_horizon: int = 10
    lookback_window: int = 5
    trend_detection_threshold: float = 0.1
    seasonality_period: int = 7
    confidence_level: float = 0.95
    model_selection_method: str = 'auto'  # 'auto', 'linear', 'tree', 'ensemble'
    cross_validation_folds: int = 5
    random_state: int = 42


@dataclass
class TrendPredictionResult:
    """Result of trend prediction analysis."""
    predictions: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    forecast_accuracy: Dict[str, Any]
    model_performance: Dict[str, Any]
    confidence_intervals: Dict[str, Any]
    recommendations: List[str]
    summary: Dict[str, Any]


@dataclass
class TrendAnalysis:
    """Individual trend analysis result."""
    feature_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'cyclical'
    trend_strength: float  # 0-1
    trend_confidence: float  # 0-1
    seasonality_detected: bool
    seasonality_strength: float
    change_points: List[int]
    trend_description: str


class TrendPredictor:
    """Advanced trend predictor for simulation comparison data."""
    
    def __init__(self, config: Optional[TrendPredictionConfig] = None):
        """Initialize trend predictor.
        
        Args:
            config: Configuration for trend prediction
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for trend prediction")
        
        self.config = config or TrendPredictionConfig()
        self._initialize_models()
        
        logger.info("TrendPredictor initialized")
    
    def _initialize_models(self):
        """Initialize prediction models."""
        self.regression_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.config.random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.config.random_state)
        }
        
        if XGBOOST_AVAILABLE:
            self.regression_models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.config.random_state
            )
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
    
    def predict_trends(self, 
                      results: List[SimulationComparisonResult],
                      features: Optional[List[str]] = None,
                      time_series: bool = True) -> TrendPredictionResult:
        """Predict trends in simulation comparison data.
        
        Args:
            results: List of simulation comparison results
            features: List of features to predict
            time_series: Whether to treat data as time series
            
        Returns:
            Trend prediction result
        """
        logger.info("Predicting trends in simulation data")
        
        if len(results) < 3:
            logger.warning("Insufficient data for trend prediction")
            return self._create_empty_result()
        
        # Extract time series data
        time_series_data = self._extract_time_series_data(results, features)
        
        if time_series_data.empty:
            logger.warning("No time series data extracted")
            return self._create_empty_result()
        
        # Analyze trends
        trend_analysis = self._analyze_trends(time_series_data)
        
        # Make predictions
        predictions = self._make_predictions(time_series_data, trend_analysis)
        
        # Calculate forecast accuracy
        forecast_accuracy = self._calculate_forecast_accuracy(time_series_data, predictions)
        
        # Evaluate model performance
        model_performance = self._evaluate_model_performance(time_series_data, predictions)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(predictions)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trend_analysis, predictions)
        
        # Generate summary
        summary = self._generate_prediction_summary(
            trend_analysis, predictions, forecast_accuracy, model_performance
        )
        
        return TrendPredictionResult(
            predictions=predictions,
            trend_analysis=trend_analysis,
            forecast_accuracy=forecast_accuracy,
            model_performance=model_performance,
            confidence_intervals=confidence_intervals,
            recommendations=recommendations,
            summary=summary
        )
    
    def _extract_time_series_data(self, 
                                 results: List[SimulationComparisonResult],
                                 features: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract time series data from simulation results."""
        time_series_data = []
        
        for i, result in enumerate(results):
            # Use comparison time as timestamp, or create sequential index
            if hasattr(result.comparison_summary, 'comparison_time'):
                timestamp = result.comparison_summary.comparison_time
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except:
                        timestamp = datetime.now() - timedelta(days=len(results)-i)
            else:
                timestamp = datetime.now() - timedelta(days=len(results)-i)
            
            row = {
                'timestamp': timestamp,
                'simulation_id': i,
                'total_differences': result.comparison_summary.total_differences,
                'config_differences': result.comparison_summary.config_differences,
                'database_differences': result.comparison_summary.database_differences,
                'log_differences': result.comparison_summary.log_differences,
                'metrics_differences': result.comparison_summary.metrics_differences,
                'severity_numeric': self._severity_to_numeric(result.comparison_summary.severity)
            }
            
            # Extract metrics features
            if result.metrics_comparison.metric_differences:
                for metric, diff in result.metrics_comparison.metric_differences.items():
                    if isinstance(diff, dict):
                        row[f'metric_{metric}_change'] = diff.get('percentage_change', 0)
                        row[f'metric_{metric}_abs_change'] = abs(diff.get('percentage_change', 0))
            
            # Extract performance features
            if result.metrics_comparison.performance_comparison:
                for metric, comp in result.metrics_comparison.performance_comparison.items():
                    if isinstance(comp, dict):
                        ratio = comp.get('ratio', 1.0)
                        row[f'perf_{metric}_ratio'] = ratio
                        row[f'perf_{metric}_improvement'] = max(0, ratio - 1.0)
            
            # Extract error features
            if result.log_comparison.error_differences:
                total_errors = 0
                for error_type, diff in result.log_comparison.error_differences.items():
                    if isinstance(diff, dict):
                        total_errors += abs(diff.get('difference', 0))
                row['total_error_changes'] = total_errors
            
            time_series_data.append(row)
        
        df = pd.DataFrame(time_series_data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Select specific features if requested
        if features:
            available_features = [f for f in features if f in df.columns]
            if available_features:
                df = df[['timestamp', 'simulation_id'] + available_features]
        
        return df
    
    def _severity_to_numeric(self, severity: str) -> int:
        """Convert severity string to numeric value."""
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return severity_map.get(severity.lower(), 0)
    
    def _analyze_trends(self, time_series_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        logger.debug("Analyzing trends in time series data")
        
        trend_analyses = {}
        numeric_columns = time_series_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['simulation_id']:
                continue
            
            values = time_series_data[column].values
            if len(values) < 3:
                continue
            
            # Detect trend direction and strength
            trend_direction, trend_strength = self._detect_trend(values)
            
            # Detect seasonality
            seasonality_detected, seasonality_strength = self._detect_seasonality(values)
            
            # Detect change points
            change_points = self._detect_change_points(values)
            
            # Calculate trend confidence
            trend_confidence = self._calculate_trend_confidence(values, trend_direction)
            
            # Generate trend description
            trend_description = self._generate_trend_description(
                column, trend_direction, trend_strength, seasonality_detected
            )
            
            trend_analyses[column] = {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'trend_confidence': trend_confidence,
                'seasonality_detected': seasonality_detected,
                'seasonality_strength': seasonality_strength,
                'change_points': change_points,
                'trend_description': trend_description
            }
        
        return trend_analyses
    
    def _detect_trend(self, values: np.ndarray) -> Tuple[str, float]:
        """Detect trend direction and strength."""
        if len(values) < 2:
            return 'stable', 0.0
        
        # Calculate linear trend using least squares
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if abs(slope) < self.config.trend_detection_threshold:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Calculate trend strength (0-1)
        strength = min(1.0, abs(r_value))
        
        return direction, strength
    
    def _detect_seasonality(self, values: np.ndarray) -> Tuple[bool, float]:
        """Detect seasonality in time series."""
        if len(values) < 2 * self.config.seasonality_period:
            return False, 0.0
        
        try:
            # Calculate autocorrelation for seasonality period
            autocorr = np.corrcoef(values[:-self.config.seasonality_period], 
                                 values[self.config.seasonality_period:])[0, 1]
            
            # Check if autocorrelation is significant
            seasonality_detected = autocorr > 0.3
            seasonality_strength = max(0.0, autocorr)
            
            return seasonality_detected, seasonality_strength
            
        except:
            return False, 0.0
    
    def _detect_change_points(self, values: np.ndarray) -> List[int]:
        """Detect change points in time series."""
        if len(values) < 4:
            return []
        
        change_points = []
        
        # Simple change point detection using rolling statistics
        window_size = max(2, len(values) // 4)
        
        for i in range(window_size, len(values) - window_size):
            before_mean = np.mean(values[i-window_size:i])
            after_mean = np.mean(values[i:i+window_size])
            
            # Check if there's a significant change
            if abs(after_mean - before_mean) > 2 * np.std(values):
                change_points.append(i)
        
        return change_points
    
    def _calculate_trend_confidence(self, values: np.ndarray, trend_direction: str) -> float:
        """Calculate confidence in trend detection."""
        if len(values) < 3:
            return 0.0
        
        # Calculate R-squared for linear trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Confidence based on R-squared and p-value
        r_squared = r_value ** 2
        p_confidence = 1.0 - p_value if p_value < 1.0 else 0.0
        
        # Combine R-squared and p-value confidence
        confidence = (r_squared + p_confidence) / 2.0
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_trend_description(self, column: str, direction: str, 
                                  strength: float, seasonality: bool) -> str:
        """Generate human-readable trend description."""
        strength_desc = "strong" if strength > 0.7 else "moderate" if strength > 0.4 else "weak"
        
        if direction == 'increasing':
            desc = f"{column} shows a {strength_desc} increasing trend"
        elif direction == 'decreasing':
            desc = f"{column} shows a {strength_desc} decreasing trend"
        else:
            desc = f"{column} shows a stable trend"
        
        if seasonality:
            desc += " with seasonal patterns"
        
        return desc
    
    def _make_predictions(self, time_series_data: pd.DataFrame, 
                         trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions for future values."""
        logger.debug("Making predictions for future values")
        
        predictions = {}
        numeric_columns = time_series_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['simulation_id'] or column not in trend_analysis:
                continue
            
            values = time_series_data[column].values
            if len(values) < 3:
                continue
            
            # Prepare features for prediction
            X, y = self._prepare_prediction_features(values)
            
            if len(X) < 2:
                continue
            
            # Select best model for this feature
            best_model = self._select_best_model(X, y, column)
            
            if best_model is None:
                continue
            
            # Train model
            best_model.fit(X, y)
            
            # Make predictions
            future_predictions = self._predict_future_values(best_model, values, X)
            
            # Calculate prediction intervals
            prediction_intervals = self._calculate_prediction_intervals(
                best_model, X, y, future_predictions
            )
            
            predictions[column] = {
                'future_values': future_predictions,
                'prediction_intervals': prediction_intervals,
                'model_name': best_model.__class__.__name__,
                'trend_info': trend_analysis[column]
            }
        
        return predictions
    
    def _prepare_prediction_features(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for time series prediction."""
        lookback = self.config.lookback_window
        
        if len(values) <= lookback:
            return np.array([]), np.array([])
        
        X = []
        y = []
        
        for i in range(lookback, len(values)):
            X.append(values[i-lookback:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def _select_best_model(self, X: np.ndarray, y: np.ndarray, column: str) -> Optional[Any]:
        """Select the best model for prediction."""
        if len(X) < 2:
            return None
        
        best_model = None
        best_score = -np.inf
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(self.config.cross_validation_folds, len(X)-1))
        
        for model_name, model in self.regression_models.items():
            try:
                # Scale features
                scaler = self.scalers['standard']
                X_scaled = scaler.fit_transform(X)
                
                # Cross-validation
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
            except Exception as e:
                logger.warning(f"Error with model {model_name} for {column}: {e}")
                continue
        
        return best_model
    
    def _predict_future_values(self, model: Any, values: np.ndarray, 
                             X: np.ndarray) -> List[float]:
        """Predict future values using trained model."""
        lookback = self.config.lookback_window
        horizon = self.config.forecast_horizon
        
        # Scale features
        scaler = self.scalers['standard']
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, values[lookback:])
        
        # Make predictions
        future_values = []
        current_sequence = values[-lookback:].copy()
        
        for _ in range(horizon):
            # Scale current sequence
            current_scaled = scaler.transform(current_sequence.reshape(1, -1))
            
            # Predict next value
            next_value = model.predict(current_scaled)[0]
            future_values.append(float(next_value))
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_value
        
        return future_values
    
    def _calculate_prediction_intervals(self, model: Any, X: np.ndarray, y: np.ndarray,
                                      predictions: List[float]) -> Dict[str, List[float]]:
        """Calculate prediction intervals for forecasts."""
        # Calculate residuals
        scaler = self.scalers['standard']
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        residuals = y - y_pred
        
        # Calculate standard error
        std_error = np.std(residuals)
        
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        margin_error = z_score * std_error
        
        lower_bounds = [pred - margin_error for pred in predictions]
        upper_bounds = [pred + margin_error for pred in predictions]
        
        return {
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'confidence_level': self.config.confidence_level
        }
    
    def _calculate_forecast_accuracy(self, time_series_data: pd.DataFrame, 
                                   predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate forecast accuracy metrics."""
        accuracy_metrics = {}
        
        for column, pred_data in predictions.items():
            if column not in time_series_data.columns:
                continue
            
            actual_values = time_series_data[column].values
            if len(actual_values) < 2:
                continue
            
            # Use last few values for accuracy calculation
            n_test = min(3, len(actual_values) // 2)
            actual_test = actual_values[-n_test:]
            
            # Simple accuracy: compare with trend
            if len(actual_test) > 1:
                actual_trend = np.polyfit(range(len(actual_test)), actual_test, 1)[0]
                predicted_trend = np.polyfit(range(len(pred_data['future_values'][:n_test])), 
                                          pred_data['future_values'][:n_test], 1)[0]
                
                trend_accuracy = 1.0 - abs(actual_trend - predicted_trend) / (abs(actual_trend) + 1e-8)
                trend_accuracy = max(0.0, min(1.0, trend_accuracy))
            else:
                trend_accuracy = 0.5
            
            accuracy_metrics[column] = {
                'trend_accuracy': trend_accuracy,
                'prediction_horizon': len(pred_data['future_values']),
                'confidence_level': pred_data['prediction_intervals']['confidence_level']
            }
        
        return accuracy_metrics
    
    def _evaluate_model_performance(self, time_series_data: pd.DataFrame,
                                  predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance."""
        performance_metrics = {}
        
        for column, pred_data in predictions.items():
            if column not in time_series_data.columns:
                continue
            
            values = time_series_data[column].values
            if len(values) < 3:
                continue
            
            # Calculate basic performance metrics
            model_name = pred_data['model_name']
            
            performance_metrics[column] = {
                'model_name': model_name,
                'prediction_horizon': len(pred_data['future_values']),
                'trend_direction': pred_data['trend_info']['trend_direction'],
                'trend_strength': pred_data['trend_info']['trend_strength'],
                'trend_confidence': pred_data['trend_info']['trend_confidence']
            }
        
        return performance_metrics
    
    def _calculate_confidence_intervals(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall confidence intervals."""
        confidence_summary = {}
        
        for column, pred_data in predictions.items():
            intervals = pred_data['prediction_intervals']
            confidence_summary[column] = {
                'confidence_level': intervals['confidence_level'],
                'average_interval_width': np.mean([
                    upper - lower for upper, lower in zip(intervals['upper_bound'], intervals['lower_bound'])
                ]),
                'max_interval_width': max([
                    upper - lower for upper, lower in zip(intervals['upper_bound'], intervals['lower_bound'])
                ])
            }
        
        return confidence_summary
    
    def _generate_recommendations(self, trend_analysis: Dict[str, Any],
                                predictions: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trend analysis and predictions."""
        recommendations = []
        
        # Analyze overall trends
        increasing_trends = []
        decreasing_trends = []
        high_confidence_trends = []
        
        for column, analysis in trend_analysis.items():
            if analysis['trend_direction'] == 'increasing' and analysis['trend_strength'] > 0.5:
                increasing_trends.append(column)
            elif analysis['trend_direction'] == 'decreasing' and analysis['trend_strength'] > 0.5:
                decreasing_trends.append(column)
            
            if analysis['trend_confidence'] > 0.7:
                high_confidence_trends.append(column)
        
        # Generate recommendations
        if increasing_trends:
            recommendations.append(f"Monitor increasing trends in: {', '.join(increasing_trends)}")
        
        if decreasing_trends:
            recommendations.append(f"Investigate decreasing trends in: {', '.join(decreasing_trends)}")
        
        if high_confidence_trends:
            recommendations.append(f"High confidence trends detected in: {', '.join(high_confidence_trends)}")
        
        # Check for concerning predictions
        for column, pred_data in predictions.items():
            future_values = pred_data['future_values']
            if future_values and any(v > np.mean(future_values) * 2 for v in future_values):
                recommendations.append(f"Potential outlier predicted for {column}")
        
        if not recommendations:
            recommendations.append("No significant trends or concerns detected")
        
        return recommendations
    
    def _generate_prediction_summary(self, trend_analysis: Dict[str, Any],
                                   predictions: Dict[str, Any],
                                   forecast_accuracy: Dict[str, Any],
                                   model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of prediction results."""
        total_features = len(trend_analysis)
        strong_trends = sum(1 for analysis in trend_analysis.values() 
                          if analysis['trend_strength'] > 0.7)
        high_confidence_trends = sum(1 for analysis in trend_analysis.values() 
                                   if analysis['trend_confidence'] > 0.7)
        seasonal_patterns = sum(1 for analysis in trend_analysis.values() 
                              if analysis['seasonality_detected'])
        
        avg_accuracy = np.mean([metrics['trend_accuracy'] for metrics in forecast_accuracy.values()]) if forecast_accuracy else 0.0
        
        return {
            'total_features_analyzed': total_features,
            'strong_trends_detected': strong_trends,
            'high_confidence_trends': high_confidence_trends,
            'seasonal_patterns_detected': seasonal_patterns,
            'average_forecast_accuracy': avg_accuracy,
            'prediction_horizon': self.config.forecast_horizon,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _create_empty_result(self) -> TrendPredictionResult:
        """Create empty prediction result."""
        return TrendPredictionResult(
            predictions={},
            trend_analysis={},
            forecast_accuracy={},
            model_performance={},
            confidence_intervals={},
            recommendations=["Insufficient data for trend prediction"],
            summary={
                'total_features_analyzed': 0,
                'strong_trends_detected': 0,
                'high_confidence_trends': 0,
                'seasonal_patterns_detected': 0,
                'average_forecast_accuracy': 0.0,
                'analysis_timestamp': datetime.now().isoformat()
            }
        )
    
    def export_prediction_results(self, result: TrendPredictionResult, 
                                output_path: Union[str, Path]) -> str:
        """Export prediction results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'predictions': result.predictions,
            'trend_analysis': result.trend_analysis,
            'forecast_accuracy': result.forecast_accuracy,
            'model_performance': result.model_performance,
            'confidence_intervals': result.confidence_intervals,
            'recommendations': result.recommendations,
            'summary': result.summary,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Trend prediction results exported to {output_path}")
        return str(output_path)