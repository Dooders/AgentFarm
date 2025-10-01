# Time Series Analysis Guide

## Overview

The analysis module provides comprehensive time series analysis capabilities with both basic statistical methods and advanced modeling techniques. This guide covers all available time series analysis features.

## Table of Contents

1. [Basic Time Series Analysis](#basic-time-series-analysis)
2. [Advanced Time Series Modeling](#advanced-time-series-modeling)
3. [Usage Examples](#usage-examples)
4. [Visualization Outputs](#visualization-outputs)
5. [Statistical Methods Reference](#statistical-methods-reference)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Basic Time Series Analysis

### Available Methods

The `analyze_temporal_patterns()` method provides 8 comprehensive time series analysis techniques:

#### 1. Stationarity Testing
- **Augmented Dickey-Fuller (ADF) Test**: Tests for unit root (non-stationarity)
- **KPSS Test**: Tests for trend stationarity
- **Critical Values**: 1%, 5%, and 10% significance levels
- **Interpretation**: p-value < 0.05 indicates stationarity (ADF) or non-stationarity (KPSS)

#### 2. Trend Analysis
- **Linear Regression**: Fits linear trend to time series
- **R-squared**: Measures goodness of fit
- **Statistical Significance**: Tests if trend is significantly different from zero
- **Trend Direction**: Classifies as increasing, decreasing, or stable

#### 3. Seasonality Detection
- **Seasonal Decomposition**: Separates trend, seasonal, and residual components
- **Seasonality Strength**: Measures relative importance of seasonal component
- **Periodogram Analysis**: Identifies dominant frequencies
- **Seasonal Patterns**: Detects periodic behavior in data

#### 4. Change Point Detection
- **Peak Detection**: Identifies local maxima using signal processing
- **Trough Detection**: Identifies local minima
- **Statistical Significance**: Tests significance of detected changes
- **Change Magnitude**: Quantifies size of detected changes

#### 5. Autocorrelation Analysis
- **Autocorrelation Function**: Calculates correlations at different lags
- **Significant Lags**: Identifies lags with significant autocorrelation
- **Pattern Recognition**: Detects periodic patterns in data
- **Memory Effects**: Measures persistence in time series

#### 6. Cross-correlation Analysis
- **Pearson Correlation**: Measures linear relationships between series
- **Statistical Significance**: Tests significance of correlations
- **Correlation Strength**: Classifies as weak, moderate, or strong
- **Lead-Lag Relationships**: Identifies which series leads/lags

### Usage

```python
from analysis.simulation_analysis import SimulationAnalyzer

# Initialize analyzer
analyzer = SimulationAnalyzer("simulation.db", random_seed=42)

# Run basic time series analysis
temporal_results = analyzer.analyze_temporal_patterns(simulation_id=1)

# Access results
for series_name, analysis in temporal_results["time_series_analysis"].items():
    print(f"\nTime Series: {series_name}")
    
    # Stationarity
    adf_test = analysis["stationarity"]["adf_test"]
    print(f"  Stationary (ADF): {adf_test['is_stationary']} (p={adf_test['p_value']:.3f})")
    
    kpss_test = analysis["stationarity"]["kpss_test"]
    print(f"  Stationary (KPSS): {kpss_test['is_stationary']} (p={kpss_test['p_value']:.3f})")
    
    # Trend
    trend = analysis["trend"]["linear_trend"]
    print(f"  Trend: {trend['trend_direction']} (R²={trend['r_squared']:.3f})")
    print(f"  Trend Significant: {trend['significant_trend']}")
    
    # Seasonality
    if "decomposition" in analysis["seasonality"]:
        seasonal = analysis["seasonality"]["decomposition"]
        print(f"  Has Seasonality: {seasonal['has_seasonality']}")
        print(f"  Seasonal Strength: {seasonal['seasonal_strength']:.3f}")
    
    # Autocorrelation
    autocorr = analysis["autocorrelation"]
    print(f"  Has Autocorrelation: {autocorr['has_autocorrelation']}")
    print(f"  Max Autocorr: {autocorr['max_autocorr']:.3f}")

# Cross-correlations
cross_corr = temporal_results["cross_correlations"]
for comparison, result in cross_corr.items():
    if "error" not in result:
        print(f"\n{comparison}:")
        print(f"  Correlation: {result['correlation']:.3f}")
        print(f"  Significant: {result['significant']}")
        print(f"  Strength: {result['strength']}")
```

## Advanced Time Series Modeling

### Available Methods

The `analyze_advanced_time_series_models()` method provides 6 advanced modeling techniques:

#### 1. ARIMA Modeling
- **Auto Parameter Selection**: Automatically selects optimal (p,d,q) parameters
- **Model Comparison**: Uses AIC/BIC for model selection
- **Forecasting**: Generates multi-step ahead forecasts
- **Confidence Intervals**: Provides forecast uncertainty bounds
- **Model Diagnostics**: Ljung-Box test for residual autocorrelation
- **Residual Analysis**: Tests for white noise residuals

#### 2. Vector Autoregression (VAR)
- **Multivariate Modeling**: Models multiple time series simultaneously
- **Lag Order Selection**: Automatically selects optimal lag length
- **Granger Causality**: Tests causal relationships between series
- **Impulse Response**: Analyzes dynamic responses to shocks
- **Forecast Error Variance Decomposition**: Measures contribution of each variable

#### 3. Exponential Smoothing
- **Simple Exponential Smoothing**: For data without trend or seasonality
- **Holt's Linear Trend**: For data with trend but no seasonality
- **Holt-Winters Seasonal**: For data with both trend and seasonality
- **Model Selection**: Automatically selects best method based on AIC/BIC
- **Forecasting**: Generates forecasts with trend and seasonal components

#### 4. Model Comparison
- **Information Criteria**: AIC and BIC for model comparison
- **Best Model Selection**: Automatically identifies optimal model
- **Performance Metrics**: Compares forecast accuracy
- **Statistical Tests**: Tests for model adequacy

#### 5. Advanced Forecasting
- **Multi-step Ahead**: Generates forecasts for multiple future periods
- **Confidence Intervals**: Provides uncertainty quantification
- **Forecast Evaluation**: Measures forecast accuracy
- **Model Validation**: Tests forecast reliability

#### 6. Model Diagnostics
- **Residual Analysis**: Tests for white noise residuals
- **Ljung-Box Test**: Tests for residual autocorrelation
- **Model Adequacy**: Validates model assumptions
- **Statistical Significance**: Tests model parameters

### Usage

```python
# Run advanced time series modeling
advanced_results = analyzer.analyze_advanced_time_series_models(simulation_id=1)

# Access ARIMA models
print("=== ARIMA MODELS ===")
for series_name, arima_result in advanced_results["arima_models"].items():
    if "error" not in arima_result:
        print(f"\n{series_name}:")
        print(f"  Order: ARIMA{arima_result['model_order']}")
        print(f"  AIC: {arima_result['aic']:.2f}")
        print(f"  BIC: {arima_result['bic']:.2f}")
        print(f"  Forecast: {arima_result['forecast'][:5]}...")
        
        # Model diagnostics
        if "ljung_box_test" in arima_result:
            lb_test = arima_result["ljung_box_test"]
            print(f"  Residuals are white noise: {lb_test['residuals_white_noise']}")
        
        # Residual statistics
        residuals_stats = arima_result["residuals_stats"]
        print(f"  Residual Mean: {residuals_stats['mean']:.3f}")
        print(f"  Residual Std: {residuals_stats['std']:.3f}")

# Access VAR model
print("\n=== VAR MODEL ===")
var_result = advanced_results["var_model"]
if "error" not in var_result:
    print(f"Lag Order: {var_result['model_order']}")
    print(f"AIC: {var_result['aic']:.2f}")
    print(f"BIC: {var_result['bic']:.2f}")
    
    # Granger causality
    gc_results = var_result["granger_causality"]
    significant_causes = [k for k, v in gc_results.items() if v["significant"]]
    print(f"Significant Granger Causality: {significant_causes}")
    
    # Forecast
    print(f"VAR Forecast: {var_result['forecast'][:3]}...")

# Access exponential smoothing
print("\n=== EXPONENTIAL SMOOTHING ===")
for series_name, exp_result in advanced_results["exponential_smoothing"].items():
    if "error" not in exp_result:
        print(f"\n{series_name}:")
        print(f"  Best Model: {exp_result['best_model']}")
        print(f"  AIC: {exp_result['model_info']['aic']:.2f}")
        print(f"  BIC: {exp_result['model_info']['bic']:.2f}")
        print(f"  Forecast: {exp_result['forecast'][:5]}...")

# Model comparison
print("\n=== MODEL COMPARISON ===")
for series_name, comparison in advanced_results["model_comparison"].items():
    print(f"\n{series_name}:")
    print(f"  Best Model: {comparison['best_model']}")
    print(f"  Comparison: {comparison['comparison']}")
```

## Integration with Other Analysis Methods

The time series analysis methods integrate seamlessly with other analysis capabilities:

### Population Dynamics Analysis
```python
# Run population dynamics analysis with time series components
pop_results = analyzer.analyze_population_dynamics(simulation_id=1)
# Includes temporal patterns in population changes
```

### Resource Distribution Analysis
```python
# Run resource distribution analysis
res_results = analyzer.analyze_resource_distribution(simulation_id=1)
# Includes temporal patterns in resource allocation
```

### Agent Interaction Analysis
```python
# Run agent interaction analysis
int_results = analyzer.analyze_agent_interactions(simulation_id=1)
# Includes temporal patterns in interaction frequencies
```

### Advanced Machine Learning Analysis
```python
# Run advanced ML analysis with temporal features
ml_results = analyzer.analyze_with_advanced_ml(simulation_id=1)
# Includes time series features in ML models
```

## Usage Examples

### Complete Analysis Workflow

```python
# Run complete analysis (includes all time series methods)
complete_results = analyzer.run_complete_analysis(simulation_id=1, significance_level=0.05)

# Access all time series results
print("=== BASIC TIME SERIES ANALYSIS ===")
temporal_patterns = complete_results["temporal_patterns"]
print(f"Time series analyzed: {len(temporal_patterns['time_series_analysis'])}")

print("\n=== ADVANCED TIME SERIES MODELING ===")
advanced_models = complete_results["advanced_time_series_models"]
print(f"ARIMA models fitted: {len([k for k, v in advanced_models['arima_models'].items() if 'error' not in v])}")
print(f"VAR model fitted: {'Yes' if 'error' not in advanced_models['var_model'] else 'No'}")
print(f"Exponential smoothing models: {len([k for k, v in advanced_models['exponential_smoothing'].items() if 'error' not in v])}")

# Validation report
if "validation_report" in complete_results:
    validation = complete_results["validation_report"]
    print(f"\n=== VALIDATION REPORT ===")
    print(f"Overall Valid: {validation['overall_valid']}")
    print(f"Success Rate: {validation['summary']['success_rate']:.1%}")
```

### Custom Analysis

```python
# Analyze specific time series with custom parameters
temporal_results = analyzer.analyze_temporal_patterns(simulation_id=1)

# Focus on specific series
system_agents_analysis = temporal_results["time_series_analysis"]["system_agents"]

# Check stationarity
if system_agents_analysis["stationarity"]["adf_test"]["is_stationary"]:
    print("System agents time series is stationary")
else:
    print("System agents time series is non-stationary - consider differencing")

# Check for trend
trend = system_agents_analysis["trend"]["linear_trend"]
if trend["significant_trend"]:
    print(f"Significant {trend['trend_direction']} trend detected")
    print(f"Trend strength (R²): {trend['r_squared']:.3f}")

# Check for seasonality
if "decomposition" in system_agents_analysis["seasonality"]:
    seasonal = system_agents_analysis["seasonality"]["decomposition"]
    if seasonal["has_seasonality"]:
        print(f"Seasonal pattern detected with strength: {seasonal['seasonal_strength']:.3f}")
```

## Visualization Outputs

### Basic Time Series Analysis
- **File**: `temporal_analysis_sim_{id}.png`
- **Panels**: 9 comprehensive panels
- **Content**:
  - Main time series plot with confidence bands
  - Trend analysis with R² values
  - Stationarity test results (ADF p-values)
  - Autocorrelation function plots
  - Seasonal decomposition pie charts
  - Cross-correlation heatmaps
  - Statistical summary panels

### Advanced Time Series Modeling
- **File**: `advanced_time_series_models_sim_{id}.png`
- **Panels**: 9 advanced modeling panels
- **Content**:
  - Time series with ARIMA forecasts and confidence bands
  - ARIMA model AIC comparison
  - Granger causality results
  - Exponential smoothing forecasts
  - Model comparison results
  - Residual analysis plots
  - Forecast accuracy metrics
  - Comprehensive summary statistics

## Statistical Methods Reference

### Stationarity Tests

#### Augmented Dickey-Fuller (ADF) Test
- **Purpose**: Tests for unit root (non-stationarity)
- **Null Hypothesis**: Series has unit root (non-stationary)
- **Alternative Hypothesis**: Series is stationary
- **Interpretation**: p-value < 0.05 → reject null → series is stationary

#### KPSS Test
- **Purpose**: Tests for trend stationarity
- **Null Hypothesis**: Series is trend stationary
- **Alternative Hypothesis**: Series has unit root
- **Interpretation**: p-value > 0.05 → fail to reject null → series is trend stationary

### Trend Analysis

#### Linear Trend
- **Method**: Ordinary Least Squares (OLS) regression
- **Model**: y = α + βt + ε
- **Significance**: t-test for β ≠ 0
- **Goodness of Fit**: R-squared

### Seasonality Analysis

#### Seasonal Decomposition
- **Method**: Additive decomposition
- **Components**: y(t) = T(t) + S(t) + R(t)
- **Seasonality Strength**: Var(S) / Var(y)

#### Periodogram
- **Purpose**: Frequency domain analysis
- **Method**: Fourier transform
- **Output**: Power spectral density

### ARIMA Modeling

#### Parameter Selection
- **p**: Autoregressive order
- **d**: Differencing order
- **q**: Moving average order
- **Selection**: Grid search with AIC/BIC

#### Model Diagnostics
- **Ljung-Box Test**: Tests residual autocorrelation
- **Null Hypothesis**: Residuals are white noise
- **Interpretation**: p-value > 0.05 → residuals are white noise

### Vector Autoregression (VAR)

#### Granger Causality
- **Purpose**: Tests causal relationships
- **Null Hypothesis**: Variable X does not Granger-cause variable Y
- **Method**: F-test on lagged values
- **Interpretation**: p-value < 0.05 → X Granger-causes Y

### Exponential Smoothing

#### Model Types
- **Simple**: For data without trend or seasonality
- **Holt**: For data with trend but no seasonality
- **Holt-Winters**: For data with both trend and seasonality

#### Model Selection
- **Criteria**: AIC, BIC, SSE
- **Method**: Fit all models, select best based on criteria

## Best Practices

### Data Requirements
- **Minimum Points**: 20 for basic analysis, 50 for advanced modeling
- **Data Quality**: Handle missing values appropriately
- **Frequency**: Consistent time intervals recommended

### Model Selection
- **Information Criteria**: Use AIC/BIC for model comparison
- **Cross-validation**: Validate models on out-of-sample data
- **Diagnostics**: Always check model assumptions

### Interpretation
- **Statistical Significance**: Consider practical significance
- **Confidence Intervals**: Always report uncertainty
- **Model Validation**: Test assumptions and diagnostics

### Performance
- **Computational Efficiency**: ARIMA can be slow for large datasets
- **Memory Usage**: VAR models require more memory
- **Convergence**: Some models may not converge

## Troubleshooting

### Common Issues

#### Insufficient Data
```
Error: Insufficient data for time series analysis
Solution: Ensure at least 20 data points for basic analysis, 50 for advanced modeling
```

#### Model Convergence
```
Error: ARIMA model failed to converge
Solution: Try different parameter ranges or simpler models
```

#### Memory Issues
```
Error: VAR model requires too much memory
Solution: Reduce number of variables or use smaller lag orders
```

#### Missing Dependencies
```
Error: ModuleNotFoundError: No module named 'statsmodels'
Solution: Install required packages: pip install statsmodels scipy
```

### Performance Optimization

#### Large Datasets
- Use data sampling for initial exploration
- Consider parallel processing for multiple models
- Optimize parameter search ranges

#### Memory Management
- Clear large objects after use
- Use chunked processing for very large datasets
- Monitor memory usage during analysis

### Error Handling

#### Robust Analysis
```python
try:
    results = analyzer.analyze_advanced_time_series_models(simulation_id=1)
    if "error" in results:
        print(f"Analysis failed: {results['error']}")
    else:
        print("Analysis completed successfully")
except Exception as e:
    print(f"Unexpected error: {e}")
```

#### Validation
```python
# Check data quality before analysis
steps = analyzer.session.query(SimulationStepModel).filter(
    SimulationStepModel.simulation_id == simulation_id
).count()

if steps < 50:
    print("Warning: Insufficient data for advanced modeling")
    # Use basic analysis instead
    results = analyzer.analyze_temporal_patterns(simulation_id)
```

## Conclusion

The time series analysis capabilities provide comprehensive tools for understanding temporal patterns in simulation data. With 14 statistical methods covering both basic analysis and advanced modeling, the system is suitable for research, publication, and industrial applications.

Key strengths:
- ✅ **Comprehensive Coverage**: 14 statistical methods
- ✅ **Advanced Modeling**: ARIMA, VAR, exponential smoothing
- ✅ **Professional Visualizations**: Publication-ready plots
- ✅ **Robust Validation**: Statistical tests and diagnostics
- ✅ **Production Ready**: Error handling and performance optimization

For additional support or feature requests, refer to the main analysis module documentation.