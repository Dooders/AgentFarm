# Time Series Analysis Module

## Overview

The analysis module provides comprehensive time series analysis capabilities for simulation data, featuring both basic statistical methods and advanced modeling techniques. This module is designed for research, publication, and industrial applications.

## 🚀 Quick Start

```python
from analysis.simulation_analysis import SimulationAnalyzer

# Initialize analyzer
analyzer = SimulationAnalyzer("simulation.db", random_seed=42)

# Run complete time series analysis
results = analyzer.run_complete_analysis(simulation_id=1, significance_level=0.05)

# Access time series results
temporal_patterns = results["temporal_patterns"]
advanced_models = results["advanced_time_series_models"]
```

## 📊 Available Methods

### Basic Time Series Analysis (8 methods)

1. **Stationarity Testing**
   - Augmented Dickey-Fuller (ADF) test
   - KPSS test for trend stationarity
   - Critical values and p-values

2. **Trend Analysis**
   - Linear trend detection with R²
   - Statistical significance testing
   - Trend direction classification

3. **Seasonality Detection**
   - Seasonal decomposition (additive model)
   - Seasonality strength calculation
   - Periodogram analysis

4. **Change Point Detection**
   - Peak and trough identification
   - Signal processing methods
   - Statistical significance of changes

5. **Autocorrelation Analysis**
   - Lag-based correlation analysis
   - Significant lag identification
   - Autocorrelation function plots

6. **Cross-correlation Analysis**
   - Relationships between time series
   - Pearson correlation with significance
   - Correlation strength classification

### Advanced Time Series Modeling (6 methods)

1. **ARIMA Modeling**
   - Auto parameter selection (p,d,q)
   - Model comparison using AIC/BIC
   - Forecasting with confidence intervals
   - Residual diagnostics (Ljung-Box test)

2. **Vector Autoregression (VAR)**
   - Multivariate time series modeling
   - Automatic lag order selection
   - Granger causality testing
   - Impulse response analysis

3. **Exponential Smoothing**
   - Simple exponential smoothing
   - Holt's linear trend method
   - Holt-Winters seasonal method
   - Model selection based on AIC/BIC

4. **Model Comparison**
   - Information criteria comparison
   - Best model selection
   - Performance metrics

5. **Advanced Forecasting**
   - Multi-step ahead forecasting
   - Confidence intervals for predictions
   - Forecast accuracy evaluation

6. **Model Diagnostics**
   - Residual analysis and testing
   - Model adequacy checking
   - Statistical significance testing

## 💻 Usage Examples

### Basic Time Series Analysis

```python
# Run basic time series analysis
temporal_results = analyzer.analyze_temporal_patterns(simulation_id=1)

# Access results
for series_name, analysis in temporal_results["time_series_analysis"].items():
    print(f"\nTime Series: {series_name}")
    
    # Stationarity
    adf_test = analysis["stationarity"]["adf_test"]
    print(f"  Stationary (ADF): {adf_test['is_stationary']} (p={adf_test['p_value']:.3f})")
    
    # Trend
    trend = analysis["trend"]["linear_trend"]
    print(f"  Trend: {trend['trend_direction']} (R²={trend['r_squared']:.3f})")
    
    # Seasonality
    if "decomposition" in analysis["seasonality"]:
        seasonal = analysis["seasonality"]["decomposition"]
        print(f"  Seasonality: {seasonal['has_seasonality']} (strength={seasonal['seasonal_strength']:.3f})")
    
    # Autocorrelation
    autocorr = analysis["autocorrelation"]
    print(f"  Autocorrelation: {autocorr['has_autocorrelation']} (max={autocorr['max_autocorr']:.3f})")

# Cross-correlations
cross_corr = temporal_results["cross_correlations"]
for comparison, result in cross_corr.items():
    if "error" not in result:
        print(f"\n{comparison}:")
        print(f"  Correlation: {result['correlation']:.3f}")
        print(f"  Significant: {result['significant']}")
        print(f"  Strength: {result['strength']}")
```

### Advanced Time Series Modeling

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

# Access VAR model
print("\n=== VAR MODEL ===")
var_result = advanced_results["var_model"]
if "error" not in var_result:
    print(f"Lag Order: {var_result['model_order']}")
    print(f"AIC: {var_result['aic']:.2f}")
    
    # Granger causality
    gc_results = var_result["granger_causality"]
    significant_causes = [k for k, v in gc_results.items() if v["significant"]]
    print(f"Significant Granger Causality: {significant_causes}")

# Access exponential smoothing
print("\n=== EXPONENTIAL SMOOTHING ===")
for series_name, exp_result in advanced_results["exponential_smoothing"].items():
    if "error" not in exp_result:
        print(f"\n{series_name}:")
        print(f"  Best Model: {exp_result['best_model']}")
        print(f"  AIC: {exp_result['model_info']['aic']:.2f}")
        print(f"  Forecast: {exp_result['forecast'][:5]}...")

# Model comparison
print("\n=== MODEL COMPARISON ===")
for series_name, comparison in advanced_results["model_comparison"].items():
    print(f"\n{series_name}:")
    print(f"  Best Model: {comparison['best_model']}")
```

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

## 📈 Visualization Outputs

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

## 🔧 Configuration

### Data Requirements
- **Minimum Points**: 20 for basic analysis, 50 for advanced modeling
- **Data Quality**: Handle missing values appropriately
- **Frequency**: Consistent time intervals recommended

### Parameters
```python
# Significance level for statistical tests
significance_level = 0.05  # 5% significance level

# Random seed for reproducibility
random_seed = 42

# Analysis parameters
analyzer = SimulationAnalyzer("simulation.db", random_seed=random_seed)
results = analyzer.run_complete_analysis(simulation_id=1, significance_level=significance_level)
```

## 📚 Statistical Methods Reference

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

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python3 run_tests.py

# Run specific test file
python3 -m unittest tests.test_simulation_analysis

# Run with verbose output
python3 -m unittest tests.test_simulation_analysis -v
```

### Test Coverage

The test suite covers:
- ✅ Basic time series analysis methods
- ✅ Advanced time series modeling
- ✅ ARIMA model fitting and diagnostics
- ✅ VAR model and Granger causality
- ✅ Exponential smoothing methods
- ✅ Model comparison and selection
- ✅ Error handling and edge cases
- ✅ Visualization generation
- ✅ Integration with complete analysis workflow

## 🚨 Troubleshooting

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

## 📖 Additional Resources

- [Time Series Analysis Guide](TIME_SERIES_ANALYSIS_GUIDE.md) - Comprehensive guide
- [Phase 2 Improvements](PHASE_2_IMPROVEMENTS.md) - Technical details
- [Test Documentation](tests/) - Test suite documentation

## 🎯 Key Features

- ✅ **14 Statistical Methods**: Comprehensive time series analysis
- ✅ **Advanced Modeling**: ARIMA, VAR, exponential smoothing
- ✅ **Professional Visualizations**: Publication-ready plots
- ✅ **Robust Validation**: Statistical tests and diagnostics
- ✅ **Production Ready**: Error handling and performance optimization
- ✅ **Reproducible**: Random seed management and validation
- ✅ **Well Tested**: Comprehensive test suite
- ✅ **Documented**: Extensive documentation and examples

## 📄 License

This module is part of the simulation analysis framework and follows the same licensing terms.

---

**Ready for research, publication, and industrial applications!** 🚀