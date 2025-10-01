# Analysis Module User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Time Series Analysis](#time-series-analysis)
4. [Advanced Modeling](#advanced-modeling)
5. [Statistical Methods](#statistical-methods)
6. [Visualization](#visualization)
7. [Reproducibility](#reproducibility)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Getting Started

### Installation

Ensure you have the required dependencies installed:

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn sqlalchemy
```

### Basic Setup

```python
from analysis.simulation_analysis import SimulationAnalyzer

# Initialize the analyzer
analyzer = SimulationAnalyzer("path/to/simulation.db", random_seed=42)
```

### Database Requirements

The analysis module expects a SQLite database with the following tables:
- `simulation_step_models`: Time series data for each simulation step
- `agent_models`: Individual agent data
- `action_models`: Agent action data
- `resource_models`: Resource data
- `simulations`: Simulation metadata

## Basic Usage

### Running a Complete Analysis

```python
# Run complete analysis with all methods
results = analyzer.run_complete_analysis(simulation_id=1, significance_level=0.05)

# Access different analysis components
population_dynamics = results["population_dynamics"]
resource_distribution = results["resource_distribution"]
agent_interactions = results["agent_interactions"]
temporal_patterns = results["temporal_patterns"]
advanced_models = results["advanced_time_series_models"]
ml_results = results["advanced_ml"]
```

### Individual Analysis Methods

```python
# Population dynamics analysis
pop_results = analyzer.analyze_population_dynamics(simulation_id=1)

# Resource distribution analysis
res_results = analyzer.analyze_resource_distribution(simulation_id=1)

# Agent interaction analysis
int_results = analyzer.analyze_agent_interactions(simulation_id=1)

# Critical event detection
events = analyzer.identify_critical_events(simulation_id=1, significance_level=0.05)
```

## Time Series Analysis

### Basic Time Series Analysis

The `analyze_temporal_patterns()` method provides comprehensive time series analysis:

```python
# Run basic time series analysis
temporal_results = analyzer.analyze_temporal_patterns(simulation_id=1)

# Access results for each time series
for series_name, analysis in temporal_results["time_series_analysis"].items():
    print(f"\n=== {series_name.upper()} ===")
    
    # Stationarity tests
    adf_test = analysis["stationarity"]["adf_test"]
    kpss_test = analysis["stationarity"]["kpss_test"]
    
    print(f"ADF Test: p-value = {adf_test['p_value']:.4f}")
    print(f"  Stationary: {adf_test['is_stationary']}")
    
    print(f"KPSS Test: p-value = {kpss_test['p_value']:.4f}")
    print(f"  Stationary: {kpss_test['is_stationary']}")
    
    # Trend analysis
    trend = analysis["trend"]["linear_trend"]
    print(f"Trend: {trend['trend_direction']}")
    print(f"R-squared: {trend['r_squared']:.4f}")
    print(f"Significant: {trend['significant_trend']}")
    
    # Seasonality
    if "decomposition" in analysis["seasonality"]:
        seasonal = analysis["seasonality"]["decomposition"]
        print(f"Has Seasonality: {seasonal['has_seasonality']}")
        print(f"Seasonal Strength: {seasonal['seasonal_strength']:.4f}")
    
    # Autocorrelation
    autocorr = analysis["autocorrelation"]
    print(f"Has Autocorrelation: {autocorr['has_autocorrelation']}")
    print(f"Max Autocorrelation: {autocorr['max_autocorr']:.4f}")

# Cross-correlations between time series
print("\n=== CROSS-CORRELATIONS ===")
for comparison, result in temporal_results["cross_correlations"].items():
    if "error" not in result:
        print(f"{comparison}:")
        print(f"  Correlation: {result['correlation']:.4f}")
        print(f"  Significant: {result['significant']}")
        print(f"  Strength: {result['strength']}")
```

### Advanced Time Series Modeling

The `analyze_advanced_time_series_models()` method provides advanced modeling:

```python
# Run advanced time series modeling
advanced_results = analyzer.analyze_advanced_time_series_models(simulation_id=1)

# ARIMA Models
print("=== ARIMA MODELS ===")
for series_name, arima_result in advanced_results["arima_models"].items():
    if "error" not in arima_result:
        print(f"\n{series_name}:")
        print(f"  Model Order: ARIMA{arima_result['model_order']}")
        print(f"  AIC: {arima_result['aic']:.2f}")
        print(f"  BIC: {arima_result['bic']:.2f}")
        
        # Forecasts
        forecast = arima_result['forecast']
        print(f"  Forecast (next 5 steps): {forecast[:5]}")
        
        # Model diagnostics
        if "ljung_box_test" in arima_result:
            lb_test = arima_result["ljung_box_test"]
            print(f"  Residuals are white noise: {lb_test['residuals_white_noise']}")
        
        # Residual statistics
        residuals_stats = arima_result["residuals_stats"]
        print(f"  Residual Mean: {residuals_stats['mean']:.4f}")
        print(f"  Residual Std: {residuals_stats['std']:.4f}")

# VAR Model
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
    forecast = var_result['forecast']
    print(f"VAR Forecast (next 3 steps): {forecast[:3]}")

# Exponential Smoothing
print("\n=== EXPONENTIAL SMOOTHING ===")
for series_name, exp_result in advanced_results["exponential_smoothing"].items():
    if "error" not in exp_result:
        print(f"\n{series_name}:")
        print(f"  Best Model: {exp_result['best_model']}")
        print(f"  AIC: {exp_result['model_info']['aic']:.2f}")
        print(f"  BIC: {exp_result['model_info']['bic']:.2f}")
        print(f"  Forecast (next 5 steps): {exp_result['forecast'][:5]}")

# Model Comparison
print("\n=== MODEL COMPARISON ===")
for series_name, comparison in advanced_results["model_comparison"].items():
    print(f"\n{series_name}:")
    print(f"  Best Model: {comparison['best_model']}")
    for model_name, metrics in comparison['comparison'].items():
        print(f"    {model_name}: AIC={metrics['aic']:.2f}, BIC={metrics['bic']:.2f}")
```

## Advanced Modeling

### Machine Learning Analysis

```python
# Run advanced ML analysis
ml_results = analyzer.analyze_with_advanced_ml(simulation_id=1, target_variable="population_dominance")

# Feature Selection Results
print("=== FEATURE SELECTION ===")
feature_selection = ml_results["feature_selection"]
for method, result in feature_selection.items():
    if "error" not in result:
        print(f"{method}: {result['n_features']} features selected")
        print(f"  Selected features: {result['selected_features'][:5]}...")

# Individual Models
print("\n=== INDIVIDUAL MODELS ===")
individual_models = ml_results["individual_models"]
for model_name, result in individual_models.items():
    if "error" not in result:
        print(f"{model_name}:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  CV Mean: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"  Feature Importance: {result['feature_importance'][:3]}...")

# Ensemble Models
print("\n=== ENSEMBLE MODELS ===")
ensemble_models = ml_results["ensemble_models"]
for model_name, result in ensemble_models.items():
    if "error" not in result:
        print(f"{model_name}:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  CV Mean: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")

# Best Model
print(f"\nBest Model: {ml_results['best_model']}")
print(f"Best Test Accuracy: {ml_results['performance_comparison'][ml_results['best_model']]['test_accuracy']:.4f}")
```

## Statistical Methods

### Effect Size Calculations

```python
# Access effect sizes from population dynamics analysis
pop_results = analyzer.analyze_population_dynamics(simulation_id=1)
pairwise_comparisons = pop_results["statistical_analysis"]["pairwise_comparisons"]

for comparison, result in pairwise_comparisons.items():
    if "error" not in result and "effect_sizes" in result:
        effect_sizes = result["effect_sizes"]
        print(f"\n{comparison}:")
        print(f"  Cohen's d: {effect_sizes['cohens_d']:.4f} ({effect_sizes['cohens_d_interpretation']})")
        print(f"  Hedges' g: {effect_sizes['hedges_g']:.4f}")
        print(f"  Glass's delta: {effect_sizes['glass_delta']:.4f}")
        print(f"  CLES: {effect_sizes['cles']:.4f}")
        print(f"  Eta-squared: {effect_sizes['eta_squared']:.4f} ({effect_sizes['eta_squared_interpretation']})")
```

### Power Analysis

```python
# Access power analysis results
for comparison, result in pairwise_comparisons.items():
    if "error" not in result and "power_analysis" in result:
        power_analysis = result["power_analysis"]
        print(f"\n{comparison} - Power Analysis:")
        print(f"  Observed Power: {power_analysis['observed_power']:.4f}")
        print(f"  Power Interpretation: {power_analysis['power_interpretation']}")
        print(f"  Effect Size: {power_analysis['effect_size']:.4f}")
        print(f"  Type II Error Rate: {power_analysis['type_ii_error_rate']:.4f}")
        print(f"  Sample Size for 80% Power: {power_analysis['sample_size_for_80_power']}")
```

### Critical Event Detection

```python
# Identify critical events with custom significance level
events = analyzer.identify_critical_events(simulation_id=1, significance_level=0.01)

print(f"Found {len(events)} potential events")
significant_events = [e for e in events if e['is_significant']]
print(f"Significant events: {len(significant_events)}")

for event in significant_events[:5]:  # Show first 5 significant events
    print(f"\nStep {event['step']} - {event['agent_type']}:")
    print(f"  Change Rate: {event['change_rate']:.4f}")
    print(f"  Z-Score: {event['z_score']:.4f}")
    print(f"  P-Value: {event['p_value']:.4f}")
    print(f"  Previous Value: {event['previous_value']}")
    print(f"  Current Value: {event['current_value']}")
```

## Visualization

### Generated Visualizations

The analysis module automatically generates high-quality visualizations:

#### Basic Time Series Analysis
- **File**: `temporal_analysis_sim_{id}.png`
- **Content**: 9-panel comprehensive visualization including:
  - Main time series plot with confidence bands
  - Trend analysis with R² values
  - Stationarity test results
  - Autocorrelation function plots
  - Seasonal decomposition
  - Cross-correlation heatmaps
  - Statistical summary panels

#### Advanced Time Series Modeling
- **File**: `advanced_time_series_models_sim_{id}.png`
- **Content**: 9-panel advanced modeling visualization including:
  - Time series with ARIMA forecasts and confidence bands
  - ARIMA model AIC comparison
  - Granger causality results
  - Exponential smoothing forecasts
  - Model comparison results
  - Residual analysis plots
  - Forecast accuracy metrics

#### Other Analysis Visualizations
- Population dynamics plots with confidence intervals
- Resource distribution box plots with statistical annotations
- Agent interaction heatmaps with significance indicators
- Critical event timeline plots
- Machine learning performance comparisons

### Customizing Visualizations

```python
# The visualizations are automatically generated with professional styling
# They include:
# - High resolution (300 DPI)
# - Professional color schemes
# - Statistical annotations
# - Confidence intervals and error bars
# - Publication-ready formatting
```

## Reproducibility

### Random Seed Management

```python
# Set random seed for reproducible results
analyzer = SimulationAnalyzer("simulation.db", random_seed=42)

# All analysis methods will use the same seed
results1 = analyzer.run_complete_analysis(simulation_id=1)
results2 = analyzer.run_complete_analysis(simulation_id=1)

# Results should be identical
assert results1["metadata"]["random_seed_used"] == results2["metadata"]["random_seed_used"]
```

### Reproducibility Reports

```python
# Complete analysis includes reproducibility information
results = analyzer.run_complete_analysis(simulation_id=1)

# Access reproducibility metadata
metadata = results["metadata"]
print(f"Analysis Timestamp: {metadata['analysis_timestamp']}")
print(f"Random Seed: {metadata['random_seed_used']}")
print(f"Analysis Version: {metadata['analysis_version']}")

# Validation report
if "validation_report" in results:
    validation = results["validation_report"]
    print(f"Overall Valid: {validation['overall_valid']}")
    print(f"Success Rate: {validation['summary']['success_rate']:.1%}")
```

## Troubleshooting

### Common Issues

#### Insufficient Data
```python
# Check data requirements before analysis
steps = analyzer.session.query(SimulationStepModel).filter(
    SimulationStepModel.simulation_id == simulation_id
).count()

if steps < 20:
    print("Warning: Insufficient data for basic analysis (minimum 20 points)")
elif steps < 50:
    print("Warning: Insufficient data for advanced modeling (minimum 50 points)")
else:
    print(f"Data sufficient: {steps} data points available")
```

#### Model Convergence Issues
```python
# Handle model convergence failures gracefully
try:
    results = analyzer.analyze_advanced_time_series_models(simulation_id=1)
    
    # Check for errors in results
    for series_name, arima_result in results["arima_models"].items():
        if "error" in arima_result:
            print(f"ARIMA failed for {series_name}: {arima_result['error']}")
    
    if "error" in results["var_model"]:
        print(f"VAR failed: {results['var_model']['error']}")
        
except Exception as e:
    print(f"Analysis failed: {e}")
```

#### Memory Issues
```python
# For large datasets, consider sampling
import random

# Sample data for initial exploration
all_steps = analyzer.session.query(SimulationStepModel).filter(
    SimulationStepModel.simulation_id == simulation_id
).all()

if len(all_steps) > 1000:
    # Sample 1000 points for analysis
    sampled_steps = random.sample(all_steps, 1000)
    # Use sampled data for analysis
```

### Error Handling

```python
# Robust error handling
def safe_analysis(analyzer, simulation_id):
    try:
        results = analyzer.run_complete_analysis(simulation_id)
        return results
    except Exception as e:
        print(f"Analysis failed: {e}")
        return {"error": str(e)}

# Use safe analysis
results = safe_analysis(analyzer, simulation_id=1)
if "error" not in results:
    print("Analysis completed successfully")
else:
    print(f"Analysis failed: {results['error']}")
```

## Best Practices

### Data Preparation

1. **Ensure Data Quality**:
   ```python
   # Check for missing values
   steps = analyzer.session.query(SimulationStepModel).filter(
       SimulationStepModel.simulation_id == simulation_id
   ).all()
   
   missing_data = any(step.system_agents is None for step in steps)
   if missing_data:
       print("Warning: Missing data detected")
   ```

2. **Validate Data Requirements**:
   ```python
   # Check minimum data requirements
   min_points = 20  # For basic analysis
   if len(steps) < min_points:
       print(f"Insufficient data: {len(steps)} < {min_points}")
   ```

### Statistical Interpretation

1. **Consider Practical Significance**:
   ```python
   # Don't just look at p-values
   p_value = 0.03
   effect_size = 0.15  # Small effect
   
   if p_value < 0.05 and effect_size > 0.2:
       print("Statistically and practically significant")
   elif p_value < 0.05 and effect_size <= 0.2:
       print("Statistically significant but small practical effect")
   ```

2. **Report Confidence Intervals**:
   ```python
   # Always report confidence intervals
   ci_lower = 0.45
   ci_upper = 0.55
   mean_value = 0.50
   
   print(f"Mean: {mean_value:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
   ```

### Performance Optimization

1. **Use Appropriate Sampling**:
   ```python
   # For very large datasets
   if len(steps) > 10000:
       # Use systematic sampling
       step_size = len(steps) // 1000
       sampled_steps = steps[::step_size]
   ```

2. **Monitor Memory Usage**:
   ```python
   import psutil
   
   # Check memory usage
   memory_percent = psutil.virtual_memory().percent
   if memory_percent > 80:
       print("Warning: High memory usage")
   ```

## Examples

### Complete Analysis Workflow

```python
#!/usr/bin/env python3
"""
Complete analysis workflow example
"""

from analysis.simulation_analysis import SimulationAnalyzer
import json

def main():
    # Initialize analyzer
    analyzer = SimulationAnalyzer("simulation.db", random_seed=42)
    
    # Run complete analysis
    print("Running complete analysis...")
    results = analyzer.run_complete_analysis(simulation_id=1, significance_level=0.05)
    
    # Print summary
    print(f"\nAnalysis completed successfully!")
    print(f"Analysis version: {results['metadata']['analysis_version']}")
    print(f"Random seed: {results['metadata']['random_seed_used']}")
    
    # Validation results
    if "validation_report" in results:
        validation = results["validation_report"]
        print(f"Validation passed: {validation['overall_valid']}")
        print(f"Success rate: {validation['summary']['success_rate']:.1%}")
    
    # Save results
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Results saved to analysis_results.json")
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    
    # Population dynamics
    pop_dyn = results["population_dynamics"]
    kruskal_test = pop_dyn["statistical_analysis"]["kruskal_wallis"]
    print(f"Population differences significant: {kruskal_test['significant_difference']} (p={kruskal_test['p_value']:.4f})")
    
    # Critical events
    events = results["critical_events"]
    significant_events = [e for e in events if e['is_significant']]
    print(f"Critical events found: {len(events)} total, {len(significant_events)} significant")
    
    # Time series analysis
    temporal = results["temporal_patterns"]
    print(f"Time series analyzed: {len(temporal['time_series_analysis'])}")
    
    # Advanced modeling
    advanced = results["advanced_time_series_models"]
    arima_models = len([k for k, v in advanced["arima_models"].items() if "error" not in v])
    print(f"ARIMA models fitted: {arima_models}")
    
    # Machine learning
    ml = results["advanced_ml"]
    print(f"Best ML model: {ml['best_model']}")
    print(f"Best accuracy: {ml['performance_comparison'][ml['best_model']]['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

### Custom Analysis Pipeline

```python
#!/usr/bin/env python3
"""
Custom analysis pipeline example
"""

from analysis.simulation_analysis import SimulationAnalyzer
import pandas as pd

def custom_analysis_pipeline(analyzer, simulation_id):
    """Custom analysis pipeline focusing on specific aspects"""
    
    results = {}
    
    # 1. Basic time series analysis
    print("Running time series analysis...")
    temporal_results = analyzer.analyze_temporal_patterns(simulation_id)
    results["temporal"] = temporal_results
    
    # 2. Advanced modeling for specific series
    print("Running advanced modeling...")
    advanced_results = analyzer.analyze_advanced_time_series_models(simulation_id)
    results["advanced"] = advanced_results
    
    # 3. Extract key insights
    insights = extract_insights(temporal_results, advanced_results)
    results["insights"] = insights
    
    return results

def extract_insights(temporal_results, advanced_results):
    """Extract key insights from analysis results"""
    
    insights = {
        "stationarity": {},
        "trends": {},
        "seasonality": {},
        "forecasts": {},
        "causality": {}
    }
    
    # Stationarity insights
    for series_name, analysis in temporal_results["time_series_analysis"].items():
        adf_test = analysis["stationarity"]["adf_test"]
        insights["stationarity"][series_name] = {
            "is_stationary": adf_test["is_stationary"],
            "p_value": adf_test["p_value"]
        }
    
    # Trend insights
    for series_name, analysis in temporal_results["time_series_analysis"].items():
        trend = analysis["trend"]["linear_trend"]
        insights["trends"][series_name] = {
            "direction": trend["trend_direction"],
            "strength": trend["r_squared"],
            "significant": trend["significant_trend"]
        }
    
    # Seasonality insights
    for series_name, analysis in temporal_results["time_series_analysis"].items():
        if "decomposition" in analysis["seasonality"]:
            seasonal = analysis["seasonality"]["decomposition"]
            insights["seasonality"][series_name] = {
                "has_seasonality": seasonal["has_seasonality"],
                "strength": seasonal["seasonal_strength"]
            }
    
    # Forecast insights
    for series_name, arima_result in advanced_results["arima_models"].items():
        if "error" not in arima_result:
            insights["forecasts"][series_name] = {
                "model": f"ARIMA{arima_result['model_order']}",
                "forecast": arima_result["forecast"][:5],  # Next 5 steps
                "aic": arima_result["aic"]
            }
    
    # Causality insights
    if "error" not in advanced_results["var_model"]:
        gc_results = advanced_results["var_model"]["granger_causality"]
        significant_causes = [k for k, v in gc_results.items() if v["significant"]]
        insights["causality"] = {
            "significant_relationships": significant_causes,
            "total_tests": len(gc_results)
        }
    
    return insights

def main():
    # Initialize analyzer
    analyzer = SimulationAnalyzer("simulation.db", random_seed=42)
    
    # Run custom analysis
    results = custom_analysis_pipeline(analyzer, simulation_id=1)
    
    # Print insights
    print("\n=== ANALYSIS INSIGHTS ===")
    
    print("\nStationarity:")
    for series, info in results["insights"]["stationarity"].items():
        print(f"  {series}: {'Stationary' if info['is_stationary'] else 'Non-stationary'} (p={info['p_value']:.4f})")
    
    print("\nTrends:")
    for series, info in results["insights"]["trends"].items():
        print(f"  {series}: {info['direction']} trend (R²={info['strength']:.4f}, significant={info['significant']})")
    
    print("\nSeasonality:")
    for series, info in results["insights"]["seasonality"].items():
        print(f"  {series}: {'Has seasonality' if info['has_seasonality'] else 'No seasonality'} (strength={info['strength']:.4f})")
    
    print("\nForecasts:")
    for series, info in results["insights"]["forecasts"].items():
        print(f"  {series}: {info['model']} (AIC={info['aic']:.2f})")
        print(f"    Next 5 steps: {info['forecast']}")
    
    print("\nCausality:")
    causality = results["insights"]["causality"]
    print(f"  Significant relationships: {len(causality['significant_relationships'])}/{causality['total_tests']}")
    for relationship in causality["significant_relationships"]:
        print(f"    {relationship}")

if __name__ == "__main__":
    main()
```

This user guide provides comprehensive coverage of all analysis module features with practical examples and best practices for effective usage.