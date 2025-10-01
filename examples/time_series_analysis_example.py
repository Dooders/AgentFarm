#!/usr/bin/env python3
"""
Time Series Analysis Example

This example demonstrates comprehensive time series analysis capabilities
including basic statistical methods and advanced modeling techniques.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add the analysis module to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.simulation_analysis import SimulationAnalyzer


def demonstrate_basic_time_series_analysis(analyzer, simulation_id):
    """Demonstrate basic time series analysis methods."""
    
    print("=" * 60)
    print("BASIC TIME SERIES ANALYSIS")
    print("=" * 60)
    
    # Run basic time series analysis
    print("Running basic time series analysis...")
    temporal_results = analyzer.analyze_temporal_patterns(simulation_id)
    
    # Display results for each time series
    for series_name, analysis in temporal_results["time_series_analysis"].items():
        print(f"\n--- {series_name.upper().replace('_', ' ')} ---")
        
        # Stationarity tests
        adf_test = analysis["stationarity"]["adf_test"]
        kpss_test = analysis["stationarity"]["kpss_test"]
        
        print(f"Stationarity Tests:")
        print(f"  ADF Test: p-value = {adf_test['p_value']:.4f}")
        print(f"    Stationary: {adf_test['is_stationary']}")
        print(f"  KPSS Test: p-value = {kpss_test['p_value']:.4f}")
        print(f"    Stationary: {kpss_test['is_stationary']}")
        
        # Trend analysis
        trend = analysis["trend"]["linear_trend"]
        print(f"\nTrend Analysis:")
        print(f"  Direction: {trend['trend_direction']}")
        print(f"  R-squared: {trend['r_squared']:.4f}")
        print(f"  Significant: {trend['significant_trend']}")
        print(f"  Slope: {trend['slope']:.4f}")
        
        # Seasonality
        if "decomposition" in analysis["seasonality"]:
            seasonal = analysis["seasonality"]["decomposition"]
            print(f"\nSeasonality Analysis:")
            print(f"  Has Seasonality: {seasonal['has_seasonality']}")
            print(f"  Seasonal Strength: {seasonal['seasonal_strength']:.4f}")
            print(f"  Trend Strength: {seasonal['trend_strength']:.4f}")
            print(f"  Residual Strength: {seasonal['residual_strength']:.4f}")
        
        # Autocorrelation
        autocorr = analysis["autocorrelation"]
        print(f"\nAutocorrelation Analysis:")
        print(f"  Has Autocorrelation: {autocorr['has_autocorrelation']}")
        print(f"  Max Autocorrelation: {autocorr['max_autocorr']:.4f}")
        print(f"  Significant Lags: {autocorr['significant_lags']}")
        
        # Change points
        if "change_points" in analysis:
            change_points = analysis["change_points"]
            print(f"\nChange Point Detection:")
            print(f"  Peaks Found: {len(change_points.get('peaks', []))}")
            print(f"  Troughs Found: {len(change_points.get('troughs', []))}")
    
    # Cross-correlations
    print(f"\n--- CROSS-CORRELATIONS ---")
    cross_corr = temporal_results["cross_correlations"]
    for comparison, result in cross_corr.items():
        if "error" not in result:
            print(f"{comparison.replace('_', ' ').title()}:")
            print(f"  Correlation: {result['correlation']:.4f}")
            print(f"  Significant: {result['significant']}")
            print(f"  Strength: {result['strength']}")
            print(f"  P-value: {result['p_value']:.4f}")
    
    return temporal_results


def demonstrate_advanced_time_series_modeling(analyzer, simulation_id):
    """Demonstrate advanced time series modeling methods."""
    
    print("\n" + "=" * 60)
    print("ADVANCED TIME SERIES MODELING")
    print("=" * 60)
    
    # Run advanced time series modeling
    print("Running advanced time series modeling...")
    advanced_results = analyzer.analyze_advanced_time_series_models(simulation_id)
    
    # ARIMA Models
    print(f"\n--- ARIMA MODELS ---")
    arima_models = advanced_results["arima_models"]
    for series_name, arima_result in arima_models.items():
        if "error" not in arima_result:
            print(f"\n{series_name.replace('_', ' ').title()}:")
            print(f"  Model Order: ARIMA{arima_result['model_order']}")
            print(f"  AIC: {arima_result['aic']:.2f}")
            print(f"  BIC: {arima_result['bic']:.2f}")
            print(f"  Log Likelihood: {arima_result['model_summary']['log_likelihood']:.2f}")
            
            # Forecasts
            forecast = arima_result['forecast']
            ci_lower = arima_result['forecast_ci_lower']
            ci_upper = arima_result['forecast_ci_upper']
            print(f"  Forecast (next 5 steps):")
            for i, (f, l, u) in enumerate(zip(forecast[:5], ci_lower[:5], ci_upper[:5])):
                print(f"    Step {i+1}: {f:.2f} (95% CI: [{l:.2f}, {u:.2f}])")
            
            # Model diagnostics
            if "ljung_box_test" in arima_result:
                lb_test = arima_result["ljung_box_test"]
                print(f"  Ljung-Box Test:")
                print(f"    Statistic: {lb_test['statistic']:.4f}")
                print(f"    P-value: {lb_test['p_value']:.4f}")
                print(f"    Residuals are white noise: {lb_test['residuals_white_noise']}")
            
            # Residual statistics
            residuals_stats = arima_result["residuals_stats"]
            print(f"  Residual Statistics:")
            print(f"    Mean: {residuals_stats['mean']:.4f}")
            print(f"    Std: {residuals_stats['std']:.4f}")
            print(f"    Skewness: {residuals_stats['skewness']:.4f}")
            print(f"    Kurtosis: {residuals_stats['kurtosis']:.4f}")
        else:
            print(f"\n{series_name.replace('_', ' ').title()}: Error - {arima_result['error']}")
    
    # VAR Model
    print(f"\n--- VECTOR AUTOREGRESSION (VAR) MODEL ---")
    var_result = advanced_results["var_model"]
    if "error" not in var_result:
        print(f"Lag Order: {var_result['model_order']}")
        print(f"AIC: {var_result['aic']:.2f}")
        print(f"BIC: {var_result['bic']:.2f}")
        
        # Granger causality
        gc_results = var_result["granger_causality"]
        print(f"\nGranger Causality Tests:")
        significant_causes = []
        for test_name, result in gc_results.items():
            cause, effect = test_name.split('_causes_')
            print(f"  {cause.replace('_', ' ').title()} → {effect.replace('_', ' ').title()}:")
            print(f"    F-statistic: {result['statistic']:.4f}")
            print(f"    P-value: {result['p_value']:.4f}")
            print(f"    Significant: {result['significant']}")
            if result['significant']:
                significant_causes.append(test_name)
        
        print(f"\nSignificant Granger Causality Relationships: {len(significant_causes)}")
        for relationship in significant_causes:
            cause, effect = relationship.split('_causes_')
            print(f"  • {cause.replace('_', ' ').title()} causes {effect.replace('_', ' ').title()}")
        
        # VAR Forecast
        forecast = var_result['forecast']
        print(f"\nVAR Forecast (next 3 steps):")
        for i, step_forecast in enumerate(forecast[:3]):
            print(f"  Step {i+1}: {step_forecast}")
    else:
        print(f"VAR Model: Error - {var_result['error']}")
    
    # Exponential Smoothing
    print(f"\n--- EXPONENTIAL SMOOTHING ---")
    exp_smoothing = advanced_results["exponential_smoothing"]
    for series_name, exp_result in exp_smoothing.items():
        if "error" not in exp_result:
            print(f"\n{series_name.replace('_', ' ').title()}:")
            print(f"  Best Model: {exp_result['best_model']}")
            model_info = exp_result['model_info']
            print(f"  AIC: {model_info['aic']:.2f}")
            print(f"  BIC: {model_info['bic']:.2f}")
            print(f"  SSE: {model_info['sse']:.2f}")
            
            # Forecast
            forecast = exp_result['forecast']
            print(f"  Forecast (next 5 steps): {forecast[:5]}")
            
            # Model-specific information
            if exp_result['best_model'] == 'holt_winters':
                print(f"  Seasonal Period: {model_info.get('seasonal_period', 'N/A')}")
        else:
            print(f"\n{series_name.replace('_', ' ').title()}: Error - {exp_result['error']}")
    
    # Model Comparison
    print(f"\n--- MODEL COMPARISON ---")
    model_comparison = advanced_results["model_comparison"]
    for series_name, comparison in model_comparison.items():
        print(f"\n{series_name.replace('_', ' ').title()}:")
        print(f"  Best Model: {comparison['best_model']}")
        print(f"  Model Comparison:")
        for model_name, metrics in comparison['comparison'].items():
            print(f"    {model_name}: AIC={metrics['aic']:.2f}, BIC={metrics['bic']:.2f}")
    
    return advanced_results


def demonstrate_complete_analysis(analyzer, simulation_id):
    """Demonstrate complete analysis workflow."""
    
    print("\n" + "=" * 60)
    print("COMPLETE ANALYSIS WORKFLOW")
    print("=" * 60)
    
    # Run complete analysis
    print("Running complete analysis...")
    results = analyzer.run_complete_analysis(simulation_id, significance_level=0.05)
    
    # Print analysis metadata
    metadata = results["metadata"]
    print(f"\nAnalysis Metadata:")
    print(f"  Timestamp: {metadata['analysis_timestamp']}")
    print(f"  Random Seed: {metadata['random_seed_used']}")
    print(f"  Analysis Version: {metadata['analysis_version']}")
    print(f"  Significance Level: {metadata['significance_level']}")
    print(f"  Data Quality Checks: {metadata['data_quality_checks']}")
    
    # Statistical methods used
    print(f"\nStatistical Methods Used:")
    for method in metadata['statistical_methods_used']:
        print(f"  • {method}")
    
    # Validation report
    if "validation_report" in results:
        validation = results["validation_report"]
        print(f"\nValidation Report:")
        print(f"  Overall Valid: {validation['overall_valid']}")
        print(f"  Success Rate: {validation['summary']['success_rate']:.1%}")
        print(f"  Total Validations: {validation['summary']['total_validations']}")
        print(f"  Passed Validations: {validation['summary']['passed_validations']}")
        
        if not validation['overall_valid']:
            print(f"  Errors Found:")
            for error in validation['errors']:
                print(f"    • {error}")
    
    # Key findings summary
    print(f"\nKey Findings Summary:")
    
    # Population dynamics
    pop_dyn = results["population_dynamics"]
    kruskal_test = pop_dyn["statistical_analysis"]["kruskal_wallis"]
    print(f"  Population Dynamics:")
    print(f"    Significant differences: {kruskal_test['significant_difference']} (p={kruskal_test['p_value']:.4f})")
    
    # Critical events
    events = results["critical_events"]
    significant_events = [e for e in events if e['is_significant']]
    print(f"  Critical Events:")
    print(f"    Total events: {len(events)}")
    print(f"    Significant events: {len(significant_events)}")
    
    # Time series analysis
    temporal = results["temporal_patterns"]
    print(f"  Time Series Analysis:")
    print(f"    Series analyzed: {len(temporal['time_series_analysis'])}")
    
    # Advanced modeling
    advanced = results["advanced_time_series_models"]
    arima_models = len([k for k, v in advanced["arima_models"].items() if "error" not in v])
    var_success = "error" not in advanced["var_model"]
    exp_models = len([k for k, v in advanced["exponential_smoothing"].items() if "error" not in v])
    print(f"  Advanced Modeling:")
    print(f"    ARIMA models fitted: {arima_models}")
    print(f"    VAR model fitted: {var_success}")
    print(f"    Exponential smoothing models: {exp_models}")
    
    # Machine learning
    ml = results["advanced_ml"]
    print(f"  Machine Learning:")
    print(f"    Best model: {ml['best_model']}")
    best_accuracy = ml['performance_comparison'][ml['best_model']]['test_accuracy']
    print(f"    Best accuracy: {best_accuracy:.4f}")
    
    return results


def save_results_to_file(results, filename="time_series_analysis_results.json"):
    """Save analysis results to a JSON file."""
    
    print(f"\nSaving results to {filename}...")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    # Recursively convert all numpy types
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy_types(obj)
    
    # Convert and save
    converted_results = recursive_convert(results)
    
    with open(filename, 'w') as f:
        json.dump(converted_results, f, indent=2, default=str)
    
    print(f"Results saved successfully to {filename}")


def main():
    """Main function to run the time series analysis example."""
    
    print("TIME SERIES ANALYSIS EXAMPLE")
    print("=" * 60)
    print("This example demonstrates comprehensive time series analysis")
    print("capabilities including basic statistical methods and advanced")
    print("modeling techniques.")
    print("=" * 60)
    
    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = SimulationAnalyzer("simulation.db", random_seed=42)
    
    # Simulation ID to analyze
    simulation_id = 1
    
    try:
        # 1. Basic time series analysis
        temporal_results = demonstrate_basic_time_series_analysis(analyzer, simulation_id)
        
        # 2. Advanced time series modeling
        advanced_results = demonstrate_advanced_time_series_modeling(analyzer, simulation_id)
        
        # 3. Complete analysis workflow
        complete_results = demonstrate_complete_analysis(analyzer, simulation_id)
        
        # 4. Save results
        save_results_to_file(complete_results)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("  • temporal_analysis_sim_1.png - Basic time series visualization")
        print("  • advanced_time_series_models_sim_1.png - Advanced modeling visualization")
        print("  • time_series_analysis_results.json - Complete analysis results")
        print("  • analysis_results/simulation_1_analysis.json - Standard analysis output")
        print("  • analysis_results/reproducibility_report_sim_1.json - Reproducibility report")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("Please check that:")
        print("  1. The database file exists and is accessible")
        print("  2. The simulation ID exists in the database")
        print("  3. There is sufficient data for analysis (minimum 20 points for basic, 50 for advanced)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)