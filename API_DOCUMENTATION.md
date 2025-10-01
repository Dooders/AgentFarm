# Analysis Module API Documentation

## Overview

The analysis module provides comprehensive statistical analysis capabilities for simulation data, including time series analysis, machine learning validation, and advanced statistical methods.

## Module Structure

```
analysis/
├── simulation_analysis.py      # Main analysis module
├── simulation_comparison.py    # Cross-simulation analysis
└── reproducibility.py         # Reproducibility framework
```

## Core Classes

### SimulationAnalyzer

The main analysis class providing comprehensive statistical analysis capabilities.

#### Constructor

```python
SimulationAnalyzer(db_path: str, random_seed: int = 42)
```

**Parameters:**
- `db_path` (str): Path to the SQLite database file
- `random_seed` (int): Random seed for reproducible results (default: 42)

**Returns:** SimulationAnalyzer instance

**Example:**
```python
analyzer = SimulationAnalyzer("simulation.db", random_seed=42)
```

#### Methods

##### analyze_population_dynamics(simulation_id: int) -> Dict[str, Any]

Analyzes population dynamics with statistical validation.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze

**Returns:** Dictionary containing:
- `dataframe`: Population data over time
- `statistical_analysis`: Kruskal-Wallis test, pairwise comparisons, confidence intervals
- `effect_sizes`: Cohen's d, Hedges' g, eta-squared calculations
- `power_analysis`: Statistical power analysis

**Statistical Methods:**
- Kruskal-Wallis test for multi-group comparison
- Mann-Whitney U test for pairwise comparisons
- 95% confidence intervals for mean populations
- Effect size calculations (Cohen's d, Hedges' g, eta-squared)
- Statistical power analysis

**Example:**
```python
results = analyzer.analyze_population_dynamics(simulation_id=1)
print(f"Kruskal-Wallis p-value: {results['statistical_analysis']['kruskal_wallis']['p_value']}")
```

##### analyze_resource_distribution(simulation_id: int) -> Dict[str, Any]

Analyzes resource distribution patterns with statistical validation.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze

**Returns:** Dictionary containing:
- `resource_stats`: Descriptive statistics by resource type
- `statistical_analysis`: ANOVA test, post-hoc tests, confidence intervals

**Statistical Methods:**
- One-way ANOVA for resource type comparison
- Tukey's HSD post-hoc tests
- 95% confidence intervals
- Effect size calculations

**Example:**
```python
results = analyzer.analyze_resource_distribution(simulation_id=1)
anova_result = results['statistical_analysis']['anova_test']
print(f"ANOVA p-value: {anova_result['p_value']}")
```

##### analyze_agent_interactions(simulation_id: int) -> Dict[str, Any]

Analyzes agent interaction patterns with statistical validation.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze

**Returns:** Dictionary containing:
- `interaction_patterns`: Interaction frequency by agent type
- `interaction_matrix`: Contingency table of interactions
- `statistical_analysis`: Chi-square test, interaction rates, confidence intervals

**Statistical Methods:**
- Chi-square test for independence
- Wilson score confidence intervals for interaction rates
- Effect size calculations

**Example:**
```python
results = analyzer.analyze_agent_interactions(simulation_id=1)
chi_square = results['statistical_analysis']['chi_square_test']
print(f"Chi-square p-value: {chi_square['p_value']}")
```

##### analyze_generational_survival(simulation_id: int) -> Dict[str, Any]

Analyzes generational survival patterns with statistical validation.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze

**Returns:** Dictionary containing:
- `survival_rates`: Survival rates by generation
- `statistical_analysis`: Confidence intervals, trend analysis

**Statistical Methods:**
- Wilson score confidence intervals for survival rates
- Trend analysis for survival patterns

**Example:**
```python
results = analyzer.analyze_generational_survival(simulation_id=1)
survival_ci = results['statistical_analysis']['survival_rate_ci']
```

##### identify_critical_events(simulation_id: int, significance_level: float = 0.05) -> List[Dict[str, Any]]

Identifies critical events using statistical change point detection.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze
- `significance_level` (float): Statistical significance level (default: 0.05)

**Returns:** List of critical events, each containing:
- `step`: Simulation step number
- `agent_type`: Type of agent affected
- `change_rate`: Rate of change
- `z_score`: Statistical z-score
- `p_value`: Statistical p-value
- `is_significant`: Whether the change is statistically significant

**Statistical Methods:**
- Z-score based change detection
- Configurable significance levels (0.01, 0.05, 0.1)
- Statistical significance testing

**Example:**
```python
events = analyzer.identify_critical_events(simulation_id=1, significance_level=0.01)
significant_events = [e for e in events if e['is_significant']]
print(f"Found {len(significant_events)} significant events")
```

##### analyze_temporal_patterns(simulation_id: int) -> Dict[str, Any]

Performs comprehensive time series analysis.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze

**Returns:** Dictionary containing:
- `time_series_analysis`: Analysis for each time series
- `cross_correlations`: Correlations between time series

**Statistical Methods:**
- Augmented Dickey-Fuller (ADF) test for stationarity
- KPSS test for trend stationarity
- Linear trend analysis with R²
- Seasonal decomposition
- Periodogram analysis
- Change point detection
- Autocorrelation analysis
- Cross-correlation analysis

**Example:**
```python
results = analyzer.analyze_temporal_patterns(simulation_id=1)
for series_name, analysis in results["time_series_analysis"].items():
    adf_test = analysis["stationarity"]["adf_test"]
    print(f"{series_name}: Stationary = {adf_test['is_stationary']}")
```

##### analyze_advanced_time_series_models(simulation_id: int) -> Dict[str, Any]

Performs advanced time series modeling including ARIMA, VAR, and exponential smoothing.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze

**Returns:** Dictionary containing:
- `arima_models`: ARIMA model results for each time series
- `var_model`: Vector Autoregression model results
- `exponential_smoothing`: Exponential smoothing results
- `model_comparison`: Model comparison and selection

**Statistical Methods:**
- ARIMA modeling with auto parameter selection
- Vector Autoregression (VAR) modeling
- Exponential smoothing (Simple, Holt, Holt-Winters)
- Granger causality testing
- Model comparison using AIC/BIC
- Forecasting with confidence intervals
- Model diagnostics (Ljung-Box test)

**Example:**
```python
results = analyzer.analyze_advanced_time_series_models(simulation_id=1)
for series_name, arima_result in results["arima_models"].items():
    if "error" not in arima_result:
        print(f"{series_name}: ARIMA{arima_result['model_order']}")
        print(f"  AIC: {arima_result['aic']:.2f}")
        print(f"  Forecast: {arima_result['forecast'][:5]}")
```

##### analyze_with_advanced_ml(simulation_id: int, target_variable: str = "population_dominance") -> Dict[str, Any]

Performs advanced machine learning analysis with ensemble methods.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze
- `target_variable` (str): Target variable for ML analysis (default: "population_dominance")

**Returns:** Dictionary containing:
- `feature_selection`: Results from multiple feature selection methods
- `individual_models`: Performance of individual ML models
- `ensemble_models`: Performance of ensemble models
- `hyperparameter_tuning`: Best hyperparameters and performance
- `performance_comparison`: Comprehensive performance comparison

**Statistical Methods:**
- Feature selection (Univariate, RFE, Model-based)
- Individual models (Random Forest, Gradient Boosting, Logistic Regression, SVM, Decision Tree)
- Ensemble methods (Voting, Bagging)
- Cross-validation with stratified splits
- Hyperparameter tuning with GridSearchCV
- Performance metrics (accuracy, AUC, precision, recall)

**Example:**
```python
results = analyzer.analyze_with_advanced_ml(simulation_id=1)
best_model = results["best_model"]
print(f"Best model: {best_model}")
print(f"Test accuracy: {results['individual_models'][best_model]['test_accuracy']:.3f}")
```

##### analyze_agent_decisions(simulation_id: int) -> pd.DataFrame

Analyzes patterns in agent decision-making.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze

**Returns:** DataFrame containing decision patterns and statistics

**Example:**
```python
decisions_df = analyzer.analyze_agent_decisions(simulation_id=1)
print(decisions_df.head())
```

##### run_complete_analysis(simulation_id: int, significance_level: float = 0.05) -> Dict[str, Any]

Runs a complete analysis including all available methods.

**Parameters:**
- `simulation_id` (int): ID of the simulation to analyze
- `significance_level` (float): Statistical significance level (default: 0.05)

**Returns:** Dictionary containing:
- All individual analysis results
- Validation report
- Reproducibility report
- Metadata with analysis information

**Example:**
```python
results = analyzer.run_complete_analysis(simulation_id=1, significance_level=0.05)
print(f"Analysis version: {results['metadata']['analysis_version']}")
print(f"Validation passed: {results['validation_report']['overall_valid']}")
```

#### Helper Methods

##### _calculate_effect_sizes(data1: pd.Series, data2: pd.Series) -> Dict[str, float]

Calculates comprehensive effect size measures.

**Parameters:**
- `data1` (pd.Series): First data series
- `data2` (pd.Series): Second data series

**Returns:** Dictionary containing:
- `cohens_d`: Cohen's d effect size
- `hedges_g`: Hedges' g effect size
- `glass_delta`: Glass's delta effect size
- `cles`: Common Language Effect Size
- `eta_squared`: Eta-squared effect size
- Interpretations for effect sizes

##### _calculate_power_analysis(data1: pd.Series, data2: pd.Series, p_value: float) -> Dict[str, Any]

Performs statistical power analysis.

**Parameters:**
- `data1` (pd.Series): First data series
- `data2` (pd.Series): Second data series
- `p_value` (float): Observed p-value

**Returns:** Dictionary containing:
- `observed_power`: Observed statistical power
- `power_interpretation`: Interpretation of power
- `effect_size`: Effect size calculation
- `type_ii_error_rate`: Type II error rate
- `power_for_effects`: Power for different effect sizes
- `sample_size_for_80_power`: Sample size needed for 80% power

## SimulationComparison

Class for comparing results across multiple simulations.

### Constructor

```python
SimulationComparison(db_path: str)
```

### Methods

#### cluster_simulations(df: pd.DataFrame, max_clusters: Optional[int] = None) -> Dict[str, Any]

Clusters simulations based on their characteristics.

**Parameters:**
- `df` (pd.DataFrame): Simulation data
- `max_clusters` (Optional[int]): Maximum number of clusters

**Returns:** Dictionary containing:
- `optimal_k`: Optimal number of clusters
- `silhouette_score`: Silhouette score for validation
- `cluster_labels`: Cluster assignments
- `cluster_centers`: Cluster centroids

#### build_predictive_model(df: pd.DataFrame, target_column: str = "population_dominance") -> Dict[str, Any]

Builds predictive models for simulation outcomes.

**Parameters:**
- `df` (pd.DataFrame): Simulation data
- `target_column` (str): Target variable for prediction

**Returns:** Dictionary containing:
- `classification_report`: Model performance metrics
- `feature_importance`: Feature importance scores
- `performance_summary`: Cross-validation results

## Reproducibility Framework

### ReproducibilityManager

Manages reproducibility aspects of analysis.

#### Constructor

```python
ReproducibilityManager(random_seed: Optional[int] = None)
```

#### Methods

##### get_run_metadata(analysis_params: Dict[str, Any], data_hash: str = "") -> Dict[str, Any]

Returns metadata for the current analysis run.

##### get_analysis_hash(params: Dict[str, Any], data_hash: str = "") -> str

Generates a hash for the analysis based on parameters and data.

### AnalysisValidator

Validates the consistency and quality of analysis results.

#### Methods

##### validate_complete_analysis(results: Dict[str, Any]) -> Dict[str, Any]

Performs comprehensive validation of all analysis results.

### create_reproducibility_report(analysis_params: Dict[str, Any], analysis_results: Dict[str, Any], output_path: Path) -> Path

Creates a JSON report containing all information necessary to reproduce an analysis.

## Data Models

### SimulationStepModel

Represents a single step in a simulation.

**Attributes:**
- `simulation_id`: ID of the simulation
- `step_number`: Step number in the simulation
- `system_agents`: Number of system agents
- `independent_agents`: Number of independent agents
- `control_agents`: Number of control agents
- `total_agents`: Total number of agents
- `resource_efficiency`: Resource efficiency metric
- `average_agent_health`: Average agent health
- `average_reward`: Average reward

### AgentModel

Represents an individual agent.

**Attributes:**
- `agent_id`: Unique agent identifier
- `agent_type`: Type of agent (system, independent, control)
- `health`: Agent health value
- `resources`: Agent resource value
- `reward`: Agent reward value

### ActionModel

Represents an action taken by an agent.

**Attributes:**
- `simulation_id`: ID of the simulation
- `step_number`: Step number when action occurred
- `agent_id`: ID of the agent taking the action
- `action_type`: Type of action
- `action_target_id`: ID of the action target

### ResourceModel

Represents resource data at a simulation step.

**Attributes:**
- `simulation_id`: ID of the simulation
- `step_number`: Step number
- `resource_type`: Type of resource
- `amount`: Amount of resource

## Error Handling

The analysis module includes comprehensive error handling:

### Insufficient Data Errors
- Minimum data requirements for each analysis type
- Graceful degradation when data is insufficient
- Clear error messages with requirements

### Model Convergence Errors
- Handling of model fitting failures
- Fallback to simpler models when needed
- Error reporting with diagnostic information

### Statistical Test Errors
- Handling of edge cases in statistical tests
- Robust error handling for invalid data
- Clear error messages for troubleshooting

## Performance Considerations

### Memory Usage
- Efficient data loading and processing
- Memory management for large datasets
- Chunked processing for very large simulations

### Computational Efficiency
- Optimized statistical calculations
- Parallel processing where applicable
- Caching of intermediate results

### Scalability
- Handles simulations with thousands of steps
- Efficient database queries
- Optimized visualization generation

## Dependencies

### Required Packages
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization
- `statsmodels`: Advanced statistical models
- `scikit-learn`: Machine learning algorithms
- `sqlalchemy`: Database ORM

### Optional Packages
- `psutil`: System memory information (for reproducibility)

## Configuration

### Database Configuration
- SQLite database path
- Connection parameters
- Query optimization settings

### Analysis Configuration
- Random seed for reproducibility
- Significance levels for statistical tests
- Visualization settings
- Output directory configuration

## Output Formats

### JSON Results
- Structured analysis results
- Metadata and validation information
- Reproducibility reports

### Visualization Files
- High-resolution PNG files (300 DPI)
- Professional styling and formatting
- Comprehensive multi-panel layouts

### Log Files
- Detailed analysis logs
- Error and warning messages
- Performance metrics

## Best Practices

### Data Preparation
- Ensure sufficient data points for analysis
- Handle missing values appropriately
- Validate data quality before analysis

### Statistical Interpretation
- Consider practical significance alongside statistical significance
- Report confidence intervals
- Validate model assumptions

### Performance Optimization
- Use appropriate data sampling for large datasets
- Monitor memory usage during analysis
- Optimize database queries

### Reproducibility
- Set random seeds for consistent results
- Document analysis parameters
- Save reproducibility reports

## Troubleshooting

### Common Issues
- Insufficient data for analysis
- Model convergence failures
- Memory limitations
- Missing dependencies

### Solutions
- Check data requirements
- Try simpler models
- Optimize memory usage
- Install missing packages

### Support
- Check documentation for detailed examples
- Review error messages for specific guidance
- Validate input data format and quality