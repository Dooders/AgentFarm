# Health and Resource Dynamics Analysis Guide

This guide explains how to interpret the results from the health and resource dynamics analysis system, which examines relationships between agent health, resources, and behavior patterns.

## Overview

The analysis consists of three main components:
1. Cross-correlation analysis between resources and health
2. Fourier analysis to detect periodic patterns
3. Multivariate regression model for health prediction

**Data Source**: The analysis uses simulation data retrieved via `DataRetriever.get_simulation_data()`, which provides comprehensive agent state information including health, resources, age, and action history.

## Cross-Correlation Analysis

### Purpose
Identifies relationships between resource availability and health levels, including any time delays between resource changes and health impacts.

### Interpreting Results
```python
corr_results = {
    'peak_lag': 3,         # Time steps between cause and effect
    'peak_correlation': 0.8 # Strength of relationship (-1 to 1)
}
```

- **Peak Lag**
  - Positive: Resource changes lead health changes
  - Negative: Health changes lead resource changes
  - Zero: Changes occur simultaneously

- **Peak Correlation**
  - Close to 1: Strong positive relationship
  - Close to -1: Strong negative relationship
  - Close to 0: Weak or no relationship

### Visualization (cross_correlation.png)
- X-axis: Time lag in simulation steps
- Y-axis: Correlation coefficient
- Peak indicates strongest relationship point

## Fourier Analysis

### Purpose
Detects periodic patterns in health and resource levels over time.

### Interpreting Results
```python
fourier_results = {
    'metric': ['health', 'resources'],
    'dominant_frequency': [0.1, 0.05],    # Cycles per time step
    'period': [10, 20]                    # Time steps per cycle
}
```

- **Dominant Frequency**
  - Higher: More rapid oscillations
  - Lower: Slower cycles
  
- **Period**
  - Number of time steps to complete one cycle
  - Useful for identifying natural rhythms in the system

### Visualization (fourier_analysis.png)
- Top plot: Health frequency components
- Bottom plot: Resource frequency components
- Peaks indicate dominant cycles

## Health Prediction Model

### Purpose
Predicts agent health based on multiple factors using multivariate regression.

### Model Features
1. Age
2. Resource level
3. Recent action count (actions in last 10 time steps)

### Interpreting Results
```python
model_results = {
    'r2_score': 0.75,  # Prediction accuracy (0-1)
    'feature_importance': {
        'age': -0.3,           # Negative impact
        'resource_level': 0.6,  # Positive impact
        'recent_actions': -0.2  # Negative impact
    }
}
```

- **R-squared Score**
  - Close to 1: Model explains most health variation
  - Close to 0: Poor predictive power

- **Feature Importance**
  - Positive coefficients: Features that increase health
  - Negative coefficients: Features that decrease health
  - Larger magnitudes: Stronger effects

## Common Patterns to Look For

### 1. Resource-Health Lag
- Typical lag: 2-5 time steps
- Larger lags suggest slower health response
- Very short lags indicate immediate effects

### 2. Periodic Behaviors
- Resource gathering cycles
- Health recovery patterns
- Population-driven fluctuations

### 3. Critical Thresholds
- Points where health sharply declines
- Resource levels that trigger behavior changes
- Age-related health impacts

## Using the Analysis

### 1. Policy Optimization
```python
# Example of using analysis to adjust policies
if corr_results['peak_lag'] > 5:
    # Increase urgency of resource gathering
    agent.gather_threshold *= 0.8
```

### 2. Risk Assessment
```python
# Example of risk evaluation
def assess_health_risk(agent):
    predicted_health = model.predict([
        agent.age,
        agent.resource_level,
        agent.recent_actions
    ])
    return predicted_health < critical_threshold
```

### 3. System Tuning
```python
# Example of system parameter adjustment
if fourier_results['period']['resources'] < 10:
    # Adjust resource regeneration rate
    environment.resource_regen_rate *= 1.2
```

## Best Practices

1. **Regular Analysis**
   - Run analysis at fixed intervals
   - Track changes in patterns over time
   - Compare across different scenarios

2. **Context Consideration**
   - Consider environmental conditions
   - Account for population dynamics
   - Note any special events or changes

3. **Validation**
   - Cross-validate predictions
   - Compare with observed behaviors
   - Test on different agent types

## Troubleshooting

### Common Issues

1. **Weak Correlations**
   - Check for non-linear relationships
   - Consider additional variables
   - Verify data quality

2. **Unclear Periodicity**
   - Increase sampling frequency
   - Check for overlapping cycles
   - Consider external influences

3. **Poor Predictions**
   - Add relevant features
   - Check for outliers
   - Consider non-linear models

## Additional Analysis Features

The implementation also includes:
- **Health Strategy Clustering**: Identifies different agent health management patterns using K-means clustering
- **Comprehensive Visualizations**: Time series plots, distribution analysis, and age impact studies
- **Rolling Correlation Analysis**: Shows how health-resource relationships change over time

## References

1. Time Series Analysis: Box, G. E., et al. (2015)
2. Fourier Analysis in Biological Systems: Smith, D. R. (2018)
3. Multi-Agent Health Systems: Johnson, M. K. (2020) 