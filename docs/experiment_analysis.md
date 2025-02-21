# FARM Experiment Analysis

This document outlines the available experiment analysis capabilities in the FARM system, focusing on comparative analysis across multiple simulations.

## Available Analysis Tools

### 1. Comparative Analysis (`farm/analysis/comparative_analysis.py`)

Compares metrics across multiple simulations including:

- Population dynamics
- Resource utilization 
- Health and reward distributions
- Birth/death rates
- Agent lifespans
- Population vs simulation duration
- Population vs agent age

Key functions:
```python
compare_simulations(search_path: str, analysis_path: str)
plot_comparative_metrics(df: pd.DataFrame, output_dir: str)
```

### 2. Action Type Distribution (`farm/analysis/action_type_distribution.py`)

Analyzes action patterns across simulations:

- Action frequency distribution
- Action-reward correlations
- Chi-square tests for action-success relationships
- Temporal action patterns

Key functions:
```python
calculate_action_frequencies(actions_df: pd.DataFrame)
calculate_action_correlations(actions_df: pd.DataFrame)
```

### 3. Health Resource Dynamics (`farm/analysis/health_resource_dynamics.py`)

Studies health and resource relationships:

- Cross-correlation analysis
- Fourier analysis of cycles
- Health prediction modeling
- Strategy clustering

Key functions:
```python
analyze_health_resource_dynamics(db_path: str)
analyze_health_strategies(data: pd.DataFrame)
```

### 4. Learning Experience Analysis (`farm/analysis/learning_experience.py`)

Evaluates learning outcomes:

- Reward vs loss relationships
- State change impact
- Action selection patterns
- Learning efficiency over time

Key functions:
```python
analyze_learning_experiences(db_path: str)
plot_learning_metrics(rl_data: pd.DataFrame)
```

### 5. Reproduction Analysis (`farm/analysis/reproduction_diagnosis.py`)

Examines reproduction patterns:

- Success/failure rates
- Resource level impacts
- Generational analysis
- Population sustainability

Key functions:
```python
analyze_reproduction_patterns() -> Dict
plot_diagnostics(metrics: Dict)
```

### 6. Reward Efficiency Analysis (`farm/analysis/reward_efficiency.py`)

Studies reward optimization:

- Action-specific rewards
- Agent type efficiency
- Resource-reward relationships
- Strategy effectiveness

Key functions:
```python
analyze_reward_efficiency(data: pd.DataFrame)
reward_efficiency_pipeline(db_path: str)
```

## Usage Examples

### Basic Comparative Analysis
```python
from farm.analysis.comparative_analysis import compare_simulations

# Compare multiple simulations
compare_simulations(
    search_path="experiments/initial_experiments/databases",
    analysis_path="experiments/analysis_results"
)
```

### Action Pattern Analysis
```python
from farm.analysis.action_type_distribution import main as analyze_actions

# Analyze action patterns
analyze_actions(engine)  # SQLAlchemy engine connected to database
```

### Health-Resource Analysis
```python
from farm.analysis.health_resource_dynamics import analyze_health_resource_dynamics

# Analyze health and resource dynamics
analyze_health_resource_dynamics("simulations/simulation.db")
```

## Output Formats

Analysis results are provided in multiple formats:

1. **Visualizations**: PNG files with plots and charts
2. **CSV Data**: Raw numerical data for further analysis
3. **JSON Reports**: Structured analysis results
4. **Markdown Reports**: Human-readable summaries
5. **HTML Dashboards**: Interactive visualization dashboards

## Key Metrics Tracked

1. **Population Dynamics**
   - Mean/median/mode population
   - Population stability
   - Growth/decline rates

2. **Resource Management**
   - Resource efficiency
   - Distribution patterns
   - Sustainability metrics

3. **Agent Performance**
   - Health trends
   - Reward accumulation
   - Action effectiveness

4. **Learning Outcomes**
   - Strategy adaptation
   - Reward optimization
   - Knowledge transfer

5. **System Stability**
   - Population sustainability
   - Resource balance
   - Agent type distribution

## Future Enhancements

Planned analysis capabilities:

1. **Network Analysis**
   - Agent interaction patterns
   - Resource flow networks
   - Information propagation

2. **Evolutionary Analysis**
   - Genetic diversity metrics
   - Adaptation rates
   - Fitness landscapes

3. **Environmental Impact**
   - Resource depletion patterns
   - Carrying capacity analysis
   - Sustainability metrics

4. **Agent Behavior**
   - Decision tree analysis
   - Strategy classification
   - Behavioral clustering

5. **Performance Optimization**
   - Bottleneck identification
   - Resource utilization
   - System efficiency metrics
