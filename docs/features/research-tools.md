# Research Tools

## Overview

AgentFarm provides a comprehensive suite of research tools designed specifically for academic and scientific research. These tools enable systematic experimentation, rigorous analysis, reproducible results, and advanced logging for tracking simulation dynamics.

## Key Capabilities

### Parameter Sweep Experiments
- **Grid Search**: Systematically explore parameter combinations
- **Random Search**: Efficiently sample parameter spaces
- **Adaptive Sampling**: Focus on interesting parameter regions
- **Multi-Dimensional Sweeps**: Vary multiple parameters simultaneously

### Comparative Analysis Framework
- **Multi-Run Comparisons**: Compare results across simulation runs
- **Statistical Testing**: Apply rigorous statistical tests
- **Sensitivity Analysis**: Identify influential parameters
- **Variance Analysis**: Understand sources of variation

### Experiment Replication Tools
- **Deterministic Simulations**: Ensure reproducible results with fixed random seeds
- **Configuration Management**: Track and version experiment configurations
- **Result Verification**: Validate replication accuracy
- **Batch Processing**: Run multiple replications efficiently

### Structured Logging System
- **Rich Contextual Logs**: Professional-grade logging with structlog
- **Machine-Readable Output**: JSON logs for automated analysis
- **Multiple Formats**: Console (colored), JSON, and plain text formats
- **Performance Optimization**: Log sampling to reduce overhead
- **Security**: Automatic censoring of sensitive data

## Parameter Sweep Experiments

### Grid Search

Systematically explore parameter combinations:

```python
from farm.experiments import ParameterSweep, GridSearch

# Define parameter ranges
parameters = {
    'num_agents': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'mutation_rate': [0.001, 0.01, 0.1]
}

# Create grid search
grid_search = GridSearch(parameters)

# Run experiments
results = []
for params in grid_search:
    simulation = Simulation(params)
    result = simulation.run()
    results.append({
        'parameters': params,
        'outcome': result.final_population,
        'metrics': result.metrics
    })
```

### Random Search

Efficiently sample parameter spaces:

```python
from farm.experiments import RandomSearch

# Define parameter distributions
param_distributions = {
    'num_agents': ('uniform', 50, 200),
    'learning_rate': ('log_uniform', 0.001, 0.1),
    'mutation_rate': ('log_uniform', 0.0001, 0.1)
}

# Create random search
random_search = RandomSearch(
    param_distributions,
    n_samples=100,
    random_seed=42
)

# Run experiments
results = random_search.run_experiments(
    simulation_factory=Simulation,
    n_parallel=4
)
```

### Adaptive Parameter Sampling

Focus on interesting regions of parameter space:

```python
from farm.experiments import AdaptiveSampler

# Create adaptive sampler
sampler = AdaptiveSampler(
    parameters=parameters,
    objective='maximize',
    metric='final_population'
)

# Iteratively sample and refine
for iteration in range(10):
    # Get next parameter set
    params = sampler.suggest()
    
    # Run simulation
    result = Simulation(params).run()
    
    # Update sampler
    sampler.update(params, result.final_population)

# Get best parameters
best_params = sampler.get_best_parameters()
```

## Comparative Analysis

### Multi-Run Comparison

Compare multiple simulation runs:

```python
from farm.analysis import ComparativeAnalysis

# Load simulation results
simulations = [
    'sim_baseline',
    'sim_high_learning',
    'sim_low_mutation'
]

# Create comparative analysis
analysis = ComparativeAnalysis(simulations)

# Compare key metrics
comparison = analysis.compare_metrics(
    metrics=['population', 'avg_fitness', 'diversity'],
    statistical_tests=['t_test', 'anova', 'mann_whitney']
)

# Generate comparison report
report = analysis.generate_report(
    include_visualizations=True,
    include_statistics=True
)
```

### Statistical Testing

Apply rigorous statistical tests:

```python
from farm.analysis import StatisticalTester

tester = StatisticalTester()

# Compare two conditions
t_stat, p_value = tester.t_test(
    group1=baseline_results,
    group2=treatment_results,
    alternative='two-sided'
)

# ANOVA for multiple groups
f_stat, p_value = tester.anova(
    *[results for results in experiment_groups]
)

# Non-parametric tests
u_stat, p_value = tester.mann_whitney_u(
    group1=baseline_results,
    group2=treatment_results
)
```

### Sensitivity Analysis

Identify influential parameters:

```python
from farm.analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer()

# Perform Sobol sensitivity analysis
sensitivity_indices = analyzer.sobol_analysis(
    parameter_ranges=parameters,
    output_metric='final_population',
    n_samples=1000
)

# Visualize sensitivity
analyzer.plot_sensitivity(
    sensitivity_indices,
    plot_type='bar'
)

# Get most influential parameters
top_parameters = analyzer.get_top_parameters(n=5)
```

## Experiment Replication

### Deterministic Simulations

Ensure reproducible results:

```python
from farm.core.simulation import Simulation

# Set random seed for reproducibility
config = SimulationConfig(
    random_seed=42,
    deterministic=True
)

# Run simulation
simulation1 = Simulation(config)
result1 = simulation1.run()

# Replicate with same seed
simulation2 = Simulation(config)
result2 = simulation2.run()

# Verify replication
assert result1.final_population == result2.final_population
assert np.allclose(result1.metrics, result2.metrics)
```

### Configuration Versioning

Track experiment configurations:

```python
from farm.experiments import ConfigurationManager

manager = ConfigurationManager()

# Save configuration
config_id = manager.save_configuration(
    config=config,
    description="Baseline experiment",
    tags=['baseline', 'v1.0']
)

# Load configuration
loaded_config = manager.load_configuration(config_id)

# List configurations
configs = manager.list_configurations(
    tags=['baseline'],
    date_range=('2024-01-01', '2024-12-31')
)
```

### Batch Replication

Run multiple replications efficiently:

```python
from farm.experiments import ReplicationRunner

runner = ReplicationRunner()

# Run multiple replications
replications = runner.run_replications(
    config=base_config,
    n_replications=30,
    random_seeds=range(100, 130),
    n_parallel=6
)

# Analyze replication variance
variance_analysis = runner.analyze_variance(replications)

# Get confidence intervals
ci = runner.get_confidence_intervals(
    metric='final_population',
    confidence=0.95
)
```

## Structured Logging

### Basic Logging Setup

Configure professional-grade logging:

```python
from farm.utils import configure_logging, get_logger

# Configure logging
configure_logging(
    environment="development",  # or "production"
    log_level="INFO",
    log_format="console"  # or "json", "plain"
)

# Get logger
logger = get_logger(__name__)

# Log with context
logger.info(
    "simulation_started",
    simulation_id="sim_001",
    num_agents=100,
    num_steps=1000
)
```

### Structured Context

Add rich contextual information:

```python
from farm.utils.logging import bind_contextvars

# Bind simulation context
bind_contextvars(
    simulation_id="sim_001",
    experiment_name="parameter_sweep",
    replicate=1
)

# All subsequent logs include context
logger.info("agent_created", agent_id=1, agent_type="forager")
logger.debug("action_executed", agent_id=1, action="move", target=(10, 15))
```

### Performance-Optimized Logging

Use log sampling for high-frequency events:

```python
from farm.utils.logging import SamplingLogger

# Create sampling logger
sampler = SamplingLogger(
    logger=logger,
    sample_rate=0.1  # Log 10% of events
)

# Use for high-frequency events
for step in range(10000):
    for agent in agents:
        # Only logs 10% of actions
        sampler.debug(
            "agent_step",
            step=step,
            agent_id=agent.id,
            action=agent.current_action
        )
```

### Log Analysis

Analyze structured logs:

```python
from farm.utils.logging import LogAnalyzer

analyzer = LogAnalyzer('logs/simulation.json')

# Extract metrics from logs
metrics = analyzer.extract_metrics(
    event_type='simulation_step',
    fields=['population', 'avg_health', 'resources']
)

# Find anomalies
anomalies = analyzer.find_anomalies(
    threshold=3.0  # 3 standard deviations
)

# Generate log summary
summary = analyzer.generate_summary()
```

## Data Collection and Analysis

### Experiment Database

Store and query experiment results:

```python
from farm.data.experiment_db import ExperimentDatabase

db = ExperimentDatabase('experiments.db')

# Store experiment result
db.store_experiment(
    experiment_id='exp_001',
    configuration=config,
    results=results,
    metadata={
        'researcher': 'Jane Doe',
        'hypothesis': 'Higher learning rates improve fitness',
        'date': '2024-01-15'
    }
)

# Query experiments
experiments = db.query_experiments(
    parameter_ranges={'learning_rate': (0.01, 0.1)},
    outcome_threshold={'final_population': 50}
)
```

### Advanced Analytics

Perform sophisticated analyses:

```python
from farm.analysis import AdvancedAnalyzer

analyzer = AdvancedAnalyzer()

# Causal analysis
causal_effects = analyzer.causal_analysis(
    treatment='high_learning_rate',
    outcome='fitness',
    confounders=['initial_population', 'environment_size']
)

# Clustering analysis
clusters = analyzer.cluster_experiments(
    features=['learning_rate', 'mutation_rate', 'final_population'],
    n_clusters=5,
    method='kmeans'
)

# Pattern mining
patterns = analyzer.mine_patterns(
    min_support=0.1,
    min_confidence=0.8
)
```

## Benchmarking Suite

### Performance Benchmarks

Test and optimize performance:

```python
from benchmarks import BenchmarkRunner

runner = BenchmarkRunner()

# Run benchmark suite
results = runner.run_benchmarks(
    benchmarks=['memory_db', 'spatial_index', 'agent_update'],
    configurations=[
        {'num_agents': 100},
        {'num_agents': 1000},
        {'num_agents': 10000}
    ]
)

# Generate benchmark report
runner.generate_report(
    results,
    output_path='benchmarks/report.md'
)
```

### Profiling

Identify performance bottlenecks:

```python
from farm.profiling import Profiler

with Profiler(output_file='profile.prof'):
    simulation = Simulation(config)
    simulation.run()

# Analyze profile
from farm.profiling import ProfileAnalyzer

analyzer = ProfileAnalyzer('profile.prof')
analyzer.print_top_functions(n=20)
analyzer.visualize_call_graph()
```

## Best Practices

### Experimental Design
- **Control Variables**: Keep parameters constant when testing specific factors
- **Adequate Replication**: Use sufficient replications for statistical power
- **Randomization**: Properly randomize experimental conditions
- **Documentation**: Document all experimental decisions and observations

### Data Management
- **Organized Storage**: Use consistent naming and directory structure
- **Metadata**: Store comprehensive metadata with results
- **Backups**: Regularly backup experiment data
- **Version Control**: Track code and configuration versions

### Statistical Analysis
- **Appropriate Tests**: Choose tests suitable for your data and hypotheses
- **Multiple Comparisons**: Correct for multiple testing when needed
- **Effect Sizes**: Report effect sizes, not just p-values
- **Confidence Intervals**: Provide uncertainty estimates

## Related Documentation

- [Experiments Documentation](../experiments.md)
- [Experiment Quick Start](../ExperimentQuickStart.md)
- [Experiment Analysis](../experiment_analysis.md)
- [Logging Guide](../logging_guide.md)
- [Logging Quick Reference](../LOGGING_QUICK_REFERENCE.md)
- [Benchmarking Report](../benchmarks/reports/0.1.0/benchmark_profiling_summary_report.md)
- [Deterministic Simulations](../deterministic_simulations.md)

## Examples

For practical examples:
- [Logging Examples](../../examples/logging_examples.py)
- [One of a Kind Experiments](../experiments/one_of_a_kind/README.md)
- [Memory Agent Experiments](../experiments/memory_agent/README.md)
- [Benchmarks README](../../benchmarks/README.md)
