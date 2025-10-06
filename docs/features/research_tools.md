# Research Tools

![Feature](https://img.shields.io/badge/feature-research%20tools-red)

## Table of Contents

1. [Overview](#overview)
   - [Why Research Tools Matter](#why-research-tools-matter)
2. [Core Capabilities](#core-capabilities)
   - [1. Parameter Sweep Experiments](#1-parameter-sweep-experiments)
     - [Basic Parameter Sweeps](#basic-parameter-sweeps)
     - [Automated Parameter Space Exploration](#automated-parameter-space-exploration)
     - [Adaptive Parameter Sweeps](#adaptive-parameter-sweeps)
   - [2. Comparative Analysis Framework](#2-comparative-analysis-framework)
     - [Multi-Simulation Comparison](#multi-simulation-comparison)
     - [Cross-Experiment Analysis](#cross-experiment-analysis)
     - [Effect Size Analysis](#effect-size-analysis)
     - [Clustering Analysis](#clustering-analysis)
   - [3. Experiment Replication Tools](#3-experiment-replication-tools)
     - [Reproducibility Management](#reproducibility-management)
     - [Exact Replication](#exact-replication)
     - [Validation and Verification](#validation-and-verification)
     - [Cross-Validation of Results](#cross-validation-of-results)
   - [4. Structured Logging System](#4-structured-logging-system)
     - [Basic Logging](#basic-logging)
     - [Context Management](#context-management)
     - [Specialized Context Managers](#specialized-context-managers)
     - [Performance Logging](#performance-logging)
     - [Agent-Specific Logging](#agent-specific-logging)
     - [Log Sampling](#log-sampling)
3. [Advanced Research Workflows](#advanced-research-workflows)
   - [Multi-Stage Experiments](#multi-stage-experiments)
   - [A/B Testing Framework](#ab-testing-framework)
4. [Example: Complete Research Workflow](#example-complete-research-workflow)
5. [Additional Resources](#additional-resources)
   - [Documentation](#documentation)
   - [Examples](#examples)
   - [Tools](#tools)
6. [Support](#support)

---

## Overview

AgentFarm provides a comprehensive suite of research tools designed to support rigorous scientific investigation through agent-based modeling. From parameter sweeps to comparative analysis and experiment replication, these tools enable researchers to conduct systematic studies with confidence in their results.

### Why Research Tools Matter

Effective research tools enable:
- **Systematic Investigation**: Structured parameter space exploration
- **Reproducible Results**: Consistent experiment replication
- **Statistical Rigor**: Proper experimental design and analysis
- **Efficient Workflow**: Automated experiment management
- **Publication Quality**: Professional-grade reporting and visualization

---

## Core Capabilities

### 1. Parameter Sweep Experiments

Systematically explore parameter spaces to identify optimal configurations and understand system sensitivities.

#### Basic Parameter Sweeps

```python
from farm.runners.experiment_runner import ExperimentRunner
from farm.config import SimulationConfig
import itertools

# Define parameter ranges
population_sizes = [10, 25, 50, 100]
resource_rates = [0.01, 0.02, 0.03]
agent_ratios = [0.2, 0.5, 0.8]  # Proportion of System agents

# Create base configuration
base_config = SimulationConfig(
    width=100,
    height=100,
    max_steps=1000,
    seed=42
)

# Initialize experiment runner
experiment = ExperimentRunner(
    base_config=base_config,
    experiment_name="parameter_sweep_study"
)

# Generate all parameter combinations
variations = []
for pop_size, regen_rate, ratio in itertools.product(
    population_sizes, resource_rates, agent_ratios
):
    num_system = int(pop_size * ratio)
    num_independent = pop_size - num_system
    
    variations.append({
        'system_agents': num_system,
        'independent_agents': num_independent,
        'resource_regen_rate': regen_rate
    })

print(f"Running {len(variations)} parameter combinations...")

# Run parameter sweep
experiment.run_iterations(
    num_iterations=len(variations),
    config_variations=variations,
    num_steps=1000
)

# Generate comparative report
experiment.generate_report()

# Results show which parameters lead to:
# - Highest population sustainability
# - Best resource efficiency
# - Optimal agent type balance
```

#### Automated Parameter Space Exploration

Use grid search or random search for large parameter spaces:

```python
from sklearn.model_selection import ParameterGrid
import numpy as np

# Define parameter grid
param_grid = {
    'width': [50, 100, 150],
    'height': [50, 100, 150],
    'system_agents': [10, 25, 50],
    'independent_agents': [10, 25, 50],
    'resource_regen_rate': [0.01, 0.02, 0.03],
    'initial_resources': [100, 200, 300],
    'learning_rate': [0.0001, 0.001, 0.01],
    'epsilon_decay': [0.99, 0.995, 0.999]
}

# Generate all combinations (can be very large!)
all_combinations = list(ParameterGrid(param_grid))
print(f"Total combinations: {len(all_combinations)}")

# Random sampling for large spaces
sample_size = 100
sampled_params = np.random.choice(
    all_combinations, 
    size=sample_size, 
    replace=False
)

# Run sampled parameter sweep
for i, params in enumerate(sampled_params):
    print(f"Running configuration {i+1}/{sample_size}")
    
    config = SimulationConfig(**params)
    result = run_simulation(config)
    
    # Store results with parameters
    results_db.store_result(params, result)

# Analyze results to find optimal parameters
best_params = results_db.find_optimal_configuration(
    metric='population_sustainability',
    direction='maximize'
)
```

#### Adaptive Parameter Sweeps

Use Bayesian optimization to efficiently explore parameter space:

> **Note**: The `BayesianParameterOptimizer` class is planned for a future release. Currently, parameter sweeps can be implemented using grid search or random sampling with external optimization libraries.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np

class BayesianParameterOptimizer:
    """Efficiently explore parameter space using Bayesian optimization."""
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize optimizer.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        
        # Initialize Gaussian Process
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6
        )
        
        self.X_observed = []
        self.y_observed = []
        
    def suggest_next_params(self) -> Dict[str, float]:
        """Suggest next parameter configuration to try."""
        if len(self.X_observed) < 5:
            # Random exploration for first few iterations
            return {
                name: np.random.uniform(bounds[0], bounds[1])
                for name, bounds in self.param_bounds.items()
            }
        
        # Fit GP on observed data
        self.gp.fit(self.X_observed, self.y_observed)
        
        # Acquisition function: Expected Improvement
        best_y = max(self.y_observed)
        
        # Sample candidates
        n_candidates = 1000
        candidates = np.random.uniform(
            low=[b[0] for b in self.param_bounds.values()],
            high=[b[1] for b in self.param_bounds.values()],
            size=(n_candidates, len(self.param_names))
        )
        
        # Predict mean and std for candidates
        mu, sigma = self.gp.predict(candidates, return_std=True)
        
        # Expected improvement
        with np.errstate(divide='warn'):
            imp = mu - best_y
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        # Select best candidate
        best_idx = np.argmax(ei)
        best_candidate = candidates[best_idx]
        
        return {
            name: value 
            for name, value in zip(self.param_names, best_candidate)
        }
    
    def observe(self, params: Dict[str, float], result: float):
        """Record observation from experiment."""
        param_vector = [params[name] for name in self.param_names]
        self.X_observed.append(param_vector)
        self.y_observed.append(result)

# Use Bayesian optimization
param_bounds = {
    'system_agents': (10, 100),
    'independent_agents': (10, 100),
    'resource_regen_rate': (0.01, 0.05),
    'learning_rate': (0.0001, 0.01)
}

optimizer = BayesianParameterOptimizer(param_bounds)

# Run optimization loop
for iteration in range(50):
    # Get next parameters to try
    params = optimizer.suggest_next_params()
    
    # Run simulation
    config = SimulationConfig(**params)
    result = run_simulation(config)
    
    # Evaluate performance
    score = evaluate_performance(result)
    
    # Record observation
    optimizer.observe(params, score)
    
    print(f"Iteration {iteration}: Score = {score:.3f}")
    print(f"  Parameters: {params}")

# Get best found parameters
best_idx = np.argmax(optimizer.y_observed)
best_params = {
    name: optimizer.X_observed[best_idx][i]
    for i, name in enumerate(optimizer.param_names)
}

print(f"\nBest parameters found:")
for name, value in best_params.items():
    print(f"  {name}: {value:.4f}")
```

---

### 2. Comparative Analysis Framework

Compare multiple simulations or experiments to identify patterns and significant differences.

#### Multi-Simulation Comparison

```python
from farm.database.simulation_comparison import SimulationComparator
import pandas as pd

# Initialize comparator
comparator = SimulationComparator("experiment.db")

# Load multiple simulations
simulation_ids = [1, 2, 3, 4, 5]
sim_data = comparator.load_simulation_data(simulation_ids)

# Compare population outcomes
population_comparison = comparator.compare_population_dynamics(
    simulation_ids=simulation_ids
)

print("Population Dynamics Comparison:")
print(population_comparison.describe())

# Statistical tests for differences
from scipy.stats import kruskal, mannwhitneyu

# Test for significant differences
final_populations = [
    get_final_population(sim_id) 
    for sim_id in simulation_ids
]

# Kruskal-Wallis test (non-parametric ANOVA)
h_stat, p_value = kruskal(*final_populations)
print(f"\nKruskal-Wallis Test:")
print(f"  H-statistic: {h_stat:.3f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print("  Significant differences found between simulations")
else:
    print("  No significant differences found")

# Pairwise comparisons
print("\nPairwise Comparisons:")
for i in range(len(simulation_ids)):
    for j in range(i+1, len(simulation_ids)):
        u_stat, p = mannwhitneyu(
            final_populations[i], 
            final_populations[j]
        )
        print(f"  Sim {simulation_ids[i]} vs Sim {simulation_ids[j]}: "
              f"p = {p:.4f}")
```

#### Cross-Experiment Analysis

Compare entire experimental conditions:

```python
from farm.core.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("experiments")

# Register experiments
exp1_id = tracker.register_experiment(
    "low_resources",
    config_low_resources,
    "exp1.db"
)

exp2_id = tracker.register_experiment(
    "high_resources",
    config_high_resources,
    "exp2.db"
)

exp3_id = tracker.register_experiment(
    "variable_resources",
    config_variable_resources,
    "exp3.db"
)

# Compare experiments
comparison = tracker.compare_experiments(
    [exp1_id, exp2_id, exp3_id],
    metrics=[
        'final_population',
        'avg_lifespan',
        'resource_efficiency',
        'cooperation_rate'
    ]
)

# Generate comparison report
tracker.generate_comparison_report(
    [exp1_id, exp2_id, exp3_id],
    output_file="experiment_comparison.html"
)

# Statistical comparison
comparison_summary = tracker.generate_comparison_summary(
    [exp1_id, exp2_id, exp3_id]
)

print(comparison_summary)
```

#### Effect Size Analysis

Measure practical significance of differences:

```python
from scipy.stats import cohen_d

def calculate_effect_sizes(group1, group2):
    """Calculate effect sizes between two groups."""
    
    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(group1) + np.var(group2)) / 2
    )
    cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Interpretation
    if abs(cohen_d) < 0.2:
        interpretation = "negligible"
    elif abs(cohen_d) < 0.5:
        interpretation = "small"
    elif abs(cohen_d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        'cohens_d': cohen_d,
        'interpretation': interpretation,
        'mean_diff': np.mean(group1) - np.mean(group2),
        'std_diff': np.std(group1) - np.std(group2)
    }

# Compare experimental groups
control_results = get_experiment_results("control")
treatment_results = get_experiment_results("treatment")

effect_sizes = calculate_effect_sizes(
    control_results['final_population'],
    treatment_results['final_population']
)

print(f"Effect Size Analysis:")
print(f"  Cohen's d: {effect_sizes['cohens_d']:.3f}")
print(f"  Interpretation: {effect_sizes['interpretation']}")
print(f"  Mean difference: {effect_sizes['mean_diff']:.2f}")
```

#### Clustering Analysis

Group similar simulations together:

```python
from analysis.simulation_comparison import SimulationComparator

comparator = SimulationComparator("multi_sim.db")

# Cluster simulations by outcomes
clustering_results = comparator.cluster_simulations(
    sim_data,
    max_clusters=5
)

print(f"Identified {clustering_results['optimal_clusters']} clusters")
print(f"Silhouette score: {clustering_results['silhouette_score']:.3f}")

# Analyze cluster characteristics
for cluster_id, characteristics in clustering_results['cluster_profiles'].items():
    print(f"\nCluster {cluster_id}:")
    for feature, value in characteristics.items():
        print(f"  {feature}: {value:.2f}")

# Visualize clusters
comparator.visualize_clusters(
    clustering_results,
    output_path="cluster_visualization.png"
)
```

---

### 3. Experiment Replication Tools

Ensure reproducible research through proper experiment management and replication.

#### Reproducibility Management

```python
from analysis.reproducibility import ReproducibilityManager, create_reproducibility_report

# Initialize reproducibility manager
repro_manager = ReproducibilityManager(seed=42)

# Capture environment
env_info = repro_manager.get_environment_info()

print("Environment Information:")
print(f"  Python: {env_info['python_version']}")
print(f"  Platform: {env_info['platform']}")
print(f"  Timestamp: {env_info['timestamp']}")

# Run analysis with reproducibility tracking
analysis_params = {
    'simulation_db': 'simulation.db',
    'analysis_types': ['population', 'resources', 'behavior'],
    'confidence_level': 0.95
}

# Create analysis hash for identification
analysis_hash = repro_manager.create_analysis_hash(analysis_params)
print(f"\nAnalysis Hash: {analysis_hash}")

# Run analysis
results = run_comprehensive_analysis(analysis_params)

# Create reproducibility report
report_path = create_reproducibility_report(
    analysis_params=analysis_params,
    results=results,
    output_path="reproducibility_report.json"
)

print(f"Reproducibility report saved: {report_path}")
```

#### Exact Replication

Replicate experiments with identical conditions:

```python
def replicate_experiment(original_config: Dict, num_replications: int = 10):
    """Run exact replications of an experiment."""
    
    results = []
    
    for i in range(num_replications):
        print(f"Running replication {i+1}/{num_replications}")
        
        # Create identical configuration
        config = SimulationConfig(**original_config)
        
        # Use different seed for each replication
        config.seed = original_config['seed'] + i
        
        # Run simulation
        result = run_simulation(config)
        results.append(result)
    
    # Analyze consistency
    final_populations = [r['surviving_agents'] for r in results]
    
    consistency_report = {
        'mean': np.mean(final_populations),
        'std': np.std(final_populations),
        'min': np.min(final_populations),
        'max': np.max(final_populations),
        'cv': np.std(final_populations) / np.mean(final_populations),  # Coefficient of variation
        'replications': num_replications
    }
    
    print("\nReplication Consistency:")
    print(f"  Mean final population: {consistency_report['mean']:.2f}")
    print(f"  Std deviation: {consistency_report['std']:.2f}")
    print(f"  Coefficient of variation: {consistency_report['cv']:.2%}")
    
    # Test for significant variation
    if consistency_report['cv'] > 0.15:  # 15% threshold
        print("  WARNING: High variation between replications!")
    else:
        print("  Good consistency across replications")
    
    return results, consistency_report

# Replicate an experiment
original_config = {
    'width': 100,
    'height': 100,
    'system_agents': 25,
    'independent_agents': 25,
    'resource_regen_rate': 0.02,
    'max_steps': 1000,
    'seed': 42
}

replications, consistency = replicate_experiment(original_config, num_replications=20)
```

#### Validation and Verification

Validate analysis results for correctness:

```python
from analysis.reproducibility import AnalysisValidator

validator = AnalysisValidator()

# Validate single analysis
validation = validator.validate_analysis_result(
    analysis_type='population_dynamics',
    result=population_analysis_results
)

print("Validation Results:")
print(f"  Valid: {validation['valid']}")
print(f"  Checks passed: {validation['checks_passed']}/{validation['total_checks']}")

if not validation['valid']:
    print("\nErrors found:")
    for error in validation['errors']:
        print(f"  - {error}")

if validation['warnings']:
    print("\nWarnings:")
    for warning in validation['warnings']:
        print(f"  - {warning}")

# Validate complete analysis
complete_validation = validator.validate_complete_analysis(all_results)

print(f"\nOverall Validation:")
print(f"  Valid: {complete_validation['overall_valid']}")
print(f"  Success rate: {complete_validation['summary']['success_rate']:.1%}")
```

#### Cross-Validation of Results

Verify results are consistent across runs:

```python
def cross_validate_experiment(config, num_folds=5):
    """Cross-validate experiment results."""
    
    fold_results = []
    
    for fold in range(num_folds):
        # Run with different seed
        fold_config = config.copy()
        fold_config.seed = config.seed + fold * 1000
        
        result = run_simulation(fold_config)
        fold_results.append(result)
    
    # Compare consistency
    metrics = ['surviving_agents', 'total_resources_consumed', 'avg_lifespan']
    
    comparison = {}
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        comparison[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'cv': np.std(values) / np.mean(values),
            'values': values
        }
    
    # Validate consistency
    for metric, stats in comparison.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  CV: {stats['cv']:.2%}")
        
        if stats['cv'] < 0.1:
            print(f"  ✓ Excellent consistency")
        elif stats['cv'] < 0.2:
            print(f"  ✓ Good consistency")
        else:
            print(f"  ⚠ High variation - investigate!")
    
    return comparison
```

---

### 4. Structured Logging System

Professional-grade logging with rich contextual information.

#### Basic Logging

```python
from farm.utils import configure_logging, get_logger

# Configure logging
configure_logging(
    environment="development",  # or "production"
    log_level="INFO"
)

# Get logger for your module
logger = get_logger(__name__)

# Event-style logging (preferred)
logger.info("simulation_started", 
           simulation_id="sim_001",
           num_agents=100, 
           num_steps=1000)

logger.debug("agent_action",
            agent_id="agent_042",
            action_type="gather",
            reward=0.5)

logger.warning("low_resources",
              simulation_id="sim_001",
              resource_level=15,
              threshold=20)

logger.error("simulation_failed",
            error_type="OutOfMemory",
            memory_used_gb=8.5)
```

#### Context Management

Add context automatically to all log messages:

```python
from farm.utils import log_context, bind_context, unbind_context

# Scoped context (automatic cleanup)
with log_context(simulation_id="sim_001", experiment="param_sweep"):
    logger.info("step_completed", step=100)
    # Logs include simulation_id and experiment automatically
    
    run_simulation_step()
    # All logs in this block have context

# Global context (persists)
bind_context(user_id="researcher_001", lab="complex_systems")
logger.info("analysis_started")  # Includes user_id and lab

# Remove context
unbind_context("user_id")

# Logger-level binding
sim_logger = logger.bind(component="spatial_index", version="2.0")
sim_logger.info("index_updated")  # Always includes component and version
```

#### Specialized Context Managers

```python
from farm.utils import log_simulation, log_step, log_experiment

# Simulation context
with log_simulation(simulation_id="sim_001", num_agents=100, seed=42):
    logger.info("simulation_initialized")
    # Run simulation
    # All logs include simulation context

# Step context
for step in range(1000):
    with log_step(step_number=step, simulation_id="sim_001"):
        # Process step
        logger.debug("processing_agents", active_agents=75)
        # Logs include step_number and simulation_id

# Experiment context
with log_experiment(experiment_id="exp_001", name="resource_study"):
    logger.info("experiment_started")
    # Run multiple simulations
    # All logs include experiment context
```

#### Performance Logging

Track performance automatically:

```python
from farm.utils import log_performance

@log_performance(operation_name="spatial_query", slow_threshold_ms=100.0)
def find_nearby_agents(position, radius):
    """Find agents within radius of position."""
    # Implementation
    return nearby_agents

# Automatically logs:
# - Execution time
# - Slow operations (> threshold)
# - Operation name and parameters
```

#### Agent-Specific Logging

Specialized logger for agent actions:

```python
from farm.utils import AgentLogger

agent_logger = AgentLogger(
    agent_id="agent_042",
    agent_type="SystemAgent"
)

# Log actions
agent_logger.log_action(
    action_type="gather",
    success=True,
    reward=0.75,
    resource_amount=5
)

# Log interactions
agent_logger.log_interaction(
    interaction_type="share",
    target_agent="agent_013",
    resource_amount=3
)

# Log state changes
agent_logger.log_state_change(
    old_state="exploring",
    new_state="gathering",
    reason="resource_detected"
)

# Log death
agent_logger.log_death(
    cause="starvation",
    final_resources=0,
    lifespan=245
)
```

#### Log Sampling

Reduce log volume for high-frequency events:

```python
from farm.utils import LogSampler

# Sample 10% of events
sampler = LogSampler(sample_rate=0.1)

for agent in agents:
    if sampler.should_log():
        logger.debug("agent_processed",
                    agent_id=agent.id,
                    position=agent.position)
    
    # Process agent (always happens)
    process_agent(agent)

# Adaptive sampling based on conditions
adaptive_sampler = LogSampler(sample_rate=0.01)  # 1% default

# Increase sampling when errors occur
if error_rate > 0.05:
    adaptive_sampler.set_sample_rate(0.5)  # 50% during errors
```

---

## Advanced Research Workflows

### Multi-Stage Experiments

Run complex multi-stage experimental protocols:

```python
class MultiStageExperiment:
    """Manager for multi-stage experimental protocols."""
    
    def __init__(self, name: str):
        self.name = name
        self.stages = []
        self.results = {}
        
    def add_stage(self, stage_name: str, config: SimulationConfig,
                  num_iterations: int):
        """Add a stage to the experiment."""
        self.stages.append({
            'name': stage_name,
            'config': config,
            'num_iterations': num_iterations
        })
        
    def run(self):
        """Execute all stages sequentially."""
        for stage in self.stages:
            print(f"\n=== Running Stage: {stage['name']} ===")
            
            stage_results = []
            for i in range(stage['num_iterations']):
                result = run_simulation(stage['config'])
                stage_results.append(result)
            
            self.results[stage['name']] = stage_results
            
            # Analyze stage
            self._analyze_stage(stage['name'], stage_results)
            
    def _analyze_stage(self, stage_name: str, results: List):
        """Analyze results from a stage."""
        populations = [r['surviving_agents'] for r in results]
        
        print(f"\nStage {stage_name} Results:")
        print(f"  Mean population: {np.mean(populations):.2f}")
        print(f"  Std deviation: {np.std(populations):.2f}")
        print(f"  Range: [{np.min(populations)}, {np.max(populations)}]")

# Use multi-stage experiment
experiment = MultiStageExperiment("progressive_difficulty")

# Stage 1: Easy conditions
experiment.add_stage(
    "easy",
    SimulationConfig(resource_regen_rate=0.05, max_steps=500),
    num_iterations=10
)

# Stage 2: Medium conditions
experiment.add_stage(
    "medium",
    SimulationConfig(resource_regen_rate=0.03, max_steps=750),
    num_iterations=10
)

# Stage 3: Hard conditions
experiment.add_stage(
    "hard",
    SimulationConfig(resource_regen_rate=0.01, max_steps=1000),
    num_iterations=10
)

experiment.run()
```

### A/B Testing Framework

Compare two experimental conditions rigorously:

```python
class ABTest:
    """Framework for A/B testing in simulations."""
    
    def __init__(self, name: str, control_config: SimulationConfig,
                 treatment_config: SimulationConfig):
        self.name = name
        self.control_config = control_config
        self.treatment_config = treatment_config
        
    def run(self, num_samples: int = 30, alpha: float = 0.05):
        """Run A/B test with statistical analysis."""
        
        print(f"Running A/B Test: {self.name}")
        print(f"  Sample size per group: {num_samples}")
        print(f"  Significance level: {alpha}")
        
        # Run control group
        print("\nRunning control group...")
        control_results = [
            run_simulation(self.control_config)
            for _ in range(num_samples)
        ]
        
        # Run treatment group
        print("Running treatment group...")
        treatment_results = [
            run_simulation(self.treatment_config)
            for _ in range(num_samples)
        ]
        
        # Analyze
        return self._analyze_results(
            control_results,
            treatment_results,
            alpha
        )
    
    def _analyze_results(self, control, treatment, alpha):
        """Statistical analysis of A/B test."""
        from scipy.stats import ttest_ind, mannwhitneyu
        
        # Extract metric
        control_values = [r['surviving_agents'] for r in control]
        treatment_values = [r['surviving_agents'] for r in treatment]
        
        # T-test
        t_stat, t_p = ttest_ind(control_values, treatment_values)
        
        # Mann-Whitney U (non-parametric)
        u_stat, u_p = mannwhitneyu(control_values, treatment_values)
        
        # Effect size
        effect_size = (np.mean(treatment_values) - np.mean(control_values)) / \
                     np.sqrt((np.var(control_values) + np.var(treatment_values)) / 2)
        
        # Report
        report = {
            'control_mean': np.mean(control_values),
            'treatment_mean': np.mean(treatment_values),
            'difference': np.mean(treatment_values) - np.mean(control_values),
            't_statistic': t_stat,
            't_p_value': t_p,
            'u_statistic': u_stat,
            'u_p_value': u_p,
            'effect_size': effect_size,
            'significant': u_p < alpha
        }
        
        print(f"\n=== A/B Test Results ===")
        print(f"Control mean: {report['control_mean']:.2f}")
        print(f"Treatment mean: {report['treatment_mean']:.2f}")
        print(f"Difference: {report['difference']:.2f}")
        print(f"Effect size (Cohen's d): {report['effect_size']:.3f}")
        print(f"p-value: {report['u_p_value']:.4f}")
        
        if report['significant']:
            print("✓ Statistically significant difference found")
        else:
            print("✗ No significant difference found")
        
        return report

# Run A/B test
test = ABTest(
    name="resource_regeneration",
    control_config=SimulationConfig(resource_regen_rate=0.02),
    treatment_config=SimulationConfig(resource_regen_rate=0.03)
)

results = test.run(num_samples=30)
```

---

## Example: Complete Research Workflow

```python
#!/usr/bin/env python3
"""
Complete research workflow example.
Demonstrates parameter sweep, comparison, and replication.
"""

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.runners.experiment_runner import ExperimentRunner
from farm.core.experiment_tracker import ExperimentTracker
from farm.utils import configure_logging, get_logger, log_experiment
from analysis.reproducibility import ReproducibilityManager, create_reproducibility_report
import itertools

def main():
    # Step 1: Configure logging
    configure_logging(environment="research", log_level="INFO")
    logger = get_logger(__name__)
    
    with log_experiment(experiment_id="param_sweep_001", name="resource_study"):
        logger.info("research_workflow_started")
        
        # Step 2: Initialize reproducibility
        repro_manager = ReproducibilityManager(seed=42)
        env_info = repro_manager.get_environment_info()
        logger.info("environment_captured", env_info=env_info)
        
        # Step 3: Define parameter space
        param_space = {
            'resource_regen_rates': [0.01, 0.02, 0.03],
            'agent_ratios': [0.25, 0.50, 0.75],
            'population_sizes': [50, 75, 100]
        }
        
        # Step 4: Run parameter sweep
        logger.info("starting_parameter_sweep", param_space=param_space)
        
        tracker = ExperimentTracker("research_experiments")
        experiment_ids = []
        
        for regen_rate, ratio, pop_size in itertools.product(
            param_space['resource_regen_rates'],
            param_space['agent_ratios'],
            param_space['population_sizes']
        ):
            # Create configuration
            config = SimulationConfig(
                width=100,
                height=100,
                system_agents=int(pop_size * ratio),
                independent_agents=int(pop_size * (1-ratio)),
                resource_regen_rate=regen_rate,
                max_steps=1000,
                seed=42
            )
            
            # Run experiment
            exp_name = f"regen{regen_rate}_ratio{ratio}_pop{pop_size}"
            logger.info("running_experiment", name=exp_name)
            
            result = run_simulation(config)
            
            # Register with tracker
            exp_id = tracker.register_experiment(
                name=exp_name,
                config=config.to_dict(),
                db_path=result['db_path']
            )
            experiment_ids.append(exp_id)
            
            logger.info("experiment_completed",
                       experiment_id=exp_id,
                       final_population=result['surviving_agents'])
        
        # Step 5: Comparative analysis
        logger.info("running_comparative_analysis",
                   num_experiments=len(experiment_ids))
        
        comparison = tracker.compare_experiments(
            experiment_ids,
            metrics=['final_population', 'avg_lifespan', 'resource_efficiency']
        )
        
        logger.info("comparison_completed",
                   shape=comparison.shape)
        
        # Step 6: Generate report
        logger.info("generating_comparison_report")
        
        tracker.generate_comparison_report(
            experiment_ids,
            output_file="research_comparison_report.html"
        )
        
        # Step 7: Replication study
        logger.info("starting_replication_study")
        
        # Find best configuration
        best_exp_id = experiment_ids[comparison['final_population'].idxmax()]
        best_config = tracker.metadata['experiments'][best_exp_id]['config']
        
        # Replicate 20 times
        replication_results = []
        for i in range(20):
            rep_config = SimulationConfig(**best_config)
            rep_config.seed = 42 + i
            
            result = run_simulation(rep_config)
            replication_results.append(result)
        
        # Analyze consistency
        populations = [r['surviving_agents'] for r in replication_results]
        cv = np.std(populations) / np.mean(populations)
        
        logger.info("replication_analysis",
                   mean=np.mean(populations),
                   std=np.std(populations),
                   cv=cv)
        
        # Step 8: Create reproducibility report
        logger.info("creating_reproducibility_report")
        
        analysis_params = {
            'param_space': param_space,
            'num_experiments': len(experiment_ids),
            'num_replications': 20,
            'best_config': best_config
        }
        
        report_path = create_reproducibility_report(
            analysis_params=analysis_params,
            results={'comparison': comparison.to_dict(),
                    'replications': replication_results},
            output_path="reproducibility_report.json"
        )
        
        logger.info("research_workflow_completed",
                   report_path=str(report_path))
        
        print(f"\n=== Research Workflow Complete ===")
        print(f"Total experiments: {len(experiment_ids)}")
        print(f"Comparison report: research_comparison_report.html")
        print(f"Reproducibility report: {report_path}")

if __name__ == "__main__":
    main()
```

---

## Additional Resources

### Documentation
- [Experiment QuickStart](ExperimentQuickStart.md) - Getting started guide
- [Logging Guide](logging_guide.md) - Comprehensive logging documentation
- [Logging Quick Reference](LOGGING_QUICK_REFERENCE.md) - Quick logging reference
- [Data Analysis](data_visualization.md) - Analysis and visualization tools

### Examples
- [Parameter Sweep Example](examples/parameter_sweep.py)
- [Comparative Analysis Example](examples/comparative_analysis.py)
- [Replication Study Example](examples/replication_study.py)

### Tools
- [Experiment Runner](runners/experiment_runner.py) - Experiment execution
- [Experiment Tracker](core/experiment_tracker.py) - Experiment management
- [Simulation Comparator](database/simulation_comparison.py) - Comparison tools

---

## Support

For research tools questions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [Full documentation index](README.md)
- **Examples**: Check `examples/` directory for more samples

---

**Ready to conduct rigorous research?** Start with [Parameter Sweeps](#parameter-sweep-experiments) or explore our [Comparative Analysis](#comparative-analysis-framework) tools!
