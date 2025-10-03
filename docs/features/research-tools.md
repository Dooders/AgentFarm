# Research Tools

## Overview

AgentFarm provides a comprehensive suite of research tools designed specifically for academic and scientific research. These tools enable systematic experimentation, reproducible results, rigorous statistical analysis, and comprehensive logging for tracking simulation dynamics. By providing infrastructure for common research tasks, AgentFarm allows you to focus on scientific questions rather than research infrastructure.

The design emphasizes reproducibility, recognizing that computational experiments must be as reproducible as laboratory experiments. The tools support systematic exploration of parameter spaces, appropriate replication strategies, and statistical rigor through significance testing and effect size estimation. Together, these capabilities promote scientific best practices and produce trustworthy, publishable results.

## Parameter Sweep Experiments

Parameter sweeps systematically vary parameters to understand their effects on outcomes. AgentFarm provides sophisticated infrastructure for designing, executing, and analyzing these experiments, handling the logistics of generating parameter combinations, managing resources, and organizing results.

Grid search exhaustively explores all combinations of discrete parameter values. When you specify ranges for parameters, the system generates the Cartesian product, creating a complete factorial design. This guarantees comprehensive coverage and can reveal interaction effects between parameters that one-at-a-time variations miss.

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

The computational expense of grid search grows exponentially with parameters, making it impractical for high-dimensional spaces. For these scenarios, random search samples parameter combinations randomly from specified distributions. Surprisingly, random search often finds good parameter regions more quickly in high dimensions because it doesn't waste effort on nearby points that often yield similar results.

Adaptive sampling intelligently focuses effort on interesting parameter regions. Rather than committing to all combinations upfront, adaptive methods iteratively select parameters based on results obtained so far. Bayesian optimization builds a statistical model of how parameters affect outcomes and uses it to identify values likely to produce extreme or interesting results.

The infrastructure handles parallel execution automatically, distributing runs across available resources. You specify the degree of parallelism, and the system manages work distribution, monitors progress, handles failures, and collects results systematically.

## Comparative Analysis

Understanding results requires comparing outcomes across different conditions, parameters, or modeling choices. AgentFarm provides a comprehensive comparative analysis framework that facilitates rigorous, statistically sound comparisons.

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

Multi-run comparison loads results from multiple simulations and compares them systematically. The system aligns data across simulations, computes summary statistics for each condition, identifies metrics showing substantial differences, and visualizes comparisons through side-by-side plots and effect size displays.

Statistical testing distinguishes genuine effects from random variation. AgentFarm implements tests appropriate for different data types and designs. T-tests compare means between two conditions. Mann-Whitney U tests provide nonparametric alternatives when distributional assumptions are questionable. ANOVA extends comparison to multiple groups. Chi-square tests assess categorical data. These include appropriate corrections for multiple comparisons when testing many hypotheses simultaneously.

Beyond testing whether differences exist, effect size estimation quantifies the magnitude of differences. Statistical significance depends partly on sample size - large samples can make tiny differences statistically significant. Effect sizes like Cohen's d measure practical importance. AgentFarm automatically computes appropriate effect sizes with confidence intervals, providing a complete picture of both statistical significance and practical importance.

Sensitivity analysis quantifies how outputs depend on inputs. Rather than comparing discrete conditions, sensitivity analysis treats parameters as continuous and measures the rate of change of outputs with respect to inputs. Sobol indices decompose output variance into components attributable to each parameter and their interactions, revealing which parameters most strongly influence outcomes.

## Experiment Replication

Reproducibility stands as a cornerstone of scientific research. Computational research offers unique opportunities for perfect reproducibility but requires careful engineering. AgentFarm prioritizes reproducibility through deterministic simulations, comprehensive provenance tracking, validation tools, and replication management.

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
```

Deterministic simulations ensure that running a simulation multiple times with the same configuration produces identical results. This is achieved through careful control of random number generation using specified seeds, deterministic floating-point arithmetic, and fixed iteration orders. The ability to reproduce simulations exactly is invaluable for debugging, verifying that code changes don't alter behavior, and allowing collaborators to reproduce your results.

Configuration management tracks all parameters and settings, storing complete configurations alongside results. This ensures you can always reconstruct how simulations were configured, compare configurations across experiments, and share configurations with collaborators. The system supports versioning so you can track how experimental designs evolve, annotate configurations with descriptions, and tag them with metadata.

Batch replication facilitates running many replications of each configuration, essential for understanding stochastic variation. The system manages generating unique random seeds for each replication while maintaining an overall experimental seed for reproducibility, executes replications in parallel, monitors progress, and handles failures gracefully. Results are automatically aggregated to compute means, confidence intervals, and summary statistics.

## Structured Logging

Logging provides a continuous record of simulation execution, invaluable for debugging, understanding dynamics, and providing audit trails. AgentFarm implements professional-grade structured logging through structlog, providing rich contextual information, machine-readable output, flexible formatting, and performance optimization.

```python
from farm.utils import configure_logging, get_logger

# Configure logging
configure_logging(
    environment="development",
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

Structured logging treats log events as data structures with named fields rather than unstructured text. Each event includes the message along with structured context like simulation ID, timestep, agent ID, and action type. This makes logs far more useful for automated analysis, enables powerful querying and filtering, and allows consistent formatting across the codebase.

The system provides multiple output formats. Console logging for development uses color-coding and human-readable formatting. JSON logging for production creates machine-readable logs suitable for automated analysis. Plain text provides compatibility with traditional tools. The format can be selected through configuration without code modification.

Contextual logging automatically includes relevant context without requiring explicit passing to every logging call. Context bindings established at simulation or agent scope are automatically included in subsequent log messages. This dramatically reduces boilerplate while ensuring messages include sufficient context.

Performance optimization ensures comprehensive logging doesn't slow simulations unacceptably. The system implements lazy evaluation where expensive formatting only occurs if messages will actually be output. Log level filtering removes debug messages in production without incurring formatting cost. Sampling allows logging a fraction of high-frequency events, providing visibility without overwhelming storage.

## Log Analysis

The value of logging is realized through analysis tools that extract insights from log data.

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

Metric extraction parses logs to compute derived metrics not explicitly tracked during simulation. Log mining reveals aspects of system behavior that might not be captured by explicit instrumentation. Anomaly detection identifies unusual events or patterns indicating problems or interesting phenomena. Log summarization condenses detailed logs into high-level overviews for quick assessment and reporting.

## Experiment Database

For research involving many simulations, organizing and querying results efficiently becomes essential. AgentFarm provides an experiment database that stores results from all simulations in a queryable repository.

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

The database maintains complete records including full parameter configurations, summary statistics and key metrics, paths to detailed outputs, and metadata about execution environment. Query capabilities let you find simulations matching criteria, retrieve all simulations exploring a parameter, identify outliers with unusual outcomes, and compare simulations across parameter ranges.

## Benchmarking Suite

For researchers concerned with performance or developers optimizing code, AgentFarm includes comprehensive benchmarking and profiling tools.

```bash
# Run all benchmarks
python -m benchmarks.run_benchmarks

# Run specific benchmark
python -m benchmarks.run_benchmarks --benchmark spatial_index

# Run with custom configuration
python -m benchmarks.run_benchmarks --num-agents 10000 --num-steps 1000
```

The benchmarking suite includes standardized performance tests covering all major components. Benchmarks can be run across configurations to understand how performance scales, compared across versions to detect regressions, and used to evaluate different algorithms. Profiling tools provide detailed analysis of where time is spent, tracking line-level execution time, function call counts, and memory allocation.

## Best Practices

Effective use of research tools requires attention to experimental design and statistical methodology. Control variables carefully, use adequate replication for statistical power, randomize appropriately to prevent biases, and pilot test before large-scale experiments.

Statistical analysis should use methods appropriate for your data and questions. Report effect sizes along with p-values. Correct for multiple comparisons when testing many hypotheses. Use confidence intervals to characterize uncertainty. These fundamentals distinguish publishable research from exploratory analysis.

## Related Documentation

For detailed information, see [Experiments Documentation](../experiments.md), [Experiment Quick Start](../ExperimentQuickStart.md), [Experiment Analysis](../experiment_analysis.md), [Logging Guide](../logging_guide.md), [Logging Quick Reference](../LOGGING_QUICK_REFERENCE.md), and [Deterministic Simulations](../deterministic_simulations.md).

## Examples

Practical examples can be found in [Logging Examples](../../examples/logging_examples.py), [One of a Kind Experiments](../experiments/one_of_a_kind/README.md), [Memory Agent Experiments](../experiments/memory_agent/README.md), and [Benchmarks README](../../benchmarks/README.md).
