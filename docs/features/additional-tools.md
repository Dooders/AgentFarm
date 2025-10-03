# Additional Tools

## Overview

AgentFarm includes a rich set of additional tools and utilities that enhance the simulation, analysis, and research experience. These tools support interactive exploration through Jupyter notebooks, performance optimization through benchmarking, advanced research workflows, and evolutionary analysis through genome embeddings.

These additional capabilities complement the core simulation and analysis features, providing specialized tools for particular workflows and research questions. Whether you're exploring data interactively, optimizing performance, conducting academic research, or analyzing evolutionary dynamics, AgentFarm provides tools to support your work.

## Interactive Notebooks

Jupyter notebooks provide an ideal environment for interactive data exploration and analysis. AgentFarm integrates seamlessly with Jupyter, bringing visualization and analysis capabilities into the rich ecosystem of interactive Python notebooks.

```python
# In Jupyter notebook
from farm.notebooks import SimulationExplorer

# Create interactive explorer
explorer = SimulationExplorer('simulation.db')

# Interactive visualization
explorer.show_population_timeline(interactive=True)
explorer.show_agent_details(agent_id=1)
explorer.show_heatmap(step=500)
```

Interactive widgets let you create custom controls for exploring simulation data. You can add sliders, dropdowns, and other controls that update visualizations dynamically as you adjust parameters.

```python
import ipywidgets as widgets
from IPython.display import display

# Step slider
step_slider = widgets.IntSlider(
    value=500,
    min=0,
    max=1000,
    step=10,
    description='Step:'
)

# Metric selector
metric_selector = widgets.Dropdown(
    options=['health', 'resources', 'age'],
    value='health',
    description='Metric:'
)

# Interactive update function
@widgets.interact(step=step_slider, metric=metric_selector)
def update_visualization(step, metric):
    explorer.plot_agents(step=step, color_by=metric)
```

Live monitoring capabilities let you watch simulations as they run, with charts updating in real-time. This is invaluable for debugging, understanding dynamics, and creating engaging demonstrations.

```python
from farm.notebooks import LiveMonitor

# Create live monitor
monitor = LiveMonitor()

# Start simulation with monitoring
simulation = Simulation(config)
simulation.add_callback(monitor.update)

# Display live charts
monitor.show_live_charts([
    'population',
    'avg_health',
    'avg_resources'
])

# Run simulation (charts update automatically)
simulation.run()
```

Data exploration tools provide interactive tables, statistical summaries, and correlation matrices that you can filter, sort, and drill into. This exploratory analysis helps you understand your data before conducting formal analyses.

## Benchmarking Suite

For researchers concerned with computational performance or developers optimizing code, AgentFarm includes comprehensive benchmarking tools.

```bash
# Run all benchmarks
python -m benchmarks.run_benchmarks

# Run specific benchmark
python -m benchmarks.run_benchmarks --benchmark spatial_index

# Run with custom configuration
python -m benchmarks.run_benchmarks --num-agents 10000 --num-steps 1000
```

The benchmarking suite includes standardized performance tests covering all major system components - agent updates, spatial queries, data persistence, and analysis operations. Benchmarks can be run across different configurations to understand how performance scales, compared across versions to detect regressions, and used to evaluate different algorithms or implementations.

```python
from benchmarks import BenchmarkSuite

# Create benchmark suite
suite = BenchmarkSuite()

# Add benchmarks
suite.add_benchmark(
    name='agent_update',
    function=benchmark_agent_update,
    configurations=[
        {'num_agents': 100},
        {'num_agents': 1000},
        {'num_agents': 10000}
    ]
)

# Run suite
results = suite.run()

# Generate report
suite.generate_report(results, output='benchmark_report.html')
```

Profiling tools provide detailed analysis of where time is spent. Line profilers show execution time for each line of code. Function profilers show call counts and cumulative time. Memory profilers track allocation and identify leaks. These tools are essential for understanding performance and guiding optimization.

```python
from farm.profiling import profile_function, ProfileReport

# Profile a function
@profile_function
def simulate_step(simulation):
    simulation.step()

# Run with profiling
simulation = Simulation(config)
for step in range(1000):
    simulate_step(simulation)

# Generate profile report
report = ProfileReport()
report.generate('profile_report.html')
```

Performance visualization creates graphical representations of profiling data that make bottlenecks immediately apparent. Flame graphs show the call stack with width proportional to time spent. Timeline visualizations show how execution unfolds. These help you quickly grasp performance characteristics.

## Research Tools

AgentFarm includes specialized analysis tools for academic research, providing capabilities beyond standard simulation analysis.

```python
from farm.research import (
    EvolutionaryAnalyzer,
    SocialNetworkAnalyzer,
    EcologicalAnalyzer
)

# Evolutionary analysis
evo_analyzer = EvolutionaryAnalyzer('simulation.db')
fitness_landscape = evo_analyzer.compute_fitness_landscape()
selection_pressure = evo_analyzer.estimate_selection_pressure()

# Social network analysis
social_analyzer = SocialNetworkAnalyzer('simulation.db')
centrality = social_analyzer.compute_centrality_measures()
communities = social_analyzer.detect_communities()

# Ecological analysis
eco_analyzer = EcologicalAnalyzer('simulation.db')
diversity = eco_analyzer.compute_diversity_indices()
stability = eco_analyzer.assess_stability()
```

Advanced statistical analysis tools provide capabilities beyond basic descriptive statistics. Time series analysis includes autocorrelation, stationarity testing, and spectral analysis. Distribution fitting identifies which probability distributions best match your data. Hypothesis testing includes parametric and nonparametric tests with appropriate multiple comparison corrections.

```python
from farm.research.statistics import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Time series analysis
acf = analyzer.autocorrelation(time_series, max_lag=50)
stationarity = analyzer.test_stationarity(time_series)

# Distribution fitting
best_fit = analyzer.fit_distribution(data, distributions=[
    'normal', 'lognormal', 'exponential', 'gamma'
])

# Hypothesis testing
result = analyzer.hypothesis_test(
    group1=control_data,
    group2=treatment_data,
    test='t_test',
    alternative='greater'
)
```

Data export tools prepare simulation data for analysis in other software. You can export to R-compatible formats, MATLAB, NetCDF for large datasets, and other specialized formats used in different research domains.

## Genome Embeddings

For simulations involving evolution over many generations, analyzing genetic diversity and structure becomes important. AgentFarm includes machine learning tools for genome analysis through embedding techniques.

```python
from farm.research.genome_embeddings import GenomeEmbedder

# Create embedder
embedder = GenomeEmbedder(
    embedding_dim=32,
    architecture='autoencoder'
)

# Train on genome data
genomes = load_genomes('simulation.db')
embedder.train(genomes, epochs=100)

# Embed genomes
embeddings = embedder.embed(genomes)
```

Similarity analysis using these embeddings lets you find agents with similar genetic profiles, study how genetic diversity changes over time, identify genetic bottlenecks, and understand how genetic distance relates to phenotypic differences.

```python
# Find similar genomes
target_genome = genomes[0]
similar = embedder.find_similar(
    target_genome,
    k=10,
    metric='cosine'
)

# Compute pairwise similarities
similarity_matrix = embedder.compute_similarity_matrix(
    genomes,
    metric='euclidean'
)
```

Clustering groups genomes by characteristics, revealing genetic structure in the population. Combined with temporal tracking, this shows how genetic structure evolves over generations.

```python
from farm.research.genome_embeddings import GenomeClusterer

# Create clusterer
clusterer = GenomeClusterer(embeddings)

# Cluster genomes
clusters = clusterer.cluster(
    n_clusters=5,
    method='kmeans'
)

# Analyze clusters
for cluster_id in range(5):
    characteristics = clusterer.analyze_cluster(cluster_id)
    print(f"Cluster {cluster_id}: {characteristics}")
```

Visualization of genome space uses dimensionality reduction techniques like t-SNE to plot high-dimensional genome data in two or three dimensions. This creates intuitive visual representations of genetic relationships and evolutionary trajectories.

```python
# Visualize genome space
embedder.visualize_genome_space(
    method='tsne',
    color_by='generation',
    annotate_outliers=True
)

# Visualize evolutionary trajectory
embedder.visualize_trajectory(
    start_generation=0,
    end_generation=100,
    highlight_selection_events=True
)
```

## Command-Line Tools

AgentFarm provides command-line tools for automation and scripting, making it easy to incorporate simulations into larger computational workflows.

```bash
# Run simulation
farm simulate --config config.yaml --output results/

# Analyze results
farm analyze --input results/simulation.db --report analysis.html

# Compare simulations
farm compare --simulations sim1.db sim2.db sim3.db --output comparison.pdf

# Export data
farm export --input simulation.db --format csv --output exports/
```

Batch processing capabilities let you process multiple simulations systematically. You can run many simulations in parallel, analyze entire directories of results, and generate reports for multiple experiments.

```bash
# Batch run simulations
farm batch-run --config-dir configs/ --output-dir results/ --parallel 4

# Batch analysis
farm batch-analyze --input-dir results/ --output-dir reports/
```

Workflow automation tools let you define multi-step analysis pipelines that execute automatically. This ensures consistency and reproducibility across analyses.

```python
from farm.tools import Workflow

# Define workflow
workflow = Workflow('analysis_pipeline')

workflow.add_step('load_data', load_simulation_data)
workflow.add_step('preprocess', preprocess_data)
workflow.add_step('analyze', perform_analysis)
workflow.add_step('visualize', create_visualizations)
workflow.add_step('report', generate_report)

# Execute workflow
workflow.execute(input_data='simulation.db')
```

## Utility Functions

AgentFarm includes various utility functions that simplify common tasks in simulation work.

Configuration helpers assist with configuration management - merging configurations, validating settings, generating defaults, and converting between formats.

```python
from farm.utils import ConfigUtils

# Merge configurations
merged = ConfigUtils.merge_configs(base_config, override_config)

# Validate configuration
errors = ConfigUtils.validate_config(config)

# Generate default configuration
default = ConfigUtils.generate_default_config()
```

Data utilities provide helper functions for common data manipulations - resampling time series, smoothing data, normalizing features, and other preprocessing operations.

```python
from farm.utils import DataUtils

# Resample time series
resampled = DataUtils.resample_timeseries(
    data,
    original_frequency=1,
    target_frequency=10
)

# Smooth data
smoothed = DataUtils.smooth_data(
    data,
    method='moving_average',
    window_size=10
)

# Normalize data
normalized = DataUtils.normalize(
    data,
    method='min_max',
    feature_range=(0, 1)
)
```

## Related Documentation

For detailed information, see [Benchmarks README](../../benchmarks/README.md), [Benchmarking Report](../../benchmarks/reports/0.1.0/benchmark_profiling_summary_report.md), [Module Overview](../module_overview.md), [User Guide](../user-guide.md), and [Developer Guide](../developer-guide.md).

## Examples

Practical examples can be found in [Usage Examples](../usage_examples.md), [Memory Agent Experiments](../experiments/memory_agent/README.md), and interactive notebook examples if available in your installation.
