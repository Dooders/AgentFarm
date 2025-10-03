# Additional Tools

## Overview

AgentFarm includes a rich set of additional tools and utilities that enhance the simulation, analysis, and research experience. These tools support interactive exploration, performance optimization, advanced research workflows, and evolutionary analysis.

## Key Capabilities

### Interactive Notebooks
- **Jupyter Integration**: Interactive data exploration and analysis
- **Visualization Tools**: Rich plotting and charting capabilities
- **Live Updates**: Real-time simulation monitoring
- **Custom Widgets**: Interactive controls for parameter tuning

### Benchmarking Suite
- **Performance Testing**: Comprehensive performance benchmarks
- **Profiling Tools**: Identify bottlenecks and optimize code
- **Comparison Tools**: Compare performance across versions
- **Automated Reports**: Generate detailed performance reports

### Research Tools
- **Analysis Modules**: Specialized tools for academic research
- **Statistical Tools**: Advanced statistical analysis capabilities
- **Data Export**: Export data in research-friendly formats
- **Citation Tools**: Generate citations and references

### Genome Embeddings
- **Machine Learning**: ML tools for analyzing genetic evolution
- **Similarity Analysis**: Find similar genomes in population
- **Clustering**: Group genomes by characteristics
- **Visualization**: Visualize genome space and evolution

## Interactive Notebooks

### Jupyter Integration

Use Jupyter notebooks for interactive analysis:

```python
# Install Jupyter support
# pip install jupyter ipywidgets matplotlib

# In Jupyter notebook
from farm.notebooks import SimulationExplorer

# Create interactive explorer
explorer = SimulationExplorer('simulation.db')

# Interactive visualization
explorer.show_population_timeline(interactive=True)
explorer.show_agent_details(agent_id=1)
explorer.show_heatmap(step=500)
```

### Interactive Widgets

Create custom interactive controls:

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

### Live Monitoring

Monitor simulations in real-time:

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

### Data Exploration

Explore simulation data interactively:

```python
from farm.notebooks import DataExplorer

# Create explorer
explorer = DataExplorer('simulation.db')

# Interactive data table
explorer.show_agent_table(
    step=500,
    sortable=True,
    filterable=True
)

# Statistical summary
explorer.show_statistical_summary()

# Correlation matrix
explorer.show_correlation_matrix()
```

## Benchmarking Suite

### Performance Benchmarks

Run comprehensive benchmarks:

```bash
# Run all benchmarks
python -m benchmarks.run_benchmarks

# Run specific benchmark
python -m benchmarks.run_benchmarks --benchmark spatial_index

# Run with custom configuration
python -m benchmarks.run_benchmarks --num-agents 10000 --num-steps 1000
```

### Benchmark Configuration

Configure benchmark parameters:

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

### Profiling Tools

Profile code to identify bottlenecks:

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

### Memory Profiling

Track memory usage:

```python
from farm.profiling import memory_profile

@memory_profile
def run_simulation(config):
    simulation = Simulation(config)
    return simulation.run()

# Run with memory profiling
result = run_simulation(config)

# View memory usage
print(f"Peak memory: {result.peak_memory_mb} MB")
print(f"Memory leaked: {result.memory_leaked_mb} MB")
```

### Benchmark Comparison

Compare performance across versions:

```python
from benchmarks import BenchmarkComparison

# Load benchmark results
comparison = BenchmarkComparison([
    'benchmarks/v1.0/results.json',
    'benchmarks/v1.1/results.json',
    'benchmarks/v1.2/results.json'
])

# Compare performance
comparison.plot_comparison(
    benchmark='agent_update',
    metric='time_ms'
)

# Statistical analysis
comparison.statistical_comparison(
    test='t_test',
    alpha=0.05
)
```

## Research Tools

### Analysis Modules

Specialized analysis tools:

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

### Statistical Tools

Advanced statistical analysis:

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

### Data Export for Research

Export data in research-friendly formats:

```python
from farm.research.export import ResearchExporter

exporter = ResearchExporter('simulation.db')

# Export to R-compatible format
exporter.export_to_r(
    output_file='data.RData',
    include_metadata=True
)

# Export to MATLAB format
exporter.export_to_matlab(
    output_file='data.mat',
    version='7.3'
)

# Export to NetCDF (for large datasets)
exporter.export_to_netcdf(
    output_file='data.nc',
    compression='zlib'
)
```

### Citation and Documentation

Generate citations and documentation:

```python
from farm.research import CitationGenerator

# Generate citation
citation = CitationGenerator.generate_citation(
    simulation_id='sim_001',
    format='bibtex'
)

# Generate methods section
methods = CitationGenerator.generate_methods_section(
    simulation_id='sim_001',
    include_parameters=True,
    style='academic'
)

# Generate data availability statement
data_statement = CitationGenerator.generate_data_statement(
    simulation_id='sim_001',
    repository='zenodo'
)
```

## Genome Embeddings

### Machine Learning for Genomes

Analyze genetic evolution using ML:

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

### Similarity Analysis

Find similar genomes:

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

### Genome Clustering

Group genomes by characteristics:

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
    cluster_genomes = clusterer.get_cluster_genomes(cluster_id)
    characteristics = clusterer.analyze_cluster(cluster_id)
    print(f"Cluster {cluster_id}: {characteristics}")
```

### Genome Space Visualization

Visualize genome evolution:

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

# Animate evolution
embedder.create_evolution_animation(
    output_file='genome_evolution.mp4',
    fps=10
)
```

### Genetic Architecture Analysis

Analyze genetic architecture:

```python
from farm.research.genome_embeddings import GeneticArchitectureAnalyzer

analyzer = GeneticArchitectureAnalyzer(embeddings, genomes)

# Identify important genes
important_genes = analyzer.identify_important_genes(
    method='feature_importance',
    top_n=20
)

# Analyze epistasis
epistasis = analyzer.analyze_epistasis(
    gene_pairs=important_genes
)

# Compute heritability
heritability = analyzer.estimate_heritability(
    trait='fitness',
    method='genomic_relm'
)
```

## Command-Line Tools

### CLI Interface

Use command-line tools for automation:

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

### Batch Processing

Process multiple simulations:

```bash
# Batch run simulations
farm batch-run --config-dir configs/ --output-dir results/ --parallel 4

# Batch analysis
farm batch-analyze --input-dir results/ --output-dir reports/
```

### Workflow Automation

Create automated workflows:

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

### Configuration Helpers

Utilities for configuration management:

```python
from farm.utils import ConfigUtils

# Merge configurations
merged = ConfigUtils.merge_configs(base_config, override_config)

# Validate configuration
errors = ConfigUtils.validate_config(config)

# Generate default configuration
default = ConfigUtils.generate_default_config()

# Convert between formats
ConfigUtils.yaml_to_json('config.yaml', 'config.json')
```

### Data Utilities

Helper functions for data manipulation:

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

- [Benchmarks README](../../benchmarks/README.md)
- [Benchmarking Report](../../benchmarks/reports/0.1.0/benchmark_profiling_summary_report.md)
- [Module Overview](../module_overview.md)
- [User Guide](../user-guide.md)
- [Developer Guide](../developer-guide.md)

## Examples

For practical examples:
- [Usage Examples](../usage_examples.md)
- [Memory Agent Experiments](../experiments/memory_agent/README.md)
- [Interactive Notebooks](../../examples/notebooks/)
