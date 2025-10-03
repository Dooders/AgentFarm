# Data & Visualization

## Overview

AgentFarm provides comprehensive data collection, analysis, and visualization tools that transform raw simulation data into meaningful insights. The system automatically tracks an extensive array of metrics throughout simulation execution while providing flexible, powerful tools for visual exploration and automated report generation.

Understanding complex simulation dynamics requires multiple perspectives. Summary statistics might hide important details visible in time series plots. Aggregate metrics might obscure interesting individual variation visible in scatter plots. Spatial patterns might not be apparent without heatmaps showing agent movement. The platform provides all these views and more, enabling you to build a multi-faceted understanding of your simulations.

## Automated Data Collection

At the foundation is a robust automatic data collection system that runs continuously during simulations, capturing detailed information about every aspect of system state and dynamics. This ensures data is never lost and all simulations produce comparable datasets suitable for systematic analysis.

Agent-level data tracks the state of every agent at each timestep - health levels, resource inventories, spatial positions, current actions, social relationships, learning progress, and any custom attributes. This granular data enables detailed analysis of individual trajectories, identification of exceptional individuals, and understanding of within-population variation.

Population-level metrics aggregate agent data to characterize the population as a whole. The system tracks population size and demographics, summary statistics of agent attributes, diversity measures quantifying heterogeneity, and spatial distributions. These provide macro-level views essential for understanding emergent phenomena.

Environmental data captures resource levels across locations, environmental quality metrics, spatial gradients, and histories of environmental change. This enables analysis of agent-environment feedback loops, resource depletion dynamics, and how environmental heterogeneity influences population structure.

Interaction data records every interaction between agents with detailed metadata - the agents involved, interaction type, context and location, and outcomes for each participant. This comprehensive logging enables network analysis, identification of keystone individuals, and understanding of how social structure emerges and evolves.

## Real-Time Visualization

For interactive exploration and monitoring of ongoing simulations, AgentFarm provides real-time visualization that updates continuously as simulations run. This serves multiple purposes - debugging and validation, hypothesis generation as patterns suggest new questions, and presentation where live simulations create engaging demonstrations.

```python
from farm.visualization import RealTimeVisualizer

# Create visualizer
visualizer = RealTimeVisualizer(
    window_size=(800, 600),
    update_frequency=10  # Update every 10 steps
)

# Run simulation with visualization
simulation = Simulation(config)
simulation.run(visualizer=visualizer)
```

The spatial visualization displays agent positions and movements in an animated view. Agents can be color-coded by attributes like health or resources, allowing you to see at a glance how these properties distribute across space. Movement trails visualize recent trajectories, revealing spatial strategies like territoriality or migration. Resource distributions can be overlaid as heatmaps showing how agents respond to environmental heterogeneity.

Real-time metric plots display time series of key indicators updating as the simulation progresses. You can monitor population size, average fitness, resource depletion, diversity indices, and custom metrics. Multiple metrics can be plotted simultaneously to reveal relationships and tradeoffs. Interactive controls let you adjust parameters on the fly, zoom in on specific regions, pause and resume, and even modify parameters while running for exploratory experiments.

## Static Visualization

After simulations complete, comprehensive static visualization tools enable detailed analysis and creation of publication-quality figures. Static visualizations can be more sophisticated than real-time displays since computation time is less constrained.

```python
from farm.visualization import SimulationVisualizer

visualizer = SimulationVisualizer(simulation_id="sim_001")

# Spatial visualization
visualizer.plot_agent_positions(
    step=500,
    color_by='health',
    size_by='resources'
)

# Population over time
visualizer.plot_population_timeline(
    show_births=True,
    show_deaths=True
)

# Resource distribution
visualizer.plot_resource_distribution(
    resource_type='food',
    visualization_type='heatmap'
)
```

Spatial visualization tools create detailed maps of agent distributions and environmental features at specific timepoints or averaged over periods. Agents can be represented as points with size and color encoding multiple attributes. Density heatmaps reveal clustering patterns. Spatial statistics can be overlaid to test for significant deviations from randomness.

Population timeline visualizations chart how properties change over simulation courses. Line plots track single metrics while stacked area charts show how components contribute to totals. Confidence bands represent variation across replications. Event markers highlight significant occurrences like population crashes or transitions.

## Network Visualization

Given the importance of interactions in agent-based models, AgentFarm provides specialized tools for constructing and visualizing interaction networks. Network visualization transforms interaction data into graph representations where nodes represent agents and edges represent interactions, revealing social structure that might not be apparent from individual interaction records.

```python
from farm.visualization import NetworkVisualizer

net_viz = NetworkVisualizer(simulation_id="sim_001")

# Interaction network
net_viz.plot_interaction_network(
    step_range=(0, 1000),
    interaction_types=['cooperation', 'competition'],
    layout='force_directed'
)

# Social network
net_viz.plot_social_network(
    node_size_by='centrality',
    node_color_by='generation',
    show_communities=True
)
```

Interaction networks can be constructed using various criteria - all interactions, only strong or frequent interactions, or specific types like cooperation. Edge weights represent interaction frequency or strength. Node attributes like size and color encode agent properties, creating rich visualizations that integrate social structure with agent characteristics.

Social network analysis metrics are computed automatically - degree centrality measuring connectivity, betweenness centrality identifying agents that bridge groups, clustering coefficients quantifying local connectivity, and community detection revealing subgroup structure. Network layout algorithms position nodes to reveal structure using force-directed, hierarchical, circular, or spatial layouts.

## Charting and Plotting

Beyond specialized visualizations, AgentFarm provides general-purpose charting tools for creating standard statistical plots and exploratory data analysis visualizations.

```python
from farm.visualization import ChartGenerator

charts = ChartGenerator(simulation_id="sim_001")

# Multi-line chart
charts.plot_metrics_over_time(
    metrics=['avg_health', 'avg_resources', 'population'],
    normalize=True,
    show_confidence_intervals=True
)

# Distribution histograms
charts.plot_distribution(
    metric='agent_age',
    step=1000,
    bins=50,
    show_kde=True
)

# Correlation heatmap
charts.plot_correlation_heatmap(
    metrics=['health', 'resources', 'age', 'offspring'],
    method='pearson'
)
```

Time-series charts are fundamental for simulation data. Line charts track metric evolution, revealing trends, cycles, and transitions. Multi-line charts compare multiple metrics. Area charts emphasize magnitudes and can be stacked to show component contributions. Confidence bands indicate uncertainty across replications.

Distribution plots characterize spread and shape at specific timepoints. Histograms bin continuous data while kernel density estimates provide smoothed representations. Box plots compactly summarize distributions with quartiles and outliers. Violin plots combine box plots with density estimates. These are essential for understanding population heterogeneity.

Scatter plots explore relationships between variable pairs, revealing correlations and nonlinear dependencies. Points can be color-coded by third variables. Trend lines and confidence regions summarize relationships statistically. Correlation matrices use color to encode correlation strength, making patterns of covariation immediately apparent.

## Automated Report Generation

To streamline documenting and communicating results, AgentFarm includes automated report generation combining visualizations, statistical summaries, and narrative text into comprehensive documents.

```python
from farm.reporting import ReportGenerator

generator = ReportGenerator(simulation_id="sim_001")

# Generate summary report
report = generator.generate_summary_report(
    include_sections=[
        'overview',
        'population_dynamics',
        'key_metrics',
        'visualizations',
        'statistical_analysis'
    ],
    format='html'
)

report.save('reports/simulation_summary.html')
```

Summary reports provide high-level overviews suitable for quick assessment. They include key metrics and values, population dynamics visualizations, significant events, comparisons to baselines, and methodological notes. Analysis reports dive deeper with detailed statistical analysis, extensive visualizations, hypothesis tests, and discussion of findings.

Comparison reports facilitate systematic evaluation of parameter effects or treatments. They present results from multiple runs in parallel with side-by-side visualizations highlighting differences, statistical tests assessing significance, and summary tables quantifying effect sizes.

## Export Formats

AgentFarm supports exporting data in various formats to facilitate analysis with external tools and sharing with collaborators.

```python
from farm.data.export import DataExporter

exporter = DataExporter(simulation_id="sim_001")

# Export to CSV
exporter.export_to_csv(
    data_type='agent_history',
    output_path='exports/agent_data.csv'
)

# Export to JSON
exporter.export_to_json(
    data_type='all',
    output_path='exports/simulation_data.json'
)

# Export to HDF5 (for large datasets)
exporter.export_to_hdf5(
    output_path='exports/simulation_data.h5',
    compression='gzip'
)
```

CSV provides universal format readable by virtually every analysis tool. JSON creates hierarchical, self-describing files preserving structure and data types. HDF5 provides efficient storage for very large datasets with compression and chunking for efficient access. Database export writes to SQLite for efficient querying using SQL.

## Interactive Dashboards

For deep exploratory analysis, AgentFarm provides interactive dashboard capabilities combining multiple coordinated visualizations with interactive controls.

```python
from farm.visualization import InteractiveDashboard
import ipywidgets as widgets

# Create interactive dashboard
dashboard = InteractiveDashboard(simulation_id="sim_001")

# Add interactive controls
@widgets.interact(
    step=widgets.IntSlider(min=0, max=1000, value=500),
    metric=widgets.Dropdown(options=['health', 'resources', 'age'])
)
def update_visualization(step, metric):
    dashboard.plot_agents(step=step, color_by=metric)
```

Jupyter notebook integration brings visualization into the rich ecosystem of interactive Python notebooks. You can create interactive widgets controlling visualization parameters, combine code and visualizations in single documents, and export notebooks as reproducible analysis documents.

Web-based dashboards provide rich interactive experiences accessible through browsers without local software installation. These include coordinated multiple views, interactive filtering, animated playback, and real-time updates for ongoing simulations.

## Performance Optimization

Visualizing large datasets presents performance challenges addressed through several strategies. Data downsampling reduces points plotted when full resolution exceeds what can be meaningfully displayed. Intelligent downsampling preserves important features like peaks and outliers while reducing volume.

Level-of-detail rendering adjusts detail based on zoom level and viewing context. When viewing entire simulations, simplified representations suffice. When zooming in, full detail becomes visible. Caching of rendered visualizations avoids redundant computation. Progressive rendering displays incremental results while completing expensive visualizations, providing immediate feedback.

## Related Documentation

For more details, see [Data System](./data-system.md), [Data Services](../data/data_services.md), [Data Retrieval](../data/data_retrieval.md), [Metrics Documentation](../metrics.md), and [Analysis Documentation](../data/analysis/Analysis.md).

## Examples

Practical examples can be found in [Usage Examples](../usage_examples.md), [Service Usage Examples](../data/service_usage_examples.md), [Experiment Analysis](../experiment_analysis.md), and [One of a Kind Experiments](../experiments/one_of_a_kind/README.md).
