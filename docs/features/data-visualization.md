# Data & Visualization

## Overview

AgentFarm provides comprehensive data collection, analysis, and visualization tools to help researchers understand simulation dynamics, identify patterns, and communicate findings. The system automatically tracks metrics throughout simulations and provides flexible tools for visualization and reporting.

## Key Capabilities

### Comprehensive Data Collection
- **Agent Metrics**: Track health, resources, position, actions, and custom attributes
- **Population Metrics**: Monitor population size, demographics, and distribution
- **Environmental Metrics**: Record resource levels, spatial distributions, and conditions
- **Interaction Data**: Capture agent-agent and agent-environment interactions
- **Time-Series Data**: Store temporal evolution of all metrics

### Visualization Tools
- **Real-Time Visualization**: Watch simulations as they run
- **Spatial Visualization**: View agent positions and environmental features
- **Network Visualization**: Display interaction networks and relationships
- **Statistical Plots**: Generate histograms, scatter plots, and distributions
- **Temporal Plots**: Visualize trends and patterns over time

### Charting and Plotting
- **Line Charts**: Track metrics over time
- **Heatmaps**: Visualize spatial distributions and correlations
- **Bar Charts**: Compare populations, groups, and categories
- **Scatter Plots**: Explore relationships between variables
- **Box Plots**: Analyze distributions and outliers

### Automated Report Generation
- **Summary Reports**: Generate comprehensive simulation summaries
- **Analysis Reports**: Produce detailed statistical analyses
- **Comparison Reports**: Compare multiple simulation runs
- **Custom Reports**: Create reports with custom metrics and visualizations

## Data Collection System

### Automatic Data Tracking

AgentFarm automatically collects data during simulations:

```python
from farm.core.simulation import Simulation

# Data is collected automatically
simulation = Simulation(config)
results = simulation.run()

# Access collected data
agent_data = results.agent_history
population_data = results.population_metrics
environment_data = results.environment_metrics
```

### Custom Metrics

Define and track custom metrics:

```python
from farm.data import MetricCollector

class CustomMetricCollector(MetricCollector):
    def collect(self, simulation, step):
        metrics = {}
        
        # Custom metric: average cooperation score
        metrics['avg_cooperation'] = np.mean([
            agent.cooperation_score 
            for agent in simulation.agents
        ])
        
        # Custom metric: resource inequality (Gini coefficient)
        metrics['resource_inequality'] = self.calculate_gini(
            [agent.resources for agent in simulation.agents]
        )
        
        # Custom metric: spatial clustering
        metrics['spatial_clustering'] = self.calculate_clustering(
            [agent.position for agent in simulation.agents]
        )
        
        return metrics
```

## Data Access and Retrieval

### Database Interface

Access simulation data efficiently:

```python
from farm.data.services import SimulationDataService

# Create service for data access
service = SimulationDataService(simulation_id="sim_001")

# Query agent data
agents = service.get_agents_at_step(step=500)
agent_history = service.get_agent_history(agent_id=1)

# Query population data
population_stats = service.get_population_statistics()
generation_data = service.get_generation_data(generation=10)

# Query interaction data
interactions = service.get_interactions(
    step_range=(0, 1000),
    interaction_type="cooperation"
)
```

### Data Filtering and Aggregation

Filter and aggregate data for analysis:

```python
# Filter agents by criteria
high_performers = service.filter_agents(
    min_health=80,
    min_resources=50,
    alive=True
)

# Aggregate metrics
aggregated = service.aggregate_metrics(
    metrics=['health', 'resources', 'age'],
    aggregation='mean',
    group_by='generation'
)
```

## Visualization

### Real-Time Visualization

Visualize simulations as they run:

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

### Static Visualizations

Generate static visualizations from completed simulations:

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

### Network Visualization

Visualize interaction networks:

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

## Charting and Plotting

### Time-Series Charts

Create detailed time-series visualizations:

```python
from farm.visualization import ChartGenerator

charts = ChartGenerator(simulation_id="sim_001")

# Multi-line chart
charts.plot_metrics_over_time(
    metrics=['avg_health', 'avg_resources', 'population'],
    normalize=True,
    show_confidence_intervals=True
)

# Stacked area chart
charts.plot_stacked_area(
    metrics=['food', 'water', 'shelter'],
    title='Resource Distribution Over Time'
)
```

### Statistical Plots

Generate statistical visualizations:

```python
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

# Box plot comparison
charts.plot_boxplot_comparison(
    metric='fitness',
    group_by='generation',
    show_outliers=True
)
```

### Spatial Heatmaps

Visualize spatial distributions:

```python
# Agent density heatmap
charts.plot_density_heatmap(
    step=500,
    resolution=50,
    colormap='viridis'
)

# Resource distribution heatmap
charts.plot_resource_heatmap(
    resource_type='food',
    step_range=(0, 1000),
    aggregation='mean'
)
```

## Automated Reports

### Summary Reports

Generate comprehensive simulation summaries:

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

### Analysis Reports

Create detailed analysis reports:

```python
# Analysis report
analysis_report = generator.generate_analysis_report(
    analyses=[
        'behavioral_patterns',
        'interaction_analysis',
        'spatial_analysis',
        'evolutionary_trends'
    ],
    include_recommendations=True
)

analysis_report.save('reports/detailed_analysis.pdf')
```

### Comparison Reports

Compare multiple simulation runs:

```python
# Multi-simulation comparison
comparison = generator.generate_comparison_report(
    simulation_ids=['sim_001', 'sim_002', 'sim_003'],
    compare_metrics=['population', 'avg_fitness', 'resource_usage'],
    statistical_tests=True
)

comparison.save('reports/simulation_comparison.html')
```

## Export Formats

### Data Export

Export data in various formats:

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

### Visualization Export

Export visualizations:

```python
# Export as image
visualizer.save_figure(
    'visualizations/population_timeline.png',
    dpi=300,
    format='png'
)

# Export as interactive HTML
visualizer.save_interactive(
    'visualizations/interactive_network.html',
    include_controls=True
)

# Export animation
visualizer.create_animation(
    output_path='visualizations/simulation_animation.mp4',
    fps=30,
    duration=60
)
```

## Interactive Dashboards

### Jupyter Notebooks

Use interactive notebooks for exploration:

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

### Web-Based Dashboards

Create web-based exploration tools:

```python
from farm.api.server import start_api_server

# Start API server with visualization endpoints
start_api_server(port=5000)

# Access dashboard at http://localhost:5000/dashboard
```

## Performance Optimization

### Data Storage
- **Database Indexing**: Optimize queries with proper indexing
- **Batch Operations**: Use batch inserts for better performance
- **Compression**: Compress large datasets to save space
- **Lazy Loading**: Load data on-demand to reduce memory usage

### Visualization Performance
- **Downsampling**: Reduce data points for large datasets
- **Level of Detail**: Adjust detail based on zoom level
- **Caching**: Cache rendered visualizations
- **Progressive Rendering**: Render in stages for large datasets

## Related Documentation

- [Data System](./data-system.md)
- [Data Services](../data/data_services.md)
- [Data Retrieval](../data/data_retrieval.md)
- [Metrics Documentation](../metrics.md)
- [Analysis Documentation](../data/analysis/Analysis.md)

## Examples

For practical examples:
- [Usage Examples](../usage_examples.md)
- [Service Usage Examples](../data/service_usage_examples.md)
- [Experiment Analysis](../experiment_analysis.md)
- [One of a Kind Experiments](../experiments/one_of_a_kind/README.md)
