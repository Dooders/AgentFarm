# Data & Visualization

![Feature](https://img.shields.io/badge/feature-data%20%26%20viz-orange)

## Overview

AgentFarm provides a comprehensive data collection and visualization system that captures every aspect of your simulations, from individual agent actions to population-level dynamics. With powerful analysis tools, flexible data access patterns, and rich visualization capabilities, you can extract deep insights from your simulation data.

### Why Data & Visualization Matter

Effective data management and visualization enable:
- **Deep Insights**: Understand complex simulation dynamics
- **Pattern Discovery**: Identify trends and emergent behaviors
- **Model Validation**: Verify simulation correctness
- **Results Communication**: Share findings effectively
- **Research Reproducibility**: Document and replicate experiments

---

## Core Capabilities

### 1. Comprehensive Data Collection and Metrics

AgentFarm automatically captures detailed data about every aspect of your simulation.

#### Automatic Data Logging

Every simulation run automatically logs data to a structured database:

```python
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation

# Configure simulation
config = SimulationConfig(
    width=100,
    height=100,
    system_agents=20,
    independent_agents=20,
    max_steps=1000,
    
    # Database configuration
    use_in_memory_db=False,  # Use persistent database
    persist_db_on_completion=True
)

# Run simulation - data is automatically logged
results = run_simulation(config)

# Access database
db_path = results['db_path']
print(f"Data saved to: {db_path}")

# Database contains:
# - Agent states at each step
# - All actions and their outcomes
# - Resource distribution over time
# - Population dynamics
# - Combat events
# - Reproduction events
# - Learning experiences
# - And much more!
```

#### Data Schema

AgentFarm uses a comprehensive database schema organized hierarchically:

**Experiment Level:**
- Experiment metadata (name, description, hypothesis)
- Variable configurations
- Results summary

**Simulation Level:**
- Simulation configuration
- Start/end times
- Summary statistics

**Agent Level:**
- Agent lifecycle (birth/death)
- Agent properties (type, position, resources, health)
- Genome information
- Generation tracking

**Event Level:**
- Agent actions (move, gather, attack, share, reproduce)
- State changes
- Combat incidents
- Health changes
- Social interactions
- Learning experiences

```python
# Example: Database structure
"""
experiments/
├── experiment_001/
│   ├── metadata
│   └── simulations/
│       ├── sim_001/
│       │   ├── agents
│       │   ├── actions
│       │   ├── states
│       │   └── events
│       └── sim_002/
│           └── ...
"""
```

#### Metrics Collection

AgentFarm tracks hundreds of metrics automatically:

**Population Metrics:**
```python
from farm.database.data_retrieval import DataRetriever

retriever = DataRetriever(db_path)

# Population over time
pop_data = retriever.get_population_stats()
# Returns:
# {
#     'total_agents': [100, 98, 95, ...],
#     'system_agents': [50, 49, 47, ...],
#     'independent_agents': [50, 49, 48, ...],
#     'steps': [0, 1, 2, ...]
# }

# Birth and death rates
demographics = retriever.get_demographics()
# Returns:
# {
#     'births_per_step': 0.5,
#     'deaths_per_step': 0.3,
#     'avg_lifespan': 234.5,
#     'reproduction_success_rate': 0.65
# }
```

**Resource Metrics:**
```python
# Resource distribution
resource_stats = retriever.get_resource_statistics()
# Returns:
# {
#     'total_resources': 450,
#     'avg_per_agent': 22.5,
#     'resource_depletion_rate': 0.02,
#     'gathering_efficiency': 0.75
# }

# Resource by agent type
resource_by_type = retriever.get_resources_by_agent_type()
# Returns:
# {
#     'SystemAgent': {'avg': 25.3, 'std': 5.2},
#     'IndependentAgent': {'avg': 19.8, 'std': 7.1}
# }
```

**Behavioral Metrics:**
```python
from farm.database.analyzers import ActionStatsAnalyzer

analyzer = ActionStatsAnalyzer(db_path)

# Action frequency
action_stats = analyzer.get_action_statistics()
# Returns:
# {
#     'move': {'count': 5234, 'success_rate': 0.98},
#     'gather': {'count': 3421, 'success_rate': 0.72},
#     'attack': {'count': 876, 'success_rate': 0.45},
#     'share': {'count': 1543, 'success_rate': 0.89},
#     'reproduce': {'count': 234, 'success_rate': 0.67}
# }

# Action rewards
reward_stats = analyzer.get_reward_statistics()
# Returns average reward per action type
```

**Performance Metrics:**
```python
# Simulation performance
perf_metrics = retriever.get_performance_metrics()
# Returns:
# {
#     'total_runtime': 125.4,  # seconds
#     'steps_per_second': 7.98,
#     'actions_per_second': 159.6,
#     'avg_step_time': 0.125,
#     'database_write_time': 12.3
# }
```

#### Custom Metrics

Define your own metrics:

```python
from farm.database.database import SimulationDatabase
import pandas as pd

def calculate_cooperation_index(db_path: str) -> float:
    """Calculate custom cooperation metric."""
    db = SimulationDatabase(db_path)
    
    # Query data
    query = """
    SELECT 
        COUNT(CASE WHEN action_type = 'share' THEN 1 END) as shares,
        COUNT(CASE WHEN action_type = 'attack' THEN 1 END) as attacks,
        COUNT(*) as total_actions
    FROM agent_actions
    """
    
    df = pd.read_sql(query, db.engine)
    
    # Calculate cooperation index
    shares = df['shares'].iloc[0]
    attacks = df['attacks'].iloc[0]
    total = df['total_actions'].iloc[0]
    
    cooperation_index = (shares - attacks) / total if total > 0 else 0
    
    return cooperation_index

# Use custom metric
coop_index = calculate_cooperation_index("simulation.db")
print(f"Cooperation Index: {coop_index:.3f}")
# Values:
# > 0: Cooperative behavior dominant
# < 0: Competitive behavior dominant
# = 0: Balanced
```

---

### 2. Simulation Visualization Tools

AgentFarm provides multiple visualization approaches for different use cases.

#### Interactive Real-Time Visualization

Watch your simulation unfold in real-time:

```python
from farm.core.visualization import SimulationVisualizer
import tkinter as tk

# Create window
root = tk.Tk()
root.title("AgentFarm Visualizer")

# Initialize visualizer
visualizer = SimulationVisualizer(
    parent=root,
    db_path="simulation.db"
)

# Start visualization
root.mainloop()

# Features:
# - Real-time agent rendering
# - Resource distribution overlay
# - Population statistics cards
# - Timeline scrubbing
# - Play/pause controls
# - Speed adjustment
# - Agent tracking
# - Birth/death animations
```

**Visualization Elements:**
- **Agent Rendering**: Color-coded by type, sized by resources
- **Resource Overlay**: Glowing resource locations
- **Population Cards**: Live statistics display
- **Timeline**: Step-by-step navigation
- **Metrics Graphs**: Real-time population/resource trends

#### Static Visualization

Generate visualization images programmatically:

```python
from farm.core.visualization import render_simulation_frame
from PIL import Image

# Render specific step
image = render_simulation_frame(
    db_path="simulation.db",
    step_number=500,
    width=800,
    height=800,
    show_resources=True,
    show_trails=True
)

# Save image
image.save("simulation_step_500.png")

# Create animation
frames = []
for step in range(0, 1000, 10):
    frame = render_simulation_frame("simulation.db", step)
    frames.append(frame)

# Save as GIF
frames[0].save(
    "simulation_animation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
```

#### Spatial Heatmaps

Visualize spatial distributions:

```python
import matplotlib.pyplot as plt
import numpy as np
from farm.database.analyzers import SpatialAnalyzer

analyzer = SpatialAnalyzer(db_path)

# Agent density heatmap
density_map = analyzer.get_density_heatmap(step_range=(0, 1000))

plt.figure(figsize=(10, 8))
plt.imshow(density_map, cmap='hot', interpolation='nearest')
plt.colorbar(label='Agent Visits')
plt.title('Agent Activity Density')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.savefig('density_heatmap.png')

# Resource depletion heatmap
depletion_map = analyzer.get_resource_depletion_map()

plt.figure(figsize=(10, 8))
plt.imshow(depletion_map, cmap='RdYlGn_r', interpolation='nearest')
plt.colorbar(label='Resource Depletion Rate')
plt.title('Resource Depletion Patterns')
plt.savefig('depletion_heatmap.png')

# Combat hotspots
combat_map = analyzer.get_combat_heatmap()

plt.figure(figsize=(10, 8))
plt.imshow(combat_map, cmap='Reds', interpolation='nearest')
plt.colorbar(label='Combat Events')
plt.title('Combat Hotspots')
plt.savefig('combat_heatmap.png')
```

---

### 3. Charting and Plotting Utilities

AgentFarm includes comprehensive charting capabilities for analysis.

#### Population Charts

```python
from farm.charts import chart_simulation

# Population dynamics
chart_simulation.plot_population_dynamics(
    db_path="simulation.db",
    output_path="charts/"
)
# Generates: population_dynamics.png
# Shows: Total, System, Independent, Control agents over time

# Births and deaths
chart_simulation.plot_births_and_deaths(
    db_path="simulation.db",
    output_path="charts/"
)
# Generates: births_deaths.png
# Shows: Birth/death rates over time

# Births/deaths by type
chart_simulation.plot_births_and_deaths_by_type(
    db_path="simulation.db",
    output_path="charts/"
)
# Generates: births_deaths_by_type.png
# Shows: Separate trends for each agent type

# Lifespan distribution
chart_simulation.plot_agent_lifespan_histogram(
    db_path="simulation.db",
    output_path="charts/"
)
# Generates: lifespan_histogram.png
# Shows: Distribution of agent lifespans
```

#### Resource Charts

```python
from farm.charts import chart_resources

# Resource over time
chart_resources.plot_resource_dynamics(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Total resources, consumption rate, regeneration

# Resource distribution
chart_resources.plot_resource_distribution(
    db_path="simulation.db",
    step_number=500,
    output_path="charts/"
)
# Shows: Spatial resource distribution at specific step

# Resource efficiency
chart_simulation.plot_resource_efficiency(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Resources gathered per agent over time

# Resource sharing
chart_simulation.plot_resource_sharing(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Sharing frequency and amounts by agent type
```

#### Action Charts

```python
from farm.charts import chart_actions

# Action frequency
chart_actions.plot_action_frequency_over_time(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: How often each action is performed over time

# Action type distribution
chart_actions.plot_action_type_distribution(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Pie chart of action type proportions

# Action rewards
chart_actions.plot_rewards_by_action_type(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Average reward for each action type

# Action success rates
chart_actions.plot_action_success_rates(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Success rate for each action over time
```

#### Agent Charts

```python
from farm.charts import chart_agents

# Agent types over time
chart_agents.plot_agent_types_over_time(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Proportion of each agent type over time

# Reproduction success
chart_agents.plot_reproduction_success_rate(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Reproduction success rate by agent type

# Lineage analysis
chart_agents.plot_lineage_size(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Size of different agent lineages

# Health and age
chart_simulation.plot_agent_health_and_age(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Correlation between health and age
```

#### Evolutionary Charts

```python
# Generational analysis
chart_simulation.plot_generational_analysis(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Population by generation, fitness trends

# Evolutionary metrics
chart_simulation.plot_evolutionary_metrics(
    db_path="simulation.db",
    output_path="charts/"
)
# Shows: Mutation rates, selection pressure, diversity
```

#### Comparative Charts

Compare multiple simulations:

```python
from farm.database.simulation_comparison import SimulationComparison

# Create comparison
comparison = SimulationComparison([
    "simulation_1.db",
    "simulation_2.db",
    "simulation_3.db"
])

# Compare population dynamics
comparison.plot_population_comparison(output_path="charts/")

# Compare resource efficiency
comparison.plot_resource_efficiency_comparison(output_path="charts/")

# Compare survival rates
comparison.plot_survival_comparison(output_path="charts/")

# Statistical comparison
comparison.generate_comparison_report(output_path="comparison_report.pdf")
```

---

### 4. Automated Report Generation

AgentFarm can automatically generate comprehensive analysis reports.

#### Chart Analyzer

Generate reports with AI-powered insights:

```python
from farm.charts.chart_analyzer import ChartAnalyzer
from farm.database.database import SimulationDatabase

# Initialize analyzer
db = SimulationDatabase("simulation.db")
analyzer = ChartAnalyzer(
    database=db,
    output_dir="analysis_output/",
    save_charts=True
)

# Generate all charts and analyses
analyses = analyzer.analyze_all_charts()

# Access insights
print("=== Simulation Analysis ===\n")
for chart_name, analysis in analyses.items():
    print(f"\n{chart_name}:")
    print(analysis)

# Generates:
# - All standard charts
# - Natural language insights for each chart
# - Key findings and trends
# - Anomaly detection
# - Recommendations
```

**Example Output:**
```
Population Dynamics Analysis:
The simulation shows a stable population that begins with 50 agents and 
maintains an average of 45±5 agents throughout the 1000-step duration. 
System agents show higher survival rates (0.78) compared to Independent 
agents (0.62), suggesting cooperative strategies provide an advantage in 
this environment.

Key Findings:
1. Population stability achieved after initial 100-step adjustment period
2. System agents dominate final population (ratio: 2:1)
3. Three distinct population growth phases observed
4. Resource availability strongly correlated with population size (r=0.87)

Recommendations:
- Investigate the competitive advantage of System agents
- Analyze resource gathering strategies between agent types
- Examine the transition points between growth phases
```

#### Comprehensive Reports

Generate full experimental reports:

```python
from farm.analysis.service import AnalysisService, AnalysisRequest
from pathlib import Path

# Initialize service
service = AnalysisService(config_service)

# Generate comprehensive report
request = AnalysisRequest(
    module_name="comprehensive",
    experiment_path=Path("experiments/cooperation_study"),
    output_path=Path("reports/"),
    group="all"
)

result = service.run(request)

# Report includes:
# - Executive summary
# - Population analysis
# - Resource dynamics
# - Behavioral patterns
# - Statistical tests
# - Visualizations
# - Conclusions and recommendations
```

#### Custom Report Templates

Create custom report formats:

```python
from jinja2 import Template
import json

def generate_custom_report(db_path: str, output_path: str):
    """Generate custom HTML report."""
    
    # Collect data
    retriever = DataRetriever(db_path)
    
    data = {
        'simulation_id': retriever.get_simulation_id(),
        'total_steps': retriever.get_total_steps(),
        'final_population': retriever.get_final_population(),
        'charts': {
            'population': 'charts/population_dynamics.png',
            'resources': 'charts/resource_dynamics.png',
            'actions': 'charts/action_distribution.png'
        },
        'metrics': retriever.get_all_metrics(),
        'insights': analyzer.generate_insights()
    }
    
    # Load template
    template = Template(open('report_template.html').read())
    
    # Render report
    html = template.render(**data)
    
    # Save
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {output_path}")

# Use custom report
generate_custom_report("simulation.db", "custom_report.html")
```

---

## Advanced Data Access

### Repository Pattern

Access data through specialized repositories:

```python
from farm.database.repositories import (
    AgentRepository,
    ActionRepository,
    PopulationRepository,
    ResourceRepository
)
from farm.database.database import SimulationDatabase

# Initialize database
db = SimulationDatabase("simulation.db")

# Agent repository
agent_repo = AgentRepository(db)

# Get specific agent
agent = agent_repo.get_agent_by_id("agent_001")
print(f"Agent: {agent.agent_id}, Type: {agent.agent_type}")

# Get all agents of type
system_agents = agent_repo.get_agents_by_type("SystemAgent")
print(f"Found {len(system_agents)} System agents")

# Get agents alive at step
alive_at_500 = agent_repo.get_agents_alive_at_step(500)

# Action repository
action_repo = ActionRepository(db)

# Get agent actions
agent_actions = action_repo.get_actions_by_agent("agent_001")
print(f"Agent performed {len(agent_actions)} actions")

# Get actions by type
attacks = action_repo.get_actions_by_type("attack")
print(f"Total attacks: {len(attacks)}")

# Population repository
pop_repo = PopulationRepository(db)

# Get population time series
pop_series = pop_repo.get_population_time_series()
# Returns DataFrame with columns: step, total, system, independent, control

# Get population statistics
pop_stats = pop_repo.get_population_statistics()
# Returns: mean, std, min, max, peak_step, etc.
```

### Service Layer

Use services for complex operations:

```python
from farm.database.services import ActionsService, PopulationService

# Actions service
actions_service = ActionsService(db)

# Comprehensive action analysis
action_analysis = actions_service.analyze_all_actions()
# Returns:
# {
#     'statistics': {...},
#     'patterns': {...},
#     'anomalies': [...],
#     'recommendations': [...]
# }

# Action sequence analysis
sequences = actions_service.find_common_sequences(min_length=3)
# Returns most common action sequences

# Population service
pop_service = PopulationService(db)

# Population dynamics analysis
dynamics = pop_service.analyze_population_dynamics()
# Returns:
# {
#     'growth_rate': 0.02,
#     'stability_index': 0.85,
#     'diversity_index': 0.67,
#     'extinction_risk': 0.12
# }
```

### Direct SQL Queries

For advanced users, direct database access:

```python
import pandas as pd
from sqlalchemy import create_engine

# Create engine
engine = create_engine('sqlite:///simulation.db')

# Custom query
query = """
SELECT 
    a.agent_type,
    COUNT(*) as action_count,
    AVG(a.reward) as avg_reward,
    SUM(CASE WHEN a.success = 1 THEN 1 ELSE 0 END) as success_count
FROM agent_actions a
WHERE a.action_type = 'gather'
GROUP BY a.agent_type
"""

results = pd.read_sql(query, engine)
print(results)

# Complex analysis query
query = """
WITH agent_stats AS (
    SELECT 
        agent_id,
        agent_type,
        (death_time - birth_time) as lifespan,
        starting_resources,
        final_resources
    FROM agents
    WHERE death_time IS NOT NULL
)
SELECT 
    agent_type,
    AVG(lifespan) as avg_lifespan,
    AVG(final_resources - starting_resources) as avg_resource_gain,
    COUNT(*) as count
FROM agent_stats
GROUP BY agent_type
ORDER BY avg_lifespan DESC
"""

lifespan_analysis = pd.read_sql(query, engine)
print(lifespan_analysis)
```

---

## Data Export

### Export Formats

Export data in various formats:

```python
from farm.database.data_retrieval import DataRetriever

retriever = DataRetriever("simulation.db")

# Export to CSV
retriever.export_to_csv(
    output_dir="exports/",
    tables=['agents', 'actions', 'states']
)

# Export to JSON
retriever.export_to_json(
    output_file="simulation_data.json",
    include_metadata=True
)

# Export to Parquet (efficient for large datasets)
retriever.export_to_parquet(
    output_dir="exports/parquet/",
    compression='snappy'
)

# Export specific data
agents_df = retriever.get_agents_dataframe()
agents_df.to_csv("agents.csv", index=False)

actions_df = retriever.get_actions_dataframe()
actions_df.to_csv("actions.csv", index=False)
```

### Data Pipelines

Create automated data processing pipelines:

```python
from pathlib import Path

def create_analysis_pipeline(simulation_dir: Path):
    """Automated analysis pipeline."""
    
    # 1. Load data
    print("Loading simulation data...")
    db_path = simulation_dir / "simulation.db"
    db = SimulationDatabase(str(db_path))
    
    # 2. Generate charts
    print("Generating charts...")
    chart_dir = simulation_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    
    analyzer = ChartAnalyzer(db, output_dir=chart_dir)
    analyzer.generate_all_charts()
    
    # 3. Run analyses
    print("Running analyses...")
    analysis_service = AnalysisService(config_service)
    
    request = AnalysisRequest(
        module_name="comprehensive",
        experiment_path=simulation_dir,
        output_path=simulation_dir / "analysis"
    )
    
    analysis_service.run(request)
    
    # 4. Generate report
    print("Generating report...")
    generate_custom_report(
        str(db_path),
        str(simulation_dir / "report.html")
    )
    
    # 5. Export data
    print("Exporting data...")
    retriever = DataRetriever(str(db_path))
    retriever.export_to_csv(output_dir=str(simulation_dir / "exports"))
    
    print(f"Pipeline complete! Results in: {simulation_dir}")

# Run pipeline
create_analysis_pipeline(Path("experiments/exp_001"))
```

---

## Performance Optimization

### Efficient Data Storage

```python
# Configure database for performance
config = SimulationConfig(
    # Use in-memory database for speed
    use_in_memory_db=True,
    
    # Persist to disk when complete
    persist_db_on_completion=True,
    
    # Buffer settings
    log_buffer_size=1000,  # Batch writes
    commit_interval_seconds=30,  # Commit frequency
    
    # Database pragmas
    db_pragma_profile="performance",  # Optimized settings
    db_cache_size_mb=200,  # Larger cache
    db_journal_mode="WAL"  # Write-ahead logging
)
```

### Lazy Loading

Load data on-demand:

> **Note**: The `LazyDataRetriever` class is planned for a future release. Currently, memory-efficient data retrieval can be implemented using pandas chunked reading or custom SQL LIMIT/OFFSET queries.

```python
# Custom lazy loading implementation
from farm.database.database import SimulationDatabase
import pandas as pd

class CustomLazyRetriever:
    def __init__(self, db_path, chunk_size=1000):
        self.db = SimulationDatabase(db_path)
        self.chunk_size = chunk_size

    def get_step_data(self, step):
        """Load data for specific step only."""
        query = f"""
        SELECT * FROM agent_actions
        WHERE step = {step}
        """
        return pd.read_sql(query, self.db.engine)

# Usage
retriever = CustomLazyRetriever("large_simulation.db")

# Data loaded only when accessed
for step in range(0, 10000, 100):
    # Only loads data for this step
    step_data = retriever.get_step_data(step)
    process_step(step_data)
    # Data released after use
```

### Batch Processing

Process large datasets efficiently:

```python
def process_large_simulation(db_path: str, batch_size: int = 1000):
    """Process simulation data in batches."""
    
    engine = create_engine(f'sqlite:///{db_path}')
    
    # Process actions in batches
    offset = 0
    while True:
        query = f"""
        SELECT * FROM agent_actions 
        LIMIT {batch_size} OFFSET {offset}
        """
        
        batch = pd.read_sql(query, engine)
        
        if len(batch) == 0:
            break
            
        # Process batch
        process_action_batch(batch)
        
        offset += batch_size
        
        print(f"Processed {offset} actions...")
```

---

## Example: Complete Data & Visualization Workflow

```python
#!/usr/bin/env python3
"""
Complete example: Data collection, analysis, and visualization workflow.
"""

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.database.database import SimulationDatabase
from farm.charts.chart_analyzer import ChartAnalyzer
from farm.database.data_retrieval import DataRetriever
from pathlib import Path

def main():
    print("=== AgentFarm Data & Visualization Demo ===\n")
    
    # Step 1: Run simulation with comprehensive logging
    print("Step 1: Running simulation...")
    config = SimulationConfig(
        width=100,
        height=100,
        system_agents=25,
        independent_agents=25,
        max_steps=1000,
        use_in_memory_db=False,
        persist_db_on_completion=True,
        seed=42
    )
    
    results = run_simulation(config)
    db_path = results['db_path']
    print(f"  Simulation complete! Data saved to: {db_path}")
    
    # Step 2: Access and explore data
    print("\nStep 2: Exploring simulation data...")
    db = SimulationDatabase(db_path)
    retriever = DataRetriever(db_path)
    
    # Get summary statistics
    pop_stats = retriever.get_population_stats()
    print(f"  Initial population: {pop_stats['initial']}")
    print(f"  Final population: {pop_stats['final']}")
    print(f"  Peak population: {pop_stats['peak']} (step {pop_stats['peak_step']})")
    
    resource_stats = retriever.get_resource_statistics()
    print(f"  Resource efficiency: {resource_stats['gathering_efficiency']:.2%}")
    
    # Step 3: Generate visualizations
    print("\nStep 3: Generating charts...")
    output_dir = Path("visualization_demo")
    output_dir.mkdir(exist_ok=True)
    
    analyzer = ChartAnalyzer(
        database=db,
        output_dir=output_dir,
        save_charts=True
    )
    
    # Generate all charts with AI analysis
    analyses = analyzer.analyze_all_charts()
    print(f"  Generated {len(analyses)} charts with analyses")
    
    # Step 4: Export data
    print("\nStep 4: Exporting data...")
    export_dir = output_dir / "exports"
    export_dir.mkdir(exist_ok=True)
    
    retriever.export_to_csv(
        output_dir=str(export_dir),
        tables=['agents', 'actions', 'simulation_steps']
    )
    print(f"  Data exported to: {export_dir}")
    
    # Step 5: Generate comprehensive report
    print("\nStep 5: Generating report...")
    
    report_data = {
        'simulation_id': results['simulation_id'],
        'config': config.to_dict(),
        'results': results,
        'population_stats': pop_stats,
        'resource_stats': resource_stats,
        'analyses': analyses
    }
    
    generate_html_report(report_data, output_dir / "report.html")
    print(f"  Report generated: {output_dir / 'report.html'}")
    
    # Step 6: Display key insights
    print("\n=== Key Insights ===")
    for chart_name, analysis in list(analyses.items())[:3]:
        print(f"\n{chart_name}:")
        print(f"  {analysis[:200]}...")  # First 200 chars
    
    print(f"\n=== Complete! ===")
    print(f"All outputs in: {output_dir}")

if __name__ == "__main__":
    main()
```

---

## Additional Resources

### Documentation
- [Database Schema](data/database_schema.md) - Detailed schema documentation
- [Data API](data/data_api.md) - Data access patterns
- [Repositories](data/repositories.md) - Repository documentation
- [Metrics](metrics.md) - Available metrics
- [Analysis System](analysis/README.md) - Analysis modules

### Examples
- [Visualization Examples](examples/visualization_examples.py)
- [Data Analysis Examples](examples/data_analysis_examples.py)
- [Custom Charts](examples/custom_charts.py)

### Tools
- [Chart Utilities](charts/) - Charting functions
- [Data Retrieval](database/data_retrieval.py) - Data access
- [Analyzers](database/analyzers/) - Analysis tools

---

## Support

For data and visualization questions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [Full documentation index](README.md)
- **Examples**: Check `examples/` directory for more samples

---

**Ready to visualize your simulations?** Start with the [Interactive Visualizer](#interactive-real-time-visualization) or explore our [Charting Utilities](#charting-and-plotting-utilities)!
