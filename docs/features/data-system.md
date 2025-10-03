# Data System

## Overview

AgentFarm features a comprehensive, layered data architecture designed for efficient storage, retrieval, and analysis of simulation data. The system follows the repository pattern and provides multiple abstraction levels, from low-level database access to high-level coordinated services.

## Key Capabilities

### Comprehensive Data Architecture
- **Database Layer**: SQLite-based storage with optimized schema
- **Repository Layer**: Clean data access patterns following repository design
- **Analyzer Layer**: Specialized components for different analysis types
- **Service Layer**: High-level coordinated operations with error handling

### Advanced Analytics
- **Action Statistics**: Detailed analysis of agent actions and outcomes
- **Behavioral Clustering**: Group agents by behavior patterns
- **Causal Analysis**: Identify cause-and-effect relationships
- **Pattern Recognition**: Discover temporal and spatial patterns

### Flexible Data Access
- **Query Builder**: Construct complex queries with fluent interface
- **Efficient Retrieval**: Optimized queries with proper indexing
- **Filtering**: Rich filtering capabilities across all data types
- **Aggregation**: Built-in aggregation functions for analytics

### High-Level Services
- **Coordinated Operations**: Services that combine multiple data sources
- **Error Handling**: Robust error handling and recovery
- **Caching**: Intelligent caching for frequently accessed data
- **Transaction Management**: Ensure data consistency

### Multi-Simulation Support
- **Experiment Database**: Compare multiple simulation runs
- **Cross-Simulation Analysis**: Analyze patterns across experiments
- **Result Aggregation**: Combine results from multiple runs
- **Parameter Tracking**: Link outcomes to configurations

## Architecture Layers

### Database Layer

The foundation of the data system:

```python
from farm.data.database import SimulationDatabase

# Create database connection
db = SimulationDatabase('simulation.db')

# Low-level database access
with db.connection() as conn:
    cursor = conn.execute(
        "SELECT * FROM agents WHERE health > ?",
        (50,)
    )
    results = cursor.fetchall()
```

### Repository Layer

Clean data access patterns:

```python
from farm.data.repositories import AgentRepository, ActionRepository

# Agent repository
agent_repo = AgentRepository(database)

# Retrieve agents
agent = agent_repo.get_by_id(agent_id=1)
agents = agent_repo.get_all(step=500)
agents_filtered = agent_repo.filter(
    min_health=50,
    max_age=100,
    alive=True
)

# Action repository
action_repo = ActionRepository(database)

# Query actions
actions = action_repo.get_by_agent(agent_id=1)
action_counts = action_repo.get_action_counts(
    step_range=(0, 1000)
)
```

### Analyzer Layer

Specialized analysis components:

```python
from farm.data.analyzers import (
    ActionAnalyzer,
    BehaviorAnalyzer,
    TemporalPatternAnalyzer,
    SpatialAnalyzer
)

# Action analysis
action_analyzer = ActionAnalyzer(database)
action_stats = action_analyzer.analyze_action_distribution()
success_rates = action_analyzer.calculate_success_rates()

# Behavior clustering
behavior_analyzer = BehaviorAnalyzer(database)
clusters = behavior_analyzer.cluster_behaviors(
    n_clusters=5,
    features=['action_frequency', 'resource_usage', 'social_score']
)

# Temporal patterns
temporal_analyzer = TemporalPatternAnalyzer(database)
patterns = temporal_analyzer.find_patterns(
    window_size=50,
    min_support=0.2
)

# Spatial analysis
spatial_analyzer = SpatialAnalyzer(database)
clustering = spatial_analyzer.analyze_spatial_clustering()
hotspots = spatial_analyzer.identify_hotspots()
```

### Service Layer

High-level coordinated operations:

```python
from farm.data.services import SimulationDataService

# Create service
service = SimulationDataService(simulation_id="sim_001")

# Comprehensive analysis
analysis = service.analyze_simulation(
    include_behaviors=True,
    include_interactions=True,
    include_evolution=True
)

# Generate insights
insights = service.generate_insights()

# Export results
service.export_results(
    format='csv',
    output_dir='exports/'
)
```

## Database Schema

### Core Tables

The database schema is optimized for simulation data:

```sql
-- Agents table
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    simulation_id TEXT,
    agent_id INTEGER,
    step INTEGER,
    health REAL,
    resources REAL,
    position_x REAL,
    position_y REAL,
    age INTEGER,
    generation INTEGER,
    alive BOOLEAN
);

-- Actions table
CREATE TABLE actions (
    id INTEGER PRIMARY KEY,
    simulation_id TEXT,
    step INTEGER,
    agent_id INTEGER,
    action_type TEXT,
    target_id INTEGER,
    success BOOLEAN,
    reward REAL
);

-- Interactions table
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY,
    simulation_id TEXT,
    step INTEGER,
    agent1_id INTEGER,
    agent2_id INTEGER,
    interaction_type TEXT,
    outcome TEXT
);
```

### Indexes

Optimized indexes for common queries:

```sql
-- Agent queries
CREATE INDEX idx_agents_step ON agents(simulation_id, step);
CREATE INDEX idx_agents_id ON agents(simulation_id, agent_id);

-- Action queries
CREATE INDEX idx_actions_agent ON actions(simulation_id, agent_id);
CREATE INDEX idx_actions_step ON actions(simulation_id, step);

-- Interaction queries
CREATE INDEX idx_interactions_agents ON interactions(agent1_id, agent2_id);
CREATE INDEX idx_interactions_step ON interactions(simulation_id, step);
```

## Data Access Patterns

### Query Building

Construct complex queries with fluent interface:

```python
from farm.data.query import QueryBuilder

# Build complex query
query = (QueryBuilder(database)
    .select('agents')
    .where('health', '>', 50)
    .where('generation', '>=', 10)
    .order_by('resources', 'DESC')
    .limit(100)
)

results = query.execute()
```

### Efficient Retrieval

Optimize data retrieval:

```python
# Use pagination for large result sets
page_size = 100
offset = 0

while True:
    agents = agent_repo.get_all(
        step=500,
        limit=page_size,
        offset=offset
    )
    
    if not agents:
        break
    
    process_agents(agents)
    offset += page_size
```

### Batch Operations

Perform batch operations efficiently:

```python
# Batch insert
agent_repo.insert_batch([
    {'agent_id': 1, 'health': 100, 'resources': 50},
    {'agent_id': 2, 'health': 90, 'resources': 60},
    # ... more agents
])

# Batch update
agent_repo.update_batch(
    agent_ids=[1, 2, 3],
    updates={'health': 110}
)
```

## Advanced Analytics

### Action Statistics

Analyze agent actions in detail:

```python
from farm.data.analyzers import ActionAnalyzer

analyzer = ActionAnalyzer(database)

# Action distribution
distribution = analyzer.get_action_distribution(
    step_range=(0, 1000),
    group_by='agent_type'
)

# Success rates
success_rates = analyzer.calculate_success_rates(
    action_types=['hunt', 'gather', 'reproduce']
)

# Action sequences
sequences = analyzer.find_common_sequences(
    min_length=3,
    min_support=0.1
)

# Temporal patterns
temporal = analyzer.analyze_temporal_patterns(
    window_size=50,
    stride=10
)
```

### Behavioral Clustering

Group agents by behavior:

```python
from farm.data.analyzers import BehaviorAnalyzer

analyzer = BehaviorAnalyzer(database)

# Cluster agents
clusters = analyzer.cluster_agents(
    features=[
        'action_diversity',
        'resource_efficiency',
        'social_score',
        'exploration_rate'
    ],
    n_clusters=5,
    method='kmeans'
)

# Analyze cluster characteristics
for cluster_id, agents in clusters.items():
    characteristics = analyzer.analyze_cluster(cluster_id)
    print(f"Cluster {cluster_id}: {characteristics}")

# Find behavioral transitions
transitions = analyzer.find_behavioral_transitions(
    time_window=100
)
```

### Causal Analysis

Identify cause-and-effect relationships:

```python
from farm.data.analyzers import CausalAnalyzer

analyzer = CausalAnalyzer(database)

# Analyze causal relationships
effects = analyzer.analyze_causal_effects(
    treatment='high_resources',
    outcome='reproduction_success',
    confounders=['age', 'health', 'generation']
)

# Find causal chains
chains = analyzer.find_causal_chains(
    start_event='resource_collected',
    end_event='reproduction',
    max_length=5
)

# Estimate treatment effects
ate = analyzer.estimate_treatment_effect(
    treatment='learned_behavior',
    outcome='survival_rate',
    method='propensity_score'
)
```

### Pattern Recognition

Discover patterns in simulation data:

```python
from farm.data.analyzers import PatternAnalyzer

analyzer = PatternAnalyzer(database)

# Temporal patterns
temporal_patterns = analyzer.find_temporal_patterns(
    metrics=['population', 'resources', 'diversity'],
    pattern_types=['trend', 'cycle', 'anomaly']
)

# Spatial patterns
spatial_patterns = analyzer.find_spatial_patterns(
    clustering_method='dbscan',
    min_samples=5
)

# Behavioral patterns
behavioral_patterns = analyzer.find_behavioral_patterns(
    sequence_length=5,
    min_support=0.15
)
```

## Multi-Simulation Support

### Experiment Database

Compare multiple simulations:

```python
from farm.data.experiment_db import ExperimentDatabase

exp_db = ExperimentDatabase('experiments.db')

# Store multiple simulations
for sim in simulations:
    exp_db.store_simulation(
        simulation_id=sim.id,
        parameters=sim.config,
        results=sim.results
    )

# Query across simulations
results = exp_db.query_simulations(
    parameter_range={'learning_rate': (0.01, 0.1)},
    min_population=50
)

# Aggregate results
aggregated = exp_db.aggregate_results(
    metric='final_population',
    group_by='learning_rate'
)
```

### Cross-Simulation Analysis

Analyze patterns across experiments:

```python
from farm.data.services import ExperimentAnalysisService

service = ExperimentAnalysisService(exp_db)

# Compare simulation outcomes
comparison = service.compare_simulations(
    simulation_ids=['sim_001', 'sim_002', 'sim_003'],
    metrics=['population', 'diversity', 'resources']
)

# Find optimal parameters
optimal = service.find_optimal_parameters(
    objective='maximize',
    metric='final_population',
    n_best=10
)

# Analyze parameter effects
effects = service.analyze_parameter_effects(
    parameters=['learning_rate', 'mutation_rate'],
    outcome='fitness'
)
```

## Data Persistence

### Checkpointing

Save and restore simulation state:

```python
from farm.data.checkpoint import CheckpointManager

manager = CheckpointManager(database)

# Save checkpoint
manager.save_checkpoint(
    simulation=simulation,
    step=500,
    checkpoint_id='cp_500'
)

# Restore from checkpoint
simulation = manager.restore_checkpoint('cp_500')

# List checkpoints
checkpoints = manager.list_checkpoints(simulation_id='sim_001')
```

### Data Migration

Migrate data between versions:

```python
from farm.data.migration import DataMigrator

migrator = DataMigrator()

# Migrate database schema
migrator.migrate_schema(
    from_version='1.0',
    to_version='2.0',
    database_path='simulation.db'
)

# Export for migration
migrator.export_data(
    database='old_simulation.db',
    output_format='json',
    output_path='migration_data.json'
)

# Import migrated data
migrator.import_data(
    database='new_simulation.db',
    input_path='migration_data.json'
)
```

## Performance Optimization

### Query Optimization
- **Proper Indexing**: Use indexes for frequently queried columns
- **Query Planning**: Analyze and optimize query execution plans
- **Batch Operations**: Use batch operations for multiple updates
- **Connection Pooling**: Reuse database connections

### Memory Management
- **Lazy Loading**: Load data on-demand
- **Pagination**: Use pagination for large result sets
- **Streaming**: Stream large datasets instead of loading all at once
- **Caching**: Cache frequently accessed data

### Storage Optimization
- **Compression**: Compress large datasets
- **Pruning**: Remove unnecessary historical data
- **Archiving**: Archive old simulation data
- **Vacuum**: Regularly vacuum SQLite databases

## Related Documentation

- [Database Schema](../data/database_schema.md)
- [Repositories](../data/repositories.md)
- [Data Services](../data/data_services.md)
- [Data Retrieval](../data/data_retrieval.md)
- [Analyzers Documentation](../data/analyzers/README.md)
- [Service Usage Examples](../data/service_usage_examples.md)

## Examples

For practical examples:
- [Usage Examples](../usage_examples.md)
- [Service Usage Examples](../data/service_usage_examples.md)
- [Action Data Documentation](../action_data.md)
- [Analysis Examples](../data/analysis/Analysis.md)
