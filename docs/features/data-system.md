# Data System

## Overview

AgentFarm features a sophisticated, layered data architecture that provides efficient storage, flexible retrieval, and powerful analysis capabilities for simulation data. The architecture follows the repository pattern that separates data access logic from business logic, creating clean abstractions that make the codebase maintainable and extensible.

This carefully designed infrastructure lets you work with simulation data at the level of abstraction most appropriate for your task. At the bottom, the database layer provides persistent storage with SQL-based queries. Above that, the repository layer offers clean object-oriented interfaces without exposing database details. The analyzer layer implements specialized analytical algorithms. At the top, the service layer coordinates multiple repositories and analyzers for high-level capabilities.

## Database Layer

The database layer provides reliable persistent storage for the vast amounts of data generated during simulations. AgentFarm uses SQLite as its default database engine, chosen for its simplicity, zero-configuration deployment, excellent performance for read-heavy workloads, and portability. SQLite databases are single files that can be easily copied, backed up, and shared.

The database schema is carefully designed to efficiently store simulation data while supporting queries needed for analysis. The schema uses appropriate data types to minimize storage and maximize query performance. Normalization reduces redundancy and ensures consistency. Denormalization is applied selectively when it significantly improves query performance for common access patterns.

Indexes are crucial for query performance, especially as databases grow to millions of records. AgentFarm automatically creates indexes on columns frequently used in WHERE clauses and JOIN operations, dramatically accelerating queries that would otherwise require full table scans. The indexing strategy is continuously refined based on profiling of actual query patterns.

Transaction management ensures data consistency even in the presence of errors or crashes. All related writes are wrapped in transactions so that either all succeed or none do, preventing partial updates that would leave the database inconsistent. For long-running simulations, periodic commits prevent unbounded memory growth while maintaining consistency.

## Repository Layer

The repository layer implements the repository pattern, encapsulating data access logic behind clean object-oriented interfaces. Repositories provide methods for querying and persisting domain objects without exposing how operations are implemented. This abstraction allows underlying storage to change without affecting code that uses repositories, promotes testability by allowing repositories to be mocked, and provides a natural place for caching and optimization.

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

AgentRepository provides access to agent data through methods that speak in domain terms rather than SQL. You can retrieve agents by ID, get all agents at a timestep, filter by criteria like health or status, and track individual agent histories. These methods hide the complexity of constructing queries and transforming database rows into domain objects.

ActionRepository specializes in accessing action data, providing methods to retrieve actions by agent, by type, and action counts and distributions. Actions are central to understanding behavior, and the repository provides convenient access patterns without requiring complex SQL queries.

InteractionRepository manages interaction data, supporting queries for interactions between specific agents, interaction networks over time periods, counts by type, and spatial patterns. This is essential for social network analysis and understanding emergent social structure.

Each repository implements common patterns like filtering with flexible criteria, pagination for large result sets, bulk operations for efficiency, and caching of frequently accessed data. These ensure repositories are not only convenient but also performant.

## Analyzer Layer

While repositories provide data access, analyzers implement analytical algorithms that transform raw data into insights. The analyzer layer consists of specialized components focused on particular analysis types, each implementing sophisticated algorithms appropriate for its domain.

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
```

ActionAnalyzer implements algorithms for understanding agent action patterns. It computes action frequency distributions, calculates success rates, identifies common action sequences, and analyzes temporal patterns. The analyzer applies statistical methods like sequence mining and time series analysis to reveal structure in action data.

BehaviorAnalyzer groups agents by behavioral similarity using clustering and classification. It defines behavioral features that characterize strategies, applies clustering algorithms to identify behavioral types, tracks behavioral changes over time, and compares behaviors across conditions. This reveals the diversity of strategies and how diversity emerges and evolves.

TemporalPatternAnalyzer specializes in discovering patterns in time series data. It identifies trends showing directional changes, detects cycles using spectral analysis, recognizes changepoints where behavior shifts, and measures autocorrelations indicating system memory. These analyses characterize system dynamics.

SpatialAnalyzer examines spatial distributions and patterns using spatial statistics. It measures clustering to determine whether agents aggregate or disperse, computes density gradients, identifies hotspots, and analyzes territoriality. This reveals how agents organize in space and how spatial structure affects interactions.

## Service Layer

The service layer provides the highest abstraction, coordinating multiple repositories and analyzers to implement complex analytical workflows. Services handle orchestration of multi-step analyses, error handling and recovery, progress reporting, and caching to avoid redundant computation.

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

SimulationDataService provides comprehensive access to all data from a specific simulation through a unified interface. Rather than working with multiple repositories separately, you interact with a single service that coordinates data access, handles initialization, manages connections, and transforms results into convenient formats.

AnalysisService coordinates complete analytical pipelines combining multiple analyzers. For example, comprehensive behavioral analysis might involve ActionAnalyzer to characterize behaviors, BehaviorAnalyzer to cluster similar agents, NetworkAnalyzer to examine social structure, and EvolutionaryAnalyzer to understand how behaviors evolve. The service orchestrates these analyses and presents integrated results.

ExperimentAnalysisService specializes in analyzing results across multiple simulations. It loads data from multiple databases, aligns data to enable comparison, applies statistical tests, computes effect sizes, and generates comparative visualizations. This makes systematic experiments and rigorous analysis straightforward.

## Database Schema Design

The database schema reflects careful thinking about data representation, query patterns, and performance. Core tables represent fundamental entities: agents, actions, interactions, resources, and metadata.

The agents table stores agent states at each timestep with columns for agent identifiers, temporal information, state variables, spatial positions, and status. The table is indexed by simulation ID, timestep, and agent ID to support common query patterns.

The actions table records every action with columns identifying the agent and timestep, action type and target, success or failure, and outcomes. Actions are central to understanding behavior, and the table supports queries about frequencies, success rates, temporal patterns, and agent-specific histories.

The interactions table stores pairwise interactions including participants, interaction type, spatial context, and outcomes. This underpins social network analysis and study of emergent social structure.

Schema versioning and migration support ensure databases remain compatible as the platform evolves. When schema changes are necessary, migration scripts transform existing databases automatically, preserving data while updating structure.

## Advanced Analytics

Beyond basic data access and standard analyses, AgentFarm's data system supports sophisticated analytical techniques providing deep insights.

Causal analysis tools help identify cause-and-effect relationships even in observational data. The system implements propensity score matching for creating balanced comparison groups, instrumental variable approaches leveraging exogenous variation, and graphical causal models for reasoning about causal structure.

Behavioral clustering groups agents by similarity in behavioral patterns using machine learning. The system extracts behavioral features, applies clustering algorithms including k-means and hierarchical clustering, and analyzes resulting clusters. This reveals behavioral diversity and specialist versus generalist strategies.

Pattern mining discovers frequent patterns in sequential data like action sequences. Association rule mining finds combinations that frequently occur together. Sequence mining identifies common behavioral motifs. These exploratory techniques generate hypotheses about mechanisms that can be tested through targeted experiments.

## Multi-Simulation Support

Research often involves many simulations with varying parameters. Managing and analyzing this collection requires infrastructure beyond what's needed for single simulations.

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

The experiment database stores metadata and key results from all simulations in a research project, creating a searchable catalog. Each entry includes complete parameter configurations, summary statistics, provenance information, and annotations. Cross-simulation queries enable finding simulations matching criteria and comparing simulations differing in specific ways.

## Performance Optimization

Efficient data operations are crucial for maintaining good performance as databases grow. AgentFarm implements multiple optimization strategies.

Query optimization ensures database queries execute efficiently through appropriate indexing, query planning, and caching. The system creates indexes automatically for common patterns, uses explain plans to verify queries use indexes effectively, and caches results when appropriate.

Batch operations process multiple records together rather than one at a time, dramatically reducing overhead. Bulk inserts add many records in a single transaction. Batch updates modify many records together. These can be orders of magnitude faster than record-by-record processing.

Lazy loading defers data retrieval until actually needed, avoiding unnecessary work. Connection pooling reuses database connections rather than creating new ones for each operation. Caching stores results of expensive operations for reuse, dramatically speeding up iterative analysis workflows.

## Related Documentation

For detailed information, see [Database Schema](../data/database_schema.md), [Repositories](../data/repositories.md), [Data Services](../data/data_services.md), [Data Retrieval](../data/data_retrieval.md), [Analyzers Documentation](../data/analyzers/README.md), and [Service Usage Examples](../data/service_usage_examples.md).

## Examples

Practical examples can be found in [Usage Examples](../usage_examples.md), [Service Usage Examples](../data/service_usage_examples.md), [Action Data Documentation](../action_data.md), and [Analysis Examples](../data/analysis/Analysis.md).
