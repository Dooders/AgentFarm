# Data System

![Feature](https://img.shields.io/badge/feature-data%20system-blue)

## Overview

AgentFarm's Data System provides a comprehensive, layered architecture for managing simulation data with unprecedented depth and flexibility. From raw database storage to high-level analytical services, the system offers powerful tools for data persistence, retrieval, analysis, and insight generation.

### Why a Comprehensive Data System Matters

A robust data architecture enables:
- **Complete Observability**: Track every aspect of simulations
- **Flexible Analysis**: Query and analyze data in multiple ways
- **Scalable Storage**: Handle simulations of any size
- **Research Reproducibility**: Ensure consistent, reliable data access
- **Performance**: Optimized queries and efficient data structures

---

## Core Capabilities

### 1. Comprehensive Data Architecture

AgentFarm implements a clean, layered architecture that separates concerns and provides clear interfaces at each level.

#### Architectural Layers

```
┌─────────────────────────────────────┐
│         Services Layer              │  ← High-level coordination
│  (ActionsService, PopulationService)│
├─────────────────────────────────────┤
│       Repositories Layer            │  ← Data access abstraction
│  (ActionRepo, AgentRepo, PopRepo)  │
├─────────────────────────────────────┤
│        Analyzers Layer              │  ← Analysis implementations
│  (Stats, Behavior, Causal, etc.)   │
├─────────────────────────────────────┤
│         Database Layer              │  ← Persistence & ORM
│  (SQLAlchemy Models, Migrations)   │
└─────────────────────────────────────┘
```

**Benefits:**
- **Separation of Concerns**: Each layer has a single responsibility
- **Testability**: Layers can be tested independently
- **Flexibility**: Swap implementations without affecting other layers
- **Maintainability**: Changes are localized to specific layers

#### Database Layer

The foundation of the data system using SQLAlchemy ORM:

```python
from farm.database.database import SimulationDatabase

# Initialize database
db = SimulationDatabase("simulation.db")

# Database provides:
# - SQLAlchemy ORM models
# - Transaction management
# - Connection pooling
# - Schema migrations
# - Multi-simulation support

# Access database engine
engine = db.engine

# Get session
with db.get_session() as session:
    # Perform database operations
    agents = session.query(AgentModel).all()
```

**Database Schema** includes tables for:
- **Experiments**: Top-level experiment metadata
- **Simulations**: Simulation runs and configurations
- **Agents**: Agent lifecycle and properties
- **Actions**: All agent actions and outcomes
- **States**: Agent states at each timestep
- **Resources**: Resource distribution and changes
- **Events**: Special events (births, deaths, combat)
- **Learning**: Learning experiences and metrics

#### Repository Layer

Repositories provide clean data access interfaces:

```python
from farm.database.repositories import (
    AgentRepository,
    ActionRepository,
    PopulationRepository,
    ResourceRepository
)

# Initialize repositories
agent_repo = AgentRepository(session_manager)
action_repo = ActionRepository(session_manager)
pop_repo = PopulationRepository(session_manager)

# Repositories provide:
# - Filtered queries
# - Aggregated data
# - Joins and relationships
# - Transaction safety
# - Consistent interfaces
```

**Key Repositories:**

```python
# Agent Repository - Complete agent data
agent = agent_repo.get_agent_by_id("agent_001")
agent_actions = agent_repo.get_actions_by_agent_id("agent_001")
agent_states = agent_repo.get_states_by_agent_id("agent_001")
agent_metrics = agent_repo.get_agent_performance_metrics("agent_001")
agent_children = agent_repo.get_agent_children("agent_001")

# Action Repository - Flexible action queries
actions = action_repo.get_actions_by_scope(
    scope="SIMULATION",
    agent_id="agent_001",
    step_range=(100, 200)
)

# Population Repository - Population analytics
population_data = pop_repo.get_population_data(session, scope="SIMULATION")
agent_distribution = pop_repo.get_agent_type_distribution(session)
evolution_metrics = pop_repo.evolution(session, scope="SIMULATION")

# Resource Repository - Resource analysis
resource_states = resource_repo.get_resource_states(
    step_range=(0, 1000)
)
resource_consumption = resource_repo.get_consumption_by_type()
```

#### Analyzers Layer

Specialized analyzers process data into insights:

```python
from farm.database.analyzers import (
    ActionStatsAnalyzer,
    BehaviorClusteringAnalyzer,
    CausalAnalyzer,
    ResourceImpactAnalyzer,
    TemporalPatternAnalyzer
)

# Action Statistics
action_analyzer = ActionStatsAnalyzer(action_repo)
stats = action_analyzer.analyze(scope="SIMULATION")
# Returns detailed metrics for each action type

# Behavior Clustering
behavior_analyzer = BehaviorClusteringAnalyzer(action_repo)
clusters = behavior_analyzer.analyze(scope="SIMULATION")
# Identifies distinct behavioral patterns

# Causal Analysis
causal_analyzer = CausalAnalyzer(action_repo)
causal_results = causal_analyzer.analyze(
    action_type="gather",
    scope="SIMULATION"
)
# Analyzes cause-effect relationships

# Resource Impact
resource_analyzer = ResourceImpactAnalyzer(action_repo)
impacts = resource_analyzer.analyze(scope="SIMULATION")
# Measures resource efficiency by action

# Temporal Patterns
temporal_analyzer = TemporalPatternAnalyzer(action_repo)
patterns = temporal_analyzer.analyze(scope="SIMULATION")
# Identifies time-based patterns
```

#### Services Layer

High-level services coordinate multiple components:

```python
from farm.database.services import ActionsService, PopulationService

# Actions Service - Comprehensive action analysis
actions_service = ActionsService(action_repo)

# Run multiple analyses at once
comprehensive_analysis = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats', 'behavior', 'causal', 'resource']
)

# Results include:
# - action_stats: Statistical metrics
# - behavior_clusters: Behavioral patterns
# - causal_analysis: Cause-effect relationships
# - resource_impacts: Resource efficiency

# Get high-level summary
summary = actions_service.get_action_summary(scope="SIMULATION")
# Returns easy-to-digest metrics for each action

# Population Service - Population analytics
pop_service = PopulationService(pop_repo)

# Get comprehensive population statistics
pop_stats = pop_service.execute(session)
# Returns: population metrics, variance, distribution, evolution
```

---

### 2. Advanced Analytics

AgentFarm provides sophisticated analytical capabilities for deep insights.

#### Action Statistics Analysis

Comprehensive metrics for every action type:

```python
from farm.database.analyzers import ActionStatsAnalyzer

analyzer = ActionStatsAnalyzer(action_repo)
stats = analyzer.analyze(scope="SIMULATION")

# Stats include for each action:
for action_stat in stats:
    print(f"Action: {action_stat.action_type}")
    print(f"  Count: {action_stat.count}")
    print(f"  Success rate: {action_stat.success_rate:.2%}")
    print(f"  Average reward: {action_stat.avg_reward:.3f}")
    print(f"  Std deviation: {action_stat.std_reward:.3f}")
    print(f"  Median reward: {action_stat.median_reward:.3f}")
    print(f"  25th percentile: {action_stat.q1_reward:.3f}")
    print(f"  75th percentile: {action_stat.q3_reward:.3f}")
    print(f"  Confidence interval: [{action_stat.ci_lower:.3f}, {action_stat.ci_upper:.3f}]")
    print(f"  Variance: {action_stat.variance:.3f}")

# Agent-specific analysis
agent_stats = analyzer.analyze(
    scope="AGENT",
    agent_id="agent_001"
)

# Time-range analysis
period_stats = analyzer.analyze(
    scope="EPISODE",
    step_range=(500, 1000)
)
```

#### Behavioral Clustering

Automatically discover behavioral patterns:

```python
from farm.database.analyzers import BehaviorClusteringAnalyzer

analyzer = BehaviorClusteringAnalyzer(action_repo)

# Configure clustering
analyzer.clustering_method = "dbscan"  # or "spectral", "hierarchical"
analyzer.dim_reduction_method = "pca"  # for visualization
analyzer.n_components = 2

# Run analysis
clustering = analyzer.analyze(scope="SIMULATION")

# Explore clusters
print(f"Found {len(clustering.clusters)} behavioral clusters:")
for cluster_name, agent_ids in clustering.clusters.items():
    print(f"\n{cluster_name} ({len(agent_ids)} agents):")
    
    # Cluster characteristics
    chars = clustering.cluster_characteristics[cluster_name]
    print(f"  Move frequency: {chars.get('move_freq', 0):.2%}")
    print(f"  Attack frequency: {chars.get('attack_freq', 0):.2%}")
    print(f"  Share frequency: {chars.get('share_freq', 0):.2%}")
    print(f"  Gather frequency: {chars.get('gather_freq', 0):.2%}")
    
    # Performance metrics
    perf = clustering.cluster_performance[cluster_name]
    print(f"  Avg reward: {perf.get('avg_reward', 0):.3f}")
    print(f"  Avg resources: {perf.get('avg_resources', 0):.2f}")
    print(f"  Survival rate: {perf.get('survival_rate', 0):.2%}")

# Visualization data (if dim reduction enabled)
if clustering.reduced_features:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    for cluster_name, agent_ids in clustering.clusters.items():
        # Plot cluster in reduced space
        cluster_points = [
            clustering.reduced_features[aid] 
            for aid in agent_ids
        ]
        xs = [p[0] for p in cluster_points]
        ys = [p[1] for p in cluster_points]
        plt.scatter(xs, ys, label=cluster_name, alpha=0.6)
    
    plt.legend()
    plt.title("Behavioral Clusters (PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("behavior_clusters.png")
```

#### Causal Analysis

Understand cause-effect relationships:

```python
from farm.database.analyzers import CausalAnalyzer

analyzer = CausalAnalyzer(action_repo)

# Analyze causal impact of gathering
gather_causal = analyzer.analyze(
    action_type="gather",
    scope="SIMULATION"
)

print(f"Gather Action Causal Analysis:")
print(f"  Causal impact: {gather_causal.causal_impact:.3f}")
print(f"  (Average reward when gathering)")

print(f"\nState Transitions:")
for transition, probability in gather_causal.state_transition_probs.items():
    print(f"  {transition}: {probability:.2%}")

# Example output:
# gather|success_True,resource_change_+2.0,gathered_3: 60%
# share|success_True,resource_change_-1.0,shared_2: 30%
# gather|success_False,resource_change_0.0: 10%

# Interpretation:
# After a gather action:
# - 60% chance of another successful gather (+2 resources)
# - 30% chance of transitioning to sharing
# - 10% chance of failed gather

# Compare all actions
for action_type in ['move', 'gather', 'attack', 'share']:
    analysis = analyzer.analyze(
        action_type=action_type,
        scope="SIMULATION"
    )
    print(f"{action_type}: impact={analysis.causal_impact:.3f}")
```

#### Pattern Recognition

Discover temporal and sequence patterns:

```python
from farm.database.analyzers import (
    SequencePatternAnalyzer,
    TemporalPatternAnalyzer
)

# Sequence Pattern Analysis
seq_analyzer = SequencePatternAnalyzer(action_repo)
patterns = seq_analyzer.analyze(scope="SIMULATION")

print("Common Action Sequences:")
for pattern in patterns:
    print(f"  {' → '.join(pattern.sequence)}")
    print(f"    Frequency: {pattern.count}")
    print(f"    Average reward: {pattern.avg_reward:.3f}")
    print(f"    Success rate: {pattern.success_rate:.2%}")

# Example output:
# move → gather → move: Frequency=1234, Reward=1.2, Success=0.85
# gather → share → gather: Frequency=876, Reward=0.8, Success=0.72

# Temporal Pattern Analysis
temporal_analyzer = TemporalPatternAnalyzer(action_repo)
temporal_patterns = temporal_analyzer.analyze(scope="SIMULATION")

print("\nTemporal Patterns:")
for pattern in temporal_patterns:
    print(f"  Time period: steps {pattern.start_step}-{pattern.end_step}")
    print(f"  Dominant action: {pattern.dominant_action}")
    print(f"  Action distribution: {pattern.action_distribution}")
    print(f"  Trend: {pattern.trend}")  # 'increasing', 'decreasing', 'stable'
```

---

### 3. Flexible Data Access

Multiple ways to access and query simulation data.

#### Repository Pattern

Type-safe, structured data access:

```python
from farm.database.repositories import AgentRepository

agent_repo = AgentRepository(session_manager)

# Get specific agent
agent = agent_repo.get_agent_by_id("agent_001")
print(f"Agent: {agent.agent_id}")
print(f"  Type: {agent.agent_type}")
print(f"  Born: step {agent.birth_time}")
print(f"  Died: step {agent.death_time}")
print(f"  Generation: {agent.generation}")

# Get agent's complete info
agent_info = agent_repo.get_agent_info("agent_001")
# Includes: agent data, stats, current state

# Get agent's actions
actions = agent_repo.get_actions_by_agent_id("agent_001")
print(f"Agent performed {len(actions)} actions")

# Get agent's state history
states = agent_repo.get_agent_state_history("agent_001")
for state in states:
    print(f"  Step {state.step}: resources={state.resource_level}, health={state.health}")

# Get agent's offspring
children = agent_repo.get_agent_children("agent_001")
print(f"Agent has {len(children)} offspring")

# Get agent performance metrics
metrics = agent_repo.get_agent_performance_metrics("agent_001")
print(f"Performance Metrics:")
print(f"  Total actions: {metrics['total_actions']}")
print(f"  Success rate: {metrics['success_rate']:.2%}")
print(f"  Avg reward: {metrics['avg_reward']:.3f}")
print(f"  Resources gained: {metrics['resources_gained']}")
print(f"  Lifespan: {metrics['lifespan']} steps")
```

#### Scope-Based Filtering

Query data at different granularities:

```python
from farm.database.enums import AnalysisScope

# Simulation-wide analysis
sim_actions = action_repo.get_actions_by_scope(
    scope=AnalysisScope.SIMULATION
)

# Episode analysis (time period)
episode_actions = action_repo.get_actions_by_scope(
    scope=AnalysisScope.EPISODE,
    step_range=(100, 500)
)

# Agent-specific analysis
agent_actions = action_repo.get_actions_by_scope(
    scope=AnalysisScope.AGENT,
    agent_id="agent_001"
)

# Step-specific analysis
step_actions = action_repo.get_actions_by_scope(
    scope=AnalysisScope.STEP,
    step=250
)

# Combined filters
filtered_actions = action_repo.get_actions_by_scope(
    scope=AnalysisScope.EPISODE,
    agent_id="agent_001",
    step_range=(100, 200)
)
```

#### Direct SQL Queries

For advanced use cases, direct database access:

```python
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine('sqlite:///simulation.db')

# Custom SQL query
query = """
SELECT 
    a.agent_type,
    act.action_type,
    COUNT(*) as action_count,
    AVG(act.reward) as avg_reward,
    SUM(CASE WHEN act.success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
FROM agent_actions act
JOIN agents a ON act.agent_id = a.agent_id
GROUP BY a.agent_type, act.action_type
ORDER BY a.agent_type, action_count DESC
"""

results = pd.read_sql(query, engine)
print(results)

# Complex analytical query
advanced_query = """
WITH agent_performance AS (
    SELECT 
        agent_id,
        agent_type,
        (death_time - birth_time) as lifespan,
        (SELECT COUNT(*) 
         FROM agent_actions 
         WHERE agent_id = a.agent_id AND success = 1) as successful_actions,
        (SELECT AVG(reward) 
         FROM agent_actions 
         WHERE agent_id = a.agent_id) as avg_reward
    FROM agents a
    WHERE death_time IS NOT NULL
)
SELECT 
    agent_type,
    AVG(lifespan) as avg_lifespan,
    AVG(successful_actions) as avg_successful_actions,
    AVG(avg_reward) as overall_avg_reward,
    COUNT(*) as agent_count
FROM agent_performance
GROUP BY agent_type
ORDER BY avg_lifespan DESC
"""

performance_summary = pd.read_sql(advanced_query, engine)
print(performance_summary)
```

---

### 4. High-Level Services

Coordinated analysis operations with built-in error handling.

#### ActionsService

Comprehensive action analysis coordination:

```python
from farm.database.services import ActionsService

actions_service = ActionsService(action_repo)

# Run all analyses at once
comprehensive = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats', 'behavior', 'causal', 'resource', 'temporal']
)

# Access results
print("=== Action Statistics ===")
for stat in comprehensive['action_stats']:
    print(f"{stat.action_type}: {stat.avg_reward:.3f} avg reward")

print("\n=== Behavioral Clusters ===")
for cluster, agents in comprehensive['behavior_clusters'].clusters.items():
    print(f"{cluster}: {len(agents)} agents")

print("\n=== Causal Impacts ===")
for causal in comprehensive['causal_analysis']:
    print(f"{causal.action_type}: {causal.causal_impact:.3f} impact")

print("\n=== Resource Impacts ===")
for impact in comprehensive['resource_impacts']:
    print(f"{impact.action_type}: {impact.resource_efficiency:.3f} efficiency")

print("\n=== Temporal Patterns ===")
for pattern in comprehensive['temporal_patterns']:
    print(f"Period {pattern.start_step}-{pattern.end_step}: {pattern.dominant_action}")

# Get simplified summary
summary = actions_service.get_action_summary(scope="SIMULATION")

for action_type, metrics in summary.items():
    print(f"\n{action_type}:")
    print(f"  Success rate: {metrics['success_rate']:.2%}")
    print(f"  Avg reward: {metrics['avg_reward']:.3f}")
    print(f"  Frequency: {metrics['frequency']:.2%}")
    print(f"  Resource efficiency: {metrics['resource_efficiency']:.3f}")
```

#### PopulationService

Population-level analytics:

```python
from farm.database.services import PopulationService

pop_service = PopulationService(pop_repo)

# Get comprehensive population statistics
with db.get_session() as session:
    stats = pop_service.execute(session)
    
    # Population metrics
    print("=== Population Metrics ===")
    print(f"Total agents: {stats.population_metrics.total_agents}")
    print(f"System agents: {stats.population_metrics.system_agents}")
    print(f"Independent agents: {stats.population_metrics.independent_agents}")
    print(f"Control agents: {stats.population_metrics.control_agents}")
    print(f"Peak population: {stats.population_metrics.peak_population}")
    
    # Population variance
    print("\n=== Population Variance ===")
    print(f"Variance: {stats.population_variance.variance:.2f}")
    print(f"Std deviation: {stats.population_variance.standard_deviation:.2f}")
    print(f"Coefficient of variation: {stats.population_variance.coefficient_variation:.2f}")
    
    # Basic statistics
    basic_stats = pop_service.basic_population_statistics(session)
    print("\n=== Basic Statistics ===")
    print(f"Average population: {basic_stats.avg_population:.2f}")
    print(f"Final step: {basic_stats.death_step}")
    print(f"Resources consumed: {basic_stats.resources_consumed:.2f}")
    print(f"Resources available: {basic_stats.resources_available:.2f}")
```

---

### 5. Multi-Simulation Support

Store and compare multiple simulation runs in a single database.

#### Experiment Database

Manage multiple simulations:

```python
from farm.database.experiment_database import ExperimentDatabase

# Create experiment database
exp_db = ExperimentDatabase(
    db_path="my_experiment.db",
    experiment_id="resource_study_001"
)

# Create contexts for different simulations
sim1_context = exp_db.create_simulation_context("sim_low_resources")
sim2_context = exp_db.create_simulation_context("sim_high_resources")
sim3_context = exp_db.create_simulation_context("sim_variable_resources")

# Run simulations with their contexts
config1 = SimulationConfig(resource_regen_rate=0.01)
result1 = run_simulation(config1, logger=sim1_context.logger)

config2 = SimulationConfig(resource_regen_rate=0.05)
result2 = run_simulation(config2, logger=sim2_context.logger)

config3 = SimulationConfig(resource_regen_rate=0.03)
result3 = run_simulation(config3, logger=sim3_context.logger)

# All data stored in same database with simulation_id tags
```

#### Cross-Simulation Queries

Query across multiple simulations:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///my_experiment.db')
Session = sessionmaker(bind=engine)

with Session() as session:
    # Query agents from specific simulations
    from farm.database.models import AgentModel
    
    agents = session.query(AgentModel).filter(
        AgentModel.simulation_id.in_([
            'sim_low_resources',
            'sim_high_resources'
        ])
    ).all()
    
    # Compare outcomes across simulations
    from farm.database.models import Simulation
    
    simulations = session.query(Simulation).all()
    
    for sim in simulations:
        print(f"\nSimulation: {sim.simulation_id}")
        print(f"  Final population: {sim.results_summary.get('final_population')}")
        print(f"  Average lifespan: {sim.results_summary.get('avg_lifespan')}")
        print(f"  Resources consumed: {sim.results_summary.get('resources_consumed')}")
```

#### Comparative Analysis

Compare simulations systematically:

```python
from analysis.simulation_comparison import SimulationComparator

comparator = SimulationComparator("my_experiment.db")

# Load simulation data
simulation_ids = ['sim_low_resources', 'sim_high_resources', 'sim_variable_resources']
sim_data = comparator.load_simulation_data(simulation_ids)

# Cluster simulations by outcomes
clustering = comparator.cluster_simulations(sim_data)

print(f"Identified {clustering['optimal_clusters']} simulation clusters")
print(f"Silhouette score: {clustering['silhouette_score']:.3f}")

for cluster_id, profile in clustering['cluster_profiles'].items():
    print(f"\nCluster {cluster_id}:")
    for metric, value in profile.items():
        print(f"  {metric}: {value:.2f}")

# Statistical comparison
comparison_stats = comparator.compare_population_dynamics(simulation_ids)

print("\nPopulation Dynamics Comparison:")
print(comparison_stats.describe())
```

---

## Advanced Features

### Custom Analyzers

Create your own analyzers:

```python
from farm.database.repositories.action_repository import ActionRepository
from typing import Optional, Tuple

class CustomAnalyzer:
    """Custom analyzer for specific research questions."""
    
    def __init__(self, repository: ActionRepository):
        self.repository = repository
        
    def analyze(
        self,
        scope: str = "SIMULATION",
        agent_id: Optional[str] = None,
        step_range: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """Perform custom analysis."""
        
        # Get actions
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step_range=step_range
        )
        
        # Custom analysis logic
        results = {
            'metric1': self._calculate_metric1(actions),
            'metric2': self._calculate_metric2(actions),
            'insights': self._generate_insights(actions)
        }
        
        return results
    
    def _calculate_metric1(self, actions):
        # Custom metric calculation
        pass
    
    def _calculate_metric2(self, actions):
        # Another custom metric
        pass
    
    def _generate_insights(self, actions):
        # Generate insights
        pass

# Use custom analyzer
custom = CustomAnalyzer(action_repo)
results = custom.analyze(scope="SIMULATION")
```

### Performance Optimization

Optimize data access for large simulations:

```python
# Use in-memory database for speed
config = SimulationConfig(
    use_in_memory_db=True,
    persist_db_on_completion=True
)

# Batch queries
# Instead of N queries
for agent_id in agent_ids:
    agent = agent_repo.get_agent_by_id(agent_id)
    
# Use single query
agents = session.query(AgentModel).filter(
    AgentModel.agent_id.in_(agent_ids)
).all()

# Use indexes
# Ensure key columns are indexed
CREATE INDEX idx_actions_agent ON agent_actions(agent_id);
CREATE INDEX idx_actions_step ON agent_actions(step);
CREATE INDEX idx_actions_type ON agent_actions(action_type);

# Limit data ranges
# Query specific time periods
actions = action_repo.get_actions_by_scope(
    scope="EPISODE",
    step_range=(1000, 2000)  # Only load relevant data
)
```

---

## Example: Complete Data System Usage

```python
#!/usr/bin/env python3
"""
Complete data system usage example.
Demonstrates all layers of the data architecture.
"""

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from farm.database.database import SimulationDatabase
from farm.database.session_manager import SessionManager
from farm.database.repositories import (
    AgentRepository,
    ActionRepository,
    PopulationRepository
)
from farm.database.services import ActionsService, PopulationService
from farm.database.analyzers import (
    ActionStatsAnalyzer,
    BehaviorClusteringAnalyzer,
    CausalAnalyzer
)

def main():
    print("=== AgentFarm Data System Demo ===\n")
    
    # Step 1: Run simulation (creates database)
    print("Step 1: Running simulation...")
    config = SimulationConfig(
        width=100,
        height=100,
        system_agents=25,
        independent_agents=25,
        max_steps=1000,
        seed=42
    )
    
    results = run_simulation(config)
    db_path = results['db_path']
    print(f"  Database created: {db_path}")
    
    # Step 2: Initialize database layer
    print("\nStep 2: Initializing database layer...")
    db = SimulationDatabase(db_path)
    session_manager = SessionManager(db_path)
    print("  Database layer ready")
    
    # Step 3: Initialize repositories
    print("\nStep 3: Initializing repositories...")
    agent_repo = AgentRepository(session_manager)
    action_repo = ActionRepository(session_manager)
    pop_repo = PopulationRepository(session_manager)
    print("  Repositories initialized")
    
    # Step 4: Use repositories directly
    print("\nStep 4: Repository-level access...")
    agents = agent_repo.get_all_agents()
    print(f"  Total agents: {len(agents)}")
    
    first_agent = agents[0]
    agent_actions = agent_repo.get_actions_by_agent_id(first_agent.agent_id)
    print(f"  Agent {first_agent.agent_id} performed {len(agent_actions)} actions")
    
    # Step 5: Use analyzers
    print("\nStep 5: Analyzer-level analysis...")
    
    stats_analyzer = ActionStatsAnalyzer(action_repo)
    stats = stats_analyzer.analyze(scope="SIMULATION")
    print(f"  Action statistics for {len(stats)} action types")
    
    behavior_analyzer = BehaviorClusteringAnalyzer(action_repo)
    behavior_analyzer.clustering_method = "dbscan"
    clusters = behavior_analyzer.analyze(scope="SIMULATION")
    print(f"  Found {len(clusters.clusters)} behavioral clusters")
    
    # Step 6: Use services
    print("\nStep 6: Service-level coordination...")
    
    actions_service = ActionsService(action_repo)
    comprehensive = actions_service.analyze_actions(
        scope="SIMULATION",
        analysis_types=['stats', 'behavior', 'causal']
    )
    print(f"  Comprehensive analysis complete")
    print(f"    - Action stats: ✓")
    print(f"    - Behavior clusters: ✓")
    print(f"    - Causal analysis: ✓")
    
    with db.get_session() as session:
        pop_service = PopulationService(pop_repo)
        pop_stats = pop_service.execute(session)
        print(f"\n  Population statistics:")
        print(f"    - Total agents: {pop_stats.population_metrics.total_agents}")
        print(f"    - Peak population: {pop_stats.population_metrics.peak_population}")
        print(f"    - Population variance: {pop_stats.population_variance.variance:.2f}")
    
    # Step 7: High-level insights
    print("\nStep 7: Extracting insights...")
    
    summary = actions_service.get_action_summary(scope="SIMULATION")
    
    print("\n  Action Performance Summary:")
    for action_type, metrics in summary.items():
        print(f"    {action_type}:")
        print(f"      Success rate: {metrics['success_rate']:.2%}")
        print(f"      Avg reward: {metrics['avg_reward']:.3f}")
        print(f"      Frequency: {metrics['frequency']:.2%}")
    
    print("\n  Behavioral Clusters:")
    for cluster_name, agent_ids in clusters.clusters.items():
        print(f"    {cluster_name}: {len(agent_ids)} agents")
        chars = clusters.cluster_characteristics[cluster_name]
        print(f"      Characteristics: {chars}")
    
    print("\n=== Data System Demo Complete ===")

if __name__ == "__main__":
    main()
```

---

## Additional Resources

### Documentation
- [Database Schema](data/database_schema.md) - Complete schema documentation
- [Data API](data/data_api.md) - API overview
- [Repositories](data/repositories.md) - Repository documentation
- [Services](data/data_services.md) - Services documentation
- [Analyzers](data/analyzers/README.md) - Analyzer documentation

### Examples
- [Data Access Examples](examples/data_access.py)
- [Custom Analyzer Example](examples/custom_analyzer.py)
- [Multi-Simulation Example](examples/multi_simulation.py)

### API Reference
- [Models](api_reference.md#models) - Database models
- [Repositories](api_reference.md#repositories) - Repository API
- [Services](api_reference.md#services) - Service API

---

## Support

For data system questions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/Dooders/AgentFarm/issues)
- **Documentation**: [Full documentation index](README.md)
- **Examples**: Check `examples/` directory for more samples

---

**Ready to explore your data?** Start with the [Repository Layer](#repository-layer) or dive into [Advanced Analytics](#advanced-analytics)!
