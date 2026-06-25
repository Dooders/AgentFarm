# Service Usage Examples

This document provides comprehensive examples of how to use the AgentFarm services for data analysis. Each example demonstrates real-world usage patterns with actual service methods and data structures.

## Overview

The AgentFarm system provides high-level services that coordinate multiple analyzers to provide comprehensive insights into simulation data. These services abstract away the complexity of individual analyzers and provide convenient interfaces for common analysis tasks.

## Available Services

### 1. ActionsService
Coordinates multiple action-related analyzers for comprehensive action analysis.

### 2. PopulationService
Provides population-level analysis and statistics.

## ActionsService Usage Examples

### Basic Action Analysis

```python
from farm.database.services.actions_service import ActionsService
from farm.database.repositories.action_repository import ActionRepository
from farm.database.session_manager import SessionManager

# Initialize service
session_manager = SessionManager("path/to/simulation.db")
session = session_manager.create_session()
repository = ActionRepository(session)
actions_service = ActionsService(repository)

# Perform comprehensive action analysis
results = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats', 'behavior', 'resource']
)

# Access results by analysis type
action_stats = results['stats']
behavior_clusters = results['behavior']
resource_impacts = results['resource']

# Print action statistics
for metric in action_stats:
    print(f"Action: {metric.action_type}")
    print(f"  Count: {metric.count}")
    print(f"  Avg Reward: {metric.avg_reward:.2f}")
    print(f"  Frequency: {metric.frequency:.2%}")
```

### Agent-Specific Analysis

```python
# Analyze actions for a specific agent
agent_results = actions_service.analyze_actions(
    scope="SIMULATION",
    agent_id="agent_001",
    analysis_types=['stats', 'sequence', 'temporal']
)

# Get action summary for the agent
agent_summary = actions_service.get_action_summary(
    scope="SIMULATION",
    agent_id="agent_001"
)

print("Agent Action Summary:")
for action_type, metrics in agent_summary.items():
    print(f"  {action_type}:")
    for metric_name, value in metrics.items():
        print(f"    {metric_name}: {value:.2f}")
```

### Time-Based Analysis

```python
# Analyze actions over a specific time period
period_results = actions_service.analyze_actions(
    scope="EPISODE",
    step_range=(100, 200),
    analysis_types=['stats', 'temporal', 'resource']
)

# Compare early vs late period
early_results = actions_service.analyze_actions(
    scope="EPISODE",
    step_range=(0, 100),
    analysis_types=['stats']
)

late_results = actions_service.analyze_actions(
    scope="EPISODE",
    step_range=(900, 1000),
    analysis_types=['stats']
)

# Compare performance over time
for early_metric, late_metric in zip(early_results['stats'], late_results['stats']):
    if early_metric.action_type == late_metric.action_type:
        reward_change = late_metric.avg_reward - early_metric.avg_reward
        print(f"{early_metric.action_type}: {reward_change:+.2f} reward change")
```

### Comprehensive Analysis Workflow

```python
# Perform all available analyses
comprehensive_results = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats', 'behavior', 'causal', 'decision', 
                   'resource', 'sequence', 'temporal']
)

# Analyze behavioral patterns
behavior_clusters = comprehensive_results['behavior']
print(f"Found {len(behavior_clusters.clusters)} behavioral clusters:")
for cluster_name, agents in behavior_clusters.clusters.items():
    print(f"  {cluster_name}: {len(agents)} agents")

# Analyze causal relationships
causal_analysis = comprehensive_results['causal']
for causal in causal_analysis:
    print(f"Cause: {causal.cause_action}")
    print(f"Effect: {causal.effect_action}")
    print(f"Strength: {causal.correlation_strength:.2f}")

# Analyze decision patterns
decision_patterns = comprehensive_results['decision']
print(f"Decision diversity: {decision_patterns.diversity_index:.2f}")
print(f"Common patterns: {len(decision_patterns.common_patterns)}")

# Analyze resource impacts
resource_impacts = comprehensive_results['resource']
for impact in resource_impacts:
    print(f"{impact.action_type}: {impact.resource_efficiency:.2f} efficiency")

# Analyze action sequences
sequences = comprehensive_results['sequence']
for sequence in sequences[:5]:  # Top 5 sequences
    print(f"Sequence: {sequence.sequence}")
    print(f"  Count: {sequence.count}")
    print(f"  Probability: {sequence.probability:.2%}")

# Analyze temporal patterns
temporal_patterns = comprehensive_results['temporal']
for pattern in temporal_patterns:
    print(f"Action: {pattern.action_type}")
    print(f"  Time periods: {len(pattern.time_distribution)}")
    print(f"  Avg reward: {sum(pattern.reward_progression) / len(pattern.reward_progression):.2f}")
```

### Error Handling and Best Practices

```python
try:
    # Perform analysis with error handling
    results = actions_service.analyze_actions(
        scope="SIMULATION",
        analysis_types=['stats', 'behavior']
    )
    
    # Check if results are valid
    if 'stats' in results and results['stats']:
        print("Analysis completed successfully")
    else:
        print("No action data found")
        
except Exception as e:
    print(f"Analysis failed: {e}")
    # Fallback to basic analysis
    basic_results = actions_service.analyze_actions(
        scope="SIMULATION",
        analysis_types=['stats']
    )
```

### Performance Optimization

```python
# Use specific analysis types for better performance
quick_stats = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats']  # Only basic statistics
)

# Use step ranges for large datasets
large_dataset_results = actions_service.analyze_actions(
    scope="EPISODE",
    step_range=(0, 1000),  # Limit data range
    analysis_types=['stats', 'resource']
)

# Cache results for repeated analysis
cached_results = {}
for agent_id in ["agent_001", "agent_002", "agent_003"]:
    if agent_id not in cached_results:
        cached_results[agent_id] = actions_service.analyze_actions(
            scope="SIMULATION",
            agent_id=agent_id,
            analysis_types=['stats']
        )
```

## PopulationService Usage Examples

### Basic Population Analysis

```python
from farm.database.services.population_service import PopulationService
from farm.database.repositories.population_repository import PopulationRepository

# Initialize service
session_manager = SessionManager("path/to/simulation.db")
session = session_manager.create_session()
repository = PopulationRepository(session)
population_service = PopulationService(repository)

# Get comprehensive population statistics
stats = population_service.execute(session)

# Access population metrics
print(f"Total Agents: {stats.population_metrics.total_agents}")
print(f"System Agents: {stats.population_metrics.system_agents}")
print(f"Independent Agents: {stats.population_metrics.independent_agents}")
print(f"Control Agents: {stats.population_metrics.control_agents}")

# Access variance statistics
print(f"Population Variance: {stats.population_variance.variance:.2f}")
print(f"Standard Deviation: {stats.population_variance.standard_deviation:.2f}")
print(f"Coefficient of Variation: {stats.population_variance.coefficient_variation:.2f}")
```

### Population Data Analysis

```python
# Get basic population statistics
basic_stats = population_service.basic_population_statistics(session)

if basic_stats:
    print(f"Average Population: {basic_stats.avg_population:.1f}")
    print(f"Peak Population: {basic_stats.peak_population}")
    print(f"Death Step: {basic_stats.death_step}")
    print(f"Resource Efficiency: {basic_stats.resource_efficiency:.2%}")
else:
    print("No population data available")

# Get agent type distribution
type_distribution = population_service.agent_type_distribution(session)

if type_distribution:
    print(f"System Agents: {type_distribution.system_agents}")
    print(f"Independent Agents: {type_distribution.independent_agents}")
    print(f"Control Agents: {type_distribution.control_agents}")
    print(f"Dominant Type: {type_distribution.dominant_type}")
```

### Population Trends Analysis

```python
# Get population data over time
population_data = population_service.population_data(session)

if population_data:
    # Analyze population trends
    populations = [p.population for p in population_data]
    steps = [p.step for p in population_data]
    
    # Calculate growth rate
    if len(populations) > 1:
        growth_rate = (populations[-1] - populations[0]) / len(populations)
        print(f"Average growth rate: {growth_rate:.2f} agents per step")
    
    # Find peak population
    peak_population = max(populations)
    peak_step = steps[populations.index(peak_population)]
    print(f"Peak population: {peak_population} at step {peak_step}")
    
    # Analyze stability
    variance = sum((p - sum(populations)/len(populations))**2 for p in populations) / len(populations)
    print(f"Population variance: {variance:.2f}")
```

## Advanced Usage Patterns

### Cross-Service Analysis

```python
# Combine action and population analysis
actions_service = ActionsService(ActionRepository(session))
population_service = PopulationService(PopulationRepository(session))

# Get action patterns for different population phases
early_actions = actions_service.analyze_actions(
    scope="EPISODE",
    step_range=(0, 100),
    analysis_types=['stats']
)

late_actions = actions_service.analyze_actions(
    scope="EPISODE",
    step_range=(900, 1000),
    analysis_types=['stats']
)

# Compare action patterns with population changes
population_stats = population_service.execute(session)

print("Action patterns during population changes:")
for early_metric, late_metric in zip(early_actions['stats'], late_actions['stats']):
    if early_metric.action_type == late_metric.action_type:
        frequency_change = late_metric.frequency - early_metric.frequency
        print(f"{early_metric.action_type}: {frequency_change:+.2%} frequency change")
```

### Custom Analysis Workflows

```python
def analyze_agent_performance(agent_id: str, session):
    """Custom workflow for analyzing individual agent performance."""
    
    actions_service = ActionsService(ActionRepository(session))
    
    # Get comprehensive agent analysis
    agent_results = actions_service.analyze_actions(
        scope="SIMULATION",
        agent_id=agent_id,
        analysis_types=['stats', 'sequence', 'temporal']
    )
    
    # Extract key metrics
    stats = agent_results['stats']
    sequences = agent_results['sequence']
    temporal = agent_results['temporal']
    
    # Calculate performance score
    total_actions = sum(metric.count for metric in stats)
    avg_reward = sum(metric.avg_reward * metric.count for metric in stats) / total_actions
    sequence_diversity = len(sequences)
    
    return {
        'agent_id': agent_id,
        'total_actions': total_actions,
        'avg_reward': avg_reward,
        'sequence_diversity': sequence_diversity,
        'temporal_trends': len(temporal)
    }

# Use custom workflow
agent_performance = analyze_agent_performance("agent_001", session)
print(f"Agent Performance: {agent_performance}")
```

### Batch Analysis

```python
def batch_analyze_simulations(simulation_paths: List[str]):
    """Analyze multiple simulations in batch."""
    
    results = {}
    
    for sim_path in simulation_paths:
        session_manager = SessionManager(sim_path)
        session = session_manager.create_session()
        
        # Initialize services
        actions_service = ActionsService(ActionRepository(session))
        population_service = PopulationService(PopulationRepository(session))
        
        # Perform analysis
        action_results = actions_service.analyze_actions(
            scope="SIMULATION",
            analysis_types=['stats', 'resource']
        )
        
        population_results = population_service.execute(session)
        
        # Store results
        results[sim_path] = {
            'actions': action_results,
            'population': population_results
        }
        
        session.close()
    
    return results

# Use batch analysis
simulation_paths = [
    "simulation_1.db",
    "simulation_2.db",
    "simulation_3.db"
]

batch_results = batch_analyze_simulations(simulation_paths)

# Compare results across simulations
for sim_path, results in batch_results.items():
    print(f"\nSimulation: {sim_path}")
    
    # Compare action patterns
    action_stats = results['actions']['stats']
    total_actions = sum(metric.count for metric in action_stats)
    print(f"  Total Actions: {total_actions}")
    
    # Compare population metrics
    pop_metrics = results['population'].population_metrics
    print(f"  Peak Population: {pop_metrics.total_agents}")
```

## Error Handling and Troubleshooting

### Common Issues and Solutions

```python
# Issue: No data found
try:
    results = actions_service.analyze_actions(scope="SIMULATION")
except Exception as e:
    if "No data" in str(e):
        print("No action data found in simulation")
        # Check if simulation has any data
        actions = repository.get_actions_by_scope("SIMULATION")
        if not actions:
            print("Simulation contains no action data")
    else:
        raise e

# Issue: Memory issues with large datasets
try:
    # Use step ranges for large datasets
    results = actions_service.analyze_actions(
        scope="SIMULATION",
        step_range=(0, 1000),  # Limit data range
        analysis_types=['stats']  # Limit analysis types
    )
except MemoryError:
    print("Dataset too large, consider using smaller step ranges")

# Issue: Invalid analysis types
try:
    results = actions_service.analyze_actions(
        scope="SIMULATION",
        analysis_types=['invalid_type']
    )
except ValueError as e:
    print(f"Invalid analysis type: {e}")
    # Use valid analysis types
    valid_types = ['stats', 'behavior', 'causal', 'decision', 
                   'resource', 'sequence', 'temporal']
    print(f"Valid types: {valid_types}")
```

### Performance Monitoring

```python
import time

def monitor_analysis_performance():
    """Monitor analysis performance and provide feedback."""
    
    start_time = time.time()
    
    # Perform analysis
    results = actions_service.analyze_actions(
        scope="SIMULATION",
        analysis_types=['stats', 'behavior', 'resource']
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Analysis completed in {duration:.2f} seconds")
    
    # Check result quality
    if 'stats' in results and results['stats']:
        print(f"Analyzed {len(results['stats'])} action types")
    else:
        print("Warning: No action statistics generated")
    
    return results

# Use performance monitoring
results = monitor_analysis_performance()
```

## Best Practices

### 1. Service Initialization
- Always use proper session management
- Initialize services with appropriate repositories
- Close sessions when done

### 2. Analysis Scope
- Use appropriate scopes for your use case
- Consider data volume when selecting ranges
- Use step ranges for large datasets

### 3. Error Handling
- Always wrap service calls in try-catch blocks
- Check for empty or null results
- Provide meaningful error messages

### 4. Performance Optimization
- Use specific analysis types when possible
- Cache results for repeated analysis
- Use step ranges for large datasets

### 5. Data Validation
- Verify data exists before analysis
- Check result quality and completeness
- Validate against expected patterns

## Notes

- All examples use actual service methods and data structures
- Error handling patterns reflect real-world usage scenarios
- Performance considerations are based on actual system behavior
- Integration patterns demonstrate proper service coordination
- Examples can be adapted for specific analysis requirements 