# AgentFarm Analysis Capabilities

This document provides a comprehensive overview of the analysis capabilities available in the Farm simulation's analyzer package. Each analyzer focuses on specific aspects of agent behavior and simulation dynamics.

## Overview

The analysis system provides modular, extensible tools for processing simulation data into meaningful insights. Analyzers are organized by domain and can be used individually or in combination for comprehensive analysis.

## Core Analyzers

### 1. Action Stats Analyzer
- Calculates frequency and reward statistics for each action type
- Analyzes interaction rates and performance metrics
- Provides detailed statistical measures of reward distributions
- Computes temporal, resource, and decision-making patterns

**Detailed Documentation**: [Action Stats Analyzer](Action.md)

### 2. Agent Analysis
- Basic agent information and metrics
- Exploration vs exploitation behavior analysis
- Adversarial interaction analysis
- Collaborative behavior patterns
- Learning curve analysis
- Risk-reward analysis
- Resilience and adaptation metrics
- Environmental impact assessment
- Conflict analysis
- Counterfactual analysis

**Detailed Documentation**: [Agent Analysis](Agent.md)

### 3. Behavior Clustering Analyzer
- Groups agents based on behavioral patterns
- Supports multiple clustering algorithms (DBSCAN, Spectral, Hierarchical)
- Provides dimensionality reduction for visualization
- Calculates cluster characteristics and performance metrics

### 4. Causal Analyzer
- Examines cause-effect relationships between actions
- Calculates transition probabilities between states
- Analyzes action impact on resources and rewards
- Identifies trigger patterns and resolution strategies

## Pattern Analysis

### 5. Decision Pattern Analyzer
- Identifies behavioral trends and statistics
- Calculates action frequencies and reward statistics
- Measures decision diversity
- Analyzes co-occurrence patterns

### 6. Sequence Pattern Analyzer
- Identifies common action sequences
- Calculates sequence probabilities
- Tracks pattern frequencies

### 7. Temporal Pattern Analyzer
- Analyzes patterns over time
- Calculates rolling averages and trends
- Segments events and analyzes period-specific metrics

## Population & Resource Analysis

### 8. Population Analyzer
- Tracks population dynamics
- Calculates survival rates
- Analyzes agent distributions
- Measures population momentum and variance

### 9. Resource Analyzer
- Analyzes resource distribution patterns
- Tracks consumption statistics
- Identifies resource hotspots
- Calculates efficiency metrics

### 10. Resource Impact Analyzer
- Analyzes how actions affect resource availability
- Calculates resource efficiency metrics
- Identifies resource optimization opportunities

## Spatial Analysis

### 11. Spatial Analyzer
- Integrates location and movement analysis
- Identifies clustering patterns
- Analyzes position effects on performance

### 12. Location Analyzer
- Analyzes position-specific patterns
- Calculates location-based performance metrics
- Identifies popular areas and bottlenecks

### 13. Movement Analyzer
- Tracks movement patterns and trajectories
- Calculates path statistics
- Analyzes directional preferences

## Learning Analysis

### 14. Learning Analyzer
- Tracks learning progress metrics
- Analyzes module performance
- Calculates learning efficiency
- Provides comprehensive learning statistics

## Analysis Architecture

### Common Interface
All analyzers follow a consistent interface pattern:

```python
def analyze(
    self,
    scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
    agent_id: Optional[str] = None,
    step: Optional[int] = None,
    step_range: Optional[Tuple[int, int]] = None
) -> AnalysisResult:
```

### Analysis Scopes
- **SIMULATION**: Complete simulation data
- **EPISODE**: Specific episode or time period
- **AGENT**: Individual agent analysis
- **STEP**: Single simulation step

### Filtering Options
- **agent_id**: Filter for specific agent
- **step**: Analyze specific timestep
- **step_range**: Analyze range of timesteps

## Usage Examples

### Basic Action Analysis
```python
from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer
from farm.database.repositories.action_repository import ActionRepository

repository = ActionRepository(session)
analyzer = ActionStatsAnalyzer(repository)
stats = analyzer.analyze(scope="SIMULATION")
for metric in stats:
    print(f"{metric.action_type}: {metric.avg_reward:.2f} avg reward")
```

### Population Analysis
```python
from farm.database.analyzers.population_analyzer import PopulationAnalyzer
from farm.database.repositories.population_repository import PopulationRepository

repository = PopulationRepository(session)
analyzer = PopulationAnalyzer(repository)
stats = analyzer.analyze_comprehensive_statistics()
print(f"Peak population: {stats.population_metrics.total_agents}")
```

### Behavioral Clustering
```python
from farm.database.analyzers.behavior_clustering_analyzer import BehaviorClusteringAnalyzer
from farm.database.repositories.action_repository import ActionRepository

repository = ActionRepository(session)
analyzer = BehaviorClusteringAnalyzer(repository)
clusters = analyzer.analyze()
for cluster_name, agents in clusters.clusters.items():
    print(f"{cluster_name}: {len(agents)} agents")
```

### Comprehensive Agent Analysis
```python
from farm.database.analyzers.agent_analyzer import AgentAnalysis
from farm.database.repositories.agent_repository import AgentRepository

repository = AgentRepository(session)
analyzer = AgentAnalysis(repository)

# Basic agent information
basic_info = analyzer.analyze(agent_id="agent_1")

# Exploration vs exploitation analysis
explore_stats = analyzer.analyze_exploration_exploitation(agent_id="agent_1")

# Learning curve analysis
learning = analyzer.analyze_learning_curve(agent_id="agent_1")

print(f"Agent Type: {basic_info.agent_type}")
print(f"Exploration Rate: {explore_stats.exploration_rate:.2%}")
print(f"Learning Progress: {learning.mistake_reduction:.2%}")
```

## Integration with Services

Analyzers are typically used through high-level services that coordinate multiple analyzers:

```python
from farm.database.services.actions_service import ActionsService
from farm.database.repositories.action_repository import ActionRepository

repository = ActionRepository(session)
actions_service = ActionsService(repository)
results = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats', 'behavior', 'causal']
)
```

## Performance Considerations

### Data Volume
- Use appropriate scopes to limit data processing
- Consider step ranges for large simulations
- Leverage filtering by agent_id when possible

### Caching
- Analyzers cache common calculations
- Results are cached for repeated queries
- Consider data freshness requirements

### Optimization
- Use specific analyzers rather than comprehensive analysis when possible
- Leverage database indexes for filtering
- Consider batch processing for large datasets

## Best Practices

1. **Start with Basic Analysis**
   - Begin with simple statistics and metrics
   - Add complexity as needed
   - Use appropriate scopes for your use case

2. **Combine Analyzers**
   - Use multiple analyzers for comprehensive insights
   - Cross-reference results between analyzers
   - Look for patterns across different analysis types

3. **Consider Context**
   - Account for simulation parameters
   - Consider environmental factors
   - Compare against population averages

4. **Validate Results**
   - Check for data quality issues
   - Verify statistical significance
   - Consider alternative interpretations

## Cross-References

For detailed information on specific analyzers:

- **Action Analysis**: [Action Stats Analyzer](Action.md)
- **Agent Analysis**: [Agent Analysis](Agent.md)
- **Services**: [Services Documentation](../data_services.md)
- **Data API**: [Data API Overview](../data_api.md)

## Notes

- All analyzers support filtering by scope, agent_id, step, and step_range
- Most analyzers provide both detailed and summary statistics
- Analysis results are returned as structured data types for consistency
- Many analyzers support multiple analysis methods and configurations
- Analyzers integrate with the repository pattern for data access
- Services provide high-level coordination between multiple analyzers