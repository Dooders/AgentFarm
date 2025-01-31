AgentFarm Analysis Capabilities
===========================================

This document provides an overview of the analysis capabilities available in the Farm 
simulation's analyzer package. Each analyzer focuses on specific aspects of agent behavior 
and simulation dynamics.

Core Analyzers
-------------

1. Action Stats Analyzer
   - Calculates frequency and reward statistics for each action type
   - Analyzes interaction rates and performance metrics
   - Provides detailed statistical measures of reward distributions
   - Computes temporal, resource, and decision-making patterns

2. Agent Analyzer
   - Basic agent information and metrics
   - Exploration vs exploitation behavior analysis
   - Adversarial interaction analysis
   - Collaborative behavior patterns
   - Learning curve analysis
   - Risk-reward analysis
   - Resilience and adaptation metrics
   - Environmental impact assessment

3. Behavior Clustering Analyzer
   - Groups agents based on behavioral patterns
   - Supports multiple clustering algorithms (DBSCAN, Spectral, Hierarchical)
   - Provides dimensionality reduction for visualization
   - Calculates cluster characteristics and performance metrics

4. Causal Analyzer
   - Examines cause-effect relationships between actions
   - Calculates transition probabilities between states
   - Analyzes action impact on resources and rewards
   - Identifies trigger patterns and resolution strategies

Pattern Analysis
--------------

5. Decision Pattern Analyzer
   - Identifies behavioral trends and statistics
   - Calculates action frequencies and reward statistics
   - Measures decision diversity
   - Analyzes co-occurrence patterns

6. Sequence Pattern Analyzer
   - Identifies common action sequences
   - Calculates sequence probabilities
   - Tracks pattern frequencies

7. Temporal Pattern Analyzer
   - Analyzes patterns over time
   - Calculates rolling averages and trends
   - Segments events and analyzes period-specific metrics

Population & Resource Analysis
---------------------------

8. Population Analyzer
   - Tracks population dynamics
   - Calculates survival rates
   - Analyzes agent distributions
   - Measures population momentum and variance

9. Resource Analyzer
   - Analyzes resource distribution patterns
   - Tracks consumption statistics
   - Identifies resource hotspots
   - Calculates efficiency metrics

Spatial Analysis
--------------

10. Spatial Analyzer
    - Integrates location and movement analysis
    - Identifies clustering patterns
    - Analyzes position effects on performance

11. Location Analyzer
    - Analyzes position-specific patterns
    - Calculates location-based performance metrics
    - Identifies popular areas and bottlenecks

12. Movement Analyzer
    - Tracks movement patterns and trajectories
    - Calculates path statistics
    - Analyzes directional preferences

Learning Analysis
---------------

13. Learning Analyzer
    - Tracks learning progress metrics
    - Analyzes module performance
    - Calculates learning efficiency
    - Provides comprehensive learning statistics

Usage Examples
------------

Basic Action Analysis:
```python
from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer

analyzer = ActionStatsAnalyzer(repository)
stats = analyzer.analyze(scope="SIMULATION")
for metric in stats:
    print(f"{metric.action_type}: {metric.avg_reward:.2f} avg reward")
```

Population Analysis:
```python
from farm.database.analyzers.population_analyzer import PopulationAnalyzer

analyzer = PopulationAnalyzer(repository)
stats = analyzer.analyze_comprehensive_statistics()
print(f"Peak population: {stats.population_metrics.total_agents}")
```

Behavioral Clustering:
```python
from farm.database.analyzers.behavior_clustering_analyzer import BehaviorClusteringAnalyzer

analyzer = BehaviorClusteringAnalyzer(repository)
clusters = analyzer.analyze()
for cluster_name, agents in clusters.clusters.items():
    print(f"{cluster_name}: {len(agents)} agents")
```

Notes
-----
- All analyzers support filtering by scope, agent_id, step, and step_range
- Most analyzers provide both detailed and summary statistics
- Analysis results are returned as structured data types for consistency
- Many analyzers support multiple analysis methods and configurations


This documentation provides a comprehensive overview of the analysis capabilities available in the Farm simulation's analyzer package. It includes:

1. A high-level overview of each analyzer
2. Key capabilities and metrics provided
3. Grouping of analyzers by analysis type
4. Usage examples for common analysis tasks
5. Notes about common features and capabilities