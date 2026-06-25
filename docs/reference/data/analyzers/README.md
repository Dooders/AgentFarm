# AgentFarm Analyzers Documentation

This document provides comprehensive documentation for all analyzers available in the AgentFarm system. Each analyzer focuses on specific aspects of simulation data and provides detailed insights into agent behavior, population dynamics, and system performance.

## Overview

The AgentFarm analysis system consists of multiple specialized analyzers that process simulation data to extract meaningful insights. Each analyzer is designed to work with specific data types and provides focused analysis capabilities.

## Analyzer Categories

### 1. Action Analysis
- **ActionStatsAnalyzer**: Comprehensive action statistics and metrics
- **CausalAnalyzer**: Cause-effect relationships between actions
- **DecisionPatternAnalyzer**: Decision-making patterns and trends
- **SequencePatternAnalyzer**: Action sequence analysis
- **TemporalPatternAnalyzer**: Time-based pattern analysis

### 2. Behavioral Analysis
- **BehaviorClusteringAnalyzer**: Agent behavior clustering and classification
- **AgentAnalysis**: Individual agent behavior analysis

### 3. Resource Analysis
- **ResourceImpactAnalyzer**: Resource utilization and efficiency analysis

### 4. Population Analysis
- **PopulationAnalyzer**: Population dynamics and statistics

### 5. Learning Analysis
- **LearningAnalyzer**: Learning progress and adaptation analysis

## Common Interface

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

### Analysis Parameters

- **scope**: Analysis scope (SIMULATION, EPISODE, AGENT, STEP)
- **agent_id**: Filter for specific agent (optional)
- **step**: Analyze specific timestep (optional)
- **step_range**: Analyze range of timesteps (optional)

### Analysis Scopes

- **SIMULATION**: Complete simulation data
- **EPISODE**: Specific episode or time period
- **AGENT**: Individual agent analysis
- **STEP**: Single simulation step

## Usage Patterns

### Basic Analysis
```python
from farm.database.analyzers.action_stats_analyzer import ActionStatsAnalyzer
from farm.database.repositories.action_repository import ActionRepository

repository = ActionRepository(session)
analyzer = ActionStatsAnalyzer(repository)
results = analyzer.analyze(scope="SIMULATION")
```

### Agent-Specific Analysis
```python
results = analyzer.analyze(
    scope="SIMULATION",
    agent_id="agent_001"
)
```

### Time-Based Analysis
```python
results = analyzer.analyze(
    scope="EPISODE",
    step_range=(100, 200)
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

## Integration with Services

Analyzers are typically used through high-level services:

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

## Best Practices

### 1. Start with Basic Analysis
- Begin with simple statistics and metrics
- Add complexity as needed
- Use appropriate scopes for your use case

### 2. Combine Analyzers
- Use multiple analyzers for comprehensive insights
- Cross-reference results between analyzers
- Look for patterns across different analysis types

### 3. Consider Context
- Account for simulation parameters
- Consider environmental factors
- Compare against population averages

### 4. Validate Results
- Check for data quality issues
- Verify statistical significance
- Consider alternative interpretations

## Error Handling

All analyzers handle common issues:
- Missing data gracefully
- Incomplete histories
- Invalid time ranges
- Null values in metrics

## Cross-References

For detailed information on specific analyzers:

- **Action Analysis**: [Action Stats Analyzer](ActionStatsAnalyzer.md)
- **Behavioral Analysis**: [Behavior Clustering Analyzer](BehaviorClusteringAnalyzer.md)
- **Causal Analysis**: [Causal Analyzer](CausalAnalyzer.md)
- **Decision Patterns**: [Decision Pattern Analyzer](DecisionPatternAnalyzer.md)
- **Resource Analysis**: [Resource Impact Analyzer](ResourceImpactAnalyzer.md)
- **Sequence Analysis**: [Sequence Pattern Analyzer](SequencePatternAnalyzer.md)
- **Temporal Analysis**: [Temporal Pattern Analyzer](TemporalPatternAnalyzer.md)
- **Population Analysis**: [Population Analyzer](PopulationAnalyzer.md)
- **Learning Analysis**: [Learning Analyzer](LearningAnalyzer.md)

## Notes

- All analyzers support filtering by scope, agent_id, step, and step_range
- Most analyzers provide both detailed and summary statistics
- Analysis results are returned as structured data types for consistency
- Many analyzers support multiple analysis methods and configurations
- Analyzers integrate with the repository pattern for data access
- Services provide high-level coordination between multiple analyzers 