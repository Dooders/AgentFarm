# Actions Analysis Module

The Actions Analysis Module provides comprehensive analysis of agent actions, interactions, and decision-making patterns within the simulation database. It enables deep insights into agent behaviors, strategies, and performance.

The module is structured into several key components:

- **Repositories**: Data access layers that retrieve agent action data from the database.
- **Analyzers**: Classes that process action data to generate insights and analytics.
- **Services**: High-level interfaces that coordinate repositories and analyzers to perform complex analyses.

## Key Components

### AgentActionRepository

The `AgentActionRepository` class provides methods to query and retrieve agent actions based on various filtering criteria such as scope, agent ID, and step numbers.

```python:database/repositories/agent_action_repository.py
class AgentActionRepository:
    """
    Repository class for managing agent action records in the database.

    This class provides methods to query and retrieve agent actions based on various
    filtering criteria such as scope, agent ID, and step numbers.

    Args:
        session_manager (SessionManager): Session manager for database operations.
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize repository with session manager.

        Parameters
        ----------
        session_manager : SessionManager
            Session manager instance for database operations
        """
        self.session_manager = session_manager

    def get_actions_by_scope(
        self,
        scope: Union[str, AnalysisScope],
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[AgentActionData]:
        """Retrieve agent actions filtered by scope and other optional parameters.

        Parameters
        ----------
        scope : Union[str, AnalysisScope]
            The scope to filter actions by (e.g., 'SIMULATION', 'EPISODE')
        agent_id : Optional[int], optional
            Specific agent ID to filter by. Defaults to None.
        """
        # Implementation details...
```

### Analyzers

The Analysis module provides various analytical tools and methods to process and analyze simulation data. Each analyzer focuses on a specific aspect of the data, allowing for modular and extensible analysis capabilities.

#### ActionStatsAnalyzer

Analyzes statistics and patterns of agent actions in a simulation.

```python:analysis/action_stats_analyzer.py
class ActionStatsAnalyzer:
    """
    Analyzes statistics and patterns of agent actions in a simulation.

    This class processes action data to generate metrics including frequency, rewards,
    interaction rates, and various patterns of agent behavior.
    """

    def __init__(self, repository: AgentActionRepository):
        """
        Initialize the ActionStatsAnalyzer.

        Args:
            repository (AgentActionRepository): Repository for accessing agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> List[ActionMetrics]:
        """
        Analyze action statistics based on specified scope and filters.

        Returns:
            List[ActionMetrics]: List of action metrics objects containing statistical data.
        """
        # Implementation details...
```

#### Other Analyzers

- **BehaviorClusteringAnalyzer**: Groups agents by behavioral patterns and strategies.
- **CausalAnalyzer**: Analyzes cause-effect relationships for actions.
- **DecisionPatternAnalyzer**: Analyzes comprehensive decision-making patterns.
- **ResourceImpactAnalyzer**: Analyzes resource impacts of different actions.
- **SequencePatternAnalyzer**: Analyzes sequential action patterns and transitions.
- **TemporalPatternAnalyzer**: Analyzes action patterns over time.

### ActionsService

The `ActionsService` class provides a high-level interface for analyzing agent actions using various analyzers. It orchestrates different types of analysis on agent actions including:

- Basic action statistics and metrics
- Behavioral patterns and clustering
- Causal relationships
- Decision patterns
- Resource impacts
- Action sequences
- Temporal patterns

```python:services/actions_service.py
class ActionsService:
    """
    High-level service for analyzing agent actions using various analyzers.

    Attributes:
        action_repository (AgentActionRepository): Repository for accessing agent action data
        stats_analyzer (ActionStatsAnalyzer): Analyzer for basic action statistics
        behavior_analyzer (BehaviorClusteringAnalyzer): Analyzer for behavioral patterns
        causal_analyzer (CausalAnalyzer): Analyzer for causal relationships
        decision_analyzer (DecisionPatternAnalyzer): Analyzer for decision patterns
        resource_analyzer (ResourceImpactAnalyzer): Analyzer for resource impacts
        sequence_analyzer (SequencePatternAnalyzer): Analyzer for action sequences
        temporal_analyzer (TemporalPatternAnalyzer): Analyzer for temporal patterns
    """

    def __init__(self, action_repository: AgentActionRepository):
        """
        Initialize ActionsService with required repositories and analyzers.

        Args:
            action_repository (AgentActionRepository): Repository for accessing agent action data
        """
        self.action_repository = action_repository

        # Initialize analyzers
        self.stats_analyzer = ActionStatsAnalyzer(action_repository)
        self.behavior_analyzer = BehaviorClusteringAnalyzer(action_repository)
        self.causal_analyzer = CausalAnalyzer(action_repository)
        self.decision_analyzer = DecisionPatternAnalyzer(action_repository)
        self.resource_analyzer = ResourceImpactAnalyzer(action_repository)
        self.sequence_analyzer = SequencePatternAnalyzer(action_repository)
        self.temporal_analyzer = TemporalPatternAnalyzer(action_repository)

    def analyze_actions(
        self,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
        analysis_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on agent actions.

        Args:
            scope (Union[str, AnalysisScope], optional): Scope of analysis. Defaults to SIMULATION.
            agent_id (Optional[int], optional): Specific agent ID to analyze. Defaults to None.
            step (Optional[int], optional): Specific simulation step to analyze. Defaults to None.
            step_range (Optional[Tuple[int, int]], optional): Range of steps to analyze. Defaults to None.
            analysis_types (Optional[List[str]], optional): List of analysis types to perform. Defaults to all.

        Returns:
            Dict[str, Any]: Dictionary containing analysis results keyed by analysis type.
        """
        results = {}

        if analysis_types is None:
            analysis_types = [
                "stats", "behavior", "causal", "decision",
                "resource", "sequence", "temporal"
            ]

        # Basic action statistics
        if "stats" in analysis_types:
            results["action_stats"] = self.stats_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        # Behavioral clustering
        if "behavior" in analysis_types:
            results["behavior_clusters"] = self.behavior_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        # Causal analysis
        if "causal" in analysis_types:
            results["causal_analysis"] = self.causal_analyzer.analyze(
                action_type='attack',  # Example action type
                scope=scope,
                agent_id=agent_id,
                step_range=step_range
            )

        # Decision pattern analysis
        if "decision" in analysis_types:
            results["decision_patterns"] = self.decision_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        # Resource impact analysis
        if "resource" in analysis_types:
            results["resource_impacts"] = self.resource_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        # Sequence pattern analysis
        if "sequence" in analysis_types:
            results["sequence_patterns"] = self.sequence_analyzer.analyze(
                scope, agent_id, step, step_range
            )

        # Temporal pattern analysis
        if "temporal" in analysis_types:
            results["temporal_patterns"] = self.temporal_analyzer.analyze(
                scope, agent_id, step_range
            )

        return results
```

## Data Types

- **AgentActionData**: Structured representation of individual actions.
- **ActionMetrics**: Statistical metrics for action types.
- **TimePattern**: Temporal evolution patterns.
- **ResourceImpact**: Resource consumption/generation metrics.
- **DecisionPatterns**: Decision-making analysis results.
- **SequencePattern**: Action sequence statistics.
- **CausalAnalysis**: Cause-effect relationship data.
- **BehaviorClustering**: Agent behavioral groupings.

## Usage Example

```python
from database.session_manager import SessionManager
from database.repositories.agent_action_repository import AgentActionRepository
from services.actions_service import ActionsService

# Initialize session manager and repositories
session_manager = SessionManager('sqlite:///simulation.db')
action_repository = AgentActionRepository(session_manager)

# Initialize services
actions_service = ActionsService(action_repository)

# Perform comprehensive analysis on agent actions
results = actions_service.analyze_actions(
    scope='SIMULATION',
    agent_id=123,
    analysis_types=['stats', 'behavior', 'causal']
)

# Access analysis results
action_stats = results.get('action_stats')
behavior_clusters = results.get('behavior_clusters')
causal_analysis = results.get('causal_analysis')
```

## Conclusion

The updated Actions Analysis Module leverages a modular architecture that separates concerns across Repositories, Analyzers, and Services. This design enhances maintainability and scalability by decoupling data access from processing logic and providing high-level interfaces for complex analyses. Users can utilize the `ActionsService` to perform comprehensive analysis on agent actions without needing to manage the underlying data retrieval and analysis processes.
