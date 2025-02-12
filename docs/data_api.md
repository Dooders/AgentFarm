# Data API Documentation

## Introduction

This document provides an overview of the Data API being developed, which is designed to support simulation state persistence, analysis, and data retrieval functionalities. The API is structured into several key components, including the Database module, Analysis module, Repositories, and Services. Each component plays a vital role in managing data flow and processing within the simulation environment.

## Overview of the Data API Components

### Database Module (`database`)

The Database module is responsible for handling all interactions with the underlying database using SQLAlchemy ORM. It provides classes and methods to persist simulation data, manage configurations, and ensure data integrity. Key components of the Database module include:

- **`SimulationDatabase`**: The main database interface class that manages connections, sessions, and provides high-level methods for data operations.

- **Models** (`database/models.py`): Defines the ORM models representing different entities in the simulation, such as `Agent`, `AgentState`, `ResourceState`, `SimulationStep`, and others.

- **Data Retrieval** (`database/data_retrieval.py`): Contains classes like `DataRetriever` that facilitate querying and fetching data from the database for analysis purposes.

### Analysis Module (`analysis`)

The Analysis module provides various analytical tools and methods to process and analyze simulation data. It includes classes and functions to perform statistical analysis, clustering, causal inference, and pattern recognition on the data retrieved from the database. Key components of the Analysis module include:

- **`ActionStatsAnalyzer`** (`analysis/action_stats_analyzer.py`): Analyzes statistics and patterns of agent actions in a simulation.

- **`BehaviorClusteringAnalyzer`** (`analysis/behavior_clustering_analyzer.py`): Analyzes agent behaviors and clusters them based on their action patterns.

- **`CausalAnalyzer`** (`analysis/causal_analyzer.py`): Analyzes causal relationships between agent actions and their outcomes.

- **`DecisionPatternAnalyzer`** (`analysis/decision_pattern_analyzer.py`): Analyzes decision patterns from agent actions to identify behavioral trends and statistics.

- **`TemporalPatternAnalyzer`** (`analysis/temporal_pattern_analyzer.py`): Analyzes temporal patterns in agent actions over time.

- **`SequencePatternAnalyzer`** (`analysis/sequence_pattern_analyzer.py`): Analyzes sequences of agent actions to identify patterns and their probabilities.

- **`ResourceImpactAnalyzer`** (`analysis/resource_impact_analyzer.py`): Analyzes the resource impact of agent actions in a simulation.

- **`HealthResourceDynamics`** (`analysis/health_resource_dynamics.py`): Analyzes health and resource dynamics of agents over time using clustering and statistical methods.

- **`LearningExperienceAnalyzer`** (`analysis/learning_experience.py`): Analyzes learning experiences and performance metrics from simulation data.

- **`RewardEfficiencyAnalyzer`** (`analysis/reward_efficiency.py`): Analyzes reward efficiency by action type and agent group.

- **`SimulationAnalyzer`** (`analysis/simulation_analyzer.py`): Provides overall analysis of the simulation, including survival rates, resource distribution, and population balance.

Each analyzer focuses on a specific aspect of the data, allowing for modular and extensible analysis capabilities.

### Repositories (`database/repositories`)

Repositories act as data access layers that encapsulate the logic required to access data sources. They provide methods to query the database and retrieve data in a structured way, helping to decouple data access from business logic. Key repositories include:

- **`AgentActionRepository`** (`database/repositories/agent_action_repository.py`): Provides methods to query agent actions based on various filters and scopes.

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
          scope : str
              The scope to filter actions by (e.g., 'SIMULATION', 'EPISODE')
          agent_id : Optional[int], optional
              Specific agent ID to filter by. Defaults to None.
          """
          # Implementation details...
  ```

### Services (`services`)

Services provide high-level interfaces for performing complex operations using the repositories and analyzers. They orchestrate the flow of data between the repositories and the analysis modules, offering convenient methods for application logic to consume. Key services include:

- **`ActionsService`** (`services/actions_service.py`): High-level service for analyzing agent actions using various analyzers.

  ```python:services/actions_service.py
  class ActionsService:
      """
      High-level service for analyzing agent actions using various analyzers.

      This service orchestrates different types of analysis on agent actions including:
      - Basic action statistics and metrics
      - Behavioral patterns and clustering
      - Causal relationships
      - Decision patterns
      - Resource impacts
      - Action sequences
      - Temporal patterns

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
      ) -> Dict[str, Union[Any]]:
          """
          Perform comprehensive analysis on agent actions.

          Args:
              scope (Union[str, AnalysisScope], optional): Scope of analysis. Defaults to SIMULATION.
              agent_id (Optional[int], optional): Specific agent ID to analyze. Defaults to None.
              step (Optional[int], optional): Specific simulation step to analyze. Defaults to None.
              step_range (Optional[Tuple[int, int]], optional): Range of steps to analyze. Defaults to None.
              analysis_types (Optional[List[str]], optional): List of analysis types to perform. Defaults to all.

          Returns:
              Dict[str, Union[Any]]: Dictionary containing analysis results keyed by analysis type.
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

### Interaction Between Components

The Data API is designed with a modular architecture where each component interacts with others to perform its functions:

- The **Services** use **Repositories** to access data from the **Database**.

- **Analyzers** in the **Analysis Module** rely on **Repositories** to retrieve data required for analysis.

- **Services** coordinate **Analyzers** to perform complex operations and provide high-level interfaces for application logic.

- The **Database Module** provides the foundational data models and data access mechanisms that the other components build upon.

## Usage Example

Here's an example of how the components might be used together to perform an analysis:

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

The Data API provides a structured and extensible framework for managing simulation data and performing analyses. By separating concerns across the Database, Analysis, Repositories, and Services modules, the API facilitates maintainability and scalability. Users can leverage the high-level Services to perform complex analyses without needing to delve into the underlying data access and processing logic.
