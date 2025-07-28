# Data API Documentation

## Introduction

This document provides an overview of the Data API architecture, which is designed to support simulation state persistence, analysis, and data retrieval functionalities. The API is structured into several key components that work together to manage data flow and processing within the simulation environment.

## Architecture Overview

The Data API follows a layered architecture with clear separation of concerns:

```
┌─────────────────┐
│   Services      │  High-level coordination
├─────────────────┤
│  Repositories   │  Data access layer
├─────────────────┤
│   Analyzers     │  Analysis implementations
├─────────────────┤
│   Database      │  Data storage & models
└─────────────────┘
```

### Component Relationships

- **Services** orchestrate complex operations using repositories and analyzers
- **Repositories** provide data access interfaces to the database
- **Analyzers** process data into meaningful metrics and insights
- **Database** stores simulation data using SQLAlchemy ORM

## Core Components

### Database Module

The Database module handles all interactions with the underlying database using SQLAlchemy ORM. For detailed schema information and data model overview, see [Database Schema](database_schema.md).

Key components:
- **`SimulationDatabase`**: Main database interface for single simulations
- **`ExperimentDatabase`**: Extended interface for multi-simulation experiments
- **Models**: ORM models representing simulation entities
- **Data Retrieval**: Classes for querying and fetching data

### Analysis Module

The Analysis module provides various analytical tools for processing simulation data. For comprehensive analyzer documentation, see [Analysis Overview](analysis/Analysis.md).

Available analyzers include:
- **ActionStatsAnalyzer**: Action statistics and patterns
- **BehaviorClusteringAnalyzer**: Behavioral clustering and grouping
- **CausalAnalyzer**: Causal relationships between actions
- **DecisionPatternAnalyzer**: Decision patterns and trends
- **ResourceImpactAnalyzer**: Resource impact analysis
- **TemporalPatternAnalyzer**: Temporal patterns and trends
- **SequencePatternAnalyzer**: Action sequence analysis
- **PopulationAnalyzer**: Population-level statistics
- **AgentAnalyzer**: Individual agent analysis
- **LearningAnalyzer**: Learning experience analysis

### Repositories

Repositories act as data access layers that encapsulate database query logic. They provide methods to query the database and retrieve data in a structured way, helping to decouple data access from business logic.

Key repositories:
- **`ActionRepository`**: Query agent actions with filtering and analysis
- **`AgentRepository`**: Access agent data, states, and lifecycle information
- **`PopulationRepository`**: Population-level statistics and dynamics
- **`ResourceRepository`**: Resource state and distribution analysis
- **`LearningRepository`**: Learning experience and module performance data
- **`SimulationRepository`**: Simulation metadata and configuration
- **`GUIRepository`**: GUI-specific data access patterns

**Detailed Documentation**: [Repository Documentation](repositories.md)

### Services

Services provide high-level interfaces for performing complex operations. For detailed service documentation, see [Services Documentation](data_services.md).

Available services:
- **`ActionsService`**: Comprehensive action analysis and coordination
- **`PopulationService`**: Population-level analysis and statistics

**Detailed Documentation**: [Services Documentation](data_services.md)

## Component Interactions

The Data API is designed with a modular architecture where each component interacts with others to perform its functions:

- **Services** use **Repositories** to access data from the **Database**
- **Analyzers** rely on **Repositories** to retrieve data required for analysis
- **Services** coordinate **Analyzers** to perform complex operations
- The **Database Module** provides the foundational data models and access mechanisms

## Data Flow

### Typical Usage Pattern

1. **Initialize Components**: Set up database connection and repositories
2. **Configure Services**: Create service instances with appropriate repositories
3. **Execute Analysis**: Use services to coordinate multiple analyzers
4. **Process Results**: Handle structured analysis results

### Integration Points

- **Database Layer**: Provides data persistence and retrieval
- **Repository Layer**: Offers data access abstractions
- **Analyzer Layer**: Implements specific analysis algorithms
- **Service Layer**: Coordinates complex operations across components

## Cross-References

For detailed information on specific components:

- **Database Schema & Model**: [Database Schema](database_schema.md)
- **Analysis Capabilities**: [Analysis Overview](analysis/Analysis.md)
- **Service Documentation**: [Services Documentation](data_services.md)
- **Data Retrieval**: [Data Retrieval](data_retrieval.md)

## Architecture Principles

### 1. **Separation of Concerns**
- Each component has a specific responsibility
- Clear interfaces between layers
- Minimal coupling between components

### 2. **Modularity**
- Components can be used independently
- Easy to extend with new analyzers or services
- Consistent interfaces across components

### 3. **Performance Optimization**
- Efficient data access patterns
- Caching at appropriate layers
- Optimized query strategies

### 4. **Error Handling**
- Graceful degradation
- Meaningful error messages
- Data consistency guarantees

## Best Practices

1. **Use Services for High-Level Operations**
   - Services provide the most convenient interfaces
   - They handle coordination between multiple components
   - Error handling and validation are built-in

2. **Choose Appropriate Analysis Types**
   - Request only needed analysis types for performance
   - Use specific analyzers for focused analysis
   - Consider data volume when selecting scopes

3. **Leverage Repository Patterns**
   - Use repositories for custom data access patterns
   - Implement caching where appropriate
   - Follow established query patterns

4. **Database Considerations**
   - Use appropriate database contexts for multi-simulation
   - Consider performance implications of queries
   - Leverage indexes for common query patterns

## Usage Examples

### Basic Action Analysis
```python
from farm.database.repositories.action_repository import ActionRepository
from farm.database.services.actions_service import ActionsService

# Initialize repository and service
action_repo = ActionRepository(session)
actions_service = ActionsService(action_repo)

# Perform comprehensive action analysis
results = actions_service.analyze_actions(
    scope="SIMULATION",
    analysis_types=['stats', 'behavior', 'causal']
)
```

### Population Analysis
```python
from farm.database.repositories.population_repository import PopulationRepository
from farm.database.analyzers.population_analyzer import PopulationAnalyzer

# Initialize repository and analyzer
pop_repo = PopulationRepository(session)
pop_analyzer = PopulationAnalyzer(pop_repo)

# Get population statistics
stats = pop_analyzer.analyze_comprehensive_statistics()
```

### Agent Analysis
```python
from farm.database.repositories.agent_repository import AgentRepository
from farm.database.analyzers.agent_analyzer import AgentAnalyzer

# Initialize repository and analyzer
agent_repo = AgentRepository(session)
agent_analyzer = AgentAnalyzer(agent_repo)

# Analyze specific agent
agent_stats = agent_analyzer.analyze(agent_id="agent_001")
```

## Conclusion

The Data API provides a structured and extensible framework for managing simulation data and performing analyses. By separating concerns across the Database, Analysis, Repositories, and Services modules, the API facilitates maintainability and scalability. Users can leverage the high-level Services to perform complex analyses without needing to delve into the underlying data access and processing logic.

For implementation details and specific usage examples, refer to the specialized documentation files listed in the Cross-References section.
