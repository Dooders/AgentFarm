"""Core interfaces for decoupling dependencies in the AgentFarm system.

This module defines protocols and abstract base classes that allow different
components to communicate without tight coupling.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar


class AgentProtocol(Protocol):
    """Protocol defining the interface that agents must implement for metrics tracking."""

    agent_id: str
    position: Tuple[float, float]
    resource_level: float
    current_health: float
    starting_health: float
    starvation_counter: int
    is_defending: bool
    total_reward: float
    birth_time: float
    alive: bool
    genome_id: Optional[str]
    generation: int

    def get_state_dict(self) -> Dict[str, Any]:
        """Get the agent's state as a dictionary."""
        ...


class MetricsTrackerProtocol(Protocol):
    """Protocol for metrics tracking functionality."""

    def record_birth(self) -> None:
        """Record a birth event."""
        ...

    def record_death(self) -> None:
        """Record a death event."""
        ...

    def record_combat_encounter(self) -> None:
        """Record a combat encounter."""
        ...

    def record_successful_attack(self) -> None:
        """Record a successful attack."""
        ...

    def record_resources_shared(self, amount: float) -> None:
        """Record resources shared between agents."""
        ...

    def calculate_metrics(
        self, agent_objects: Dict[str, AgentProtocol], resources: Any, time: int, config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Calculate metrics for the current simulation state."""
        ...

    def update_metrics(
        self,
        metrics: Dict[str, Any],
        db: Optional[Any] = None,
        time: Optional[int] = None,
        agent_objects: Optional[Dict[str, AgentProtocol]] = None,
        resources: Optional[Any] = None,
    ) -> None:
        """Update environment metrics and log to database."""
        ...


class EnvironmentProtocol(Protocol):
    """Protocol defining the interface that environments must implement."""

    def get_agents(self) -> Dict[str, AgentProtocol]:
        """Get all agents in the environment."""
        ...

    def get_resources(self) -> Any:
        """Get all resources in the environment."""
        ...

    def get_current_time(self) -> int:
        """Get the current simulation time."""
        ...


class DatabaseProtocol(Protocol):
    """Enhanced protocol for database operations.
    
    This protocol defines the interface for database instances that provide
    data logging, repository access, and configuration management. Implementations
    should support both logging operations and data retrieval.
    """

    @property
    def logger(self) -> "DataLoggerProtocol":
        """Get the data logger instance.
        
        Returns
        -------
        DataLoggerProtocol
            The logger for buffered data operations
        """
        ...

    def log_step(self, step_number: int, agent_states: Any, resource_states: Any, metrics: Dict[str, Any]) -> None:
        """Log a simulation step to the database.
        
        Parameters
        ----------
        step_number : int
            Current simulation step
        agent_states : Any
            Collection of agent states
        resource_states : Any
            Collection of resource states
        metrics : Dict[str, Any]
            Step-level metrics
        """
        ...

    def export_data(self, filepath: str, format: str = "csv", **kwargs) -> None:
        """Export simulation data to a file.
        
        Parameters
        ----------
        filepath : str
            Path where the export file will be saved
        format : str
            Export format (csv, excel, json, parquet)
        **kwargs : dict
            Additional export options
        """
        ...

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        ...

    def get_configuration(self) -> Dict[str, Any]:
        """Retrieve the current simulation configuration.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        ...

    def save_configuration(self, config: Dict[str, Any]) -> None:
        """Save simulation configuration to the database.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to save
        """
        ...


class DatabaseFactoryProtocol(Protocol):
    """Protocol for database factory operations."""

    def setup_db(
        self, db_path: Optional[str], simulation_id: str, config: Optional[Dict[str, Any]] = None
    ) -> "DatabaseProtocol":
        """Setup and initialize a database instance."""
        ...


class ChartAnalyzerProtocol(Protocol):
    """Protocol for chart analysis functionality."""

    def analyze_all_charts(self, output_dir: Optional[str] = None, database: Optional[Any] = None) -> Dict[str, Any]:
        """Analyze all charts and return results."""
        ...


# Generic type variable for repository protocol
T = TypeVar("T")


class DataLoggerProtocol(Protocol):
    """Protocol for data logging operations.
    
    This protocol defines the interface for logging simulation data including
    agent actions, states, health incidents, and other events. Implementations
    should handle buffered batch operations for performance.
    """

    def log_agent_action(
        self,
        step_number: int,
        agent_id: str,
        action_type: str,
        action_target_id: Optional[str] = None,
        resources_before: Optional[float] = None,
        resources_after: Optional[float] = None,
        reward: Optional[float] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Log an agent action.
        
        Parameters
        ----------
        step_number : int
            Current simulation step
        agent_id : str
            ID of the agent performing the action
        action_type : str
            Type of action being performed
        action_target_id : Optional[str]
            ID of the target (if any)
        resources_before : Optional[float]
            Resource level before action
        resources_after : Optional[float]
            Resource level after action
        reward : Optional[float]
            Reward received for action
        details : Optional[Dict]
            Additional action details
        """
        ...

    def log_step(
        self,
        step_number: int,
        agent_states: List[Tuple],
        resource_states: List[Tuple],
        metrics: Dict[str, Any],
    ) -> None:
        """Log a complete simulation step.
        
        Parameters
        ----------
        step_number : int
            Current simulation step
        agent_states : List[Tuple]
            List of agent state tuples
        resource_states : List[Tuple]
            List of resource state tuples
        metrics : Dict[str, Any]
            Step metrics
        """
        ...

    def log_agent(
        self,
        agent_id: str,
        birth_time: int,
        agent_type: str,
        position: Tuple[float, float],
        initial_resources: float,
        starting_health: float,
        starvation_counter: int,
        genome_id: Optional[str] = None,
        generation: int = 0,
        action_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a new agent creation.
        
        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent
        birth_time : int
            Time step when agent was created
        agent_type : str
            Type of agent
        position : Tuple[float, float]
            Initial (x, y) coordinates
        initial_resources : float
            Starting resource level
        starting_health : float
            Maximum health points
        starvation_counter : int
            Current starvation counter value
        genome_id : Optional[str]
            Genome identifier
        generation : int
            Generation number
        action_weights : Optional[Dict[str, float]]
            Action weights/probabilities
        """
        ...

    def log_health_incident(
        self,
        step_number: int,
        agent_id: str,
        health_before: float,
        health_after: float,
        cause: str,
        details: Optional[Dict] = None,
    ) -> None:
        """Log a health incident.
        
        Parameters
        ----------
        step_number : int
            Current simulation step
        agent_id : str
            ID of affected agent
        health_before : float
            Health before incident
        health_after : float
            Health after incident
        cause : str
            Cause of health change
        details : Optional[Dict]
            Additional incident details
        """
        ...

    def flush_all_buffers(self) -> None:
        """Flush all buffered data to the database."""
        ...


class RepositoryProtocol(Protocol[T]):
    """Generic repository protocol for data access operations.
    
    This protocol defines the standard CRUD interface for accessing and
    manipulating entities in the database. Implementations should provide
    transaction safety and error handling.
    """

    def add(self, entity: T) -> None:
        """Add a new entity to the repository.
        
        Parameters
        ----------
        entity : T
            The entity to add
        """
        ...

    def get_by_id(self, entity_id: Any) -> Optional[T]:
        """Retrieve an entity by its ID.
        
        Parameters
        ----------
        entity_id : Any
            The ID of the entity
            
        Returns
        -------
        Optional[T]
            The entity if found, None otherwise
        """
        ...

    def update(self, entity: T) -> None:
        """Update an existing entity.
        
        Parameters
        ----------
        entity : T
            The entity to update
        """
        ...

    def delete(self, entity: T) -> None:
        """Delete an entity from the repository.
        
        Parameters
        ----------
        entity : T
            The entity to delete
        """
        ...
