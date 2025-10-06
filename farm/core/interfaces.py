"""Core interfaces for decoupling dependencies in the AgentFarm system.

This module defines protocols and abstract base classes that allow different
components to communicate without tight coupling.
"""

from typing import Any, Dict, Optional, Protocol, Tuple


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
    """Protocol for database operations."""

    def log_step(self, step_number: int, agent_states: Any, resource_states: Any, metrics: Dict[str, Any]) -> None:
        """Log a simulation step to the database."""
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
