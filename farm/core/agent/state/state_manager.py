"""
State management for agents.

Manages agent state including position, orientation, genealogy, and metadata.
Provides centralized state access and ensures consistency across the agent.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentGenealogy:
    """Genealogy information for tracking agent lineage."""
    
    generation: int = 0
    """Generation number (0 for first generation)."""
    
    parent_ids: list[str] = None
    """IDs of parent agents."""
    
    genome_id: str = ""
    """Unique genome identifier."""
    
    birth_time: int = 0
    """Simulation step when agent was created."""
    
    death_time: Optional[int] = None
    """Simulation step when agent died, None if still alive."""
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.parent_ids is None:
            self.parent_ids = []


@dataclass
class StateSnapshot:
    """Complete snapshot of agent state at a point in time."""
    
    agent_id: str
    position: tuple[float, float]
    orientation: float = 0.0
    """Heading in degrees (0 = north/up, 90 = east/right)."""
    
    # Genealogy
    generation: int = 0
    parent_ids: list[str] = None
    genome_id: str = ""
    birth_time: int = 0
    death_time: Optional[int] = None
    
    # Components delegate their state here
    resource_level: float = 0.0
    health: float = 100.0
    is_defending: bool = False
    defense_timer: int = 0
    
    # Metadata
    total_reward: float = 0.0
    age: int = 0
    alive: bool = True
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.parent_ids is None:
            self.parent_ids = []


class StateManager:
    """
    Manages agent state including position, orientation, and genealogy.
    
    Provides centralized state management that components can update through
    well-defined methods, ensuring consistency and making state changes
    auditable for debugging and analysis.
    """
    
    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        generation: int = 0,
        parent_ids: Optional[list[str]] = None,
        genome_id: str = "",
        birth_time: int = 0,
    ):
        """
        Initialize state manager.
        
        Args:
            agent_id: Unique agent identifier
            position: Initial (x, y) position
            generation: Generation number
            parent_ids: IDs of parent agents
            genome_id: Unique genome identifier
            birth_time: Simulation step when born
        """
        self.agent_id = agent_id
        self.position = position
        self.orientation = 0.0
        
        # Genealogy
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.genome_id = genome_id
        self.birth_time = birth_time
        self.death_time: Optional[int] = None
        
        # Component state (updated by components)
        self.resource_level = 0.0
        self.health = 100.0
        self.is_defending = False
        self.defense_timer = 0
        self.total_reward = 0.0
        self.alive = True
    
    def update_position(self, position: tuple[float, float]) -> None:
        """Update agent position."""
        self.position = position
    
    def update_orientation(self, orientation: float) -> None:
        """Update agent orientation (heading in degrees)."""
        self.orientation = orientation
    
    def set_dead(self, death_time: int) -> None:
        """Mark agent as dead."""
        self.alive = False
        self.death_time = death_time
    
    def snapshot(self, current_time: int) -> StateSnapshot:
        """
        Create a snapshot of current state.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            StateSnapshot: Complete state snapshot
        """
        return StateSnapshot(
            agent_id=self.agent_id,
            position=self.position,
            orientation=self.orientation,
            generation=self.generation,
            parent_ids=self.parent_ids.copy(),
            genome_id=self.genome_id,
            birth_time=self.birth_time,
            death_time=self.death_time,
            resource_level=self.resource_level,
            health=self.health,
            is_defending=self.is_defending,
            defense_timer=self.defense_timer,
            total_reward=self.total_reward,
            age=current_time - self.birth_time if self.birth_time >= 0 else 0,
            alive=self.alive,
        )
