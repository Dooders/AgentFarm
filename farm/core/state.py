from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from farm.core.environment import Environment


class BaseState(BaseModel):
    """Base class for all state representations in the simulation.

    This class provides common functionality and validation for state objects.
    All state values should be normalized to range [0,1] for stable learning
    and consistent processing across different state implementations.

    Attributes:
        DIMENSIONS (ClassVar[int]): Number of dimensions in the state vector

    Methods:
        to_tensor: Convert state to tensor format for neural network input
        to_dict: Convert state to dictionary representation
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
        arbitrary_types_allowed=True
    )

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert state to tensor format for neural network input.

        Subclasses must implement this to define how their attributes are
        converted into a numeric tensor suitable for model input.

        Args:
            device: Target device (CPU/GPU)

        Returns:
            torch.Tensor: Tensor representation of this state
        """
        raise NotImplementedError("Subclasses must implement to_tensor")

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary with human-readable keys.

        Subclasses must implement this to return a stable, serialization-
        friendly mapping of state values for logging, DB storage, or export.

        Returns:
            Dict[str, Any]: Dictionary representation of state
        """
        raise NotImplementedError("Subclasses must implement to_dict")


class AgentState(BaseState):
    """State representation for agent decision making.

    Captures the current state of an individual agent including its position,
    resources, health, genealogy, and other key attributes. All continuous values are
    normalized to [0,1] for consistency with other state representations.

    Attributes:
        agent_id (str): Unique identifier for the agent
        step_number (int): Current simulation step
        position_x (float): Normalized X coordinate in environment
        position_y (float): Normalized Y coordinate in environment
        position_z (float): Normalized Z coordinate (usually 0 for 2D)
        resource_level (float): Current resource amount [0,1]
        current_health (float): Current health level [0,1]
        is_defending (bool): Whether agent is in defensive stance
        total_reward (float): Cumulative reward received
        age (int): Number of steps agent has existed
        generation (int): Generation number (0 for first generation)
        parent_ids (list[str]): IDs of parent agents
        genome_id (str): Unique genome identifier
        birth_time (int): Simulation step when agent was created
        death_time (Optional[int]): Simulation step when agent died, None if still alive
        orientation (float): Agent heading in degrees (0 = north/up, 90 = east/right)
        alive (bool): Whether agent is currently alive

    Example:
        >>> state = AgentState(
        ...     agent_id="agent_1",
        ...     step_number=100,
        ...     position_x=0.5,
        ...     position_y=0.3,
        ...     position_z=0.0,
        ...     resource_level=0.7,
        ...     current_health=0.9,
        ...     is_defending=False,
        ...     total_reward=10.5,
        ...     age=50,
        ...     generation=2,
        ...     parent_ids=["parent_1", "parent_2"],
        ...     genome_id="genome_123",
        ...     birth_time=50,
        ...     death_time=None,
        ...     orientation=45.0,
        ...     alive=True
        ... )
    """

    # Required fields from AgentStateModel
    agent_id: str
    step_number: int
    position_x: float
    position_y: float
    position_z: float
    resource_level: float
    current_health: float
    is_defending: bool
    total_reward: float
    age: int
    
    # Genealogy fields
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)
    genome_id: str = ""
    birth_time: int = 0
    death_time: Optional[int] = None
    
    # Additional agent state
    orientation: float = 0.0
    alive: bool = True

    @classmethod
    def from_raw_values(
        cls,
        agent_id: str,
        step_number: int,
        position_x: float,
        position_y: float,
        position_z: float,
        resource_level: float,
        current_health: float,
        is_defending: bool,
        total_reward: float,
        age: int,
        generation: int = 0,
        parent_ids: Optional[list[str]] = None,
        genome_id: str = "",
        birth_time: int = 0,
        death_time: Optional[int] = None,
        orientation: float = 0.0,
        alive: bool = True,
    ) -> "AgentState":
        """Create a state instance from raw values.

        Args:
            agent_id: Unique identifier for the agent
            step_number: Current simulation step
            position_x: X coordinate
            position_y: Y coordinate
            position_z: Z coordinate (usually 0 for 2D)
            resource_level: Current resource amount
            current_health: Current health level
            is_defending: Whether agent is in defensive stance
            total_reward: Cumulative reward received
            age: Number of steps agent has existed
            generation: Generation number (0 for first generation)
            parent_ids: IDs of parent agents
            genome_id: Unique genome identifier
            birth_time: Simulation step when agent was created
            death_time: Simulation step when agent died, None if still alive
            orientation: Agent heading in degrees (0 = north/up, 90 = east/right)
            alive: Whether agent is currently alive

        Returns:
            AgentState: State instance with provided values
        """
        return cls(
            agent_id=agent_id,
            step_number=step_number,
            position_x=position_x,
            position_y=position_y,
            position_z=position_z,
            resource_level=resource_level,
            current_health=current_health,
            is_defending=is_defending,
            total_reward=total_reward,
            age=age,
            generation=generation,
            parent_ids=parent_ids or [],
            genome_id=genome_id,
            birth_time=birth_time,
            death_time=death_time,
            orientation=orientation,
            alive=alive,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "step_number": self.step_number,
            "position": (self.position_x, self.position_y, self.position_z),
            "resource_level": self.resource_level,
            "current_health": self.current_health,
            "is_defending": self.is_defending,
            "total_reward": self.total_reward,
            "age": self.age,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "genome_id": self.genome_id,
            "birth_time": self.birth_time,
            "death_time": self.death_time,
            "orientation": self.orientation,
            "alive": self.alive,
        }

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert state to tensor format for neural network input."""
        return torch.FloatTensor(
            [
                self.position_x,
                self.position_y,
                self.position_z,
                self.resource_level,
                self.current_health,
                self.is_defending,
                self.total_reward,
                self.age,
                self.generation,
                self.orientation / 360.0,  # Normalize orientation to [0,1]
                float(self.alive),
            ]
        ).to(device)


class EnvironmentState(BaseState):
    """State representation for the simulation environment.

    Captures the overall state of the environment including resource distribution,
    agent populations, and global metrics. All values are normalized to [0,1].

    Attributes:
        normalized_resource_density (float): Density of resources in environment
        normalized_agent_density (float): Density of agents in environment
        normalized_system_ratio (float): Ratio of system agents to total agents
        normalized_resource_availability (float): Average resource amount availability
        normalized_time (float): Current simulation time relative to max steps
        DIMENSIONS (ClassVar[int]): Number of dimensions in state vector

    Example:
        >>> state = EnvironmentState(
        ...     normalized_resource_density=0.4,
        ...     normalized_agent_density=0.3,
        ...     normalized_system_ratio=0.6,
        ...     normalized_resource_availability=0.7,
        ...     normalized_time=0.5
        ... )
    """

    normalized_resource_density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Density of resources relative to environment area. "
        "0 = no resources, 1 = maximum expected density",
    )

    normalized_agent_density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Density of agents relative to environment area. "
        "0 = no agents, 1 = at population capacity",
    )

    normalized_system_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of system agents to total agents. "
        "0 = all independent, 1 = all system agents",
    )

    normalized_resource_availability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average resource amount across all resources. "
        "0 = all depleted, 1 = all at maximum capacity",
    )

    normalized_time: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current simulation time normalized by maximum steps. "
        "0 = start, 1 = end of simulation",
    )

    # Class constants
    DIMENSIONS: ClassVar[int] = 5
    MAX_EXPECTED_RESOURCES: ClassVar[int] = 100  # Adjust based on your simulation
    MAX_STEPS: ClassVar[int] = 1000  # Maximum simulation steps

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert environment state to tensor format.

        Args:
            device (torch.device): Device to place tensor on (CPU/GPU)

        Returns:
            torch.Tensor: 1D tensor of shape (DIMENSIONS,) containing state values
        """
        return torch.FloatTensor(
            [
                self.normalized_resource_density,
                self.normalized_agent_density,
                self.normalized_system_ratio,
                self.normalized_resource_availability,
                self.normalized_time,
            ]
        ).to(device)

    @classmethod
    def from_environment(cls, env: "Environment") -> "EnvironmentState":
        """Create a normalized state from an Environment instance.

        Args:
            env (Environment): Environment instance to create state from

        Returns:
            EnvironmentState: Normalized state representation

        Example:
            >>> state = EnvironmentState.from_environment(env)
        """
        # Calculate environment area
        env_area = env.width * env.height

        # Calculate densities
        resource_density = len(env.resources) / cls.MAX_EXPECTED_RESOURCES

        alive_agents = [a for a in env.agent_objects if a.alive]
        max_population = env.config.population.max_population if env.config else 3000
        agent_density = len(alive_agents) / max_population

        # Calculate system agent ratio (no system agents in current implementation)
        system_ratio = 0.0

        # Calculate resource availability
        max_possible = env.max_resource or (
            env.config.resources.max_resource_amount if env.config else 30
        )
        avg_resource = (
            sum(r.amount for r in env.resources) / (len(env.resources) * max_possible)
            if env.resources
            else 0.0
        )

        return cls(
            normalized_resource_density=resource_density,
            normalized_agent_density=agent_density,
            normalized_system_ratio=system_ratio,
            normalized_resource_availability=avg_resource,
            normalized_time=env.time / cls.MAX_STEPS,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert state to dictionary with descriptive keys.

        Returns:
            Dict[str, float]: Dictionary containing state values

        Example:
            >>> state.to_dict()
            {
                'resource_density': 0.4,
                'agent_density': 0.3,
                'system_agent_ratio': 0.6,
                'resource_availability': 0.7,
                'simulation_progress': 0.5
            }
        """
        return {
            "resource_density": self.normalized_resource_density,
            "agent_density": self.normalized_agent_density,
            "system_agent_ratio": self.normalized_system_ratio,
            "resource_availability": self.normalized_resource_availability,
            "simulation_progress": self.normalized_time,
        }


class ModelState(BaseModel):
    """State representation for machine learning models in the simulation.

    Captures the current state of a model including its learning parameters,
    performance metrics, and architecture information in their raw form.

    Attributes:
        learning_rate (float): Current learning rate
        epsilon (float): Current exploration rate
        latest_loss (Optional[float]): Most recent training loss
        latest_reward (Optional[float]): Most recent reward
        memory_size (int): Current number of experiences in memory
        memory_capacity (int): Maximum memory capacity
        steps (int): Total training steps taken
        architecture (Dict[str, Any]): Network architecture information
        training_metrics (Dict[str, float]): Recent training performance metrics

    Example:
        >>> state = ModelState.from_move_module(agent.move_module)
        >>> print(state.training_metrics['avg_reward'])
    """

    learning_rate: float = Field(
        ..., description="Current learning rate used by optimizer"
    )

    epsilon: float = Field(..., description="Current exploration rate (epsilon)")

    latest_loss: Optional[float] = Field(
        None, description="Most recent training loss value"
    )

    latest_reward: Optional[float] = Field(
        None, description="Most recent reward received"
    )

    memory_size: int = Field(..., description="Current number of experiences in memory")

    memory_capacity: int = Field(
        ..., description="Maximum capacity of experience memory"
    )

    steps: int = Field(..., description="Total number of training steps taken")

    architecture: Dict[str, Any] = Field(
        ..., description="Summary of network architecture including layer sizes"
    )

    training_metrics: Dict[str, float] = Field(
        ..., description="Recent training performance metrics"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True
    )

    @classmethod
    def from_move_module(cls, move_module) -> "ModelState":
        """Create a state representation from a MoveModule instance.

        Args:
            move_module (MoveModule): Move module instance to create state from

        Returns:
            ModelState: Current state of the move module

        Example:
            >>> state = ModelState.from_move_module(agent.move_module)
        """
        # Get architecture summary
        architecture = {
            "input_dim": move_module.q_network.network[0].in_features,
            "hidden_sizes": [
                layer.out_features
                for layer in move_module.q_network.network
                if isinstance(layer, torch.nn.Linear)
            ][:-1],
            "output_dim": move_module.q_network.network[-1].out_features,
        }

        # Get recent metrics
        recent_losses = [
            loss for loss in move_module.losses[-1000:] if loss is not None
        ]
        recent_rewards = move_module.episode_rewards[-1000:]

        training_metrics = {
            "avg_loss": (
                sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            ),
            "avg_reward": (
                sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            ),
            "min_reward": min(recent_rewards) if recent_rewards else 0.0,
            "max_reward": max(recent_rewards) if recent_rewards else 0.0,
            "std_reward": float(np.std(recent_rewards)) if recent_rewards else 0.0,
        }

        return cls(
            learning_rate=move_module.optimizer.param_groups[0]["lr"],
            epsilon=move_module.epsilon,
            latest_loss=recent_losses[-1] if recent_losses else None,
            latest_reward=recent_rewards[-1] if recent_rewards else None,
            memory_size=len(move_module.memory),
            memory_capacity=move_module.memory.maxlen if move_module.memory else 0,  # type: ignore
            steps=move_module.steps,
            architecture=architecture,
            training_metrics=training_metrics,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all state values

        Example:
            >>> state.to_dict()
            {
                'learning_rate': 0.001,
                'epsilon': 0.3,
                'latest_loss': 0.5,
                'latest_reward': 1.2,
                'memory_usage': {'current': 1000, 'capacity': 10000},
                'steps': 5000,
                'architecture': {'input_dim': 4, 'hidden_sizes': [64, 64], 'output_dim': 4},
                'metrics': {'avg_loss': 0.4, 'avg_reward': 1.1, ...}
            }
        """
        return {
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "latest_loss": self.latest_loss,
            "latest_reward": self.latest_reward,
            "memory_usage": {
                "current": self.memory_size,
                "capacity": self.memory_capacity,
            },
            "steps": self.steps,
            "architecture": self.architecture,
            "metrics": self.training_metrics,
        }

    def __str__(self) -> str:
        """Human-readable string representation of model state.

        Returns:
            str: Formatted string with key model information
        """
        loss_str = f"{self.latest_loss:.3f}" if self.latest_loss is not None else "None"
        reward_str = f"{self.latest_reward:.3f}" if self.latest_reward is not None else "None"
        return (
            f"ModelState(lr={self.learning_rate:.6f}, "
            f"ε={self.epsilon:.3f}, "
            f"loss={loss_str}, "
            f"reward={reward_str}, "
            f"memory={self.memory_size}/{self.memory_capacity}, "
            f"steps={self.steps})"
        )


class SimulationState(BaseState):
    """State representation for the overall simulation.

    Captures the current state of the entire simulation including time progression,
    population metrics, resource metrics, and performance indicators. All values
    are normalized to [0,1] for consistency with other state representations.

    Attributes:
        normalized_time_progress (float): Current simulation progress
        normalized_population_size (float): Current total population relative to capacity
        normalized_survival_rate (float): Portion of original agents still alive
        normalized_resource_efficiency (float): Resource utilization effectiveness
        normalized_system_performance (float): System agents' performance metric
        DIMENSIONS (ClassVar[int]): Number of dimensions in state vector
        MAX_POPULATION (ClassVar[int]): Maximum expected population for normalization

    Example:
        >>> state = SimulationState(
        ...     normalized_time_progress=0.5,
        ...     normalized_population_size=0.7,
        ...     normalized_survival_rate=0.8,
        ...     normalized_resource_efficiency=0.6,
        ...     normalized_system_performance=0.75
        ... )
    """

    normalized_time_progress: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current simulation step relative to total steps. "
        "0 = start, 1 = completion",
    )

    normalized_population_size: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current population relative to maximum capacity. "
        "0 = empty, 1 = at capacity",
    )

    normalized_survival_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Portion of original agents still alive. "
        "0 = none survived, 1 = all survived",
    )

    normalized_resource_efficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Resource utilization effectiveness. "
        "0 = inefficient, 1 = optimal usage",
    )

    normalized_system_performance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="System agents' performance metric. "
        "0 = poor performance, 1 = optimal performance",
    )

    # Class constants
    DIMENSIONS: ClassVar[int] = 5
    MAX_POPULATION: ClassVar[int] = 1000  # Adjust based on simulation needs

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert simulation state to tensor format.

        Args:
            device (torch.device): Device to place tensor on (CPU/GPU)

        Returns:
            torch.Tensor: 1D tensor of shape (DIMENSIONS,) containing state values
        """
        return torch.FloatTensor(
            [
                self.normalized_time_progress,
                self.normalized_population_size,
                self.normalized_survival_rate,
                self.normalized_resource_efficiency,
                self.normalized_system_performance,
            ]
        ).to(device)

    def to_dict(self) -> Dict[str, float]:
        """Convert state to dictionary with descriptive keys.

        Returns:
            Dict[str, float]: Dictionary containing state values

        Example:
            >>> state.to_dict()
            {
                'time_progress': 0.5,
                'population_size': 0.7,
                'survival_rate': 0.8,
                'resource_efficiency': 0.6,
                'system_performance': 0.75
            }
        """
        return {
            "time_progress": self.normalized_time_progress,
            "population_size": self.normalized_population_size,
            "survival_rate": self.normalized_survival_rate,
            "resource_efficiency": self.normalized_resource_efficiency,
            "system_performance": self.normalized_system_performance,
        }

    @classmethod
    def from_environment(
        cls, environment: "Environment", num_steps: int
    ) -> "SimulationState":
        """Create a SimulationState instance from current environment state.

        Args:
            environment (Environment): Current simulation environment
            num_steps (int): Total number of simulation steps

        Returns:
            SimulationState: Current state of the simulation
        """
        # Calculate normalized time progress
        time_progress = environment.time / num_steps

        # Calculate population metrics
        current_population = len([a for a in environment.agent_objects if a.alive])
        initial_population = environment.get_initial_agent_count()
        max_population = cls.MAX_POPULATION

        # Calculate survival rate (capped at 1.0 to handle reproduction)
        survival_rate = min(
            current_population / initial_population if initial_population > 0 else 0.0,
            1.0,
        )

        # Calculate resource efficiency
        total_resources = sum(resource.amount for resource in environment.resources)
        max_resource_amount = (
            environment.config.resources.max_resource_amount if environment.config else 30
        )
        max_resources = max_resource_amount * len(environment.resources)
        resource_efficiency = (
            total_resources / max_resources if max_resources > 0 else 0.0
        )

        # Since we're not using system agents yet, set performance to 0
        system_performance = 0.0

        # Add clamping to ensure normalized_resource_efficiency doesn't exceed 1.0
        normalized_resource_efficiency = min(resource_efficiency, 1.0)

        return cls(
            normalized_time_progress=time_progress,
            normalized_population_size=min(current_population / max_population, 1.0),
            normalized_survival_rate=survival_rate,
            normalized_resource_efficiency=normalized_resource_efficiency,
            normalized_system_performance=system_performance,
        )

    def get_agent_genealogy(self) -> Dict[str, Any]:
        """Get genealogical information about the current agent population.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - max_generation: Highest generation number reached
            - generation_counts: Count of agents per generation
            - lineage_lengths: Distribution of lineage lengths
            - survival_rates: Survival rates by generation
        """
        return {
            "normalized_max_generation": self.normalized_time_progress,
            "generation_distribution": self.normalized_population_size,
            "lineage_survival": self.normalized_survival_rate,
            "evolutionary_progress": self.normalized_system_performance,
        }


class AgentStateManager:
    """
    Mutable wrapper around immutable AgentState for runtime updates.
    
    Provides a mutable interface for updating agent state during simulation
    while maintaining the benefits of immutable state objects for consistency
    and thread safety. Components can update state through well-defined methods,
    ensuring consistency and making state changes auditable for debugging and analysis.
    """
    
    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        step_number: int = 0,
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
            step_number: Current simulation step
            generation: Generation number
            parent_ids: IDs of parent agents
            genome_id: Unique genome identifier
            birth_time: Simulation step when born
        """
        self._state = AgentState.from_raw_values(
            agent_id=agent_id,
            step_number=step_number,
            position_x=position[0],
            position_y=position[1],
            position_z=0.0,
            resource_level=0.0,
            current_health=1.0,
            is_defending=False,
            total_reward=0.0,
            age=0,
            generation=generation,
            parent_ids=parent_ids or [],
            genome_id=genome_id,
            birth_time=birth_time,
            death_time=None,
            orientation=0.0,
            alive=True,
        )
    
    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self._state.agent_id
    
    @property
    def position(self) -> tuple[float, float]:
        """Get current position."""
        return (self._state.position_x, self._state.position_y)
    
    @property
    def orientation(self) -> float:
        """Get current orientation."""
        return self._state.orientation
    
    @property
    def generation(self) -> int:
        """Get generation number."""
        return self._state.generation
    
    @property
    def parent_ids(self) -> list[str]:
        """Get parent IDs."""
        return self._state.parent_ids.copy()
    
    @property
    def genome_id(self) -> str:
        """Get genome ID."""
        return self._state.genome_id
    
    @property
    def birth_time(self) -> int:
        """Get birth time."""
        return self._state.birth_time
    
    @property
    def death_time(self) -> Optional[int]:
        """Get death time."""
        return self._state.death_time
    
    @property
    def resource_level(self) -> float:
        """Get resource level."""
        return self._state.resource_level
    
    @property
    def health(self) -> float:
        """Get health level."""
        return self._state.current_health
    
    @property
    def is_defending(self) -> bool:
        """Get defense status."""
        return self._state.is_defending
    
    @property
    def total_reward(self) -> float:
        """Get total reward."""
        return self._state.total_reward
    
    @property
    def age(self) -> int:
        """Get age."""
        return self._state.age
    
    @property
    def alive(self) -> bool:
        """Get alive status."""
        return self._state.alive
    
    def update_position(self, position: tuple[float, float]) -> None:
        """Update agent position."""
        self._state = self._state.model_copy(update={
            "position_x": position[0],
            "position_y": position[1],
        })
    
    def update_orientation(self, orientation: float) -> None:
        """Update agent orientation (heading in degrees)."""
        self._state = self._state.model_copy(update={
            "orientation": orientation,
        })
    
    def update_resource_level(self, resource_level: float) -> None:
        """Update resource level."""
        self._state = self._state.model_copy(update={
            "resource_level": resource_level,
        })
    
    def update_health(self, health: float) -> None:
        """Update health level."""
        self._state = self._state.model_copy(update={
            "current_health": health,
        })
    
    def set_defending(self, is_defending: bool) -> None:
        """Set defense status."""
        self._state = self._state.model_copy(update={
            "is_defending": is_defending,
        })
    
    def add_reward(self, reward: float) -> None:
        """Add to total reward."""
        self._state = self._state.model_copy(update={
            "total_reward": self._state.total_reward + reward,
        })
    
    def set_dead(self, death_time: int) -> None:
        """Mark agent as dead."""
        self._state = self._state.model_copy(update={
            "alive": False,
            "death_time": death_time,
        })
    
    def update_step(self, step_number: int) -> None:
        """Update current step and age."""
        self._state = self._state.model_copy(update={
            "step_number": step_number,
            "age": step_number - self._state.birth_time,
        })
    
    def snapshot(self, current_time: int) -> AgentState:
        """
        Create a snapshot of current state.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            AgentState: Complete state snapshot
        """
        return self._state.model_copy(update={
            "step_number": current_time,
            "age": current_time - self._state.birth_time,
        })
    
    def get_state(self) -> AgentState:
        """Get current immutable state."""
        return self._state
