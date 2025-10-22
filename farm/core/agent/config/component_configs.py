"""
Configuration dataclasses for individual agent components.

Each component has its own configuration object that defines its behavior,
parameters, and thresholds. These are frozen dataclasses to ensure immutability
and make configurations safe to share across agents.
"""

from dataclasses import dataclass, field
from typing import Optional

from farm.core.decision.config import DecisionConfig


@dataclass(frozen=True)
class MovementConfig:
    """Configuration for agent movement and position handling."""
    
    max_movement: float = 8.0
    """Maximum distance agent can move in a single step."""
    
    perception_radius: int = 5
    """Radius for spatial awareness queries."""


@dataclass(frozen=True)
class ResourceConfig:
    """Configuration for resource management and starvation mechanics."""
    
    base_consumption_rate: float = 1.0
    """Resources consumed per turn for basic maintenance."""
    
    starvation_threshold: int = 100
    """Number of steps agent can survive with zero resources."""
    
    offspring_initial_resources: float = 10.0
    """Starting resources for newly created offspring."""
    
    offspring_cost: float = 5.0
    """Resources consumed by parent when reproducing."""


@dataclass(frozen=True)
class CombatConfig:
    """Configuration for combat mechanics and health."""
    
    starting_health: float = 100.0
    """Maximum health points for agent."""
    
    base_attack_strength: float = 10.0
    """Base damage dealt per attack."""
    
    base_defense_strength: float = 5.0
    """Defense bonus when in defensive stance."""
    
    defense_damage_reduction: float = 0.5
    """Damage reduction multiplier when defending (0.5 = 50% reduction)."""
    
    defense_timer_duration: int = 3
    """Number of turns defensive stance lasts."""


@dataclass(frozen=True)
class PerceptionConfig:
    """Configuration for agent perception and observation."""
    
    perception_radius: int = 5
    """Radius for perception grid."""
    
    position_discretization_method: str = "floor"
    """How to discretize positions: 'floor', 'ceil', or 'round'."""


@dataclass(frozen=True)
class ReproductionConfig:
    """Configuration for reproduction mechanics."""
    
    offspring_initial_resources: float = 10.0
    """Starting resources for newly created offspring."""
    
    offspring_cost: float = 5.0
    """Resources consumed by parent when reproducing."""


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for reward calculation and tracking."""
    
    # Reward scales
    resource_reward_scale: float = 1.0
    """Multiplier for resource-based rewards."""
    
    health_reward_scale: float = 0.5
    """Multiplier for health-based rewards."""
    
    # Survival rewards
    survival_bonus: float = 0.1
    """Reward for staying alive each step."""
    
    death_penalty: float = -10.0
    """Penalty for dying."""
    
    age_bonus: float = 0.01
    """Small bonus for each step survived."""
    
    # Action-specific rewards (can be extended)
    combat_success_bonus: float = 2.0
    """Bonus for successful combat actions."""
    
    reproduction_bonus: float = 5.0
    """Bonus for successful reproduction."""
    
    cooperation_bonus: float = 1.0
    """Bonus for cooperative actions."""
    
    # History tracking
    max_history_length: int = 1000
    """Maximum number of rewards to keep in history."""
    
    recent_window: int = 100
    """Window size for recent reward calculations."""
    
    # Reward calculation strategy
    use_delta_rewards: bool = True
    """Whether to use delta-based rewards (better for RL)."""
    
    normalize_rewards: bool = False
    """Whether to normalize rewards to [-1, 1] range."""


@dataclass(frozen=False)
class AgentComponentConfig:
    """
    Complete agent configuration combining all component configs.
    
    This is the master configuration object that defines all aspects
    of an agent's behavior. It can be passed to the factory to create
    agents with specific configurations.
    """
    
    movement: MovementConfig = field(default_factory=MovementConfig)
    """Movement component configuration."""
    
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    """Resource management configuration."""
    
    combat: CombatConfig = field(default_factory=CombatConfig)
    """Combat mechanics configuration."""
    
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    """Perception configuration."""
    
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    """Reproduction configuration."""
    
    reward: RewardConfig = field(default_factory=RewardConfig)
    """Reward calculation configuration."""
    
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    """Decision-making configuration."""
    
    @classmethod
    def default(cls) -> "AgentComponentConfig":
        """Create a default configuration with standard values."""
        return cls()
    
    @classmethod
    def aggressive(cls) -> "AgentComponentConfig":
        """Create an aggressive agent configuration with high combat stats."""
        return cls(
            combat=CombatConfig(
                starting_health=150.0,
                base_attack_strength=20.0,
                base_defense_strength=8.0,
            ),
            resource=ResourceConfig(
                base_consumption_rate=2.0,
                offspring_cost=10.0,
            ),
        )
    
    @classmethod
    def defensive(cls) -> "AgentComponentConfig":
        """Create a defensive agent configuration with high health."""
        return cls(
            combat=CombatConfig(
                starting_health=200.0,
                base_attack_strength=5.0,
                base_defense_strength=15.0,
                defense_damage_reduction=0.7,
            ),
            resource=ResourceConfig(
                base_consumption_rate=1.5,
            ),
        )
    
    @classmethod
    def efficient(cls) -> "AgentComponentConfig":
        """Create an efficient agent configuration with low resource consumption."""
        return cls(
            resource=ResourceConfig(
                base_consumption_rate=0.5,
                offspring_cost=3.0,
            ),
            combat=CombatConfig(
                starting_health=80.0,
            ),
        )
    
    @classmethod
    def from_simulation_config(cls, sim_config) -> "AgentComponentConfig":
        """
        Create component configuration from main simulation configuration.
        
        Args:
            sim_config: Main simulation configuration object
            
        Returns:
            AgentComponentConfig with values from simulation config
        """
        # Extract values from simulation config with fallbacks
        agent_behavior = getattr(sim_config, 'agent_behavior', None)
        
        if agent_behavior:
            # Use values from agent_behavior config
            base_consumption_rate = getattr(agent_behavior, 'base_consumption_rate', 1.0)
            starvation_threshold = getattr(agent_behavior, 'starvation_threshold', 10)
            offspring_cost = getattr(agent_behavior, 'offspring_cost', 5.0)
            offspring_initial_resources = getattr(agent_behavior, 'offspring_initial_resources', 10.0)
            
            # Combat config
            starting_health = getattr(agent_behavior, 'starting_health', 100.0)
            base_attack_strength = getattr(agent_behavior, 'base_attack_strength', 10.0)
            base_defense_strength = getattr(agent_behavior, 'base_defense_strength', 5.0)
        else:
            # Fallback to default values
            base_consumption_rate = 1.0
            starvation_threshold = 10
            offspring_cost = 5.0
            offspring_initial_resources = 10.0
            starting_health = 100.0
            base_attack_strength = 10.0
            base_defense_strength = 5.0
        
        return cls(
            resource=ResourceConfig(
                base_consumption_rate=base_consumption_rate,
                starvation_threshold=starvation_threshold,
                offspring_cost=offspring_cost,
                offspring_initial_resources=offspring_initial_resources,
            ),
            combat=CombatConfig(
                starting_health=starting_health,
                base_attack_strength=base_attack_strength,
                base_defense_strength=base_defense_strength,
            ),
        )