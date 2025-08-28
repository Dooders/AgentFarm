"""Pydantic-based configuration system for action modules.

This module provides a unified configuration system using Pydantic for validation
and type safety. It reduces duplication by providing a base DQN configuration
that can be extended by specific action modules with only their unique parameters.
"""

from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field, field_validator


class BaseDQNConfig(BaseModel):
    """Base configuration for all DQN-based action modules.

    This class provides sensible defaults for all DQN parameters and can be
    extended by specific action modules to add their unique parameters.

    Attributes:
        target_update_freq: Frequency of hard target network updates (unused with soft updates)
        memory_size: Maximum size of experience replay buffer
        learning_rate: Learning rate for the Adam optimizer
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate for epsilon-greedy strategy
        epsilon_min: Minimum exploration rate
        epsilon_decay: Decay rate for exploration (multiplied each step)
        dqn_hidden_size: Number of neurons in hidden layers
        batch_size: Number of experiences to sample for training
        tau: Soft update parameter for target network (0 < tau < 1)
        seed: Random seed for reproducibility
    """

    # Core DQN parameters
    target_update_freq: int = Field(
        default=100, description="Frequency of hard target network updates"
    )
    memory_size: int = Field(
        default=10000, description="Maximum size of experience replay buffer"
    )
    learning_rate: float = Field(
        default=0.001, description="Learning rate for the Adam optimizer"
    )
    gamma: float = Field(default=0.99, description="Discount factor for future rewards")
    epsilon_start: float = Field(
        default=1.0, description="Initial exploration rate for epsilon-greedy strategy"
    )
    epsilon_min: float = Field(default=0.01, description="Minimum exploration rate")
    epsilon_decay: float = Field(
        default=0.995, description="Decay rate for exploration (multiplied each step)"
    )
    dqn_hidden_size: int = Field(
        default=64, description="Number of neurons in hidden layers"
    )
    batch_size: int = Field(
        default=32, description="Number of experiences to sample for training"
    )
    tau: float = Field(
        default=0.005,
        description="Soft update parameter for target network (0 < tau < 1)",
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )

    @field_validator("tau")
    def validate_tau(cls, v):
        """Validate tau is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("tau must be between 0 and 1")
        return v

    @field_validator("gamma")
    def validate_gamma(cls, v):
        """Validate gamma is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("gamma must be between 0 and 1")
        return v

    @field_validator("epsilon_start", "epsilon_min")
    def validate_epsilon(cls, v):
        """Validate epsilon values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("epsilon values must be between 0 and 1")
        return v

    @field_validator("epsilon_decay")
    def validate_epsilon_decay(cls, v):
        """Validate epsilon decay is between 0 and 1."""
        if not 0 < v <= 1:
            raise ValueError("epsilon_decay must be between 0 and 1")
        return v


class AttackConfig(BaseDQNConfig):
    """Configuration for attack action module.

    Extends BaseDQNConfig with attack-specific parameters for combat behavior.
    """

    # Attack-specific parameters
    base_cost: float = Field(
        default=-0.2, description="Base resource cost for attempting an attack"
    )
    success_reward: float = Field(
        default=1.0, description="Reward multiplier for successful attacks"
    )
    failure_penalty: float = Field(
        default=-0.3, description="Penalty for failed attack attempts"
    )
    defense_threshold: float = Field(
        default=0.3, description="Health ratio threshold for defensive behavior"
    )
    defense_boost: float = Field(
        default=2.0, description="Multiplier for defense action when health is low"
    )
    range: float = Field(default=20.0, description="Range for attack actions")
    base_damage: float = Field(default=10.0, description="Base damage for attacks")
    kill_reward: float = Field(
        default=5.0, description="Reward for killing an opponent"
    )


class GatherConfig(BaseDQNConfig):
    """Configuration for gather action module.

    Extends BaseDQNConfig with gathering-specific parameters for resource collection.
    """

    # Gathering-specific parameters
    success_reward: float = Field(
        default=1.0, description="Base reward for successful resource gathering"
    )
    failure_penalty: float = Field(
        default=-0.1, description="Base penalty for failed gathering attempts"
    )
    efficiency_multiplier: float = Field(
        default=0.5, description="Multiplier for efficiency bonuses"
    )
    cost_multiplier: float = Field(
        default=0.3, description="Multiplier for movement cost penalties"
    )
    min_resource_threshold: float = Field(
        default=0.1, description="Minimum resource amount worth gathering"
    )
    max_wait_steps: int = Field(
        default=5, description="Maximum steps to wait for resource regeneration"
    )
    range: int = Field(default=30, description="Range for gathering actions")
    max_amount: int = Field(
        default=3, description="Maximum amount that can be gathered per action"
    )


class MoveConfig(BaseDQNConfig):
    """Configuration for move action module.

    Extends BaseDQNConfig with movement-specific parameters for navigation.
    """

    # Movement-specific parameters
    base_cost: float = Field(
        default=-0.1, description="Base cost for any movement action"
    )
    resource_approach_reward: float = Field(
        default=0.3, description="Reward for moving closer to resources"
    )
    resource_retreat_penalty: float = Field(
        default=-0.2, description="Penalty for moving away from resources"
    )
    max_movement: int = Field(
        default=8, description="Maximum movement distance per action"
    )


class ReproduceConfig(BaseDQNConfig):
    """Configuration for reproduce action module.

    Extends BaseDQNConfig with reproduction-specific parameters for population dynamics.
    """

    # Reproduction-specific parameters
    success_reward: float = Field(
        default=1.0, description="Reward given for successful reproduction"
    )
    failure_penalty: float = Field(
        default=-0.2, description="Penalty for failed reproduction attempts"
    )
    offspring_survival_bonus: float = Field(
        default=0.5, description="Bonus for maintaining resources after reproduction"
    )
    population_balance_bonus: float = Field(
        default=0.3, description="Bonus for maintaining optimal population levels"
    )
    min_health_ratio: float = Field(
        default=0.5, description="Minimum health ratio required for reproduction"
    )
    min_resource_ratio: float = Field(
        default=0.6, description="Minimum resource ratio required for reproduction"
    )
    ideal_density_radius: float = Field(
        default=50.0, description="Radius for calculating local population density"
    )
    max_local_density: float = Field(
        default=0.7, description="Maximum allowed local population density"
    )
    min_space_required: float = Field(
        default=20.0,
        description="Minimum space required between agents for reproduction",
    )
    offspring_cost: int = Field(default=3, description="Cost of creating offspring")
    min_reproduction_resources: int = Field(
        default=8, description="Minimum resources required for reproduction"
    )


class SelectConfig(BaseDQNConfig):
    """Configuration for select action module.

    Extends BaseDQNConfig with selection-specific parameters for action prioritization.
    """

    # Base action weights
    move_weight: float = Field(
        default=0.3, description="Base probability weight for move actions"
    )
    gather_weight: float = Field(
        default=0.3, description="Base probability weight for gather actions"
    )
    share_weight: float = Field(
        default=0.15, description="Base probability weight for share actions"
    )
    attack_weight: float = Field(
        default=0.1, description="Base probability weight for attack actions"
    )
    reproduce_weight: float = Field(
        default=0.15, description="Base probability weight for reproduce actions"
    )

    # Algorithm selection (optional traditional ML path)
    algorithm_type: str = Field(
        default="dqn",
        description=(
            "Action algorithm type: one of ['dqn','mlp','svm','random_forest','gradient_boost','naive_bayes','knn']"
        ),
    )
    algorithm_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters for algorithm constructor"
    )
    feature_engineering: List[str] = Field(
        default_factory=list, description="Optional feature engineering flags"
    )
    ensemble_size: int = Field(
        default=1, description="Optional ensemble size for algorithms that support it"
    )
    use_exploration_bonus: bool = Field(
        default=True, description="If true, add small exploration bonus to probabilities"
    )

    # State-based multipliers
    move_mult_no_resources: float = Field(
        default=1.5, description="Multiplier for move when no resources nearby"
    )
    gather_mult_low_resources: float = Field(
        default=1.5, description="Multiplier for gather when resources are low"
    )
    share_mult_wealthy: float = Field(
        default=1.3, description="Multiplier for share when agent is wealthy"
    )
    share_mult_poor: float = Field(
        default=0.5, description="Multiplier for share when agent is poor"
    )
    attack_mult_desperate: float = Field(
        default=1.4, description="Multiplier for attack when desperate (starving)"
    )
    attack_mult_stable: float = Field(
        default=0.6, description="Multiplier for attack when stable"
    )
    reproduce_mult_wealthy: float = Field(
        default=1.4, description="Multiplier for reproduce when wealthy"
    )
    reproduce_mult_poor: float = Field(
        default=0.3, description="Multiplier for reproduce when poor"
    )

    # Thresholds
    attack_starvation_threshold: float = Field(
        default=0.5, description="Threshold for desperate attack behavior"
    )
    attack_defense_threshold: float = Field(
        default=0.3, description="Threshold for defensive attack behavior"
    )
    reproduce_resource_threshold: float = Field(
        default=0.7, description="Threshold for reproduction resource requirements"
    )

    @field_validator(
        "move_weight",
        "gather_weight",
        "share_weight",
        "attack_weight",
        "reproduce_weight",
    )
    def validate_weights(cls, v):
        """Validate weights are non-negative."""
        if v < 0:
            raise ValueError("weights must be non-negative")
        return v

    @field_validator("algorithm_type")
    def validate_algorithm_type(cls, v):
        valid = [
            "dqn",
            "mlp",
            "svm",
            "random_forest",
            "gradient_boost",
            "naive_bayes",
            "knn",
        ]
        if v not in valid:
            raise ValueError(f"Algorithm must be one of: {valid}")
        return v


class ShareConfig(BaseDQNConfig):
    """Configuration for share action module.

    Extends BaseDQNConfig with sharing-specific parameters for cooperative behavior.
    """

    # Sharing-specific parameters
    range: float = Field(
        default=30.0, description="Maximum distance for sharing interactions"
    )
    min_amount: int = Field(
        default=1, description="Minimum resources required to initiate sharing"
    )
    success_reward: float = Field(
        default=0.3, description="Base reward for successful sharing actions"
    )
    failure_penalty: float = Field(
        default=-0.1, description="Penalty for failed sharing attempts"
    )
    base_cost: float = Field(default=-0.05, description="Base cost for sharing actions")
    altruism_bonus: float = Field(
        default=0.2, description="Extra reward for sharing with agents in need"
    )
    cooperation_memory: int = Field(
        default=100,
        description="Number of recent interactions to remember for cooperation scoring",
    )
    max_resources: int = Field(
        default=30, description="Maximum possible resources for normalization purposes"
    )
    max_amount: int = Field(
        default=5, description="Maximum amount that can be shared per action"
    )
    threshold: float = Field(default=0.3, description="Threshold for sharing decisions")
    cooperation_bonus: float = Field(
        default=0.2, description="Bonus for cooperative behavior"
    )
    altruism_factor: float = Field(
        default=1.2, description="Factor for altruistic sharing"
    )
    cooperation_score_threshold: float = Field(
        default=0.5, description="Threshold for considering an agent cooperative"
    )


# Default configuration instances
DEFAULT_ATTACK_CONFIG = AttackConfig()
DEFAULT_GATHER_CONFIG = GatherConfig()
DEFAULT_MOVE_CONFIG = MoveConfig()
DEFAULT_REPRODUCE_CONFIG = ReproduceConfig()
DEFAULT_SELECT_CONFIG = SelectConfig()
DEFAULT_SHARE_CONFIG = ShareConfig()


def create_config_from_dict(
    config_dict: Dict[str, Any], config_class: type
) -> BaseDQNConfig:
    """Create a configuration instance from a dictionary.

    This function allows creating configuration instances from dictionaries,
    which is useful for loading configurations from YAML files or other sources.

    Args:
        config_dict: Dictionary containing configuration parameters
        config_class: The configuration class to instantiate

    Returns:
        Configuration instance of the specified class
    """
    return config_class(**config_dict)


def merge_configs(
    base_config: BaseDQNConfig, override_config: Dict[str, Any]
) -> BaseDQNConfig:
    """Merge a base configuration with override values.

    This function allows updating specific parameters in a configuration
    while keeping the rest unchanged.

    Args:
        base_config: Base configuration to merge from
        override_config: Dictionary of parameters to override

    Returns:
        New configuration instance with merged values
    """
    base_dict = base_config.dict()
    base_dict.update(override_config)
    return type(base_config)(**base_dict)
