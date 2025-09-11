"""Pydantic-based configuration system for action modules.

This module provides a unified configuration system using Pydantic for validation
and type safety. It reduces duplication by providing a base DQN configuration
that can be extended by specific action modules with only their unique parameters.
"""

from typing import Any, Dict, List, Optional, TypeVar

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
    @classmethod
    def validate_tau(cls, v):
        """Validate tau is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("tau must be between 0 and 1")
        return v

    @field_validator("gamma")
    @classmethod
    def validate_gamma(cls, v):
        """Validate gamma is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("gamma must be between 0 and 1")
        return v

    @field_validator("epsilon_start", "epsilon_min")
    @classmethod
    def validate_epsilon(cls, v):
        """Validate epsilon values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("epsilon values must be between 0 and 1")
        return v

    @field_validator("epsilon_decay")
    @classmethod
    def validate_epsilon_decay(cls, v):
        """Validate epsilon decay is between 0 and 1."""
        if not 0 < v <= 1:
            raise ValueError("epsilon_decay must be between 0 and 1")
        return v


class DecisionConfig(BaseDQNConfig):
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

    # Algorithm selection (traditional ML path)
    algorithm_type: str = Field(
        default="dqn",
        description=(
            "Action algorithm type: one of ['dqn','mlp','svm','random_forest','gradient_boost','naive_bayes','knn','ppo','sac','a2c','ddpg','ddqn','fallback']"
        ),
    )
    algorithm_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for algorithm constructor",
    )

    # RL algorithm configuration
    rl_state_dim: int = Field(
        default=8, description="State dimension for RL algorithms"
    )
    rl_buffer_size: int = Field(
        default=10000, description="Experience replay buffer size for RL algorithms"
    )
    rl_batch_size: int = Field(
        default=32, description="Batch size for RL algorithm training"
    )
    rl_train_freq: int = Field(
        default=4, description="How often to train RL algorithms (every N steps)"
    )
    feature_engineering: List[str] = Field(
        default_factory=list, description="Optional feature engineering flags"
    )
    ensemble_size: int = Field(
        default=1, description="Optional ensemble size for algorithms that support it"
    )
    use_exploration_bonus: bool = Field(
        default=True,
        description="If true, add small exploration bonus to probabilities",
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
    @classmethod
    def validate_weights(cls, v):
        """Validate weights are non-negative."""
        if v < 0:
            raise ValueError("weights must be non-negative")
        return v

    @field_validator("algorithm_type")
    @classmethod
    def validate_algorithm_type(cls, v):
        valid = [
            "dqn",
            "mlp",
            "svm",
            "random_forest",
            "gradient_boost",
            "naive_bayes",
            "knn",
            # RL algorithms
            "ppo",
            "sac",
            "a2c",
            "ddpg",
            # Additional supported algorithms
            "ddqn",  # Double DQN (alias for dqn with DDQN features)
            "fallback",  # Fallback random algorithm
        ]
        # Instead of raising an error, log a warning and return 'fallback'
        # This allows the DecisionModule to handle invalid algorithms gracefully
        if v not in valid:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Invalid algorithm type '{v}'. Valid options: {valid}. "
                "Falling back to 'fallback' algorithm."
            )
            return "fallback"
        return v


# Type variable for configuration classes
ConfigT = TypeVar("ConfigT", bound=BaseDQNConfig)


# Default configuration instances
DEFAULT_DECISION_CONFIG = DecisionConfig()
DEFAULT_DQN_CONFIG = BaseDQNConfig()


def create_config_from_dict(
    config_dict: Dict[str, Any], config_class: type[ConfigT]
) -> ConfigT:
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
    base_dict = base_config.model_dump()
    base_dict.update(override_config)
    return type(base_config)(**base_dict)
