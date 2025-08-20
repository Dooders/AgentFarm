import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class VisualizationConfig:
    canvas_size: Tuple[int, int] = (400, 400)
    padding: int = 20
    background_color: str = "black"
    max_animation_frames: int = 5
    animation_min_delay: int = 50
    max_resource_amount: int = 30
    resource_colors: Dict[str, int] = field(
        default_factory=lambda: {"glow_red": 150, "glow_green": 255, "glow_blue": 50}
    )
    resource_size: int = 2
    agent_radius_scale: int = 2
    birth_radius_scale: int = 4
    death_mark_scale: float = 1.5
    agent_colors: Dict[str, str] = field(
        default_factory=lambda: {"SystemAgent": "blue", "IndependentAgent": "red"}
    )
    min_font_size: int = 10
    font_scale_factor: int = 40
    font_family: str = "arial"
    death_mark_color: List[int] = field(default_factory=lambda: [255, 0, 0])
    birth_mark_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    metric_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "total_agents": "#4a90e2",
            "system_agents": "#50c878",
            "independent_agents": "#e74c3c",
            "total_resources": "#f39c12",
            "average_agent_resources": "#9b59b6",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert visualization config to a JSON-serializable dictionary."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualizationConfig":
        """Create visualization config from a dictionary."""
        return cls(**data)


@dataclass
class DQNProfile:
    """Reusable DQN hyperparameter profile.

    Defines a set of DQN learning hyperparameters that can be referenced and
    reused across different actions. Action-specific overrides can be applied
    on top of a base profile.
    """

    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    hidden_size: int = 64
    tau: float = 0.005
    target_update_freq: int = 100
    seed: Optional[int] = None


@dataclass 
class ActionConfig:
    """Per-action customizations that reference a profile and apply overrides."""

    profile: str = "default"
    rewards: Dict[str, float] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RedisMemoryConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = True
    environment: str = "default"


@dataclass
class SimulationConfig:
    # Environment settings
    width: int = 100
    height: int = 100

    # Agent settings
    system_agents: int = 10
    independent_agents: int = 10
    control_agents: int = 10
    initial_resource_level: int = 0
    max_population: int = 3000
    starvation_threshold: int = 0
    max_starvation_time: int = 15
    offspring_cost: int = 3
    min_reproduction_resources: int = 8
    offspring_initial_resources: int = 5
    perception_radius: int = 2
    base_attack_strength: int = 2
    base_defense_strength: int = 2
    # Agent type ratios
    agent_type_ratios: Dict[str, float] = field(
        default_factory=lambda: {
            "SystemAgent": 0.33,
            "IndependentAgent": 0.33,
            "ControlAgent": 0.34,
        }
    )
    seed: Optional[int] = 1234567890

    # Resource settings
    initial_resources: int = 20
    resource_regen_rate: float = 0.1
    resource_regen_amount: int = 2
    max_resource_amount: int = 30

    # Agent behavior settings
    base_consumption_rate: float = 0.15
    max_movement: int = 8
    gathering_range: int = 30
    max_gather_amount: int = 3
    territory_range: int = 30

    # Learning parameters
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 2000
    batch_size: int = 32
    training_frequency: int = 4
    dqn_hidden_size: int = 24
    tau: float = 0.005

    # Combat Parameters
    starting_health: float = 100.0
    attack_range: float = 20.0
    attack_base_damage: float = 10.0
    attack_kill_reward: float = 5.0

    # Agent-specific parameters
    agent_parameters: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "SystemAgent": {
                "gather_efficiency_multiplier": 0.4,
                "gather_cost_multiplier": 0.4,
                "min_resource_threshold": 0.2,
                "share_weight": 0.3,
                "attack_weight": 0.05,
            },
            "IndependentAgent": {
                "gather_efficiency_multiplier": 0.7,
                "gather_cost_multiplier": 0.2,
                "min_resource_threshold": 0.05,
                "share_weight": 0.05,
                "attack_weight": 0.25,
            },
            "ControlAgent": {
                "gather_efficiency_multiplier": 0.55,
                "gather_cost_multiplier": 0.3,
                "min_resource_threshold": 0.125,
                "share_weight": 0.15,
                "attack_weight": 0.15,
            },
        }
    )

    # Visualization settings (separate config)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Action probability adjustment parameters
    social_range: int = 30  # Range for social interactions (share/attack)

    # Movement multipliers
    move_mult_no_resources: float = 1.5  # Multiplier when no resources nearby

    # Gathering multipliers
    gather_mult_low_resources: float = 1.5  # Multiplier when resources needed

    # Sharing multipliers
    share_mult_wealthy: float = 1.3  # Multiplier when agent has excess resources
    share_mult_poor: float = 0.5  # Multiplier when agent needs resources

    # Attack multipliers
    attack_starvation_threshold: float = (
        0.5  # Starvation risk threshold for desperate behavior
    )
    attack_mult_desperate: float = 1.4  # Multiplier when desperate for resources
    attack_mult_stable: float = 0.6  # Multiplier when resource stable

    # Add to the main configuration section, before visualization settings
    max_wait_steps: int = 10  # Maximum steps to wait between gathering attempts

    # Database configuration
    use_in_memory_db: bool = False  # Whether to use in-memory database
    persist_db_on_completion: bool = True  # Whether to persist in-memory DB to disk after simulation
    in_memory_db_memory_limit_mb: Optional[int] = None  # Memory limit for in-memory DB (None = no limit)
    in_memory_tables_to_persist: Optional[List[str]] = None  # Tables to persist (None = all tables)
    
    # Database pragma settings
    db_pragma_profile: str = "balanced"  # Options: "balanced", "performance", "safety", "memory"
    db_cache_size_mb: int = 200  # Cache size in MB
    db_synchronous_mode: str = "NORMAL"  # Options: "OFF", "NORMAL", "FULL"
    db_journal_mode: str = "WAL"  # Options: "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"
    db_custom_pragmas: Dict[str, str] = field(default_factory=dict)  # Custom pragma overrides

    # Redis configuration
    redis: RedisMemoryConfig = field(default_factory=RedisMemoryConfig)

    # Unified config (profiles + action configs) - optional and used during load
    profiles: Dict[str, DQNProfile] = field(default_factory=dict)
    action_configs: Dict[str, ActionConfig] = field(default_factory=dict)

    # Gathering Module Parameters
    gather_target_update_freq: int = 100
    gather_memory_size: int = 10000
    gather_learning_rate: float = 0.001
    gather_gamma: float = 0.99
    gather_epsilon_start: float = 1.0
    gather_epsilon_min: float = 0.01
    gather_epsilon_decay: float = 0.995
    gather_dqn_hidden_size: int = 64
    gather_batch_size: int = 32
    gather_tau: float = 0.005
    gather_success_reward: float = 0.5
    gather_failure_penalty: float = -0.1
    gather_base_cost: float = -0.05
    gather_distance_penalty_factor: float = 0.1
    gather_resource_threshold: float = 0.2
    gather_competition_penalty: float = -0.2
    gather_efficiency_bonus: float = 0.3

    # Sharing Module Parameters
    share_range: float = 30.0
    share_target_update_freq: int = 100
    share_memory_size: int = 10000
    share_learning_rate: float = 0.001
    share_gamma: float = 0.99
    share_epsilon_start: float = 1.0
    share_epsilon_min: float = 0.01
    share_epsilon_decay: float = 0.995
    share_dqn_hidden_size: int = 64
    share_batch_size: int = 32
    share_tau: float = 0.005
    share_success_reward: float = 0.5
    share_failure_penalty: float = -0.1
    share_base_cost: float = -0.05
    min_share_amount: int = 1
    max_share_amount: int = 5
    share_threshold: float = 0.3
    share_cooperation_bonus: float = 0.2
    share_altruism_factor: float = 1.2
    cooperation_memory: int = 100  # Number of past interactions to remember
    cooperation_score_threshold: float = (
        0.5  # Threshold for considering an agent cooperative
    )

    # Movement Module Parameters
    move_target_update_freq: int = 100
    move_memory_size: int = 10000
    move_learning_rate: float = 0.001
    move_gamma: float = 0.99
    move_epsilon_start: float = 1.0
    move_epsilon_min: float = 0.01
    move_epsilon_decay: float = 0.995
    move_dqn_hidden_size: int = 64
    move_batch_size: int = 32
    move_reward_history_size: int = 100
    move_epsilon_adapt_threshold: float = 0.1
    move_epsilon_adapt_factor: float = 1.5
    move_min_reward_samples: int = 10
    move_tau: float = 0.005
    move_base_cost: float = -0.1
    move_resource_approach_reward: float = 0.3
    move_resource_retreat_penalty: float = -0.2

    # Attack Module Parameters
    attack_target_update_freq: int = 100
    attack_memory_size: int = 10000
    attack_learning_rate: float = 0.001
    attack_gamma: float = 0.99
    attack_epsilon_start: float = 1.0
    attack_epsilon_min: float = 0.01
    attack_epsilon_decay: float = 0.995
    attack_dqn_hidden_size: int = 64
    attack_batch_size: int = 32
    attack_tau: float = 0.005
    attack_base_cost: float = -0.2
    attack_success_reward: float = 1.0
    attack_failure_penalty: float = -0.3
    attack_defense_threshold: float = 0.3
    attack_defense_boost: float = 2.0
    attack_kill_reward: float = 5.0
    attack_range: float = 20.0
    attack_base_damage: float = 10.0

    simulation_steps: int = 100  # Default value

    curriculum_phases: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"steps": 100, "enabled_actions": ["move", "gather"]},
        {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
        {"steps": -1, "enabled_actions": ["move", "gather", "share", "attack", "reproduce"]}
    ])  # -1 for remaining steps

    @classmethod
    def from_yaml(cls, file_path: str) -> "SimulationConfig":
        """Load configuration from a YAML file."""
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle visualization config separately
        vis_config = config_dict.pop("visualization", {})

        # Detect and process unified profiles/actions format if present
        profiles_raw = config_dict.pop("dqn_profiles", None)
        actions_raw = config_dict.pop("actions", None)

        flattened: Dict[str, Any] = {}

        if profiles_raw is not None and isinstance(profiles_raw, dict):
            # Normalize nested profiles into dataclasses
            profiles: Dict[str, DQNProfile] = {}
            for name, data in profiles_raw.items():
                # YAML merges should already be resolved by safe_load
                profiles[name] = DQNProfile(**data)

            # Set global/unprefixed DQN params from the default profile if not explicitly provided
            default_profile: Optional[DQNProfile] = profiles.get("default") or next(iter(profiles.values()), None)
            if default_profile is not None:
                if "learning_rate" not in config_dict:
                    config_dict["learning_rate"] = default_profile.learning_rate
                if "gamma" not in config_dict:
                    config_dict["gamma"] = default_profile.gamma
                if "epsilon_start" not in config_dict:
                    config_dict["epsilon_start"] = default_profile.epsilon_start
                if "epsilon_min" not in config_dict:
                    config_dict["epsilon_min"] = default_profile.epsilon_min
                if "epsilon_decay" not in config_dict:
                    config_dict["epsilon_decay"] = default_profile.epsilon_decay
                if "memory_size" not in config_dict:
                    config_dict["memory_size"] = default_profile.memory_size
                if "batch_size" not in config_dict:
                    config_dict["batch_size"] = default_profile.batch_size
                if "dqn_hidden_size" not in config_dict:
                    config_dict["dqn_hidden_size"] = default_profile.hidden_size
                if "tau" not in config_dict:
                    config_dict["tau"] = default_profile.tau
                if "target_update_freq" not in config_dict:
                    config_dict["target_update_freq"] = default_profile.target_update_freq

            # If actions are provided, map them into flattened legacy fields
            if actions_raw and isinstance(actions_raw, dict):
                action_cfgs: Dict[str, ActionConfig] = {}

                def apply_profile_to_action(action_name: str, action_data: Dict[str, Any]) -> None:
                    action = ActionConfig(**action_data)
                    action_cfgs[action_name] = action

                    # Get base profile
                    profile_name = action.profile
                    base_profile = profiles.get(profile_name, DQNProfile())
                    # Start from base profile values
                    effective = {
                        "learning_rate": base_profile.learning_rate,
                        "gamma": base_profile.gamma,
                        "epsilon_start": base_profile.epsilon_start,
                        "epsilon_min": base_profile.epsilon_min,
                        "epsilon_decay": base_profile.epsilon_decay,
                        "memory_size": base_profile.memory_size,
                        "batch_size": base_profile.batch_size,
                        "hidden_size": base_profile.hidden_size,
                        "tau": base_profile.tau,
                        "target_update_freq": base_profile.target_update_freq,
                    }
                    # Apply overrides on top
                    for key, value in action.overrides.items():
                        effective[key] = value

                    # Map effective DQN params to legacy per-action fields
                    prefix = action_name.lower()
                    # Only map known actions to avoid collisions
                    if prefix in {"move", "attack", "gather", "share"}:
                        flattened[f"{prefix}_learning_rate"] = effective["learning_rate"]
                        flattened[f"{prefix}_gamma"] = effective["gamma"]
                        flattened[f"{prefix}_epsilon_start"] = effective["epsilon_start"]
                        flattened[f"{prefix}_epsilon_min"] = effective["epsilon_min"]
                        flattened[f"{prefix}_epsilon_decay"] = effective["epsilon_decay"]
                        flattened[f"{prefix}_memory_size"] = effective["memory_size"]
                        flattened[f"{prefix}_batch_size"] = effective["batch_size"]
                        flattened[f"{prefix}_dqn_hidden_size"] = effective["hidden_size"]
                        flattened[f"{prefix}_tau"] = effective["tau"]
                        flattened[f"{prefix}_target_update_freq"] = effective["target_update_freq"]

                        # Also set global hidden size if not already defined
                        if "dqn_hidden_size" not in config_dict:
                            config_dict["dqn_hidden_size"] = effective["hidden_size"]

                        # Map rewards/costs to known legacy names when present
                        costs = action.costs or {}
                        rewards = action.rewards or {}

                        if prefix == "move":
                            if "base" in costs:
                                flattened["move_base_cost"] = costs["base"]
                            if "approach_resource" in rewards:
                                flattened["move_resource_approach_reward"] = rewards["approach_resource"]
                            if "retreat_penalty" in rewards:
                                flattened["move_resource_retreat_penalty"] = rewards["retreat_penalty"]
                        elif prefix == "attack":
                            if "base" in costs:
                                flattened["attack_base_cost"] = costs["base"]
                            if "success" in rewards:
                                flattened["attack_success_reward"] = rewards["success"]
                            if "failure_penalty" in rewards:
                                flattened["attack_failure_penalty"] = rewards["failure_penalty"]
                            if "kill" in rewards:
                                flattened["attack_kill_reward"] = rewards["kill"]
                            # Common overrides for attack behavior
                            if "defense_threshold" in effective:
                                flattened["attack_defense_threshold"] = effective["defense_threshold"]
                            if "defense_boost" in effective:
                                flattened["attack_defense_boost"] = effective["defense_boost"]
                        elif prefix == "gather":
                            if "base" in costs:
                                flattened["gather_base_cost"] = costs["base"]
                            if "success" in rewards:
                                flattened["gather_success_reward"] = rewards["success"]
                            if "failure_penalty" in rewards:
                                flattened["gather_failure_penalty"] = rewards["failure_penalty"]
                            # Additional gather tuning via overrides
                            for key_map in (
                                ("distance_penalty_factor", "gather_distance_penalty_factor"),
                                ("resource_threshold", "gather_resource_threshold"),
                                ("competition_penalty", "gather_competition_penalty"),
                                ("efficiency_bonus", "gather_efficiency_bonus"),
                                ("max_wait_steps", "max_wait_steps"),
                            ):
                                src, dst = key_map
                                if src in effective:
                                    flattened[dst] = effective[src]
                        elif prefix == "share":
                            if "base" in costs:
                                flattened["share_base_cost"] = costs["base"]
                            if "success" in rewards:
                                flattened["share_success_reward"] = rewards["success"]
                            if "failure_penalty" in rewards:
                                flattened["share_failure_penalty"] = rewards["failure_penalty"]
                            # Additional share overrides
                            for key_map in (
                                ("range", "share_range"),
                                ("min_share_amount", "min_share_amount"),
                                ("max_share_amount", "max_share_amount"),
                                ("share_threshold", "share_threshold"),
                                ("cooperation_bonus", "share_cooperation_bonus"),
                                ("altruism_factor", "share_altruism_factor"),
                            ):
                                src, dst = key_map
                                if src in effective:
                                    flattened[dst] = effective[src]

                for action_name, action_data in actions_raw.items():
                    if isinstance(action_data, dict):
                        apply_profile_to_action(action_name, action_data)

                # Persist parsed structures for visibility
                config_dict["profiles"] = profiles
                config_dict["action_configs"] = action_cfgs

        # Merge flattened values on top of any existing keys
        config_dict.update(flattened)

        # Attach visualization dataclass last
        config_dict["visualization"] = VisualizationConfig(**vis_config)

        return cls(**config_dict)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        # Convert to dictionary, handling visualization config specially
        config_dict = self.__dict__.copy()
        config_dict["visualization"] = self.visualization.to_dict()

        with open(file_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if key == "visualization":
                config_dict[key] = self.visualization.to_dict()
            else:
                config_dict[key] = value
        return config_dict

    def copy(self):
        """Create a deep copy of the configuration."""
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create configuration from a dictionary."""
        # Handle visualization config specially
        vis_data = data.pop("visualization", {})
        data["visualization"] = VisualizationConfig(**vis_data)
        return cls(**data)
