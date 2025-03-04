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

    @classmethod
    def from_yaml(cls, file_path: str) -> "SimulationConfig":
        """Load configuration from a YAML file."""
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle visualization config separately
        vis_config = config_dict.pop("visualization", {})
        config_dict["visualization"] = VisualizationConfig(**vis_config)

        return cls(**config_dict)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        # Convert to dictionary, handling visualization config specially
        config_dict = self.__dict__.copy()
        config_dict["visualization"] = self.visualization.__dict__

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
