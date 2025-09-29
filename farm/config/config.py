import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

from farm.core.observations import ObservationConfig


@dataclass
class SpatialIndexConfig:
    """Configuration for spatial indexing and batch updates."""

    enable_batch_updates: bool = True
    region_size: float = 50.0
    max_batch_size: int = 100
    max_regions: int = 1000
    enable_quadtree_indices: bool = False
    enable_spatial_hash_indices: bool = False
    spatial_hash_cell_size: Optional[float] = None
    performance_monitoring: bool = True
    debug_queries: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert SpatialIndexConfig to a JSON-serializable dictionary."""
        return {
            "enable_batch_updates": self.enable_batch_updates,
            "region_size": self.region_size,
            "max_batch_size": self.max_batch_size,
            "max_regions": self.max_regions,
            "enable_quadtree_indices": self.enable_quadtree_indices,
            "enable_spatial_hash_indices": self.enable_spatial_hash_indices,
            "spatial_hash_cell_size": self.spatial_hash_cell_size,
            "performance_monitoring": self.performance_monitoring,
            "debug_queries": self.debug_queries,
        }


@dataclass
class EnvironmentConfig:
    """Configuration for simulation environment settings."""

    width: int = 100
    height: int = 100
    position_discretization_method: str = "floor"  # Options: "floor", "round", "ceil"
    use_bilinear_interpolation: bool = (
        True  # Whether to use bilinear interpolation for resources
    )
    spatial_index: Optional[SpatialIndexConfig] = None


@dataclass
class PopulationConfig:
    """Configuration for agent population settings."""

    system_agents: int = 10
    independent_agents: int = 10
    control_agents: int = 10
    max_population: int = 3000
    agent_type_ratios: Dict[str, float] = field(
        default_factory=lambda: {
            "SystemAgent": 0.33,
            "IndependentAgent": 0.33,
            "ControlAgent": 0.34,
        }
    )


@dataclass
class ResourceConfig:
    """Configuration for resource system settings."""

    initial_resources: int = 20
    resource_regen_rate: float = 0.1
    resource_regen_amount: int = 2
    max_resource_amount: int = 30
    memmap_delete_on_close: bool = False  # Delete memmap files when Environment closes


@dataclass
class LearningConfig:
    """Configuration for reinforcement learning parameters."""

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


@dataclass
class CombatConfig:
    """Configuration for combat system settings."""

    starting_health: float = 100.0
    attack_range: float = 20.0
    attack_base_damage: float = 10.0
    attack_kill_reward: float = 5.0


@dataclass
class AgentBehaviorConfig:
    """Configuration for agent behavior parameters."""

    base_consumption_rate: float = 0.15
    max_movement: int = 8
    gathering_range: int = 30
    max_gather_amount: int = 3
    territory_range: int = 30
    perception_radius: int = 2
    base_attack_strength: int = 2
    base_defense_strength: int = 2
    initial_resource_level: int = 0
    starvation_threshold: float = 100
    offspring_cost: int = 3
    min_reproduction_resources: int = 8
    offspring_initial_resources: int = 5
    social_range: int = 30  # Range for social interactions (share/attack)
    move_mult_no_resources: float = 1.5  # Multiplier when no resources nearby
    gather_mult_low_resources: float = 1.5  # Multiplier when resources needed
    share_mult_wealthy: float = 1.3  # Multiplier when agent has excess resources
    share_mult_poor: float = 0.5  # Multiplier when agent needs resources
    attack_starvation_threshold: float = (
        0.5  # Starvation risk threshold for desperate behavior
    )
    attack_mult_desperate: float = 1.4  # Multiplier when desperate for resources
    attack_mult_stable: float = 0.6  # Multiplier when resource stable
    max_wait_steps: int = 10  # Maximum steps to wait between gathering attempts


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""

    use_in_memory_db: bool = False  # Whether to use in-memory database
    persist_db_on_completion: bool = (
        True  # Whether to persist in-memory DB to disk after simulation
    )
    in_memory_db_memory_limit_mb: Optional[int] = (
        None  # Memory limit for in-memory DB (None = no limit)
    )
    in_memory_tables_to_persist: Optional[List[str]] = (
        None  # Tables to persist (None = all tables)
    )
    db_pragma_profile: str = (
        "balanced"  # Options: "balanced", "performance", "safety", "memory"
    )
    db_cache_size_mb: int = 200  # Cache size in MB
    db_synchronous_mode: str = "NORMAL"  # Options: "OFF", "NORMAL", "FULL"
    db_journal_mode: str = (
        "WAL"  # Options: "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"
    )
    db_custom_pragmas: Dict[str, str] = field(
        default_factory=dict
    )  # Custom pragma overrides


@dataclass
class DeviceConfig:
    """Configuration for neural network device settings."""

    device_preference: str = (
        "auto"  # Options: "auto", "cpu", "cuda", "cuda:X" (specific GPU)
    )
    device_fallback: bool = (
        True  # Whether to fallback to CPU if preferred device unavailable
    )
    device_memory_fraction: Optional[float] = (
        None  # GPU memory fraction to use (0.0-1.0)
    )
    device_validate_compatibility: bool = (
        True  # Whether to validate tensor compatibility when moving devices
    )


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning phases."""

    curriculum_phases: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"steps": 100, "enabled_actions": ["move", "gather"]},
            {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
            {
                "steps": -1,
                "enabled_actions": ["move", "gather", "share", "attack", "reproduce"],
            },
        ]
    )  # -1 for remaining steps


@dataclass
class LoggingConfig:
    """Configuration for logging and debugging."""

    debug: bool = False
    verbose_logging: bool = False


@dataclass
class VersioningConfig:
    """Configuration for versioning metadata."""

    config_version: Optional[str] = None
    config_created_at: Optional[str] = None
    config_description: Optional[str] = None


@dataclass
class ModuleConfig:
    """Configuration for specialized learning modules."""

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

        def _jsonify(val):
            if isinstance(val, tuple):
                return list(val)
            if isinstance(val, dict):
                return {k: _jsonify(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_jsonify(v) for v in val]
            return val

        return {
            key: _jsonify(value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualizationConfig":
        """Create visualization config from a dictionary."""
        # Convert canvas_size from list back to tuple if needed
        if "canvas_size" in data and isinstance(data["canvas_size"], list):
            data = data.copy()  # Don't modify the original
            data["canvas_size"] = tuple(data["canvas_size"])
        return cls(**data)


@dataclass
class RedisMemoryConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = True
    environment: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert RedisMemoryConfig to a JSON-serializable dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password,
            "decode_responses": self.decode_responses,
            "environment": self.environment,
        }


@dataclass
class SimulationConfig:
    """Main simulation configuration using composition of focused sub-configs."""

    # Core simulation settings
    simulation_steps: int = 100  # Default value
    max_steps: int = 1000  # Maximum steps for environment termination
    seed: Optional[int] = 1234567890

    # Agent-specific parameters (kept at top level for backward compatibility)
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

    # Nested configuration objects
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    agent_behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    modules: ModuleConfig = field(default_factory=ModuleConfig)

    # Existing nested configs (kept for backward compatibility)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    observation: Optional[ObservationConfig] = None
    redis: RedisMemoryConfig = field(default_factory=RedisMemoryConfig)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        # Convert to dictionary, handling visualization and redis configs specially
        config_dict = self.to_dict()

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        config_dict = {}

        # Handle top-level fields
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            elif key == "visualization":
                config_dict[key] = self.visualization.to_dict()
            elif key == "redis":
                config_dict[key] = self.redis.to_dict()
            elif key == "observation":
                if self.observation:
                    obs_dict = self.observation.model_dump()
                    # Convert torch dtype to string for JSON serialization
                    if "dtype" in obs_dict and hasattr(obs_dict["dtype"], "__name__"):
                        obs_dict["dtype"] = obs_dict["dtype"].__name__
                    elif "dtype" in obs_dict:
                        obs_dict["dtype"] = str(obs_dict["dtype"])
                    # Convert StorageMode enum to string for JSON serialization
                    if "storage_mode" in obs_dict and hasattr(
                        obs_dict["storage_mode"], "value"
                    ):
                        obs_dict["storage_mode"] = obs_dict["storage_mode"].value
                    config_dict[key] = obs_dict
                else:
                    config_dict[key] = None
            elif key in [
                "environment",
                "population",
                "resources",
                "learning",
                "combat",
                "agent_behavior",
                "database",
                "device",
                "curriculum",
                "logging",
                "versioning",
                "modules",
            ]:
                # Convert nested config objects to dicts
                for k, v in value.__dict__.items():
                    if k.startswith("_"):
                        continue
                    elif key == "environment" and k == "spatial_index" and v is not None:
                        # Handle SpatialIndexConfig specially
                        if hasattr(v, 'to_dict'):
                            config_dict[f"{key}.{k}"] = v.to_dict()
                        else:
                            config_dict[f"{key}.{k}"] = v
                    else:
                        config_dict[f"{key}.{k}"] = v
            else:
                config_dict[key] = value
        return config_dict

    def copy(self):
        """Create a deep copy of the configuration."""
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create configuration from a dictionary."""
        # Handle backward compatibility: convert flat structure to nested
        nested_data = cls._convert_flat_to_nested(data)

        # Handle visualization config specially
        vis_data = nested_data.pop("visualization", {})
        if isinstance(vis_data, VisualizationConfig):
            nested_data["visualization"] = vis_data
        else:
            nested_data["visualization"] = VisualizationConfig(**vis_data)

        # Handle redis config specially
        redis_data = nested_data.pop("redis", {})
        if isinstance(redis_data, RedisMemoryConfig):
            nested_data["redis"] = redis_data
        else:
            nested_data["redis"] = RedisMemoryConfig(**redis_data)

        # Handle observation config specially
        obs_data = nested_data.pop("observation", None)
        if obs_data:
            if isinstance(obs_data, dict):
                nested_data["observation"] = ObservationConfig(**obs_data)
            else:
                nested_data["observation"] = obs_data

        return cls(**nested_data)

    @classmethod
    def _convert_flat_to_nested(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat configuration structure to nested structure for backward compatibility."""
        nested_data = data.copy()

        # Handle dotted notation keys (e.g., "environment.width" -> nested structure)
        dotted_keys = {}
        for key, value in list(nested_data.items()):
            if "." in key:
                parts = key.split(".")
                if len(parts) == 2:
                    parent, child = parts
                    if parent not in dotted_keys:
                        dotted_keys[parent] = {}
                    dotted_keys[parent][child] = value
                    nested_data.pop(key)

        # Convert dotted keys to config objects
        for parent, config_dict in dotted_keys.items():
            if parent == "environment":
                nested_data[parent] = EnvironmentConfig(**config_dict)
            elif parent == "population":
                nested_data[parent] = PopulationConfig(**config_dict)
            elif parent == "resources":
                nested_data[parent] = ResourceConfig(**config_dict)
            elif parent == "learning":
                nested_data[parent] = LearningConfig(**config_dict)
            elif parent == "combat":
                nested_data[parent] = CombatConfig(**config_dict)
            elif parent == "agent_behavior":
                nested_data[parent] = AgentBehaviorConfig(**config_dict)
            elif parent == "database":
                nested_data[parent] = DatabaseConfig(**config_dict)
            elif parent == "device":
                nested_data[parent] = DeviceConfig(**config_dict)
            elif parent == "curriculum":
                nested_data[parent] = CurriculumConfig(**config_dict)
            elif parent == "logging":
                nested_data[parent] = LoggingConfig(**config_dict)
            elif parent == "versioning":
                nested_data[parent] = VersioningConfig(**config_dict)
            elif parent == "modules":
                nested_data[parent] = ModuleConfig(**config_dict)

        # First, extract all module-related fields to avoid conflicts
        # Only extract fields that are specifically for the specialized learning modules
        # Exclude fields that belong to other configs like agent_behavior
        module_fields = {}
        module_specific_fields = {
            # Gathering module
            "gather_target_update_freq",
            "gather_memory_size",
            "gather_learning_rate",
            "gather_gamma",
            "gather_epsilon_start",
            "gather_epsilon_min",
            "gather_epsilon_decay",
            "gather_dqn_hidden_size",
            "gather_batch_size",
            "gather_tau",
            "gather_success_reward",
            "gather_failure_penalty",
            "gather_base_cost",
            "gather_distance_penalty_factor",
            "gather_resource_threshold",
            "gather_competition_penalty",
            "gather_efficiency_bonus",
            # Sharing module
            "share_range",
            "share_target_update_freq",
            "share_memory_size",
            "share_learning_rate",
            "share_gamma",
            "share_epsilon_start",
            "share_epsilon_min",
            "share_epsilon_decay",
            "share_dqn_hidden_size",
            "share_batch_size",
            "share_tau",
            "share_success_reward",
            "share_failure_penalty",
            "share_base_cost",
            "min_share_amount",
            "max_share_amount",
            "share_threshold",
            "share_cooperation_bonus",
            "share_altruism_factor",
            "cooperation_memory",
            "cooperation_score_threshold",
            # Movement module
            "move_target_update_freq",
            "move_memory_size",
            "move_learning_rate",
            "move_gamma",
            "move_epsilon_start",
            "move_epsilon_min",
            "move_epsilon_decay",
            "move_dqn_hidden_size",
            "move_batch_size",
            "move_reward_history_size",
            "move_epsilon_adapt_threshold",
            "move_epsilon_adapt_factor",
            "move_min_reward_samples",
            "move_tau",
            "move_base_cost",
            "move_resource_approach_reward",
            "move_resource_retreat_penalty",
            # Attack module
            "attack_target_update_freq",
            "attack_memory_size",
            "attack_learning_rate",
            "attack_gamma",
            "attack_epsilon_start",
            "attack_epsilon_min",
            "attack_epsilon_decay",
            "attack_dqn_hidden_size",
            "attack_batch_size",
            "attack_tau",
            "attack_base_cost",
            "attack_success_reward",
            "attack_failure_penalty",
            "attack_defense_threshold",
            "attack_defense_boost",
            "attack_kill_reward",
        }

        for field in module_specific_fields:
            if field in nested_data:
                module_fields[field] = nested_data.pop(field)

        if module_fields:
            nested_data["modules"] = ModuleConfig(**module_fields)

        # Mapping of flat keys to nested config classes
        config_mappings = {
            "environment": (
                EnvironmentConfig,
                [
                    "width",
                    "height",
                    "position_discretization_method",
                    "use_bilinear_interpolation",
                ],
            ),
            "population": (
                PopulationConfig,
                [
                    "system_agents",
                    "independent_agents",
                    "control_agents",
                    "max_population",
                    "agent_type_ratios",
                ],
            ),
            "resources": (
                ResourceConfig,
                [
                    "initial_resources",
                    "resource_regen_rate",
                    "resource_regen_amount",
                    "max_resource_amount",
                ],
            ),
            "learning": (
                LearningConfig,
                [
                    "learning_rate",
                    "gamma",
                    "epsilon_start",
                    "epsilon_min",
                    "epsilon_decay",
                    "memory_size",
                    "batch_size",
                    "training_frequency",
                    "dqn_hidden_size",
                    "tau",
                ],
            ),
            "combat": (
                CombatConfig,
                [
                    "starting_health",
                    "attack_range",
                    "attack_base_damage",
                    "attack_kill_reward",
                ],
            ),
            "agent_behavior": (
                AgentBehaviorConfig,
                [
                    "base_consumption_rate",
                    "max_movement",
                    "gathering_range",
                    "max_gather_amount",
                    "territory_range",
                    "perception_radius",
                    "base_attack_strength",
                    "base_defense_strength",
                    "initial_resource_level",
                    "starvation_threshold",
                    "offspring_cost",
                    "min_reproduction_resources",
                    "offspring_initial_resources",
                    "social_range",
                    "move_mult_no_resources",
                    "gather_mult_low_resources",
                    "share_mult_wealthy",
                    "share_mult_poor",
                    "attack_starvation_threshold",
                    "attack_mult_desperate",
                    "attack_mult_stable",
                    "max_wait_steps",
                ],
            ),
            "database": (
                DatabaseConfig,
                [
                    "use_in_memory_db",
                    "persist_db_on_completion",
                    "in_memory_db_memory_limit_mb",
                    "in_memory_tables_to_persist",
                    "db_pragma_profile",
                    "db_cache_size_mb",
                    "db_synchronous_mode",
                    "db_journal_mode",
                    "db_custom_pragmas",
                ],
            ),
            "device": (
                DeviceConfig,
                [
                    "device_preference",
                    "device_fallback",
                    "device_memory_fraction",
                    "device_validate_compatibility",
                ],
            ),
            "curriculum": (CurriculumConfig, ["curriculum_phases"]),
            "logging": (LoggingConfig, ["debug", "verbose_logging"]),
            "versioning": (
                VersioningConfig,
                ["config_version", "config_created_at", "config_description"],
            ),
        }

        for config_name, (config_class, fields) in config_mappings.items():
            config_dict = {}
            for field in fields:
                if field in nested_data:
                    config_dict[field] = nested_data.pop(field)

            if config_dict:
                nested_data[config_name] = config_class(**config_dict)

        return nested_data

    @classmethod
    def from_yaml(cls, file_path: str) -> "SimulationConfig":
        """
        Load configuration from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            SimulationConfig: Loaded configuration
        """
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_centralized_config(
        cls,
        environment: str = "development",
        profile: Optional[str] = None,
        config_dir: str = "farm/config",
        use_cache: bool = True,
        strict_validation: bool = False,
        auto_repair: bool = False,
    ) -> "SimulationConfig":
        """
        Load configuration from the centralized config structure.

        Args:
            environment: Environment name (development, production, testing)
            profile: Optional profile name (benchmark, simulation, research)
            config_dir: Base configuration directory
            use_cache: Whether to use caching for performance
            strict_validation: Whether to treat warnings as errors
            auto_repair: Whether to attempt automatic repair of validation errors

        Returns:
            SimulationConfig: Loaded and merged configuration

        Raises:
            ConfigurationError: If validation fails and auto_repair is disabled
        """
        from .cache import OptimizedConfigLoader

        # Use cache loader for basic loading
        loader = OptimizedConfigLoader()
        config = loader.load_centralized_config(
            environment=environment,
            profile=profile,
            config_dir=config_dir,
            use_cache=use_cache,
        )

        # Apply validation and repair if requested
        if strict_validation or auto_repair:
            from .validation import SafeConfigLoader

            loader = SafeConfigLoader()
            config, status_info = loader.load_config_safely(
                environment=environment,
                profile=profile,
                config_dir=config_dir,
                strict_validation=strict_validation,
                auto_repair=auto_repair,
            )

        return config

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge override dictionary into base dictionary.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Dict: Merged dictionary
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = SimulationConfig._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def generate_version_hash(self) -> str:
        """
        Generate a unique hash for this configuration.

        Returns:
            str: SHA256 hash of the configuration content
        """
        # Convert config to dict and remove versioning fields for consistent hashing
        config_dict = self.to_dict()
        # Remove versioning fields (they may be in dotted format or flat format)
        versioning_keys = [
            "versioning.config_version",
            "versioning.config_created_at",
            "versioning.config_description",
            "config_version",
            "config_created_at",
            "config_description",
        ]
        for key in versioning_keys:
            config_dict.pop(key, None)

        # Convert to JSON string with sorted keys for consistent hashing
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        # Use a 32-character prefix to reduce collision risk while keeping filenames manageable
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:32]

    def version_config(self, description: Optional[str] = None) -> "SimulationConfig":
        """
        Create a versioned copy of this configuration.

        Args:
            description: Optional description of this configuration version

        Returns:
            SimulationConfig: Versioned configuration
        """
        versioned_config = copy.deepcopy(self)
        versioned_config.versioning.config_version = self.generate_version_hash()
        versioned_config.versioning.config_created_at = datetime.now().isoformat()
        versioned_config.versioning.config_description = description
        return versioned_config

    def save_versioned_config(
        self, directory: str, description: Optional[str] = None
    ) -> str:
        """
        Save this configuration as a versioned file.

        Args:
            directory: Directory to save the configuration
            description: Optional description of this configuration

        Returns:
            str: Path to the saved configuration file
        """
        os.makedirs(directory, exist_ok=True)

        # Version the config if not already versioned
        if not self.versioning.config_version:
            versioned_config = self.version_config(description)
        else:
            versioned_config = self

        filename = f"config_{versioned_config.versioning.config_version}.yaml"
        filepath = os.path.join(directory, filename)

        versioned_config.to_yaml(filepath)
        return filepath

    @classmethod
    def load_versioned_config(cls, directory: str, version: str) -> "SimulationConfig":
        """
        Load a specific versioned configuration.

        Args:
            directory: Directory containing versioned configs
            version: Version hash to load

        Returns:
            SimulationConfig: Loaded configuration

        Raises:
            FileNotFoundError: If version doesn't exist
        """
        filepath = os.path.join(directory, f"config_{version}.yaml")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Configuration version {version} not found in {directory}"
            )

        # Load directly from YAML file
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested configs
        vis_config = config_dict.pop("visualization", {})
        config_dict["visualization"] = VisualizationConfig(**vis_config)

        redis_config = config_dict.pop("redis", {})
        config_dict["redis"] = RedisMemoryConfig(**redis_config)

        obs_config = config_dict.pop("observation", None)
        if obs_config:
            # Lazy load ObservationConfig to avoid heavy dependencies in lightweight contexts
            try:
                from farm.core.observations import ObservationConfig

                config_dict["observation"] = ObservationConfig(**obs_config)
            except ImportError:
                # If import fails, store as raw dict and let caller handle it
                config_dict["observation"] = obs_config

        return cls(**cls._convert_flat_to_nested(config_dict))

    @classmethod
    def list_config_versions(cls, directory: str) -> List[Dict[str, Any]]:
        """
        List all available configuration versions in a directory.

        Args:
            directory: Directory to scan for versioned configs

        Returns:
            List[Dict]: List of version information
        """
        if not os.path.exists(directory):
            return []

        versions = []
        for filename in os.listdir(directory):
            if filename.startswith("config_") and filename.endswith(".yaml"):
                version_hash = filename.replace("config_", "").replace(".yaml", "")
                filepath = os.path.join(directory, filename)

                try:
                    config = cls.from_yaml(filepath)
                    versions.append(
                        {
                            "version": version_hash,
                            "created_at": config.versioning.config_created_at,
                            "description": config.versioning.config_description,
                            "filepath": filepath,
                        }
                    )
                except Exception:
                    # Skip invalid config files
                    continue

        return sorted(versions, key=lambda x: x.get("created_at", ""), reverse=True)

    def diff_config(self, other: "SimulationConfig") -> Dict[str, Any]:
        """
        Compare this configuration with another and return differences.

        Args:
            other: Configuration to compare against

        Returns:
            Dict: Dictionary containing differences
        """

        def _dict_diff(d1: Dict, d2: Dict, path: str = "") -> Dict[str, Any]:
            """Recursively find differences between two dictionaries."""
            diff = {}

            # Keys in d1 but not in d2
            for key in d1.keys() - d2.keys():
                diff[f"{path}.{key}" if path else key] = {
                    "self": d1[key],
                    "other": None,
                }

            # Keys in d2 but not in d1
            for key in d2.keys() - d1.keys():
                diff[f"{path}.{key}" if path else key] = {
                    "self": None,
                    "other": d2[key],
                }

            # Keys in both
            for key in d1.keys() & d2.keys():
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    nested_diff = _dict_diff(
                        d1[key], d2[key], f"{path}.{key}" if path else key
                    )
                    diff.update(nested_diff)
                elif d1[key] != d2[key]:
                    diff[f"{path}.{key}" if path else key] = {
                        "self": d1[key],
                        "other": d2[key],
                    }

            return diff

        self_dict = self.to_dict()
        other_dict = other.to_dict()

        # Remove versioning fields from comparison
        versioning_keys = [
            "versioning.config_version",
            "versioning.config_created_at",
            "versioning.config_description",
            "config_version",
            "config_created_at",
            "config_description",
        ]
        for d in [self_dict, other_dict]:
            for key in versioning_keys:
                d.pop(key, None)

        return _dict_diff(self_dict, other_dict)
