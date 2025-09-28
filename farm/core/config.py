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
    # Environment settings
    width: int = 100
    height: int = 100

    # Position discretization settings
    position_discretization_method: str = "floor"  # Options: "floor", "round", "ceil"
    use_bilinear_interpolation: bool = (
        True  # Whether to use bilinear interpolation for resources
    )

    # Agent settings
    system_agents: int = 10
    independent_agents: int = 10
    control_agents: int = 10
    initial_resource_level: int = 0
    max_population: int = 3000
    starvation_threshold: int = 100
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

    # Observation settings
    observation: Optional[ObservationConfig] = None

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
    persist_db_on_completion: bool = (
        True  # Whether to persist in-memory DB to disk after simulation
    )
    in_memory_db_memory_limit_mb: Optional[int] = (
        None  # Memory limit for in-memory DB (None = no limit)
    )
    in_memory_tables_to_persist: Optional[List[str]] = (
        None  # Tables to persist (None = all tables)
    )

    # Database pragma settings
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

    # Redis configuration
    redis: RedisMemoryConfig = field(default_factory=RedisMemoryConfig)

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
    max_steps: int = 1000  # Maximum steps for environment termination

    # Device configuration for neural network computations
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

    # Logging and debugging
    debug: bool = False
    verbose_logging: bool = False

    # Configuration versioning
    config_version: Optional[str] = None
    config_created_at: Optional[str] = None
    config_description: Optional[str] = None

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        # Convert to dictionary, handling visualization and redis configs specially
        config_dict = self.to_dict()

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if key == "visualization":
                config_dict[key] = self.visualization.to_dict()
            elif key == "redis":
                config_dict[key] = self.redis.to_dict()
            elif key == "observation":
                if self.observation:
                    obs_dict = self.observation.model_dump()
                    # Convert torch dtype to string for JSON serialization
                    if 'dtype' in obs_dict and hasattr(obs_dict['dtype'], '__name__'):
                        obs_dict['dtype'] = obs_dict['dtype'].__name__
                    elif 'dtype' in obs_dict:
                        obs_dict['dtype'] = str(obs_dict['dtype'])
                    # Convert StorageMode enum to string for JSON serialization
                    if 'storage_mode' in obs_dict and hasattr(obs_dict['storage_mode'], 'value'):
                        obs_dict['storage_mode'] = obs_dict['storage_mode'].value
                    config_dict[key] = obs_dict
                else:
                    config_dict[key] = None
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

        # Handle redis config specially
        redis_data = data.pop("redis", {})
        data["redis"] = RedisMemoryConfig(**redis_data)

        # Handle observation config specially
        obs_data = data.pop("observation", None)
        if obs_data:
            data["observation"] = ObservationConfig(**obs_data)

        return cls(**data)

    @classmethod
    def from_yaml(cls, file_path: str) -> "SimulationConfig":
        """
        Load configuration from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            SimulationConfig: Loaded configuration
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_centralized_config(
        cls,
        environment: str = "development",
        profile: Optional[str] = None,
        config_dir: str = "config",
        use_cache: bool = True,
        strict_validation: bool = False,
        auto_repair: bool = True
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
        from farm.core.config_cache import OptimizedConfigLoader

        # Use cache loader for basic loading
        loader = OptimizedConfigLoader()
        config = loader.load_centralized_config(
            environment=environment,
            profile=profile,
            config_dir=config_dir,
            use_cache=use_cache
        )

        # Apply validation and repair if requested
        if strict_validation or auto_repair:
            from farm.core.config_validation import SafeConfigLoader
            loader = SafeConfigLoader()
            config, status_info = loader.load_config_safely(
                environment=environment,
                profile=profile,
                config_dir=config_dir,
                strict_validation=strict_validation,
                auto_repair=auto_repair
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
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
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
        config_dict.pop('config_version', None)
        config_dict.pop('config_created_at', None)
        config_dict.pop('config_description', None)

        # Convert to JSON string with sorted keys for consistent hashing
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:16]  # Short hash

    def version_config(self, description: Optional[str] = None) -> 'SimulationConfig':
        """
        Create a versioned copy of this configuration.

        Args:
            description: Optional description of this configuration version

        Returns:
            SimulationConfig: Versioned configuration
        """
        versioned_config = copy.deepcopy(self)
        versioned_config.config_version = self.generate_version_hash()
        versioned_config.config_created_at = datetime.now().isoformat()
        versioned_config.config_description = description
        return versioned_config

    def save_versioned_config(self, directory: str, description: Optional[str] = None) -> str:
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
        if not self.config_version:
            versioned_config = self.version_config(description)
        else:
            versioned_config = self

        filename = f"config_{versioned_config.config_version}.yaml"
        filepath = os.path.join(directory, filename)

        versioned_config.to_yaml(filepath)
        return filepath

    @classmethod
    def load_versioned_config(cls, directory: str, version: str) -> 'SimulationConfig':
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
            raise FileNotFoundError(f"Configuration version {version} not found in {directory}")

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
            from farm.core.observations import ObservationConfig
            config_dict["observation"] = ObservationConfig(**obs_config)

        return cls(**config_dict)

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
            if filename.startswith('config_') and filename.endswith('.yaml'):
                version_hash = filename.replace('config_', '').replace('.yaml', '')
                filepath = os.path.join(directory, filename)

                try:
                    config = cls.from_yaml(filepath)
                    versions.append({
                        'version': version_hash,
                        'created_at': config.config_created_at,
                        'description': config.config_description,
                        'filepath': filepath
                    })
                except Exception:
                    # Skip invalid config files
                    continue

        return sorted(versions, key=lambda x: x.get('created_at', ''), reverse=True)

    def diff_config(self, other: 'SimulationConfig') -> Dict[str, Any]:
        """
        Compare this configuration with another and return differences.

        Args:
            other: Configuration to compare against

        Returns:
            Dict: Dictionary containing differences
        """
        def _dict_diff(d1: Dict, d2: Dict, path: str = '') -> Dict[str, Any]:
            """Recursively find differences between two dictionaries."""
            diff = {}

            # Keys in d1 but not in d2
            for key in d1.keys() - d2.keys():
                diff[f"{path}.{key}" if path else key] = {'self': d1[key], 'other': None}

            # Keys in d2 but not in d1
            for key in d2.keys() - d1.keys():
                diff[f"{path}.{key}" if path else key] = {'self': None, 'other': d2[key]}

            # Keys in both
            for key in d1.keys() & d2.keys():
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    nested_diff = _dict_diff(d1[key], d2[key], f"{path}.{key}" if path else key)
                    diff.update(nested_diff)
                elif d1[key] != d2[key]:
                    diff[f"{path}.{key}" if path else key] = {'self': d1[key], 'other': d2[key]}

            return diff

        self_dict = self.to_dict()
        other_dict = other.to_dict()

        # Remove versioning fields from comparison
        for d in [self_dict, other_dict]:
            d.pop('config_version', None)
            d.pop('config_created_at', None)
            d.pop('config_description', None)

        return _dict_diff(self_dict, other_dict)