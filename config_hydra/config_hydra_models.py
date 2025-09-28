"""
Pydantic models for Hydra configuration validation.

This module provides structured configuration models using Pydantic for
strong type validation, better error messages, and improved developer experience.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class VisualizationConfig(BaseModel):
    """Pydantic model for visualization configuration."""
    
    canvas_size: Tuple[int, int] = Field(default=(400, 400), description="Canvas dimensions (width, height)")
    padding: int = Field(default=20, ge=0, description="Canvas padding in pixels")
    background_color: str = Field(default="black", description="Background color")
    max_animation_frames: int = Field(default=5, ge=1, description="Maximum animation frames")
    animation_min_delay: int = Field(default=50, ge=1, description="Minimum animation delay in ms")
    max_resource_amount: int = Field(default=30, ge=1, description="Maximum resource amount for visualization")
    resource_colors: Dict[str, int] = Field(
        default_factory=lambda: {"glow_red": 150, "glow_green": 255, "glow_blue": 50},
        description="Resource color configuration"
    )
    resource_size: int = Field(default=2, ge=1, description="Resource size in pixels")
    agent_radius_scale: int = Field(default=2, ge=1, description="Agent radius scale factor")
    birth_radius_scale: int = Field(default=4, ge=1, description="Birth radius scale factor")
    death_mark_scale: float = Field(default=1.5, ge=0.1, description="Death mark scale factor")
    agent_colors: Dict[str, str] = Field(
        default_factory=lambda: {"SystemAgent": "blue", "IndependentAgent": "red"},
        description="Agent color mapping"
    )
    min_font_size: int = Field(default=10, ge=1, description="Minimum font size")
    font_scale_factor: int = Field(default=40, ge=1, description="Font scale factor")
    font_family: str = Field(default="arial", description="Font family")
    death_mark_color: List[int] = Field(
        default_factory=lambda: [255, 0, 0],
        description="Death mark color (RGB)"
    )
    birth_mark_color: List[int] = Field(
        default_factory=lambda: [255, 255, 255],
        description="Birth mark color (RGB)"
    )
    metric_colors: Dict[str, str] = Field(
        default_factory=lambda: {
            "total_agents": "#4a90e2",
            "system_agents": "#50c878",
            "independent_agents": "#e74c3c",
            "total_resources": "#f39c12",
            "average_agent_resources": "#9b59b6",
        },
        description="Metric color mapping"
    )
    
    @field_validator('canvas_size')
    @classmethod
    def validate_canvas_size(cls, v):
        """Validate canvas size dimensions."""
        if len(v) != 2:
            raise ValueError("Canvas size must be a tuple of (width, height)")
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        if width > 10000 or height > 10000:
            raise ValueError("Canvas dimensions too large (max 10000x10000)")
        return v
    
    @field_validator('death_mark_color', 'birth_mark_color')
    @classmethod
    def validate_color_rgb(cls, v):
        """Validate RGB color values."""
        if len(v) != 3:
            raise ValueError("Color must be RGB tuple with 3 values")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("RGB values must be between 0 and 255")
        return v


class RedisMemoryConfig(BaseModel):
    """Pydantic model for Redis memory configuration."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    environment: str = Field(default="development", description="Environment name")
    password: Optional[str] = Field(default=None, description="Redis password")
    timeout: int = Field(default=5, ge=1, description="Connection timeout in seconds")
    max_connections: int = Field(default=10, ge=1, description="Maximum connections")
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v):
        """Validate host format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Host cannot be empty")
        return v.strip()


class AgentParameters(BaseModel):
    """Pydantic model for agent-specific parameters."""
    
    gather_efficiency_multiplier: float = Field(default=0.4, ge=0.0, le=2.0, description="Gather efficiency multiplier")
    gather_cost_multiplier: float = Field(default=0.3, ge=0.0, le=2.0, description="Gather cost multiplier")
    min_resource_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Minimum resource threshold")
    share_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Resource sharing weight")
    attack_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Attack behavior weight")
    cooperation_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Cooperation threshold")
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Learning rate")
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Exploration rate")
    
    @field_validator('share_weight', 'attack_weight')
    @classmethod
    def validate_behavior_weights(cls, v, info):
        """Validate that behavior weights are reasonable."""
        if info.data and 'share_weight' in info.data and 'attack_weight' in info.data:
            total = info.data.get('share_weight', 0) + info.data.get('attack_weight', 0)
            if total > 1.0:
                logger.warning(f"Total behavior weights ({total}) exceed 1.0")
        return v


class AgentTypeRatios(BaseModel):
    """Pydantic model for agent type ratios."""
    
    SystemAgent: float = Field(default=0.33, ge=0.0, le=1.0, description="System agent ratio")
    IndependentAgent: float = Field(default=0.33, ge=0.0, le=1.0, description="Independent agent ratio")
    ControlAgent: float = Field(default=0.34, ge=0.0, le=1.0, description="Control agent ratio")
    
    @model_validator(mode='after')
    def validate_ratios_sum(self):
        """Validate that ratios sum to approximately 1.0."""
        total = self.SystemAgent + self.IndependentAgent + self.ControlAgent
        if abs(total - 1.0) > 0.01:  # Allow small floating point differences
            raise ValueError(f"Agent type ratios must sum to 1.0, got {total:.3f}")
        return self


class HydraSimulationConfig(BaseModel):
    """Pydantic model for complete Hydra simulation configuration."""
    
    # Environment settings
    width: int = Field(default=100, ge=10, le=10000, description="Environment width")
    height: int = Field(default=100, ge=10, le=10000, description="Environment height")
    
    # Position discretization settings
    position_discretization_method: str = Field(
        default="floor", 
        pattern="^(floor|round|ceil)$",
        description="Position discretization method"
    )
    use_bilinear_interpolation: bool = Field(default=True, description="Use bilinear interpolation for resources")
    
    # Agent settings
    system_agents: int = Field(default=10, ge=0, le=1000, description="Number of system agents")
    independent_agents: int = Field(default=10, ge=0, le=1000, description="Number of independent agents")
    control_agents: int = Field(default=10, ge=0, le=1000, description="Number of control agents")
    
    # Simulation settings
    max_steps: int = Field(default=1000, ge=1, le=1000000, description="Maximum simulation steps")
    max_population: int = Field(default=100, ge=1, le=10000, description="Maximum population")
    simulation_id: str = Field(default="simulation", min_length=1, max_length=100, description="Simulation identifier")
    
    # Database settings
    use_in_memory_db: bool = Field(default=False, description="Use in-memory database")
    in_memory_db_memory_limit_mb: int = Field(default=1000, ge=1, le=100000, description="In-memory DB memory limit in MB")
    persist_db_on_completion: bool = Field(default=True, description="Persist database on completion")
    db_pragma_profile: str = Field(
        default="balanced",
        pattern="^(safety|balanced|performance)$",
        description="Database pragma profile"
    )
    db_cache_size_mb: int = Field(default=100, ge=1, le=10000, description="Database cache size in MB")
    
    # Learning parameters
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Learning rate")
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0, description="Initial epsilon value")
    epsilon_min: float = Field(default=0.01, ge=0.0, le=1.0, description="Minimum epsilon value")
    epsilon_decay: float = Field(default=0.995, ge=0.0, le=1.0, description="Epsilon decay rate")
    
    # Debug settings
    debug: bool = Field(default=False, description="Enable debug mode")
    verbose_logging: bool = Field(default=False, description="Enable verbose logging")
    
    # Additional common fields (optional)
    environment: Optional[str] = Field(default=None, description="Environment name")
    in_memory_db_memory_limit_mb: Optional[int] = Field(default=None, description="In-memory DB memory limit in MB")
    initial_resource_level: Optional[int] = Field(default=None, description="Initial resource level")
    starvation_threshold: Optional[int] = Field(default=None, description="Starvation threshold")
    offspring_cost: Optional[int] = Field(default=None, description="Offspring cost")
    min_reproduction_resources: Optional[int] = Field(default=None, description="Minimum reproduction resources")
    offspring_initial_resources: Optional[int] = Field(default=None, description="Offspring initial resources")
    perception_radius: Optional[int] = Field(default=None, description="Perception radius")
    base_attack_strength: Optional[float] = Field(default=None, description="Base attack strength")
    base_defense_strength: Optional[float] = Field(default=None, description="Base defense strength")
    seed: Optional[int] = Field(default=None, description="Random seed")
    initial_resources: Optional[int] = Field(default=None, description="Initial resources")
    resource_regen_rate: Optional[float] = Field(default=None, description="Resource regeneration rate")
    resource_regen_amount: Optional[int] = Field(default=None, description="Resource regeneration amount")
    max_resource_amount: Optional[int] = Field(default=None, description="Maximum resource amount")
    base_consumption_rate: Optional[float] = Field(default=None, description="Base consumption rate")
    max_movement: Optional[int] = Field(default=None, description="Maximum movement")
    gathering_range: Optional[int] = Field(default=None, description="Gathering range")
    max_gather_amount: Optional[int] = Field(default=None, description="Maximum gather amount")
    territory_range: Optional[int] = Field(default=None, description="Territory range")
    gamma: Optional[float] = Field(default=None, description="Gamma value")
    memory_size: Optional[int] = Field(default=None, description="Memory size")
    batch_size: Optional[int] = Field(default=None, description="Batch size")
    training_frequency: Optional[int] = Field(default=None, description="Training frequency")
    dqn_hidden_size: Optional[int] = Field(default=None, description="DQN hidden size")
    tau: Optional[float] = Field(default=None, description="Tau value")
    starting_health: Optional[float] = Field(default=None, description="Starting health")
    attack_range: Optional[float] = Field(default=None, description="Attack range")
    attack_base_damage: Optional[float] = Field(default=None, description="Attack base damage")
    attack_kill_reward: Optional[float] = Field(default=None, description="Attack kill reward")
    in_memory_tables_to_persist: Optional[Any] = Field(default=None, description="In-memory tables to persist")
    db_synchronous_mode: Optional[str] = Field(default=None, description="Database synchronous mode")
    db_journal_mode: Optional[str] = Field(default=None, description="Database journal mode")
    db_custom_pragmas: Optional[Dict[str, Any]] = Field(default=None, description="Database custom pragmas")
    device_preference: Optional[str] = Field(default=None, description="Device preference")
    device_fallback: Optional[bool] = Field(default=None, description="Device fallback")
    device_memory_fraction: Optional[float] = Field(default=None, description="Device memory fraction")
    device_validate_compatibility: Optional[bool] = Field(default=None, description="Device validate compatibility")
    
    # Nested configurations
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization settings")
    redis: RedisMemoryConfig = Field(default_factory=RedisMemoryConfig, description="Redis settings")
    agent_type_ratios: AgentTypeRatios = Field(default_factory=AgentTypeRatios, description="Agent type ratios")
    agent_parameters: Dict[str, AgentParameters] = Field(
        default_factory=lambda: {
            "SystemAgent": AgentParameters(),
            "IndependentAgent": AgentParameters(),
            "ControlAgent": AgentParameters()
        },
        description="Agent-specific parameters"
    )
    
    @field_validator('epsilon_start', 'epsilon_min')
    @classmethod
    def validate_epsilon_values(cls, v, info):
        """Validate epsilon values are consistent."""
        if info.data and 'epsilon_start' in info.data and 'epsilon_min' in info.data:
            if info.data['epsilon_start'] < info.data['epsilon_min']:
                raise ValueError("epsilon_start must be >= epsilon_min")
        return v
    
    @field_validator('epsilon_decay')
    @classmethod
    def validate_epsilon_decay(cls, v):
        """Validate epsilon decay is reasonable."""
        if v <= 0.9:
            logger.warning(f"Epsilon decay ({v}) is quite aggressive")
        return v
    
    @model_validator(mode='after')
    def validate_agent_population(self):
        """Validate agent population settings."""
        total_agents = (
            self.system_agents + 
            self.independent_agents + 
            self.control_agents
        )
        max_pop = self.max_population
        
        if total_agents > max_pop:
            raise ValueError(f"Total agents ({total_agents}) exceeds max population ({max_pop})")
        
        return self
    
    @model_validator(mode='after')
    def validate_environment_size(self):
        """Validate environment size is reasonable."""
        width = self.width
        height = self.height
        
        if width * height > 10000000:  # 10M cells
            logger.warning(f"Large environment size: {width}x{height} ({width*height:,} cells)")
        
        return self
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True
        extra = "allow"  # Allow extra fields for flexibility
        validate_by_name = True


class HydraEnvironmentConfig(BaseModel):
    """Pydantic model for environment-specific configuration overrides."""
    
    debug: bool = Field(default=False, description="Enable debug mode")
    verbose_logging: bool = Field(default=False, description="Enable verbose logging")
    max_steps: int = Field(default=1000, ge=1, le=1000000, description="Maximum simulation steps")
    max_population: int = Field(default=100, ge=1, le=10000, description="Maximum population")
    use_in_memory_db: bool = Field(default=False, description="Use in-memory database")
    persist_db_on_completion: bool = Field(default=True, description="Persist database on completion")
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Learning rate")
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0, description="Initial epsilon value")
    epsilon_min: float = Field(default=0.01, ge=0.0, le=1.0, description="Minimum epsilon value")
    db_pragma_profile: str = Field(
        default="balanced",
        pattern="^(safety|balanced|performance)$",
        description="Database pragma profile"
    )
    db_cache_size_mb: int = Field(default=100, ge=1, le=10000, description="Database cache size in MB")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "allow"  # Allow extra fields for flexibility
        validate_by_name = True


class HydraAgentConfig(BaseModel):
    """Pydantic model for agent-specific configuration overrides."""
    
    agent_parameters: Dict[str, AgentParameters] = Field(
        description="Agent-specific parameters"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "allow"  # Allow extra fields for flexibility
        validate_by_name = True


def validate_config_dict(config_dict: Dict[str, Any]) -> HydraSimulationConfig:
    """
    Validate a configuration dictionary using Pydantic models.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        Validated HydraSimulationConfig instance
        
    Raises:
        ValidationError: If configuration validation fails
    """
    try:
        return HydraSimulationConfig(**config_dict)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def validate_environment_config(config_dict: Dict[str, Any]) -> HydraEnvironmentConfig:
    """
    Validate an environment configuration dictionary.
    
    Args:
        config_dict: Environment configuration dictionary to validate
        
    Returns:
        Validated HydraEnvironmentConfig instance
        
    Raises:
        ValidationError: If configuration validation fails
    """
    try:
        return HydraEnvironmentConfig(**config_dict)
    except Exception as e:
        logger.error(f"Environment configuration validation failed: {e}")
        raise


def validate_agent_config(config_dict: Dict[str, Any]) -> HydraAgentConfig:
    """
    Validate an agent configuration dictionary.
    
    Args:
        config_dict: Agent configuration dictionary to validate
        
    Returns:
        Validated HydraAgentConfig instance
        
    Raises:
        ValidationError: If configuration validation fails
    """
    try:
        return HydraAgentConfig(**config_dict)
    except Exception as e:
        logger.error(f"Agent configuration validation failed: {e}")
        raise