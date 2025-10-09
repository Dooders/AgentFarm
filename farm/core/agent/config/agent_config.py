"""
Type-safe configuration classes for agent components.

These immutable value objects replace the verbose get_nested_then_flat pattern
with clean, typed configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class MovementConfig:
    """
    Configuration for agent movement component.

    Attributes:
        max_movement: Maximum distance agent can move per turn
        position_discretization_method: How to discretize positions ("floor", "round", "ceil")
    """

    max_movement: float = 8.0
    position_discretization_method: str = "floor"

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_movement < 0:
            raise ValueError("max_movement must be non-negative")
        if self.position_discretization_method not in ("floor", "round", "ceil"):
            raise ValueError(
                f"Invalid discretization method: {self.position_discretization_method}"
            )


@dataclass(frozen=True)
class ResourceConfig:
    """
    Configuration for agent resource component.

    Attributes:
        base_consumption_rate: Resources consumed per turn
        starvation_threshold: Turns without resources before death
        initial_resources: Starting resource level for new agents
    """

    base_consumption_rate: int = 1
    starvation_threshold: int = 100
    initial_resources: int = 10

    def __post_init__(self):
        """Validate configuration values."""
        if self.base_consumption_rate < 0:
            raise ValueError("base_consumption_rate must be non-negative")
        if self.starvation_threshold < 0:
            raise ValueError("starvation_threshold must be non-negative")
        if self.initial_resources < 0:
            raise ValueError("initial_resources must be non-negative")


@dataclass(frozen=True)
class CombatConfig:
    """
    Configuration for agent combat component.

    Attributes:
        starting_health: Initial and maximum health
        base_attack_strength: Base damage dealt in attacks
        base_defense_strength: Defense bonus when defending
        defense_reduction: Damage reduction multiplier when defending (0.5 = 50% reduction)
        defense_duration: Turns to stay in defensive stance
    """

    starting_health: float = 100.0
    base_attack_strength: float = 10.0
    base_defense_strength: float = 5.0
    defense_reduction: float = 0.5
    defense_duration: int = 1

    def __post_init__(self):
        """Validate configuration values."""
        if self.starting_health <= 0:
            raise ValueError("starting_health must be positive")
        if self.base_attack_strength < 0:
            raise ValueError("base_attack_strength must be non-negative")
        if self.base_defense_strength < 0:
            raise ValueError("base_defense_strength must be non-negative")
        if not 0 <= self.defense_reduction <= 1:
            raise ValueError("defense_reduction must be between 0 and 1")
        if self.defense_duration < 0:
            raise ValueError("defense_duration must be non-negative")


@dataclass(frozen=True)
class ReproductionConfig:
    """
    Configuration for agent reproduction component.

    Attributes:
        offspring_cost: Resources consumed by parent to reproduce
        offspring_initial_resources: Starting resources for offspring
        reproduction_threshold: Minimum resources needed to reproduce
    """

    offspring_cost: int = 5
    offspring_initial_resources: int = 10
    reproduction_threshold: int = 8  # Reduced from 20 to 8 to allow reproduction with starting resources

    def __post_init__(self):
        """Validate configuration values."""
        if self.offspring_cost < 0:
            raise ValueError("offspring_cost must be non-negative")
        if self.offspring_initial_resources < 0:
            raise ValueError("offspring_initial_resources must be non-negative")
        if self.reproduction_threshold < 0:
            raise ValueError("reproduction_threshold must be non-negative")


@dataclass(frozen=True)
class PerceptionConfig:
    """
    Configuration for agent perception component.

    Attributes:
        perception_radius: How far agent can perceive environment
        perception_grid_size: Size of perception grid (derived from radius)
    """

    perception_radius: int = 5

    @property
    def perception_grid_size(self) -> int:
        """Calculated grid size based on radius."""
        return 2 * self.perception_radius + 1

    def __post_init__(self):
        """Validate configuration values."""
        if self.perception_radius < 0:
            raise ValueError("perception_radius must be non-negative")


@dataclass(frozen=True)
class AgentConfig:
    """
    Complete agent configuration combining all component configs.

    This is an immutable value object that provides type-safe access to
    all configuration values, replacing the verbose get_nested_then_flat pattern.

    Example:
        >>> config = AgentConfig(
        ...     movement=MovementConfig(max_movement=10.0),
        ...     resource=ResourceConfig(base_consumption_rate=2)
        ... )
        >>> print(config.movement.max_movement)  # 10.0
        >>> print(config.resource.base_consumption_rate)  # 2
    """

    movement: MovementConfig = field(default_factory=MovementConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "AgentConfig":
        """
        Create AgentConfig from a dictionary (e.g., from YAML/JSON).

        Args:
            config_dict: Dictionary with optional keys: "movement", "resource",
                        "combat", "reproduction", "perception"

        Returns:
            AgentConfig: Configured instance with validated values

        Example:
            >>> config = AgentConfig.from_dict({
            ...     "movement": {"max_movement": 12.0},
            ...     "resource": {"base_consumption_rate": 2},
            ... })
        """
        return AgentConfig(
            movement=MovementConfig(**config_dict.get("movement", {})),
            resource=ResourceConfig(**config_dict.get("resource", {})),
            combat=CombatConfig(**config_dict.get("combat", {})),
            reproduction=ReproductionConfig(**config_dict.get("reproduction", {})),
            perception=PerceptionConfig(**config_dict.get("perception", {})),
        )

    @staticmethod
    def from_legacy_config(legacy_config: Any) -> "AgentConfig":
        """
        Create AgentConfig from legacy configuration object.

        This provides a migration path from the old get_nested_then_flat pattern
        to the new type-safe configuration.

        Args:
            legacy_config: Old-style configuration object with nested attributes

        Returns:
            AgentConfig: New-style configuration

        Note:
            This is a compatibility shim that will be removed after full migration.
        """
        from farm.utils.config_utils import get_nested_then_flat

        def safe_get(nested_attr: str, attr_name: str, default, expected_types):
            """Safely get config value with fallback."""
            try:
                return get_nested_then_flat(
                    config=legacy_config,
                    nested_parent_attr=nested_attr,
                    nested_attr_name=attr_name,
                    flat_attr_name=attr_name,
                    default_value=default,
                    expected_types=expected_types,
                )
            except Exception:
                return default

        # Extract movement config
        movement = MovementConfig(
            max_movement=safe_get(
                "agent_behavior", "max_movement", 8.0, (int, float)
            ),
            position_discretization_method=safe_get(
                "environment", "position_discretization_method", "floor", str
            ),
        )

        # Extract resource config
        resource = ResourceConfig(
            base_consumption_rate=safe_get(
                "agent_behavior", "base_consumption_rate", 1, (int, float)
            ),
            starvation_threshold=safe_get(
                "agent_behavior", "starvation_threshold", 100, (int, float)
            ),
            initial_resources=safe_get(
                "agent_behavior", "offspring_initial_resources", 10, (int, float)
            ),
        )

        # Extract combat config
        combat = CombatConfig(
            starting_health=safe_get(
                "combat", "starting_health", 100.0, (int, float)
            ),
            base_attack_strength=safe_get(
                "agent_behavior", "base_attack_strength", 10.0, (int, float)
            ),
            base_defense_strength=safe_get(
                "agent_behavior", "base_defense_strength", 5.0, (int, float)
            ),
        )

        # Extract reproduction config
        reproduction = ReproductionConfig(
            offspring_cost=safe_get(
                "agent_behavior", "offspring_cost", 5, (int, float)
            ),
            offspring_initial_resources=safe_get(
                "agent_behavior", "offspring_initial_resources", 10, (int, float)
            ),
        )

        # Extract perception config
        perception = PerceptionConfig(
            perception_radius=safe_get(
                "agent_behavior", "perception_radius", 5, (int, float)
            ),
        )

        return AgentConfig(
            movement=movement,
            resource=resource,
            combat=combat,
            reproduction=reproduction,
            perception=perception,
        )