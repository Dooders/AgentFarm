"""
Compatibility layer for migrating from BaseAgent to AgentCore.

This module provides adapters and utilities to ease the transition from the
monolithic BaseAgent to the new component-based AgentCore system.
"""

import warnings
from typing import Optional, TYPE_CHECKING
from farm.core.agent.core import AgentCore
from farm.core.agent.factory import AgentFactory
from farm.core.agent.config.agent_config import AgentConfig
from farm.core.agent.behaviors import DefaultAgentBehavior

if TYPE_CHECKING:
    from farm.core.services.interfaces import ISpatialQueryService

try:
    from farm.utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BaseAgentAdapter:
    """
    Adapter providing BaseAgent-compatible API using AgentCore internally.

    This class allows existing code to use the new AgentCore system without
    modification. It implements the most commonly used BaseAgent methods and
    properties, delegating to the underlying AgentCore and components.

    Usage:
        >>> # Old code using BaseAgent
        >>> agent = BaseAgent(...)
        >>> agent.position
        >>> agent.resource_level
        >>> agent.act()

        >>> # New code using adapter (drop-in replacement)
        >>> agent = BaseAgentAdapter.from_old_style(...)
        >>> agent.position  # Works!
        >>> agent.resource_level  # Works!
        >>> agent.act()  # Works!

    Migration Strategy:
        1. Replace BaseAgent with BaseAgentAdapter
        2. Verify behavior is unchanged
        3. Gradually refactor to use AgentCore directly
        4. Remove adapter when migration complete

    Note:
        This adapter is a temporary migration aid. New code should use
        AgentCore directly for better type safety and clarity.
    """

    def __init__(self, agent_core: AgentCore):
        """
        Initialize adapter wrapping an AgentCore.

        Args:
            agent_core: The underlying AgentCore instance
        """
        self._core = agent_core

        # Issue deprecation warning
        warnings.warn(
            "BaseAgentAdapter is deprecated and will be removed in a future version. "
            "Please migrate to using AgentCore directly.",
            DeprecationWarning,
            stacklevel=2
        )

    # ===== Identity Properties =====

    @property
    def agent_id(self) -> str:
        """Agent unique identifier."""
        return self._core.agent_id

    @property
    def alive(self) -> bool:
        """Whether agent is alive."""
        return self._core.alive

    @property
    def agent_type(self) -> str:
        """Agent type identifier."""
        # Try to get from state manager, fallback to class name
        return getattr(self._core, 'agent_type', self._core.__class__.__name__)

    # ===== Position Properties =====

    @property
    def position(self) -> tuple[float, float]:
        """Current 2D position."""
        return self._core.position

    @position.setter
    def position(self, value: tuple[float, float]) -> None:
        """Set position (for backward compatibility)."""
        self._core.state_manager.set_position(value)

    @property
    def orientation(self) -> float:
        """Current orientation in degrees."""
        return self._core.state_manager.orientation

    @orientation.setter
    def orientation(self, value: float) -> None:
        """Set orientation."""
        self._core.state_manager.set_orientation(value)

    # ===== Resource Properties =====

    @property
    def resource_level(self) -> int:
        """Current resource level."""
        resource = self._core.get_component("resource")
        return resource.level if resource else 0

    @resource_level.setter
    def resource_level(self, value: int) -> None:
        """Set resource level (for backward compatibility)."""
        resource = self._core.get_component("resource")
        if resource:
            resource.set_level(value)

    # ===== Health Properties =====

    @property
    def current_health(self) -> float:
        """Current health points."""
        combat = self._core.get_component("combat")
        return combat.health if combat else 0.0

    @current_health.setter
    def current_health(self, value: float) -> None:
        """Set health (for backward compatibility)."""
        combat = self._core.get_component("combat")
        if combat:
            combat.set_health(value)

    @property
    def starting_health(self) -> float:
        """Maximum health points."""
        combat = self._core.get_component("combat")
        return combat.max_health if combat else 0.0

    @property
    def is_defending(self) -> bool:
        """Whether agent is in defensive stance."""
        combat = self._core.get_component("combat")
        return combat.is_defending if combat else False

    # ===== Lifecycle Properties =====

    @property
    def birth_time(self) -> int:
        """Simulation step when agent was born."""
        return self._core.state_manager.birth_time

    @property
    def generation(self) -> int:
        """Generation number."""
        return self._core.state_manager.generation

    @property
    def genome_id(self) -> str:
        """Genome identifier."""
        return self._core.state_manager.genome_id

    # ===== Action Methods =====

    def act(self) -> None:
        """Execute one simulation turn."""
        self._core.act()

    def terminate(self) -> None:
        """Handle agent death."""
        self._core.terminate()

    def update_position(self, new_position: tuple[float, float]) -> None:
        """Update agent position (legacy method)."""
        self._core.state_manager.set_position(new_position)

    # ===== Combat Methods =====

    def handle_combat(self, attacker, damage: float) -> float:
        """Handle incoming attack (legacy method)."""
        combat = self._core.get_component("combat")
        if combat:
            return combat.take_damage(damage)
        return 0.0

    def take_damage(self, damage: float) -> bool:
        """Apply damage to agent (legacy method)."""
        combat = self._core.get_component("combat")
        if combat:
            combat.take_damage(damage)
            return True
        return False

    @property
    def attack_strength(self) -> float:
        """Calculate attack strength (legacy property)."""
        combat = self._core.get_component("combat")
        if combat:
            return combat._calculate_attack_damage()
        return 0.0

    @property
    def defense_strength(self) -> float:
        """Calculate defense strength (legacy property)."""
        combat = self._core.get_component("combat")
        if combat:
            return combat.get_defense_strength()
        return 0.0

    # ===== State Management =====

    def get_state(self):
        """Get agent state (legacy method)."""
        return self._core.get_state_dict()

    # ===== Access to Core =====

    @property
    def core(self) -> AgentCore:
        """
        Get underlying AgentCore for gradual migration.

        This allows code to access new functionality while still using
        the adapter for compatibility.

        Example:
            >>> agent = BaseAgentAdapter(...)
            >>> # Use old API
            >>> agent.position
            >>> # Access new features
            >>> movement = agent.core.get_component("movement")
            >>> movement.move_to((100, 100))
        """
        return self._core

    # ===== Factory Methods =====

    @classmethod
    def from_old_style(
        cls,
        agent_id: str,
        position: tuple[float, float],
        resource_level: int,
        spatial_service: "ISpatialQueryService",
        environment: Optional[object] = None,
        **kwargs
    ) -> "BaseAgentAdapter":
        """
        Create adapter from old-style BaseAgent parameters.

        This factory method accepts the same parameters as the old BaseAgent
        constructor, making it a drop-in replacement.

        Args:
            agent_id: Unique identifier
            position: Starting position
            resource_level: Initial resources
            spatial_service: Spatial query service
            environment: Optional environment reference
            **kwargs: Other parameters (mostly ignored for compatibility)

        Returns:
            BaseAgentAdapter: Adapter wrapping new AgentCore

        Example:
            >>> # Old code
            >>> agent = BaseAgent(
            ...     agent_id="agent_001",
            ...     position=(10, 20),
            ...     resource_level=100,
            ...     spatial_service=spatial_service
            ... )

            >>> # New code (drop-in replacement)
            >>> agent = BaseAgentAdapter.from_old_style(
            ...     agent_id="agent_001",
            ...     position=(10, 20),
            ...     resource_level=100,
            ...     spatial_service=spatial_service
            ... )
        """
        # Extract services from kwargs
        time_service = kwargs.get('time_service')
        lifecycle_service = kwargs.get('lifecycle_service')
        config_obj = kwargs.get('config')

        # Convert to new config if needed
        if config_obj:
            try:
                config = AgentConfig.from_legacy_config(config_obj)
            except Exception:
                config = AgentConfig()
        else:
            config = AgentConfig()

        # Create factory
        factory = AgentFactory(
            spatial_service=spatial_service,
            default_config=config,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
        )

        # Create agent
        agent_core = factory.create_default_agent(
            agent_id=agent_id,
            position=position,
            initial_resources=resource_level,
            config=config,
        )

        # Wrap in adapter
        return cls(agent_core)

    def __repr__(self) -> str:
        """String representation."""
        return f"BaseAgentAdapter(core={repr(self._core)})"


def migrate_to_core(adapter: BaseAgentAdapter) -> AgentCore:
    """
    Extract AgentCore from adapter.

    Use this to gradually migrate code from adapter to direct AgentCore usage.

    Args:
        adapter: BaseAgentAdapter instance

    Returns:
        AgentCore: Underlying core

    Example:
        >>> # Stage 1: Use adapter
        >>> agent = BaseAgentAdapter.from_old_style(...)
        >>> agent.act()

        >>> # Stage 2: Migrate to core
        >>> core = migrate_to_core(agent)
        >>> movement = core.get_component("movement")
        >>> movement.move_to((100, 100))
    """
    return adapter.core


def is_new_agent(agent: object) -> bool:
    """
    Check if agent uses new AgentCore system.

    Args:
        agent: Agent instance to check

    Returns:
        bool: True if using new system

    Example:
        >>> if is_new_agent(agent):
        ...     # Use new API
        ...     agent.get_component("movement")
        ... else:
        ...     # Use old API
        ...     agent.position
    """
    return isinstance(agent, (AgentCore, BaseAgentAdapter))


def get_core(agent: object) -> Optional[AgentCore]:
    """
    Get AgentCore from any agent type.

    Args:
        agent: Agent instance (BaseAgent, BaseAgentAdapter, or AgentCore)

    Returns:
        AgentCore or None

    Example:
        >>> core = get_core(agent)
        >>> if core:
        ...     movement = core.get_component("movement")
    """
    if isinstance(agent, AgentCore):
        return agent
    elif isinstance(agent, BaseAgentAdapter):
        return agent.core
    return None