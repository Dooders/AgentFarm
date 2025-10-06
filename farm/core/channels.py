"""
Dynamic Channel System for Agent Observations

This module provides a dynamic, extensible channel system for agent observations.
It allows users to define custom observation channels without modifying the core
observation code, while maintaining backward compatibility with the existing
Channel enum.

The channel system is designed to handle different types of observation data:
- INSTANT channels: Overwritten each tick with fresh data (e.g., current health, visibility)
- DYNAMIC channels: Persist across ticks and decay over time (e.g., damage trails, known empty cells)
- PERSISTENT channels: Persist indefinitely until explicitly cleared

Key Components:
    - ChannelBehavior: Enum defining how channels behave (instant, dynamic, persistent)
    - ChannelHandler: Abstract base class for channel-specific processing logic
    - ChannelRegistry: Registry for managing dynamic channel registration
    - Core channel handlers for all standard observation types

The system is designed to be backward compatible with the existing Channel enum
while providing a flexible foundation for adding custom observation channels.

Usage:
    # Register a custom channel
    class MyCustomHandler(ChannelHandler):
        def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
            # Custom processing logic here
            pass

    # Register and use
    custom_handler = MyCustomHandler("MY_CUSTOM", ChannelBehavior.DYNAMIC, gamma=0.95)
    register_channel(custom_handler)

    # Use in observations
    agent_obs.perceive_world(..., my_custom_data=custom_data)

Channel Types and Behaviors:
    - SELF_HP (INSTANT): Agent's current health normalized to [0,1]
    - ALLIES_HP (INSTANT): Visible allies' health at their positions
    - ENEMIES_HP (INSTANT): Visible enemies' health at their positions
    - RESOURCES (INSTANT): Resource availability in the world
    - OBSTACLES (INSTANT): Obstacle/passability information
    - TERRAIN_COST (INSTANT): Movement cost for different terrain types
    - VISIBILITY (INSTANT): Field-of-view visibility mask
    - KNOWN_EMPTY (DYNAMIC): Previously observed empty cells with decay
    - DAMAGE_HEAT (DYNAMIC): Recent damage events with decay
    - TRAILS (DYNAMIC): Agent movement trails with decay
    - ALLY_SIGNAL (DYNAMIC): Ally communication signals with decay
    - GOAL (INSTANT): Goal/waypoint positions
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from farm.core.observations import ObservationConfig


class ChannelBehavior(IntEnum):
    """
    Defines how a channel behaves during observation updates.

    This enum controls the temporal behavior of observation channels and determines
    how they are handled during the observation update cycle.

    Attributes:
        INSTANT: Channel data is completely overwritten each tick with fresh data.
                Used for current state information that doesn't need to persist.
                These channels are cleared at the beginning of each observation update.
                Example: Current health, visibility, immediate threats.

        DYNAMIC: Channel data persists across ticks and decays over time using
                a gamma factor. Used for information that should fade gradually.
                These channels have their values multiplied by a decay factor each tick.
                Example: Damage trails, movement trails, known empty cells.

        PERSISTENT: Channel data persists indefinitely until explicitly cleared.
                   Used for information that should remain until manually reset.
                   These channels are never automatically cleared or decayed.
                   Example: Permanent landmarks, long-term memory, learned behaviors.
    """

    INSTANT = 0  # Overwritten each tick with fresh data
    DYNAMIC = 1  # Persists across ticks and decays over time
    PERSISTENT = 2  # Persists indefinitely until explicitly cleared


class ChannelHandler(ABC):
    """
    Base class for channel-specific processing logic.

    This abstract class defines the interface that custom channel handlers must implement
    to add new observation channels to the system. Each handler is responsible for
    processing world data and writing it to the appropriate channel in the observation tensor.

    Channel handlers are the core of the dynamic channel system, allowing users to
    define custom observation channels without modifying the core observation code.
    Each handler encapsulates the logic for one specific type of observation data.

    Subclasses must implement the `process` method to define how their channel
    data is extracted from the world state and written to the observation tensor.

    Attributes:
        name: Unique identifier for this channel (used for registration and lookup)
        behavior: How this channel behaves over time (INSTANT, DYNAMIC, or PERSISTENT)
        gamma: Optional decay rate for DYNAMIC channels (0.0 to 1.0).
              Higher values = slower decay, 1.0 = no decay. Only used for DYNAMIC channels.

    Example:
        >>> class MyCustomHandler(ChannelHandler):
        ...     def __init__(self):
        ...         super().__init__("MY_CUSTOM", ChannelBehavior.DYNAMIC, gamma=0.95)
        ...
        ...     def process(self, observation, channel_idx, config, agent_world_pos, **kwargs):
        ...         # observation is an AgentObservation instance
        ...         custom_data = kwargs.get("my_custom_data", [])
        ...         # Use sparse methods for efficiency: observation._store_sparse_point(...)
        ...         # Or direct tensor access: observation.tensor()[channel_idx] = ...
        ...         pass
    """

    def __init__(self, name: str, behavior: ChannelBehavior, gamma: Optional[float] = None):
        """
        Initialize a channel handler.

        Args:
            name: Unique identifier for this channel (e.g., "SELF_HP", "DAMAGE_HEAT").
                 This name is used for registration, lookup, and debugging.
            behavior: How this channel behaves over time. Must be a ChannelBehavior enum value.
                     Determines whether the channel is INSTANT, DYNAMIC, or PERSISTENT.
            gamma: Optional decay rate for DYNAMIC channels. Should be between 0.0 and 1.0.
                  Higher values mean slower decay (1.0 = no decay). If None, the channel
                  will use config-specific gamma values or default behavior.
        """
        self.name = name
        self.behavior = behavior
        self.gamma = gamma  # Decay rate for DYNAMIC channels

    @abstractmethod
    def process(
        self,
        observation,  # AgentObservation instance (not just tensor)
        channel_idx: int,  # Index of this channel
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,  # Channel-specific data
    ) -> None:
        """
        Process world data and write to the observation channel.

        This abstract method must be implemented by subclasses to define how their
        channel data is extracted from the world state and written to the observation tensor.
        The method is called each tick during the observation update process.

        The observation parameter is an AgentObservation instance that provides access to
        sparse storage methods for memory-efficient channel updates. Implementations should
        use the observation's sparse storage methods (_store_sparse_point, _store_sparse_points,
        _store_sparse_grid) when available, with fallback to direct tensor access.

        Args:
            observation: AgentObservation instance providing sparse storage interface and
                        tensor access. Use observation.tensor()[channel_idx] for direct tensor
                        access or observation._store_sparse_* methods for efficient sparse storage.
            channel_idx: Index of this channel in the observation tensor (0-based).
            config: Observation configuration object containing parameters like R (radius),
                   device, torch_dtype, gamma values, etc.
            agent_world_pos: Agent's current position in world coordinates as (y, x) tuple,
                           where y is the row coordinate and x is the column coordinate.
            **kwargs: Channel-specific data passed from AgentObservation.perceive_world().
                     Common keys include:
                     - self_hp01: Agent's normalized health [0,1]
                     - allies: List of (world_y, world_x, hp) tuples for visible allies
                     - enemies: List of (world_y, world_x, hp) tuples for visible enemies
                     - world_layers: Dict of world layer tensors (resources, obstacles, etc.)
                     - recent_damage_world: List of (world_y, world_x, intensity) damage events
                     - trails_world_points: List of (world_y, world_x, intensity) trail points
                     - ally_signals_world: List of (world_y, world_x, intensity) signal points
                     - goal_world_pos: Goal position (world_y, world_x) if available

        Note:
            Implementations should convert world coordinates to local observation coordinates:
            local_y = world_y - agent_world_pos[0] + config.R
            local_x = world_x - agent_world_pos[1] + config.R

            For memory efficiency, prefer sparse storage methods when available:
            - observation._store_sparse_point(channel_idx, y, x, value) for single points
            - observation._store_sparse_points(channel_idx, points) for multiple points
            - observation._store_sparse_grid(channel_idx, grid) for full grids
        """
        raise NotImplementedError("Subclasses must implement this method")

    def decay(self, observation, channel_idx: int, config=None) -> None:
        """
        Apply temporal decay to this channel if it's DYNAMIC.

        This method is called automatically each tick for channels with DYNAMIC behavior
        to simulate the gradual fading of transient information over time. The decay
        factor is either the handler's own gamma value or a config-specific gamma value.

        For DYNAMIC channels, this method multiplies the channel data by the decay factor,
        causing old information to fade away gradually. The decay is applied uniformly
        across all spatial positions in the channel.

        Args:
            observation: AgentObservation instance providing sparse storage interface.
            channel_idx: Index of this channel in the observation tensor (0-based).
            config: Optional ObservationConfig object that may contain channel-specific
                   gamma values. If provided and the channel has a config-specific gamma,
                   it will be used instead of the handler's gamma.

        Note:
            This method only applies decay if behavior == ChannelBehavior.DYNAMIC.
            For INSTANT and PERSISTENT channels, this method does nothing.
        """
        if self.behavior == ChannelBehavior.DYNAMIC and self.gamma is not None:
            self._safe_decay_sparse_channel(observation, channel_idx, self.gamma)

    def _safe_store_sparse_point(self, observation, channel_idx: int, y: int, x: int, value: float) -> None:
        """
        Safely store a single point using sparse storage with fallback to direct tensor access.

        This utility method centralizes the hasattr pattern for storing single points,
        providing a consistent interface across all channel handlers.

        Args:
            observation: AgentObservation instance providing sparse storage interface.
            channel_idx: Index of this channel in the observation tensor (0-based).
            y: Row coordinate in local observation space.
            x: Column coordinate in local observation space.
            value: Value to store at the specified position.
        """
        if hasattr(observation, "_store_sparse_point"):
            observation._store_sparse_point(channel_idx, y, x, value)
        else:
            # Fallback to direct tensor access (backward compatibility)
            if hasattr(observation, "tensor"):
                observation.tensor()[channel_idx, y, x] = value
            elif isinstance(observation, torch.Tensor):
                observation[channel_idx, y, x] = value

    def _safe_store_sparse_points(self, observation, channel_idx: int, points: list, accumulate: bool = True) -> None:
        """
        Safely store multiple points using sparse storage with fallback to direct tensor access.

        This utility method centralizes the hasattr pattern for storing multiple points,
        providing a consistent interface across all channel handlers.

        Args:
            observation: AgentObservation instance providing sparse storage interface.
            channel_idx: Index of this channel in the observation tensor (0-based).
            points: List of (y, x, value) tuples to store.
            accumulate: If True, accumulate values by taking max with existing values.
                       If False, overwrite existing values.
        """
        if hasattr(observation, "_store_sparse_points"):
            observation._store_sparse_points(channel_idx, points, accumulate=accumulate)
        else:
            # Fallback to direct tensor access (backward compatibility)
            for y, x, value in points:
                if hasattr(observation, "tensor"):
                    tensor = observation.tensor()
                    if accumulate:
                        tensor[channel_idx, y, x] = max(tensor[channel_idx, y, x].item(), value)
                    else:
                        tensor[channel_idx, y, x] = value
                elif isinstance(observation, torch.Tensor):
                    if accumulate:
                        observation[channel_idx, y, x] = max(observation[channel_idx, y, x].item(), value)
                    else:
                        observation[channel_idx, y, x] = value

    def _safe_store_sparse_grid(self, observation, channel_idx: int, grid: torch.Tensor) -> None:
        """
        Safely store a full grid using sparse storage with fallback to direct tensor access.

        This utility method centralizes the hasattr pattern for storing full grids,
        providing a consistent interface across all channel handlers.

        Args:
            observation: AgentObservation instance providing sparse storage interface.
            channel_idx: Index of this channel in the observation tensor (0-based).
            grid: Full grid tensor to store in the channel.
        """
        if hasattr(observation, "_store_sparse_grid"):
            observation._store_sparse_grid(channel_idx, grid)
        else:
            # Fallback to direct tensor access (backward compatibility)
            if hasattr(observation, "tensor"):
                observation.tensor()[channel_idx].copy_(grid)
            elif isinstance(observation, torch.Tensor):
                observation[channel_idx].copy_(grid)

    def _safe_decay_sparse_channel(self, observation, channel_idx: int, gamma: float) -> None:
        """
        Safely apply decay to a channel using sparse-aware decay with fallback.

        This utility method centralizes the hasattr pattern for channel decay,
        providing a consistent interface across all channel handlers.

        Args:
            observation: AgentObservation instance providing sparse storage interface.
            channel_idx: Index of this channel in the observation tensor (0-based).
            gamma: Decay factor to apply (0.0 to 1.0).
        """
        if hasattr(observation, "_decay_sparse_channel"):
            observation._decay_sparse_channel(channel_idx, gamma)
        else:
            # Fallback to direct tensor access (backward compatibility)
            if hasattr(observation, "tensor"):
                observation.tensor()[channel_idx] *= gamma
            elif isinstance(observation, torch.Tensor):
                observation[channel_idx] *= gamma

    def _safe_clear_sparse_channel(self, observation, channel_idx: int) -> None:
        """
        Safely clear a channel using sparse-aware clearing with fallback.

        This utility method centralizes the hasattr pattern for channel clearing,
        providing a consistent interface across all channel handlers.

        Args:
            observation: AgentObservation instance providing sparse storage interface.
            channel_idx: Index of this channel in the observation tensor (0-based).
        """
        if hasattr(observation, "_clear_sparse_channel"):
            observation._clear_sparse_channel(channel_idx)
        else:
            # Fallback to direct tensor access (backward compatibility)
            if hasattr(observation, "tensor"):
                observation.tensor()[channel_idx].zero_()
            elif isinstance(observation, torch.Tensor):
                observation[channel_idx].zero_()

    def _transform_entities_to_observation_coords(
        self, entities: list, agent_world_pos: Tuple[int, int], angle: float, R: int
    ) -> List[Tuple[int, int, float]]:
        """
        Transform world entity coordinates to local observation coordinates.

        This utility method handles the common coordinate transformation logic used
        by multiple channel handlers. It converts world coordinates to local observation
        coordinates, applies optional rotation, and filters valid positions.

        Args:
            entities: List of (world_y, world_x, value) tuples representing entities
            agent_world_pos: Agent's world position (y, x)
            angle: Rotation angle in degrees (0 = no rotation)
            R: Observation radius (used for coordinate offset)

        Returns:
            List of (obs_y, obs_x, value) tuples in observation coordinate space
        """
        use_rotation = angle != 0.0
        if use_rotation:
            a = math.radians(angle)
            cos_a = math.cos(a)
            sin_a = math.sin(a)

        # Convert to local coordinates (rotate by -angle to align facing 'up') and filter valid positions
        points = []
        for world_y, world_x, value in entities:
            dy = float(world_y - agent_world_pos[0])
            dx = float(world_x - agent_world_pos[1])
            if use_rotation:
                dxp = dx * cos_a + dy * sin_a
                dyp = -dx * sin_a + dy * cos_a
            else:
                dxp = dx
                dyp = dy
            y = int(round(R + dyp))
            x = int(round(R + dxp))
            if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
                points.append((y, x, float(value)))

        return points

    def clear(self, observation, channel_idx: int) -> None:
        """
        Clear this channel if it's INSTANT.

        This method is called automatically each tick for channels with INSTANT behavior
        to reset the channel data before writing fresh information. INSTANT channels
        represent current state that should be completely overwritten each observation cycle.

        The method sets all values in the channel to zero, ensuring no stale data
        remains from the previous tick.

        Args:
            observation: AgentObservation instance providing sparse storage interface.
            channel_idx: Index of this channel in the observation tensor (0-based).

        Note:
            This method only clears channels if behavior == ChannelBehavior.INSTANT.
            For DYNAMIC and PERSISTENT channels, this method does nothing to preserve
            their temporal behavior.
        """
        if self.behavior == ChannelBehavior.INSTANT:
            self._safe_clear_sparse_channel(observation, channel_idx)


class ChannelRegistry:
    """
    Registry for managing dynamic observation channels.

    This class serves as the central registry for all observation channels in the system.
    It maintains bidirectional mappings between channel names and their indices, enabling
    dynamic registration of custom channels while preserving backward compatibility.

    The registry is the core of the dynamic channel system, allowing users to register
    custom channel handlers at runtime. It automatically assigns indices to new channels
    and provides efficient lookup in both directions (name → index and index → name).

    Key features:
    - Dynamic channel registration with automatic index assignment
    - Manual index assignment for backward compatibility
    - Efficient bidirectional lookup (name ↔ index)
    - Batch operations for applying decay and clearing across all channels
    - Prevention of duplicate registrations and index conflicts
    - Thread-safe operations (though not currently implemented with locks)

    Attributes:
        _handlers: Internal mapping of channel names to their ChannelHandler objects
        _name_to_index: Internal mapping of channel names to their assigned indices
        _index_to_name: Internal mapping of channel indices to their names
        _next_index: Next available index for automatic assignment during registration
    """

    def __init__(self):
        """Initialize an empty channel registry."""
        self._handlers: Dict[str, ChannelHandler] = {}
        self._name_to_index: Dict[str, int] = {}
        self._index_to_name: Dict[int, str] = {}
        self._next_index = 0

    # --- Test support utilities (public on purpose) ---
    def snapshot_state(self) -> Dict[str, Any]:
        """
        Take a snapshot of the internal registry state for test save/restore.

        Returns:
            Dictionary containing shallow copies of internal mappings and next index.
        """
        return {
            "handlers": self._handlers.copy(),
            "name_to_index": self._name_to_index.copy(),
            "index_to_name": self._index_to_name.copy(),
            "next_index": self._next_index,
        }

    def reset(self) -> None:
        """Reset the registry to an empty state. Intended for tests."""
        self._handlers.clear()
        self._name_to_index.clear()
        self._index_to_name.clear()
        self._next_index = 0

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore the registry from a snapshot captured by snapshot_state().

        Args:
            snapshot: Snapshot dictionary returned by snapshot_state().
        """
        self._handlers = snapshot["handlers"].copy()
        self._name_to_index = snapshot["name_to_index"].copy()
        self._index_to_name = snapshot["index_to_name"].copy()
        self._next_index = snapshot["next_index"]

    def register(self, handler: ChannelHandler, index: Optional[int] = None) -> int:
        """
        Register a channel handler.

        Args:
            handler: The channel handler to register
            index: Optional specific index to assign (for backward compatibility).
                  If None, the next available index will be used automatically.

        Returns:
            The assigned channel index

        Raises:
            ValueError: If the channel name is already registered or if the
                       specified index is already assigned to another channel
        """
        if handler.name in self._handlers:
            raise ValueError(f"Channel '{handler.name}' already registered")

        if index is None:
            index = self._next_index
            self._next_index += 1
        else:
            if index in self._index_to_name:
                raise ValueError(f"Channel index {index} already assigned to '{self._index_to_name[index]}'")
            self._next_index = max(self._next_index, index + 1)

        self._handlers[handler.name] = handler
        self._name_to_index[handler.name] = index
        self._index_to_name[index] = handler.name

        return index

    def get_handler(self, name: str) -> ChannelHandler:
        """
        Get a channel handler by name.

        Args:
            name: The name of the channel

        Returns:
            The channel handler object

        Raises:
            KeyError: If the channel name is not registered
        """
        if name not in self._handlers:
            raise KeyError(f"Channel '{name}' not registered")
        return self._handlers[name]

    def get_index(self, name: str) -> int:
        """
        Get the index of a channel by name.

        Args:
            name: The name of the channel

        Returns:
            The channel index

        Raises:
            KeyError: If the channel name is not registered
        """
        if name not in self._name_to_index:
            raise KeyError(f"Channel '{name}' not registered")
        return self._name_to_index[name]

    def get_name(self, index: int) -> str:
        """
        Get the name of a channel by index.

        Args:
            index: The channel index

        Returns:
            The channel name

        Raises:
            KeyError: If the channel index is not registered
        """
        if index not in self._index_to_name:
            raise KeyError(f"Channel index {index} not registered")
        return self._index_to_name[index]

    def get_all_handlers(self) -> Dict[str, ChannelHandler]:
        """
        Get all registered handlers.

        Returns:
            A copy of the handlers dictionary
        """
        return self._handlers.copy()

    @property
    def num_channels(self) -> int:
        """
        Get the total number of registered channels.

        Returns:
            The number of registered channels
        """
        return len(self._handlers)

    @property
    def max_index(self) -> int:
        """
        Get the highest channel index currently registered.

        Returns:
            The highest channel index, or -1 if no channels are registered
        """
        if not self._index_to_name:
            return -1
        return max(self._index_to_name.keys())

    def apply_decay(self, observation: torch.Tensor, config: "ObservationConfig") -> None:
        """
        Apply decay to all DYNAMIC channels.

        This method is called each tick to apply temporal decay to all DYNAMIC
        channels in the observation tensor.

        Args:
            observation: The full observation tensor
            config: Observation configuration that may contain channel-specific gamma values
        """
        for name, handler in self._handlers.items():
            channel_idx = self._name_to_index[name]
            handler.decay(observation, channel_idx, config)

    def clear_instant(self, observation: torch.Tensor) -> None:
        """
        Clear all INSTANT channels.

        This method is called each tick to clear all INSTANT channels in
        preparation for fresh data.

        Args:
            observation: The full observation tensor
        """
        for name, handler in self._handlers.items():
            channel_idx = self._name_to_index[name]
            handler.clear(observation, channel_idx)


# Global registry instance
_global_registry = ChannelRegistry()


def register_channel(handler: ChannelHandler, index: Optional[int] = None) -> int:
    """
    Register a channel handler with the global registry.

    This is the main entry point for registering custom channels. The handler
    will be added to the global registry and can then be used in observations.

    Args:
        handler: The channel handler to register
        index: Optional specific index to assign (for backward compatibility)

    Returns:
        The assigned channel index

    Example:
        >>> custom_handler = MyCustomHandler("CUSTOM", ChannelBehavior.DYNAMIC, gamma=0.9)
        >>> idx = register_channel(custom_handler)
        >>> print(f"Registered channel at index {idx}")
    """
    return _global_registry.register(handler, index)


def get_channel_registry() -> ChannelRegistry:
    """
    Get the global channel registry.

    Returns:
        The global channel registry instance
    """
    return _global_registry


# Core Channel Handlers
# These implement the standard observation channels that were previously hardcoded


class SelfHPHandler(ChannelHandler):
    """
    Handler for agent's own health information.

    This channel displays the agent's current health normalized to [0,1] at the
    center of the observation (agent's position). Health is treated as an INSTANT
    channel since it represents current state.

    Data source: self_hp01 from kwargs
    Channel behavior: INSTANT (overwritten each tick)
    Position: Center of observation (R, R)
    """

    def __init__(self):
        """Initialize the self HP handler as an INSTANT channel."""
        super().__init__("SELF_HP", ChannelBehavior.INSTANT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process agent's health and write to the center of the observation.

        Uses sparse storage for single-point data to maintain memory efficiency.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of the SELF_HP channel
            config: Observation configuration
            agent_world_pos: Agent's world position (unused for self HP)
            **kwargs: Must contain 'self_hp01' with normalized health [0,1]
        """
        self_hp01 = kwargs.get("self_hp01", 0.0)
        R = config.R

        # Use sparse storage utility method for consistent handling
        self._safe_store_sparse_point(observation, channel_idx, R, R, float(self_hp01))


class AlliesHPHandler(ChannelHandler):
    """
    Handler for visible allies' health information.

    This channel displays the health of visible allies at their relative positions
    in the observation. Health values are normalized to [0,1] and multiple allies
    at the same position will show the maximum health value.

    Data source: allies list from kwargs
    Channel behavior: INSTANT (overwritten each tick)
    Position: Relative to agent position, clipped to observation bounds
    """

    def __init__(self):
        """Initialize the allies HP handler as an INSTANT channel."""
        super().__init__("ALLIES_HP", ChannelBehavior.INSTANT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process allies' health and write to their relative positions.

        Uses sparse storage for point entities to maintain memory efficiency.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of the ALLIES_HP channel
            config: Observation configuration
            agent_world_pos: Agent's world position (y, x)
            **kwargs: Must contain 'allies' list of (y, x, hp) tuples
        """
        allies = kwargs.get("allies", [])
        if not allies:
            return

        R = config.R
        ay, ax = agent_world_pos
        angle = float(kwargs.get("agent_orientation", 0.0)) % 360.0
        use_rotation = angle != 0.0
        if use_rotation:
            a = math.radians(angle)
            cos_a = math.cos(a)
            sin_a = math.sin(a)

        # Convert to local coordinates (rotate by -angle to align facing 'up') and filter valid positions
        points = self._transform_entities_to_observation_coords(allies, agent_world_pos, angle, R)

        # Use sparse storage utility method for consistent handling (no accumulation)
        # This maintains encapsulation by avoiding direct access to observation.sparse_channels
        # and instead uses the public sparse storage interface
        self._safe_store_sparse_points(observation, channel_idx, points, accumulate=False)


class EnemiesHPHandler(ChannelHandler):
    """
    Handler for visible enemies' health information.

    This channel displays the health of visible enemies at their relative positions
    in the observation. Health values are normalized to [0,1] and multiple enemies
    at the same position will show the maximum health value.

    Data source: enemies list from kwargs
    Channel behavior: INSTANT (overwritten each tick)
    Position: Relative to agent position, clipped to observation bounds
    """

    def __init__(self):
        """Initialize the enemies HP handler as an INSTANT channel."""
        super().__init__("ENEMIES_HP", ChannelBehavior.INSTANT)

    def process(
        self,
        observation,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process enemies' health and write to their relative positions.

        Uses sparse storage for point entities to maintain memory efficiency.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of the ENEMIES_HP channel
            config: Observation configuration
            agent_world_pos: Agent's world position (y, x)
            **kwargs: Must contain 'enemies' list of (y, x, hp) tuples
        """
        enemies = kwargs.get("enemies", [])
        if not enemies:
            return

        R = config.R
        ay, ax = agent_world_pos
        angle = float(kwargs.get("agent_orientation", 0.0)) % 360.0
        use_rotation = angle != 0.0
        if use_rotation:
            a = math.radians(angle)
            cos_a = math.cos(a)
            sin_a = math.sin(a)

        # Convert to local coordinates (rotate by -angle to align facing 'up') and filter valid positions
        points = self._transform_entities_to_observation_coords(enemies, agent_world_pos, angle, R)

        # Use sparse storage utility method for consistent handling (no accumulation)
        # This maintains encapsulation by avoiding direct access to observation.sparse_channels
        # and instead uses the public sparse storage interface
        self._safe_store_sparse_points(observation, channel_idx, points, accumulate=False)


class WorldLayerHandler(ChannelHandler):
    """
    Base handler for world layer data (resources, obstacles, terrain cost).

    This handler processes world layer data by cropping it to the agent's
    local view and copying it directly to the observation channel.
    World layers represent static environmental information.

    Data source: world_layers dict from kwargs
    Channel behavior: INSTANT (overwritten each tick)
    Position: Full observation area (cropped from world layer)
    """

    def __init__(self, name: str, layer_key: str):
        """
        Initialize a world layer handler.

        Args:
            name: Channel name (e.g., "RESOURCES", "OBSTACLES")
            layer_key: Key to access the layer data in world_layers dict
        """
        super().__init__(name, ChannelBehavior.INSTANT)
        self.layer_key = layer_key

    def process(
        self,
        observation,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process world layer data and copy to observation channel.

        Uses sparse storage interface - stores full grids for dense channels.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of this world layer channel
            config: Observation configuration
            agent_world_pos: Agent's world position (y, x)
            **kwargs: Must contain 'world_layers' dict with layer data
        """
        world_layers = kwargs.get("world_layers", {})
        if self.layer_key not in world_layers:
            return

        from farm.core.observations import (  # Import here to avoid circular import
            crop_local,
            crop_local_rotated,
            rotate_local_grid,
        )

        # Accept either a full world grid (H, W) or a pre-cropped local grid (S, S)
        layer = world_layers[self.layer_key]

        # Convert numpy arrays to torch on the configured device/dtype
        # Optimize: check if already torch tensor with correct device/dtype
        if isinstance(layer, torch.Tensor):
            if layer.device != config.device or layer.dtype != config.torch_dtype:
                layer = layer.to(device=config.device, dtype=config.torch_dtype)
        else:
            layer = torch.as_tensor(layer, device=config.device, dtype=config.torch_dtype)

        # If already local sized (2R+1, 2R+1), use directly; otherwise crop from world
        R = config.R
        expected_size = config.get_local_observation_size()
        angle = float(kwargs.get("agent_orientation", 0.0)) % 360.0
        if tuple(layer.shape[-2:]) == expected_size:
            if angle != 0.0:
                final_layer = rotate_local_grid(layer, angle, pad_val=0.0)
            else:
                final_layer = layer
        else:
            if angle != 0.0:
                final_layer = crop_local_rotated(layer, agent_world_pos, R, orientation=angle, pad_val=0.0)
            else:
                final_layer = crop_local(layer, agent_world_pos, R, pad_val=0.0)

        # Use sparse storage utility method for consistent handling
        self._safe_store_sparse_grid(observation, channel_idx, final_layer)


class VisibilityHandler(ChannelHandler):
    """
    Handler for field-of-view visibility mask.

    This channel creates a circular visibility mask based on the agent's
    field-of-view radius. The mask is 1.0 for visible cells and 0.0 for
    cells outside the field of view.

    Data source: Generated from config.fov_radius
    Channel behavior: INSTANT (overwritten each tick)
    Position: Full observation area (circular mask)
    """

    def __init__(self):
        """Initialize the visibility handler as an INSTANT channel."""
        super().__init__("VISIBILITY", ChannelBehavior.INSTANT)

    def process(
        self,
        observation,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Create visibility mask and write to observation channel.

        Uses sparse storage interface - stores full visibility mask.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of the VISIBILITY channel
            config: Observation configuration (uses fov_radius, device, torch_dtype)
            agent_world_pos: Agent's world position (unused for visibility)
            **kwargs: Not used for visibility
        """
        from farm.core.observations import (
            make_disk_mask,
        )  # Import here to avoid circular import

        S = 2 * config.R + 1
        vis = make_disk_mask(
            S,
            min(config.fov_radius, config.R),
            device=config.device,
            dtype=config.torch_dtype,
        )

        # Use sparse storage utility method for consistent handling
        self._safe_store_sparse_grid(observation, channel_idx, vis)


class KnownEmptyHandler(ChannelHandler):
    """
    Handler for previously observed empty cells.

    This channel tracks cells that have been observed as empty in previous ticks.
    The information decays over time using config.gamma_known, allowing agents
    to remember recently observed empty spaces.

    Data source: Updated externally via update_known_empty
    Channel behavior: DYNAMIC (decays over time)
    Position: Full observation area
    """

    def __init__(self):
        """Initialize the known empty handler as a DYNAMIC channel."""
        super().__init__("KNOWN_EMPTY", ChannelBehavior.DYNAMIC, gamma=None)  # Use config gamma

    def decay(self, observation, channel_idx: int, config=None) -> None:
        """
        Apply decay using config gamma_known.

        Uses sparse-aware decay to maintain memory efficiency.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of the KNOWN_EMPTY channel
            config: Observation configuration (uses gamma_known)
        """
        gamma = None
        if config is not None and hasattr(config, "gamma_known"):
            gamma = config.gamma_known
        elif self.gamma is not None:
            gamma = self.gamma

        if gamma is not None:
            self._safe_decay_sparse_channel(observation, channel_idx, gamma)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        No-op process method for externally managed channel.

        This channel does not perform direct processing during the standard observation
        update cycle. Instead, it relies entirely on external updates via the
        update_known_empty function and temporal decay via the decay method using
        config.gamma_known.
        """
        # This is handled specially in update_known_empty - no direct processing needed
        return


class TransientEventHandler(ChannelHandler):
    """
    Generic handler for transient events that decay over time.

    This handler processes transient events that occur at specific world positions
    and decay over time. Events are represented as intensity values that fade
    according to a configurable gamma factor. The specific event data source is
    configured via the data_key parameter.

    Data source: Event list from kwargs using the configured data_key
    Channel behavior: DYNAMIC (decays over time)
    Position: Event positions relative to agent, clipped to observation bounds
    """

    def __init__(self, name: str, data_key: str, config_gamma_key: str):
        """
        Initialize a transient event handler.

        Args:
            name: Channel name (e.g., "DAMAGE_HEAT", "TRAILS")
            data_key: Key to access event data in kwargs (e.g., "recent_damage_world",
                     "trails_world_points", "ally_signals_world")
            config_gamma_key: Key to access gamma value in config (e.g., "gamma_dmg",
                           "gamma_trail", "gamma_sig")
        """
        super().__init__(name, ChannelBehavior.DYNAMIC, gamma=None)  # Use config gamma
        self.data_key = data_key
        self.config_gamma_key = config_gamma_key

    def decay(self, observation, channel_idx: int, config=None) -> None:
        """
        Apply decay using appropriate config gamma.

        Uses sparse-aware decay to maintain memory efficiency.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of this transient event channel
            config: Observation configuration (uses config-specific gamma)
        """
        gamma = None
        if config is not None and hasattr(config, self.config_gamma_key):
            gamma = getattr(config, self.config_gamma_key)
        elif self.gamma is not None:
            gamma = self.gamma

        if gamma is not None:
            self._safe_decay_sparse_channel(observation, channel_idx, gamma)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process transient events and write to their relative positions.

        Uses sparse storage for event points to maintain memory efficiency.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of this transient event channel
            config: Observation configuration
            agent_world_pos: Agent's world position (y, x)
            **kwargs: Must contain event data at self.data_key as list of (y, x, intensity) tuples
                     where self.data_key is configured during initialization
        """
        events = kwargs.get(self.data_key, [])
        if not events:
            return

        R = config.R
        ay, ax = agent_world_pos
        angle = float(kwargs.get("agent_orientation", 0.0)) % 360.0
        use_rotation = angle != 0.0
        if use_rotation:
            a = math.radians(angle)
            cos_a = math.cos(a)
            sin_a = math.sin(a)

        # Convert to local coordinates (rotate by -angle to align facing 'up') and filter valid positions
        points = self._transform_entities_to_observation_coords(events, agent_world_pos, angle, R)

        # Use sparse storage utility method for consistent handling (no accumulation)
        # This maintains encapsulation by avoiding direct access to observation.sparse_channels
        # and instead uses the public sparse storage interface
        self._safe_store_sparse_points(observation, channel_idx, points, accumulate=False)


class GoalHandler(ChannelHandler):
    """
    Handler for goal/waypoint positions.

    This channel displays goal positions as a single point (value 1.0) at the
    goal's relative position in the observation. If no goal is set, the channel
    remains empty.

    Data source: goal_world_pos from kwargs
    Channel behavior: INSTANT (overwritten each tick)
    Position: Goal position relative to agent, clipped to observation bounds
    """

    def __init__(self):
        """Initialize the goal handler as an INSTANT channel."""
        super().__init__("GOAL", ChannelBehavior.INSTANT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process goal position and write to its relative position.

        Uses sparse storage for single goal point to maintain memory efficiency.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of the GOAL channel
            config: Observation configuration
            agent_world_pos: Agent's world position (y, x)
            **kwargs: May contain 'goal_world_pos' as (y, x) tuple
        """
        goal_world_pos = kwargs.get("goal_world_pos")
        if goal_world_pos is None:
            return

        R = config.R
        ay, ax = agent_world_pos
        gy, gx = goal_world_pos
        angle = float(kwargs.get("agent_orientation", 0.0)) % 360.0
        use_rotation = angle != 0.0
        if use_rotation:
            a = math.radians(angle)
            cos_a = math.cos(a)
            sin_a = math.sin(a)

        dy = float(gy - ay)
        dx = float(gx - ax)
        if use_rotation:
            dxp = dx * cos_a + dy * sin_a
            dyp = -dx * sin_a + dy * cos_a
        else:
            dxp = dx
            dyp = dy
        y = int(round(R + dyp))
        x = int(round(R + dxp))

        if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
            # Use sparse storage utility method for consistent handling
            self._safe_store_sparse_point(observation, channel_idx, y, x, 1.0)


class LandmarkHandler(ChannelHandler):
    """
    Handler for permanent landmarks or waypoints.

    This channel tracks important permanent locations in the environment that
    agents should remember indefinitely. Landmarks persist until explicitly
    cleared and can represent strategic locations, resource caches, or
    important waypoints.

    Data source: landmarks_world list passed via kwargs in perceive_world()
    Channel behavior: PERSISTENT (remains until explicitly cleared)
    Position: Landmark positions relative to agent, clipped to observation bounds
    """

    def __init__(self):
        """Initialize the landmark handler as a PERSISTENT channel."""
        super().__init__("LANDMARKS", ChannelBehavior.PERSISTENT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process landmark positions and accumulate them in the observation.

        Unlike INSTANT channels, PERSISTENT channels accumulate information
        across ticks. This handler adds landmark information without clearing
        previous landmark data.

        Uses sparse storage to maintain memory efficiency for landmark accumulation.

        Args:
            observation: AgentObservation instance
            channel_idx: Index of the LANDMARKS channel
            config: Observation configuration
            agent_world_pos: Agent's world position (y, x)
            **kwargs: Must contain 'landmarks_world' list of (y, x, importance) tuples
        """
        landmarks_world = kwargs.get("landmarks_world", [])
        if not landmarks_world:
            return

        R = config.R
        ay, ax = agent_world_pos
        angle = float(kwargs.get("agent_orientation", 0.0)) % 360.0
        use_rotation = angle != 0.0
        if use_rotation:
            a = math.radians(angle)
            cos_a = math.cos(a)
            sin_a = math.sin(a)

        # Convert to local coordinates (rotate by -angle to align facing 'up') and filter valid positions
        points = self._transform_entities_to_observation_coords(landmarks_world, agent_world_pos, angle, R)

        # Use sparse storage utility method for consistent handling (with accumulation)
        # This maintains encapsulation by avoiding direct access to observation.sparse_channels
        # and instead uses the public sparse storage interface
        self._safe_store_sparse_points(observation, channel_idx, points, accumulate=True)


# Register core channel handlers with their original indices for backward compatibility
def _register_core_channels():
    """
    Register all core channel handlers with the global registry.

    This function registers all the standard observation channels with their
    original indices to maintain backward compatibility with the existing
    Channel enum. The registration order and indices match the original
    hardcoded channel system.

    Registered channels:
        0: SELF_HP - Agent's own health
        1: ALLIES_HP - Visible allies' health
        2: ENEMIES_HP - Visible enemies' health
        3: RESOURCES - Resource availability
        4: OBSTACLES - Obstacle/passability information
        5: TERRAIN_COST - Movement cost for terrain
        6: VISIBILITY - Field-of-view mask
        7: KNOWN_EMPTY - Previously observed empty cells
        8: DAMAGE_HEAT - Recent damage events
        9: TRAILS - Agent movement trails
        10: ALLY_SIGNAL - Ally communication signals
        11: GOAL - Goal/waypoint positions
        12: LANDMARKS - Permanent landmarks and waypoints
    """

    # Register with specific indices to maintain backward compatibility
    register_channel(SelfHPHandler(), 0)
    register_channel(AlliesHPHandler(), 1)
    register_channel(EnemiesHPHandler(), 2)
    register_channel(WorldLayerHandler("RESOURCES", "RESOURCES"), 3)
    register_channel(WorldLayerHandler("OBSTACLES", "OBSTACLES"), 4)
    register_channel(WorldLayerHandler("TERRAIN_COST", "TERRAIN_COST"), 5)
    register_channel(VisibilityHandler(), 6)
    register_channel(KnownEmptyHandler(), 7)
    register_channel(TransientEventHandler("DAMAGE_HEAT", "recent_damage_world", "gamma_dmg"), 8)
    register_channel(TransientEventHandler("TRAILS", "trails_world_points", "gamma_trail"), 9)
    register_channel(TransientEventHandler("ALLY_SIGNAL", "ally_signals_world", "gamma_sig"), 10)
    register_channel(GoalHandler(), 11)
    register_channel(LandmarkHandler(), 12)


# Register core channels on import
_register_core_channels()


# Backward compatibility enum
class Channel(IntEnum):
    """
    Observation channels for the agent's perception system.

    This enum provides backward compatibility while the system transitions
    to the dynamic registry approach. The indices match those used in the
    original hardcoded channel system.

    Channel Descriptions:
        SELF_HP: Agent's current health normalized to [0,1]
        ALLIES_HP: Visible allies' health at their positions
        ENEMIES_HP: Visible enemies' health at their positions
        RESOURCES: Resource availability in the world
        OBSTACLES: Obstacle/passability information (binary or passability)
        TERRAIN_COST: Movement cost for different terrain types (0..1)
        VISIBILITY: Field-of-view visibility mask (1=visible now)
        KNOWN_EMPTY: Previously observed empty cells (decays over time)
        DAMAGE_HEAT: Recent damage events (decays over time)
        TRAILS: Agent movement trails (decays over time)
        ALLY_SIGNAL: Ally communication signals (decays over time)
        GOAL: Goal/waypoint positions (waypoint projection)
        LANDMARKS: Permanent landmarks and waypoints (persistent)
    """

    SELF_HP = 0  # 1
    ALLIES_HP = 1  # 2
    ENEMIES_HP = 2  # 3
    RESOURCES = 3  # 4
    OBSTACLES = 4  # 5 (binary or passability)
    TERRAIN_COST = 5  # 6 (0..1)
    VISIBILITY = 6  # 7 (1=visible now)
    KNOWN_EMPTY = 7  # 8 (decays)
    DAMAGE_HEAT = 8  # 9 (decays)
    TRAILS = 9  # 10 (decays)
    ALLY_SIGNAL = 10  # 11 (decays)
    GOAL = 11  # 12 (waypoint projection)
    LANDMARKS = 12  # 13 (persistent)


# Dynamic channel count based on registry
NUM_CHANNELS = _global_registry.num_channels
