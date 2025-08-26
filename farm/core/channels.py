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

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from farm.core.observations import ObservationConfig


class ChannelBehavior(IntEnum):
    """
    Defines how a channel behaves during observation updates.

    This enum controls the temporal behavior of observation channels:

    - INSTANT: Channel data is completely overwritten each tick with fresh data.
              Used for current state information that doesn't need to persist.
              Example: Current health, visibility, immediate threats.

    - DYNAMIC: Channel data persists across ticks and decays over time using
              a gamma factor. Used for information that should fade gradually.
              Example: Damage trails, movement trails, known empty cells.

    - PERSISTENT: Channel data persists indefinitely until explicitly cleared.
                 Used for information that should remain until manually reset.
                 Example: Permanent landmarks, long-term memory.
    """

    INSTANT = 0  # Overwritten each tick with fresh data
    DYNAMIC = 1  # Persists across ticks and decays over time
    PERSISTENT = 2  # Persists indefinitely until explicitly cleared


class ChannelHandler(ABC):
    """
    Base class for channel-specific processing logic.

    This class defines the interface that custom channel handlers must implement
    to add new observation channels to the system. Each handler is responsible
    for processing world data and writing it to the appropriate channel.

    Handlers receive the full observation tensor and are responsible for:
    1. Processing incoming data from the world
    2. Writing processed data to the correct channel index
    3. Handling channel-specific behavior (decay, clearing, etc.)

    Attributes:
        name (str): Unique identifier for this channel
        behavior (ChannelBehavior): How this channel behaves over time
        gamma (Optional[float]): Decay rate for DYNAMIC channels (0.0 to 1.0)
                                Higher values = slower decay, 1.0 = no decay
    """

    def __init__(
        self, name: str, behavior: ChannelBehavior, gamma: Optional[float] = None
    ):
        """
        Initialize a channel handler.

        Args:
            name: Unique identifier for this channel (e.g., "SELF_HP", "DAMAGE_HEAT")
            behavior: How this channel behaves over time (INSTANT, DYNAMIC, PERSISTENT)
            gamma: Decay rate for DYNAMIC channels. Should be between 0.0 and 1.0.
                  Higher values mean slower decay. If None, will use config gamma.
        """
        self.name = name
        self.behavior = behavior
        self.gamma = gamma  # Decay rate for DYNAMIC channels

    @abstractmethod
    def process(
        self,
        observation: torch.Tensor,  # Full observation tensor
        channel_idx: int,  # Index of this channel
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,  # Channel-specific data
    ) -> None:
        """
        Process world data and write to the observation channel.

        This method is called each tick to update the channel with fresh data.
        The handler should process the incoming data and write it to the
        observation tensor at the specified channel index.

        Args:
            observation: The full observation tensor with shape (NUM_CHANNELS, 2R+1, 2R+1)
                        where R is the observation radius. This tensor is local
                        (centered on the agent's position).
            channel_idx: Index of this channel in the observation tensor
            config: Observation configuration containing parameters like R (radius),
                   device, torch_dtype, etc.
            agent_world_pos: Agent's current position in world coordinates (y, x)
            **kwargs: Channel-specific data passed from perceive_world. Common keys:
                     - self_hp01: Agent's normalized health [0,1]
                     - allies: List of (y, x, hp) tuples for visible allies
                     - enemies: List of (y, x, hp) tuples for visible enemies
                     - world_layers: Dict of world layer data (resources, obstacles, etc.)
                     - recent_damage_world: List of (y, x, intensity) damage events
                     - trails_world_points: List of (y, x, intensity) trail points
                     - ally_signals_world: List of (y, x, intensity) signal points
                     - goal_world_pos: Goal position (y, x) if available
        """
        pass

    def decay(self, observation: torch.Tensor, channel_idx: int, config=None) -> None:
        """
        Apply decay to this channel if it's DYNAMIC.

        This method is called each tick for DYNAMIC channels to apply temporal decay.
        The decay factor is either the handler's gamma or a config-specific gamma.

        Args:
            observation: The full observation tensor
            channel_idx: Index of this channel
            config: Optional config object that may contain channel-specific gamma values
        """
        if self.behavior == ChannelBehavior.DYNAMIC and self.gamma is not None:
            observation[channel_idx] *= self.gamma

    def clear(self, observation: torch.Tensor, channel_idx: int) -> None:
        """
        Clear this channel if it's INSTANT.

        This method is called each tick for INSTANT channels to prepare for fresh data.

        Args:
            observation: The full observation tensor
            channel_idx: Index of this channel
        """
        if self.behavior == ChannelBehavior.INSTANT:
            observation[channel_idx].zero_()


class ChannelRegistry:
    """
    Registry for managing dynamic observation channels.

    This class maintains a mapping of channel names to their handlers and indices,
    allowing for dynamic registration of custom channels while maintaining
    backward compatibility with the original Channel enum.

    The registry provides:
    - Dynamic channel registration with automatic or manual index assignment
    - Lookup by name or index
    - Batch operations for decay and clearing
    - Backward compatibility with existing Channel enum indices

    Attributes:
        _handlers: Mapping of channel names to their handler objects
        _name_to_index: Mapping of channel names to their indices
        _index_to_name: Mapping of channel indices to their names
        _next_index: Next available index for automatic assignment
    """

    def __init__(self):
        """Initialize an empty channel registry."""
        self._handlers: Dict[str, ChannelHandler] = {}
        self._name_to_index: Dict[str, int] = {}
        self._index_to_name: Dict[int, str] = {}
        self._next_index = 0

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
                raise ValueError(
                    f"Channel index {index} already assigned to '{self._index_to_name[index]}'"
                )
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

    def apply_decay(
        self, observation: torch.Tensor, config: "ObservationConfig"
    ) -> None:
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

        Args:
            observation: Full observation tensor
            channel_idx: Index of the SELF_HP channel
            config: Observation configuration
            agent_world_pos: Agent's world position (unused for self HP)
            **kwargs: Must contain 'self_hp01' with normalized health [0,1]
        """
        self_hp01 = kwargs.get("self_hp01", 0.0)
        R = config.R
        observation[channel_idx, R, R] = float(self_hp01)


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

        Args:
            observation: Full observation tensor
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

        for ally_y, ally_x, ally_hp in allies:
            dy = ally_y - ay
            dx = ally_x - ax
            y = R + dy
            x = R + dx
            if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
                observation[channel_idx, y, x] = max(
                    observation[channel_idx, y, x].item(), float(ally_hp)
                )


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
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process enemies' health and write to their relative positions.

        Args:
            observation: Full observation tensor
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

        for enemy_y, enemy_x, enemy_hp in enemies:
            dy = enemy_y - ay
            dx = enemy_x - ax
            y = R + dy
            x = R + dx
            if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
                observation[channel_idx, y, x] = max(
                    observation[channel_idx, y, x].item(), float(enemy_hp)
                )


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
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Process world layer data and copy to observation channel.

        Args:
            observation: Full observation tensor
            channel_idx: Index of this world layer channel
            config: Observation configuration
            agent_world_pos: Agent's world position (y, x)
            **kwargs: Must contain 'world_layers' dict with layer data
        """
        world_layers = kwargs.get("world_layers", {})
        if self.layer_key not in world_layers:
            return

        from farm.core.observations import (
            crop_local,
        )  # Import here to avoid circular import

        R = config.R
        crop = crop_local(world_layers[self.layer_key], agent_world_pos, R, pad_val=0.0)
        observation[channel_idx].copy_(crop)


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
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        Create visibility mask and write to observation channel.

        Args:
            observation: Full observation tensor
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
        observation[channel_idx] = vis


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
        super().__init__(
            "KNOWN_EMPTY", ChannelBehavior.DYNAMIC, gamma=None
        )  # Use config gamma

    def decay(self, observation: torch.Tensor, channel_idx: int, config=None) -> None:
        """
        Apply decay using config gamma_known.

        Args:
            observation: Full observation tensor
            channel_idx: Index of the KNOWN_EMPTY channel
            config: Observation configuration (uses gamma_known)
        """
        if config is not None and hasattr(config, "gamma_known"):
            observation[channel_idx] *= config.gamma_known
        elif self.gamma is not None:
            observation[channel_idx] *= self.gamma

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        """
        No direct processing - handled externally.

        This channel is updated externally via the update_known_empty function
        rather than through the standard process method.
        """
        # This is handled specially in update_known_empty - no direct processing needed
        pass


class TransientEventHandler(ChannelHandler):
    """
    Base handler for transient events (damage, trails, signals).

    This handler processes transient events that occur at specific world positions
    and decay over time. Events are represented as intensity values that fade
    according to a configurable gamma factor.

    Data source: Event lists from kwargs (recent_damage_world, trails_world_points, etc.)
    Channel behavior: DYNAMIC (decays over time)
    Position: Event positions relative to agent, clipped to observation bounds
    """

    def __init__(self, name: str, data_key: str, config_gamma_key: str):
        """
        Initialize a transient event handler.

        Args:
            name: Channel name (e.g., "DAMAGE_HEAT", "TRAILS")
            data_key: Key to access event data in kwargs
            config_gamma_key: Key to access gamma value in config
        """
        super().__init__(name, ChannelBehavior.DYNAMIC, gamma=None)  # Use config gamma
        self.data_key = data_key
        self.config_gamma_key = config_gamma_key

    def decay(self, observation: torch.Tensor, channel_idx: int, config=None) -> None:
        """
        Apply decay using appropriate config gamma.

        Args:
            observation: Full observation tensor
            channel_idx: Index of this transient event channel
            config: Observation configuration (uses config-specific gamma)
        """
        if config is not None and hasattr(config, self.config_gamma_key):
            gamma = getattr(config, self.config_gamma_key)
            observation[channel_idx] *= gamma
        elif self.gamma is not None:
            observation[channel_idx] *= self.gamma

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

        Args:
            observation: Full observation tensor
            channel_idx: Index of this transient event channel
            config: Observation configuration
            agent_world_pos: Agent's world position (y, x)
            **kwargs: Must contain event data at self.data_key as list of (y, x, intensity) tuples
        """
        events = kwargs.get(self.data_key, [])
        if not events:
            return

        R = config.R
        ay, ax = agent_world_pos

        for event_y, event_x, intensity in events:
            dy = event_y - ay
            dx = event_x - ax
            y = R + dy
            x = R + dx
            if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
                observation[channel_idx, y, x] = max(
                    observation[channel_idx, y, x].item(), float(intensity)
                )


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

        Args:
            observation: Full observation tensor
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
        dy = gy - ay
        dx = gx - ax
        y = R + dy
        x = R + dx
        if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
            observation[channel_idx, y, x] = 1.0


class LandmarkHandler(ChannelHandler):
    """
    Handler for permanent landmarks or waypoints.

    This channel tracks important permanent locations in the environment that
    agents should remember indefinitely. Landmarks persist until explicitly
    cleared and can represent strategic locations, resource caches, or
    important waypoints.

    Data source: landmarks_world list from kwargs
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

        Args:
            observation: Full observation tensor
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

        for landmark_y, landmark_x, importance in landmarks_world:
            dy = landmark_y - ay
            dx = landmark_x - ax
            y = R + dy
            x = R + dx
            if 0 <= y < 2 * R + 1 and 0 <= x < 2 * R + 1:
                # Accumulate landmark importance (don't overwrite existing values)
                current_value = observation[channel_idx, y, x].item()
                observation[channel_idx, y, x] = max(current_value, float(importance))


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
    register_channel(
        TransientEventHandler("DAMAGE_HEAT", "recent_damage_world", "gamma_dmg"), 8
    )
    register_channel(
        TransientEventHandler("TRAILS", "trails_world_points", "gamma_trail"), 9
    )
    register_channel(
        TransientEventHandler("ALLY_SIGNAL", "ally_signals_world", "gamma_sig"), 10
    )
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
