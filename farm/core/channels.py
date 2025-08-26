"""
Dynamic Channel System for Agent Observations

This module provides a dynamic, extensible channel system for agent observations.
It allows users to define custom observation channels without modifying the core
observation code, while maintaining backward compatibility with the existing
Channel enum.

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
    """Defines how a channel behaves during observation updates."""

    INSTANT = 0  # Overwritten each tick with fresh data
    DYNAMIC = 1  # Persists across ticks and decays over time
    PERSISTENT = 2  # Persists indefinitely until explicitly cleared


class ChannelHandler(ABC):
    """Base class for channel-specific processing logic.

    This class defines the interface that custom channel handlers must implement
    to add new observation channels to the system. Each handler is responsible
    for processing world data and writing it to the appropriate channel.
    """

    def __init__(
        self, name: str, behavior: ChannelBehavior, gamma: Optional[float] = None
    ):
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
        """Process world data and write to the observation channel.

        Args:
            observation: The full observation tensor (NUM_CHANNELS, 2R+1, 2R+1)
            channel_idx: Index of this channel in the observation tensor
            config: Observation configuration
            agent_world_pos: Agent's position in world coordinates
            **kwargs: Channel-specific data passed from perceive_world
        """
        pass

    def decay(self, observation: torch.Tensor, channel_idx: int, config=None) -> None:
        """Apply decay to this channel if it's DYNAMIC."""
        if self.behavior == ChannelBehavior.DYNAMIC and self.gamma is not None:
            observation[channel_idx] *= self.gamma

    def clear(self, observation: torch.Tensor, channel_idx: int) -> None:
        """Clear this channel if it's INSTANT."""
        if self.behavior == ChannelBehavior.INSTANT:
            observation[channel_idx].zero_()


class ChannelRegistry:
    """Registry for managing dynamic observation channels.

    This class maintains a mapping of channel names to their handlers and indices,
    allowing for dynamic registration of custom channels while maintaining
    backward compatibility with the original Channel enum.
    """

    def __init__(self):
        self._handlers: Dict[str, ChannelHandler] = {}
        self._name_to_index: Dict[str, int] = {}
        self._index_to_name: Dict[int, str] = {}
        self._next_index = 0

    def register(self, handler: ChannelHandler, index: Optional[int] = None) -> int:
        """Register a channel handler.

        Args:
            handler: The channel handler to register
            index: Optional specific index to assign (for backward compatibility)

        Returns:
            The assigned channel index
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
        """Get a channel handler by name."""
        if name not in self._handlers:
            raise KeyError(f"Channel '{name}' not registered")
        return self._handlers[name]

    def get_index(self, name: str) -> int:
        """Get the index of a channel by name."""
        if name not in self._name_to_index:
            raise KeyError(f"Channel '{name}' not registered")
        return self._name_to_index[name]

    def get_name(self, index: int) -> str:
        """Get the name of a channel by index."""
        if index not in self._index_to_name:
            raise KeyError(f"Channel index {index} not registered")
        return self._index_to_name[index]

    def get_all_handlers(self) -> Dict[str, ChannelHandler]:
        """Get all registered handlers."""
        return self._handlers.copy()

    @property
    def num_channels(self) -> int:
        """Get the total number of registered channels."""
        return len(self._handlers)

    def apply_decay(
        self, observation: torch.Tensor, config: "ObservationConfig"
    ) -> None:
        """Apply decay to all DYNAMIC channels."""
        for name, handler in self._handlers.items():
            channel_idx = self._name_to_index[name]
            handler.decay(observation, channel_idx, config)

    def clear_instant(self, observation: torch.Tensor) -> None:
        """Clear all INSTANT channels."""
        for name, handler in self._handlers.items():
            channel_idx = self._name_to_index[name]
            handler.clear(observation, channel_idx)


# Global registry instance
_global_registry = ChannelRegistry()


def register_channel(handler: ChannelHandler, index: Optional[int] = None) -> int:
    """Register a channel handler with the global registry."""
    return _global_registry.register(handler, index)


def get_channel_registry() -> ChannelRegistry:
    """Get the global channel registry."""
    return _global_registry


# Core Channel Handlers
# These implement the standard observation channels that were previously hardcoded


class SelfHPHandler(ChannelHandler):
    """Handler for agent's own health information."""

    def __init__(self):
        super().__init__("SELF_HP", ChannelBehavior.INSTANT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
        self_hp01 = kwargs.get("self_hp01", 0.0)
        R = config.R
        observation[channel_idx, R, R] = float(self_hp01)


class AlliesHPHandler(ChannelHandler):
    """Handler for visible allies' health information."""

    def __init__(self):
        super().__init__("ALLIES_HP", ChannelBehavior.INSTANT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
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
    """Handler for visible enemies' health information."""

    def __init__(self):
        super().__init__("ENEMIES_HP", ChannelBehavior.INSTANT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
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
    """Base handler for world layer data (resources, obstacles, terrain cost)."""

    def __init__(self, name: str, layer_key: str):
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
        world_layers = kwargs.get("world_layers", {})
        if self.layer_key not in world_layers:
            return

        from farm.core.observations import (
            crop_egocentric,
        )  # Import here to avoid circular import

        R = config.R
        crop = crop_egocentric(
            world_layers[self.layer_key], agent_world_pos, R, pad_val=0.0
        )
        observation[channel_idx].copy_(crop)


class VisibilityHandler(ChannelHandler):
    """Handler for field-of-view visibility mask."""

    def __init__(self):
        super().__init__("VISIBILITY", ChannelBehavior.INSTANT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
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
    """Handler for previously observed empty cells."""

    def __init__(self):
        super().__init__(
            "KNOWN_EMPTY", ChannelBehavior.DYNAMIC, gamma=None
        )  # Use config gamma

    def decay(self, observation: torch.Tensor, channel_idx: int, config=None) -> None:
        """Apply decay using config gamma_known."""
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
        # This is handled specially in update_known_empty - no direct processing needed
        pass


class TransientEventHandler(ChannelHandler):
    """Base handler for transient events (damage, trails, signals)."""

    def __init__(self, name: str, data_key: str, config_gamma_key: str):
        super().__init__(name, ChannelBehavior.DYNAMIC, gamma=None)  # Use config gamma
        self.data_key = data_key
        self.config_gamma_key = config_gamma_key

    def decay(self, observation: torch.Tensor, channel_idx: int, config=None) -> None:
        """Apply decay using appropriate config gamma."""
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
    """Handler for goal/waypoint positions."""

    def __init__(self):
        super().__init__("GOAL", ChannelBehavior.INSTANT)

    def process(
        self,
        observation: torch.Tensor,
        channel_idx: int,
        config: "ObservationConfig",
        agent_world_pos: Tuple[int, int],
        **kwargs,
    ) -> None:
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


# Register core channel handlers with their original indices for backward compatibility
def _register_core_channels():
    """Register all core channel handlers with the global registry."""

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


# Register core channels on import
_register_core_channels()


# Backward compatibility enum
class Channel(IntEnum):
    """Observation channels for the agent's perception system.

    This enum provides backward compatibility while the system transitions
    to the dynamic registry approach.
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


# Dynamic channel count based on registry
NUM_CHANNELS = _global_registry.num_channels
