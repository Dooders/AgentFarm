from __future__ import annotations

from typing import List, Optional, Tuple, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    # Forward-type references to avoid import cycles at runtime
    from farm.core.spatial_index import SpatialIndex
    from farm.core.resources import Resource


@runtime_checkable
class SpatialService(Protocol):
    """Abstraction for spatial queries and world bounds.

    Agents and other consumers depend on this interface instead of the
    concrete Environment, enabling dependency injection and easier testing.
    Implementations in production should delegate to the SpatialIndex for
    efficient O(log n) queries via KD-trees.
    """

    def get_nearby_agents(self, position: Tuple[float, float], radius: float) -> List[object]:
        """Return agents within ``radius`` of ``position``.

        Implementations may include the querying agent if present at ``position``.
        Returned agents should be limited to currently-alive instances.
        """
        ...

    def get_nearby_resources(self, position: Tuple[float, float], radius: float) -> List["Resource"]:
        """Return resources within ``radius`` of ``position``."""
        ...

    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional["Resource"]:
        """Return the nearest resource to ``position`` or ``None`` if none exist."""
        ...

    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Return True if ``position`` lies within inclusive world bounds."""
        ...

    def get_dimensions(self) -> Tuple[int, int]:
        """Return world dimensions as ``(width, height)``."""
        ...

    def mark_positions_dirty(self) -> None:
        """Signal that agent positions changed (e.g., after movement)."""
        ...


class SpatialIndexAdapter(SpatialService):
    """Production adapter that delegates queries to a SpatialIndex.

    Construct this in the Environment and inject into agents to decouple
    agent logic from Environment while reusing the KD-tree index.
    """

    def __init__(self, spatial_index: "SpatialIndex", width: int, height: int) -> None:
        self._index = spatial_index
        self._width = width
        self._height = height

    def get_nearby_agents(self, position: Tuple[float, float], radius: float) -> List[object]:
        return self._index.get_nearby_agents(position, radius)

    def get_nearby_resources(self, position: Tuple[float, float], radius: float) -> List["Resource"]:
        return self._index.get_nearby_resources(position, radius)

    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional["Resource"]:
        # Optional helper, available on SpatialIndex
        if hasattr(self._index, "get_nearest_resource"):
            return self._index.get_nearest_resource(position)  # type: ignore[no-any-return]
        return None

    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        x, y = position
        return (0 <= x <= self._width) and (0 <= y <= self._height)

    def get_dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    def mark_positions_dirty(self) -> None:
        # Delegate to SpatialIndex so it can rebuild KD-trees on next update
        self._index.mark_positions_dirty()