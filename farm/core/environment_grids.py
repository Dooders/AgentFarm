"""World-sized environmental grid layers backed by memory-mapped storage.

This module provides :class:`EnvironmentalGridManager`, a small helper that
holds the dense environmental observation layers (``OBSTACLES``,
``TERRAIN_COST``, ``VISIBILITY``) for an entire simulation world.

Each layer is shaped ``(height, width)``. When memmap support is enabled the
underlying buffers live on disk through :class:`MemmapManager`; otherwise they
are plain in-RAM ``numpy.ndarray`` objects with the same interface. This
keeps caller code identical regardless of the backing strategy.

Typical usage::

    grids = EnvironmentalGridManager(
        height=cfg.environment.height,
        width=cfg.environment.width,
        memmap_config=cfg.memmap,
        simulation_id=env.simulation_id,
    )
    grids.set("OBSTACLES", obstacles_world_grid)
    win = grids.get_window("OBSTACLES", y0, y1, x0, x1)

Agents perceive their local environment via :meth:`get_window`; the manager
takes care of bounds checking and zero-padding outside the world.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from farm.core.memmap_manager import MemmapManager, Shape2D
from farm.utils.logging import get_logger

logger = get_logger(__name__)


#: Names of the environmental layers the manager supports out of the box.
ENVIRONMENTAL_LAYER_NAMES: Tuple[str, ...] = ("OBSTACLES", "TERRAIN_COST", "VISIBILITY")


class EnvironmentalGridManager:
    """Hold dense world-sized environmental layers (with optional memmap backing).

    Parameters
    ----------
    height, width:
        World dimensions in grid units.
    memmap_config:
        Optional ``MemmapConfig``-like object. When ``use_for_environmental``
        is truthy, layers are backed by memmap files via
        :class:`MemmapManager`; otherwise they are kept in plain numpy
        arrays (the public API is identical either way).
    simulation_id:
        Optional id passed through to :class:`MemmapManager` so backing
        files do not collide between simulations on the same host.
    layer_names:
        Names of the layers to provision up-front. Defaults to
        ``ENVIRONMENTAL_LAYER_NAMES``.
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        memmap_config: Any = None,
        simulation_id: Optional[str] = None,
        layer_names: Iterable[str] = ENVIRONMENTAL_LAYER_NAMES,
    ) -> None:
        self.shape: Shape2D = (int(height), int(width))
        self.layer_names: Tuple[str, ...] = tuple(layer_names)

        self._use_memmap: bool = bool(
            getattr(memmap_config, "use_for_environmental", False)
        )
        self._dtype = np.dtype(
            getattr(memmap_config, "dtype", "float32") if memmap_config else "float32"
        )

        self._manager: Optional[MemmapManager] = None
        self._arrays: Dict[str, np.ndarray] = {}

        if self._use_memmap:
            try:
                self._manager = MemmapManager(
                    directory=getattr(memmap_config, "directory", None),
                    simulation_id=simulation_id,
                    namespace="env",
                    default_dtype=self._dtype,
                    default_mode=getattr(memmap_config, "mode", "w+"),
                )
                for name in self.layer_names:
                    arr = self._manager.create(name, self.shape, fill=0.0)
                    self._arrays[name] = arr
                logger.info(
                    "environmental_grids_initialized",
                    backend="memmap",
                    layers=self.layer_names,
                    shape=self.shape,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "environmental_grids_memmap_init_failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                # Fall back to in-RAM arrays so simulations keep running.
                self._manager = None
                self._use_memmap = False

        if not self._use_memmap:
            # RAM mode is lazy by default: we only allocate a world-sized layer
            # when it is first written/read as an array. Pure window reads from
            # untouched layers can be served as all-zeros without allocating
            # the backing grid.
            logger.debug(
                "environmental_grids_initialized",
                backend="ram",
                layers=self.layer_names,
                shape=self.shape,
                allocation="lazy",
            )

    # ------------------------------------------------------------------
    # Layer access
    # ------------------------------------------------------------------

    @property
    def has_memmap(self) -> bool:
        """``True`` when the underlying layers are memmap-backed."""

        return bool(self._use_memmap and self._manager is not None)

    def __contains__(self, name: object) -> bool:  # pragma: no cover - trivial
        return isinstance(name, str) and name in self.layer_names

    def _ensure_layer(self, name: str) -> np.ndarray:
        if name not in self.layer_names:
            raise KeyError(
                f"Environmental layer '{name}' not registered (known: {sorted(self.layer_names)})"
            )
        arr = self._arrays.get(name)
        if arr is None:
            arr = np.zeros(self.shape, dtype=self._dtype)
            self._arrays[name] = arr
        return arr

    def get(self, name: str) -> np.ndarray:
        """Return the underlying array (memmap or ndarray) for ``name``."""

        if self.has_memmap:
            try:
                return self._arrays[name]
            except KeyError:
                raise KeyError(
                    f"Environmental layer '{name}' not registered (known: {sorted(self._arrays)})"
                ) from None
        return self._ensure_layer(name)

    def names(self) -> Tuple[str, ...]:
        """Names of the registered environmental layers."""

        return tuple(self.layer_names)

    def set(self, name: str, data: np.ndarray) -> None:
        """Replace the contents of layer ``name`` with ``data``."""

        arr = self.get(name)
        if data.shape != arr.shape:
            raise ValueError(
                f"Cannot set layer '{name}': expected shape {arr.shape}, got {data.shape}"
            )
        arr[:] = data
        if self.has_memmap:
            self._manager.flush(name)

    def fill(self, name: str, value: float) -> None:
        """Fill layer ``name`` with the scalar ``value``."""

        arr = self.get(name)
        arr[:] = value
        if self.has_memmap:
            self._manager.flush(name)

    def get_window(
        self,
        name: str,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        out_dtype: Any = np.float32,
    ) -> np.ndarray:
        """Return a zero-padded ``(y1-y0, x1-x0)`` window for ``name``."""

        if self.has_memmap:
            return self._manager.get_window(
                name, y0, y1, x0, x1, out_dtype=out_dtype
            )
        if name not in self.layer_names:
            raise KeyError(
                f"Environmental layer '{name}' not registered (known: {sorted(self.layer_names)})"
            )
        # Plain-array path mirrors MemmapManager.get_window behavior, with
        # the same fast path for windows fully inside the array.
        arr = self._arrays.get(name)
        H, W = self.shape
        y0i = int(y0)
        y1i = int(y1)
        x0i = int(x0)
        x1i = int(x1)
        h = y1i - y0i
        w = x1i - x0i
        target_dtype = np.dtype(out_dtype)
        if h <= 0 or w <= 0:
            return np.zeros((max(0, h), max(0, w)), dtype=target_dtype)
        if arr is None:
            # Untouched RAM-backed layers are implicitly all-zeros.
            return np.zeros((h, w), dtype=target_dtype)

        if 0 <= y0i and y1i <= H and 0 <= x0i and x1i <= W:
            return np.array(arr[y0i:y1i, x0i:x1i], dtype=target_dtype, copy=True)

        out = np.zeros((h, w), dtype=target_dtype)
        ys0 = max(0, y0i)
        ys1 = min(H, y1i)
        xs0 = max(0, x0i)
        xs1 = min(W, x1i)
        if ys1 <= ys0 or xs1 <= xs0:
            return out
        ty0 = ys0 - y0i
        tx0 = xs0 - x0i
        view = arr[ys0:ys1, xs0:xs1]
        if view.dtype == out.dtype:
            out[ty0 : ty0 + (ys1 - ys0), tx0 : tx0 + (xs1 - xs0)] = view
        else:
            out[ty0 : ty0 + (ys1 - ys0), tx0 : tx0 + (xs1 - xs0)] = view.astype(
                out.dtype, copy=False
            )
        return out

    def total_size_bytes(self) -> int:
        """Total disk-backed bytes when using memmap, in-RAM bytes otherwise."""

        if self.has_memmap:
            return self._manager.total_size_bytes()
        # Report only allocated RAM layers (lazy allocation).
        return sum(int(arr.nbytes) for arr in self._arrays.values())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush all memmap layers (no-op for in-RAM mode)."""

        if self.has_memmap:
            self._manager.flush()

    def close(self, *, delete_files: bool = False) -> None:
        """Release backing resources (and optionally delete memmap files)."""

        if self._manager is not None:
            self._manager.close_all(delete_files=delete_files)
            self._manager = None
        self._arrays.clear()
