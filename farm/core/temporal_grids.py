"""World-sized temporal channel grids backed by memory-mapped storage.

Temporal channels (``DAMAGE_HEAT``, ``TRAILS``, ``ALLY_SIGNAL``) carry
information that fades over time according to a per-channel decay factor
(``gamma``). Historically they were stored only as sparse per-agent
observation buffers, which made it cheap when few events occurred but
required every agent to repeatedly project the same world events into its
own coordinate frame.

:class:`TemporalGridManager` provides an alternative shared representation:
each channel is a single dense ``(height, width)`` grid of intensities.
Events are deposited once into the shared grid; agents read their
egocentric windows from it; decay is applied to the grid (in place) once per
simulation tick. When memmap-backing is enabled the grids live on disk so
even very large worlds avoid loading the full state into RAM.

Per-channel decay factors come from the simulation's
``ObservationConfig`` so the temporal-grid view stays consistent with the
sparse channel handlers (``gamma_dmg``, ``gamma_trail``, ``gamma_sig``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from farm.core.memmap_manager import MemmapManager, Shape2D
from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TemporalChannelSpec:
    """Description of one temporal channel grid.

    Attributes
    ----------
    name:
        Channel name as registered in the channel registry (uppercase).
    storage_name:
        Lowercase token used to derive the on-disk filename.
    config_gamma_key:
        Attribute on ``ObservationConfig`` that supplies the decay factor.
    default_gamma:
        Fallback decay factor used when the config does not provide one.
    """

    name: str
    storage_name: str
    config_gamma_key: str
    default_gamma: float


#: Built-in temporal channels that mirror the sparse handlers in
#: :mod:`farm.core.channels`.
DEFAULT_TEMPORAL_CHANNEL_SPECS: Tuple[TemporalChannelSpec, ...] = (
    TemporalChannelSpec("DAMAGE_HEAT", "damage_heat", "gamma_dmg", 0.85),
    TemporalChannelSpec("TRAILS", "trails", "gamma_trail", 0.90),
    TemporalChannelSpec("ALLY_SIGNAL", "ally_signal", "gamma_sig", 0.92),
)


class TemporalGridManager:
    """Manage world-sized temporal channel grids with optional memmap backing.

    Parameters
    ----------
    height, width:
        World dimensions in grid units.
    memmap_config:
        ``MemmapConfig``-like object. When ``use_for_temporal`` is truthy
        each channel is stored in a memmap file via :class:`MemmapManager`.
    simulation_id:
        Optional identifier propagated into memmap filenames.
    channel_specs:
        Iterable of :class:`TemporalChannelSpec` describing the channels to
        manage. Defaults to :data:`DEFAULT_TEMPORAL_CHANNEL_SPECS`.
    observation_config:
        Optional ``ObservationConfig`` used to look up per-channel gamma
        values. When ``None`` the spec defaults are used.
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        memmap_config: Any = None,
        simulation_id: Optional[str] = None,
        channel_specs: Iterable[TemporalChannelSpec] = DEFAULT_TEMPORAL_CHANNEL_SPECS,
        observation_config: Any = None,
    ) -> None:
        self.shape: Shape2D = (int(height), int(width))
        self._specs: Dict[str, TemporalChannelSpec] = {
            spec.name: spec for spec in channel_specs
        }
        self._observation_config = observation_config

        self._use_memmap: bool = bool(
            getattr(memmap_config, "use_for_temporal", False)
        )
        self._dtype = np.dtype(
            getattr(memmap_config, "dtype", "float32") if memmap_config else "float32"
        )

        self._manager: Optional[MemmapManager] = None
        self._arrays: Dict[str, np.ndarray] = {}

        # Per-channel "has any non-zero data" hint. Set to True on deposit,
        # left True until decay drops everything below a small epsilon.
        # Lets ``Environment._get_observation`` short-circuit when the
        # entire world grid for a channel is known empty, avoiding both
        # the window slice and the dense channel write.
        self._has_data: Dict[str, bool] = {name: False for name in self._specs}

        if self._use_memmap:
            try:
                self._manager = MemmapManager(
                    directory=getattr(memmap_config, "directory", None),
                    simulation_id=simulation_id,
                    namespace="temporal",
                    default_dtype=self._dtype,
                    default_mode=getattr(memmap_config, "mode", "w+"),
                )
                for spec in self._specs.values():
                    arr = self._manager.create(
                        spec.storage_name, self.shape, fill=0.0
                    )
                    self._arrays[spec.name] = arr
                logger.info(
                    "temporal_grids_initialized",
                    backend="memmap",
                    channels=tuple(self._specs.keys()),
                    shape=self.shape,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "temporal_grids_memmap_init_failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                self._manager = None
                self._use_memmap = False

        if not self._use_memmap:
            # RAM mode allocates channel grids lazily. Channels with no events
            # stay allocation-free and are treated as all-zeros on read.
            logger.debug(
                "temporal_grids_initialized",
                backend="ram",
                channels=tuple(self._specs.keys()),
                shape=self.shape,
                allocation="lazy",
            )

    # ------------------------------------------------------------------
    # Layer access
    # ------------------------------------------------------------------

    @property
    def has_memmap(self) -> bool:
        """``True`` when channel grids are memmap-backed."""

        return bool(self._use_memmap and self._manager is not None)

    def channel_names(self) -> Tuple[str, ...]:
        """Names of the registered temporal channels."""

        return tuple(self._specs.keys())

    def __contains__(self, name: object) -> bool:  # pragma: no cover - trivial
        return isinstance(name, str) and name in self._specs

    def _ensure_channel(self, name: str) -> np.ndarray:
        if name not in self._specs:
            raise KeyError(
                f"Temporal channel '{name}' not registered (known: {sorted(self._specs)})"
            )
        arr = self._arrays.get(name)
        if arr is None:
            arr = np.zeros(self.shape, dtype=self._dtype)
            self._arrays[name] = arr
        return arr

    def get(self, name: str) -> np.ndarray:
        """Return the underlying array for channel ``name``."""

        if self.has_memmap:
            try:
                return self._arrays[name]
            except KeyError:
                raise KeyError(
                    f"Temporal channel '{name}' not registered (known: {sorted(self._arrays)})"
                ) from None
        return self._ensure_channel(name)

    # ------------------------------------------------------------------
    # Event deposit / decay
    # ------------------------------------------------------------------

    def _resolve_storage_name(self, name: str) -> str:
        try:
            return self._specs[name].storage_name
        except KeyError:
            raise KeyError(
                f"Temporal channel '{name}' not registered"
            ) from None

    def deposit(
        self,
        name: str,
        events: Iterable[Tuple[int, int, float]],
        *,
        accumulate: bool = True,
        clip_max: Optional[float] = 1.0,
    ) -> None:
        """Insert ``(y, x, intensity)`` events into channel ``name``.

        Out-of-bounds events are ignored. When ``accumulate`` is true the
        deposited intensity is added to the existing value; otherwise the
        existing value is replaced. Values are clamped to ``clip_max`` when
        provided.
        """

        arr = self.get(name)
        H, W = self.shape
        any_change = False
        for y, x, intensity in events:
            yi, xi = int(y), int(x)
            if not (0 <= yi < H and 0 <= xi < W):
                continue
            if accumulate:
                new_val = float(arr[yi, xi]) + float(intensity)
            else:
                new_val = float(intensity)
            if clip_max is not None and new_val > clip_max:
                new_val = float(clip_max)
            arr[yi, xi] = new_val
            any_change = True
        if any_change:
            self._has_data[name] = True
            if self.has_memmap:
                self._manager.flush(self._resolve_storage_name(name))

    def has_any_data(self, name: str) -> bool:
        """Return ``True`` when channel ``name`` may contain non-zero data.

        Set conservatively: ``True`` after any successful :meth:`deposit`,
        cleared by :meth:`apply_decay` only when decay shrinks the maximum
        value below the floor at which it would round to zero in the
        backing dtype. Callers can use this to short-circuit window
        extraction when the entire world grid for the channel is empty.
        """

        return self._has_data.get(name, False)

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
        """Return a zero-padded window for channel ``name``."""

        if self.has_memmap:
            return self._manager.get_window(
                self._resolve_storage_name(name),
                y0,
                y1,
                x0,
                x1,
                out_dtype=out_dtype,
            )
        if name not in self._specs:
            raise KeyError(
                f"Temporal channel '{name}' not registered (known: {sorted(self._specs)})"
            )
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
            # Undeposited RAM-backed channels are implicitly all-zeros.
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

    def gamma_for(self, name: str) -> float:
        """Return the decay gamma for channel ``name``.

        Reads from the attached observation config when available, otherwise
        falls back to the spec default.
        """

        spec = self._specs[name]
        if self._observation_config is not None and hasattr(
            self._observation_config, spec.config_gamma_key
        ):
            value = getattr(self._observation_config, spec.config_gamma_key)
            if value is not None:
                return float(value)
        return float(spec.default_gamma)

    def apply_decay(self, name: Optional[str] = None) -> None:
        """Multiply one (or all) channel grids by their gamma factor in place.

        Decay is applied directly to the backing array, so memmap-backed
        grids stream pages through memory rather than loading the full grid
        at once. After decay, the grid is flushed to disk.
        """

        if name is None:
            for channel_name in self._specs:
                self.apply_decay(channel_name)
            return
        spec = self._specs[name]
        gamma = self.gamma_for(name)
        if gamma is None or gamma == 1.0:
            return
        if not self._has_data.get(name, False):
            # Nothing was ever deposited. Skip the in-place multiply.
            # The flag is sticky once set; checking ``arr.max()`` on a
            # large memmap to flip it back would defeat the purpose.
            return
        arr = self._arrays.get(name)
        if arr is None:
            return
        np.multiply(arr, gamma, out=arr)
        if self.has_memmap:
            self._manager.flush(spec.storage_name)

    def total_size_bytes(self) -> int:
        """Total disk-backed bytes when memmap is enabled, in-RAM otherwise."""

        if self.has_memmap:
            return self._manager.total_size_bytes()
        return sum(int(arr.nbytes) for arr in self._arrays.values())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush all memmap channels (no-op for in-RAM mode)."""

        if self.has_memmap:
            self._manager.flush()

    def close(self, *, delete_files: bool = False) -> None:
        """Release backing resources (and optionally delete memmap files)."""

        if self._manager is not None:
            self._manager.close_all(delete_files=delete_files)
            self._manager = None
        self._arrays.clear()
        self._has_data = {name: False for name in self._specs}

    def clear_all(self) -> None:
        """Reset all temporal channels to zero and clear activity flags.

        In RAM mode this only touches channels that were actually allocated.
        In memmap mode all registered channels are dense-backed and are
        zeroed explicitly.
        """

        for name, arr in self._arrays.items():
            arr[:] = 0
            if self.has_memmap:
                self._manager.flush(self._resolve_storage_name(name))
        self._has_data = {name: False for name in self._specs}
