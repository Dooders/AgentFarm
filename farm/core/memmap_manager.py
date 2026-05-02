"""Generic memory-mapped array manager.

This module provides ``MemmapManager``, a small reusable component for
creating, accessing, and tearing down disk-backed ``numpy.memmap`` arrays.

It is designed to be the single shared mechanism behind every memmap-backed
state structure in AgentFarm (resource grids, environmental grids such as
OBSTACLES/TERRAIN_COST/VISIBILITY, and temporal channel grids such as
DAMAGE_HEAT/TRAILS/ALLY_SIGNAL).

Key design properties:

- **Single Responsibility**: only handles memmap lifecycle (create / lookup /
  flush / cleanup) and windowed reads with zero-padding.
- **Multiprocess safety**: filenames embed the OS process id and an optional
  ``simulation_id`` so concurrent simulations on the same host do not collide.
- **Cross-platform**: relies only on ``numpy.memmap`` and the standard library
  for filesystem operations; the same code path works on Linux, macOS, and
  Windows.
- **No global state**: every ``MemmapManager`` owns its own files and indexes
  them by name within a single instance.

The manager is intentionally minimal – higher-level grid managers
(``EnvironmentalGridManager``, ``TemporalGridManager``, etc.) compose it and
add domain semantics on top.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from farm.utils.logging import get_logger

logger = get_logger(__name__)


# Public type aliases -------------------------------------------------------

#: A 2D shape ``(height, width)``.
Shape2D = Tuple[int, int]


def get_zero_padded_window_2d(
    arr: Optional[np.ndarray],
    shape_hw: Shape2D,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    out_dtype: Any = np.float32,
) -> np.ndarray:
    """Extract ``arr[y0:y1, x0:x1]`` with zero-padding outside ``shape_hw``.

    Shared by :class:`MemmapManager` and RAM-backed grid managers so bounds
    handling and dtype coercion stay consistent across backends.
    """

    H, W = int(shape_hw[0]), int(shape_hw[1])
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
    if view.dtype == target_dtype:
        out[ty0 : ty0 + (ys1 - ys0), tx0 : tx0 + (xs1 - xs0)] = view
    else:
        out[ty0 : ty0 + (ys1 - ys0), tx0 : tx0 + (xs1 - xs0)] = view.astype(
            target_dtype, copy=False
        )
    return out


@dataclass(frozen=True)
class MemmapArrayInfo:
    """Metadata describing a single memmap-backed array."""

    name: str
    path: str
    shape: Shape2D
    dtype: np.dtype
    mode: str

    @property
    def size_bytes(self) -> int:
        """Total on-disk size of the array in bytes."""

        return int(np.prod(self.shape)) * int(self.dtype.itemsize)


def sanitize_for_filename(value: str, max_length: int = 64) -> str:
    """Replace filesystem-unsafe characters and clip to ``max_length``."""

    cleaned = "".join(
        c if (c.isalnum() or c in ("-", "_")) else "-" for c in str(value)
    )
    return cleaned[:max_length]


class MemmapManager:
    """Manage a named collection of disk-backed ``numpy.memmap`` arrays.

    Parameters
    ----------
    directory:
        Directory in which to create ``.dat`` files. Defaults to the system
        temporary directory.
    simulation_id:
        Optional identifier embedded in filenames so multiple simulations on
        the same host do not collide.
    namespace:
        Optional namespace prefix applied to every filename (in addition to
        ``simulation_id``). Useful when multiple managers share the same
        directory (e.g. one for resources and one for environmental grids).
    default_dtype:
        Default numpy dtype used when ``create`` is invoked without one.
    default_mode:
        Default memmap mode (``"w+"`` truncates, ``"r+"`` opens existing).

    Notes
    -----
    Filenames follow the pattern::

        [<namespace>_]<name>[_<simulation_id>]_p<pid>_<H>x<W>.dat

    Each component is sanitized to keep cross-platform filesystem
    compatibility. A ``MemmapManager`` instance owns the files it creates and
    will optionally remove them on :meth:`close_all`.
    """

    def __init__(
        self,
        directory: Optional[str] = None,
        *,
        simulation_id: Optional[str] = None,
        namespace: Optional[str] = None,
        default_dtype: Union[str, np.dtype] = "float32",
        default_mode: str = "w+",
    ) -> None:
        self.directory: str = directory or tempfile.gettempdir()
        os.makedirs(self.directory, exist_ok=True)
        self.simulation_id: Optional[str] = simulation_id
        self.namespace: Optional[str] = namespace
        self.default_dtype: np.dtype = np.dtype(default_dtype)
        self.default_mode: str = default_mode

        self._arrays: Dict[str, np.memmap] = {}
        # Plain ``ndarray`` views of each memmap, built once via
        # ``np.asarray``. Slicing through the ``np.memmap`` subclass adds
        # measurable per-call overhead; the plain view exposes the same
        # buffer with the regular ndarray dispatch path. We still keep the
        # memmap object in ``_arrays`` for ``flush()`` and lifecycle.
        self._views: Dict[str, np.ndarray] = {}
        self._infos: Dict[str, MemmapArrayInfo] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_filename(self, name: str, shape: Shape2D) -> str:
        parts: List[str] = []
        if self.namespace:
            parts.append(sanitize_for_filename(self.namespace))
        parts.append(sanitize_for_filename(name))
        if self.simulation_id:
            parts.append(sanitize_for_filename(self.simulation_id))
        parts.append(f"p{os.getpid()}")
        h, w = int(shape[0]), int(shape[1])
        parts.append(f"{h}x{w}")
        return "_".join(parts) + ".dat"

    def create(
        self,
        name: str,
        shape: Shape2D,
        *,
        dtype: Optional[Union[str, np.dtype]] = None,
        mode: Optional[str] = None,
        fill: Optional[float] = 0.0,
    ) -> np.memmap:
        """Create (or reopen) a memmap array under ``name``.

        Parameters
        ----------
        name:
            Logical name of the array (e.g. ``"obstacles"``).
        shape:
            ``(height, width)`` tuple describing the array.
        dtype:
            Numpy dtype (defaults to the manager's ``default_dtype``).
        mode:
            ``numpy.memmap`` mode. Defaults to ``default_mode``. Use ``"r+"``
            to reuse an existing file without truncating.
        fill:
            Value to broadcast into the new array after creation. Pass
            ``None`` to skip initialization (useful when reopening).

        Returns
        -------
        numpy.memmap
            The newly created (or reopened) memmap array.
        """

        if name in self._arrays:
            raise ValueError(f"Memmap array '{name}' already exists in this manager")

        resolved_dtype = np.dtype(dtype) if dtype is not None else self.default_dtype
        resolved_mode = mode or self.default_mode
        shape = (int(shape[0]), int(shape[1]))

        filename = self._build_filename(name, shape)
        path = os.path.join(self.directory, filename)

        try:
            arr = np.memmap(path, dtype=resolved_dtype, mode=resolved_mode, shape=shape)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "memmap_create_failed",
                name=name,
                path=path,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

        if fill is not None and resolved_mode in ("w+",):
            arr[:] = fill
            arr.flush()

        info = MemmapArrayInfo(
            name=name,
            path=path,
            shape=shape,
            dtype=resolved_dtype,
            mode=resolved_mode,
        )
        self._arrays[name] = arr
        self._views[name] = np.asarray(arr)
        self._infos[name] = info
        logger.debug(
            "memmap_array_created",
            name=name,
            path=path,
            shape=shape,
            dtype=str(resolved_dtype),
            mode=resolved_mode,
        )
        return arr

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        """Return ``True`` when an array named ``name`` is registered."""

        return name in self._arrays

    def __contains__(self, name: object) -> bool:  # pragma: no cover - trivial
        return isinstance(name, str) and name in self._arrays

    def get(self, name: str) -> np.memmap:
        """Return the memmap array registered under ``name``."""

        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError(f"Memmap array '{name}' is not registered") from None

    def info(self, name: str) -> MemmapArrayInfo:
        """Return :class:`MemmapArrayInfo` for ``name``."""

        try:
            return self._infos[name]
        except KeyError:
            raise KeyError(f"Memmap array '{name}' is not registered") from None

    def names(self) -> Iterable[str]:
        """Iterate over registered array names."""

        return tuple(self._arrays.keys())

    def total_size_bytes(self) -> int:
        """Sum of on-disk sizes of all registered arrays."""

        return sum(info.size_bytes for info in self._infos.values())

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def write(self, name: str, data: np.ndarray) -> None:
        """Overwrite the contents of ``name`` with ``data`` (broadcasted)."""

        arr = self.get(name)
        arr[:] = data
        arr.flush()

    def fill(self, name: str, value: float) -> None:
        """Fill the entire array ``name`` with the scalar ``value``."""

        arr = self.get(name)
        arr[:] = value
        arr.flush()

    def scale(self, name: str, factor: float) -> None:
        """Multiply the array ``name`` in-place by ``factor`` (and flush)."""

        arr = self.get(name)
        # In-place multiply keeps memmap semantics and avoids a temp copy.
        np.multiply(arr, factor, out=arr)
        arr.flush()

    def add_at(self, name: str, y: int, x: int, value: float, clip_max: Optional[float] = None) -> None:
        """Increment ``arr[y, x]`` by ``value``, clipping to ``clip_max``.

        Out-of-bounds coordinates are silently ignored so callers can pass
        world coordinates without bounds checking.
        """

        arr = self.get(name)
        h, w = self._infos[name].shape
        if 0 <= y < h and 0 <= x < w:
            new_val = float(arr[y, x]) + float(value)
            if clip_max is not None and new_val > clip_max:
                new_val = clip_max
            arr[y, x] = new_val

    # ------------------------------------------------------------------
    # Window reads
    # ------------------------------------------------------------------

    def get_window(
        self,
        name: str,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        normalize_by: Optional[float] = None,
        out_dtype: Union[str, np.dtype] = np.float32,
    ) -> np.ndarray:
        """Return ``arr[y0:y1, x0:x1]`` with zero-padding outside bounds.

        Parameters
        ----------
        name:
            Array to slice from.
        y0, y1, x0, x1:
            Half-open window bounds in array coordinates.
        normalize_by:
            If provided and positive, divide the resulting window by this
            value and clip to ``[0, 1]``.
        out_dtype:
            Numpy dtype of the returned array (default ``float32``).

        Returns
        -------
        numpy.ndarray
            A freshly-allocated array of shape ``(y1 - y0, x1 - x0)``. The
            window is *never* a view into the underlying memmap so callers can
            safely mutate the result.
        """

        arr = self._views[name]
        out = get_zero_padded_window_2d(
            arr, arr.shape, y0, y1, x0, x1, out_dtype=out_dtype
        )

        if normalize_by is not None and normalize_by > 0:
            np.divide(out, float(normalize_by), out=out)
            np.clip(out, 0.0, 1.0, out=out)
        return out

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def flush(self, name: Optional[str] = None) -> None:
        """Flush a single array (or all of them when ``name`` is ``None``)."""

        if name is None:
            for arr in self._arrays.values():
                try:
                    arr.flush()
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning(
                        "memmap_flush_failed",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
            return
        try:
            self._arrays[name].flush()
        except KeyError:
            raise KeyError(f"Memmap array '{name}' is not registered") from None

    def close(self, name: str, *, delete_file: bool = False) -> None:
        """Drop a single array from the manager and optionally delete its file."""

        arr = self._arrays.pop(name, None)
        self._views.pop(name, None)
        info = self._infos.pop(name, None)
        if arr is not None:
            try:
                arr.flush()
            except Exception:  # pragma: no cover - best effort
                pass
            del arr
        if info is not None and delete_file and os.path.exists(info.path):
            try:
                os.remove(info.path)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning(
                    "memmap_delete_failed",
                    path=info.path,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

    def close_all(self, *, delete_files: bool = False) -> None:
        """Close every registered array, optionally deleting backing files."""

        for name in list(self._arrays.keys()):
            self.close(name, delete_file=delete_files)
