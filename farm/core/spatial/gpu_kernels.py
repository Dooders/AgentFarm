"""GPU-accelerated spatial computation kernels using PyTorch.

These kernels accelerate common spatial queries (radius search, nearest
neighbour) via vectorised distance-matrix operations.  When a CUDA device is
available the work is dispatched there; otherwise the same tensor code runs on
CPU, still benefiting from vectorised parallelism over a pure-Python loop.

Typical usage
-------------
>>> from farm.core.spatial.gpu_kernels import SpatialGpuKernels
>>> kernels = SpatialGpuKernels()          # auto-detect GPU
>>> indices = kernels.radius_query(positions, query_point, radius=20.0)
>>> nearest_idx = kernels.nearest_query(positions, query_point)

For bulk / batch queries (many query points at once):
>>> results = kernels.batch_radius_query(positions, query_points, radius=20.0)
>>> results = kernels.batch_nearest_query(positions, query_points)

Notes
-----
- Import of ``torch`` is deferred so that the rest of the spatial package
  continues to work on environments that do not have PyTorch installed.
- GPU acceleration is most beneficial when *N* (number of candidate positions)
  is large.  For small entity counts the per-call tensor-setup overhead
  exceeds the saving; the ``gpu_threshold`` parameter in :class:`SpatialGpuKernels`
  controls the cross-over point.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from farm.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def is_gpu_available() -> bool:
    """Return ``True`` if a CUDA GPU is available via PyTorch."""
    return _TORCH_AVAILABLE and torch.cuda.is_available()


def get_spatial_device(prefer_gpu: bool = True) -> "Optional[torch.device]":
    """Return the best device for spatial tensor operations.

    Parameters
    ----------
    prefer_gpu:
        When *True*, try to return a CUDA device; fall back to CPU if CUDA is
        not available.  When *False*, always returns a CPU device.

    Returns
    -------
    torch.device or None
        ``None`` only when PyTorch is not installed at all.
    """
    if not _TORCH_AVAILABLE:
        return None  # type: ignore[return-value]
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Low-level numpy fallbacks (no torch required)
# ---------------------------------------------------------------------------


def _np_pairwise_distances(
    positions: np.ndarray, query_points: np.ndarray
) -> np.ndarray:
    """Pure-NumPy pairwise Euclidean distances, shape ``(M, N)``."""
    # positions: (N, 2),  query_points: (M, 2)
    diff = query_points[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (M, N, 2)
    return np.sqrt((diff**2).sum(axis=-1))  # (M, N)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SpatialGpuKernels:
    """GPU-accelerated (or vectorised-CPU) spatial kernels.

    Parameters
    ----------
    use_gpu:
        Whether to prefer GPU execution.  When *True* and no CUDA device is
        found, falls back to CPU tensors transparently.
    gpu_threshold:
        Minimum number of candidate entities before the GPU/tensor path is
        preferred over a direct call.  Below this count the tensor-setup
        overhead is not worthwhile.  Default 256.
    device:
        Explicit ``torch.device`` to use.  Auto-selected when ``None``.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        gpu_threshold: int = 256,
        device: "Optional[torch.device]" = None,
    ) -> None:
        self.use_gpu = use_gpu
        self.gpu_threshold = gpu_threshold

        if device is not None:
            self._device = device
        elif _TORCH_AVAILABLE:
            self._device = get_spatial_device(prefer_gpu=use_gpu)
        else:
            self._device = None

        self._on_gpu: bool = (
            self._device is not None and self._device.type == "cuda"
        )

        logger.debug(
            "spatial_gpu_kernels_init",
            device=str(self._device),
            on_gpu=self._on_gpu,
            torch_available=_TORCH_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> "Optional[torch.device]":
        """The active compute device (``None`` when torch is unavailable)."""
        return self._device

    @property
    def on_gpu(self) -> bool:
        """``True`` when computations will run on a CUDA device."""
        return self._on_gpu

    # ------------------------------------------------------------------
    # Single-query helpers
    # ------------------------------------------------------------------

    def radius_query(
        self,
        positions: np.ndarray,
        query_point: Tuple[float, float],
        radius: float,
    ) -> List[int]:
        """Return indices of *positions* within *radius* of *query_point*.

        Parameters
        ----------
        positions:
            Candidate entity positions, shape ``(N, 2)``.
        query_point:
            Query ``(x, y)`` tuple.
        radius:
            Search radius (same units as positions).

        Returns
        -------
        list of int
            Indices into *positions* whose distance to *query_point* is ``<= radius``.
        """
        if len(positions) == 0:
            return []
        qp = np.asarray(query_point, dtype=np.float64).reshape(1, 2)
        results = self.batch_radius_query(positions, qp, radius)
        return results[0]

    def nearest_query(
        self,
        positions: np.ndarray,
        query_point: Tuple[float, float],
    ) -> int:
        """Return index of the nearest position to *query_point*.

        Parameters
        ----------
        positions:
            Candidate entity positions, shape ``(N, 2)``.
        query_point:
            Query ``(x, y)`` tuple.

        Returns
        -------
        int
            Index into *positions* of the nearest entity, or ``-1`` if
            *positions* is empty.
        """
        if len(positions) == 0:
            return -1
        qp = np.asarray(query_point, dtype=np.float64).reshape(1, 2)
        indices = self.batch_nearest_query(positions, qp)
        return int(indices[0])

    # ------------------------------------------------------------------
    # Batch query methods (primary GPU-accelerated paths)
    # ------------------------------------------------------------------

    def batch_radius_query(
        self,
        positions: np.ndarray,
        query_points: np.ndarray,
        radius: float,
    ) -> List[List[int]]:
        """Return neighbour indices for every query point in *query_points*.

        This is the primary GPU-accelerated path.  The full ``(M, N)``
        distance matrix is computed in a single batched operation.

        Parameters
        ----------
        positions:
            Candidate entity positions, shape ``(N, 2)``.
        query_points:
            Query positions, shape ``(M, 2)``.
        radius:
            Search radius.

        Returns
        -------
        list of lists of int
            ``results[i]`` contains the indices of all positions within
            *radius* of ``query_points[i]``.
        """
        positions = np.asarray(positions, dtype=np.float64)
        query_points = np.asarray(query_points, dtype=np.float64)

        n_entities = len(positions)
        m_queries = len(query_points)

        if n_entities == 0 or m_queries == 0:
            return [[] for _ in range(m_queries)]

        # ------ GPU / torch path ------
        if _TORCH_AVAILABLE and self._device is not None:
            pos_t = torch.as_tensor(positions, dtype=torch.float64, device=self._device)
            qpt_t = torch.as_tensor(query_points, dtype=torch.float64, device=self._device)
            # torch.cdist: (M, N) pairwise L2 distances
            dist_t = torch.cdist(qpt_t, pos_t)  # (M, N)
            mask = dist_t <= float(radius)       # (M, N) bool
            mask_cpu = mask.cpu()
            return [
                mask_cpu[i].nonzero(as_tuple=False).squeeze(1).tolist()
                for i in range(m_queries)
            ]

        # ------ NumPy CPU fallback ------
        dists = _np_pairwise_distances(positions, query_points)  # (M, N)
        mask = dists <= radius
        return [list(np.where(mask[i])[0]) for i in range(m_queries)]

    def batch_nearest_query(
        self,
        positions: np.ndarray,
        query_points: np.ndarray,
    ) -> np.ndarray:
        """Return the nearest-position index for every query point.

        Parameters
        ----------
        positions:
            Candidate entity positions, shape ``(N, 2)``.
        query_points:
            Query positions, shape ``(M, 2)``.

        Returns
        -------
        ndarray, shape ``(M,)``
            Each element is the index into *positions* of the nearest entity
            for the corresponding query point.  Values are ``-1`` for any
            query point when *positions* is empty.
        """
        positions = np.asarray(positions, dtype=np.float64)
        query_points = np.asarray(query_points, dtype=np.float64)

        n_entities = len(positions)
        m_queries = len(query_points)

        if n_entities == 0:
            return np.full(m_queries, -1, dtype=np.int64)
        if m_queries == 0:
            return np.empty(0, dtype=np.int64)

        # ------ GPU / torch path ------
        if _TORCH_AVAILABLE and self._device is not None:
            pos_t = torch.as_tensor(positions, dtype=torch.float64, device=self._device)
            qpt_t = torch.as_tensor(query_points, dtype=torch.float64, device=self._device)
            dist_t = torch.cdist(qpt_t, pos_t)   # (M, N)
            idx_t = dist_t.argmin(dim=1)          # (M,)
            return idx_t.cpu().numpy().astype(np.int64)

        # ------ NumPy CPU fallback ------
        dists = _np_pairwise_distances(positions, query_points)  # (M, N)
        return dists.argmin(axis=1).astype(np.int64)

    def pairwise_distances(
        self,
        positions: np.ndarray,
        query_points: np.ndarray,
    ) -> np.ndarray:
        """Compute the full pairwise distance matrix.

        Parameters
        ----------
        positions:
            Candidate positions, shape ``(N, 2)``.
        query_points:
            Query positions, shape ``(M, 2)``.

        Returns
        -------
        ndarray, shape ``(M, N)``
            ``distances[i, j]`` is the Euclidean distance between
            ``query_points[i]`` and ``positions[j]``.
        """
        positions = np.asarray(positions, dtype=np.float64)
        query_points = np.asarray(query_points, dtype=np.float64)

        if len(positions) == 0 or len(query_points) == 0:
            return np.empty((len(query_points), len(positions)), dtype=np.float64)

        if _TORCH_AVAILABLE and self._device is not None:
            pos_t = torch.as_tensor(positions, dtype=torch.float64, device=self._device)
            qpt_t = torch.as_tensor(query_points, dtype=torch.float64, device=self._device)
            dist_t = torch.cdist(qpt_t, pos_t)
            return dist_t.cpu().numpy()

        return _np_pairwise_distances(positions, query_points)
