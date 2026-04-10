"""Peak RAM and storage profiling harness for the recombination pipeline.

This module provides a lightweight, cross-platform harness for measuring
**peak resident memory** (RAM) during forward passes and **storage** (on-disk
/ state-dict byte footprints) for each stage of the recombination pipeline:
parent → student → quantized → child.

Two RAM tracking back-ends are supported and composed automatically:

* **CPU** – ``tracemalloc`` (Python allocator delta) and ``psutil`` process
  RSS snapshots.  The *tracemalloc peak* is the more reliable signal because
  it captures only the net Python heap delta introduced inside the measured
  block rather than the accumulated process lifetime peak.
* **GPU** – ``torch.cuda.max_memory_allocated`` / ``torch.cuda.reset_peak_memory_stats``
  when a CUDA device is present.

Quick start
-----------
::

    import numpy as np
    import torch
    from farm.core.decision.base_dqn import StudentQNetwork
    from farm.core.decision.training.memory_profiler import (
        profile_model_stage,
        PipelineMemoryReport,
    )

    model = StudentQNetwork(input_dim=8, output_dim=4, parent_hidden_size=64)
    states = np.random.standard_normal((256, 8)).astype("float32")

    profile = profile_model_stage("student", model, states, batch_size=64)
    print(profile.to_dict())

    # Profile multiple stages in one shot
    report = PipelineMemoryReport(stages=[profile])
    print(report.to_dict())

Known limitations
-----------------
* ``tracemalloc`` tracks only **Python-managed** allocations; it does not
  capture memory allocated by PyTorch's C++ ATen allocator or BLAS
  libraries.  Use process RSS as a complementary signal.
* ``psutil`` RSS on Linux reads from ``/proc/<pid>/status``; it reports
  virtual memory pages that are currently resident and may fluctuate due to
  OS paging activity and the kernel's lazy-eviction policy.
* ``resource.getrusage`` units differ by OS: bytes on Linux, kilobytes on
  macOS.  The field ``process_peak_rss_ru`` is reported in the OS-native
  units (see ``ru_maxrss`` man page); callers should note this in
  publication tables.
* GPU tracking with ``torch.cuda.max_memory_allocated`` reports the
  **PyTorch caching allocator** high-watermark, not the true device
  memory.  It resets to zero after ``reset_peak_memory_stats``; other
  concurrent kernels may inflate the reading.
* Allocator warm-up effects: on the first forward pass PyTorch may allocate
  extra buffers.  Pass ``n_warmup`` > 0 to exclude these from measurements.
* Nested :func:`profile_peak_ram`: if ``tracemalloc`` was already tracing
  before entry, ``tracemalloc_peak_bytes`` is the peak since tracing began
  (process-wide), not isolated to the inner block alone.
* ``state_dict_bytes`` sums tensor element counts per ``state_dict`` entry;
  unusual parameter sharing can make the sum exceed unique physical storage.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Generator, List, Optional

try:
    import resource as _resource
except ImportError:  # Windows
    _resource = None  # type: ignore[assignment]

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]

import numpy as np
import torch
import torch.nn as nn

from farm.utils.logging import get_logger

from .quantize_ptq import _estimate_tensor_bytes

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _rss_bytes() -> Optional[int]:
    """Return current process RSS in bytes, or ``None`` if psutil is absent."""
    if _psutil is None:
        return None
    return int(_psutil.Process(os.getpid()).memory_info().rss)


def _process_peak_rss_ru() -> Optional[int]:
    """Return ``resource.getrusage`` peak RSS in OS-native units (see module notes)."""
    if _resource is None:
        return None
    try:
        return int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ValueError, OSError):
        return None


def _cuda_peak_bytes(device: torch.device) -> Optional[int]:
    """Return peak CUDA memory (bytes) allocated since last reset, or ``None``."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _state_dict_bytes(model: nn.Module) -> int:
    """Estimate in-memory byte footprint of all tensors in *model*'s state dict.

    Handles quantized model state dicts that contain non-tensor entries such
    as ``torch.dtype`` objects inside ``PackedParams``.

    Each ``state_dict`` key is summed separately; tensors that share storage
    across keys still contribute once per key, so totals can exceed unique
    backing storage in edge cases.
    """
    return _estimate_tensor_bytes(model.state_dict())


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@dataclass
class PeakRAMSample:
    """Peak RAM measurements captured inside a :func:`profile_peak_ram` block.

    Attributes
    ----------
    tracemalloc_peak_bytes:
        Peak Python heap (bytes) from ``tracemalloc.get_traced_memory`` at
        block exit.  If tracing was already active before this block, the
        value is the peak since tracing began, not block-local.
    rss_bytes_before:
        Process RSS (bytes) at block entry.  ``None`` if ``psutil`` absent.
    rss_bytes_after:
        Process RSS (bytes) at block exit.  ``None`` if ``psutil`` absent.
    rss_delta_bytes:
        ``rss_bytes_after - rss_bytes_before``.  May be negative due to GC.
        ``None`` if either snapshot is unavailable.
    process_peak_rss_ru:
        ``resource.getrusage`` lifetime peak RSS in OS-native units (bytes on
        Linux, kilobytes on macOS).  This is a *process lifetime* peak, not
        limited to the block.
    cuda_peak_bytes:
        Peak CUDA allocator memory (bytes) since the last reset, or ``None``
        on CPU.
    elapsed_seconds:
        Wall-clock duration of the measured block in seconds.
    """

    tracemalloc_peak_bytes: Optional[int] = None
    rss_bytes_before: Optional[int] = None
    rss_bytes_after: Optional[int] = None
    rss_delta_bytes: Optional[int] = None
    process_peak_rss_ru: Optional[int] = None
    cuda_peak_bytes: Optional[int] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@contextmanager
def profile_peak_ram(
    device: Optional[torch.device] = None,
) -> Generator[PeakRAMSample, None, None]:
    """Context manager that measures peak RAM during an arbitrary code block.

    Yields a :class:`PeakRAMSample` that is populated *after* the block exits.

    If ``tracemalloc`` is already tracing when the block is entered, this
    context manager does **not** call ``stop()`` on exit, so outer tracing
    stays enabled.  In that case ``tracemalloc_peak_bytes`` reflects the
    process-wide peak since tracing started, not allocations confined to
    this block.

    Parameters
    ----------
    device:
        Optional ``torch.device``.  When it is a CUDA device, peak GPU
        memory is also recorded.

    Example
    -------
    ::

        with profile_peak_ram() as sample:
            output = model(batch)
        print(sample.tracemalloc_peak_bytes)
    """
    dev = device or torch.device("cpu")
    sample = PeakRAMSample()

    _reset_cuda_peak(dev)
    rss_before = _rss_bytes()
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    t0 = time.perf_counter()

    try:
        yield sample
    finally:
        elapsed = time.perf_counter() - t0
        _current, peak_tm = tracemalloc.get_traced_memory()
        if not already_tracing:
            tracemalloc.stop()

        rss_after = _rss_bytes()
        cuda_peak = _cuda_peak_bytes(dev)
        peak_ru = _process_peak_rss_ru()

        sample.tracemalloc_peak_bytes = int(peak_tm)
        sample.rss_bytes_before = rss_before
        sample.rss_bytes_after = rss_after
        sample.rss_delta_bytes = (
            (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None
        )
        sample.process_peak_rss_ru = peak_ru
        sample.cuda_peak_bytes = cuda_peak
        sample.elapsed_seconds = elapsed


# ---------------------------------------------------------------------------
# Per-stage profile
# ---------------------------------------------------------------------------


@dataclass
class StageMemoryProfile:
    """Memory profile for a single pipeline stage (e.g. ``"parent"``).

    Attributes
    ----------
    stage:
        Human-readable label, e.g. ``"parent"``, ``"student"``,
        ``"quantized"``, ``"child"``.
    batch_size:
        Number of samples used in each forward pass.
    n_forward_passes:
        Total number of measured forward passes (warmup excluded).
    state_dict_bytes:
        Estimated in-memory byte footprint of the model state dict.
    checkpoint_bytes:
        On-disk checkpoint size in bytes, or ``None`` if no path was given.
    peak_ram:
        :class:`PeakRAMSample` from the measured forward passes.
    device:
        Device string (e.g. ``"cpu"``, ``"cuda:0"``).
    notes:
        Free-form list of warnings or limitation notices.
    """

    stage: str
    batch_size: int
    n_forward_passes: int
    state_dict_bytes: int
    checkpoint_bytes: Optional[int]
    peak_ram: PeakRAMSample
    device: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public profiling function
# ---------------------------------------------------------------------------


def profile_model_stage(
    stage: str,
    model: nn.Module,
    states: np.ndarray,
    batch_size: int = 64,
    n_warmup: int = 3,
    n_forward_passes: int = 5,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> StageMemoryProfile:
    """Profile peak RAM and storage for one pipeline stage.

    A *stage* is any ``eval``-mode ``nn.Module`` corresponding to a step in
    the recombination pipeline: parent, student, quantized student, or child.
    The function:

    1. Runs ``n_warmup`` un-timed forward passes to prime allocator caches.
    2. Measures peak RAM over ``n_forward_passes`` consecutive forward passes
       (all batches in one :func:`profile_peak_ram` block).
    3. Records state-dict byte footprint and optional on-disk size.

    Parameters
    ----------
    stage:
        Human-readable stage label (e.g. ``"parent"``, ``"student"``).
    model:
        The ``nn.Module`` to profile.  It is placed in ``eval()`` mode.
    states:
        NumPy array of shape ``(N, input_dim)`` with ``dtype=float32``.
        Must be non-empty.  ``batch_size`` is clamped to ``N``.
    batch_size:
        Number of samples per forward pass.  Clamped to ``len(states)``.
    n_warmup:
        Number of un-timed forward passes before measurement.
    n_forward_passes:
        Number of timed forward passes to include in the measurement block.
    checkpoint_path:
        Optional path to an on-disk checkpoint file; when provided the file
        size is recorded in :attr:`StageMemoryProfile.checkpoint_bytes`.
    device:
        Target device.  Defaults to ``torch.device("cpu")``.

    Returns
    -------
    StageMemoryProfile
    """
    dev = device or torch.device("cpu")
    model = model.to(dev)
    model.eval()

    states_arr = np.asarray(states, dtype=np.float32)
    if states_arr.size == 0:
        raise ValueError("states must be non-empty (at least one row).")
    effective_bs = min(batch_size, len(states_arr))
    batch_np = states_arr[:effective_bs]
    batch_t = torch.from_numpy(batch_np).to(dev)

    notes: List[str] = []

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(batch_t)
            if hasattr(out, "dequantize") and out.is_quantized:
                out = out.dequantize()

    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)

    # Measure
    with profile_peak_ram(device=dev) as sample:
        with torch.no_grad():
            for _ in range(n_forward_passes):
                out = model(batch_t)
                if hasattr(out, "dequantize") and out.is_quantized:
                    out = out.dequantize()
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(dev)

    sd_bytes = _state_dict_bytes(model)
    ckpt_bytes: Optional[int] = None
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            ckpt_bytes = os.path.getsize(checkpoint_path)
        else:
            notes.append(
                f"checkpoint_path {checkpoint_path!r} is not a readable file; checkpoint_bytes omitted."
            )

    if _psutil is None:
        notes.append("psutil not installed; RSS metrics unavailable.")
    if _resource is None:
        notes.append("resource module unavailable (Windows?); process_peak_rss_ru not recorded.")
    if dev.type != "cuda":
        notes.append(
            "CUDA profiling skipped (CPU device). "
            "tracemalloc captures Python heap only; ATen/BLAS allocations are not included."
        )
    else:
        notes.append(
            "cuda_peak_bytes reflects PyTorch caching allocator high-watermark "
            "since last reset; concurrent kernels may inflate the reading."
        )

    logger.info(
        "stage_profiled",
        stage=stage,
        batch_size=effective_bs,
        n_forward_passes=n_forward_passes,
        state_dict_bytes=sd_bytes,
        tracemalloc_peak_bytes=sample.tracemalloc_peak_bytes,
        rss_delta_bytes=sample.rss_delta_bytes,
    )

    return StageMemoryProfile(
        stage=stage,
        batch_size=effective_bs,
        n_forward_passes=n_forward_passes,
        state_dict_bytes=sd_bytes,
        checkpoint_bytes=ckpt_bytes,
        peak_ram=sample,
        device=str(dev),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-stage report
# ---------------------------------------------------------------------------


@dataclass
class PipelineMemoryReport:
    """Aggregate peak-RAM and storage report for all pipeline stages.

    Parameters
    ----------
    stages:
        List of :class:`StageMemoryProfile` instances, one per stage.

    Example
    -------
    ::

        report = PipelineMemoryReport(stages=[parent_profile, student_profile])
        import json
        print(json.dumps(report.to_dict(), indent=2))
    """

    stages: List[StageMemoryProfile] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain Python dict suitable for ``json.dump``."""
        return {
            "stages": [s.to_dict() for s in self.stages],
            "summary": self._build_summary(),
        }

    def _build_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for s in self.stages:
            entry: Dict[str, Any] = {
                "state_dict_bytes": s.state_dict_bytes,
                "checkpoint_bytes": s.checkpoint_bytes,
                "tracemalloc_peak_bytes": s.peak_ram.tracemalloc_peak_bytes,
                "rss_delta_bytes": s.peak_ram.rss_delta_bytes,
            }
            if s.peak_ram.cuda_peak_bytes is not None:
                entry["cuda_peak_bytes"] = s.peak_ram.cuda_peak_bytes
            summary[s.stage] = entry
        return summary
