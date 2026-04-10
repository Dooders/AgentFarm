#!/usr/bin/env python3
"""Multi-device inference latency benchmarking harness.

Measures **median inference latency** (and optional throughput) for one or more
Q-network checkpoints across devices (CPU and/or CUDA).  Designed for
reproducible cross-device comparisons of parent, student, int8-quantized, and
child networks with identical input shapes.

How to run
----------
::

    # Benchmark a single checkpoint on CPU (random weights, no checkpoint needed)
    python scripts/benchmark_inference.py --report-dir reports/bench

    # Benchmark parent + student checkpoints on CPU with a real states file
    python scripts/benchmark_inference.py \\
        --parent-ckpt  checkpoints/parent.pt \\
        --student-ckpt checkpoints/student.pt \\
        --states-file  data/replay_states.npy \\
        --report-dir   reports/bench

    # Compare CPU vs CUDA latency
    python scripts/benchmark_inference.py \\
        --parent-ckpt checkpoints/parent.pt \\
        --devices     cpu,cuda \\
        --batch-sizes 1,16,64 \\
        --report-dir  reports/bench

    # Benchmark all four roles from a checkpoint directory
    python scripts/benchmark_inference.py \\
        --checkpoint-dir checkpoints/experiment \\
        --devices        cpu,cuda \\
        --report-dir     reports/bench \\
        --allow-unsafe-unpickle

Architecture flags (``--input-dim``, ``--output-dim``, ``--hidden-size``)
must match the values used when checkpoints were trained.  The defaults
(8, 4, 64) are the standard AgentFarm experiment dimensions.

Inputs
------
- ``--parent-ckpt``: :class:`BaseQNetwork` state-dict checkpoint.
- ``--student-ckpt``: :class:`StudentQNetwork` state-dict checkpoint.
- ``--int8-ckpt``: quantized full-model checkpoint (requires
  ``--allow-unsafe-unpickle``).
- ``--child-ckpt``: :class:`BaseQNetwork` state-dict checkpoint (crossover
  child or fine-tuned network).
- ``--checkpoint-dir``: directory from which missing role paths are inferred
  (``parent.pt``, ``student.pt``, ``int8.pt``, ``child.pt``).
- ``--states-file``: optional NumPy ``.npy`` array of shape ``(N, input_dim)``
  with ``dtype=float32``.  When absent, a synthetic standard-normal dataset
  is generated from ``--seed``.

Outputs
-------
Results are printed to stdout as a Markdown table and (when ``--report-dir``
is provided) written to:

- ``<report-dir>/inference_benchmark.json`` — full metrics dict
  (``schema_version`` ``1.1``: run metadata includes ``states_shape`` (probe rows
  used, capped at ``max(batch_sizes)``), ``states_loaded_shape``, ``devices``,
  ``batch_sizes``, ``git_commit``, ``hostname``, etc.).
- ``<report-dir>/inference_benchmark.md`` — human-readable Markdown table.

Sample output
-------------
::

    Inference Latency Benchmark
    ===========================

    | Model   | Device | Batch |  Median (ms) |   Mean (ms) |   P95 (ms) | Throughput (batch/s) |
    |---------|--------|------:|-------------:|------------:|-----------:|---------------------:|
    | parent  | cpu    |     1 |        0.123 |       0.125 |      0.138 |              8130.1 |
    | student | cpu    |     1 |        0.087 |       0.088 |      0.095 |             11494.3 |
    | parent  | cpu    |    16 |        0.342 |       0.345 |      0.360 |             46783.6 |
    | student | cpu    |    16 |        0.241 |       0.243 |      0.258 |             66390.0 |

Related: ``RecombinationEvaluator._measure_latency``, ``StudentValidator``
latency fields, ``scripts/validate_distillation.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Allow running directly from repo root without requiring pip install -e .
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_distillation_states,
)
from farm.core.decision.training.quantize_ptq import (  # noqa: E402
    load_quantized_checkpoint,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_P95_PERCENTILE: int = 95
"""Percentile value used to compute P95 latency statistics."""

_MIN_ELAPSED_SEC: float = 1e-9
"""Minimum elapsed time guard (seconds) to avoid division-by-zero in throughput."""


def _is_supported_device_string(device_str: str) -> bool:
    s = device_str.strip()
    if s == "cpu":
        return True
    if s.startswith("cuda"):
        return True
    return False


def _provenance() -> Tuple[str, str]:
    """Return ``(git_commit, hostname)`` for reproducibility metadata; commit may be empty."""
    hostname = socket.gethostname()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root,
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        commit = ""
    return commit, hostname


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRow:
    """Single benchmark result for one model × device × batch combination."""

    model: str
    device: str
    batch_size: int
    median_ms: float
    mean_ms: float
    p95_ms: float
    throughput_batches_per_sec: float
    n_warmup: int
    n_repeats: int


@dataclass
class BenchmarkReport:
    """Full benchmark output containing all rows and run metadata.

    ``states_shape`` is the probe tensor shape (at most ``max(batch_sizes)`` rows)
    that timed forwards use; ``states_loaded_shape`` is the array shape before any
    padding to satisfy the largest batch (when applicable).
    """

    schema_version: str
    torch_version: str
    states_shape: Tuple[int, int]
    states_loaded_shape: Tuple[int, int]
    states_source: str
    devices: List[str]
    batch_sizes: List[int]
    warmup: int
    repeats: int
    throughput_repeats: int
    git_commit: str
    hostname: str
    rows: List[BenchmarkRow]

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuples to lists for JSON serialisability
        d["states_shape"] = list(self.states_shape)
        d["states_loaded_shape"] = list(self.states_loaded_shape)
        return d


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_parent(path: str, input_dim: int, output_dim: int, hidden_size: int) -> BaseQNetwork:
    net = BaseQNetwork(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size)
    if path:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(
                f"Parent checkpoint at {path!r} does not contain a state dict "
                f"(got {type(state).__name__})."
            )
        net.load_state_dict(state)
    net.eval()
    return net


def _load_student(path: str, input_dim: int, output_dim: int, hidden_size: int) -> StudentQNetwork:
    net = StudentQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        parent_hidden_size=hidden_size,
    )
    if path:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(
                f"Student checkpoint at {path!r} does not contain a state dict "
                f"(got {type(state).__name__})."
            )
        net.load_state_dict(state)
    net.eval()
    return net


def _load_int8(path: str, allow_unsafe: bool) -> nn.Module:
    if not path:
        raise ValueError("--int8-ckpt is required to benchmark an int8 model.")
    if not allow_unsafe:
        raise ValueError(
            "Loading a quantized (int8) checkpoint requires --allow-unsafe-unpickle "
            "because it uses torch.load with weights_only=False."
        )
    model, _meta = load_quantized_checkpoint(path)
    model.eval()
    return model


def _resolve_checkpoint(explicit: str, directory: str, filename: str) -> str:
    """Return explicit path if set; fall back to directory + filename."""
    if explicit:
        return explicit
    if directory:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_model_single(
    model: nn.Module,
    inp: torch.Tensor,
    n_warmup: int,
    n_repeats: int,
) -> Tuple[float, float, float]:
    """Run *n_repeats* timed single forward passes; return (median, mean, p95) in ms."""
    device = inp.device
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(inp)
    _sync_cuda(device)
    times: List[float] = []
    with torch.no_grad():
        for _ in range(n_repeats):
            _sync_cuda(device)
            t0 = time.perf_counter()
            model(inp)
            _sync_cuda(device)
            times.append((time.perf_counter() - t0) * 1_000.0)
    arr = np.asarray(times, dtype=np.float64)
    return float(np.median(arr)), float(np.mean(arr)), float(np.percentile(arr, _P95_PERCENTILE))


def _throughput_batches_per_sec(
    model: nn.Module,
    inp: torch.Tensor,
    n_warmup: int,
    n_timed: int,
) -> float:
    """Average batches/sec over *n_timed* full forward passes after warmup."""
    if n_timed <= 0 or inp.size(0) == 0:
        return 0.0
    device = inp.device
    model.eval()
    with torch.no_grad():
        for _ in range(max(n_warmup, 0)):
            model(inp)
    _sync_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_timed):
            model(inp)
            _sync_cuda(device)
    elapsed = time.perf_counter() - t0
    return float(n_timed / max(elapsed, _MIN_ELAPSED_SEC))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    named_models: List[Tuple[str, nn.Module]],
    states: np.ndarray,
    devices: List[str],
    batch_sizes: List[int],
    n_warmup: int,
    n_repeats: int,
    throughput_n_timed: int,
    *,
    states_source: str,
    states_loaded_shape: Tuple[int, int],
    git_commit: str,
    hostname: str,
) -> BenchmarkReport:
    """Run latency benchmark for each (model, device, batch_size) combination.

    Parameters
    ----------
    named_models:
        List of ``(label, model)`` pairs.  Models are expected to be on CPU
        initially; they will be moved to each target device as needed.
    states:
        Float32 array of shape ``(N, input_dim)`` — source rows for the probe tensor.
        Only the first ``max(batch_sizes)`` rows are used (see ``states_shape`` in the report).
    devices:
        Device strings, e.g. ``["cpu"]`` or ``["cpu", "cuda"]``.
    batch_sizes:
        List of batch sizes to benchmark, e.g. ``[1, 16]``.
    n_warmup:
        Number of warmup forward passes (excluded from timing).
    n_repeats:
        Number of timed forward passes used to compute median/mean/p95.
    throughput_n_timed:
        Number of forward passes used to measure throughput (batches/sec).
        Set to 0 to skip throughput measurement.
    states_source:
        Human-readable description of where *states* came from (file path or synthetic).
    states_loaded_shape:
        ``(N, input_dim)`` immediately after loading / synthesising states, **before**
        any padding used to satisfy ``max(batch_sizes)``.
    git_commit:
        Repository ``HEAD`` revision if available (else empty string).
    hostname:
        Result of :func:`socket.gethostname` for the machine that ran the benchmark.

    Returns
    -------
    BenchmarkReport
        Structured result with one :class:`BenchmarkRow` per combination.
    """
    rows: List[BenchmarkRow] = []
    max_batch = max(batch_sizes)
    # Slice the states array to the largest required batch.
    probe_cpu = torch.tensor(states[:max_batch], dtype=torch.float32)
    states_shape = (int(probe_cpu.shape[0]), int(probe_cpu.shape[1]))

    for device_str in devices:
        device = torch.device(device_str)
        for label, model_cpu in named_models:
            # Move model to target device (or keep on CPU for int8 which must stay on CPU).
            try:
                model = model_cpu.to(device)
            except (RuntimeError, TypeError) as exc:
                # Only int8 checkpoints are expected to refuse non-CPU devices; other
                # failures (OOM, invalid device) should surface to the operator.
                if device_str != "cpu" and label == "int8":
                    print(
                        f"  [warn] Cannot move {label!r} to {device_str}; "
                        "skipping (quantized models must run on CPU)."
                    )
                    continue
                raise RuntimeError(
                    f"Failed to move model {label!r} to {device_str!r}."
                ) from exc

            model.eval()
            for batch_size in batch_sizes:
                inp = probe_cpu[:batch_size].to(device)
                # Single-sample timing uses batch_size slice.
                median_ms, mean_ms, p95_ms = _time_model_single(
                    model, inp, n_warmup, n_repeats
                )
                tput = (
                    _throughput_batches_per_sec(model, inp, n_warmup, throughput_n_timed)
                    if throughput_n_timed > 0
                    else 0.0
                )
                rows.append(
                    BenchmarkRow(
                        model=label,
                        device=device_str,
                        batch_size=batch_size,
                        median_ms=median_ms,
                        mean_ms=mean_ms,
                        p95_ms=p95_ms,
                        throughput_batches_per_sec=tput,
                        n_warmup=n_warmup,
                        n_repeats=n_repeats,
                    )
                )

    return BenchmarkReport(
        schema_version="1.1",
        torch_version=torch.__version__,
        states_shape=states_shape,
        states_loaded_shape=states_loaded_shape,
        states_source=states_source,
        devices=list(devices),
        batch_sizes=list(batch_sizes),
        warmup=n_warmup,
        repeats=n_repeats,
        throughput_repeats=throughput_n_timed,
        git_commit=git_commit,
        hostname=hostname,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_table(rows: List[BenchmarkRow], show_throughput: bool) -> str:
    """Render benchmark rows as a Markdown table string."""
    if not rows:
        return "No benchmark results."

    headers = ["Model", "Device", "Batch", "Median (ms)", "Mean (ms)", "P95 (ms)"]
    if show_throughput:
        headers.append("Throughput (batch/s)")

    def _row_cells(r: BenchmarkRow) -> List[str]:
        cells = [
            r.model,
            r.device,
            str(r.batch_size),
            f"{r.median_ms:.3f}",
            f"{r.mean_ms:.3f}",
            f"{r.p95_ms:.3f}",
        ]
        if show_throughput:
            cells.append(f"{r.throughput_batches_per_sec:.1f}")
        return cells

    all_cells = [headers] + [_row_cells(r) for r in rows]
    col_widths = [max(len(c[i]) for c in all_cells) for i in range(len(headers))]

    def _fmt_row(cells: List[str], widths: List[int]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            # Right-align numeric columns (Batch and beyond index 2).
            if i >= 2:
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "| " + " | ".join(parts) + " |"

    lines: List[str] = []
    lines.append(_fmt_row(headers, col_widths))
    sep_parts = []
    for i, w in enumerate(col_widths):
        if i >= 2:
            sep_parts.append("-" * (w - 1) + ":")
        else:
            sep_parts.append("-" * w)
    lines.append("| " + " | ".join(sep_parts) + " |")
    for r in rows:
        lines.append(_fmt_row(_row_cells(r), col_widths))
    return "\n".join(lines)


def _print_report(report: BenchmarkReport, show_throughput: bool) -> None:
    print()
    print("Inference Latency Benchmark")
    print("===========================")
    print()
    print(f"States shape : {report.states_shape[0]} × {report.states_shape[1]}")
    if report.states_loaded_shape != report.states_shape:
        print(
            f"States loaded: {report.states_loaded_shape[0]} × {report.states_loaded_shape[1]} "
            "(padded/truncated to max batch for probe tensor)"
        )
    if report.states_source:
        print(f"States source: {report.states_source}")
    print(f"Devices      : {report.devices}")
    print(f"Batch sizes  : {report.batch_sizes}")
    print(f"PyTorch      : {report.torch_version}")
    print()
    print(_format_table(report.rows, show_throughput))
    print()


def _write_reports(report: BenchmarkReport, report_dir: str, show_throughput: bool) -> None:
    os.makedirs(report_dir, exist_ok=True)

    json_path = os.path.join(report_dir, "inference_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"JSON report : {json_path}")

    md_path = os.path.join(report_dir, "inference_benchmark.md")
    loaded_line = (
        f"**States loaded shape** (before padding): {report.states_loaded_shape[0]} × "
        f"{report.states_loaded_shape[1]}  \n"
        if report.states_loaded_shape != report.states_shape
        else ""
    )
    header = (
        "# Inference Latency Benchmark\n\n"
        f"**States shape** (array used): {report.states_shape[0]} × {report.states_shape[1]}  \n"
        f"{loaded_line}"
        f"**States source**: {report.states_source or 'synthetic'}  \n"
        f"**PyTorch**: {report.torch_version}  \n"
        f"**Devices**: {', '.join(report.devices)}  \n"
        f"**Batch sizes**: {', '.join(str(b) for b in report.batch_sizes)}  \n"
        f"**Warmup / repeats / throughput repeats**: "
        f"{report.warmup} / {report.repeats} / {report.throughput_repeats}  \n"
        f"**Hostname**: {report.hostname or '—'}  \n"
        f"**Git commit**: {report.git_commit or '—'}  \n\n"
    )
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(_format_table(report.rows, show_throughput))
        fh.write("\n")
    print(f"MD report   : {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-device inference latency benchmarking harness for Q-networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint paths
    ckpt = parser.add_argument_group("checkpoints")
    ckpt.add_argument(
        "--checkpoint-dir",
        default="",
        help=(
            "Directory used to infer missing checkpoint paths "
            "(parent.pt, student.pt, int8.pt, child.pt)."
        ),
    )
    ckpt.add_argument("--parent-ckpt", default="", help="Path to parent BaseQNetwork state dict.")
    ckpt.add_argument(
        "--student-ckpt", default="", help="Path to student StudentQNetwork state dict."
    )
    ckpt.add_argument(
        "--int8-ckpt",
        default="",
        help="Path to quantized (int8) full-model checkpoint.",
    )
    ckpt.add_argument("--child-ckpt", default="", help="Path to child BaseQNetwork state dict.")
    ckpt.add_argument(
        "--allow-unsafe-unpickle",
        action="store_true",
        help="Required to load quantized full-model checkpoints (trusted artifacts only).",
    )

    # Architecture
    arch = parser.add_argument_group("architecture")
    arch.add_argument("--input-dim", type=int, default=8)
    arch.add_argument("--output-dim", type=int, default=4)
    arch.add_argument("--hidden-size", type=int, default=64)

    # States
    st = parser.add_argument_group("states")
    st.add_argument(
        "--states-file",
        default="",
        help=(
            "Path to a .npy file of shape (N, input_dim) with dtype=float32.  "
            "When absent, synthetic random states are generated."
        ),
    )
    st.add_argument(
        "--n-states",
        type=int,
        default=256,
        help="Number of synthetic states to generate when --states-file is not provided.",
    )
    st.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic state generation.")

    # Benchmark settings
    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "--devices",
        default="cpu",
        help=(
            "Comma-separated list of devices to benchmark.  "
            "E.g. 'cpu' or 'cpu,cuda'.  CUDA entries are skipped when unavailable."
        ),
    )
    bench.add_argument(
        "--batch-sizes",
        default="1",
        help="Comma-separated list of batch sizes to benchmark.  E.g. '1,16,64'.",
    )
    bench.add_argument("--warmup", type=int, default=20, help="Number of warmup forward passes.")
    bench.add_argument(
        "--repeats",
        type=int,
        default=200,
        help="Number of timed forward passes for latency statistics.",
    )
    bench.add_argument(
        "--throughput-repeats",
        type=int,
        default=200,
        help="Number of forward passes used to estimate throughput (batches/sec).  0 to skip.",
    )

    # Output
    out = parser.add_argument_group("output")
    out.add_argument(
        "--report-dir",
        default="",
        help="Directory to write JSON and Markdown reports.  Skipped when empty.",
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {args.input_dim}")
    if args.output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {args.output_dim}")
    if args.hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {args.hidden_size}")
    if args.n_states <= 0:
        raise ValueError(f"n_states must be positive, got {args.n_states}")
    if args.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {args.warmup}")
    if args.repeats <= 0:
        raise ValueError(f"repeats must be positive, got {args.repeats}")
    if args.throughput_repeats < 0:
        raise ValueError(f"throughput_repeats must be non-negative, got {args.throughput_repeats}")


def _parse_int_list(raw: str, name: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} cannot be empty")
    values = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            raise ValueError(f"{name}: {p!r} is not a valid integer") from None
        if v <= 0:
            raise ValueError(f"{name}: all values must be positive, got {v}")
        values.append(v)
    return values


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    batch_sizes = _parse_int_list(args.batch_sizes, "--batch-sizes")
    devices_raw = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices_raw:
        raise ValueError("--devices cannot be empty")

    for raw_dev in devices_raw:
        if not _is_supported_device_string(raw_dev):
            raise ValueError(
                f"Unsupported device {raw_dev!r}; use 'cpu' or 'cuda' / 'cuda:N'."
            )

    # Filter CUDA if not available.
    devices: List[str] = []
    for d in devices_raw:
        if d.startswith("cuda") and not torch.cuda.is_available():
            print(f"  [warn] CUDA requested but not available; skipping device {d!r}.")
            continue
        devices.append(d)
    if not devices:
        raise RuntimeError("No valid devices available to benchmark.")

    # Load states.
    states_file = args.states_file
    states = load_distillation_states(
        states_file=states_file,
        n_states=args.n_states,
        input_dim=args.input_dim,
        seed=args.seed,
    )
    states_source = states_file if states_file else f"synthetic_standard_normal(n={args.n_states})"
    states_loaded_shape = (int(states.shape[0]), int(states.shape[1]))

    # Ensure we have enough rows for the largest batch.
    max_batch = max(batch_sizes)
    if states.shape[0] < max_batch:
        print(
            f"  [warn] states has only {states.shape[0]} rows but max batch size is {max_batch}; "
            "repeating states to fill."
        )
        repeats_needed = (max_batch + states.shape[0] - 1) // states.shape[0]
        states = np.tile(states, (repeats_needed, 1))[:max_batch]

    # Resolve checkpoint paths.
    parent_ckpt = _resolve_checkpoint(args.parent_ckpt, args.checkpoint_dir, "parent.pt")
    student_ckpt = _resolve_checkpoint(args.student_ckpt, args.checkpoint_dir, "student.pt")
    int8_ckpt = _resolve_checkpoint(args.int8_ckpt, args.checkpoint_dir, "int8.pt")
    child_ckpt = _resolve_checkpoint(args.child_ckpt, args.checkpoint_dir, "child.pt")

    # Build model list (only roles that have a resolvable or default path).
    named_models: List[Tuple[str, nn.Module]] = []

    # Always benchmark at least a parent (random weights if no checkpoint).
    if parent_ckpt or not (student_ckpt or int8_ckpt or child_ckpt):
        if parent_ckpt and not os.path.isfile(parent_ckpt):
            raise FileNotFoundError(f"Parent checkpoint not found: {parent_ckpt!r}")
        print(f"  parent  : {parent_ckpt or '(random weights)'}")
        named_models.append(
            ("parent", _load_parent(parent_ckpt, args.input_dim, args.output_dim, args.hidden_size))
        )

    if student_ckpt:
        if not os.path.isfile(student_ckpt):
            raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt!r}")
        print(f"  student : {student_ckpt}")
        named_models.append(
            ("student", _load_student(student_ckpt, args.input_dim, args.output_dim, args.hidden_size))
        )

    if int8_ckpt:
        if not os.path.isfile(int8_ckpt):
            raise FileNotFoundError(f"Int8 checkpoint not found: {int8_ckpt!r}")
        print(f"  int8    : {int8_ckpt}")
        named_models.append(("int8", _load_int8(int8_ckpt, args.allow_unsafe_unpickle)))

    if child_ckpt:
        if not os.path.isfile(child_ckpt):
            raise FileNotFoundError(f"Child checkpoint not found: {child_ckpt!r}")
        print(f"  child   : {child_ckpt}")
        named_models.append(
            ("child", _load_parent(child_ckpt, args.input_dim, args.output_dim, args.hidden_size))
        )

    print()
    print(f"Devices     : {devices}")
    print(f"Batch sizes : {batch_sizes}")
    print(f"Warmup      : {args.warmup}")
    print(f"Repeats     : {args.repeats}")

    git_commit, hostname = _provenance()

    # Run benchmark.
    report = run_benchmark(
        named_models=named_models,
        states=states,
        devices=devices,
        batch_sizes=batch_sizes,
        n_warmup=args.warmup,
        n_repeats=args.repeats,
        throughput_n_timed=args.throughput_repeats,
        states_source=states_source,
        states_loaded_shape=states_loaded_shape,
        git_commit=git_commit,
        hostname=hostname,
    )

    show_throughput = args.throughput_repeats > 0
    _print_report(report, show_throughput)

    if args.report_dir:
        _write_reports(report, args.report_dir, show_throughput)

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
