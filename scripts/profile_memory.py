#!/usr/bin/env python
"""Profile peak RAM and storage for every stage of the recombination pipeline.

This script produces **comparable** peak resident-memory and storage numbers
for each stage—parent, student, quantized student, and child—using a common
batch size and state distribution.  Results are printed to stdout and written
to a JSON report file.

Profiling back-ends
-------------------
* **CPU** – ``tracemalloc`` (Python heap delta) + ``psutil`` process RSS.
* **GPU** – ``torch.cuda.max_memory_allocated`` (PyTorch caching allocator).

Both are captured automatically; only the ones available on the current
platform appear in the report.

How to run
----------
::

    # Minimal: profile the stages you have checkpoints for.
    python scripts/profile_memory.py \\
        --parent-ckpt   checkpoints/parent_A.pt \\
        --student-ckpt  checkpoints/student_A.pt \\
        --quant-ckpt    checkpoints/student_A_int8.pt \\
        --allow-unsafe-unpickle \\
        --report-dir    reports/memory

    # Override architecture if your models differ from the defaults.
    python scripts/profile_memory.py \\
        --input-dim 16 --output-dim 8 --parent-hidden 128 \\
        --batch-size 128 --n-forward-passes 10 \\
        --device cuda:0 \\
        --parent-ckpt   checkpoints/parent_A.pt \\
        --student-ckpt  checkpoints/student_A.pt \\
        --report-dir    reports/memory

    # Provide synthetic states without checkpoint files (quick sanity check).
    python scripts/profile_memory.py \\
        --parent-ckpt   checkpoints/parent_A.pt \\
        --n-states 500 \\
        --report-dir    reports/memory

Architecture flags
------------------
``--input-dim``, ``--output-dim``, ``--parent-hidden`` must match the values
used when the checkpoints were created (see ``run_distillation.py``).

Device
------
Use ``--device cuda`` or ``--device cuda:0`` to profile on GPU (CUDA must be
available).  The default is ``cpu``.

Interpreting the report
-----------------------
``state_dict_bytes``
    Sum of ``nelement() * element_size()`` for every tensor in the model's
    state dict.  This is the **in-memory** weight footprint, not the
    on-disk pickle size.
``checkpoint_bytes``
    On-disk file size in bytes (includes Python pickle / metadata overhead).
``tracemalloc_peak_bytes``
    Peak Python heap delta during the measured forward passes.  Captures only
    Python-managed allocations; does not include ATen/BLAS C++ buffers.
``rss_delta_bytes``
    RSS change (bytes) between block entry and exit.  May be negative due to
    OS page reclamation.  Use as a *directional* signal.
``process_peak_rss_ru``
    Lifetime process peak RSS in OS-native units (bytes on Linux, kilobytes
    on macOS).  Not limited to the measured block.
``cuda_peak_bytes``
    PyTorch caching-allocator high-watermark since the last reset.  Present
    only for CUDA device runs.

Known limitations
-----------------
* tracemalloc captures only Python heap; native C++ ATen buffers are invisible
  to it.
* ``psutil`` RSS on Linux is affected by OS paging noise.
* ``resource.getrusage`` units are OS-dependent (bytes/Linux, KB/macOS).
* Quantized checkpoints (``weights_only=False``) require explicit opt-in via
  ``--allow-unsafe-unpickle`` because they are full-model pickles.
* CUDA peak reflects the PyTorch allocator watermark; concurrent GPU work may
  inflate it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Allow running directly from repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_base_qnetwork_checkpoint,
    load_float_student_checkpoint,
)
from farm.core.decision.training.memory_profiler import (  # noqa: E402
    PipelineMemoryReport,
    StageMemoryProfile,
    profile_model_stage,
)
from farm.core.decision.training.quantize_ptq import (  # noqa: E402
    load_quantized_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_parent(path: str, input_dim: int, output_dim: int, hidden: int) -> BaseQNetwork:
    return load_base_qnetwork_checkpoint(path, input_dim, output_dim, hidden)


def _load_student(path: str, input_dim: int, output_dim: int, hidden: int) -> StudentQNetwork:
    return load_float_student_checkpoint(
        path,
        input_dim,
        output_dim,
        hidden,
        not_found_template="Student checkpoint not found: {path}",
        bad_state_template="Expected a state dict at {path!r}, got {type_name}.",
    )


def _ensure_unsafe_unpickle_allowed(allow_flag: bool) -> None:
    if not allow_flag:
        raise ValueError(
            "Loading quantized checkpoints requires full-model unpickling "
            "(weights_only=False), which can execute arbitrary code. "
            "Re-run with --allow-unsafe-unpickle only for trusted checkpoints."
        )


def _print_stage(profile: StageMemoryProfile) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"Stage: {profile.stage}")
    print(f"  Device                  : {profile.device}")
    print(f"  Batch size              : {profile.batch_size}")
    print(f"  Forward passes measured : {profile.n_forward_passes}")
    print(f"  State-dict bytes        : {profile.state_dict_bytes:,}")
    if profile.checkpoint_bytes is not None:
        print(f"  Checkpoint bytes (disk) : {profile.checkpoint_bytes:,}")
    ram = profile.peak_ram
    print(f"  tracemalloc peak (bytes): {ram.tracemalloc_peak_bytes}")
    if ram.rss_bytes_before is not None:
        print(f"  RSS before (bytes)      : {ram.rss_bytes_before:,}")
    if ram.rss_bytes_after is not None:
        print(f"  RSS after (bytes)       : {ram.rss_bytes_after:,}")
    if ram.rss_delta_bytes is not None:
        print(f"  RSS delta (bytes)       : {ram.rss_delta_bytes:,}")
    if ram.process_peak_rss_ru is not None:
        print(f"  Process peak RSS (ru)   : {ram.process_peak_rss_ru:,}  (OS-native units)")
    if ram.cuda_peak_bytes is not None:
        print(f"  CUDA peak (bytes)       : {ram.cuda_peak_bytes:,}")
    print(f"  Elapsed (s)             : {ram.elapsed_seconds:.4f}")
    for note in profile.notes:
        print(f"  NOTE: {note}")


def _print_summary(report: PipelineMemoryReport) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("Pipeline memory summary")
    print(sep)
    header = f"{'Stage':<14} {'StateDict(B)':>14} {'Ckpt(B)':>12} {'TMalloc peak(B)':>17} {'RSS delta(B)':>14}"
    print(header)
    print("-" * len(header))
    for s in report.stages:
        ckpt = f"{s.checkpoint_bytes:,}" if s.checkpoint_bytes is not None else "n/a"
        tm = f"{s.peak_ram.tracemalloc_peak_bytes}" if s.peak_ram.tracemalloc_peak_bytes is not None else "n/a"
        rss = f"{s.peak_ram.rss_delta_bytes:,}" if s.peak_ram.rss_delta_bytes is not None else "n/a"
        print(f"{s.stage:<14} {s.state_dict_bytes:>14,} {ckpt:>12} {tm:>17} {rss:>14}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile peak RAM and storage for each recombination pipeline stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoints (all optional—skip stages you don't have)
    p.add_argument("--parent-ckpt", default="", help="Path to parent (teacher) checkpoint.")
    p.add_argument("--student-ckpt", default="", help="Path to float student checkpoint.")
    p.add_argument(
        "--quant-ckpt",
        default="",
        help="Path to quantized student checkpoint (full-model pickle).",
    )
    p.add_argument(
        "--child-ckpt",
        default="",
        help="Path to child (crossover / fine-tuned) checkpoint.",
    )
    p.add_argument(
        "--allow-unsafe-unpickle",
        action="store_true",
        help=(
            "Allow torch.load(weights_only=False) for quantized full-model pickles. "
            "Use only for trusted checkpoints."
        ),
    )

    # Architecture
    p.add_argument("--input-dim", type=int, default=8)
    p.add_argument("--output-dim", type=int, default=4)
    p.add_argument(
        "--parent-hidden",
        type=int,
        default=64,
        help="Hidden size of the parent (BaseQNetwork) and student networks.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for models and profiling (e.g. cpu, cuda, cuda:0).",
    )

    # States
    p.add_argument("--states-file", default="", help="NumPy .npy file of shape (N, input_dim).")
    p.add_argument("--n-states", type=int, default=512, help="Synthetic states when --states-file absent.")
    p.add_argument("--seed", type=int, default=42)

    # Profiling knobs
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for forward passes (aligned across all stages).",
    )
    p.add_argument("--n-warmup", type=int, default=3, help="Un-timed warmup passes.")
    p.add_argument(
        "--n-forward-passes",
        type=int,
        default=5,
        help="Timed forward passes per stage.",
    )

    # Output
    p.add_argument(
        "--report-dir",
        default="reports/memory_profiling",
        help="Directory for JSON output.",
    )
    p.add_argument(
        "--report-file",
        default="",
        help="Explicit JSON output path (overrides --report-dir).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is False.")

    # Load or generate states
    if args.states_file:
        states = np.load(args.states_file).astype("float32")
        if states.ndim != 2:
            raise SystemExit(f"States file must be 2-D (N, input_dim), got shape {states.shape}.")
        if states.shape[1] != args.input_dim:
            raise SystemExit(
                f"States second dimension {states.shape[1]} does not match --input-dim {args.input_dim}."
            )
        print(f"Loaded states from {args.states_file}: {states.shape}")
    else:
        states = rng.standard_normal((args.n_states, args.input_dim)).astype("float32")
        print(f"Using {args.n_states} synthetic states (shape {states.shape})")
    stages: list[StageMemoryProfile] = []

    common_kwargs = dict(
        states=states,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        n_forward_passes=args.n_forward_passes,
        device=device,
    )

    # Parent stage
    if args.parent_ckpt:
        print(f"\nProfiling parent: {args.parent_ckpt}")
        parent_model = _load_parent(
            args.parent_ckpt, args.input_dim, args.output_dim, args.parent_hidden
        )
        stages.append(
            profile_model_stage(
                "parent",
                parent_model,
                checkpoint_path=args.parent_ckpt,
                **common_kwargs,
            )
        )
    else:
        print("Skipping parent stage (no --parent-ckpt).")

    # Student stage
    if args.student_ckpt:
        print(f"\nProfiling student: {args.student_ckpt}")
        student_model = _load_student(
            args.student_ckpt, args.input_dim, args.output_dim, args.parent_hidden
        )
        stages.append(
            profile_model_stage(
                "student",
                student_model,
                checkpoint_path=args.student_ckpt,
                **common_kwargs,
            )
        )
    else:
        print("Skipping student stage (no --student-ckpt).")

    # Quantized stage
    if args.quant_ckpt:
        _ensure_unsafe_unpickle_allowed(args.allow_unsafe_unpickle)
        print(f"\nProfiling quantized: {args.quant_ckpt}")
        q_model, _meta = load_quantized_checkpoint(args.quant_ckpt, device=device)
        stages.append(
            profile_model_stage(
                "quantized",
                q_model,
                checkpoint_path=args.quant_ckpt,
                **common_kwargs,
            )
        )
    else:
        print("Skipping quantized stage (no --quant-ckpt).")

    # Child stage (float student architecture)
    if args.child_ckpt:
        print(f"\nProfiling child: {args.child_ckpt}")
        child_model = _load_student(
            args.child_ckpt, args.input_dim, args.output_dim, args.parent_hidden
        )
        stages.append(
            profile_model_stage(
                "child",
                child_model,
                checkpoint_path=args.child_ckpt,
                **common_kwargs,
            )
        )
    else:
        print("Skipping child stage (no --child-ckpt).")

    if not stages:
        print("\nNo stages profiled—supply at least one checkpoint flag.")
        return

    # Print per-stage detail
    for profile in stages:
        _print_stage(profile)

    report = PipelineMemoryReport(stages=stages)
    _print_summary(report)

    # Write JSON
    if args.report_file:
        out_path = args.report_file
    else:
        os.makedirs(args.report_dir, exist_ok=True)
        out_path = os.path.join(args.report_dir, "memory_profile.json")

    report_dict = report.to_dict()
    report_dict["meta"] = {
        "batch_size": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_forward_passes": args.n_forward_passes,
        "n_states": int(states.shape[0]),
        "input_dim": int(states.shape[1]),
        "states_source": args.states_file or "synthetic_standard_normal",
        "seed": args.seed,
        "device": str(device),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2)
    print(f"\nJSON report written: {out_path}")


if __name__ == "__main__":
    main()
