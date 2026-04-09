#!/usr/bin/env python
"""Validate quantized student checkpoints against float references.

This script compares the outputs of quantized student models
(produced by ``scripts/quantize_distilled.py``) against their float
counterparts, reporting fidelity, latency, file size, and compatibility
metadata.  It mirrors the structure of ``scripts/validate_distillation.py``
and is intended to be run after the quantization step.

How to run
----------
::

    # Validate both quantized students against their float checkpoints
    python scripts/validate_quantized.py \\
        --float-dir   checkpoints/distillation \\
        --quant-dir   checkpoints/quantized \\
        --allow-unsafe-unpickle \\
        --report-dir  reports/quantization_validation

    # Validate a single pair with a real state file
    python scripts/validate_quantized.py \\
        --pair A \\
        --float-a-ckpt  checkpoints/distillation/student_A.pt \\
        --quant-a-ckpt  checkpoints/quantized/student_A_int8.pt \\
        --states-file   data/replay_states.npy

Architecture flags (default 8, 4, 64) must match the values used in
``run_distillation.py`` and ``quantize_distilled.py``.

Inputs
------
- ``--float-dir`` / ``--float-{a,b}-ckpt``: float student checkpoint(s).
- ``--quant-dir`` / ``--quant-{a,b}-ckpt``: quantized checkpoint(s)
  (saved by :class:`PostTrainingQuantizer` or :class:`QATTrainer`).
- ``--allow-unsafe-unpickle``: explicit opt-in required to load quantized
  full-model pickle checkpoints (trusted artifacts only).
- ``--states-file``: optional NumPy ``.npy`` file of shape ``(N, input_dim)``
  with ``dtype=float32``.  When absent a synthetic standard-normal dataset is
  used for quick sanity checks.

Device
------
Both models run on CPU only. Quantized checkpoints rely on packed int8
kernels that are not CUDA-compatible in this validation path.

Interpreting the report
-----------------------
The JSON report includes these top-level sections (among others):

``fidelity``
    Action agreement, Q-error, cosine similarity vs the float model, MSE /
    KL on softmax(Q), top-*k* agreement (default *k* = 3; see ``--top-k``),
    and optional ``fidelity_vs_teacher`` when a parent checkpoint is supplied.
    For dynamic-quantized models expect high top-1 agreement; static and QAT
    may be lower.
``latency``
    Single-sample inference (ms) with warmup excluded: **median** (fields
    ``float_inference_ms`` / ``quantized_inference_ms``), plus **mean** and
    **p95**.  ``latency_ratio`` uses medians.  On CPU, quantized latency can
    exceed float for tiny batches.
``throughput``
    Optional fixed-batch throughput (batches/sec) when
    ``--throughput-batch-size`` > 0.
``memory``
    Best-effort RSS snapshots (``psutil``) and ``resource.getrusage`` peak
    (platform-dependent units for the latter).
``size``
    On-disk checkpoint sizes in bytes.  Quantized checkpoints include
    Python pickle overhead so the raw ratio can differ from the
    theoretical 4× (float32 → int8) saving.
``compatibility``
    PyTorch version, quantization mode/backend/dtype, and a ``compatible``
    boolean (``True`` when the forward pass completed without error).

``passed``
    ``True`` when all threshold checks pass.  Set ``--report-only`` to
    always emit a report without failing.

Known limitations
-----------------
- Static quantization performance may vary across PyTorch minor versions.
- Checkpoint sizes reflect Python pickle + metadata overhead and are not
  a pure measure of weight storage.
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

from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_base_qnetwork_checkpoint,
    load_distillation_states,
    load_float_student_checkpoint,
)
from farm.core.decision.training.quantize_ptq import (  # noqa: E402
    QuantizedValidationThresholds,
    QuantizedValidator,
    load_quantized_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(pair: str, explicit: str, directory: str, template: str) -> str:
    if explicit:
        return explicit
    if directory:
        return os.path.join(directory, template.format(pair=pair))
    return ""


def _resolve_parent_ckpt(pair: str, explicit: str, *search_dirs: str) -> str:
    """Return path to parent (teacher) checkpoint if explicit or found under a directory."""
    if explicit:
        return explicit
    for d in search_dirs:
        if d:
            candidate = os.path.join(d, f"parent_{pair}.pt")
            if os.path.isfile(candidate):
                return candidate
    return ""


def _print_report(pair: str, report_dict: dict) -> None:
    sep = "=" * 72
    compat = report_dict.get("compatibility", {})
    fidelity = report_dict.get("fidelity", {})
    latency = report_dict.get("latency", {})
    size = report_dict.get("size", {})

    print(f"\n{sep}")
    print(f"Quantization validation report: student_{pair}")
    print(sep)
    print(f"Compatible      : {compat.get('compatible', 'unknown')}")
    print(f"PyTorch version : {compat.get('pytorch_version', 'unknown')}")
    print(f"Mode            : {compat.get('quantization_mode', 'unknown')}")
    print(f"Dtype           : {compat.get('quantization_dtype', 'unknown')}")
    print(f"Backend         : {compat.get('quantization_backend', 'unknown')}")
    print()
    print(f"Action agreement    : {fidelity.get('action_agreement', 0)*100:.2f}%")
    print(f"Mean Q-error        : {fidelity.get('mean_q_error', 0):.6f}")
    print(f"Max Q-error         : {fidelity.get('max_q_error', 0):.6f}")
    print(f"Cosine similarity   : {fidelity.get('mean_cosine_similarity', 0):.6f}")
    print(f"States evaluated    : {fidelity.get('n_states', 0)}")
    print()
    print(f"Float latency median (ms)   : {latency.get('float_inference_ms', 0):.4f}")
    print(f"Float latency mean / p95    : {latency.get('float_inference_ms_mean', 0):.4f} / {latency.get('float_inference_ms_p95', 0):.4f}")
    print(f"Quant latency median (ms)   : {latency.get('quantized_inference_ms', 0):.4f}")
    print(f"Quant latency mean / p95    : {latency.get('quantized_inference_ms_mean', 0):.4f} / {latency.get('quantized_inference_ms_p95', 0):.4f}")
    print(f"Latency ratio (median)      : {latency.get('latency_ratio', 0):.4f}")
    thr = report_dict.get("throughput") or {}
    if thr.get("batch_size"):
        print()
        print(f"Throughput batch size       : {thr.get('batch_size')}")
        print(f"Float batches/s             : {thr.get('float_batches_per_sec', 0):.2f}")
        print(f"Quant batches/s             : {thr.get('quantized_batches_per_sec', 0):.2f}")
    mem = report_dict.get("memory") or {}
    if mem.get("rss_bytes_before") is not None:
        print()
        print(f"RSS before (bytes)          : {mem.get('rss_bytes_before')}")
        print(f"RSS after (bytes)           : {mem.get('rss_bytes_after')}")
        print(f"Process peak RSS (ru_maxrss): {mem.get('process_peak_rss_kb')}")
    ft = report_dict.get("fidelity") or {}
    if "mse_logits" in ft:
        print()
        print(f"MSE (float vs quant logits) : {ft.get('mse_logits', 0):.6f}")
        print(f"KL(float || quant softmax)  : {ft.get('kl_divergence_float_vs_quant', 0):.6f}")
        tka = ft.get("top_k_agreements") or {}
        if tka:
            print(f"Top-k agreements (float∈quant top-k): {tka}")
    if "fidelity_vs_teacher" in report_dict:
        print()
        print("Vs teacher (distillation-style metrics):")
        for k, v in sorted(report_dict["fidelity_vs_teacher"].items()):
            print(f"  {k}: {v:.6f}")
    print()
    float_bytes = size.get("float_checkpoint_bytes")
    quant_bytes = size.get("quantized_checkpoint_bytes")
    size_ratio = size.get("size_ratio")
    if float_bytes is not None:
        print(f"Float size (bytes)  : {float_bytes:,}")
    if quant_bytes is not None:
        print(f"Quant size (bytes)  : {quant_bytes:,}")
    if size_ratio is not None:
        print(f"Size ratio          : {size_ratio:.4f}")
    print()
    print(f"Passed              : {report_dict.get('passed', False)}")


def _ensure_unsafe_unpickle_allowed(allow_flag: bool) -> None:
    """Guard full-model pickle loading behind explicit CLI opt-in."""
    if not allow_flag:
        raise ValueError(
            "Loading quantized checkpoints requires full-model unpickling "
            "(weights_only=False), which can execute arbitrary code. "
            "Re-run with --allow-unsafe-unpickle only for trusted checkpoints."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate quantized student checkpoints against float references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pair", choices=["A", "B", "both"], default="both")

    # Float checkpoints
    p.add_argument("--float-dir", default="", help="Dir containing student_A.pt / student_B.pt.")
    p.add_argument("--float-a-ckpt", default="")
    p.add_argument("--float-b-ckpt", default="")

    # Quantized checkpoints
    p.add_argument("--quant-dir", default="", help="Dir containing student_A_int8.pt / student_B_int8.pt.")
    p.add_argument("--quant-a-ckpt", default="")
    p.add_argument("--quant-b-ckpt", default="")
    p.add_argument(
        "--allow-unsafe-unpickle",
        action="store_true",
        help=(
            "Allow loading quantized full-model pickles via torch.load(weights_only=False). "
            "Use only with trusted checkpoints."
        ),
    )

    # Architecture
    p.add_argument("--input-dim", type=int, default=8)
    p.add_argument("--output-dim", type=int, default=4)
    p.add_argument("--parent-hidden", type=int, default=64)

    # States
    p.add_argument("--states-file", default="")
    p.add_argument("--n-states", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # Latency benchmark
    p.add_argument("--latency-warmup", type=int, default=5,
                   help="Forward passes excluded from latency timing.")
    p.add_argument("--latency-repeats", type=int, default=50,
                   help="Timed single-sample forward passes (median/mean/p95).")

    p.add_argument(
        "--fidelity-batch-size",
        type=int,
        default=256,
        help="Batch size for fidelity / teacher metric forward passes.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        nargs="*",
        default=None,
        metavar="K",
        help="Top-k agreement keys (float argmax in quant top-k). Omit for default [3]; "
        "pass --top-k with no values to disable.",
    )
    p.add_argument(
        "--throughput-batch-size",
        type=int,
        default=0,
        help="If > 0, measure batches/sec on this batch size (clamped to state count).",
    )
    p.add_argument("--throughput-warmup-batches", type=int, default=2)
    p.add_argument("--throughput-timed-batches", type=int, default=10)
    p.add_argument(
        "--no-memory-metrics",
        action="store_true",
        help="Skip RSS / getrusage memory fields in the report.",
    )
    p.add_argument(
        "--teacher-dir",
        default="",
        help="Directory with parent_A.pt / parent_B.pt (optional; also searches --float-dir).",
    )
    p.add_argument("--teacher-a-ckpt", default="", help="Explicit parent_A.pt path.")
    p.add_argument("--teacher-b-ckpt", default="", help="Explicit parent_B.pt path.")

    # Thresholds
    p.add_argument("--min-action-agreement", type=float, default=0.75)
    p.add_argument("--max-mean-q-error", type=float, default=0.5)
    p.add_argument("--min-cosine-similarity", type=float, default=0.75)
    p.add_argument("--max-latency-ratio", type=float, default=2.0)
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Emit the report without applying pass/fail thresholds.",
    )

    # Output
    p.add_argument("--report-dir", default="reports/quantization_validation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device("cpu")

    states = load_distillation_states(
        args.states_file, args.n_states, args.input_dim, args.seed
    )
    pairs = ["A", "B"] if args.pair == "both" else [args.pair]

    os.makedirs(args.report_dir, exist_ok=True)

    float_ckpts = {
        "A": _resolve("A", args.float_a_ckpt, args.float_dir, "student_{pair}.pt"),
        "B": _resolve("B", args.float_b_ckpt, args.float_dir, "student_{pair}.pt"),
    }
    quant_ckpts = {
        "A": _resolve("A", args.quant_a_ckpt, args.quant_dir, "student_{pair}_int8.pt"),
        "B": _resolve("B", args.quant_b_ckpt, args.quant_dir, "student_{pair}_int8.pt"),
    }
    teacher_ckpts = {
        "A": _resolve_parent_ckpt("A", args.teacher_a_ckpt, args.teacher_dir, args.float_dir),
        "B": _resolve_parent_ckpt("B", args.teacher_b_ckpt, args.teacher_dir, args.float_dir),
    }

    thresholds = QuantizedValidationThresholds(
        min_action_agreement=args.min_action_agreement,
        max_mean_q_error=args.max_mean_q_error,
        min_cosine_similarity=args.min_cosine_similarity,
        max_latency_ratio=args.max_latency_ratio,
        report_only=args.report_only,
    )

    any_failed = False

    for pair in pairs:
        float_ckpt = float_ckpts[pair]
        quant_ckpt = quant_ckpts[pair]

        if not float_ckpt:
            raise ValueError(f"Missing float checkpoint path for pair {pair}.")
        if not quant_ckpt:
            raise ValueError(f"Missing quantized checkpoint path for pair {pair}.")

        float_model = load_float_student_checkpoint(
            float_ckpt,
            args.input_dim,
            args.output_dim,
            args.parent_hidden,
            not_found_template="Float student checkpoint not found: {path}",
            bad_state_template="Expected a state dict at {path!r}, got {type_name}.",
        )
        _ensure_unsafe_unpickle_allowed(args.allow_unsafe_unpickle)
        q_model, meta = load_quantized_checkpoint(quant_ckpt, device=device)

        teacher_path = teacher_ckpts[pair]
        teacher_model = None
        if teacher_path:
            teacher_model = load_base_qnetwork_checkpoint(
                teacher_path,
                args.input_dim,
                args.output_dim,
                args.parent_hidden,
            )
            teacher_model = teacher_model.to(device)
            teacher_model.eval()

        validator = QuantizedValidator(float_model, q_model, thresholds=thresholds, device=device)
        report = validator.validate(
            states,
            float_checkpoint_path=float_ckpt,
            quantized_checkpoint_path=quant_ckpt,
            quantization_metadata=meta,
            n_latency_warmup=args.latency_warmup,
            n_latency_repeats=args.latency_repeats,
            batch_size=args.fidelity_batch_size,
            top_k_values=args.top_k,
            throughput_batch_size=args.throughput_batch_size,
            throughput_warmup_batches=args.throughput_warmup_batches,
            throughput_timed_batches=args.throughput_timed_batches,
            track_memory=not args.no_memory_metrics,
            teacher_model=teacher_model,
        )

        report_dict = report.to_dict()
        report_dict["pair"] = pair
        report_dict["checkpoints"] = {
            "float": float_ckpt,
            "quantized": quant_ckpt,
            "teacher": teacher_path or None,
        }
        report_dict["states"] = {
            "count": int(states.shape[0]),
            "input_dim": int(states.shape[1]),
            "source": args.states_file if args.states_file else "synthetic_standard_normal",
            "seed": args.seed,
        }
        report_dict["benchmark"] = {
            "fidelity_batch_size": args.fidelity_batch_size,
            "top_k_values": ([] if args.top_k is not None and len(args.top_k) == 0 else (args.top_k or [3])),
            "latency_warmup": args.latency_warmup,
            "latency_repeats": args.latency_repeats,
            "throughput_batch_size": args.throughput_batch_size,
            "throughput_warmup_batches": args.throughput_warmup_batches,
            "throughput_timed_batches": args.throughput_timed_batches,
            "track_memory": not args.no_memory_metrics,
            "device": str(device),
        }

        _print_report(pair, report_dict)

        out_path = os.path.join(args.report_dir, f"quantization_validation_{pair}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2, allow_nan=False)
        print(f"JSON report written: {out_path}")

        if not report.passed:
            any_failed = True

    print("\nValidation complete.")
    if any_failed and not args.report_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
