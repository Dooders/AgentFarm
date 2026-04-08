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
- ``--states-file``: optional NumPy ``.npy`` file of shape ``(N, input_dim)``
  with ``dtype=float32``.  When absent a synthetic standard-normal dataset is
  used for quick sanity checks.

Device
------
Both models run on CPU by default; pass ``--device cuda`` if CUDA is
available and the quantized checkpoint is CUDA-compatible.  Note that
int8 kernel acceleration is CPU-only — CUDA simply dequantizes weights
at runtime.

Interpreting the report
-----------------------
The JSON report has four top-level sections:

``fidelity``
    Action agreement, Q-error, and cosine similarity vs the float model.
    For dynamic-quantized models expect ≥ 90 % agreement; static and QAT
    may be lower.
``latency``
    Median per-sample inference time (ms) with warmup excluded.  The
    ``latency_ratio`` (quantized/float) may exceed 1.0 on CPU for small
    batch sizes — this is expected for dynamic quantization.
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
- CUDA paths dequantize int8 weights before matmul; the latency ratio
  on GPU will typically be > 1.
- Checkpoint sizes reflect Python pickle + metadata overhead and are not
  a pure measure of weight storage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

# Allow running directly from repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import StudentQNetwork  # noqa: E402
from farm.core.decision.training.quantize_ptq import (  # noqa: E402
    QuantizedValidationThresholds,
    QuantizedValidator,
    load_quantized_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_float_student(
    path: str,
    input_dim: int,
    output_dim: int,
    parent_hidden: int,
) -> StudentQNetwork:
    model = StudentQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        parent_hidden_size=parent_hidden,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Float student checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict at {path!r}, got {type(state).__name__}.")
    model.load_state_dict(state)
    model.eval()
    return model


def _load_states(
    states_file: str,
    n_states: int,
    input_dim: int,
    seed: Optional[int],
) -> np.ndarray:
    if states_file:
        if not os.path.isfile(states_file):
            raise FileNotFoundError(f"States file not found: {states_file}")
        states = np.load(states_file).astype("float32")
        if states.ndim != 2 or states.shape[1] != input_dim:
            raise ValueError(
                f"States shape mismatch: expected (N, {input_dim}), got {states.shape}"
            )
        return states
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_states, input_dim)).astype("float32")


def _resolve(pair: str, explicit: str, directory: str, template: str) -> str:
    if explicit:
        return explicit
    if directory:
        return os.path.join(directory, template.format(pair=pair))
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
    print(f"Float latency (ms)  : {latency.get('float_inference_ms', 0):.4f}")
    print(f"Quant latency (ms)  : {latency.get('quantized_inference_ms', 0):.4f}")
    print(f"Latency ratio       : {latency.get('latency_ratio', 0):.4f}")
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
                   help="Timed single-sample forward passes (median reported).")

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

    # Device
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # Output
    p.add_argument("--report-dir", default="reports/quantization_validation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    states = _load_states(args.states_file, args.n_states, args.input_dim, args.seed)
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

        float_model = _load_float_student(
            float_ckpt, args.input_dim, args.output_dim, args.parent_hidden
        )
        q_model, meta = load_quantized_checkpoint(quant_ckpt, device=device)

        validator = QuantizedValidator(float_model, q_model, thresholds=thresholds, device=device)
        report = validator.validate(
            states,
            float_checkpoint_path=float_ckpt,
            quantized_checkpoint_path=quant_ckpt,
            quantization_metadata=meta,
            n_latency_warmup=args.latency_warmup,
            n_latency_repeats=args.latency_repeats,
        )

        report_dict = report.to_dict()
        report_dict["pair"] = pair
        report_dict["checkpoints"] = {
            "float": float_ckpt,
            "quantized": quant_ckpt,
        }
        report_dict["states"] = {
            "count": int(states.shape[0]),
            "input_dim": int(states.shape[1]),
            "source": args.states_file if args.states_file else "synthetic_standard_normal",
            "seed": args.seed,
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
