#!/usr/bin/env python
"""Quantise distilled student checkpoints to 8-bit precision.

This script reads ``student_<pair>.pt`` checkpoints produced by
``scripts/run_distillation.py`` and writes ``student_<pair>_int8.pt``
quantized counterparts (plus companion JSON metadata files) using
post-training quantization (PTQ).

Usage
-----
Quantise both students with dynamic (weight-only) int8 PTQ::

    python scripts/quantize_distilled.py \\
        --checkpoint-dir checkpoints/distillation \\
        --output-dir checkpoints/quantized

Quantise only student A::

    python scripts/quantize_distilled.py \\
        --pair A \\
        --student-a-ckpt checkpoints/distillation/student_A.pt \\
        --output-dir checkpoints/quantized

Static quantization with calibration data::

    python scripts/quantize_distilled.py \\
        --mode static \\
        --states-file data/replay_states.npy \\
        --checkpoint-dir checkpoints/distillation \\
        --output-dir checkpoints/quantized

Architecture flags
------------------
The script must know the network architecture to reconstruct the float model
for validation.  Pass ``--input-dim``, ``--output-dim``, and
``--parent-hidden`` to match the values used in ``run_distillation.py``
(defaults: 8, 4, 64).

Calibration
-----------
When ``--mode static``, the script uses ``--states-file`` (a ``.npy`` file of
shape ``(N, input_dim)``) or synthesises ``--n-states`` random normal states
with ``--seed`` for calibration.  Batch sizes and counts are controlled by
``--calibration-batches`` and ``--calibration-batch-size``.

Quantization hyperparameters
-----------------------------
======================  ============================================
``--mode``              ``dynamic`` (default) or ``static``
``--dtype``             ``qint8`` (default) or ``quint8``
``--backend``           ``qnnpack`` (default) or ``fbgemm``
``--calibration-batches``  Number of calibration batches (static only)
``--calibration-batch-size``  Batch size for calibration (static only)
======================  ============================================

Output
------
For each processed pair the script writes:

* ``<output_dir>/student_<pair>_int8.pt``  – quantized model pickle
* ``<output_dir>/student_<pair>_int8.pt.json``  – JSON metadata

A final comparison report (action agreement, Q-error) is printed to stdout
and written to ``<output_dir>/quantization_report_<pair>.json``.
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

from farm.core.decision.base_dqn import StudentQNetwork  # noqa: E402
from farm.core.decision.training.quantize_ptq import (  # noqa: E402
    PostTrainingQuantizer,
    QuantizationConfig,
    compare_outputs,
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
    """Load a float StudentQNetwork from a state-dict checkpoint."""
    model = StudentQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        parent_hidden_size=parent_hidden,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at '{path}' must be a state dict (got {type(state).__name__}). "
            "Use the checkpoint produced by run_distillation.py."
        )
    model.load_state_dict(state)
    model.eval()
    return model


def _load_states(
    states_file: str,
    n_states: int,
    input_dim: int,
    seed: int,
) -> np.ndarray:
    if states_file and os.path.isfile(states_file):
        states = np.load(states_file).astype("float32")
        print(f"  Loaded states from {states_file}: shape={states.shape}")
        return states
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n_states, input_dim)).astype("float32")
    print(f"  Using {n_states} synthetic random states (shape={states.shape})")
    return states


def _run_pair(
    pair: str,
    student_ckpt: str,
    input_dim: int,
    output_dim: int,
    parent_hidden: int,
    config: QuantizationConfig,
    calibration_states: np.ndarray,
    output_dir: str,
) -> None:
    """Quantise one student checkpoint and write outputs."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Quantising: student_{pair}  →  student_{pair}_int8")
    print(f"{sep}")

    # Load float model
    print(f"  Loading float checkpoint: {student_ckpt}")
    float_model = _load_float_student(student_ckpt, input_dim, output_dim, parent_hidden)
    float_params = sum(p.numel() for p in float_model.parameters())
    print(f"  Float model params: {float_params:,}")

    # Quantise
    quantizer = PostTrainingQuantizer(config)
    cal_states = calibration_states if config.mode == "static" else None
    print(f"  Applying {config.mode} PTQ (dtype={config.dtype}, backend={config.backend}) …")
    q_model, result = quantizer.quantize(float_model, calibration_states=cal_states)

    # Save
    out_path = os.path.join(output_dir, f"student_{pair}_int8.pt")
    os.makedirs(output_dir, exist_ok=True)
    arch_kwargs = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "parent_hidden_size": parent_hidden,
    }
    quantizer.save_checkpoint(q_model, out_path, result, arch_kwargs=arch_kwargs)
    print(f"  Quantized checkpoint: {out_path}")
    print(f"  Metadata JSON:        {out_path}.json")

    # Report quantization stats
    print(f"\n  --- Quantization summary for student_{pair} ---")
    print(f"  Linear layers quantized : {result.linear_layers_quantized}")
    print(
        f"  Float weight bytes      : {result.float_param_bytes:,}"
        f"  →  int8 weight bytes: {result.quantized_param_bytes:,}"
        f"  (≈{100*result.quantized_param_bytes/max(1,result.float_param_bytes):.0f}%)"
    )
    print(f"  Elapsed (s)             : {result.elapsed_seconds:.3f}")

    # Compare outputs
    print("\n  Comparing float vs quantized outputs …")
    cmp = compare_outputs(float_model, q_model, calibration_states)
    print(f"  Action agreement        : {cmp['action_agreement']*100:.2f}%")
    print(f"  Mean Q-error            : {cmp['mean_q_error']:.6f}")
    print(f"  Max Q-error             : {cmp['max_q_error']:.6f}")
    print(f"  Mean cosine similarity  : {cmp['mean_cosine_similarity']:.6f}")

    # Write comparison JSON
    report = {
        "pair": pair,
        "float_checkpoint": student_ckpt,
        "quantized_checkpoint": out_path,
        "quantization": result.to_dict(),
        "comparison": cmp,
    }
    report_path = os.path.join(output_dir, f"quantization_report_{pair}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    print(f"\n  Full report written: {report_path}")

    # Verify round-trip load
    print("  Verifying checkpoint round-trip load …")
    q_model_rt, _meta = load_quantized_checkpoint(out_path)
    cmp_rt = compare_outputs(float_model, q_model_rt, calibration_states[:64])
    print(f"  Round-trip action agreement: {cmp_rt['action_agreement']*100:.2f}%  ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Post-training quantization of distilled student checkpoints. "
            "Reads student_<pair>.pt, writes student_<pair>_int8.pt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pair", choices=["A", "B", "both"], default="both")

    # Checkpoints (explicit or from directory)
    p.add_argument(
        "--checkpoint-dir",
        default="",
        help="Directory containing student_A.pt / student_B.pt.",
    )
    p.add_argument("--student-a-ckpt", default="", help="Explicit path to student_A.pt.")
    p.add_argument("--student-b-ckpt", default="", help="Explicit path to student_B.pt.")

    # Architecture
    p.add_argument("--input-dim", type=int, default=8, help="State feature dimension.")
    p.add_argument("--output-dim", type=int, default=4, help="Number of actions.")
    p.add_argument("--parent-hidden", type=int, default=64, help="Teacher hidden layer width.")

    # States / calibration
    p.add_argument("--states-file", default="", help="Path to .npy calibration state file.")
    p.add_argument("--n-states", type=int, default=1000, help="Synthetic states if no file.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic states.")

    # Quantization hyperparameters
    p.add_argument(
        "--mode",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode: 'dynamic' (weight-only, no calibration) or 'static' (activation-aware).",
    )
    p.add_argument("--dtype", choices=["qint8", "quint8"], default="qint8")
    p.add_argument(
        "--backend",
        choices=["qnnpack", "fbgemm", "none"],
        default="qnnpack",
        help="Quantization backend (static mode only).",
    )
    p.add_argument("--calibration-batches", type=int, default=10)
    p.add_argument("--calibration-batch-size", type=int, default=64)

    # Output
    p.add_argument(
        "--output-dir",
        default="checkpoints/quantized",
        help="Directory to write quantized checkpoints and reports.",
    )
    return p.parse_args()


def _resolve_student_ckpt(pair: str, explicit: str, checkpoint_dir: str) -> str:
    if explicit:
        return explicit
    if checkpoint_dir:
        return os.path.join(checkpoint_dir, f"student_{pair}.pt")
    return f"student_{pair}.pt"


def main() -> None:
    args = _parse_args()

    config = QuantizationConfig(
        mode=args.mode,
        dtype=args.dtype,
        backend=args.backend,
        calibration_batches=args.calibration_batches,
        calibration_batch_size=args.calibration_batch_size,
    )

    # Load / generate states (used for calibration and comparison)
    states = _load_states(args.states_file, args.n_states, args.input_dim, args.seed)

    pairs = ["A", "B"] if args.pair == "both" else [args.pair]
    ckpt_map = {
        "A": _resolve_student_ckpt("A", args.student_a_ckpt, args.checkpoint_dir),
        "B": _resolve_student_ckpt("B", args.student_b_ckpt, args.checkpoint_dir),
    }

    for pair in pairs:
        _run_pair(
            pair=pair,
            student_ckpt=ckpt_map[pair],
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            parent_hidden=args.parent_hidden,
            config=config,
            calibration_states=states,
            output_dir=args.output_dir,
        )

    print("\nQuantization complete.")


if __name__ == "__main__":
    main()
