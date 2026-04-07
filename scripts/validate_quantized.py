#!/usr/bin/env python
"""Validate quantized student checkpoints against float references.

This script compares the outputs of quantized student models
(produced by ``scripts/quantize_distilled.py``) against their float
counterparts, reporting Q-value error, action agreement, and cosine
similarity.  It mirrors the structure of ``scripts/validate_distillation.py``
and is intended to be run after the quantization step.

Usage
-----
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

Calibration / state distribution
---------------------------------
The comparison is meaningful only when *states* come from the same distribution
as those used for distillation.  Pass ``--states-file`` for real replay data or
rely on the synthetic default for quick sanity checks.

Output
------
Prints a table of metrics and writes one JSON report per pair under
``--report-dir``.
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


def _print_report(pair: str, cmp: dict, meta: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"Quantization validation report: student_{pair}")
    print(sep)
    quant_meta = meta.get("quantization", {})
    print(f"Mode            : {quant_meta.get('mode', 'unknown')}")
    print(f"Dtype           : {quant_meta.get('dtype', 'unknown')}")
    print(f"Backend         : {quant_meta.get('backend', 'unknown')}")
    print(f"Action agreement: {cmp['action_agreement']*100:.2f}%")
    print(f"Mean Q-error    : {cmp['mean_q_error']:.6f}")
    print(f"Max Q-error     : {cmp['max_q_error']:.6f}")
    print(f"Cosine similarity: {cmp['mean_cosine_similarity']:.6f}")
    print(f"States evaluated: {cmp['n_states']}")


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

    # Output
    p.add_argument("--report-dir", default="reports/quantization_validation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

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
        q_model, meta = load_quantized_checkpoint(quant_ckpt)

        cmp = compare_outputs(float_model, q_model, states)
        _print_report(pair, cmp, meta)

        report = {
            "pair": pair,
            "float_checkpoint": float_ckpt,
            "quantized_checkpoint": quant_ckpt,
            "comparison": cmp,
            "quantization_metadata": meta,
        }
        out_path = os.path.join(args.report_dir, f"quantization_validation_{pair}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, allow_nan=False)
        print(f"JSON report written: {out_path}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
