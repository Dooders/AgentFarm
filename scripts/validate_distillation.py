#!/usr/bin/env python3
"""Validate teacher-student distillation quality from saved checkpoints.

This script evaluates current distillation quality for parent/student pairs
using the behavioural-fidelity checks implemented by ``StudentValidator``:

- action agreement (top-1 and top-k)
- KL/MSE/MAE/cosine output similarity
- parameter ratio (student vs parent size)
- inference latency ratio
- optional robustness slices

It is intended as a "current state" checkpoint before further work on
distillation + recombination experiments (see GitHub Issue #8).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

# Allow running directly from repo root without requiring pip install -e .
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.trainer_distill import (  # noqa: E402
    StudentValidator,
    ValidationThresholds,
)


def _parse_k_values(raw: str) -> List[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("k-values cannot be empty")
    parsed = [int(v) for v in values]
    invalid = [v for v in parsed if v <= 0]
    if invalid:
        raise ValueError(f"k-values must be positive integers, got: {invalid}")
    return parsed


def _resolve_checkpoint_path(
    pair: str,
    explicit_path: str,
    checkpoint_dir: str,
    filename_template: str,
) -> str:
    if explicit_path:
        return explicit_path
    if checkpoint_dir:
        return os.path.join(checkpoint_dir, filename_template.format(pair=pair))
    return ""


def _require_file(path: str, label: str) -> None:
    if not path:
        raise ValueError(
            f"Missing {label} path. Set an explicit checkpoint path or provide --checkpoint-dir."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found at: {path}")


def _load_states(states_file: str, n_states: int, input_dim: int, seed: Optional[int]) -> np.ndarray:
    if states_file:
        if not os.path.isfile(states_file):
            raise FileNotFoundError(f"States file not found: {states_file}")
        states = np.load(states_file).astype("float32")
        if states.ndim != 2:
            raise ValueError(
                f"States must be a 2D array with shape (N, input_dim), got {states.shape!r}"
            )
        if states.shape[1] != input_dim:
            raise ValueError(
                f"States input_dim mismatch: expected {input_dim}, got {states.shape[1]}"
            )
        return states

    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_states, input_dim)).astype("float32")


def _build_robustness_slices(
    base_states: np.ndarray,
    seed: Optional[int],
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Low-resource: lower signal magnitude.
    low_resource = (base_states * 0.5).astype("float32")

    # High-threat: noisy perturbation.
    high_threat = (base_states + 0.25 * rng.standard_normal(base_states.shape)).astype(
        "float32"
    )

    # Sparse observations: random feature drop.
    mask = (rng.random(base_states.shape) > 0.5).astype(np.float32)
    sparse_obs = (base_states * mask).astype("float32")

    return {
        "low_resource": low_resource,
        "high_threat": high_threat,
        "sparse_obs": sparse_obs,
    }


def _load_parent_student(
    *,
    input_dim: int,
    output_dim: int,
    parent_hidden: int,
    parent_ckpt: str,
    student_ckpt: str,
    device: torch.device,
) -> tuple[BaseQNetwork, StudentQNetwork]:
    parent = BaseQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=parent_hidden,
    ).to(device)
    student = StudentQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        parent_hidden_size=parent_hidden,
    ).to(device)

    parent_state = torch.load(parent_ckpt, map_location=device, weights_only=True)
    student_state = torch.load(student_ckpt, map_location=device, weights_only=True)
    if not isinstance(parent_state, dict):
        raise ValueError(
            f"Parent checkpoint at {parent_ckpt!r} does not contain a state dict."
        )
    if not isinstance(student_state, dict):
        raise ValueError(
            f"Student checkpoint at {student_ckpt!r} does not contain a state dict."
        )

    parent.load_state_dict(parent_state)
    student.load_state_dict(student_state)
    parent.eval()
    student.eval()
    return parent, student


def _print_report(pair: str, report: Dict[str, object]) -> None:
    top_k = report["top_k_agreements"]
    print(f"\n{'=' * 72}")
    print(f"Distillation validation report: pair {pair}")
    print(f"{'=' * 72}")
    print(f"Passed thresholds    : {report['passed']}")
    print(f"Action agreement     : {report['action_agreement']:.4f}")
    print(f"Top-k agreements     : {top_k}")
    print(f"KL divergence        : {report['kl_divergence']:.6f}")
    print(f"MSE / MAE            : {report['mse']:.6f} / {report['mae']:.6f}")
    print(f"Cosine similarity    : {report['mean_cosine_similarity']:.6f}")
    print(
        "Param ratio          : "
        f"{report['student_param_count']}/{report['parent_param_count']} "
        f"(ratio={report['param_ratio']:.4f})"
    )
    print(
        "Latency (ms)         : "
        f"parent={report['parent_inference_ms']:.4f}, "
        f"student={report['student_inference_ms']:.4f}, "
        f"ratio={report['latency_ratio']:.4f}"
    )
    if report["robustness_slice_agreements"]:
        print(f"Robustness slices    : {report['robustness_slice_agreements']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate current teacher-student distillation quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pair", choices=["A", "B", "both"], default="both")
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help=(
            "Optional directory for implicit checkpoint resolution. If set, missing "
            "paths use parent_{pair}.pt and student_{pair}.pt within this directory."
        ),
    )
    parser.add_argument("--parent-a-ckpt", default="")
    parser.add_argument("--student-a-ckpt", default="")
    parser.add_argument("--parent-b-ckpt", default="")
    parser.add_argument("--student-b-ckpt", default="")

    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--parent-hidden", type=int, default=64)
    parser.add_argument("--states-file", default="")
    parser.add_argument("--n-states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--k-values",
        default="1,2,3",
        help="Comma-separated top-k values for agreement metrics.",
    )
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument(
        "--disable-robustness-slices",
        action="store_true",
        help="Skip synthetic robustness slice checks.",
    )

    # Threshold overrides
    parser.add_argument("--min-action-agreement", type=float, default=0.85)
    parser.add_argument("--max-kl-divergence", type=float, default=0.5)
    parser.add_argument("--max-mse", type=float, default=2.0)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.8)
    parser.add_argument("--max-param-ratio", type=float, default=0.9)

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--report-dir",
        default="reports/distillation_validation",
        help="Directory to write JSON reports (one per pair).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    k_values = _parse_k_values(args.k_values)
    states = _load_states(
        states_file=args.states_file,
        n_states=args.n_states,
        input_dim=args.input_dim,
        seed=args.seed,
    )

    thresholds = ValidationThresholds(
        min_action_agreement=args.min_action_agreement,
        max_kl_divergence=args.max_kl_divergence,
        max_mse=args.max_mse,
        min_cosine_similarity=args.min_cosine_similarity,
        max_param_ratio=args.max_param_ratio,
    )

    robustness_slices = None
    if not args.disable_robustness_slices:
        robustness_slices = _build_robustness_slices(states, seed=args.seed)

    pair_to_paths = {
        "A": (
            _resolve_checkpoint_path(
                "A",
                args.parent_a_ckpt,
                args.checkpoint_dir,
                "parent_{pair}.pt",
            ),
            _resolve_checkpoint_path(
                "A",
                args.student_a_ckpt,
                args.checkpoint_dir,
                "student_{pair}.pt",
            ),
        ),
        "B": (
            _resolve_checkpoint_path(
                "B",
                args.parent_b_ckpt,
                args.checkpoint_dir,
                "parent_{pair}.pt",
            ),
            _resolve_checkpoint_path(
                "B",
                args.student_b_ckpt,
                args.checkpoint_dir,
                "student_{pair}.pt",
            ),
        ),
    }
    pairs = ["A", "B"] if args.pair == "both" else [args.pair]

    os.makedirs(args.report_dir, exist_ok=True)

    for pair in pairs:
        parent_ckpt, student_ckpt = pair_to_paths[pair]
        _require_file(parent_ckpt, f"parent_{pair} checkpoint")
        _require_file(student_ckpt, f"student_{pair} checkpoint")

        parent, student = _load_parent_student(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            parent_hidden=args.parent_hidden,
            parent_ckpt=parent_ckpt,
            student_ckpt=student_ckpt,
            device=device,
        )
        validator = StudentValidator(parent, student, thresholds=thresholds, device=device)
        report = validator.validate(
            states=states,
            robustness_slices=robustness_slices,
            k_values=k_values,
            n_latency_warmup=args.latency_warmup,
            n_latency_repeats=args.latency_repeats,
        )
        report_dict = report.to_dict()
        report_dict["pair"] = pair
        report_dict["checkpoints"] = {
            "parent": parent_ckpt,
            "student": student_ckpt,
        }
        report_dict["states"] = {
            "count": int(states.shape[0]),
            "input_dim": int(states.shape[1]),
            "source": args.states_file if args.states_file else "synthetic_standard_normal",
        }

        _print_report(pair, report_dict)
        out_path = os.path.join(args.report_dir, f"distillation_validation_{pair}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2, allow_nan=False)
        print(f"JSON report written  : {out_path}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
