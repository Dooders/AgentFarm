#!/usr/bin/env python3
"""Evaluate generalization of a crossover child on holdout and shifted-domain data.

This script wraps :class:`~farm.core.decision.training.RecombinationEvaluator` and
runs it on up to **three** state sets produced from a single replay buffer:

1. **In-distribution (ID)** – the training split (``1 - holdout_fraction``).
2. **Holdout** – the held-out test split (``holdout_fraction``).
3. **Shifted** *(optional)* – the holdout set after applying a domain-shift
   perturbation (Gaussian noise or input scaling).

JSON reports for each set are written to ``--report-dir``.  A combined
``generalization_summary.json`` is also emitted with top-level pass/fail and
a side-by-side metric comparison across sets.

How to run
----------
::

    # Minimal – use synthetic states, 20 % holdout, no shift
    python scripts/eval_generalization.py \\
        --parent-a-ckpt checkpoints/parent_A.pt \\
        --parent-b-ckpt checkpoints/parent_B.pt \\
        --child-ckpt    checkpoints/child.pt \\
        --n-states 2000 --seed 42 \\
        --report-dir    reports/generalization

    # With a real replay buffer and domain-shift evaluation
    python scripts/eval_generalization.py \\
        --parent-a-ckpt checkpoints/parent_A.pt \\
        --parent-b-ckpt checkpoints/parent_B.pt \\
        --child-ckpt    checkpoints/child.pt \\
        --states-file   data/replay_states.npy \\
        --holdout-fraction 0.2 \\
        --shift-type    gaussian_noise --shift-std 0.1 \\
        --report-dir    reports/generalization

    # With input-scaling shift
    python scripts/eval_generalization.py \\
        --parent-a-ckpt checkpoints/parent_A.pt \\
        --parent-b-ckpt checkpoints/parent_B.pt \\
        --child-ckpt    checkpoints/child.pt \\
        --states-file   data/replay_states.npy \\
        --shift-type    input_scaling --shift-scale-factor 2.0 \\
        --report-dir    reports/generalization

Output files
------------
``<report-dir>/id_report.json``
    RecombinationReport on the in-distribution split.
``<report-dir>/holdout_report.json``
    RecombinationReport on the held-out split.
``<report-dir>/shifted_report.json`` *(optional)*
    RecombinationReport on the shifted holdout set.
``<report-dir>/generalization_summary.json``
    Top-level summary comparing all evaluated sets.
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

from farm.core.decision.base_dqn import BaseQNetwork  # noqa: E402
from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_distillation_states,
)
from farm.core.decision.training.holdout_utils import (  # noqa: E402
    SHIFT_TYPES,
    apply_gaussian_noise,
    apply_input_scaling,
    split_replay_buffer,
)
from farm.core.decision.training.recombination_eval import (  # noqa: E402
    RecombinationEvaluator,
    RecombinationThresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(
    path: str,
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    label: str,
) -> BaseQNetwork:
    """Load a BaseQNetwork from a state-dict checkpoint."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{label} checkpoint not found: {path!r}")
    model = BaseQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"{label} checkpoint at {path!r} does not contain a state dict; "
            f"got {type(state).__name__}."
        )
    model.load_state_dict(state)
    model.eval()
    return model


def _parse_k_values(raw: str) -> List[int]:
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    if not parts:
        raise ValueError("--k-values cannot be empty.")
    parsed = [int(v) for v in parts]
    invalid = [v for v in parsed if v <= 0]
    if invalid:
        raise ValueError(f"k-values must be positive integers; got {invalid}.")
    return parsed


def _run_eval(
    evaluator: RecombinationEvaluator,
    states: np.ndarray,
    *,
    states_source: str,
    k_values: List[int],
    latency_warmup: int,
    latency_repeats: int,
    include_parent_baseline: bool,
    eval_batch_size: Optional[int],
    model_paths: Optional[Dict[str, Optional[str]]],
) -> dict:
    report = evaluator.evaluate(
        states,
        include_parent_baseline=include_parent_baseline,
        k_values=k_values,
        n_latency_warmup=latency_warmup,
        n_latency_repeats=latency_repeats,
        states_source=states_source,
        eval_batch_size=eval_batch_size,
        model_paths=model_paths,
    )
    return report.to_dict()


def _extract_summary_metrics(report: dict) -> dict:
    """Pull top-level agreement numbers from a RecombinationReport dict."""
    summary = report.get("summary", {})
    return {
        "child_agrees_with_parent_a": summary.get("child_agrees_with_parent_a"),
        "child_agrees_with_parent_b": summary.get("child_agrees_with_parent_b"),
        "oracle_agreement": summary.get("oracle_agreement"),
        "n_states": report.get("states", {}).get("n_states"),
        "passed": report.get("passed"),
    }


def _print_set_summary(label: str, metrics: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    n_states                 : {metrics.get('n_states')}")
    parent_a = metrics.get("child_agrees_with_parent_a")
    if parent_a is not None:
        print(f"    child ↔ parent A         : {parent_a:.4f}")
    else:
        print("    child ↔ parent A         : N/A")
    parent_b = metrics.get("child_agrees_with_parent_b")
    if parent_b is not None:
        print(f"    child ↔ parent B         : {parent_b:.4f}")
    else:
        print("    child ↔ parent B         : N/A")
    oracle = metrics.get("oracle_agreement")
    if oracle is not None:
        print(f"    oracle agreement         : {oracle:.4f}")
    print(f"    passed                   : {metrics.get('passed')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate child vs parents on in-distribution, holdout, "
            "and optionally domain-shifted state sets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint paths
    p.add_argument("--parent-a-ckpt", default="", help="Path to parent A state-dict checkpoint.")
    p.add_argument("--parent-b-ckpt", default="", help="Path to parent B state-dict checkpoint.")
    p.add_argument("--child-ckpt", default="", help="Path to child state-dict checkpoint.")

    # Architecture
    p.add_argument("--input-dim", type=int, default=8, help="Network input feature dimension.")
    p.add_argument("--output-dim", type=int, default=4, help="Number of output actions.")
    p.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size.")

    # States
    p.add_argument("--states-file", default="", help="Path to .npy states file (N, input_dim).")
    p.add_argument(
        "--n-states",
        type=int,
        default=2000,
        help="Synthetic state count when --states-file is absent.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic states.")

    # Holdout split
    p.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of states reserved for holdout evaluation (0 < f < 1).",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help=(
            "Disable shuffling before the train/holdout split.  "
            "Use only when rows are already in a random order and temporal "
            "adjacency within each split is desired."
        ),
    )

    # Domain-shift options
    p.add_argument(
        "--shift-type",
        choices=list(SHIFT_TYPES),
        default=None,
        help=(
            "Optional domain-shift perturbation applied to the holdout set.  "
            "Choose 'gaussian_noise' or 'input_scaling'.  "
            "When absent, no shifted evaluation is run."
        ),
    )
    p.add_argument(
        "--shift-std",
        type=float,
        default=0.1,
        help="Standard deviation of Gaussian noise (used when --shift-type gaussian_noise).",
    )
    p.add_argument(
        "--shift-scale-factor",
        type=float,
        default=2.0,
        help="Multiplicative scale factor (used when --shift-type input_scaling).",
    )
    p.add_argument(
        "--shift-seed",
        type=int,
        default=0,
        help="RNG seed for the noise perturbation.",
    )

    # Top-k
    p.add_argument("--k-values", default="1,2,3", help="Comma-separated top-k values.")

    # Latency
    p.add_argument("--latency-warmup", type=int, default=5)
    p.add_argument("--latency-repeats", type=int, default=50)

    # Options
    p.add_argument(
        "--include-parent-baseline",
        action="store_true",
        help="Also compute parent A vs parent B comparison.",
    )

    # Thresholds
    p.add_argument("--min-action-agreement", type=float, default=0.70)
    p.add_argument("--max-kl-divergence", type=float, default=1.0)
    p.add_argument("--max-mse", type=float, default=5.0)
    p.add_argument("--min-cosine-similarity", type=float, default=0.70)
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Emit reports without applying pass/fail thresholds.",
    )

    # Output
    p.add_argument(
        "--report-dir",
        default="reports/generalization",
        help="Directory to write JSON reports.",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Max states per forward pass (0 = single batch).",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.parent_a_ckpt or not args.parent_b_ckpt or not args.child_ckpt:
        raise ValueError(
            "All three checkpoint paths are required: "
            "--parent-a-ckpt, --parent-b-ckpt, --child-ckpt."
        )

    k_values = _parse_k_values(args.k_values)
    eval_bs = args.eval_batch_size if args.eval_batch_size > 0 else None

    # Load all states (ID + holdout combined).
    all_states = load_distillation_states(
        args.states_file, args.n_states, args.input_dim, args.seed
    )
    full_source = args.states_file if args.states_file else "synthetic_standard_normal"

    # Split into in-distribution and holdout.
    id_states, holdout_states = split_replay_buffer(
        all_states,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    print(
        f"\nSplit: {all_states.shape[0]} total → "
        f"{id_states.shape[0]} ID + {holdout_states.shape[0]} holdout"
        f" (holdout_fraction={args.holdout_fraction})"
    )

    # Build optional shifted set.
    shifted_states: Optional[np.ndarray] = None
    shift_label = ""
    if args.shift_type == "gaussian_noise":
        shifted_states = apply_gaussian_noise(holdout_states, std=args.shift_std, seed=args.shift_seed)
        shift_label = f"gaussian_noise(std={args.shift_std})"
    elif args.shift_type == "input_scaling":
        shifted_states = apply_input_scaling(holdout_states, scale_factor=args.shift_scale_factor)
        shift_label = f"input_scaling(scale={args.shift_scale_factor})"

    # Load models.
    parent_a = _load_model(
        args.parent_a_ckpt, args.input_dim, args.output_dim, args.hidden_size, "parent_a"
    )
    parent_b = _load_model(
        args.parent_b_ckpt, args.input_dim, args.output_dim, args.hidden_size, "parent_b"
    )
    child = _load_model(
        args.child_ckpt, args.input_dim, args.output_dim, args.hidden_size, "child"
    )

    thresholds = RecombinationThresholds(
        min_action_agreement=args.min_action_agreement,
        max_kl_divergence=args.max_kl_divergence,
        max_mse=args.max_mse,
        min_cosine_similarity=args.min_cosine_similarity,
        report_only=args.report_only,
    )

    evaluator = RecombinationEvaluator(
        parent_a,
        parent_b,
        child,
        thresholds=thresholds,
        device=torch.device("cpu"),
    )

    model_paths = {
        "parent_a": args.parent_a_ckpt,
        "parent_b": args.parent_b_ckpt,
        "child": args.child_ckpt,
    }

    common_kwargs: dict = dict(
        k_values=k_values,
        latency_warmup=args.latency_warmup,
        latency_repeats=args.latency_repeats,
        include_parent_baseline=args.include_parent_baseline,
        eval_batch_size=eval_bs,
        model_paths=model_paths,
    )

    # --- In-distribution evaluation ---
    print("\nEvaluating in-distribution (ID) split …")
    id_report = _run_eval(
        evaluator,
        id_states,
        states_source=f"{full_source}[id_split]",
        **common_kwargs,
    )

    # --- Holdout evaluation ---
    print("Evaluating holdout split …")
    holdout_report = _run_eval(
        evaluator,
        holdout_states,
        states_source=f"{full_source}[holdout_split]",
        **common_kwargs,
    )

    # --- Shifted evaluation (optional) ---
    shifted_report: Optional[dict] = None
    if shifted_states is not None:
        print(f"Evaluating shifted holdout ({shift_label}) …")
        shifted_report = _run_eval(
            evaluator,
            shifted_states,
            states_source=f"{full_source}[{shift_label}]",
            **common_kwargs,
        )

    # --- Build summary ---
    sets: dict = {
        "in_distribution": _extract_summary_metrics(id_report),
        "holdout": _extract_summary_metrics(holdout_report),
    }
    if shifted_report is not None:
        sets["shifted"] = _extract_summary_metrics(shifted_report)
        sets["shifted"]["shift_type"] = args.shift_type

    overall_passed = (
        bool(id_report.get("passed", True))
        and bool(holdout_report.get("passed", True))
        and (shifted_report is None or bool(shifted_report.get("passed", True)))
    ) or args.report_only

    summary: dict = {
        "overall_passed": overall_passed,
        "report_only": args.report_only,
        "holdout_fraction": args.holdout_fraction,
        "shift_type": args.shift_type,
        "sets": sets,
    }

    # --- Print ---
    sep = "=" * 72
    print(f"\n{sep}")
    print("Generalization evaluation summary")
    print(sep)
    for set_name, metrics in sets.items():
        _print_set_summary(set_name, metrics)
    print(f"\nOverall passed: {overall_passed}")

    # --- Write reports ---
    os.makedirs(args.report_dir, exist_ok=True)

    def _write(data: dict, filename: str) -> None:
        path = os.path.join(args.report_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, allow_nan=False)
        print(f"  Written: {path}")

    print("\nWriting reports …")
    _write(id_report, "id_report.json")
    _write(holdout_report, "holdout_report.json")
    if shifted_report is not None:
        _write(shifted_report, "shifted_report.json")
    _write(summary, "generalization_summary.json")

    print("\nDone.")
    if not overall_passed and not args.report_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
