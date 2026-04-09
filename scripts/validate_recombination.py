#!/usr/bin/env python3
"""Evaluate a crossover child Q-network against its parent models.

This script compares a **child** network (produced by crossover and optionally
fine-tuned) against **parent A** and **parent B** using the same offline
Q-network metrics as ``validate_distillation.py`` and
``validate_quantized.py``.

For each pair (child vs parent A) and (child vs parent B), the following
metrics are computed:

- **Top-1 / top-k action agreement**: fraction of states where both models
  choose the same action.
- **Output similarity**: KL divergence, MSE, MAE, and cosine similarity on
  Q-value logits.
- **Oracle agreement**: fraction of states where the child agrees with *at
  least one* parent.

Optionally, a **parent A vs parent B** baseline comparison is included to
show the diversity gap between the two parents.

How to run
----------
::

    # Evaluate child against both parents (checkpoints inferred from directory)
    python scripts/validate_recombination.py \\
        --checkpoint-dir checkpoints/crossover \\
        --report-dir     reports/recombination

    # Explicit paths
    python scripts/validate_recombination.py \\
        --parent-a-ckpt  checkpoints/parent_A.pt \\
        --parent-b-ckpt  checkpoints/parent_B.pt \\
        --child-ckpt     checkpoints/child.pt \\
        --states-file    data/replay_states.npy \\
        --report-dir     reports/recombination

    # Include parent-vs-parent baseline and use report-only mode
    python scripts/validate_recombination.py \\
        --checkpoint-dir checkpoints/crossover \\
        --include-parent-baseline \\
        --report-only

Architecture flags (``--input-dim``, ``--output-dim``, ``--hidden-size``)
must match the values used when the checkpoints were trained.  The defaults
(8, 4, 64) are the standard AgentFarm experiment dimensions.

Inputs
------
- ``--parent-a-ckpt`` / ``--parent-b-ckpt``: parent state-dict checkpoints
  (``torch.save(model.state_dict(), …)``).
- ``--child-ckpt``: child state-dict checkpoint.
- ``--checkpoint-dir``: optional directory for implicit path resolution.
  When set, missing paths default to
  ``parent_A.pt``, ``parent_B.pt``, and ``child.pt`` within this directory.
- ``--states-file``: optional NumPy ``.npy`` array of shape
  ``(N, input_dim)`` with ``dtype=float32``.  When absent, a synthetic
  standard-normal dataset is generated from ``--seed``.

Output
------
A JSON report is written to
``<report-dir>/recombination_validation.json`` with the following top-level
keys:

``schema_version``
    Versioned schema identifier for forward-compatible parsing.
``torch_version``
    PyTorch version at evaluation time.
``states``
    State count, feature dimensionality, and source description.
``model_paths``
    Checkpoint paths for all three models.
``comparisons``
    Per-pair metrics (``child_vs_parent_a``, ``child_vs_parent_b``,
    optionally ``parent_a_vs_parent_b``).
``summary``
    Convenience top-level agreement rates and oracle agreement.
``thresholds``
    The configured pass/fail thresholds.
``passed``
    ``True`` when all threshold-checked comparisons pass (or
    ``--report-only`` is set).

Interpreting the report
-----------------------
``child_agrees_with_parent_a`` and ``child_agrees_with_parent_b``
    Both should meet or exceed ``--min-action-agreement`` for the child to
    be considered a well-blended offspring.
``oracle_agreement``
    Fraction of states where the child matches *at least one* parent.  High
    oracle agreement alongside low individual parent agreements indicates the
    child has learned a genuinely *blended* policy.
``parent_a_vs_parent_b.action_agreement`` (optional baseline)
    Reveals the inter-parent diversity.  A low baseline means the parents
    themselves are very different, so the child is navigating a wide search
    space.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

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
from farm.core.decision.training.recombination_eval import (  # noqa: E402
    RecombinationEvaluator,
    RecombinationThresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(explicit: str, directory: str, filename: str) -> str:
    """Return explicit path if set, else join directory + filename."""
    if explicit:
        return explicit
    if directory:
        return os.path.join(directory, filename)
    return ""


def _require_file(path: str, label: str) -> None:
    if not path:
        raise ValueError(
            f"Missing {label} path.  Provide an explicit checkpoint flag or --checkpoint-dir."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found at: {path}")


def _load_model(
    path: str,
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    label: str,
) -> BaseQNetwork:
    """Load a BaseQNetwork from a state-dict checkpoint."""
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


def _print_comparison(label: str, cmp: dict) -> None:
    top_k = cmp.get("top_k_agreements", {})
    print(f"\n  --- {label} ---")
    print(f"  Action agreement      : {cmp['action_agreement']:.4f}")
    print(f"  Top-k agreements      : {top_k}")
    print(f"  KL divergence         : {cmp['kl_divergence']:.6f}")
    print(f"  MSE / MAE             : {cmp['mse']:.6f} / {cmp['mae']:.6f}")
    print(f"  Cosine similarity     : {cmp['mean_cosine_similarity']:.6f}")
    ref_lat = cmp.get("reference_inference_ms", 0.0)
    qry_lat = cmp.get("query_inference_ms", 0.0)
    print(f"  Latency (ms)          : ref={ref_lat:.4f}, query={qry_lat:.4f}")
    print(f"  Passed                : {cmp['passed']}")


def _print_report(report_dict: dict) -> None:
    sep = "=" * 72
    summary = report_dict.get("summary", {})
    states = report_dict.get("states", {})
    print(f"\n{sep}")
    print("Recombination validation report")
    print(sep)
    print(f"Schema version          : {report_dict.get('schema_version', 'unknown')}")
    print(f"PyTorch version         : {report_dict.get('torch_version', 'unknown')}")
    print(
        f"States evaluated        : {states.get('n_states', 0)} "
        f"(dim={states.get('input_dim', '?')}, source={states.get('source', '?')})"
    )
    print("\nSummary:")
    print(f"  Child ↔ Parent A agreement : {summary.get('child_agrees_with_parent_a', 0):.4f}")
    print(f"  Child ↔ Parent B agreement : {summary.get('child_agrees_with_parent_b', 0):.4f}")
    oracle = summary.get("oracle_agreement")
    if oracle is not None:
        print(f"  Oracle agreement           : {oracle:.4f}")
    print("\nComparisons:")
    for label, cmp in report_dict.get("comparisons", {}).items():
        _print_comparison(label, cmp)
    print(f"\nOverall passed          : {report_dict.get('passed', False)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a crossover child Q-network against its parent models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint paths
    p.add_argument(
        "--checkpoint-dir",
        default="",
        help=(
            "Optional directory for implicit checkpoint resolution.  When set, "
            "missing paths default to parent_A.pt, parent_B.pt, child.pt within this dir."
        ),
    )
    p.add_argument("--parent-a-ckpt", default="", help="Path to parent A state-dict checkpoint.")
    p.add_argument("--parent-b-ckpt", default="", help="Path to parent B state-dict checkpoint.")
    p.add_argument("--child-ckpt", default="", help="Path to child state-dict checkpoint.")

    # Architecture
    p.add_argument("--input-dim", type=int, default=8, help="Network input feature dimension.")
    p.add_argument("--output-dim", type=int, default=4, help="Number of output actions.")
    p.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size.")

    # States
    p.add_argument("--states-file", default="", help="Path to .npy states file (N, input_dim).")
    p.add_argument("--n-states", type=int, default=1000, help="Synthetic states to generate when --states-file is absent.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic states and reproducibility.")

    # Top-k
    p.add_argument("--k-values", default="1,2,3", help="Comma-separated top-k values.")

    # Latency
    p.add_argument("--latency-warmup", type=int, default=5)
    p.add_argument("--latency-repeats", type=int, default=50)

    # Options
    p.add_argument(
        "--include-parent-baseline",
        action="store_true",
        help="Also compute parent A vs parent B comparison (informational).",
    )

    # Thresholds
    p.add_argument("--min-action-agreement", type=float, default=0.70)
    p.add_argument("--max-kl-divergence", type=float, default=1.0)
    p.add_argument("--max-mse", type=float, default=5.0)
    p.add_argument("--min-cosine-similarity", type=float, default=0.70)
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Emit the report without applying pass/fail thresholds.",
    )

    # Output
    p.add_argument("--report-dir", default="reports/recombination", help="Directory to write the JSON report.")
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help=(
            "Max states per forward pass during evaluation (0 = single batch over all states)."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve checkpoint paths.
    parent_a_ckpt = _resolve(args.parent_a_ckpt, args.checkpoint_dir, "parent_A.pt")
    parent_b_ckpt = _resolve(args.parent_b_ckpt, args.checkpoint_dir, "parent_B.pt")
    child_ckpt = _resolve(args.child_ckpt, args.checkpoint_dir, "child.pt")

    _require_file(parent_a_ckpt, "parent_a checkpoint")
    _require_file(parent_b_ckpt, "parent_b checkpoint")
    _require_file(child_ckpt, "child checkpoint")

    k_values = _parse_k_values(args.k_values)

    # Load states.
    states = load_distillation_states(
        args.states_file, args.n_states, args.input_dim, args.seed
    )
    states_source = args.states_file if args.states_file else "synthetic_standard_normal"

    # Load models.
    parent_a = _load_model(parent_a_ckpt, args.input_dim, args.output_dim, args.hidden_size, "parent_a")
    parent_b = _load_model(parent_b_ckpt, args.input_dim, args.output_dim, args.hidden_size, "parent_b")
    child = _load_model(child_ckpt, args.input_dim, args.output_dim, args.hidden_size, "child")

    thresholds = RecombinationThresholds(
        min_action_agreement=args.min_action_agreement,
        max_kl_divergence=args.max_kl_divergence,
        max_mse=args.max_mse,
        min_cosine_similarity=args.min_cosine_similarity,
        report_only=args.report_only,
    )

    evaluator = RecombinationEvaluator(
        parent_a, parent_b, child, thresholds=thresholds
    )
    eval_bs = args.eval_batch_size if args.eval_batch_size > 0 else None
    report = evaluator.evaluate(
        states,
        include_parent_baseline=args.include_parent_baseline,
        k_values=k_values,
        n_latency_warmup=args.latency_warmup,
        n_latency_repeats=args.latency_repeats,
        states_source=states_source,
        eval_batch_size=eval_bs,
        model_paths={
            "parent_a": parent_a_ckpt,
            "parent_b": parent_b_ckpt,
            "child": child_ckpt,
        },
    )

    report_dict = report.to_dict()

    _print_report(report_dict)

    os.makedirs(args.report_dir, exist_ok=True)
    out_path = os.path.join(args.report_dir, "recombination_validation.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, allow_nan=False)
    print(f"\nJSON report written: {out_path}")

    print("\nValidation complete.")
    if not report.passed and not args.report_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
