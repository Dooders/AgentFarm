#!/usr/bin/env python3
"""Qualitative error-analysis for recombined Q-networks.

Given a set of evaluation states and three Q-network checkpoints (parent A,
parent B, and child), this script writes:

- A **CSV** of per-state disagreements, KL divergences, MSEs, cosine
  similarities, and top-*k* mismatch flags.
- A **JSON** version of the same records (including optional raw logits).
- Optionally a ``.npy`` file of hidden-layer activations for a memory-bounded
  probe subset.

How to run
----------
::

    # Explicit checkpoint paths + a saved states file:
    python scripts/analyze_recombination.py \\
        --parent-a-ckpt  checkpoints/parent_A.pt \\
        --parent-b-ckpt  checkpoints/parent_B.pt \\
        --child-ckpt     checkpoints/child.pt \\
        --states-file    data/replay_states.npy \\
        --output-dir     reports/analysis

    # Implicit paths via checkpoint directory (parent_A.pt, parent_B.pt, child.pt):
    python scripts/analyze_recombination.py \\
        --checkpoint-dir checkpoints/crossover \\
        --output-dir     reports/analysis

    # Include raw logits in JSON, worst-10 states highlighted, activations export:
    python scripts/analyze_recombination.py \\
        --checkpoint-dir checkpoints/crossover \\
        --include-logits \\
        --worst-k 10 \\
        --worst-k-criterion max_kl \\
        --activations-out reports/analysis/child_activations.npy \\
        --activation-layer-index 4 \\
        --activation-max-states 500

Architecture flags (``--input-dim``, ``--output-dim``, ``--hidden-size``)
must match the checkpoint's training dimensions.  Defaults are 8/4/64.

Outputs
-------
``<output-dir>/disagreements.csv``
    One row per evaluation state.  Columns: ``state_index``,
    ``child_action``, ``parent_a_action``, ``parent_b_action``,
    ``agrees_with_parent_{a,b}``, ``agrees_with_any_parent``,
    ``kl_child_vs_parent_{a,b}``, ``mse_child_vs_parent_{a,b}``,
    ``cosine_child_vs_parent_{a,b}``,
    ``parent_{a,b}_in_top_k_{k}`` for each *k*.

``<output-dir>/disagreements.json``
    Same records in JSON with summary counts.  Includes raw logits when
    ``--include-logits`` is set.

``<output-dir>/worst_<k>_states.json``
    JSON document containing only the *k* worst records, sorted by
    ``--worst-k-criterion``.  Only written when ``--worst-k > 0``.

``<activations-out>``
    NumPy ``.npy`` file of shape ``(N, activation_dim)`` when
    ``--activations-out`` is provided.

Related
-------
- ``scripts/validate_recombination.py`` — aggregate fidelity metrics
- :mod:`farm.core.decision.training.recombination_analysis` — Python API
- ``docs/howto/neural_recombination_runbook.md`` — integration guide
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
from farm.core.decision.training.recombination_analysis import (  # noqa: E402
    ANALYSIS_SCHEMA_VERSION,
    WORST_K_CRITERIA,
    export_disagreements_csv,
    export_disagreements_json,
    extract_activations,
    extract_disagreements,
    worst_k_states,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(explicit: str, directory: str, filename: str) -> str:
    """Return *explicit* if set, else ``<directory>/<filename>``."""
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
    """Load a ``BaseQNetwork`` from a float state-dict checkpoint."""
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


def _print_summary(records, worst_records=None, criterion: Optional[str] = None) -> None:
    """Print a human-readable summary to stdout."""
    n = len(records)
    if n == 0:
        print("No records.")
        return
    n_disagree_a = sum(not r.agrees_with_parent_a for r in records)
    n_disagree_b = sum(not r.agrees_with_parent_b for r in records)
    n_disagree_both = sum(
        not r.agrees_with_parent_a and not r.agrees_with_parent_b for r in records
    )
    n_oracle = sum(r.agrees_with_any_parent for r in records)
    mean_kl_a = sum(r.kl_child_vs_parent_a for r in records) / n
    mean_kl_b = sum(r.kl_child_vs_parent_b for r in records) / n
    mean_mse_a = sum(r.mse_child_vs_parent_a for r in records) / n
    mean_mse_b = sum(r.mse_child_vs_parent_b for r in records) / n

    sep = "=" * 72
    print(f"\n{sep}")
    print("Recombination error-analysis summary")
    print(sep)
    print(f"States evaluated           : {n}")
    print(f"Disagree with parent A     : {n_disagree_a} ({n_disagree_a / n:.1%})")
    print(f"Disagree with parent B     : {n_disagree_b} ({n_disagree_b / n:.1%})")
    print(f"Disagree with both parents : {n_disagree_both} ({n_disagree_both / n:.1%})")
    print(f"Oracle agreement           : {n_oracle} ({n_oracle / n:.1%})")
    print(f"Mean KL vs parent A        : {mean_kl_a:.6f}")
    print(f"Mean KL vs parent B        : {mean_kl_b:.6f}")
    print(f"Mean MSE vs parent A       : {mean_mse_a:.6f}")
    print(f"Mean MSE vs parent B       : {mean_mse_b:.6f}")

    if worst_records:
        print(f"\nWorst {len(worst_records)} states (criterion={criterion!r}):")
        for r in worst_records[:5]:  # show up to 5 in console
            print(
                f"  idx={r.state_index:>5}  "
                f"child={r.child_action}  "
                f"pa={r.parent_a_action}  "
                f"pb={r.parent_b_action}  "
                f"kl_a={r.kl_child_vs_parent_a:.4f}  "
                f"kl_b={r.kl_child_vs_parent_b:.4f}  "
                f"mse_a={r.mse_child_vs_parent_a:.4f}  "
                f"mse_b={r.mse_child_vs_parent_b:.4f}"
            )
        if len(worst_records) > 5:
            print(f"  … ({len(worst_records) - 5} more in JSON output)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-state error analysis for crossover child Q-networks.",
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
    p.add_argument(
        "--n-states",
        type=int,
        default=1000,
        help="Synthetic states to generate when --states-file is absent.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic states.")

    # Top-k
    p.add_argument(
        "--k-values",
        default="1,2,3",
        help="Comma-separated top-k values for mismatch flags.",
    )

    # Output
    p.add_argument(
        "--output-dir",
        default="reports/analysis",
        help="Directory to write CSV and JSON outputs.",
    )
    p.add_argument(
        "--include-logits",
        action="store_true",
        help="Include raw Q-value logit vectors in the JSON output.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Max states per forward pass during disagreement extraction.",
    )

    # Worst-k
    p.add_argument(
        "--worst-k",
        type=int,
        default=0,
        help="Write a worst-k JSON file highlighting the k states with the largest errors (0 = skip).",
    )
    p.add_argument(
        "--worst-k-criterion",
        default="max_kl",
        choices=sorted(WORST_K_CRITERIA),
        help="Criterion for selecting worst-k states.",
    )

    # Activations
    p.add_argument(
        "--activations-out",
        default="",
        help=(
            "Optional path for hidden-layer activation numpy export "
            "(e.g. reports/analysis/activations.npy).  Skipped when empty."
        ),
    )
    p.add_argument(
        "--activation-layer-index",
        type=int,
        default=4,
        help=(
            "Index into list(model.modules()) for the activation hook.  "
            "For BaseQNetwork: 4 = first hidden ReLU, 8 = second hidden ReLU."
        ),
    )
    p.add_argument(
        "--activation-max-states",
        type=int,
        default=0,
        help=(
            "Cap on states used for activation extraction (0 = all states).  "
            "Use to stay memory-bounded."
        ),
    )
    p.add_argument(
        "--activation-batch-size",
        type=int,
        default=256,
        help="Max states per forward pass during activation extraction.",
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

    # Load models.
    parent_a = _load_model(
        parent_a_ckpt, args.input_dim, args.output_dim, args.hidden_size, "parent_a"
    )
    parent_b = _load_model(
        parent_b_ckpt, args.input_dim, args.output_dim, args.hidden_size, "parent_b"
    )
    child = _load_model(
        child_ckpt, args.input_dim, args.output_dim, args.hidden_size, "child"
    )

    # Extract per-state disagreements.
    print(f"Extracting disagreements for {states.shape[0]} states …")
    records = extract_disagreements(
        parent_a,
        parent_b,
        child,
        states,
        include_logits=args.include_logits,
        k_values=k_values,
        batch_size=args.batch_size,
    )

    # Worst-k states.
    worst_records = []
    if args.worst_k > 0:
        worst_records = worst_k_states(records, k=args.worst_k, criterion=args.worst_k_criterion)

    _print_summary(records, worst_records=worst_records, criterion=args.worst_k_criterion)

    # Write outputs.
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "disagreements.csv")
    export_disagreements_csv(records, csv_path, k_values=k_values)
    print(f"\nCSV written  : {csv_path}")

    json_path = os.path.join(args.output_dir, "disagreements.json")
    export_disagreements_json(
        records,
        json_path,
        extra_metadata={
            "parent_a_ckpt": parent_a_ckpt,
            "parent_b_ckpt": parent_b_ckpt,
            "child_ckpt": child_ckpt,
            "states_source": args.states_file or "synthetic_standard_normal",
            "k_values": k_values,
        },
    )
    print(f"JSON written : {json_path}")

    if worst_records:
        worst_path = os.path.join(
            args.output_dir, f"worst_{len(worst_records)}_states.json"
        )
        worst_doc = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "criterion": args.worst_k_criterion,
            "k": len(worst_records),
            "records": [r.to_dict() for r in worst_records],
        }
        with open(worst_path, "w", encoding="utf-8") as fh:
            json.dump(worst_doc, fh, indent=2, allow_nan=False)
        print(f"Worst-k JSON : {worst_path}")

    # Activation export.
    if args.activations_out:
        max_s = args.activation_max_states if args.activation_max_states > 0 else None
        print(
            f"\nExtracting activations from layer index {args.activation_layer_index} "
            f"(max_states={max_s}) …"
        )
        acts = extract_activations(
            child,
            states,
            layer_index=args.activation_layer_index,
            max_states=max_s,
            batch_size=args.activation_batch_size,
        )
        np.save(args.activations_out, acts)
        print(f"Activations  : {args.activations_out}  shape={acts.shape}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
