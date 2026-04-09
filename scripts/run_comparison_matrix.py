#!/usr/bin/env python3
"""Compare the crossover child against parents, distilled students, and quantized counterparts.

This script orchestrates the full **comparison matrix** described in the
AgentFarm evaluation methodology.  It reuses :class:`RecombinationEvaluator`
from ``farm.core.decision.training.recombination_eval`` and
:class:`QuantizedValidator` from ``farm.core.decision.training.quantize_ptq``
so that all rows share the **same state buffer** and metric definitions.

Comparison matrix
-----------------
+------+----------------------------------+--------------------------------------+
| Row  | Left (reference A / B)           | Purpose                              |
+======+==================================+======================================+
| A    | parent_A.pt / parent_B.pt        | Baseline recombination quality vs    |
|      | (float state-dicts)              | original float parents               |
+------+----------------------------------+--------------------------------------+
| B    | student_A.pt / student_B.pt      | Child vs distilled intermediates     |
|      | (float state-dicts)              | (same metric family)                 |
+------+----------------------------------+--------------------------------------+
| C    | student_A_int8.pt /              | Effect of quantization on            |
|      | student_B_int8.pt (int8 pickles) | the comparison                       |
+------+----------------------------------+--------------------------------------+
| D    | Same as A, but child loaded as   | Deployment-aligned parity check      |
| opt. | quantized (--row-d)              | (quantized child vs float parents)   |
+------+----------------------------------+--------------------------------------+

Additionally, a per-pair **quantized vs float fidelity** report is produced
for each int8 reference used in Row C (via :class:`QuantizedValidator`), so
the "cost vs accuracy" of quantization is visible alongside the child
comparison.

How to run
----------
::

    # Minimum — only float parents + students (rows A and B) with synthetic states
    python scripts/run_comparison_matrix.py \\
        --parent-a-ckpt  checkpoints/crossover/parent_A.pt \\
        --parent-b-ckpt  checkpoints/crossover/parent_B.pt \\
        --student-a-ckpt checkpoints/distillation/student_A.pt \\
        --student-b-ckpt checkpoints/distillation/student_B.pt \\
        --child-ckpt     checkpoints/crossover/child.pt \\
        --report-dir     reports/comparison_matrix

    # Full matrix including Row C (int8 students) and Row D (quantized child)
    python scripts/run_comparison_matrix.py \\
        --parent-a-ckpt    checkpoints/crossover/parent_A.pt \\
        --parent-b-ckpt    checkpoints/crossover/parent_B.pt \\
        --student-a-ckpt   checkpoints/distillation/student_A.pt \\
        --student-b-ckpt   checkpoints/distillation/student_B.pt \\
        --student-a-int8   checkpoints/quantized/student_A_int8.pt \\
        --student-b-int8   checkpoints/quantized/student_B_int8.pt \\
        --child-ckpt       checkpoints/crossover/child.pt \\
        --child-int8       checkpoints/crossover/child_int8.pt \\
        --row-d \\
        --states-file      data/replay_states.npy \\
        --report-dir       reports/comparison_matrix

    # Checkpoint-dir shortcut (same layout as validate_recombination.py)
    python scripts/run_comparison_matrix.py \\
        --checkpoint-dir checkpoints/ \\
        --seed 42 --n-states 1000 \\
        --include-parent-baseline \\
        --report-dir reports/comparison_matrix

Architecture flags
------------------
``--input-dim``, ``--output-dim``, and ``--hidden-size`` must match the
values used when the checkpoints were trained.  Defaults (8, 4, 64) are the
standard AgentFarm experiment dimensions.

State buffer
------------
All rows use the **same** state buffer.  Pass ``--states-file`` to supply a
real replay-buffer slice (shape ``(N, input_dim)`` float32 NumPy array), or
use ``--seed`` / ``--n-states`` to generate a reproducible synthetic buffer.

Output
------
Reports are written under ``<report-dir>/``:

* ``row_A_child_vs_parents.json``         — RecombinationReport (child vs float parents)
* ``row_B_child_vs_students.json``        — RecombinationReport (child vs float students)
* ``row_C_child_vs_int8_students.json``   — RecombinationReport (child vs int8 students)
  [only when int8 checkpoint paths are given]
* ``row_C_quant_fidelity_A.json``         — QuantizedValidationReport (float vs int8, pair A)
  [only when int8 checkpoint paths are given]
* ``row_C_quant_fidelity_B.json``         — QuantizedValidationReport (float vs int8, pair B)
  [only when int8 checkpoint paths are given]
* ``row_D_child_int8_vs_parents.json``    — RecombinationReport (int8 child vs float parents)
  [only when ``--row-d`` and ``--child-int8`` are given]
* ``comparison_matrix_summary.md``        — Markdown table aggregating key metrics
* ``comparison_matrix_summary.csv``       — CSV version of the same table

Gaps / non-goals
----------------
* Online rollout returns are **not** included (no full AgentFarm env wired).
  Offline metrics (agreement, KL, MSE, MAE, cosine) are used throughout.
* Pass ``--report-only`` to skip threshold enforcement and always emit reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Allow running from repo root without pip install -e .
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_distillation_states,
)
from farm.core.decision.training.quantize_ptq import (  # noqa: E402
    QuantizedValidationThresholds,
    QuantizedValidator,
    load_quantized_checkpoint,
)
from farm.core.decision.training.recombination_eval import (  # noqa: E402
    RecombinationEvaluator,
    RecombinationThresholds,
)

# Checkpoint format tags used in report JSON.
_FMT_FLOAT = "float_state_dict"
_FMT_QUANT = "quantized_full_model"


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------


def _load_base_qnetwork(path: str, input_dim: int, output_dim: int, hidden_size: int) -> BaseQNetwork:
    model = BaseQNetwork(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at {path!r} is not a state dict (got {type(state).__name__})."
        )
    model.load_state_dict(state)
    model.eval()
    return model


def _load_student(path: str, input_dim: int, output_dim: int, hidden_size: int) -> StudentQNetwork:
    model = StudentQNetwork(input_dim=input_dim, output_dim=output_dim, parent_hidden_size=hidden_size)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at {path!r} is not a state dict (got {type(state).__name__})."
        )
    model.load_state_dict(state)
    model.eval()
    return model


def _load_quantized(path: str) -> torch.nn.Module:
    model, _meta = load_quantized_checkpoint(path, device=torch.device("cpu"))
    return model


def _require_file(path: str, label: str) -> None:
    if not path:
        raise ValueError(f"Missing {label} path.  Use the relevant flag or --checkpoint-dir.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found at: {path}")


def _resolve(explicit: str, directory: str, filename: str) -> str:
    if explicit:
        return explicit
    if directory:
        return os.path.join(directory, filename)
    return ""


# ---------------------------------------------------------------------------
# Row runners
# ---------------------------------------------------------------------------


def _run_recombination_row(
    label: str,
    ref_a: torch.nn.Module,
    ref_b: torch.nn.Module,
    child: torch.nn.Module,
    states: np.ndarray,
    states_source: str,
    thresholds: RecombinationThresholds,
    model_paths: Dict[str, str],
    model_formats: Dict[str, str],
    *,
    include_parent_baseline: bool = False,
    k_values: Optional[List[int]] = None,
    latency_warmup: int = 5,
    latency_repeats: int = 50,
    eval_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run RecombinationEvaluator and return the report dict."""
    if k_values is None:
        k_values = [1, 2, 3]
    evaluator = RecombinationEvaluator(
        ref_a,
        ref_b,
        child,
        thresholds=thresholds,
        device=torch.device("cpu"),
    )
    report = evaluator.evaluate(
        states,
        include_parent_baseline=include_parent_baseline,
        k_values=k_values,
        n_latency_warmup=latency_warmup,
        n_latency_repeats=latency_repeats,
        states_source=states_source,
        eval_batch_size=eval_batch_size,
        model_paths=model_paths,
        model_formats=model_formats,
    )
    report_dict = report.to_dict()
    report_dict["matrix_row"] = label
    return report_dict


def _run_quant_fidelity_row(
    label: str,
    float_model: torch.nn.Module,
    quant_model: torch.nn.Module,
    states: np.ndarray,
    thresholds: QuantizedValidationThresholds,
    float_path: str,
    quant_path: str,
    *,
    latency_warmup: int = 5,
    latency_repeats: int = 50,
) -> Dict[str, Any]:
    """Run QuantizedValidator and return the report dict."""
    validator = QuantizedValidator(
        float_model=float_model,
        quantized_model=quant_model,
        thresholds=thresholds,
        device=torch.device("cpu"),
    )
    report = validator.validate(
        states=states,
        n_latency_warmup=latency_warmup,
        n_latency_repeats=latency_repeats,
        float_checkpoint_path=float_path,
        quantized_checkpoint_path=quant_path,
    )
    report_dict = report.to_dict()
    report_dict["matrix_row"] = label
    return report_dict


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------


def _extract_row_summary(label: str, row_type: str, report: Dict[str, Any]) -> Dict[str, str]:
    """Extract key metrics from a report into a flat dict for the summary table."""
    row: Dict[str, str] = {"row": label, "type": row_type}
    if row_type == "recombination":
        summary = report.get("summary", {})
        comparisons = report.get("comparisons", {})
        row["child_vs_ref_a_agreement"] = f"{summary.get('child_agrees_with_parent_a', float('nan')):.4f}"
        row["child_vs_ref_b_agreement"] = f"{summary.get('child_agrees_with_parent_b', float('nan')):.4f}"
        oracle = summary.get("oracle_agreement")
        row["oracle_agreement"] = f"{oracle:.4f}" if oracle is not None else "n/a"
        cmp_a = comparisons.get("child_vs_parent_a", {})
        cmp_b = comparisons.get("child_vs_parent_b", {})
        row["kl_ref_a"] = f"{cmp_a.get('kl_divergence', float('nan')):.6f}"
        row["kl_ref_b"] = f"{cmp_b.get('kl_divergence', float('nan')):.6f}"
        row["cosine_ref_a"] = f"{cmp_a.get('mean_cosine_similarity', float('nan')):.4f}"
        row["cosine_ref_b"] = f"{cmp_b.get('mean_cosine_similarity', float('nan')):.4f}"
        row["passed"] = str(report.get("passed", "n/a"))
    elif row_type == "quantized_fidelity":
        fidelity = report.get("fidelity", {})
        latency = report.get("latency", {})
        row["action_agreement"] = f"{fidelity.get('action_agreement', float('nan')):.4f}"
        kl = fidelity.get("kl_divergence_float_vs_quant")
        if kl is None:
            kl = fidelity.get("kl_divergence")
        row["kl_divergence"] = f"{kl:.6f}" if kl is not None else "n/a"
        row["mean_cosine_similarity"] = f"{fidelity.get('mean_cosine_similarity', float('nan')):.4f}"
        row["float_ms"] = f"{latency.get('float_inference_ms', float('nan')):.4f}"
        row["quant_ms"] = f"{latency.get('quantized_inference_ms', float('nan')):.4f}"
        size = report.get("size", {})
        row["size_ratio"] = f"{size.get('size_ratio', float('nan')):.4f}" if size.get("size_ratio") is not None else "n/a"
        row["passed"] = str(report.get("passed", "n/a"))
    return row


_RECOMBI_COLUMNS = [
    "row", "type",
    "child_vs_ref_a_agreement", "child_vs_ref_b_agreement", "oracle_agreement",
    "kl_ref_a", "kl_ref_b", "cosine_ref_a", "cosine_ref_b", "passed",
]

_QUANT_COLUMNS = [
    "row", "type",
    "action_agreement", "kl_divergence", "mean_cosine_similarity",
    "float_ms", "quant_ms", "size_ratio", "passed",
]


def _write_summary(report_dir: str, rows: List[Dict[str, str]], n_states: int, input_dim: int, states_source: str) -> None:
    """Write comparison_matrix_summary.md and comparison_matrix_summary.csv."""
    # Markdown
    md_lines = [
        "# Comparison Matrix Summary",
        "",
        f"**States evaluated:** {n_states}  ",
        f"**Input dim:** {input_dim}  ",
        f"**States source:** {states_source}",
        "",
        "## Recombination rows (child vs references)",
        "",
        "Columns: child_vs_ref_a_agreement / child_vs_ref_b_agreement — fraction of states"
        " where child and reference choose the same action (top-1).",
        "",
    ]
    recombi_rows = [r for r in rows if r.get("type") == "recombination"]
    quant_rows = [r for r in rows if r.get("type") == "quantized_fidelity"]
    if recombi_rows:
        cols = _RECOMBI_COLUMNS
        header = " | ".join(cols)
        sep = " | ".join(["---"] * len(cols))
        md_lines.append(f"| {header} |")
        md_lines.append(f"| {sep} |")
        for r in recombi_rows:
            cells = " | ".join(r.get(c, "") for c in cols)
            md_lines.append(f"| {cells} |")
    if quant_rows:
        md_lines += [
            "",
            "## Quantized fidelity rows (float vs int8)",
            "",
            "Columns: action_agreement — fraction of states where float and int8 models agree (top-1).",
            "",
        ]
        cols = _QUANT_COLUMNS
        header = " | ".join(cols)
        sep = " | ".join(["---"] * len(cols))
        md_lines.append(f"| {header} |")
        md_lines.append(f"| {sep} |")
        for r in quant_rows:
            cells = " | ".join(r.get(c, "") for c in cols)
            md_lines.append(f"| {cells} |")
    md_lines += [
        "",
        "## Gaps",
        "",
        "- Online rollout returns are not included (no full AgentFarm env wired).",
        "- Only offline metrics (action agreement, KL, MSE, MAE, cosine) are reported.",
        "",
    ]
    md_path = os.path.join(report_dir, "comparison_matrix_summary.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"  Markdown summary: {md_path}")

    # CSV (all rows, all keys)
    all_keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    csv_path = os.path.join(report_dir, "comparison_matrix_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV summary      : {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full child-vs-references comparison matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint shortcuts
    p.add_argument(
        "--checkpoint-dir",
        default="",
        help=(
            "Optional base directory for implicit checkpoint resolution.  "
            "If set, missing paths are inferred using the default filenames "
            "(parent_A.pt, parent_B.pt, student_A.pt, student_B.pt, child.pt, "
            "student_A_int8.pt, student_B_int8.pt, child_int8.pt)."
        ),
    )

    # Float parent checkpoints (Row A)
    p.add_argument("--parent-a-ckpt", default="", help="Float parent A state-dict checkpoint.")
    p.add_argument("--parent-b-ckpt", default="", help="Float parent B state-dict checkpoint.")

    # Float student checkpoints (Row B)
    p.add_argument("--student-a-ckpt", default="", help="Float student A state-dict checkpoint.")
    p.add_argument("--student-b-ckpt", default="", help="Float student B state-dict checkpoint.")

    # Int8 student checkpoints (Row C)
    p.add_argument("--student-a-int8", default="", help="Int8 student A checkpoint (quantized full-model pickle).")
    p.add_argument("--student-b-int8", default="", help="Int8 student B checkpoint (quantized full-model pickle).")

    # Child checkpoint
    p.add_argument("--child-ckpt", default="", help="Child float state-dict checkpoint.")
    p.add_argument("--child-int8", default="", help="Quantized child checkpoint (for Row D).")

    # Row D
    p.add_argument(
        "--row-d",
        action="store_true",
        help="Also run Row D: quantized child vs float parents.  Requires --child-int8.",
    )

    # Architecture
    p.add_argument("--input-dim", type=int, default=8)
    p.add_argument("--output-dim", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=64)

    # States
    p.add_argument("--states-file", default="", help="Path to .npy states file (N, input_dim) float32.")
    p.add_argument("--n-states", type=int, default=1000, help="Synthetic states to generate when --states-file is absent.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible synthetic states.")

    # Top-k
    p.add_argument("--k-values", default="1,2,3", help="Comma-separated top-k values for action agreement.")

    # Evaluation options
    p.add_argument("--latency-warmup", type=int, default=5)
    p.add_argument("--latency-repeats", type=int, default=50)
    p.add_argument("--eval-batch-size", type=int, default=0, help="Max states per forward pass (0 = all states at once).")
    p.add_argument(
        "--include-parent-baseline",
        action="store_true",
        help="Include parent-A vs parent-B baseline in Row A report.",
    )

    # Thresholds (recombination)
    p.add_argument("--min-action-agreement", type=float, default=0.70)
    p.add_argument("--max-kl-divergence", type=float, default=1.0)
    p.add_argument("--max-mse", type=float, default=5.0)
    p.add_argument("--min-cosine-similarity", type=float, default=0.70)
    p.add_argument("--report-only", action="store_true", help="Skip pass/fail threshold enforcement.")

    # Output
    p.add_argument("--report-dir", default="reports/comparison_matrix")

    return p.parse_args()


def _parse_k_values(raw: str) -> List[int]:
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    if not parts:
        raise ValueError("--k-values cannot be empty.")
    parsed = [int(v) for v in parts]
    invalid = [v for v in parsed if v <= 0]
    if invalid:
        raise ValueError(f"k-values must be positive integers; got {invalid}.")
    return parsed


def _print_recombination_summary(label: str, report: Dict[str, Any]) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"Row {label}")
    print(sep)
    summary = report.get("summary", {})
    print(f"  Child ↔ Ref-A agreement : {summary.get('child_agrees_with_parent_a', 'n/a')}")
    print(f"  Child ↔ Ref-B agreement : {summary.get('child_agrees_with_parent_b', 'n/a')}")
    oracle = summary.get("oracle_agreement")
    if oracle is not None:
        print(f"  Oracle agreement        : {oracle:.4f}")
    for comp_label, cmp in report.get("comparisons", {}).items():
        kl = cmp.get("kl_divergence")
        mse = cmp.get("mse")
        cosine = cmp.get("mean_cosine_similarity")
        kl_text = f"{float(kl):.6f}" if kl is not None else "n/a"
        mse_text = f"{float(mse):.6f}" if mse is not None else "n/a"
        cosine_text = f"{float(cosine):.4f}" if cosine is not None else "n/a"
        print(
            f"  [{comp_label}] KL={kl_text}"
            f"  MSE={mse_text}"
            f"  cosine={cosine_text}"
            f"  passed={cmp.get('passed', '?')}"
        )
    print(f"  Overall passed          : {report.get('passed', 'n/a')}")


def _print_quant_summary(label: str, report: Dict[str, Any]) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"Row {label}")
    print(sep)
    fidelity = report.get("fidelity", {})
    latency = report.get("latency", {})
    size = report.get("size", {})
    print(f"  Action agreement (float vs int8) : {fidelity.get('action_agreement', 'n/a')}")
    print(f"  KL divergence                    : {fidelity.get('kl_divergence', 'n/a')}")
    print(f"  Mean cosine similarity           : {fidelity.get('mean_cosine_similarity', 'n/a')}")
    print(f"  Float inference (ms)             : {latency.get('float_inference_ms', 'n/a')}")
    print(f"  Quant inference (ms)             : {latency.get('quantized_inference_ms', 'n/a')}")
    sr = size.get("size_ratio")
    print(f"  Size ratio (quant/float)         : {sr:.4f}" if sr is not None else "  Size ratio                       : n/a")
    print(f"  Passed                           : {report.get('passed', 'n/a')}")


def main() -> None:
    args = _parse_args()
    k_values = _parse_k_values(args.k_values)
    os.makedirs(args.report_dir, exist_ok=True)
    eval_bs = args.eval_batch_size if args.eval_batch_size > 0 else None

    # Resolve checkpoint paths
    parent_a_ckpt = _resolve(args.parent_a_ckpt, args.checkpoint_dir, "parent_A.pt")
    parent_b_ckpt = _resolve(args.parent_b_ckpt, args.checkpoint_dir, "parent_B.pt")
    student_a_ckpt = _resolve(args.student_a_ckpt, args.checkpoint_dir, "student_A.pt")
    student_b_ckpt = _resolve(args.student_b_ckpt, args.checkpoint_dir, "student_B.pt")
    student_a_int8 = _resolve(args.student_a_int8, args.checkpoint_dir, "student_A_int8.pt")
    student_b_int8 = _resolve(args.student_b_int8, args.checkpoint_dir, "student_B_int8.pt")
    child_ckpt = _resolve(args.child_ckpt, args.checkpoint_dir, "child.pt")
    child_int8 = _resolve(args.child_int8, args.checkpoint_dir, "child_int8.pt")

    # Validate required checkpoints (Row A always required)
    _require_file(parent_a_ckpt, "parent_a checkpoint")
    _require_file(parent_b_ckpt, "parent_b checkpoint")
    _require_file(child_ckpt, "child checkpoint")

    # Generate / load shared state buffer (used by ALL rows)
    print("\nLoading state buffer …")
    states = load_distillation_states(
        args.states_file, args.n_states, args.input_dim, args.seed
    )
    states_source = args.states_file if args.states_file else f"synthetic_std_normal(seed={args.seed},n={args.n_states})"
    print(f"  States: shape={states.shape}, source={states_source}")

    # Shared threshold objects
    recombi_thresholds = RecombinationThresholds(
        min_action_agreement=args.min_action_agreement,
        max_kl_divergence=args.max_kl_divergence,
        max_mse=args.max_mse,
        min_cosine_similarity=args.min_cosine_similarity,
        report_only=args.report_only,
    )
    quant_thresholds = QuantizedValidationThresholds(report_only=args.report_only)

    summary_rows: List[Dict[str, str]] = []
    all_passed = True

    # ------------------------------------------------------------------
    # Row A: child vs float parents
    # ------------------------------------------------------------------
    print("\n[Row A] Child vs float parent A & parent B …")
    parent_a = _load_base_qnetwork(parent_a_ckpt, args.input_dim, args.output_dim, args.hidden_size)
    parent_b = _load_base_qnetwork(parent_b_ckpt, args.input_dim, args.output_dim, args.hidden_size)
    child_float = _load_base_qnetwork(child_ckpt, args.input_dim, args.output_dim, args.hidden_size)

    row_a = _run_recombination_row(
        "A: child vs float parents",
        parent_a, parent_b, child_float,
        states, states_source, recombi_thresholds,
        model_paths={"parent_a": parent_a_ckpt, "parent_b": parent_b_ckpt, "child": child_ckpt},
        model_formats={"parent_a": _FMT_FLOAT, "parent_b": _FMT_FLOAT, "child": _FMT_FLOAT},
        include_parent_baseline=args.include_parent_baseline,
        k_values=k_values,
        latency_warmup=args.latency_warmup,
        latency_repeats=args.latency_repeats,
        eval_batch_size=eval_bs,
    )
    out_a = os.path.join(args.report_dir, "row_A_child_vs_parents.json")
    with open(out_a, "w", encoding="utf-8") as fh:
        json.dump(row_a, fh, indent=2, allow_nan=False)
    print(f"  Report: {out_a}")
    _print_recombination_summary("A (child vs float parents)", row_a)
    summary_rows.append(_extract_row_summary("A: child vs float parents", "recombination", row_a))
    all_passed = all_passed and row_a.get("passed", True)

    # ------------------------------------------------------------------
    # Row B: child vs float students
    # ------------------------------------------------------------------
    has_students = bool(student_a_ckpt and os.path.isfile(student_a_ckpt)
                        and student_b_ckpt and os.path.isfile(student_b_ckpt))
    if has_students:
        print("\n[Row B] Child vs float student A & student B …")
        student_a = _load_student(student_a_ckpt, args.input_dim, args.output_dim, args.hidden_size)
        student_b = _load_student(student_b_ckpt, args.input_dim, args.output_dim, args.hidden_size)

        row_b = _run_recombination_row(
            "B: child vs float students",
            student_a, student_b, child_float,
            states, states_source, recombi_thresholds,
            model_paths={"parent_a": student_a_ckpt, "parent_b": student_b_ckpt, "child": child_ckpt},
            model_formats={"parent_a": _FMT_FLOAT, "parent_b": _FMT_FLOAT, "child": _FMT_FLOAT},
            k_values=k_values,
            latency_warmup=args.latency_warmup,
            latency_repeats=args.latency_repeats,
            eval_batch_size=eval_bs,
        )
        out_b = os.path.join(args.report_dir, "row_B_child_vs_students.json")
        with open(out_b, "w", encoding="utf-8") as fh:
            json.dump(row_b, fh, indent=2, allow_nan=False)
        print(f"  Report: {out_b}")
        _print_recombination_summary("B (child vs float students)", row_b)
        summary_rows.append(_extract_row_summary("B: child vs float students", "recombination", row_b))
        all_passed = all_passed and row_b.get("passed", True)
    else:
        print("\n[Row B] Skipped — student_a_ckpt / student_b_ckpt not provided or not found.")

    # ------------------------------------------------------------------
    # Row C: child vs int8 students + per-pair quantized fidelity
    # ------------------------------------------------------------------
    has_int8_students = bool(student_a_int8 and os.path.isfile(student_a_int8)
                             and student_b_int8 and os.path.isfile(student_b_int8))
    if has_int8_students:
        print("\n[Row C] Child vs int8 student A & student B …")
        q_student_a = _load_quantized(student_a_int8)
        q_student_b = _load_quantized(student_b_int8)

        row_c = _run_recombination_row(
            "C: child vs int8 students",
            q_student_a, q_student_b, child_float,
            states, states_source, recombi_thresholds,
            model_paths={"parent_a": student_a_int8, "parent_b": student_b_int8, "child": child_ckpt},
            model_formats={"parent_a": _FMT_QUANT, "parent_b": _FMT_QUANT, "child": _FMT_FLOAT},
            k_values=k_values,
            latency_warmup=args.latency_warmup,
            latency_repeats=args.latency_repeats,
            eval_batch_size=eval_bs,
        )
        out_c = os.path.join(args.report_dir, "row_C_child_vs_int8_students.json")
        with open(out_c, "w", encoding="utf-8") as fh:
            json.dump(row_c, fh, indent=2, allow_nan=False)
        print(f"  Report: {out_c}")
        _print_recombination_summary("C (child vs int8 students)", row_c)
        summary_rows.append(_extract_row_summary("C: child vs int8 students", "recombination", row_c))
        all_passed = all_passed and row_c.get("passed", True)

        # Per-pair quantized fidelity (float student vs int8 student)
        if has_students:
            for pair_label, float_model, float_path, quant_model, quant_path in [
                ("A", student_a, student_a_ckpt, q_student_a, student_a_int8),
                ("B", student_b, student_b_ckpt, q_student_b, student_b_int8),
            ]:
                print(f"\n[Row C / fidelity-{pair_label}] Float student {pair_label} vs int8 student {pair_label} …")
                qfid = _run_quant_fidelity_row(
                    f"C_fidelity_{pair_label}: float_student_{pair_label} vs int8",
                    float_model, quant_model,
                    states, quant_thresholds,
                    float_path, quant_path,
                    latency_warmup=args.latency_warmup,
                    latency_repeats=args.latency_repeats,
                )
                out_qfid = os.path.join(args.report_dir, f"row_C_quant_fidelity_{pair_label}.json")
                with open(out_qfid, "w", encoding="utf-8") as fh:
                    json.dump(qfid, fh, indent=2, allow_nan=False)
                print(f"  Report: {out_qfid}")
                _print_quant_summary(f"C / quant fidelity {pair_label}", qfid)
                summary_rows.append(_extract_row_summary(
                    f"C_fidelity_{pair_label}: float_student_{pair_label} vs int8",
                    "quantized_fidelity", qfid,
                ))
                all_passed = all_passed and qfid.get("passed", True)
        else:
            print(
                "\n[Row C / fidelity] Skipped — float student checkpoints are required for "
                "int8-vs-float fidelity reports."
            )
    else:
        print("\n[Row C] Skipped — student_a_int8 / student_b_int8 not provided or not found.")

    # ------------------------------------------------------------------
    # Row D (optional): quantized child vs float parents
    # ------------------------------------------------------------------
    if args.row_d:
        has_child_int8 = bool(child_int8 and os.path.isfile(child_int8))
        if not has_child_int8:
            print("\n[Row D] Skipped — --child-int8 not provided or not found.")
        else:
            print("\n[Row D] Quantized child vs float parent A & parent B …")
            q_child = _load_quantized(child_int8)

            row_d = _run_recombination_row(
                "D: quantized child vs float parents",
                parent_a, parent_b, q_child,
                states, states_source, recombi_thresholds,
                model_paths={"parent_a": parent_a_ckpt, "parent_b": parent_b_ckpt, "child": child_int8},
                model_formats={"parent_a": _FMT_FLOAT, "parent_b": _FMT_FLOAT, "child": _FMT_QUANT},
                include_parent_baseline=args.include_parent_baseline,
                k_values=k_values,
                latency_warmup=args.latency_warmup,
                latency_repeats=args.latency_repeats,
                eval_batch_size=eval_bs,
            )
            out_d = os.path.join(args.report_dir, "row_D_child_int8_vs_parents.json")
            with open(out_d, "w", encoding="utf-8") as fh:
                json.dump(row_d, fh, indent=2, allow_nan=False)
            print(f"  Report: {out_d}")
            _print_recombination_summary("D (quantized child vs float parents)", row_d)
            summary_rows.append(_extract_row_summary("D: quantized child vs float parents", "recombination", row_d))
            all_passed = all_passed and row_d.get("passed", True)
    else:
        print("\n[Row D] Skipped — pass --row-d to include.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nWriting summary …")
    _write_summary(args.report_dir, summary_rows, len(states), args.input_dim, states_source)

    print(f"\nAll rows overall passed: {all_passed}")
    print("Comparison matrix complete.")

    if not all_passed and not args.report_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
