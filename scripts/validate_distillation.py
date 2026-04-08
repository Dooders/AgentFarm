#!/usr/bin/env python3
"""Validate teacher-student distillation quality from saved checkpoints.

This script evaluates distillation for parent/student pairs (``parent_A`` /
``student_A``, ``parent_B`` / ``student_B``) using :class:`StudentValidator`
and optional synthetic MDP rollouts (GitHub issue #597).

**Offline (state-batch) checks**

- Action agreement (top-1 and top-k)
- KL / MSE / MAE / cosine similarity on Q logits
- Parameter ratio (student vs parent)
- Inference latency ratio
- Optional robustness slices (low resource, noisy, sparse features)

**Optional online-style checks**

- ``--rollout-episodes``: greedy-Q mean return on a **seeded linear synthetic MDP**
  with the same ``input_dim`` / ``output_dim`` as the checkpoints.  This compares
  policies under that MDP's state distribution, **not** the full AgentFarm
  simulation.  For return parity on real tasks, wire checkpoints into the same
  env + feature pipeline used in training (e.g. replay states from
  :class:`~farm.core.decision.training.collector.ExperienceCollector`).

**Outputs** (under ``--report-dir``)

- ``distillation_validation_{A|B}.json`` — full metrics; top-level ``passed`` is
  overall pass (batch thresholds + optional rollout threshold).
- ``batch_validation_passed`` — pass/fail from state-batch thresholds only.
- ``distillation_validation_{A|B}.md`` — short human-readable summary.
- ``distillation_validation_results.csv`` — one row per validated pair.
- ``distillation_validation_summary.md`` — when ``--pair both``, brief combined summary.

See also: ``scripts/run_distillation.py`` (train students), GitHub issue #597.
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

# Allow running directly from repo root without requiring pip install -e .
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.distillation_rollout import (  # noqa: E402
    RolloutComparisonResult,
    compare_parent_student_rollouts,
)
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


def _csv_row(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Single flat row for CSV (robustness slices as JSON)."""
    top_k = report_dict.get("top_k_agreements") or {}
    row = {
        "pair": report_dict.get("pair", ""),
        "parent_ckpt": (report_dict.get("checkpoints") or {}).get("parent", ""),
        "student_ckpt": (report_dict.get("checkpoints") or {}).get("student", ""),
        "action_agreement": report_dict.get("action_agreement"),
        "kl_divergence": report_dict.get("kl_divergence"),
        "mse": report_dict.get("mse"),
        "mae": report_dict.get("mae"),
        "mean_cosine_similarity": report_dict.get("mean_cosine_similarity"),
        "param_ratio": report_dict.get("param_ratio"),
        "parent_inference_ms": report_dict.get("parent_inference_ms"),
        "student_inference_ms": report_dict.get("student_inference_ms"),
        "latency_ratio": report_dict.get("latency_ratio"),
        "top_k_agreements_json": json.dumps(top_k, sort_keys=True),
        "robustness_slice_agreements_json": json.dumps(
            report_dict.get("robustness_slice_agreements") or {},
            sort_keys=True,
        ),
        "batch_validation_passed": report_dict.get("batch_validation_passed"),
        "overall_passed": report_dict.get("passed"),
    }
    rollout = report_dict.get("rollout")
    if rollout:
        row["rollout_parent_mean_return"] = rollout.get("parent_mean_return")
        row["rollout_student_mean_return"] = rollout.get("student_mean_return")
        row["rollout_relative_drop"] = rollout.get("relative_drop")
        row["rollout_passed"] = rollout.get("passed")
    else:
        row["rollout_parent_mean_return"] = ""
        row["rollout_student_mean_return"] = ""
        row["rollout_relative_drop"] = ""
        row["rollout_passed"] = ""
    return row


def _write_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_pair_markdown(path: str, report_dict: Dict[str, Any]) -> None:
    pair = report_dict.get("pair", "?")
    lines = [
        f"# Distillation validation — pair {pair}",
        "",
        f"- **Overall passed**: {report_dict.get('passed')}",
        f"- **Batch validation passed**: {report_dict.get('batch_validation_passed')}",
        "",
        "## Offline metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Action agreement | {report_dict.get('action_agreement')} |",
        f"| KL divergence | {report_dict.get('kl_divergence')} |",
        f"| MSE | {report_dict.get('mse')} |",
        f"| MAE | {report_dict.get('mae')} |",
        f"| Cosine similarity | {report_dict.get('mean_cosine_similarity')} |",
        f"| Param ratio | {report_dict.get('param_ratio')} |",
        f"| Latency ratio | {report_dict.get('latency_ratio')} |",
        "",
    ]
    slices = report_dict.get("robustness_slice_agreements") or {}
    if slices:
        lines.append("## Robustness slice agreement")
        lines.append("")
        for name, val in sorted(slices.items()):
            lines.append(f"- {name}: {val}")
        lines.append("")
    rollout = report_dict.get("rollout")
    if rollout:
        lines.extend(
            [
                "## Synthetic rollout (seeded linear MDP)",
                "",
                f"- Parent mean return: {rollout.get('parent_mean_return')}",
                f"- Student mean return: {rollout.get('student_mean_return')}",
                f"- Relative drop: {rollout.get('relative_drop')}",
                f"- Rollout passed (if threshold set): {rollout.get('passed')}",
                "",
            ]
        )
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- JSON: `distillation_validation_{pair}.json`")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _write_combined_summary(path: str, report_dicts: List[Dict[str, Any]]) -> None:
    lines = ["# Distillation validation — combined summary", ""]
    for d in report_dicts:
        p = d.get("pair", "?")
        lines.append(f"## Pair {p}")
        lines.append("")
        lines.append(f"- Overall passed: **{d.get('passed')}**")
        lines.append(f"- Batch validation passed: **{d.get('batch_validation_passed')}**")
        lines.append(f"- See `distillation_validation_{p}.md` and `.json`.")
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _print_report(pair: str, report: Dict[str, object]) -> None:
    top_k = report["top_k_agreements"]
    print(f"\n{'=' * 72}")
    print(f"Distillation validation report: pair {pair}")
    print(f"{'=' * 72}")
    print(f"Overall passed         : {report['passed']}")
    print(f"Batch validation passed: {report['batch_validation_passed']}")
    print(f"Action agreement       : {report['action_agreement']:.4f}")
    print(f"Top-k agreements       : {top_k}")
    print(f"KL divergence          : {report['kl_divergence']:.6f}")
    print(f"MSE / MAE              : {report['mse']:.6f} / {report['mae']:.6f}")
    print(f"Cosine similarity      : {report['mean_cosine_similarity']:.6f}")
    print(
        "Param ratio            : "
        f"{report['student_param_count']}/{report['parent_param_count']} "
        f"(ratio={report['param_ratio']:.4f})"
    )
    print(
        "Latency (ms)           : "
        f"parent={report['parent_inference_ms']:.4f}, "
        f"student={report['student_inference_ms']:.4f}, "
        f"ratio={report['latency_ratio']:.4f}"
    )
    if report["robustness_slice_agreements"]:
        print(f"Robustness slices      : {report['robustness_slice_agreements']}")
    rollout = report.get("rollout")
    if rollout:
        print(
            "Rollout (synthetic)    : "
            f"parent_mean={rollout['parent_mean_return']:.6f}, "
            f"student_mean={rollout['student_mean_return']:.6f}, "
            f"relative_drop={rollout['relative_drop']}, "
            f"rollout_passed={rollout['passed']}"
        )


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
    parser.add_argument(
        "--max-mae",
        type=float,
        default=None,
        help="If set, fail when MAE exceeds this value.",
    )
    parser.add_argument(
        "--max-latency-ratio",
        type=float,
        default=None,
        help="If set, fail when student/parent latency ratio exceeds this value.",
    )
    parser.add_argument(
        "--min-robustness-action-agreement",
        type=float,
        default=None,
        help="If set, each robustness slice must meet this top-1 agreement.",
    )

    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=0,
        help="If > 0, run synthetic MDP rollouts for this many episodes per network.",
    )
    parser.add_argument(
        "--rollout-max-steps",
        type=int,
        default=50,
        help="Horizon per episode for synthetic rollouts.",
    )
    parser.add_argument(
        "--rollout-base-seed",
        type=int,
        default=None,
        help="Base seed for MDP dynamics (default: --seed).",
    )
    parser.add_argument(
        "--max-relative-return-drop",
        type=float,
        default=None,
        help="If set with rollouts, fail when student return is worse than this margin vs parent.",
    )

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--report-dir",
        default="reports/distillation_validation",
        help="Directory to write JSON, CSV, and markdown reports.",
    )
    return parser.parse_args()


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Reject obviously invalid threshold and rollout flags early."""
    if not 0.0 <= args.min_action_agreement <= 1.0:
        raise ValueError("min_action_agreement must be in [0, 1]")
    if args.max_kl_divergence < 0.0:
        raise ValueError("max_kl_divergence must be >= 0")
    if args.max_mse < 0.0:
        raise ValueError("max_mse must be >= 0")
    if not -1.0 <= args.min_cosine_similarity <= 1.0:
        raise ValueError("min_cosine_similarity must be in [-1, 1]")
    if args.max_param_ratio <= 0.0:
        raise ValueError("max_param_ratio must be > 0")
    if args.max_mae is not None and args.max_mae < 0.0:
        raise ValueError("max_mae must be >= 0 when set")
    if args.max_latency_ratio is not None and args.max_latency_ratio <= 0.0:
        raise ValueError("max_latency_ratio must be > 0 when set")
    if args.min_robustness_action_agreement is not None:
        if not 0.0 <= args.min_robustness_action_agreement <= 1.0:
            raise ValueError("min_robustness_action_agreement must be in [0, 1] when set")
    if args.rollout_episodes < 0:
        raise ValueError("rollout_episodes must be >= 0")
    if args.rollout_max_steps <= 0:
        raise ValueError("rollout_max_steps must be positive")
    if args.max_relative_return_drop is not None:
        if not 0.0 <= args.max_relative_return_drop <= 1.0:
            raise ValueError("max_relative_return_drop must be in [0, 1] when set")
    if args.n_states < 1:
        raise ValueError("n_states must be >= 1")
    if args.latency_warmup < 0 or args.latency_repeats < 0:
        raise ValueError("latency_warmup and latency_repeats must be >= 0")


def _merge_rollout(
    parent: BaseQNetwork,
    student: StudentQNetwork,
    *,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    rollout_episodes: int,
    rollout_max_steps: int,
    rollout_base_seed: int,
    max_relative_return_drop: Optional[float],
) -> Optional[RolloutComparisonResult]:
    if rollout_episodes <= 0:
        return None
    return compare_parent_student_rollouts(
        parent,
        student,
        obs_dim=input_dim,
        n_actions=output_dim,
        base_seed=rollout_base_seed,
        n_episodes=rollout_episodes,
        max_steps=rollout_max_steps,
        device=device,
        max_relative_return_drop=max_relative_return_drop,
    )


def main() -> None:
    args = _parse_args()
    _validate_cli_args(args)
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
        max_mae=args.max_mae,
        max_latency_ratio=args.max_latency_ratio,
        min_robustness_action_agreement=args.min_robustness_action_agreement,
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

    rollout_base_seed = args.rollout_base_seed if args.rollout_base_seed is not None else args.seed

    csv_rows: List[Dict[str, Any]] = []
    all_report_dicts: List[Dict[str, Any]] = []

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
        batch_ok = bool(report_dict["passed"])
        report_dict["batch_validation_passed"] = batch_ok

        rollout_result = _merge_rollout(
            parent,
            student,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            device=device,
            rollout_episodes=args.rollout_episodes,
            rollout_max_steps=args.rollout_max_steps,
            rollout_base_seed=rollout_base_seed,
            max_relative_return_drop=args.max_relative_return_drop,
        )
        if rollout_result is not None:
            report_dict["rollout"] = rollout_result.to_dict()

        overall = batch_ok
        if rollout_result is not None and rollout_result.passed is False:
            overall = False
        report_dict["passed"] = overall

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

        md_path = os.path.join(args.report_dir, f"distillation_validation_{pair}.md")
        _write_pair_markdown(md_path, report_dict)
        print(f"Markdown written     : {md_path}")

        csv_rows.append(_csv_row(report_dict))
        all_report_dicts.append(report_dict)

    csv_path = os.path.join(args.report_dir, "distillation_validation_results.csv")
    _write_csv_rows(csv_path, csv_rows)
    print(f"CSV report written   : {csv_path}")

    if len(all_report_dicts) > 1:
        summary_path = os.path.join(args.report_dir, "distillation_validation_summary.md")
        _write_combined_summary(summary_path, all_report_dicts)
        print(f"Combined summary     : {summary_path}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
