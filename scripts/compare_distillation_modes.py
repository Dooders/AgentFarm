#!/usr/bin/env python
"""Reproducible hard-only vs soft-only vs blended distillation comparison.

Used to satisfy documentation acceptance criteria for soft-label distillation
(GitHub issue #596): same frozen teacher, state buffer, and student
initialization; only ``alpha`` (and thus the training objective) changes.

Run from repo root::

    python scripts/compare_distillation_modes.py

Optional: ``--epochs N``, ``--json-out path.json`` for machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.trainer_distill import (  # noqa: E402
    DistillationConfig,
    DistillationTrainer,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-dim", type=int, default=8)
    p.add_argument("--output-dim", type=int, default=4)
    p.add_argument("--parent-hidden", type=int, default=64)
    p.add_argument("--n-states", type=int, default=5000)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--json-out", default="", help="Write summary JSON to this path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base_seed = args.seed

    # Fixed teacher and state distribution (independent of training seed).
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    teacher = BaseQNetwork(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_size=args.parent_hidden,
    )
    teacher_sd = deepcopy(teacher.state_dict())

    rng = np.random.default_rng(base_seed + 10_000)
    states = rng.standard_normal((args.n_states, args.input_dim)).astype(np.float32)

    # Fixed student initialization for all three runs.
    torch.manual_seed(base_seed + 20_000)
    _template = StudentQNetwork(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        parent_hidden_size=args.parent_hidden,
    )
    student_init_sd = deepcopy(_template.state_dict())
    del _template

    modes: List[Dict[str, Any]] = [
        {"name": "hard_only", "alpha": 0.0},
        {"name": "blended", "alpha": 0.7},
        {"name": "soft_only", "alpha": 1.0},
    ]

    results: List[Dict[str, Any]] = []

    for mode in modes:
        teacher.load_state_dict(teacher_sd)
        student = StudentQNetwork(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            parent_hidden_size=args.parent_hidden,
        )
        student.load_state_dict(student_init_sd)

        cfg = DistillationConfig(
            temperature=args.temperature,
            alpha=mode["alpha"],
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            loss_fn="kl",
            seed=base_seed + 30_000,
        )
        trainer = DistillationTrainer(teacher, student, cfg)
        metrics = trainer.train(states, checkpoint_path=None)

        last_agree = metrics.action_agreements[-1] if metrics.action_agreements else None
        last_prob = metrics.mean_prob_similarities[-1] if metrics.mean_prob_similarities else None
        row = {
            "mode": mode["name"],
            "alpha": mode["alpha"],
            "best_val_loss": metrics.best_val_loss,
            "best_epoch": metrics.best_epoch,
            "final_action_agreement": last_agree,
            "final_mean_prob_similarity": last_prob,
            "final_train_loss": metrics.train_losses[-1] if metrics.train_losses else None,
            "final_train_soft_loss": metrics.train_soft_losses[-1]
            if metrics.train_soft_losses
            else None,
        }
        results.append(row)

    # Console table
    print("\nDistillation mode comparison (same teacher, states, student init)\n")
    hdr = f"{'mode':<12} {'alpha':>6}  {'agree%':>8}  {'prob_sim':>10}  {'best_val':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ag = r["final_action_agreement"]
        ps = r["final_mean_prob_similarity"]
        bv = r["best_val_loss"]
        ag_s = f"{100.0 * ag:.2f}" if ag is not None else "n/a"
        ps_s = f"{ps:.4f}" if ps is not None else "n/a"
        bv_s = f"{bv:.6f}" if bv != float("inf") else "n/a"
        print(
            f"{r['mode']:<12} {r['alpha']:>6.1f}  {ag_s:>8}  {ps_s:>10}  {bv_s:>12}"
        )
    print()

    payload = {
        "seed_base": base_seed,
        "hyperparams": {
            "input_dim": args.input_dim,
            "output_dim": args.output_dim,
            "parent_hidden": args.parent_hidden,
            "n_states": args.n_states,
            "temperature": args.temperature,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_fraction": args.val_fraction,
            "loss_fn": "kl",
        },
        "runs": results,
    }
    if args.json_out:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json_out}\n")


if __name__ == "__main__":
    main()
