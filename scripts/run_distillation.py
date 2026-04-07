#!/usr/bin/env python
"""Knowledge distillation script: train student_A from parent_A and student_B from parent_B.

Usage
-----
Train both pairs with defaults::

    python scripts/run_distillation.py

Train a single pair::

    python scripts/run_distillation.py --pair A

Override hyperparameters::

    python scripts/run_distillation.py \\
        --temperature 4.0 \\
        --alpha 0.2 \\
        --epochs 30 \\
        --lr 5e-4 \\
        --batch-size 64 \\
        --n-states 2000 \\
        --seed 42 \\
        --output-dir checkpoints/distillation

Checkpoints are saved under ``<output_dir>/student_<pair>.pt`` with a
companion ``student_<pair>.pt.json`` metadata file.

Notes
-----
When pre-trained parent checkpoints are available on disk (e.g. produced by a
prior training run) pass ``--parent-a-ckpt`` / ``--parent-b-ckpt`` to load
them.  Otherwise random weights are used, which is useful for integration
testing.

The script synthesises random states if no replay-buffer file is provided.
Pass ``--states-file <npy>`` to supply a real state distribution (shape
``(N, input_dim)`` NumPy array saved with ``np.save``).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Allow running directly from repo root without installing the package
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.trainer_distill import (  # noqa: E402
    DistillationConfig,
    DistillationTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_network(path: str, input_dim: int, output_dim: int, hidden_size: int) -> BaseQNetwork:
    """Load a ``BaseQNetwork`` from a state-dict checkpoint or return random weights."""
    net = BaseQNetwork(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size)
    if path and os.path.isfile(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(
                f"Checkpoint at '{path}' does not contain a state dict (got {type(state).__name__})."
            )
        net.load_state_dict(state)
        print(f"  Loaded parent weights from: {path}")
    else:
        print(f"  No checkpoint found at '{path}' – using random weights.")
    return net


def _generate_states(n: int, input_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, input_dim)).astype("float32")


def _run_pair(
    pair: str,
    input_dim: int,
    output_dim: int,
    parent_hidden: int,
    parent_ckpt: str,
    n_states: int,
    states_file: str,
    cfg: DistillationConfig,
    output_dir: str,
    rng: np.random.Generator,
) -> None:
    """Run distillation for one parent/student pair."""
    print(f"\n{'=' * 60}")
    print(f"Distilling: parent_{pair}  -->  student_{pair}")
    print(f"{'=' * 60}")

    # Build / load parent (teacher)
    teacher = _load_network(parent_ckpt, input_dim, output_dim, parent_hidden)

    # Build student (half hidden width)
    student = StudentQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        parent_hidden_size=parent_hidden,
    )
    parent_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Parent params : {parent_params:,}")
    print(f"  Student params: {student_params:,}  ({100*student_params/parent_params:.1f}% of parent)")

    # Prepare state buffer
    if states_file and os.path.isfile(states_file):
        states = np.load(states_file).astype("float32")
        print(f"  States loaded from {states_file}: shape={states.shape}")
    else:
        states = _generate_states(n_states, input_dim, rng)
        print(f"  Using {n_states} synthetic random states (shape={states.shape})")

    # Train
    trainer = DistillationTrainer(teacher, student, cfg)
    ckpt_path = os.path.join(output_dir, f"student_{pair}.pt")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Training for {cfg.epochs} epochs …")
    metrics = trainer.train(states, checkpoint_path=ckpt_path)

    # Summary
    print(f"\n  --- Results for student_{pair} ---")
    if metrics.train_losses:
        print(f"  Final train loss : {metrics.train_losses[-1]:.6f}")
    if metrics.val_losses:
        print(f"  Best val loss    : {metrics.best_val_loss:.6f}  (epoch {metrics.best_epoch})")
        print(f"  Action agreement : {metrics.action_agreements[-1]*100:.1f}%  (last epoch)")
    print(f"  Checkpoint saved : {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run knowledge distillation: parent_A→student_A and/or parent_B→student_B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pair",
        choices=["A", "B", "both"],
        default="both",
        help="Which parent/student pair to distil.",
    )
    # Architecture
    p.add_argument("--input-dim", type=int, default=8, help="State feature dimension.")
    p.add_argument("--output-dim", type=int, default=4, help="Number of actions.")
    p.add_argument("--parent-hidden", type=int, default=64, help="Teacher hidden layer width.")
    # Checkpoints
    p.add_argument("--parent-a-ckpt", default="", help="Path to parent_A state-dict (.pt).")
    p.add_argument("--parent-b-ckpt", default="", help="Path to parent_B state-dict (.pt).")
    # States
    p.add_argument("--n-states", type=int, default=1000, help="Number of synthetic states (if no file).")
    p.add_argument("--states-file", default="", help="Path to .npy file of states (shape N × input_dim).")
    # Distillation hyperparameters
    p.add_argument("--temperature", type=float, default=3.0, help="Softmax temperature.")
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Blending weight for the soft distillation loss (loss = alpha*soft + (1-alpha)*hard).",
    )
    p.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clip norm.")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction.")
    p.add_argument("--loss-fn", choices=["kl", "mse"], default="kl", help="Soft distillation loss.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    # Output
    p.add_argument("--output-dir", default="checkpoints/distillation", help="Directory for saved checkpoints.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = DistillationConfig(
        temperature=args.temperature,
        alpha=args.alpha,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        val_fraction=args.val_fraction,
        loss_fn=args.loss_fn,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    pairs_to_run = ["A", "B"] if args.pair == "both" else [args.pair]

    ckpt_map = {"A": args.parent_a_ckpt, "B": args.parent_b_ckpt}

    for pair in pairs_to_run:
        _run_pair(
            pair=pair,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            parent_hidden=args.parent_hidden,
            parent_ckpt=ckpt_map[pair],
            n_states=args.n_states,
            states_file=args.states_file,
            cfg=cfg,
            output_dir=args.output_dir,
            rng=rng,
        )

    print("\nDistillation complete.")


if __name__ == "__main__":
    main()
