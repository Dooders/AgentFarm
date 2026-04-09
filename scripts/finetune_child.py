#!/usr/bin/env python
"""Fine-tune a crossover child Q-network on a target dataset.

This script loads two parent checkpoints, constructs a child network via
crossover, then fine-tunes the child using a frozen reference model
(defaulting to parent A) as a soft-label teacher.  Before/after metrics are
printed and saved alongside the checkpoint.

Usage
-----
Fine-tune with defaults (synthetic states, parent A as reference)::

    python scripts/finetune_child.py \\
        --parent-a-ckpt checkpoints/parent_a.pt \\
        --parent-b-ckpt checkpoints/parent_b.pt

Provide a real state buffer and override hyperparameters::

    python scripts/finetune_child.py \\
        --parent-a-ckpt checkpoints/parent_a.pt \\
        --parent-b-ckpt checkpoints/parent_b.pt \\
        --states-file data/replay_states.npy \\
        --crossover-mode weighted \\
        --crossover-alpha 0.6 \\
        --epochs 20 \\
        --lr 5e-4 \\
        --seed 42 \\
        --output-dir checkpoints/finetuned

Notes
-----
When no parent checkpoints are provided, random weights are used (useful for
integration testing).  The crossover seed and fine-tuning seed are independent;
pass ``--crossover-seed`` to control the crossover RNG separately from
``--seed`` (which controls fine-tuning).
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

from farm.core.decision.base_dqn import BaseQNetwork  # noqa: E402
from farm.core.decision.training.crossover import (  # noqa: E402
    CROSSOVER_MODES,
    crossover_quantized_state_dict,
)
from farm.core.decision.training.finetune import (  # noqa: E402
    QUANTIZATION_APPLIED_MODES,
    FineTuner,
    FineTuningConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_network(path: str, input_dim: int, output_dim: int, hidden_size: int) -> BaseQNetwork:
    """Load a ``BaseQNetwork`` from a state-dict checkpoint.

    If *path* is empty, a network with random weights is returned.
    """
    net = BaseQNetwork(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size)
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Parent checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(
                f"Checkpoint at '{path}' does not contain a state dict "
                f"(got {type(state).__name__})."
            )
        net.load_state_dict(state)
        print(f"  Loaded network weights from: {path}")
    else:
        print("  No checkpoint provided – using random weights.")
    return net


def _load_states(states_file: str, n_states: int, input_dim: int, seed: int | None) -> np.ndarray:
    """Load ``.npy`` states or synthesise standard-normal calibration data."""
    if states_file:
        if not os.path.isfile(states_file):
            raise FileNotFoundError(f"States file not found: {states_file!r}")
        states = np.load(states_file).astype("float32")
        if states.ndim != 2:
            raise ValueError(
                f"States must be a 2-D array with shape (N, input_dim); got {states.shape!r}"
            )
        if states.shape[1] != input_dim:
            raise ValueError(
                f"States input_dim mismatch: expected {input_dim}, got {states.shape[1]}"
            )
        print(f"  Loaded states from {states_file!r}: shape={states.shape}")
        return states
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n_states, input_dim)).astype("float32")
    print(f"  Using {n_states} synthetic random states (shape={states.shape})")
    return states


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a crossover child Q-network on a target dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Architecture
    p.add_argument("--input-dim", type=int, default=8, help="State feature dimension.")
    p.add_argument("--output-dim", type=int, default=4, help="Number of actions.")
    p.add_argument("--hidden-size", type=int, default=64, help="Parent hidden layer width.")
    # Parent checkpoints
    p.add_argument("--parent-a-ckpt", default="", help="Path to parent A state-dict (.pt).")
    p.add_argument("--parent-b-ckpt", default="", help="Path to parent B state-dict (.pt).")
    # Crossover settings
    p.add_argument(
        "--crossover-mode",
        choices=list(CROSSOVER_MODES),
        default="weighted",
        help="Crossover strategy.",
    )
    p.add_argument(
        "--crossover-alpha",
        type=float,
        default=0.5,
        help="Blend / selection coefficient for crossover.",
    )
    p.add_argument("--crossover-seed", type=int, default=None, help="RNG seed for crossover.")
    # States / target dataset
    p.add_argument(
        "--n-states",
        type=int,
        default=1000,
        help="Number of synthetic states when no --states-file is provided.",
    )
    p.add_argument(
        "--states-file",
        default="",
        help="Path to .npy file of states (shape N × input_dim).",
    )
    # Fine-tuning hyperparameters
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs.")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clip norm.")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction.")
    p.add_argument("--loss-fn", choices=["kl", "mse"], default="kl", help="Soft fine-tuning loss.")
    p.add_argument("--temperature", type=float, default=3.0, help="Softmax temperature.")
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Blending weight for soft loss (loss = alpha*soft + (1-alpha)*hard).",
    )
    p.add_argument(
        "--lr-patience",
        type=int,
        default=0,
        help="ReduceLROnPlateau patience epochs (0 = disabled).",
    )
    p.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau reduction factor.",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for fine-tuning.")
    # Quantization mode
    p.add_argument(
        "--quantization-applied",
        choices=list(QUANTIZATION_APPLIED_MODES),
        default="none",
        help=(
            "Indicate whether quantization was applied to the crossover parents. "
            "When not 'none', fine-tuning uses QAT fake-quant Linear layers so the "
            "objective is minimised under the same int8 weight noise as deployment. "
            "After fine-tuning call convert() + save_quantized() to get an int8 model. "
            f"Choices: {list(QUANTIZATION_APPLIED_MODES)}"
        ),
    )
    # Output
    p.add_argument(
        "--output-dir",
        default="checkpoints/finetuned",
        help="Directory for saved checkpoints.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 60)
    print("Fine-tuning crossover child Q-network")
    print("=" * 60)

    # 1. Load parents
    print("\n[1/4] Loading parents …")
    parent_a = _load_network(args.parent_a_ckpt, args.input_dim, args.output_dim, args.hidden_size)
    parent_b = _load_network(args.parent_b_ckpt, args.input_dim, args.output_dim, args.hidden_size)

    # 2. Construct child via crossover
    print(f"\n[2/4] Crossover: mode={args.crossover_mode!r}, alpha={args.crossover_alpha} …")
    child_sd = crossover_quantized_state_dict(
        parent_a.state_dict(),
        parent_b.state_dict(),
        mode=args.crossover_mode,
        alpha=args.crossover_alpha,
        seed=args.crossover_seed,
    )
    child = BaseQNetwork(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_size=args.hidden_size,
    )
    child.load_state_dict(child_sd)

    # 3. Load target dataset
    print("\n[3/4] Preparing target dataset …")
    states = _load_states(args.states_file, args.n_states, args.input_dim, args.seed)

    # 4. Fine-tune child against parent A as reference
    print(f"\n[4/4] Fine-tuning child (reference = parent A) for {args.epochs} epochs …")
    cfg = FineTuningConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        val_fraction=args.val_fraction,
        loss_fn=args.loss_fn,
        temperature=args.temperature,
        alpha=args.alpha,
        lr_schedule_patience=args.lr_patience,
        lr_schedule_factor=args.lr_factor,
        seed=args.seed,
        quantization_applied=args.quantization_applied,
    )
    tuner = FineTuner(reference=parent_a, child=child, config=cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "child_finetuned.pt")
    metrics = tuner.finetune(states, checkpoint_path=ckpt_path)

    # Summary
    print("\n--- Fine-tuning results ---")
    if metrics.train_losses:
        print(f"  Initial val loss   : {metrics.initial_val_loss:.6f}")
        print(f"  Initial agreement  : {metrics.initial_action_agreement * 100:.1f}%")
        print(f"  Final train loss   : {metrics.train_losses[-1]:.6f}")
    if metrics.val_losses:
        print(f"  Best val loss      : {metrics.best_val_loss:.6f}  (epoch {metrics.best_epoch})")
        print(f"  Final agreement    : {metrics.action_agreements[-1] * 100:.1f}%")
        improvement = metrics.initial_val_loss - metrics.best_val_loss
        print(f"  Val loss Δ         : {improvement:+.6f}")
    print(f"  Checkpoint saved   : {ckpt_path}")
    print(f"  Metadata saved     : {ckpt_path}.json")

    # QAT mode: optionally convert and save int8 model
    if args.quantization_applied != "none":
        print(f"\n  QAT mode ({args.quantization_applied}): converting to int8 …")
        q_model = tuner.convert()
        int8_path = os.path.join(args.output_dir, "child_finetuned_qat_int8.pt")
        tuner.save_quantized(q_model, int8_path)
        print(f"  int8 model saved   : {int8_path}")
        print(f"  int8 metadata      : {int8_path}.json")

    print("\nFine-tuning complete.")


if __name__ == "__main__":
    main()
