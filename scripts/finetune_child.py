#!/usr/bin/env python
"""Fine-tune a crossover child Q-network on a target dataset.

This script loads two parent checkpoints, constructs a child network via
crossover, then fine-tunes the child using a frozen reference model
(defaulting to parent A) as a soft-label teacher.  Before/after metrics are
printed and saved alongside the checkpoint.

Usage
-----
Fine-tune with defaults from ``crossover_child_finetune`` in
``farm/config/default.yaml`` (synthetic states, parent A as reference)::

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

Use a custom YAML file for fine-tune defaults (must define the same section)::

    python scripts/finetune_child.py --config-yaml my_experiment.yaml ...

Notes
-----
When no parent checkpoints are provided, random weights are used (useful for
integration testing).  The crossover seed and fine-tuning seed are independent;
pass ``--crossover-seed`` to control the crossover RNG separately from
``--seed`` (which controls fine-tuning).

Parent checkpoint hidden width is inferred automatically from ``network.0.weight``.
If it differs from ``--hidden-size``, the inferred value is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Any, Dict

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
from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_distillation_states,
)
from farm.core.decision.training.finetune import (  # noqa: E402
    FINETUNE_OPTIMIZERS,
    QUANTIZATION_APPLIED_MODES,
    FineTuner,
    load_finetuning_config_from_yaml,
)

def _infer_hidden_size(state: Dict[str, Any], label: str) -> int:
    """Infer hidden width from ``network.0.weight`` for Base/Student checkpoints."""
    key = "network.0.weight"
    if key not in state:
        raise ValueError(
            f"{label} checkpoint missing required key {key!r}; cannot infer hidden size."
        )
    tensor = state[key]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(
            f"{label} checkpoint key {key!r} must be rank-2 tensor; got {type(tensor).__name__}."
        )
    hidden_size = int(tensor.shape[0])
    if hidden_size < 1:
        raise ValueError(f"{label} inferred hidden size must be >= 1 (got {hidden_size}).")
    return hidden_size


def _load_parent_checkpoint_auto(
    path: str,
    input_dim: int,
    output_dim: int,
    fallback_hidden_size: int,
    label: str,
) -> BaseQNetwork:
    """Load parent checkpoint and auto-detect hidden size when possible."""
    if not path:
        print("  No checkpoint provided – using random weights.")
        return BaseQNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=fallback_hidden_size,
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Parent checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at '{path}' does not contain a state dict (got {type(state).__name__})."
        )

    inferred_hidden = _infer_hidden_size(state, label)
    if inferred_hidden != fallback_hidden_size:
        print(
            f"  {label}: inferred hidden size {inferred_hidden} from checkpoint; "
            f"overriding --hidden-size {fallback_hidden_size}."
        )

    model = BaseQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=inferred_hidden,
    )
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded network weights from: {path}")
    return model


def _merged_finetune_config(args: argparse.Namespace) -> FineTuningConfig:
    yaml_path = (args.config_yaml or "").strip() or None
    base = load_finetuning_config_from_yaml(yaml_path)
    mapping = (
        ("lr", "learning_rate"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("max_grad_norm", "max_grad_norm"),
        ("val_fraction", "val_fraction"),
        ("loss_fn", "loss_fn"),
        ("temperature", "temperature"),
        ("alpha", "alpha"),
        ("lr_patience", "lr_schedule_patience"),
        ("lr_factor", "lr_schedule_factor"),
        ("seed", "seed"),
        ("quantization_applied", "quantization_applied"),
        ("optimizer", "optimizer"),
        ("early_stopping_patience", "early_stopping_patience"),
    )
    overrides = {
        field: getattr(args, attr) for attr, field in mapping if hasattr(args, attr)
    }
    return dataclasses.replace(base, **overrides)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a crossover child Q-network on a target dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Fine-tuning hyperparameters default to the ``crossover_child_finetune`` section "
            "of farm/config/default.yaml (or the file given by --config-yaml). "
            "Pass flags such as --lr or --epochs only to override those fields."
        ),
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
    # Fine-tuning: optional overrides (defaults from YAML section)
    p.add_argument(
        "--config-yaml",
        default="",
        help="YAML file containing crossover_child_finetune (default: farm/config/default.yaml).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=argparse.SUPPRESS,
        help="Override learning_rate.",
    )
    p.add_argument("--epochs", type=int, default=argparse.SUPPRESS, help="Override epochs.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Override batch_size.",
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=argparse.SUPPRESS,
        help="Override max_grad_norm.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help="Override val_fraction.",
    )
    p.add_argument(
        "--loss-fn",
        choices=["kl", "mse"],
        default=argparse.SUPPRESS,
        help="Override loss_fn.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=argparse.SUPPRESS,
        help="Override temperature.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=argparse.SUPPRESS,
        help="Override alpha (soft vs hard loss blend).",
    )
    p.add_argument(
        "--lr-patience",
        type=int,
        default=argparse.SUPPRESS,
        help="Override lr_schedule_patience (ReduceLROnPlateau; 0 = off).",
    )
    p.add_argument(
        "--lr-factor",
        type=float,
        default=argparse.SUPPRESS,
        help="Override lr_schedule_factor.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Override fine-tuning seed.",
    )
    p.add_argument(
        "--quantization-applied",
        choices=list(QUANTIZATION_APPLIED_MODES),
        default=argparse.SUPPRESS,
        help="Override quantization_applied.",
    )
    p.add_argument(
        "--optimizer",
        choices=list(FINETUNE_OPTIMIZERS),
        default=argparse.SUPPRESS,
        help="Override optimizer.",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=argparse.SUPPRESS,
        help="Override early_stopping_patience (0 = disabled).",
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
    cfg = _merged_finetune_config(args)
    yaml_note = (args.config_yaml or "").strip() or "farm/config/default.yaml"

    print("\n" + "=" * 60)
    print("Fine-tuning crossover child Q-network")
    print("=" * 60)
    print(f"\nFine-tune defaults: crossover_child_finetune ← {yaml_note!r}")

    # 1. Load parents
    print("\n[1/4] Loading parents …")
    parent_a = _load_parent_checkpoint_auto(
        args.parent_a_ckpt,
        args.input_dim,
        args.output_dim,
        args.hidden_size,
        label="parent_a",
    )
    parent_b = _load_parent_checkpoint_auto(
        args.parent_b_ckpt,
        args.input_dim,
        args.output_dim,
        args.hidden_size,
        label="parent_b",
    )

    child_hidden_size = int(parent_a.network[0].out_features)
    parent_b_hidden_size = int(parent_b.network[0].out_features)
    if child_hidden_size != parent_b_hidden_size:
        raise ValueError(
            "Parent hidden sizes must match for crossover initialization: "
            f"parent_a={child_hidden_size}, parent_b={parent_b_hidden_size}."
        )

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
        hidden_size=child_hidden_size,
    )
    child.load_state_dict(child_sd)

    # 3. Load target dataset
    print("\n[3/4] Preparing target dataset …")
    states = load_distillation_states(
        args.states_file, args.n_states, args.input_dim, cfg.seed
    )

    # 4. Fine-tune child against parent A as reference
    print(
        f"\n[4/4] Fine-tuning child (reference = parent A), "
        f"up to {cfg.epochs} epoch(s), optimizer={cfg.optimizer!r} …"
    )
    tuner = FineTuner(reference=parent_a, child=child, config=cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "child_finetuned.pt")
    metrics = tuner.finetune(states, checkpoint_path=ckpt_path)

    # Summary
    print("\n--- Fine-tuning results ---")
    if metrics.train_losses:
        print(f"  Final train loss   : {metrics.train_losses[-1]:.6f}")
    if metrics.val_losses:
        print(f"  Initial val loss   : {metrics.initial_val_loss:.6f}")
        print(f"  Initial agreement  : {metrics.initial_action_agreement * 100:.1f}%")
        print(f"  Best val loss      : {metrics.best_val_loss:.6f}  (epoch {metrics.best_epoch})")
        print(f"  Final agreement    : {metrics.action_agreements[-1] * 100:.1f}%")
        improvement = metrics.initial_val_loss - metrics.best_val_loss
        print(f"  Val loss Δ         : {improvement:+.6f}")
    if metrics.early_stopped:
        print(f"  Early stop         : yes (completed {len(metrics.train_losses)} epoch(s))")
    print(f"  Checkpoint saved   : {ckpt_path}")
    print(f"  Metadata saved     : {ckpt_path}.json")

    # QAT mode: optionally convert and save int8 model
    if cfg.quantization_applied != "none":
        print(f"\n  QAT mode ({cfg.quantization_applied}): converting to int8 …")
        q_model = tuner.convert()
        int8_path = os.path.join(args.output_dir, "child_finetuned_qat_int8.pt")
        tuner.save_quantized(q_model, int8_path)
        print(f"  int8 model saved   : {int8_path}")
        print(f"  int8 metadata      : {int8_path}.json")

    print("\nFine-tuning complete.")


if __name__ == "__main__":
    main()
