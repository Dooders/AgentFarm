#!/usr/bin/env python3
"""Run the full recombination pipeline for two CartPole-v1 parents.

This script orchestrates all steps of the neural recombination pipeline,
starting from two trained ``BaseQNetwork`` parent checkpoints (``parent_A.pt``
and ``parent_B.pt``) and finishing with a JSON validation report.

Pipeline stages
---------------
1. **Train parents** (optional) — uses shared ``cartpole_dqn_training`` logic
   to produce ``parent_A.pt`` / ``parent_B.pt`` if they don't already
   exist (or if ``--force-train`` is set).
2. **Crossover + fine-tune child** — combines parent weights via
   :func:`~farm.core.decision.training.crossover.crossover_quantized_state_dict`
   and fine-tunes the resulting child against parent A using
   :class:`~farm.core.decision.training.finetune.FineTuner`.
3. **Validate recombination** — evaluates child vs both parents via
   :class:`~farm.core.decision.training.recombination_eval.RecombinationEvaluator`
   and writes a JSON report.

How to run
----------
::

    # Full pipeline from scratch (train + recombine + validate)
    python scripts/run_cartpole_recombination.py

    # Use existing parent checkpoints
    python scripts/run_cartpole_recombination.py \\
        --parent-a-ckpt checkpoints/cartpole/parent_A.pt \\
        --parent-b-ckpt checkpoints/cartpole/parent_B.pt

    # Custom training then weighted crossover
    python scripts/run_cartpole_recombination.py \\
        --train-episodes 300 \\
        --crossover-mode weighted \\
        --crossover-alpha 0.6 \\
        --finetune-epochs 15 \\
        --output-dir checkpoints/cartpole_run1

    # Skip threshold enforcement (just produce the report)
    python scripts/run_cartpole_recombination.py --report-only

Outputs
-------
All files are written under ``<output-dir>/``:

``parent_A.pt``, ``parent_A.pt.json``
    Parent A checkpoint and metadata (training stage).
``parent_B.pt``, ``parent_B.pt.json``
    Parent B checkpoint and metadata (training stage).
``replay_states_A.npy``, ``replay_states_B.npy`` (and legacy ``replay_states.npy``)
    Per-parent replay state exports (shape ``(N, 4)``). The pipeline prefers B,
    then A, then the legacy single file when choosing a default states path.
``child_finetuned.pt``, ``child_finetuned.pt.json``
    Fine-tuned child checkpoint and metadata (recombination stage).
``recombination_validation.json``
    Full validation report (passed / comparison metrics / thresholds).

CartPole dimensions
-------------------
``input_dim = 4``  (cart pos, cart vel, pole angle, pole angular vel)
``output_dim = 2`` (push left / push right)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Optional

import torch

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts_dir = os.path.join(_repo_root, "scripts")
for _p in (_repo_root, _scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from farm.core.decision.base_dqn import BaseQNetwork  # noqa: E402
from farm.core.decision.training.crossover import (  # noqa: E402
    CROSSOVER_MODES,
    crossover_quantized_state_dict,
)
from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_base_qnetwork_checkpoint,
    load_distillation_states,
)
from farm.core.decision.training.finetune import (  # noqa: E402
    FineTuner,
    load_finetuning_config_from_yaml,
)
from farm.core.decision.training.recombination_eval import (  # noqa: E402
    RecombinationEvaluator,
    RecombinationThresholds,
)

from cartpole_dqn_training import parse_torch_device, train_cartpole_parent  # noqa: E402

# CartPole-v1 fixed dimensions
_INPUT_DIM = 4
_OUTPUT_DIM = 2


# ---------------------------------------------------------------------------
# Stage 1: train parents
# ---------------------------------------------------------------------------


def _train_parent(
    label: str,
    episodes: int,
    hidden_size: int,
    lr: float,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
    tau: float,
    memory_size: int,
    batch_size: int,
    seed: Optional[int],
    output_dir: str,
    log_every: int,
    device: torch.device,
    max_replay_states: Optional[int],
) -> str:
    """Train one CartPole parent and return its checkpoint path."""
    print(f"\n[Stage 1] Training parent_{label}  ({episodes} episodes, seed={seed})")
    result = train_cartpole_parent(
        label=label,
        episodes=episodes,
        hidden_size=hidden_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        tau=tau,
        memory_size=memory_size,
        batch_size=batch_size,
        seed=seed,
        output_dir=output_dir,
        log_every=log_every,
        device=device,
        max_replay_states=max_replay_states,
    )
    print(
        f"  ✓ parent_{label} → {result.checkpoint_path}  "
        f"(mean last-50: {result.mean_reward_last_50:.1f})"
    )
    return result.checkpoint_path


# ---------------------------------------------------------------------------
# Stage 2: crossover + fine-tune
# ---------------------------------------------------------------------------


def _infer_hidden_size(path: str, fallback: int) -> int:
    """Read first-layer width from a ``BaseQNetwork`` state dict checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Parent checkpoint not found: {path!r}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at {path!r} does not contain a state dict (got {type(state).__name__})."
        )
    key = "network.0.weight"
    if key in state:
        return int(state[key].shape[0])
    return fallback


def _load_parent_network(path: str, hidden_fallback: int) -> BaseQNetwork:
    """Load a parent ``BaseQNetwork`` with the same validation as distillation scripts."""
    hidden = _infer_hidden_size(path, hidden_fallback)
    net = load_base_qnetwork_checkpoint(
        path,
        _INPUT_DIM,
        _OUTPUT_DIM,
        hidden,
        loaded_template="  Loaded parent network from: {path}",
    )
    net.eval()
    return net


def _assert_parent_state_dicts_compatible(
    sd_a: dict,
    sd_b: dict,
    path_a: str,
    path_b: str,
) -> None:
    """Crossover requires identical keys and tensor shapes between parents."""
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    if keys_a != keys_b:
        raise ValueError(
            f"Parent checkpoints have different state-dict keys ({path_a!r} vs {path_b!r})."
        )
    for key in keys_a:
        ta, tb = sd_a[key], sd_b[key]
        if ta.shape != tb.shape:
            raise ValueError(
                f"Parent checkpoints shape mismatch on {key!r}: "
                f"{tuple(ta.shape)} vs {tuple(tb.shape)} ({path_a!r} vs {path_b!r})."
            )


def _recombine(
    parent_a_ckpt: str,
    parent_b_ckpt: str,
    states_file: str,
    hidden_size: int,
    crossover_mode: str,
    crossover_alpha: float,
    crossover_seed: Optional[int],
    finetune_epochs: int,
    finetune_lr: float,
    finetune_batch: int,
    finetune_seed: Optional[int],
    output_dir: str,
    device: torch.device,
) -> str:
    """Crossover two parents and fine-tune the child. Returns child ckpt path."""
    print("\n[Stage 2] Crossover + fine-tune")
    parent_a = _load_parent_network(parent_a_ckpt, hidden_size)
    parent_b = _load_parent_network(parent_b_ckpt, hidden_size)
    _assert_parent_state_dicts_compatible(
        parent_a.state_dict(),
        parent_b.state_dict(),
        parent_a_ckpt,
        parent_b_ckpt,
    )
    inferred_hidden = int(parent_a.network[0].out_features)
    print(f"  parents loaded  (hidden={inferred_hidden})")

    # Crossover
    child_sd = crossover_quantized_state_dict(
        parent_a.state_dict(),
        parent_b.state_dict(),
        mode=crossover_mode,
        alpha=crossover_alpha,
        seed=crossover_seed,
    )
    child = BaseQNetwork(_INPUT_DIM, _OUTPUT_DIM, hidden_size=inferred_hidden)
    child.load_state_dict(child_sd)
    print(f"  crossover done  (mode={crossover_mode!r}, alpha={crossover_alpha})")

    # Load states
    states = load_distillation_states(
        states_file, n_states=2000, input_dim=_INPUT_DIM, seed=finetune_seed
    )
    print(f"  states shape    : {states.shape}")

    # Fine-tune
    base_cfg = load_finetuning_config_from_yaml(None)
    cfg = dataclasses.replace(
        base_cfg,
        epochs=finetune_epochs,
        learning_rate=finetune_lr,
        batch_size=finetune_batch,
        seed=finetune_seed,
    )
    tuner = FineTuner(reference=parent_a, child=child, config=cfg, device=device)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "child_finetuned.pt")
    metrics = tuner.finetune(states, checkpoint_path=ckpt_path)

    if metrics.train_losses:
        print(f"  final train loss : {metrics.train_losses[-1]:.6f}")
    if metrics.val_losses:
        print(f"  best  val  loss  : {metrics.best_val_loss:.6f} (epoch {metrics.best_epoch})")
        print(f"  final agreement  : {metrics.action_agreements[-1]*100:.1f}%")
    print(f"  ✓ child → {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Stage 3: validate recombination
# ---------------------------------------------------------------------------


def _validate(
    parent_a_ckpt: str,
    parent_b_ckpt: str,
    child_ckpt: str,
    states_file: str,
    hidden_size: int,
    report_dir: str,
    include_parent_baseline: bool,
    report_only: bool,
    min_action_agreement: float,
    max_kl: float,
    max_mse: float,
    min_cosine: float,
    device: torch.device,
) -> bool:
    """Run RecombinationEvaluator and write the JSON report. Returns passed."""
    print("\n[Stage 3] Recombination validation")
    parent_a = _load_parent_network(parent_a_ckpt, hidden_size)
    parent_b = _load_parent_network(parent_b_ckpt, hidden_size)
    child = _load_parent_network(child_ckpt, hidden_size)

    states = load_distillation_states(
        states_file, n_states=2000, input_dim=_INPUT_DIM, seed=0
    )
    print(f"  evaluation states: {states.shape}")

    thresholds = RecombinationThresholds(
        min_action_agreement=min_action_agreement,
        max_kl_divergence=max_kl,
        max_mse=max_mse,
        min_cosine_similarity=min_cosine,
        report_only=report_only,
    )
    evaluator = RecombinationEvaluator(
        parent_a, parent_b, child,
        thresholds=thresholds,
        device=device,
    )
    report = evaluator.evaluate(
        states,
        include_parent_baseline=include_parent_baseline,
        k_values=[1, 2],
        states_source=states_file if states_file else "synthetic_standard_normal",
        model_paths={
            "parent_a": parent_a_ckpt,
            "parent_b": parent_b_ckpt,
            "child": child_ckpt,
        },
    )

    report_dict = report.to_dict()

    # Print summary
    sep = "=" * 60
    summary = report_dict.get("summary", {})
    print(f"\n  {sep}")
    print("  Recombination report")
    print(f"  {sep}")
    print(f"  Child ↔ Parent A agreement : {summary.get('child_agrees_with_parent_a', 0):.4f}")
    print(f"  Child ↔ Parent B agreement : {summary.get('child_agrees_with_parent_b', 0):.4f}")
    oracle = summary.get("oracle_agreement")
    if oracle is not None:
        print(f"  Oracle agreement           : {oracle:.4f}")
    print(f"  Overall passed             : {report_dict.get('passed', False)}")

    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "recombination_validation.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, allow_nan=False)
    print(f"\n  ✓ Report → {out_path}")
    return bool(report.passed)


def _default_replay_states_path(output_dir: str) -> str:
    """Prefer B's replay export, then A's, then legacy single file."""
    for name in ("replay_states_B.npy", "replay_states_A.npy", "replay_states.npy"):
        candidate = os.path.join(output_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(output_dir, "replay_states_B.npy")


def _require_parent_checkpoint(path: str, which: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Parent {which} checkpoint is not a readable file: {path!r}. "
            "Omit --parent-*-ckpt to train defaults under --output-dir, or fix the path."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the full CartPole recombination pipeline: "
            "train parents → crossover + fine-tune → validate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Existing checkpoints (skip training if provided)
    p.add_argument(
        "--parent-a-ckpt", default="",
        help="Existing parent A checkpoint. If absent, parent A is trained.",
    )
    p.add_argument(
        "--parent-b-ckpt", default="",
        help="Existing parent B checkpoint. If absent, parent B is trained.",
    )
    p.add_argument(
        "--force-train", action="store_true",
        help="Re-train parents even if checkpoints already exist.",
    )
    # Training
    p.add_argument("--train-episodes", type=int, default=200, help="Episodes per parent.")
    p.add_argument("--train-lr", type=float, default=1e-3, help="Adam LR for parent training.")
    p.add_argument("--train-gamma", type=float, default=0.99)
    p.add_argument("--train-epsilon-start", type=float, default=1.0)
    p.add_argument("--train-epsilon-min", type=float, default=0.01)
    p.add_argument("--train-epsilon-decay", type=float, default=0.995)
    p.add_argument("--train-tau", type=float, default=0.005)
    p.add_argument("--train-memory", type=int, default=10000)
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--seed-a", type=int, default=42, help="RNG seed for parent A training.")
    p.add_argument("--seed-b", type=int, default=99, help="RNG seed for parent B training.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for parent training, fine-tuning, and validation (cpu, cuda, …).",
    )
    p.add_argument(
        "--max-replay-states",
        type=int,
        default=200_000,
        help="Max replay-state rows kept per parent (-1 = unlimited; caps RAM and .npy size).",
    )
    # Architecture
    p.add_argument("--hidden-size", type=int, default=64)
    # Crossover
    p.add_argument(
        "--crossover-mode", choices=list(CROSSOVER_MODES), default="weighted",
    )
    p.add_argument("--crossover-alpha", type=float, default=0.5)
    p.add_argument("--crossover-seed", type=int, default=None)
    # Fine-tuning
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    p.add_argument("--finetune-batch", type=int, default=64)
    p.add_argument("--finetune-seed", type=int, default=0)
    # States
    p.add_argument(
        "--states-file", default="",
        help=(
            "Path to replay states .npy. Default: first existing among "
            "replay_states_B.npy, replay_states_A.npy, replay_states.npy under --output-dir."
        ),
    )
    # Validation thresholds
    p.add_argument("--min-action-agreement", type=float, default=0.50)
    p.add_argument("--max-kl-divergence", type=float, default=2.0)
    p.add_argument("--max-mse", type=float, default=10.0)
    p.add_argument("--min-cosine-similarity", type=float, default=0.50)
    p.add_argument(
        "--include-parent-baseline", action="store_true",
        help="Also compute parent A vs parent B comparison.",
    )
    p.add_argument(
        "--report-only", action="store_true",
        help="Write validation report without applying pass/fail thresholds.",
    )
    # Output
    p.add_argument("--output-dir", default="checkpoints/cartpole")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    device = parse_torch_device(args.device)
    max_replay_states: Optional[int] = (
        None if args.max_replay_states < 0 else args.max_replay_states
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print("CartPole recombination pipeline")
    print(f"Output directory : {out}")
    print(f"Device           : {device}")
    print(f"Crossover mode   : {args.crossover_mode}  (alpha={args.crossover_alpha})")
    print(f"{sep}")

    # -----------------------------------------------------------------------
    # Stage 1: train parents (skip if checkpoints exist and --force-train not set)
    # -----------------------------------------------------------------------
    parent_a_ckpt = args.parent_a_ckpt
    parent_b_ckpt = args.parent_b_ckpt

    default_a = os.path.join(out, "parent_A.pt")
    default_b = os.path.join(out, "parent_B.pt")

    train_common = dict(
        hidden_size=args.hidden_size,
        lr=args.train_lr,
        gamma=args.train_gamma,
        epsilon_start=args.train_epsilon_start,
        epsilon_min=args.train_epsilon_min,
        epsilon_decay=args.train_epsilon_decay,
        tau=args.train_tau,
        memory_size=args.train_memory,
        batch_size=args.train_batch,
        output_dir=out,
        log_every=args.log_every,
        device=device,
        max_replay_states=max_replay_states,
    )

    need_a = not parent_a_ckpt or args.force_train
    need_b = not parent_b_ckpt or args.force_train

    if need_a and (not os.path.isfile(default_a) or args.force_train):
        parent_a_ckpt = _train_parent(
            "A", episodes=args.train_episodes, seed=args.seed_a, **train_common
        )
    elif not parent_a_ckpt:
        parent_a_ckpt = default_a
        print(f"\n[Stage 1] Skipping parent A training — using {parent_a_ckpt}")

    if need_b and (not os.path.isfile(default_b) or args.force_train):
        parent_b_ckpt = _train_parent(
            "B", episodes=args.train_episodes, seed=args.seed_b, **train_common
        )
    elif not parent_b_ckpt:
        parent_b_ckpt = default_b
        print(f"\n[Stage 1] Skipping parent B training — using {parent_b_ckpt}")

    _require_parent_checkpoint(parent_a_ckpt, "A")
    _require_parent_checkpoint(parent_b_ckpt, "B")

    # -----------------------------------------------------------------------
    # Stage 2: crossover + fine-tune
    # -----------------------------------------------------------------------
    states_file = args.states_file or _default_replay_states_path(out)
    if not os.path.isfile(states_file):
        print(
            f"\n[States] No replay states file at {states_file!r} — "
            "using synthetic Gaussian states for fine-tune and validation."
        )
        states_file = ""

    child_ckpt = _recombine(
        parent_a_ckpt=parent_a_ckpt,
        parent_b_ckpt=parent_b_ckpt,
        states_file=states_file,
        hidden_size=args.hidden_size,
        crossover_mode=args.crossover_mode,
        crossover_alpha=args.crossover_alpha,
        crossover_seed=args.crossover_seed,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_batch=args.finetune_batch,
        finetune_seed=args.finetune_seed,
        output_dir=out,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Stage 3: validate
    # -----------------------------------------------------------------------
    passed = _validate(
        parent_a_ckpt=parent_a_ckpt,
        parent_b_ckpt=parent_b_ckpt,
        child_ckpt=child_ckpt,
        states_file=states_file,
        hidden_size=args.hidden_size,
        report_dir=out,
        include_parent_baseline=args.include_parent_baseline,
        report_only=args.report_only,
        min_action_agreement=args.min_action_agreement,
        max_kl=args.max_kl_divergence,
        max_mse=args.max_mse,
        min_cosine=args.min_cosine_similarity,
        device=device,
    )

    print(f"\n{sep}")
    print("Pipeline complete.")
    print(f"All outputs in    : {out}")
    print(f"Validation passed : {passed}")
    print(f"{sep}\n")

    if not passed and not args.report_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
