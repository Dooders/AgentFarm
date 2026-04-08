#!/usr/bin/env python
"""Quantization-aware training (QAT) fine-tuning for distilled student Q-networks.

This script implements the **weight-only QAT** path for ``StudentQNetwork``
checkpoints produced by ``scripts/run_distillation.py``.

Recommended workflow
--------------------
1. Distil float student (``scripts/run_distillation.py``):
   Produces ``student_A.pt`` and ``student_B.pt``.

2. PTQ baseline (``scripts/quantize_distilled.py``):
   Produces ``student_A_int8.pt`` / ``student_B_int8.pt`` (no training cost).

3. **(Optional)** If PTQ accuracy is insufficient, run QAT:
   Produces ``student_A_qat.pt`` (float QAT) and ``student_A_qat_int8.pt``
   (converted int8 checkpoint, same format as PTQ output).

QAT vs PTQ tradeoff
-------------------
* **PTQ dynamic** – fast, zero training cost; use first.  Typically ≥ 90%
  action agreement.
* **QAT weight-only** – adds fine-tuning epochs; use when PTQ agreement is
  unacceptable (e.g. < 85%) or Q-error degrades task metrics.  QAT adapts
  the student weights to int8 quantisation noise, typically recovering 1–5%
  agreement.

Usage
-----
QAT finetune pair A (from distillation checkpoints)::

    python scripts/qat_distilled.py \\
        --teacher-ckpt checkpoints/distillation/parent_A.pt \\
        --student-ckpt checkpoints/distillation/student_A.pt \\
        --pair A \\
        --output-dir checkpoints/qat

QAT finetune both pairs with a states file::

    python scripts/qat_distilled.py \\
        --pair both \\
        --checkpoint-dir checkpoints/distillation \\
        --states-file data/replay_states.npy \\
        --output-dir checkpoints/qat

QAT hyperparameters (key options)
----------------------------------
=========================  ==================================================
``--epochs``               QAT fine-tuning epochs (default: 5)
``--learning-rate``        Adam learning rate (default: 1e-4)
``--batch-size``           Mini-batch size (default: 32)
``--loss-fn``              ``mse`` (default) or ``kl`` soft-label distillation
``--temperature``          Temperature for KL loss (default: 3.0)
``--alpha``                Soft/hard blend for KL mode (default: 1.0 = soft-only)
``--dtype``                Target dtype ``qint8`` (default) or ``quint8``
``--seed``                 RNG seed for reproducibility
``--no-convert``           Save only the float QAT checkpoint; skip int8 convert
=========================  ==================================================

Architecture flags
------------------
Pass ``--input-dim``, ``--output-dim``, and ``--parent-hidden`` to match the
values used in ``run_distillation.py`` (defaults: 8, 4, 64).

Output
------
For each processed pair the script writes to ``<output-dir>/``:

* ``student_<pair>_qat.pt``         – float QAT student weights (state-dict)
* ``student_<pair>_qat.pt.json``    – training config + metrics
* ``student_<pair>_qat_int8.pt``    – converted int8 model (full pickle)
* ``student_<pair>_qat_int8.pt.json``  – QAT config + notes

A comparison report (float student vs QAT int8) is printed to stdout and
written to ``<output-dir>/qat_report_<pair>.json``.

Known limitations
-----------------
* Weight-only scope: activations remain in float32 (same as PTQ dynamic).
* ``LayerNorm`` layers are not quantized (not fuseable under fbgemm/qnnpack).
* int8 kernels accelerate on CPU only; CUDA paths dequantize weights at runtime.
* Requires PyTorch ≥ 2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Allow running directly from repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.quantize_ptq import compare_outputs  # noqa: E402
from farm.core.decision.training.quantize_qat import (  # noqa: E402
    QATConfig,
    QATTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_float_model(
    path: str,
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    is_teacher: bool,
) -> torch.nn.Module:
    """Load a float BaseQNetwork or StudentQNetwork from a state-dict checkpoint."""
    if is_teacher:
        model = BaseQNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
        )
    else:
        model = StudentQNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            parent_hidden_size=hidden_size,
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at '{path}' must be a state-dict (got {type(state).__name__}). "
            "Use checkpoints produced by run_distillation.py."
        )
    model.load_state_dict(state)
    model.eval()
    return model


def _load_states(
    states_file: str,
    n_states: int,
    input_dim: int,
    seed: int,
) -> np.ndarray:
    if states_file:
        if not os.path.isfile(states_file):
            raise FileNotFoundError(f"States file not found: {states_file!r}")
        states = np.load(states_file).astype("float32")
        if states.ndim != 2:
            raise ValueError(
                f"Loaded states must be a 2-D array with shape (N, input_dim); got {states.shape!r}"
            )
        print(f"  Loaded states from {states_file}: shape={states.shape}")
        return states
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((n_states, input_dim)).astype("float32")
    print(f"  Using {n_states} synthetic random states (shape={states.shape})")
    return states


def _resolve_ckpt(pair: str, explicit: str, checkpoint_dir: str, pattern: str) -> str:
    if explicit:
        return explicit
    if checkpoint_dir:
        return os.path.join(checkpoint_dir, pattern.format(pair=pair))
    return pattern.format(pair=pair)


def _run_pair(
    pair: str,
    teacher_ckpt: str,
    student_ckpt: str,
    input_dim: int,
    output_dim: int,
    parent_hidden: int,
    config: QATConfig,
    states: np.ndarray,
    output_dir: str,
    convert: bool,
) -> None:
    """Run QAT finetune + (optionally) convert for one teacher-student pair."""
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"QAT finetune: student_{pair}")
    print(f"{sep}")

    print(f"  Loading teacher: {teacher_ckpt}")
    teacher = _load_float_model(
        teacher_ckpt, input_dim, output_dim, parent_hidden, is_teacher=True
    )

    print(f"  Loading float student: {student_ckpt}")
    student = _load_float_model(
        student_ckpt, input_dim, output_dim, parent_hidden, is_teacher=False
    )
    float_params = sum(p.numel() for p in student.parameters())
    print(f"  Student params: {float_params:,}")

    # QAT trainer
    trainer = QATTrainer(teacher, student, config)

    float_ckpt = os.path.join(output_dir, f"student_{pair}_qat.pt")
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"\n  Running QAT ({config.epochs} epochs, lr={config.learning_rate}, "
        f"loss={config.loss_fn}, dtype={config.dtype}) …"
    )
    metrics = trainer.train(states, checkpoint_path=float_ckpt)

    print(f"\n  --- QAT training summary for student_{pair} ---")
    print(f"  Epochs trained      : {len(metrics.train_losses)}")
    print(f"  Best epoch          : {metrics.best_epoch}")
    print(f"  Best val loss       : {metrics.best_val_loss:.6f}")
    print(f"  Final train loss    : {metrics.train_losses[-1]:.6f}")
    if metrics.action_agreements:
        print(f"  Final val agreement : {metrics.action_agreements[-1]*100:.2f}%")
    print(f"  Elapsed (s)         : {metrics.elapsed_seconds:.3f}")
    print(f"  Float QAT ckpt      : {float_ckpt}")

    report: dict = {
        "pair": pair,
        "teacher_checkpoint": teacher_ckpt,
        "student_checkpoint": student_ckpt,
        "qat_config": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "loss_fn": config.loss_fn,
            "temperature": config.temperature,
            "alpha": config.alpha,
            "dtype": config.dtype,
            "scope": "weight_only",
        },
        "qat_metrics": metrics.to_dict(),
        "float_qat_checkpoint": float_ckpt,
    }

    if convert:
        arch_kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "parent_hidden_size": parent_hidden,
        }
        print("\n  Converting QAT model to int8 …")
        q_model = trainer.convert()
        int8_ckpt = os.path.join(output_dir, f"student_{pair}_qat_int8.pt")
        trainer.save_quantized(q_model, int8_ckpt, arch_kwargs=arch_kwargs)
        print(f"  int8 checkpoint     : {int8_ckpt}")
        print(f"  int8 JSON metadata  : {int8_ckpt}.json")

        # Comparison: float student vs QAT int8
        print("\n  Comparing float student vs QAT-int8 outputs …")
        cmp = compare_outputs(student, q_model, states)
        print(f"  Action agreement    : {cmp['action_agreement']*100:.2f}%")
        print(f"  Mean Q-error        : {cmp['mean_q_error']:.6f}")
        print(f"  Max Q-error         : {cmp['max_q_error']:.6f}")
        print(f"  Mean cosine sim     : {cmp['mean_cosine_similarity']:.6f}")

        report["int8_checkpoint"] = int8_ckpt
        report["comparison_float_vs_qat_int8"] = cmp

    report_path = os.path.join(output_dir, f"qat_report_{pair}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    print(f"\n  Full report written : {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Weight-only QAT fine-tuning for distilled student Q-networks.\n"
            "Reads distilled float student, runs QAT, and (optionally) converts to int8."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pair", choices=["A", "B", "both"], default="A")

    # Checkpoints
    p.add_argument(
        "--checkpoint-dir",
        default="",
        help=(
            "Directory with parent_<pair>.pt / student_<pair>.pt. "
            "Used as fallback when explicit checkpoint paths are not given."
        ),
    )
    p.add_argument("--teacher-a-ckpt", default="", help="Explicit path to parent_A.pt.")
    p.add_argument("--student-a-ckpt", default="", help="Explicit path to student_A.pt.")
    p.add_argument("--teacher-b-ckpt", default="", help="Explicit path to parent_B.pt.")
    p.add_argument("--student-b-ckpt", default="", help="Explicit path to student_B.pt.")

    # Architecture
    p.add_argument("--input-dim", type=int, default=8, help="State feature dimension.")
    p.add_argument("--output-dim", type=int, default=4, help="Number of actions.")
    p.add_argument("--parent-hidden", type=int, default=64, help="Teacher hidden layer width.")

    # States
    p.add_argument(
        "--states-file",
        default="",
        help="Path to .npy state file of shape (N, input_dim). Shared with distillation.",
    )
    p.add_argument(
        "--n-states",
        type=int,
        default=1000,
        help="Number of synthetic random states to generate if --states-file is not given.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")

    # QAT hyperparameters
    p.add_argument("--epochs", type=int, default=5, help="QAT fine-tuning epochs.")
    p.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set 0 to disable.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of states held out for validation (0 to disable).",
    )
    p.add_argument(
        "--loss-fn",
        choices=["mse", "kl"],
        default="mse",
        help="Distillation loss: 'mse' (default) or 'kl' (temperature-scaled KL).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="Softmax temperature for KL loss (ignored in mse mode).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Soft/hard blend for KL mode (1.0 = pure soft; 0.0 = pure hard).",
    )
    p.add_argument(
        "--dtype",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Target quantization dtype.",
    )
    p.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip the int8 convert step; save only the float QAT checkpoint.",
    )

    # Output
    p.add_argument(
        "--output-dir",
        default="checkpoints/qat",
        help="Directory to write QAT checkpoints and reports.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    max_grad_norm = args.max_grad_norm if args.max_grad_norm > 0 else None
    config = QATConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_grad_norm=max_grad_norm,
        val_fraction=args.val_fraction,
        seed=args.seed,
        loss_fn=args.loss_fn,
        temperature=args.temperature,
        alpha=args.alpha,
        dtype=args.dtype,
    )

    states = _load_states(args.states_file, args.n_states, args.input_dim, args.seed)

    pairs = ["A", "B"] if args.pair == "both" else [args.pair]

    ckpt_map = {
        "A": (
            _resolve_ckpt("A", args.teacher_a_ckpt, args.checkpoint_dir, "parent_{pair}.pt"),
            _resolve_ckpt("A", args.student_a_ckpt, args.checkpoint_dir, "student_{pair}.pt"),
        ),
        "B": (
            _resolve_ckpt("B", args.teacher_b_ckpt, args.checkpoint_dir, "parent_{pair}.pt"),
            _resolve_ckpt("B", args.student_b_ckpt, args.checkpoint_dir, "student_{pair}.pt"),
        ),
    }

    for pair in pairs:
        teacher_ckpt, student_ckpt = ckpt_map[pair]
        _run_pair(
            pair=pair,
            teacher_ckpt=teacher_ckpt,
            student_ckpt=student_ckpt,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            parent_hidden=args.parent_hidden,
            config=config,
            states=states,
            output_dir=args.output_dir,
            convert=not args.no_convert,
        )

    print("\nQAT complete.")


if __name__ == "__main__":
    main()
