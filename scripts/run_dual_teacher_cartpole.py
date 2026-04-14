#!/usr/bin/env python3
"""Full dual-teacher compression-first CartPole pipeline (canonical Issue #8 run).

This script is the **definitive end-to-end demonstration** of the
distill → quantize → crossover → dual-teacher fine-tune → validate workflow
described in `Dooders/AgentFarm#8`.  It differs from
``scripts/run_cartpole_recombination.py`` in three key ways:

1. **Distillation first** — parent A and parent B are each distilled into
   dedicated ``StudentQNetwork`` checkpoints *before* any crossover.
2. **Quantization before crossover** — the distilled students are compressed
   to int8 with post-training quantization (PTQ); crossover operates on the
   dequantized int8 weights.
3. **Dual-teacher fine-tuning** — the child is realigned against *both*
   teachers simultaneously using a weighted KL loss
   ``alpha_a * KL(A ‖ child) + alpha_b * KL(B ‖ child)``.  Parent B is
   no longer a passive observer: it contributes equally to every gradient
   step.

Pipeline stages
---------------
1. **Train parents** — CartPole DQN, produces ``parent_A.pt`` / ``parent_B.pt``.
2. **Distill** — ``DistillationTrainer`` on each parent, produces
   ``student_A.pt`` / ``student_B.pt``.
3. **Quantize** — ``PostTrainingQuantizer`` (dynamic PTQ), produces
   ``student_A_int8.pt`` / ``student_B_int8.pt``.
4. **Crossover** — ``crossover_quantized_state_dict``, produces
   ``child_crossover.pt`` (float32, pre-fine-tune snapshot).
5. **Dual-teacher fine-tune** — custom loop using both frozen teachers,
   produces ``child_finetuned.pt``.
6. **Validate** — per-stage + final reports:
   * ``distillation_report_A.json``, ``distillation_report_B.json``
   * ``quantization_report_A.json``, ``quantization_report_B.json``
   * ``recombination_validation.json`` (child-vs-A, child-vs-B, A-vs-B baseline)
   * ``pipeline_report.json`` — master summary of all stages

How to run
----------
::

    # Full pipeline from scratch
    python scripts/run_dual_teacher_cartpole.py --output-dir out/dual_teacher

    # Use existing parent checkpoints, skip training
    python scripts/run_dual_teacher_cartpole.py \\
        --parent-a-ckpt out/cartpole/parent_A.pt \\
        --parent-b-ckpt out/cartpole/parent_B.pt \\
        --output-dir out/dual_teacher

    # Skip threshold enforcement (report-only)
    python scripts/run_dual_teacher_cartpole.py --report-only

    # Custom distillation + weighted crossover
    python scripts/run_dual_teacher_cartpole.py \\
        --distill-epochs 20 --distill-temperature 4.0 \\
        --crossover-mode weighted --crossover-alpha 0.6 \\
        --finetune-epochs 15 --finetune-lr 5e-4 \\
        --output-dir out/dual_teacher_run2

Outputs
-------
All artefacts are written under ``<output-dir>/``:

``parent_A.pt``, ``parent_B.pt``
    DQN parent checkpoints (training stage).
``replay_states_A.npy``, ``replay_states_B.npy``
    Per-parent CartPole replay-buffer state exports.
``student_A.pt``, ``student_B.pt``
    Distilled ``StudentQNetwork`` checkpoints (distillation stage).
``student_A_int8.pt``, ``student_B_int8.pt``
    PTQ-quantized student checkpoints (quantization stage).
``child_crossover.pt``
    Float32 child produced by crossover *before* fine-tuning (snapshot).
``child_finetuned.pt``
    Dual-teacher-fine-tuned child (final deployment artifact).
``distillation_report_A.json``, ``distillation_report_B.json``
    Per-pair distillation validation (KL, MSE, action agreement, etc.).
``quantization_report_A.json``, ``quantization_report_B.json``
    PTQ fidelity/size/latency reports.
``recombination_validation.json``
    Final child-vs-A, child-vs-B, A-vs-B evaluation.
``pipeline_report.json``
    Master JSON aggregating all stage metrics and artefact paths.

CartPole dimensions
-------------------
``input_dim = 4`` (cart pos, cart vel, pole angle, pole angular vel)
``output_dim = 2`` (push left / push right)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_scripts_dir = os.path.join(_repo_root, "scripts")
for _p in (_repo_root, _scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork  # noqa: E402
from farm.core.decision.training.crossover import (  # noqa: E402
    CROSSOVER_MODES,
    ChildArchitectureSpec,
    crossover_quantized_state_dict,
    initialize_child_from_crossover,
)
from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_base_qnetwork_checkpoint,
    load_distillation_states,
    load_float_student_checkpoint,
)
from farm.core.decision.training.quantize_ptq import (  # noqa: E402
    PostTrainingQuantizer,
    QuantizationConfig,
    compare_outputs,
    load_quantized_checkpoint,
)
from farm.core.decision.training.recombination_eval import (  # noqa: E402
    RecombinationEvaluator,
    RecombinationThresholds,
)
from farm.core.decision.training.trainer_distill import (  # noqa: E402
    DistillationConfig,
    DistillationTrainer,
)

from cartpole_dqn_training import parse_torch_device, train_cartpole_parent  # noqa: E402

# CartPole-v1 fixed dimensions
_INPUT_DIM = 4
_OUTPUT_DIM = 2

# ---------------------------------------------------------------------------
# Stage 1: train CartPole parents
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
    """Train one CartPole DQN parent and return its checkpoint path."""
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
# Stage 2: distill parent A/B → student A/B
# ---------------------------------------------------------------------------


def _distill_student(
    pair: str,
    parent_ckpt: str,
    states: np.ndarray,
    hidden_size: int,
    cfg: DistillationConfig,
    output_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    """Distil one parent into a StudentQNetwork. Returns (ckpt_path, metrics_dict)."""
    print(f"\n[Stage 2] Distilling parent_{pair} → student_{pair}")
    teacher = BaseQNetwork(_INPUT_DIM, _OUTPUT_DIM, hidden_size=hidden_size)
    state = torch.load(parent_ckpt, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state)
    teacher.eval()

    student = StudentQNetwork(
        input_dim=_INPUT_DIM,
        output_dim=_OUTPUT_DIM,
        parent_hidden_size=hidden_size,
    )
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher params : {t_params:,}")
    print(f"  Student params : {s_params:,}  ({100*s_params/t_params:.1f}% of teacher)")

    trainer = DistillationTrainer(teacher, student, cfg)
    ckpt_path = os.path.join(output_dir, f"student_{pair}.pt")
    os.makedirs(output_dir, exist_ok=True)
    metrics = trainer.train(states, checkpoint_path=ckpt_path)

    print(f"  Final train loss : {metrics.train_losses[-1]:.6f}" if metrics.train_losses else "")
    if metrics.val_losses:
        print(f"  Best val loss    : {metrics.best_val_loss:.6f}  (epoch {metrics.best_epoch})")
        print(f"  Action agreement : {metrics.action_agreements[-1]*100:.1f}%  (last epoch)")
    print(f"  ✓ student_{pair} → {ckpt_path}")

    metrics_dict: Dict[str, Any] = {
        "pair": pair,
        "checkpoint": ckpt_path,
        "train_losses": metrics.train_losses,
        "val_losses": metrics.val_losses,
        "action_agreements": [float(a) for a in metrics.action_agreements],
        "best_val_loss": float(metrics.best_val_loss) if metrics.val_losses else None,
        "best_epoch": metrics.best_epoch,
    }
    return ckpt_path, metrics_dict


# ---------------------------------------------------------------------------
# Stage 3: PTQ quantize student A/B
# ---------------------------------------------------------------------------


def _quantize_student(
    pair: str,
    student_ckpt: str,
    hidden_size: int,
    states: np.ndarray,
    output_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    """Quantize one student to int8 (dynamic PTQ). Returns (ckpt_path, report_dict)."""
    print(f"\n[Stage 3] Quantizing student_{pair} → student_{pair}_int8")

    float_model = load_float_student_checkpoint(
        student_ckpt,
        _INPUT_DIM,
        _OUTPUT_DIM,
        hidden_size,
        not_found_template="Student checkpoint not found: {path}",
        bad_state_template="Checkpoint at '{path}' must be a state dict (got {type_name}).",
    )
    float_params = sum(p.numel() for p in float_model.parameters())
    print(f"  Float model params: {float_params:,}")

    config = QuantizationConfig(mode="dynamic")
    quantizer = PostTrainingQuantizer(config)
    print("  Applying dynamic PTQ (weight-only int8) …")
    q_model, result = quantizer.quantize(float_model, calibration_states=None)

    out_path = os.path.join(output_dir, f"student_{pair}_int8.pt")
    arch_kwargs = {
        "input_dim": _INPUT_DIM,
        "output_dim": _OUTPUT_DIM,
        "parent_hidden_size": hidden_size,
    }
    quantizer.save_checkpoint(q_model, out_path, result, arch_kwargs=arch_kwargs)
    print(f"  Quantized checkpoint: {out_path}")
    print(
        f"  Size reduction: {result.float_param_bytes:,} B → "
        f"{result.quantized_param_bytes:,} B  "
        f"({100*result.quantized_param_bytes/max(1,result.float_param_bytes):.0f}%)"
    )

    cmp = compare_outputs(float_model, q_model, states)
    print(f"  Action agreement vs float : {cmp['action_agreement']*100:.2f}%")
    print(f"  Mean Q-error              : {cmp['mean_q_error']:.6f}")

    report_path = os.path.join(output_dir, f"quantization_report_{pair}.json")
    report: Dict[str, Any] = {
        "pair": pair,
        "float_checkpoint": student_ckpt,
        "quantized_checkpoint": out_path,
        "quantization": result.to_dict(),
        "comparison": cmp,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    print(f"  ✓ student_{pair}_int8 → {out_path}")

    # Verify round-trip
    q_rt, _ = load_quantized_checkpoint(out_path)
    cmp_rt = compare_outputs(float_model, q_rt, states[:64])
    print(f"  Round-trip agreement: {cmp_rt['action_agreement']*100:.2f}%  ✓")

    return out_path, report


# ---------------------------------------------------------------------------
# Stage 4: crossover quantized students → float child
# ---------------------------------------------------------------------------


def _crossover(
    student_a_int8_ckpt: str,
    student_b_int8_ckpt: str,
    hidden_size: int,
    mode: str,
    alpha: float,
    seed: Optional[int],
    output_dir: str,
) -> Tuple[str, StudentQNetwork]:
    """Crossover two quantized students and save the float32 child snapshot."""
    print(f"\n[Stage 4] Crossover  (mode={mode!r}, alpha={alpha})")

    # Use initialize_child_from_crossover which handles PTQ sidecar JSON
    # and dequantizes packed weights automatically.  The PTQ checkpoints are
    # trusted artefacts written by this pipeline, so unsafe unpickling is safe.
    arch = ChildArchitectureSpec(
        input_dim=_INPUT_DIM,
        output_dim=_OUTPUT_DIM,
        hidden_size=max(16, hidden_size // 2),  # StudentQNetwork hidden
        parent_hidden_size=hidden_size,
    )
    child = initialize_child_from_crossover(
        student_a_int8_ckpt,
        student_b_int8_ckpt,
        strategy=mode,
        rng=seed,
        allow_unsafe_unpickle=True,
        architecture=arch,
        network_class=StudentQNetwork,
        alpha=alpha,
    )
    child = child  # already a StudentQNetwork in eval() mode

    ckpt_path = os.path.join(output_dir, "child_crossover.pt")
    torch.save(child.state_dict(), ckpt_path)
    print(f"  ✓ child (pre-finetune) → {ckpt_path}")
    return ckpt_path, child


# ---------------------------------------------------------------------------
# Stage 5: dual-teacher fine-tuning
# ---------------------------------------------------------------------------


def _dual_teacher_finetune(
    child: nn.Module,
    teacher_a: nn.Module,
    teacher_b: nn.Module,
    states: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    temperature: float,
    alpha_a: float,
    val_fraction: float,
    seed: Optional[int],
    output_dir: str,
    device: torch.device,
) -> Tuple[str, Dict[str, Any]]:
    """Fine-tune the child against both teachers with a dual-teacher KL loss.

    The training objective at each gradient step is::

        loss = alpha_a * KL(softmax(A/T) ‖ log_softmax(child/T))
             + (1 - alpha_a) * KL(softmax(B/T) ‖ log_softmax(child/T))

    where ``T`` is ``temperature`` and the KL is computed using
    ``F.kl_div(..., reduction="batchmean")`` following Hinton et al. 2015.

    Both teachers are frozen throughout; only child parameters receive
    gradient updates.

    Returns
    -------
    tuple of (checkpoint_path, metrics_dict)
    """
    print(
        f"\n[Stage 5] Dual-teacher fine-tune  "
        f"(alpha_a={alpha_a:.2f}, alpha_b={1-alpha_a:.2f}, "
        f"T={temperature}, epochs={epochs}, lr={lr})"
    )
    alpha_b = 1.0 - alpha_a

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    child = child.to(device)
    teacher_a = teacher_a.to(device).eval()
    teacher_b = teacher_b.to(device).eval()
    for p in teacher_a.parameters():
        p.requires_grad = False
    for p in teacher_b.parameters():
        p.requires_grad = False

    states_arr = np.asarray(states, dtype="float32")
    n = len(states_arr)
    n_val = int(n * val_fraction) if val_fraction > 0 else 0
    n_train = n - n_val
    train_states = torch.tensor(states_arr[:n_train], dtype=torch.float32, device=device)
    val_states = (
        torch.tensor(states_arr[n_train:], dtype=torch.float32, device=device)
        if n_val > 0
        else None
    )

    optimizer = torch.optim.Adam(child.parameters(), lr=lr)

    def _kl_dual(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute dual-teacher KL loss and per-teacher components."""
        with torch.no_grad():
            logits_a = teacher_a(batch)
            logits_b = teacher_b(batch)
        logits_child = child(batch)

        soft_a = F.softmax(logits_a / temperature, dim=-1)
        soft_b = F.softmax(logits_b / temperature, dim=-1)
        log_child = F.log_softmax(logits_child / temperature, dim=-1)

        kl_a = F.kl_div(log_child, soft_a, reduction="batchmean") * (temperature ** 2)
        kl_b = F.kl_div(log_child, soft_b, reduction="batchmean") * (temperature ** 2)
        loss = alpha_a * kl_a + alpha_b * kl_b
        return loss, kl_a, kl_b

    def _eval_batch(batch: torch.Tensor) -> Dict[str, float]:
        child.eval()
        with torch.no_grad():
            logits_a = teacher_a(batch)
            logits_b = teacher_b(batch)
            logits_child = child(batch)
        actions_a = logits_a.argmax(-1)
        actions_b = logits_b.argmax(-1)
        actions_child = logits_child.argmax(-1)
        agree_a = float((actions_child == actions_a).float().mean())
        agree_b = float((actions_child == actions_b).float().mean())
        soft_a = F.softmax(logits_a / temperature, dim=-1)
        soft_b = F.softmax(logits_b / temperature, dim=-1)
        log_child = F.log_softmax(logits_child / temperature, dim=-1)
        kl_a_val = float(F.kl_div(log_child, soft_a, reduction="batchmean") * (temperature ** 2))
        kl_b_val = float(F.kl_div(log_child, soft_b, reduction="batchmean") * (temperature ** 2))
        child.train()
        return {
            "agree_a": agree_a,
            "agree_b": agree_b,
            "kl_a": kl_a_val,
            "kl_b": kl_b_val,
            "dual_loss": alpha_a * kl_a_val + alpha_b * kl_b_val,
        }

    # --- before-training baseline ---
    if val_states is not None:
        baseline = _eval_batch(val_states)
        print(
            f"  Before: agree_A={baseline['agree_a']*100:.1f}%  "
            f"agree_B={baseline['agree_b']*100:.1f}%  "
            f"dual_loss={baseline['dual_loss']:.4f}"
        )
    else:
        baseline = {}

    metrics_per_epoch: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    child.train()
    for epoch in range(epochs):
        # Shuffle training states
        perm = torch.randperm(n_train, device=device)
        train_shuffled = train_states[perm]

        epoch_losses: List[float] = []
        epoch_kl_a: List[float] = []
        epoch_kl_b: List[float] = []
        for start in range(0, n_train, batch_size):
            batch = train_shuffled[start : start + batch_size]
            if len(batch) == 0:
                continue
            optimizer.zero_grad()
            loss, kl_a, kl_b = _kl_dual(batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_kl_a.append(kl_a.item())
            epoch_kl_b.append(kl_b.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_metrics: Dict[str, float] = {"train_dual_loss": train_loss}

        if val_states is not None:
            val_m = _eval_batch(val_states)
            epoch_metrics.update({f"val_{k}": v for k, v in val_m.items()})
            val_loss = val_m["dual_loss"]
            print(
                f"  Epoch {epoch+1:3d}/{epochs}: train={train_loss:.4f}  "
                f"val_dual={val_loss:.4f}  "
                f"agree_A={val_m['agree_a']*100:.1f}%  "
                f"agree_B={val_m['agree_b']*100:.1f}%"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in child.state_dict().items()}
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state_dict = {k: v.cpu().clone() for k, v in child.state_dict().items()}

        metrics_per_epoch.append(epoch_metrics)

    # Restore best weights
    if best_state_dict is not None:
        child.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    ckpt_path = os.path.join(output_dir, "child_finetuned.pt")
    torch.save(
        {k: v.cpu() for k, v in child.state_dict().items()},
        ckpt_path,
    )

    # After-training evaluation
    if val_states is not None:
        after = _eval_batch(val_states)
        print(
            f"  After : agree_A={after['agree_a']*100:.1f}%  "
            f"agree_B={after['agree_b']*100:.1f}%  "
            f"dual_loss={after['dual_loss']:.4f}"
        )
    else:
        after = {}

    print(f"  ✓ child_finetuned → {ckpt_path}")

    metrics: Dict[str, Any] = {
        "checkpoint": ckpt_path,
        "epochs": epochs,
        "temperature": temperature,
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "before": baseline,
        "after": after,
        "per_epoch": metrics_per_epoch,
        "best_val_dual_loss": float(best_val_loss),
    }
    return ckpt_path, metrics


# ---------------------------------------------------------------------------
# Stage 6: validate — per-stage distillation & final recombination reports
# ---------------------------------------------------------------------------


def _validate_distillation(
    pair: str,
    parent_ckpt: str,
    student_ckpt: str,
    hidden_size: int,
    states: np.ndarray,
    output_dir: str,
    report_only: bool,
) -> Dict[str, Any]:
    """Compare student to parent. Returns metrics dict and writes JSON report."""
    print(f"\n[Stage 6a] Distillation validation for pair {pair}")
    teacher = BaseQNetwork(_INPUT_DIM, _OUTPUT_DIM, hidden_size=hidden_size)
    t_state = torch.load(parent_ckpt, map_location="cpu", weights_only=True)
    teacher.load_state_dict(t_state)
    teacher.eval()

    student = load_float_student_checkpoint(
        student_ckpt,
        _INPUT_DIM,
        _OUTPUT_DIM,
        hidden_size,
        not_found_template="Student not found: {path}",
        bad_state_template="Bad state: {path} got {type_name}",
    )

    states_t = torch.tensor(states, dtype=torch.float32)
    with torch.no_grad():
        q_teacher = teacher(states_t).numpy()
        q_student = student(states_t).numpy()

    actions_teacher = q_teacher.argmax(-1)
    actions_student = q_student.argmax(-1)
    action_agreement = float((actions_teacher == actions_student).mean())
    mse = float(np.mean((q_teacher - q_student) ** 2))
    mae = float(np.mean(np.abs(q_teacher - q_student)))

    # KL divergence (teacher ‖ student)
    temp = 3.0
    soft_t = torch.softmax(torch.tensor(q_teacher) / temp, dim=-1)
    soft_s = torch.log_softmax(torch.tensor(q_student) / temp, dim=-1)
    kl = float(F.kl_div(soft_s, soft_t, reduction="batchmean") * (temp ** 2))

    # Cosine similarity
    cos_vals = F.cosine_similarity(
        torch.tensor(q_teacher), torch.tensor(q_student), dim=-1
    ).numpy()
    cosine = float(cos_vals.mean())

    report = {
        "pair": pair,
        "parent_checkpoint": parent_ckpt,
        "student_checkpoint": student_ckpt,
        "n_states": len(states),
        "action_agreement": action_agreement,
        "kl_divergence": kl,
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cosine,
        "passed": report_only or (action_agreement >= 0.80 and kl <= 1.0),
    }
    print(
        f"  action_agreement: {action_agreement*100:.2f}%  "
        f"kl: {kl:.4f}  mse: {mse:.6f}  cosine: {cosine:.4f}"
    )

    report_path = os.path.join(output_dir, f"distillation_report_{pair}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    print(f"  ✓ distillation_report_{pair}.json → {report_path}")
    return report


def _validate_recombination(
    parent_a_ckpt: str,
    parent_b_ckpt: str,
    child_ckpt: str,
    hidden_size: int,
    states: np.ndarray,
    output_dir: str,
    report_only: bool,
    min_action_agreement: float,
    max_kl: float,
    max_mse: float,
    min_cosine: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate child vs both teachers + A-vs-B baseline."""
    print("\n[Stage 6b] Final recombination validation")

    def _load_net(path: str) -> BaseQNetwork:
        # Child is saved as StudentQNetwork state dict; both parent and student
        # share compatible forward() since StudentQNetwork extends BaseQNetwork.
        # Load into BaseQNetwork if hidden matches, else StudentQNetwork.
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Detect hidden size from first Linear weight
        key = "network.0.weight"
        if key in state:
            h = int(state[key].shape[0])
        else:
            h = hidden_size
        net = BaseQNetwork(_INPUT_DIM, _OUTPUT_DIM, hidden_size=h)
        try:
            net.load_state_dict(state)
        except RuntimeError:
            # Fallback: StudentQNetwork (student has hidden_size // 2)
            net = StudentQNetwork(_INPUT_DIM, _OUTPUT_DIM, parent_hidden_size=hidden_size)
            net.load_state_dict(state)
        net.eval()
        return net

    parent_a = _load_net(parent_a_ckpt)
    parent_b = _load_net(parent_b_ckpt)
    child = _load_net(child_ckpt)

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
        include_parent_baseline=True,
        k_values=[1, 2],
        states_source="dual_teacher_pipeline_synthetic",
        model_paths={
            "parent_a": parent_a_ckpt,
            "parent_b": parent_b_ckpt,
            "child": child_ckpt,
        },
    )
    report_dict = report.to_dict()
    summary = report_dict.get("summary", {})
    sep = "-" * 50
    print(f"  {sep}")
    print(f"  Child ↔ Parent A  agreement : {summary.get('child_agrees_with_parent_a', 0):.4f}")
    print(f"  Child ↔ Parent B  agreement : {summary.get('child_agrees_with_parent_b', 0):.4f}")
    comparisons = report_dict.get("comparisons", {})
    a_vs_b = comparisons.get("parent_a_vs_parent_b", {})
    if a_vs_b:
        print(f"  Parent A ↔ Parent B        : {a_vs_b.get('action_agreement', 0):.4f}")
    print(f"  Overall passed              : {report_dict.get('passed', False)}")

    report_path = os.path.join(output_dir, "recombination_validation.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, allow_nan=False)
    print(f"  ✓ recombination_validation.json → {report_path}")
    return report_dict


# ---------------------------------------------------------------------------
# Master pipeline report
# ---------------------------------------------------------------------------


def _write_pipeline_report(
    output_dir: str,
    args: argparse.Namespace,
    artifacts: Dict[str, str],
    stage_metrics: Dict[str, Any],
    passed: bool,
) -> str:
    """Write pipeline_report.json aggregating all stage results."""
    report = {
        "pipeline": "dual_teacher_cartpole",
        "issue": "Dooders/AgentFarm#8",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "hidden_size": args.hidden_size,
            "distill_epochs": args.distill_epochs,
            "distill_temperature": args.distill_temperature,
            "distill_alpha": args.distill_alpha,
            "distill_lr": args.distill_lr,
            "crossover_mode": args.crossover_mode,
            "crossover_alpha": args.crossover_alpha,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "finetune_temperature": args.finetune_temperature,
            "finetune_teacher_weight_a": args.finetune_teacher_weight_a,
        },
        "artifacts": artifacts,
        "stages": stage_metrics,
        "passed": passed,
    }
    out_path = os.path.join(output_dir, "pipeline_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Dual-teacher compression-first CartPole pipeline (canonical Issue #8 run). "
            "Runs: train parents → distill → quantize → crossover → dual-teacher fine-tune → validate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Existing parent checkpoints (skip training if provided)
    p.add_argument("--parent-a-ckpt", default="", help="Existing parent A checkpoint.")
    p.add_argument("--parent-b-ckpt", default="", help="Existing parent B checkpoint.")
    p.add_argument("--force-train", action="store_true", help="Re-train parents even if checkpoints exist.")

    # Parent training
    p.add_argument("--train-episodes", type=int, default=200, help="Episodes per parent.")
    p.add_argument("--train-lr", type=float, default=1e-3)
    p.add_argument("--train-gamma", type=float, default=0.99)
    p.add_argument("--train-epsilon-start", type=float, default=1.0)
    p.add_argument("--train-epsilon-min", type=float, default=0.01)
    p.add_argument("--train-epsilon-decay", type=float, default=0.995)
    p.add_argument("--train-tau", type=float, default=0.005)
    p.add_argument("--train-memory", type=int, default=10000)
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--seed-a", type=int, default=42, help="RNG seed for parent A.")
    p.add_argument("--seed-b", type=int, default=99, help="RNG seed for parent B.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-replay-states", type=int, default=200_000, help="-1 = unlimited.")

    # Architecture
    p.add_argument("--hidden-size", type=int, default=64, help="Parent hidden layer width.")

    # Distillation
    p.add_argument("--distill-epochs", type=int, default=15)
    p.add_argument("--distill-temperature", type=float, default=3.0)
    p.add_argument("--distill-alpha", type=float, default=1.0, help="Soft-label blend (1.0=pure KL).")
    p.add_argument("--distill-lr", type=float, default=1e-3)
    p.add_argument("--distill-batch", type=int, default=64)
    p.add_argument("--distill-seed", type=int, default=None)
    p.add_argument("--n-states", type=int, default=2000, help="Synthetic states for distillation & eval.")
    p.add_argument("--states-seed", type=int, default=42)
    p.add_argument("--states-file", default="", help="Optional .npy replay states (overrides --n-states).")

    # Crossover
    p.add_argument("--crossover-mode", choices=list(CROSSOVER_MODES), default="weighted")
    p.add_argument("--crossover-alpha", type=float, default=0.5)
    p.add_argument("--crossover-seed", type=int, default=None)

    # Dual-teacher fine-tuning
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--finetune-lr", type=float, default=5e-4)
    p.add_argument("--finetune-batch", type=int, default=64)
    p.add_argument("--finetune-temperature", type=float, default=3.0)
    p.add_argument(
        "--finetune-teacher-weight-a",
        type=float,
        default=0.5,
        help="Weight for teacher A KL loss (teacher B weight = 1 - this).",
    )
    p.add_argument("--finetune-val-fraction", type=float, default=0.1)
    p.add_argument("--finetune-seed", type=int, default=0)

    # Validation thresholds
    p.add_argument("--min-action-agreement", type=float, default=0.50)
    p.add_argument("--max-kl-divergence", type=float, default=2.0)
    p.add_argument("--max-mse", type=float, default=10.0)
    p.add_argument("--min-cosine-similarity", type=float, default=0.50)
    p.add_argument("--report-only", action="store_true", help="Write reports without enforcing pass/fail.")

    # Device & output
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="checkpoints/dual_teacher_cartpole")
    return p.parse_args()


def main() -> None:  # noqa: C901  (intentionally linear pipeline)
    args = _parse_args()
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    device = parse_torch_device(args.device)
    max_replay: Optional[int] = None if args.max_replay_states < 0 else args.max_replay_states

    sep = "=" * 60
    print(f"\n{sep}")
    print("Dual-teacher CartPole pipeline  (canonical Issue #8 run)")
    print(f"Output dir : {out}")
    print(f"Device     : {device}")
    print(f"{sep}")

    artifacts: Dict[str, str] = {}
    stage_metrics: Dict[str, Any] = {}

    # -------------------------------------------------------------------
    # Stage 1: train parents
    # -------------------------------------------------------------------
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
        max_replay_states=max_replay,
    )

    if not parent_a_ckpt or args.force_train:
        if not os.path.isfile(default_a) or args.force_train:
            parent_a_ckpt = _train_parent("A", episodes=args.train_episodes, seed=args.seed_a, **train_common)
        else:
            parent_a_ckpt = default_a
            print(f"\n[Stage 1] Skipping parent A training — using {parent_a_ckpt}")
    if not os.path.isfile(parent_a_ckpt):
        raise FileNotFoundError(f"Parent A checkpoint not found: {parent_a_ckpt!r}")
    artifacts["parent_a"] = parent_a_ckpt

    if not parent_b_ckpt or args.force_train:
        if not os.path.isfile(default_b) or args.force_train:
            parent_b_ckpt = _train_parent("B", episodes=args.train_episodes, seed=args.seed_b, **train_common)
        else:
            parent_b_ckpt = default_b
            print(f"\n[Stage 1] Skipping parent B training — using {parent_b_ckpt}")
    if not os.path.isfile(parent_b_ckpt):
        raise FileNotFoundError(f"Parent B checkpoint not found: {parent_b_ckpt!r}")
    artifacts["parent_b"] = parent_b_ckpt

    # -------------------------------------------------------------------
    # Prepare shared state buffer
    # -------------------------------------------------------------------
    states_file = args.states_file
    if not states_file:
        # Prefer replay exports written by the training stage
        for name in ("replay_states_B.npy", "replay_states_A.npy", "replay_states.npy"):
            candidate = os.path.join(out, name)
            if os.path.isfile(candidate):
                states_file = candidate
                break

    states_source_label = repr(states_file) if states_file else "'synthetic Gaussian'"
    print(f"\n[States] source: {states_source_label}")
    states = load_distillation_states(
        states_file,
        n_states=args.n_states,
        input_dim=_INPUT_DIM,
        seed=args.states_seed,
    )
    print(f"  states shape: {states.shape}")

    # -------------------------------------------------------------------
    # Stage 2: distil parent A → student A, parent B → student B
    # -------------------------------------------------------------------
    dist_cfg = DistillationConfig(
        temperature=args.distill_temperature,
        alpha=args.distill_alpha,
        learning_rate=args.distill_lr,
        epochs=args.distill_epochs,
        batch_size=args.distill_batch,
        seed=args.distill_seed,
    )
    student_a_ckpt, dist_a_metrics = _distill_student(
        "A", parent_a_ckpt, states, args.hidden_size, dist_cfg, out
    )
    artifacts["student_a"] = student_a_ckpt
    stage_metrics["distillation_A"] = dist_a_metrics

    student_b_ckpt, dist_b_metrics = _distill_student(
        "B", parent_b_ckpt, states, args.hidden_size, dist_cfg, out
    )
    artifacts["student_b"] = student_b_ckpt
    stage_metrics["distillation_B"] = dist_b_metrics

    # -------------------------------------------------------------------
    # Stage 3: PTQ quantize students
    # -------------------------------------------------------------------
    student_a_int8_ckpt, quant_a_report = _quantize_student(
        "A", student_a_ckpt, args.hidden_size, states, out
    )
    artifacts["student_a_int8"] = student_a_int8_ckpt
    stage_metrics["quantization_A"] = quant_a_report

    student_b_int8_ckpt, quant_b_report = _quantize_student(
        "B", student_b_ckpt, args.hidden_size, states, out
    )
    artifacts["student_b_int8"] = student_b_int8_ckpt
    stage_metrics["quantization_B"] = quant_b_report

    # -------------------------------------------------------------------
    # Stage 4: crossover quantized students → float child
    # -------------------------------------------------------------------
    child_crossover_ckpt, child_net = _crossover(
        student_a_int8_ckpt,
        student_b_int8_ckpt,
        args.hidden_size,
        args.crossover_mode,
        args.crossover_alpha,
        args.crossover_seed,
        out,
    )
    artifacts["child_crossover"] = child_crossover_ckpt

    # -------------------------------------------------------------------
    # Stage 5: dual-teacher fine-tune
    # -------------------------------------------------------------------
    # Load both frozen teachers (as BaseQNetwork parents, not students)
    teacher_a = BaseQNetwork(_INPUT_DIM, _OUTPUT_DIM, hidden_size=args.hidden_size)
    t_state = torch.load(parent_a_ckpt, map_location="cpu", weights_only=True)
    teacher_a.load_state_dict(t_state)
    teacher_a.eval()

    teacher_b = BaseQNetwork(_INPUT_DIM, _OUTPUT_DIM, hidden_size=args.hidden_size)
    t_state = torch.load(parent_b_ckpt, map_location="cpu", weights_only=True)
    teacher_b.load_state_dict(t_state)
    teacher_b.eval()

    child_finetuned_ckpt, ft_metrics = _dual_teacher_finetune(
        child=child_net,
        teacher_a=teacher_a,
        teacher_b=teacher_b,
        states=states,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        batch_size=args.finetune_batch,
        temperature=args.finetune_temperature,
        alpha_a=args.finetune_teacher_weight_a,
        val_fraction=args.finetune_val_fraction,
        seed=args.finetune_seed,
        output_dir=out,
        device=device,
    )
    artifacts["child_finetuned"] = child_finetuned_ckpt
    stage_metrics["finetune_dual_teacher"] = ft_metrics

    # -------------------------------------------------------------------
    # Stage 6: validate
    # -------------------------------------------------------------------
    dist_val_a = _validate_distillation(
        "A", parent_a_ckpt, student_a_ckpt, args.hidden_size, states, out, args.report_only
    )
    stage_metrics["distillation_validation_A"] = dist_val_a

    dist_val_b = _validate_distillation(
        "B", parent_b_ckpt, student_b_ckpt, args.hidden_size, states, out, args.report_only
    )
    stage_metrics["distillation_validation_B"] = dist_val_b

    recomb_report = _validate_recombination(
        parent_a_ckpt=parent_a_ckpt,
        parent_b_ckpt=parent_b_ckpt,
        child_ckpt=child_finetuned_ckpt,
        hidden_size=args.hidden_size,
        states=states,
        output_dir=out,
        report_only=args.report_only,
        min_action_agreement=args.min_action_agreement,
        max_kl=args.max_kl_divergence,
        max_mse=args.max_mse,
        min_cosine=args.min_cosine_similarity,
        device=device,
    )
    stage_metrics["recombination_validation"] = recomb_report
    artifacts["recombination_report"] = os.path.join(out, "recombination_validation.json")

    # -------------------------------------------------------------------
    # Write master pipeline report
    # -------------------------------------------------------------------
    passed = bool(recomb_report.get("passed", False))
    if args.report_only:
        passed = True

    report_path = _write_pipeline_report(out, args, artifacts, stage_metrics, passed)
    artifacts["pipeline_report"] = report_path

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{sep}")
    print("Dual-teacher CartPole pipeline complete.")
    print(f"All outputs in    : {out}")
    print(f"Pipeline report   : {report_path}")
    print(f"Validation passed : {passed}")

    summary = recomb_report.get("summary", {})
    print("\nFinal metrics (child vs teachers):")
    print(f"  Child ↔ Parent A : {summary.get('child_agrees_with_parent_a', 0):.4f} agreement")
    print(f"  Child ↔ Parent B : {summary.get('child_agrees_with_parent_b', 0):.4f} agreement")
    comparisons = recomb_report.get("comparisons", {})
    a_vs_b = comparisons.get("parent_a_vs_parent_b", {})
    if a_vs_b:
        print(f"  Parent A ↔ B     : {a_vs_b.get('action_agreement', 0):.4f} agreement")
    print(f"{sep}\n")

    if not passed and not args.report_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
