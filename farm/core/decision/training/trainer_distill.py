"""Knowledge distillation training pipeline for DQN-style policies.

This module implements a teacher-student distillation trainer that teaches a
smaller *student* model to replicate the Q-value outputs of a larger (frozen)
*teacher* model.  It is designed to work with any ``nn.Module`` that accepts a
batch of state tensors and returns Q-value logits (e.g. ``BaseQNetwork`` or
``StudentQNetwork`` from :mod:`farm.core.decision.base_dqn`).

Key features
------------
- **Teacher frozen**: teacher weights are locked and set to eval mode.
- **Temperature scaling**: soft targets are produced by dividing teacher
  (and student) logits by a ``temperature`` before softmax / KL-div, with an
  optional per-epoch ``temp_decay`` multiplier.
- **Loss blending**: the distillation (soft) loss can be blended with a hard
  cross-entropy loss on the teacher's argmax action via the ``alpha``
  hyperparameter (``loss = alpha * soft + (1 - alpha) * hard``).
- **Gradient clipping**: configurable ``max_grad_norm`` keeps training stable.
- **Validation split**: an optional validation fraction of the replay buffer is
  held out and evaluated each epoch without gradient updates.
- **Checkpointing**: the student checkpoint with the best validation loss is
  saved to disk, together with a JSON metadata file.
- **Reproducibility**: an optional ``seed`` resets all RNGs before training.
- **Calibration metrics**: per-epoch mean probability similarity between
  teacher and student soft distributions is tracked alongside action agreement.

Typical usage
-------------
::

    from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
    from farm.core.decision.training.trainer_distill import (
        DistillationConfig,
        DistillationTrainer,
    )

    teacher = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    student = StudentQNetwork(input_dim=8, output_dim=4, parent_hidden_size=64)

    cfg = DistillationConfig(temperature=3.0, alpha=0.7, epochs=20)
    trainer = DistillationTrainer(teacher, student, cfg)

    import numpy as np
    states = np.random.randn(500, 8).astype("float32")
    metrics = trainer.train(states, checkpoint_path="student_best.pt")
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from farm.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DistillationConfig:
    """Hyperparameters for the knowledge distillation training loop.

    Attributes
    ----------
    temperature:
        Softmax temperature applied to both teacher and student logits when
        computing the soft (KL-divergence) distillation loss.  Higher values
        produce softer probability distributions that expose more relative
        ordering information.  Must be > 0.
    temp_decay:
        Multiplicative factor applied to ``temperature`` after each epoch
        (e.g. ``0.95`` decays temperature by 5 % per epoch).  Set to
        ``1.0`` (the default) to disable decay.  Must be in ``(0, 1]``.
    alpha:
        Blending weight for the *soft* distillation loss.  The total loss is::

            loss = alpha * soft_loss + (1 - alpha) * hard_loss

        Set ``alpha = 1.0`` for pure soft-label distillation (default).
        Set ``alpha = 0.0`` for pure hard-label supervision.
        Intermediate values blend both objectives.
    learning_rate:
        Initial learning rate for the Adam optimizer.
    epochs:
        Number of full passes over the state buffer.
    batch_size:
        Mini-batch size for each gradient update.
    max_grad_norm:
        Maximum L2-norm of gradients before clipping.  Set to ``None`` to
        disable clipping.
    val_fraction:
        Fraction of states to hold out as a validation set (0 ≤ val_fraction
        < 1).  When 0, no validation is performed.
    seed:
        Optional integer seed for reproducibility.  When set, resets
        ``random``, ``numpy``, and ``torch`` RNGs before training starts.
    loss_fn:
        Which loss function to use for the soft distillation signal.
        ``"kl"`` uses temperature-scaled KL divergence (Hinton et al. 2015);
        ``"mse"`` applies MSE directly on the raw Q-value logits (no softmax),
        which is natural for regression-style Q-value matching.
    """

    temperature: float = 3.0
    temp_decay: float = 1.0
    alpha: float = 1.0
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    max_grad_norm: Optional[float] = 1.0
    val_fraction: float = 0.1
    seed: Optional[int] = None
    loss_fn: str = "kl"  # "kl" or "mse"

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < self.temp_decay <= 1.0:
            raise ValueError("temp_decay must be in (0, 1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when specified")
        if self.loss_fn not in ("kl", "mse"):
            raise ValueError("loss_fn must be 'kl' or 'mse'")


# ---------------------------------------------------------------------------
# Distillation metrics container
# ---------------------------------------------------------------------------


@dataclass
class DistillationMetrics:
    """Per-epoch training and validation metrics produced by :class:`DistillationTrainer`.

    Attributes
    ----------
    train_losses:
        Mean total distillation loss per training epoch.
    train_soft_losses:
        Mean soft (KL/MSE) distillation loss per training epoch.
    train_hard_losses:
        Mean hard CE loss per training epoch (empty when ``alpha == 1.0``,
        i.e. pure soft-label mode, or when ``loss_fn == "mse"``).
    val_losses:
        Mean total distillation loss per validation epoch (empty when
        ``val_fraction == 0``).
    action_agreements:
        Fraction of states where the student argmax action matches the
        teacher argmax action, evaluated on the *validation* set after each
        epoch (empty when ``val_fraction == 0``).
    mean_prob_similarities:
        Mean probability similarity between teacher and student soft
        distributions on the validation set, defined as
        ``1 - mean(|p_t - p_s|)`` averaged over actions and samples.
        Values near 1 indicate close distributional agreement.
        Empty when ``val_fraction == 0``.
    best_val_loss:
        Lowest validation loss seen during training.
    best_epoch:
        Epoch index (0-based) at which the best validation loss was achieved.
    """

    train_losses: List[float] = field(default_factory=list)
    train_soft_losses: List[float] = field(default_factory=list)
    train_hard_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    action_agreements: List[float] = field(default_factory=list)
    mean_prob_similarities: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_losses": self.train_losses,
            "train_soft_losses": self.train_soft_losses,
            "train_hard_losses": self.train_hard_losses,
            "val_losses": self.val_losses,
            "action_agreements": self.action_agreements,
            "mean_prob_similarities": self.mean_prob_similarities,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }


# ---------------------------------------------------------------------------
# Distillation trainer
# ---------------------------------------------------------------------------


class DistillationTrainer:
    """Teacher-to-student knowledge distillation trainer.

    The trainer freezes the *teacher* network, then optimises the *student*
    network to minimise a combination of:

    * **Soft distillation loss** – KL divergence (or MSE) between
      temperature-scaled teacher and student outputs.
    * **Hard supervision loss** – cross-entropy against the teacher's argmax
      action (only when ``alpha < 1.0``).

    The blended objective follows the Hinton et al. (2015) convention::

        loss = alpha * soft_loss + (1 - alpha) * hard_loss

    where ``alpha = 1.0`` (the default) gives pure soft-label distillation.

    For KL mode the soft targets are computed as:

    * Teacher: ``p_t = softmax(z_t / T)``
    * Student: ``p_s = log_softmax(z_s / T)``
    * Loss:    ``KLDiv(p_t ‖ p_s) * T²``  (``log_target=False`` PyTorch form)

    Parameters
    ----------
    teacher:
        Frozen teacher ``nn.Module``.  The module is put into ``eval`` mode
        and its gradients are disabled during training.
    student:
        Trainable student ``nn.Module``.  Must accept the same input shape as
        the teacher and produce the same output shape.
    config:
        :class:`DistillationConfig` controlling all hyperparameters.
    device:
        Target PyTorch device.  Defaults to ``torch.device("cpu")``.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config: DistillationConfig = config or DistillationConfig()
        self.device: torch.device = device or torch.device("cpu")

        # Move models to device
        self.teacher: nn.Module = teacher.to(self.device)
        self.student: nn.Module = student.to(self.device)

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Optimizer for student only
        self.optimizer = torch.optim.Adam(
            self.student.parameters(), lr=self.config.learning_rate
        )

        # Mutable temperature that can be decayed across epochs
        self._current_temperature: float = self.config.temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        states: np.ndarray,
        checkpoint_path: Optional[str] = None,
    ) -> DistillationMetrics:
        """Run the distillation training loop.

        Parameters
        ----------
        states:
            NumPy array of shape ``(N, input_dim)`` representing the replay
            buffer / state distribution used for distillation.
        checkpoint_path:
            If given, the best student weights are saved to this ``.pt`` file
            and a companion ``<checkpoint_path>.json`` metadata file is written
            next to it.

        Returns
        -------
        DistillationMetrics
            Populated metrics object with per-epoch train/val losses, separate
            soft/hard loss components, action agreement, and probability
            similarity scores.
        """
        if len(states) == 0:
            raise ValueError(
                "states must be non-empty; got an array with 0 samples."
            )

        if self.config.seed is not None:
            self._set_seed(self.config.seed)

        # Reset temperature at the start of each training run
        self._current_temperature = self.config.temperature

        train_states, val_states = self._split_states(states)
        train_tensor = torch.tensor(train_states, dtype=torch.float32, device=self.device)
        val_tensor = (
            torch.tensor(val_states, dtype=torch.float32, device=self.device)
            if len(val_states) > 0
            else None
        )

        metrics = DistillationMetrics()
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        _best_train_loss: float = float("inf")

        for epoch in range(self.config.epochs):
            train_loss, soft_loss, hard_loss = self._run_epoch(train_tensor)
            metrics.train_losses.append(train_loss)
            metrics.train_soft_losses.append(soft_loss)
            if hard_loss is not None:
                metrics.train_hard_losses.append(hard_loss)

            if val_tensor is not None and len(val_tensor) > 0:
                val_loss, agreement, prob_sim = self._evaluate(val_tensor)
                metrics.val_losses.append(val_loss)
                metrics.action_agreements.append(agreement)
                metrics.mean_prob_similarities.append(prob_sim)

                if val_loss < metrics.best_val_loss:
                    metrics.best_val_loss = val_loss
                    metrics.best_epoch = epoch
                    best_state_dict = {
                        k: v.cpu().clone()
                        for k, v in self.student.state_dict().items()
                    }

                logger.info(
                    "distillation_epoch",
                    epoch=epoch,
                    temperature=round(self._current_temperature, 6),
                    train_loss=round(train_loss, 6),
                    train_soft_loss=round(soft_loss, 6),
                    train_hard_loss=round(hard_loss, 6) if hard_loss is not None else None,
                    val_loss=round(val_loss, 6),
                    action_agreement=round(agreement, 4),
                    mean_prob_similarity=round(prob_sim, 4),
                )
            else:
                # No val set – track the best training loss and its weights
                if train_loss < _best_train_loss:
                    _best_train_loss = train_loss
                    metrics.best_val_loss = train_loss
                    metrics.best_epoch = epoch
                    best_state_dict = {
                        k: v.cpu().clone()
                        for k, v in self.student.state_dict().items()
                    }

                logger.info(
                    "distillation_epoch",
                    epoch=epoch,
                    temperature=round(self._current_temperature, 6),
                    train_loss=round(train_loss, 6),
                    train_soft_loss=round(soft_loss, 6),
                    train_hard_loss=round(hard_loss, 6) if hard_loss is not None else None,
                )

            # Apply temperature decay after each epoch
            self._current_temperature *= self.config.temp_decay

        # best_state_dict is always set inside the loop (at least one epoch runs
        # because we validated len(states) > 0 above), so this guard is a safety net.
        if best_state_dict is None:
            best_state_dict = {
                k: v.cpu().clone() for k, v in self.student.state_dict().items()
            }

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, best_state_dict, metrics)

        # Load best weights back into student
        self.student.load_state_dict(
            {k: v.to(self.device) for k, v in best_state_dict.items()}
        )

        return metrics

    def evaluate_agreement(self, states: np.ndarray) -> float:
        """Compute top-1 action agreement between teacher and student.

        Parameters
        ----------
        states:
            NumPy array of shape ``(N, input_dim)``.

        Returns
        -------
        float
            Fraction of states where student argmax == teacher argmax.
        """
        tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        _, agreement, _ = self._evaluate(tensor)
        return agreement

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _split_states(
        self, states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split states into train / validation arrays."""
        n = len(states)
        if n == 0:
            return states, np.empty((0,) + states.shape[1:], dtype=states.dtype)

        val_fraction = self.config.val_fraction
        if val_fraction <= 0.0:
            return states, np.empty((0,) + states.shape[1:], dtype=states.dtype)

        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val
        if n_train <= 0:
            # Edge case: too few samples – use all for train, none for val
            return states, np.empty((0,) + states.shape[1:], dtype=states.dtype)

        indices = np.random.permutation(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        return states[train_idx], states[val_idx]

    def _iter_batches(
        self, tensor: torch.Tensor
    ) -> Generator[torch.Tensor, None, None]:
        """Yield shuffled mini-batches from a state tensor."""
        n = tensor.size(0)
        perm = torch.randperm(n, device=self.device)
        for start in range(0, n, self.config.batch_size):
            idx = perm[start : start + self.config.batch_size]
            yield tensor[idx]

    def _distillation_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the soft distillation loss between teacher and student outputs.

        When ``loss_fn == "kl"`` the loss is temperature-scaled KL divergence
        (multiplied by T² as per Hinton et al. 2015).  The teacher provides
        soft probability targets via ``softmax(z_t / T)`` and the student
        provides log-probabilities via ``log_softmax(z_s / T)``, matching the
        ``KLDivLoss(log_target=False)`` PyTorch convention.

        When ``loss_fn == "mse"`` the loss is MSE on the raw Q-value logits
        (no temperature scaling), which is natural for regression-style
        Q-value matching.
        """
        T = self._current_temperature
        if self.config.loss_fn == "mse":
            return F.mse_loss(student_logits, teacher_logits)

        # KL divergence on temperature-softened distributions (Hinton et al. 2015)
        # p_t = softmax(z_t / T)  — teacher soft targets (NOT log-space)
        # p_s = log_softmax(z_s / T) — student log-probabilities
        # KLDiv(p_s || p_t) with log_target=False: input is log-probs, target is probs
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        student_log_soft = F.log_softmax(student_logits / T, dim=-1)
        kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean", log_target=False)
        return kl * (T**2)

    def _hard_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss of student logits against teacher argmax actions."""
        hard_labels = teacher_logits.argmax(dim=-1)
        return F.cross_entropy(student_logits, hard_labels)

    def _compute_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Blend soft distillation loss with optional hard supervision.

        Returns
        -------
        total_loss : torch.Tensor
            The combined loss used for the backward pass.
        soft_loss : torch.Tensor
            The soft (KL/MSE) distillation loss component.
        hard_loss : torch.Tensor or None
            The hard CE loss component; ``None`` when ``alpha == 1.0`` (pure
            soft mode) or when ``loss_fn == "mse"`` (no hard term).
        """
        soft_loss = self._distillation_loss(teacher_logits, student_logits)
        alpha = self.config.alpha
        # MSE mode has no meaningful hard-label CE term
        if alpha == 1.0 or self.config.loss_fn == "mse":
            return soft_loss, soft_loss, None
        hard_loss = self._hard_loss(teacher_logits, student_logits)
        total = alpha * soft_loss + (1.0 - alpha) * hard_loss
        return total, soft_loss, hard_loss

    def _run_epoch(self, train_tensor: torch.Tensor) -> Tuple[float, float, Optional[float]]:
        """Run one training epoch; return (mean_total, mean_soft, mean_hard) loss."""
        self.student.train()
        total_loss_sum = 0.0
        soft_loss_sum = 0.0
        hard_loss_sum = 0.0
        n_batches = 0
        has_hard = False

        for batch in self._iter_batches(train_tensor):
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(batch)

            # Student forward
            student_logits = self.student(batch)

            total_loss, soft_loss, hard_loss = self._compute_loss(teacher_logits, student_logits)

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            soft_loss_sum += soft_loss.item()
            if hard_loss is not None:
                hard_loss_sum += hard_loss.item()
                has_hard = True
            n_batches += 1

        denom = max(n_batches, 1)
        mean_hard: Optional[float] = (hard_loss_sum / denom) if has_hard else None
        return total_loss_sum / denom, soft_loss_sum / denom, mean_hard

    @torch.no_grad()
    def _evaluate(
        self, val_tensor: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Evaluate distillation loss, action agreement, and probability similarity.

        Returns
        -------
        val_loss : float
            Mean total distillation loss on the validation set.
        action_agreement : float
            Top-1 agreement between teacher and student argmax actions.
        mean_prob_similarity : float
            Mean probability similarity between teacher and student soft
            distributions.  Computed per batch as
            ``1 - mean(|p_t - p_s|)`` where the mean is taken over all
            actions and all samples; then averaged across batches.  Values
            near 1.0 indicate close distributional alignment.
        """
        self.student.eval()
        self.teacher.eval()

        total_loss = 0.0
        n_batches = 0
        n_agree = 0
        n_total = 0
        prob_sim_sum = 0.0

        for batch in self._iter_batches(val_tensor):
            teacher_logits = self.teacher(batch)
            student_logits = self.student(batch)

            total, _, _ = self._compute_loss(teacher_logits, student_logits)
            total_loss += total.item()
            n_batches += 1

            teacher_actions = teacher_logits.argmax(dim=-1)
            student_actions = student_logits.argmax(dim=-1)
            n_agree += (teacher_actions == student_actions).sum().item()
            n_total += batch.size(0)

            # Probability similarity: 1 - mean(|p_t - p_s|) averaged over
            # all actions and all samples in the batch.  Values near 1.0
            # indicate close distributional alignment.
            T = self._current_temperature
            p_t = F.softmax(teacher_logits / T, dim=-1)
            p_s = F.softmax(student_logits / T, dim=-1)
            prob_sim_sum += (1.0 - (p_t - p_s).abs().mean()).item()

        val_loss = total_loss / max(n_batches, 1)
        agreement = n_agree / max(n_total, 1)
        mean_prob_similarity = prob_sim_sum / max(n_batches, 1)
        return val_loss, agreement, mean_prob_similarity

    def _save_checkpoint(
        self,
        path: str,
        state_dict: Dict[str, torch.Tensor],
        metrics: DistillationMetrics,
    ) -> None:
        """Save student weights and metadata to disk.

        Parameters
        ----------
        path:
            Path to ``.pt`` checkpoint file.  A sibling ``.json`` file is
            written with distillation metadata.
        state_dict:
            Student state dict to persist (already on CPU).
        metrics:
            Metrics from the completed training run.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state_dict, path)
        logger.info("distillation_checkpoint_saved", path=path)

        meta_path = path + ".json"
        metadata = {
            "config": {
                "temperature": self.config.temperature,
                "temp_decay": self.config.temp_decay,
                "alpha": self.config.alpha,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "max_grad_norm": self.config.max_grad_norm,
                "val_fraction": self.config.val_fraction,
                "seed": self.config.seed,
                "loss_fn": self.config.loss_fn,
            },
            "metrics": metrics.to_dict(),
        }
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2, allow_nan=False)
        logger.info("distillation_metadata_saved", path=meta_path)


# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------


@dataclass
class ValidationThresholds:
    """Configurable pass/fail thresholds for parent-student pair validation.

    All fields have conservative defaults that can be tightened per task.

    Attributes
    ----------
    min_action_agreement:
        Minimum required top-1 action match rate (0–1).  The student must
        agree with the parent's argmax action on at least this fraction of
        held-out states.
    max_kl_divergence:
        Maximum allowed mean KL divergence KL(parent ‖ student) computed on
        temperature-1 softmax distributions over Q-values.
    max_mse:
        Maximum allowed mean squared error between parent and student raw
        Q-value logits, averaged over states and actions.
    min_cosine_similarity:
        Minimum mean cosine similarity between parent and student Q-value
        vectors across held-out states.
    max_param_ratio:
        Maximum allowed ratio of student parameters to parent parameters.
        Ensures the student is strictly smaller than the parent.
    """

    min_action_agreement: float = 0.85
    max_kl_divergence: float = 0.5
    max_mse: float = 2.0
    min_cosine_similarity: float = 0.8
    max_param_ratio: float = 0.9


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Comprehensive behavioural-fidelity report for a parent-student pair.

    Attributes
    ----------
    action_agreement:
        Top-1 argmax action match rate on the evaluation state batch.
    top_k_agreements:
        Mapping ``{k: agreement}`` for each k in the requested set.
        Top-k agreement is the fraction of states where the parent's
        argmax action appears in the student's top-k actions.
    kl_divergence:
        Mean KL divergence KL(p_parent ‖ p_student) where probabilities
        are temperature-1 softmax distributions over Q-values.
    mse:
        Mean squared error between parent and student Q-value logits.
    mae:
        Mean absolute error between parent and student Q-value logits.
    mean_cosine_similarity:
        Mean cosine similarity between parent and student Q-value vectors.
    parent_param_count:
        Total parameter count of the parent network.
    student_param_count:
        Total parameter count of the student network.
    param_ratio:
        ``student_param_count / parent_param_count`` – lower is better.
    parent_inference_ms:
        Median single-sample inference latency of the parent (milliseconds).
    student_inference_ms:
        Median single-sample inference latency of the student (milliseconds).
    latency_ratio:
        ``student_inference_ms / parent_inference_ms`` – lower is better.
    robustness_slice_agreements:
        Per-slice top-1 action agreement dict, keyed by slice name.
        Empty when no slices were provided.
    thresholds:
        The :class:`ValidationThresholds` used to determine pass/fail.
    """

    action_agreement: float
    top_k_agreements: Dict[int, float]
    kl_divergence: float
    mse: float
    mae: float
    mean_cosine_similarity: float
    parent_param_count: int
    student_param_count: int
    param_ratio: float
    parent_inference_ms: float
    student_inference_ms: float
    latency_ratio: float
    robustness_slice_agreements: Dict[str, float]
    thresholds: ValidationThresholds

    @property
    def passed(self) -> bool:
        """Return ``True`` if all threshold checks pass."""
        t = self.thresholds
        return (
            self.action_agreement >= t.min_action_agreement
            and self.kl_divergence <= t.max_kl_divergence
            and self.mse <= t.max_mse
            and self.mean_cosine_similarity >= t.min_cosine_similarity
            and self.param_ratio <= t.max_param_ratio
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of all report fields."""
        return {
            "action_agreement": self.action_agreement,
            # JSON requires string keys
            "top_k_agreements": {str(k): v for k, v in self.top_k_agreements.items()},
            "kl_divergence": self.kl_divergence,
            "mse": self.mse,
            "mae": self.mae,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "parent_param_count": self.parent_param_count,
            "student_param_count": self.student_param_count,
            "param_ratio": self.param_ratio,
            "parent_inference_ms": self.parent_inference_ms,
            "student_inference_ms": self.student_inference_ms,
            "latency_ratio": self.latency_ratio,
            "robustness_slice_agreements": self.robustness_slice_agreements,
            "passed": self.passed,
            "thresholds": {
                "min_action_agreement": self.thresholds.min_action_agreement,
                "max_kl_divergence": self.thresholds.max_kl_divergence,
                "max_mse": self.thresholds.max_mse,
                "min_cosine_similarity": self.thresholds.min_cosine_similarity,
                "max_param_ratio": self.thresholds.max_param_ratio,
            },
        }


# ---------------------------------------------------------------------------
# Student validator
# ---------------------------------------------------------------------------


class StudentValidator:
    """Behavioural-fidelity validator for parent-student distilled pairs.

    Compares a *student* network against its *parent* (teacher) across
    multiple dimensions:

    * **Output similarity** – KL divergence, MSE, MAE, and cosine similarity
      on held-out state batches.
    * **Action agreement** – top-1 and top-k argmax match rates.
    * **Efficiency** – parameter count ratio and inference latency ratio.
    * **Robustness slices** – per-slice action agreement on named state subsets
      (e.g. low-resource, high-threat, sparse-observation regimes).

    Both models are put into ``eval`` mode with gradients disabled during all
    validation computations.

    Parameters
    ----------
    parent:
        Parent (teacher) ``nn.Module``.  Put into ``eval`` mode during
        validation; weights are never modified.
    student:
        Student ``nn.Module``.  Put into ``eval`` mode during validation.
    thresholds:
        :class:`ValidationThresholds` controlling pass/fail criteria.
        Defaults to :class:`ValidationThresholds` with conservative values.
    device:
        Target PyTorch device.  Defaults to CPU.

    Example
    -------
    ::

        from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
        from farm.core.decision.training.trainer_distill import (
            DistillationConfig, DistillationTrainer,
            StudentValidator, ValidationThresholds,
        )

        parent = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
        student = StudentQNetwork(input_dim=8, output_dim=4, parent_hidden_size=64)

        # … train student via DistillationTrainer …

        thresholds = ValidationThresholds(min_action_agreement=0.90)
        validator = StudentValidator(parent, student, thresholds=thresholds)
        report = validator.validate(eval_states, robustness_slices={"edge": edge_states})
        print(report.passed, report.to_dict())
    """

    def __init__(
        self,
        parent: nn.Module,
        student: nn.Module,
        thresholds: Optional[ValidationThresholds] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.parent: nn.Module = parent
        self.student: nn.Module = student
        self.thresholds: ValidationThresholds = thresholds or ValidationThresholds()
        self.device: torch.device = device or torch.device("cpu")
        self.parent.to(self.device)
        self.student.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        states: np.ndarray,
        robustness_slices: Optional[Dict[str, np.ndarray]] = None,
        k_values: Optional[List[int]] = None,
        n_latency_warmup: int = 5,
        n_latency_repeats: int = 50,
    ) -> ValidationReport:
        """Run all validation checks and return a :class:`ValidationReport`.

        Parameters
        ----------
        states:
            NumPy array of shape ``(N, input_dim)`` – held-out evaluation
            states that were not used during distillation training.
        robustness_slices:
            Optional mapping of slice name → state array.  Each slice is
            evaluated independently and its top-1 action agreement is stored
            in :attr:`ValidationReport.robustness_slice_agreements`.
        k_values:
            Values of *k* for top-k action agreement.  Defaults to
            ``[1, 2, 3]``.
        n_latency_warmup:
            Number of forward passes run before timing (to warm CPU/GPU
            caches and JIT).
        n_latency_repeats:
            Number of timed single-sample forward passes; the median is
            reported to reduce noise.

        Returns
        -------
        ValidationReport
        """
        if k_values is None:
            k_values = [1, 2, 3]

        if len(states) == 0:
            raise ValueError(
                "states must be non-empty; got an array with 0 samples."
            )
        invalid_k = [k for k in k_values if not isinstance(k, int) or k <= 0]
        if invalid_k:
            raise ValueError(
                f"k_values must contain positive integers; got invalid values: {invalid_k}"
            )

        tensor = torch.tensor(states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            parent_logits, student_logits = self._get_logits(tensor)

        # -- Output similarity --
        kl = self._kl_divergence(parent_logits, student_logits)
        mse = float(F.mse_loss(student_logits, parent_logits).item())
        mae = float((student_logits - parent_logits).abs().mean().item())
        cos_sim = self._cosine_similarity(parent_logits, student_logits)

        # -- Action agreement (top-1 always computed; top-k for requested values) --
        parent_actions = parent_logits.argmax(dim=-1)
        student_actions = student_logits.argmax(dim=-1)
        # Top-1 is always computed independently so action_agreement is unambiguous
        action_agreement = float((parent_actions == student_actions).float().mean().item())
        top_k_agreements: Dict[int, float] = {}
        for k in k_values:
            k_clamped = min(k, parent_logits.size(-1))
            topk_student = student_logits.topk(k_clamped, dim=-1).indices
            matches = (topk_student == parent_actions.unsqueeze(-1)).any(dim=-1)
            top_k_agreements[k] = float(matches.float().mean().item())

        # -- Robustness slices --
        slice_agreements: Dict[str, float] = {}
        if robustness_slices:
            for name, slice_states in robustness_slices.items():
                slice_agreements[name] = self._slice_agreement(slice_states)

        # -- Efficiency --
        parent_params = sum(p.numel() for p in self.parent.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        param_ratio = student_params / max(parent_params, 1)

        if n_latency_repeats > 0:
            parent_lat, student_lat = self._measure_latency(
                states, n_latency_warmup, n_latency_repeats
            )
        else:
            parent_lat, student_lat = 0.0, 0.0
        latency_ratio = student_lat / max(parent_lat, 1e-9)

        return ValidationReport(
            action_agreement=action_agreement,
            top_k_agreements=top_k_agreements,
            kl_divergence=kl,
            mse=mse,
            mae=mae,
            mean_cosine_similarity=cos_sim,
            parent_param_count=parent_params,
            student_param_count=student_params,
            param_ratio=param_ratio,
            parent_inference_ms=parent_lat,
            student_inference_ms=student_lat,
            latency_ratio=latency_ratio,
            robustness_slice_agreements=slice_agreements,
            thresholds=self.thresholds,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_logits(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run parent and student in eval mode and return their logits."""
        self.parent.eval()
        self.student.eval()
        parent_logits = self.parent(tensor)
        student_logits = self.student(tensor)
        return parent_logits, student_logits

    def _kl_divergence(
        self,
        parent_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> float:
        """Mean KL(p_parent ‖ p_student) using temperature-1 softmax."""
        p_parent = F.softmax(parent_logits, dim=-1)
        log_p_student = F.log_softmax(student_logits, dim=-1)
        kl = F.kl_div(log_p_student, p_parent, reduction="batchmean", log_target=False)
        return float(kl.item())

    def _cosine_similarity(
        self,
        parent_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> float:
        """Mean cosine similarity between parent and student Q-value vectors."""
        cos = F.cosine_similarity(parent_logits, student_logits, dim=-1)
        return float(cos.mean().item())

    def _slice_agreement(self, slice_states: np.ndarray) -> float:
        """Top-1 action agreement on a single named state slice."""
        tensor = torch.tensor(slice_states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            parent_logits, student_logits = self._get_logits(tensor)
        parent_actions = parent_logits.argmax(dim=-1)
        student_actions = student_logits.argmax(dim=-1)
        return float((parent_actions == student_actions).float().mean().item())

    def _measure_latency(
        self,
        states: np.ndarray,
        n_warmup: int,
        n_repeats: int,
    ) -> Tuple[float, float]:
        """Return median single-sample inference latency (ms) for parent and student.

        A single representative state (the first row of *states*) is used for
        all timing runs so that any per-sample overhead differences between
        models are not confounded by batch size variation.
        """
        single = torch.tensor(states[:1], dtype=torch.float32, device=self.device)

        def _time_model(model: nn.Module) -> float:
            model.eval()
            with torch.no_grad():
                for _ in range(n_warmup):
                    model(single)
            times: List[float] = []
            with torch.no_grad():
                for _ in range(n_repeats):
                    t0 = time.perf_counter()
                    model(single)
                    times.append((time.perf_counter() - t0) * 1_000.0)
            return float(np.median(times))

        parent_ms = _time_model(self.parent)
        student_ms = _time_model(self.student)
        return parent_ms, student_ms
