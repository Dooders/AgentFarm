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
  (and student) logits by a ``temperature`` before softmax / KL-div.
- **Loss blending**: the distillation (soft) loss can be blended with a hard
  cross-entropy loss on the teacher's argmax action via the ``alpha``
  hyperparameter (``loss = alpha * hard + (1 - alpha) * soft``).
- **Gradient clipping**: configurable ``max_grad_norm`` keeps training stable.
- **Validation split**: an optional validation fraction of the replay buffer is
  held out and evaluated each epoch without gradient updates.
- **Checkpointing**: the student checkpoint with the best validation loss is
  saved to disk, together with a JSON metadata file.
- **Reproducibility**: an optional ``seed`` resets all RNGs before training.

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

    cfg = DistillationConfig(temperature=3.0, alpha=0.3, epochs=20)
    trainer = DistillationTrainer(teacher, student, cfg)

    import numpy as np
    states = np.random.randn(500, 8).astype("float32")
    metrics = trainer.train(states, checkpoint_path="student_best.pt")
"""

from __future__ import annotations

import json
import os
import random
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
    alpha:
        Blending weight for the *hard* cross-entropy loss (teacher argmax
        targets).  The total loss is::

            loss = alpha * hard_ce_loss + (1 - alpha) * kl_distill_loss

        Set ``alpha = 0.0`` for pure distillation, ``alpha = 1.0`` for pure
        hard-label supervision.
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
    alpha: float = 0.0
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
        Mean distillation loss per training epoch.
    val_losses:
        Mean distillation loss per validation epoch (empty when
        ``val_fraction == 0``).
    action_agreements:
        Fraction of states where the student argmax action matches the
        teacher argmax action, evaluated on the *validation* set after each
        epoch (empty when ``val_fraction == 0``).
    best_val_loss:
        Lowest validation loss seen during training.
    best_epoch:
        Epoch index (0-based) at which the best validation loss was achieved.
    """

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    action_agreements: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "action_agreements": self.action_agreements,
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
      action (only when ``alpha > 0``).

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
            Populated metrics object with per-epoch train/val losses and
            action agreement scores.
        """
        if self.config.seed is not None:
            self._set_seed(self.config.seed)

        train_states, val_states = self._split_states(states)
        train_tensor = torch.tensor(train_states, dtype=torch.float32, device=self.device)
        val_tensor = (
            torch.tensor(val_states, dtype=torch.float32, device=self.device)
            if len(val_states) > 0
            else None
        )

        metrics = DistillationMetrics()
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(self.config.epochs):
            train_loss = self._run_epoch(train_tensor)
            metrics.train_losses.append(train_loss)

            if val_tensor is not None and len(val_tensor) > 0:
                val_loss, agreement = self._evaluate(val_tensor)
                metrics.val_losses.append(val_loss)
                metrics.action_agreements.append(agreement)

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
                    train_loss=round(train_loss, 6),
                    val_loss=round(val_loss, 6),
                    action_agreement=round(agreement, 4),
                )
            else:
                logger.info(
                    "distillation_epoch",
                    epoch=epoch,
                    train_loss=round(train_loss, 6),
                )

        # If no val set, use the minimum train loss as a proxy for best_val_loss
        if best_state_dict is None:
            best_state_dict = {
                k: v.cpu().clone() for k, v in self.student.state_dict().items()
            }
            metrics.best_epoch = self.config.epochs - 1
            metrics.best_val_loss = min(metrics.train_losses) if metrics.train_losses else 0.0

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
        _, agreement = self._evaluate(tensor)
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
        (multiplied by T² as per Hinton et al. 2015).  When ``loss_fn ==
        "mse"`` the loss is MSE on the raw Q-value logits (no temperature
        scaling), which is natural for regression-style Q-value matching.
        """
        T = self.config.temperature
        if self.config.loss_fn == "mse":
            return F.mse_loss(student_logits, teacher_logits)

        # KL divergence on temperature-softened distributions
        teacher_soft = F.log_softmax(teacher_logits / T, dim=-1)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        # kl_div expects (input=log_probs, target=log_probs) with log_target=True
        kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean", log_target=True)
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
    ) -> torch.Tensor:
        """Blend soft distillation loss with optional hard supervision."""
        soft_loss = self._distillation_loss(teacher_logits, student_logits)
        alpha = self.config.alpha
        if alpha == 0.0:
            return soft_loss
        hard_loss = self._hard_loss(teacher_logits, student_logits)
        return alpha * hard_loss + (1.0 - alpha) * soft_loss

    def _run_epoch(self, train_tensor: torch.Tensor) -> float:
        """Run one training epoch; return mean loss over all mini-batches."""
        self.student.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self._iter_batches(train_tensor):
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(batch)

            # Student forward
            student_logits = self.student(batch)

            loss = self._compute_loss(teacher_logits, student_logits)

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(
        self, val_tensor: torch.Tensor
    ) -> Tuple[float, float]:
        """Evaluate distillation loss and action agreement on a state tensor.

        Returns
        -------
        val_loss : float
            Mean distillation loss on the validation set.
        action_agreement : float
            Top-1 agreement between teacher and student argmax actions.
        """
        self.student.eval()
        self.teacher.eval()

        total_loss = 0.0
        n_batches = 0
        n_agree = 0
        n_total = 0

        for batch in self._iter_batches(val_tensor):
            teacher_logits = self.teacher(batch)
            student_logits = self.student(batch)

            loss = self._compute_loss(teacher_logits, student_logits)
            total_loss += loss.item()
            n_batches += 1

            teacher_actions = teacher_logits.argmax(dim=-1)
            student_actions = student_logits.argmax(dim=-1)
            n_agree += (teacher_actions == student_actions).sum().item()
            n_total += batch.size(0)

        val_loss = total_loss / max(n_batches, 1)
        agreement = n_agree / max(n_total, 1)
        return val_loss, agreement

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
