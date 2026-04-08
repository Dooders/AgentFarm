"""Fine-tuning pipeline for crossover child Q-networks.

After distillation → quantization → crossover the resulting child network is a
heuristic blend of its parents and may not be optimal for the task.  This
module implements a **supervised fine-tuning stage** that closes the
performance gap by optimising the child against a frozen *reference* model
(one of the parents, a distilled student, or any ``nn.Module`` that produces
Q-value logits).

The fine-tuning objective mirrors the knowledge-distillation loss from
:mod:`~farm.core.decision.training.trainer_distill` – soft KL/MSE targets from
the reference plus an optional hard cross-entropy term – applied to a *state
tensor* buffer (offline replay buffer export, on-policy rollouts, or synthetic
calibration states).

Key features
------------
- **Before / after metrics** – the reference-vs-child loss and action
  agreement are recorded *before* any weight update so improvements can be
  reported in the returned :class:`FineTuningMetrics`.
- **LR schedule** – optional :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`
  triggered by validation loss; enabled when
  ``FineTuningConfig.lr_schedule_patience > 0``.
- **Checkpointing** – best child weights (by validation loss, or by training
  loss when ``val_fraction == 0``) are saved to a ``.pt`` file with a
  companion ``.json`` metadata file, consistent with the
  :class:`~farm.core.decision.training.trainer_distill.DistillationTrainer`
  pattern.
- **Reproducibility** – optional seed resets all RNGs before training.

Typical usage
-------------
::

    from farm.core.decision.base_dqn import BaseQNetwork
    from farm.core.decision.training.crossover import initialize_child_from_crossover
    from farm.core.decision.training.finetune import FineTuningConfig, FineTuner

    parent_a = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    parent_b = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    child = initialize_child_from_crossover(parent_a, parent_b, strategy="weighted")

    # Use parent_a as reference teacher
    cfg = FineTuningConfig(learning_rate=5e-4, epochs=10, seed=42)
    tuner = FineTuner(reference=parent_a, child=child, config=cfg)

    import numpy as np
    states = np.random.randn(500, 8).astype("float32")
    metrics = tuner.finetune(states, checkpoint_path="child_finetuned.pt")
    print(f"Before: val_loss={metrics.initial_val_loss:.4f}, "
          f"agreement={metrics.initial_action_agreement:.2%}")
    print(f"After : val_loss={metrics.best_val_loss:.4f}")
"""

from __future__ import annotations

import json
import math
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
class FineTuningConfig:
    """Hyperparameters for the child network fine-tuning loop.

    Attributes
    ----------
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
        ``random``, ``numpy``, and ``torch`` RNGs before fine-tuning starts.
    loss_fn:
        Which loss function to use for the soft distillation signal.
        ``"kl"`` uses temperature-scaled KL divergence (Hinton et al. 2015);
        ``"mse"`` applies MSE directly on the raw Q-value logits (no softmax).
    temperature:
        Softmax temperature applied to both reference and child logits when
        computing the soft (KL-divergence) loss.  Higher values produce
        softer probability distributions.  Must be > 0.
    temp_decay:
        Multiplicative factor applied to ``temperature`` after each epoch.
        Set to ``1.0`` (the default) to disable decay.  Must be in ``(0, 1]``.
    alpha:
        Blending weight for the *soft* distillation loss::

            loss = alpha * soft_loss + (1 - alpha) * hard_loss

        ``alpha = 1.0`` → pure soft-label (default).
        ``alpha = 0.0`` → pure hard-label supervision.
    lr_schedule_patience:
        Number of epochs with no improvement in validation loss before the
        learning rate is reduced.  Set to ``0`` (default) to disable the
        :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` schedule.
        Requires ``val_fraction > 0`` to have effect.
    lr_schedule_factor:
        Multiplicative factor by which the learning rate is reduced when the
        schedule fires.  Must be in ``(0, 1)``.  Ignored when
        ``lr_schedule_patience == 0``.
    """

    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 32
    max_grad_norm: Optional[float] = 1.0
    val_fraction: float = 0.1
    seed: Optional[int] = None
    loss_fn: str = "kl"  # "kl" or "mse"
    temperature: float = 3.0
    temp_decay: float = 1.0
    alpha: float = 1.0
    lr_schedule_patience: int = 0
    lr_schedule_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when specified")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.loss_fn not in ("kl", "mse"):
            raise ValueError("loss_fn must be 'kl' or 'mse'")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < self.temp_decay <= 1.0:
            raise ValueError("temp_decay must be in (0, 1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.lr_schedule_patience < 0:
            raise ValueError("lr_schedule_patience must be >= 0")
        if not 0.0 < self.lr_schedule_factor < 1.0:
            raise ValueError("lr_schedule_factor must be in (0, 1)")


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------


@dataclass
class FineTuningMetrics:
    """Per-epoch training and validation metrics from :class:`FineTuner`.

    Attributes
    ----------
    initial_val_loss:
        Reference-vs-child loss computed on the validation set *before* any
        weight update.  ``float("inf")`` when ``val_fraction == 0``.
    initial_action_agreement:
        Top-1 action agreement between reference and child *before* any weight
        update.  ``0.0`` when ``val_fraction == 0``.
    train_losses:
        Mean total fine-tuning loss per training epoch.
    train_soft_losses:
        Mean soft (KL/MSE) loss per training epoch.
    train_hard_losses:
        Mean hard CE loss per training epoch (empty when ``alpha == 1.0`` or
        ``loss_fn == "mse"``).
    val_losses:
        Mean total fine-tuning loss per validation epoch (empty when
        ``val_fraction == 0``).
    action_agreements:
        Fraction of states where the child argmax action matches the reference
        argmax action, evaluated on the *validation* set after each epoch
        (empty when ``val_fraction == 0``).
    mean_prob_similarities:
        Mean probability similarity between reference and child soft
        distributions on the validation set.  Values near 1 indicate close
        distributional agreement.  Empty when ``val_fraction == 0``.
    best_val_loss:
        Lowest validation loss seen during fine-tuning.
    best_epoch:
        Epoch index (0-based) at which the best validation loss was achieved.
    """

    initial_val_loss: float = float("inf")
    initial_action_agreement: float = 0.0
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
            "initial_val_loss": self.initial_val_loss,
            "initial_action_agreement": self.initial_action_agreement,
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
# Fine-tuner
# ---------------------------------------------------------------------------


class FineTuner:
    """Fine-tune a crossover child Q-network against a frozen reference model.

    The fine-tuner optimises the *child* network to minimise a combination of:

    * **Soft distillation loss** – KL divergence (or MSE) between
      temperature-scaled reference and child outputs.
    * **Hard supervision loss** – cross-entropy against the reference's argmax
      action (only when ``alpha < 1.0``).

    The blended objective follows the Hinton et al. (2015) convention::

        loss = alpha * soft_loss + (1 - alpha) * hard_loss

    where ``alpha = 1.0`` (the default) gives pure soft-label fine-tuning.

    **Before / after comparison**: the reference-vs-child loss and action
    agreement are measured on the held-out validation set *before* any weight
    update and stored in :attr:`FineTuningMetrics.initial_val_loss` /
    :attr:`FineTuningMetrics.initial_action_agreement`.

    Parameters
    ----------
    reference:
        Frozen reference ``nn.Module`` that provides soft / hard targets.
        Typically one of the crossover parents or a distilled teacher.  The
        module is put into ``eval`` mode and its gradients are disabled.
    child:
        The child network to fine-tune.  Must accept the same input shape as
        the reference and produce the same output shape.
    config:
        :class:`FineTuningConfig` controlling all hyperparameters.
    device:
        Target PyTorch device.  Defaults to ``torch.device("cpu")``.
    """

    def __init__(
        self,
        reference: nn.Module,
        child: nn.Module,
        config: Optional[FineTuningConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config: FineTuningConfig = config or FineTuningConfig()
        self.device: torch.device = device or torch.device("cpu")

        self.reference: nn.Module = reference.to(self.device)
        self.child: nn.Module = child.to(self.device)

        # Freeze reference
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad = False

        # Optimizer for child only
        self.optimizer = torch.optim.Adam(
            self.child.parameters(), lr=self.config.learning_rate
        )

        # Optional LR scheduler (ReduceLROnPlateau)
        self.scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
        if self.config.lr_schedule_patience > 0:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.lr_schedule_patience,
                factor=self.config.lr_schedule_factor,
            )

        # Mutable temperature that can be decayed across epochs
        self._current_temperature: float = self.config.temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def finetune(
        self,
        states: np.ndarray,
        checkpoint_path: Optional[str] = None,
    ) -> FineTuningMetrics:
        """Run the fine-tuning loop and return before/after metrics.

        Parameters
        ----------
        states:
            NumPy array of shape ``(N, input_dim)`` representing the target
            dataset – e.g. an offline replay buffer export, on-policy rollout
            states, or synthetic calibration states.
        checkpoint_path:
            If given, the best child weights are saved to this ``.pt`` file
            and a companion ``<checkpoint_path>.json`` metadata file is written
            next to it.

        Returns
        -------
        FineTuningMetrics
            Populated metrics object including before-training baseline metrics
            (``initial_val_loss``, ``initial_action_agreement``) and per-epoch
            train / val losses, action agreement, and probability similarity.
        """
        if len(states) == 0:
            raise ValueError("states must be non-empty; got an array with 0 samples.")

        if self.config.seed is not None:
            self._set_seed(self.config.seed)

        # Reset temperature at the start of each fine-tuning run
        self._current_temperature = self.config.temperature

        train_states, val_states = self._split_states(states)
        train_tensor = torch.tensor(train_states, dtype=torch.float32, device=self.device)
        val_tensor = (
            torch.tensor(val_states, dtype=torch.float32, device=self.device)
            if len(val_states) > 0
            else None
        )

        metrics = FineTuningMetrics()

        # Record before-training metrics on validation set
        if val_tensor is not None and len(val_tensor) > 0:
            init_loss, init_agreement, _ = self._evaluate(val_tensor)
            metrics.initial_val_loss = init_loss
            metrics.initial_action_agreement = init_agreement
            logger.info(
                "finetune_before_training",
                initial_val_loss=round(init_loss, 6),
                initial_action_agreement=round(init_agreement, 4),
            )

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
                        for k, v in self.child.state_dict().items()
                    }

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                logger.info(
                    "finetune_epoch",
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
                        for k, v in self.child.state_dict().items()
                    }

                logger.info(
                    "finetune_epoch",
                    epoch=epoch,
                    temperature=round(self._current_temperature, 6),
                    train_loss=round(train_loss, 6),
                    train_soft_loss=round(soft_loss, 6),
                    train_hard_loss=round(hard_loss, 6) if hard_loss is not None else None,
                )

            # Apply temperature decay after each epoch
            self._current_temperature *= self.config.temp_decay

        # Safety net for zero-epoch configs (epochs >= 1 is enforced by config)
        if best_state_dict is None:
            best_state_dict = {
                k: v.cpu().clone() for k, v in self.child.state_dict().items()
            }

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, best_state_dict, metrics)

        # Load best weights back into child
        self.child.load_state_dict(
            {k: v.to(self.device) for k, v in best_state_dict.items()}
        )

        return metrics

    def evaluate_agreement(self, states: np.ndarray) -> float:
        """Compute top-1 action agreement between reference and child.

        Parameters
        ----------
        states:
            NumPy array of shape ``(N, input_dim)``.

        Returns
        -------
        float
            Fraction of states where child argmax == reference argmax.
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
        ref_logits: torch.Tensor,
        child_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the soft fine-tuning loss between reference and child outputs.

        When ``loss_fn == "kl"`` the loss is temperature-scaled KL divergence
        (multiplied by T² as per Hinton et al. 2015).  When ``loss_fn == "mse"``
        the loss is MSE on the raw Q-value logits (no temperature scaling).
        """
        T = self._current_temperature
        if self.config.loss_fn == "mse":
            return F.mse_loss(child_logits, ref_logits)

        # KL divergence on temperature-softened distributions (Hinton et al. 2015)
        ref_soft = F.softmax(ref_logits / T, dim=-1)
        child_log_soft = F.log_softmax(child_logits / T, dim=-1)
        kl = F.kl_div(child_log_soft, ref_soft, reduction="batchmean", log_target=False)
        return kl * (T**2)

    def _hard_loss(
        self,
        ref_logits: torch.Tensor,
        child_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss of child logits against reference argmax actions."""
        hard_labels = ref_logits.argmax(dim=-1)
        return F.cross_entropy(child_logits, hard_labels)

    def _compute_loss(
        self,
        ref_logits: torch.Tensor,
        child_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Blend soft fine-tuning loss with optional hard supervision.

        Returns
        -------
        total_loss : torch.Tensor
        soft_loss : torch.Tensor
        hard_loss : torch.Tensor or None
        """
        soft_loss = self._distillation_loss(ref_logits, child_logits)
        alpha = self.config.alpha
        if alpha == 1.0 or self.config.loss_fn == "mse":
            return soft_loss, soft_loss, None
        hard_loss = self._hard_loss(ref_logits, child_logits)
        total = alpha * soft_loss + (1.0 - alpha) * hard_loss
        return total, soft_loss, hard_loss

    def _run_epoch(self, train_tensor: torch.Tensor) -> Tuple[float, float, Optional[float]]:
        """Run one training epoch; return (mean_total, mean_soft, mean_hard) loss."""
        self.child.train()
        total_loss_sum = 0.0
        soft_loss_sum = 0.0
        hard_loss_sum = 0.0
        n_batches = 0
        has_hard = False

        for batch in self._iter_batches(train_tensor):
            with torch.no_grad():
                ref_logits = self.reference(batch)

            child_logits = self.child(batch)
            total_loss, soft_loss, hard_loss = self._compute_loss(ref_logits, child_logits)

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.child.parameters(), self.config.max_grad_norm
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
        """Evaluate fine-tuning loss, action agreement, and probability similarity.

        Returns
        -------
        val_loss : float
        action_agreement : float
        mean_prob_similarity : float
        """
        self.child.eval()
        self.reference.eval()

        total_loss = 0.0
        n_batches = 0
        n_agree = 0
        n_total = 0
        prob_sim_sum = 0.0

        for batch in self._iter_batches(val_tensor):
            ref_logits = self.reference(batch)
            child_logits = self.child(batch)

            total, _, _ = self._compute_loss(ref_logits, child_logits)
            total_loss += total.item()
            n_batches += 1

            ref_actions = ref_logits.argmax(dim=-1)
            child_actions = child_logits.argmax(dim=-1)
            n_agree += (ref_actions == child_actions).sum().item()
            n_total += batch.size(0)

            T = self._current_temperature
            p_ref = F.softmax(ref_logits / T, dim=-1)
            p_child = F.softmax(child_logits / T, dim=-1)
            prob_sim_sum += (1.0 - (p_ref - p_child).abs().mean()).item()

        val_loss = total_loss / max(n_batches, 1)
        agreement = n_agree / max(n_total, 1)
        mean_prob_similarity = prob_sim_sum / max(n_batches, 1)
        return val_loss, agreement, mean_prob_similarity

    def _save_checkpoint(
        self,
        path: str,
        state_dict: Dict[str, torch.Tensor],
        metrics: FineTuningMetrics,
    ) -> None:
        """Save child weights and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state_dict, path)
        logger.info("finetune_checkpoint_saved", path=path)

        meta_path = path + ".json"
        metadata = {
            "config": {
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "max_grad_norm": self.config.max_grad_norm,
                "val_fraction": self.config.val_fraction,
                "seed": self.config.seed,
                "loss_fn": self.config.loss_fn,
                "temperature": self.config.temperature,
                "final_temperature": self._current_temperature,
                "temp_decay": self.config.temp_decay,
                "alpha": self.config.alpha,
                "lr_schedule_patience": self.config.lr_schedule_patience,
                "lr_schedule_factor": self.config.lr_schedule_factor,
            },
            "metrics": _sanitize_for_json(metrics.to_dict()),
        }
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2, allow_nan=False)
        logger.info("finetune_metadata_saved", path=meta_path)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-JSON-compliant floats (inf, -inf, nan) with ``None``."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj
