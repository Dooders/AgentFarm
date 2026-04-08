"""Quantization-aware training (QAT) for distilled student Q-networks.

Overview
--------
This module implements **weight-only QAT** for ``StudentQNetwork`` (and
compatible ``BaseQNetwork``) checkpoints produced by the distillation pipeline
(see :mod:`farm.core.decision.training.trainer_distill`).

QAT scope: weight-only
~~~~~~~~~~~~~~~~~~~~~~
``BaseQNetwork`` / ``StudentQNetwork`` interleave ``LayerNorm`` layers between
every ``Linear`` pair::

    Linear → LayerNorm → ReLU → Dropout → ...

``LayerNorm`` is **not** fuseable with ``Linear`` under the standard
``fbgemm`` / ``qnnpack`` backends, making full-activation QAT require
significant graph surgery.  This module therefore applies fake-quantization
**only to ``nn.Linear`` weights** and leaves activations in ``float32``.

This matches the PTQ **dynamic** mode target (see ``quantize_ptq.py``), so
QAT-converted models can be compared directly to PTQ-dynamic outputs using
:func:`~farm.core.decision.training.quantize_ptq.compare_outputs`.

When to use QAT vs PTQ
~~~~~~~~~~~~~~~~~~~~~~
* **PTQ-dynamic** (fast, zero training cost): run first.  Usually sufficient
  when action-agreement ≥ 90 % and mean Q-error is small.
* **QAT** (slower, ~5–20 fine-tuning epochs): use when PTQ-dynamic
  action-agreement is unacceptable (e.g. < 85 %) or when Q-value error
  degrades task metrics.  QAT adapts model weights to quantization noise and
  typically recovers 1–5 % agreement vs PTQ.

Recommended recipe
~~~~~~~~~~~~~~~~~~
1. Distil float student:  ``scripts/run_distillation.py``
2. PTQ dynamic:           ``scripts/quantize_distilled.py``
3. **(If PTQ quality is insufficient)** QAT finetune:
   ``scripts/qat_distilled.py --student-ckpt <float.pt>``

This produces ``student_<pair>_qat.pt`` (float QAT checkpoint) and
``student_<pair>_qat_int8.pt`` (converted int8 model).

Quantization mechanics
~~~~~~~~~~~~~~~~~~~~~~
During QAT training, each ``nn.Linear`` weight tensor is replaced with a
**fake-quantised** copy at every forward pass using the **straight-through
estimator (STE)**:

1. Compute symmetric per-tensor ``scale = max(|W|) / 127``.
2. Quantise: ``W_q = clamp(round(W / scale), -128, 127)``.
3. Dequantise: ``W_hat = W_q * scale``.
4. Use ``W_hat`` for the matmul; gradients flow through as if the
   quantisation did not happen (STE).

After training, :func:`~QATTrainer.convert` applies
``torch.ao.quantization.quantize_dynamic`` to produce a true int8 model
identical in format to the PTQ-dynamic output.

Known limitations
~~~~~~~~~~~~~~~~~
* **Weight-only**: activations stay in ``float32``.  A gap remains vs.
  full-integer inference (e.g. on-device NPU).  Extend to activation QAT
  only if full fusion + LayerNorm replacement is acceptable.
* **CPU-only int8 kernels**: CUDA paths dequantize weights before matmul at
  runtime.  QAT does not change this behaviour.
* **PyTorch ≥ 2.0 required**.  ``torch.ao.quantization.quantize_dynamic``
  is used for the convert step; it is available in 2.x but marked deprecated
  in 2.11 in favour of ``torchao``.  Migration to ``torchao`` is a future
  non-goal.

Save / load format
~~~~~~~~~~~~~~~~~~
The QAT float checkpoint (before conversion) is saved as a plain state-dict
(``torch.save(state_dict, path)``), compatible with ``weights_only=True``.
The converted int8 model is saved as a full model pickle
(``torch.save(model, path)``), identical to the PTQ format.
Companion JSON files record config and architecture metadata.

Example
-------
::

    from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
    from farm.core.decision.training.quantize_qat import QATConfig, QATTrainer

    teacher = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    student = StudentQNetwork(input_dim=8, output_dim=4, parent_hidden_size=64)
    # ... load distilled weights into student ...

    import numpy as np
    states = np.random.randn(500, 8).astype("float32")

    cfg = QATConfig(epochs=5, learning_rate=1e-4)
    trainer = QATTrainer(teacher, student, cfg)
    metrics = trainer.train(states, checkpoint_path="student_qat.pt")

    # Convert and save int8
    q_model = trainer.convert()
    trainer.save_quantized(q_model, "student_qat_int8.pt")
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from farm.utils.logging import get_logger

from .quantize_ptq import QuantizationResult, _load_full_model_checkpoint

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim: torch.ao.quantization
# ---------------------------------------------------------------------------
try:
    import torch.ao.quantization as tq
except ImportError:  # pragma: no cover
    import torch.quantization as tq  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Weight-only fake-quantization layer
# ---------------------------------------------------------------------------


class WeightOnlyFakeQuantLinear(nn.Linear):
    """Drop-in replacement for ``nn.Linear`` with STE weight fake-quantization.

    During **training** each forward pass applies per-tensor symmetric int8
    fake-quantization to the weight matrix before the matmul.  Gradients flow
    through the quantisation step unchanged (straight-through estimator).

    During **evaluation** the full-precision weights are used unchanged.
    Call :func:`~QATTrainer.convert` after training to obtain a true int8 model.

    The fake-quantization formula is::

        scale      = max(|W|) / 127          # per-tensor symmetric
        W_q        = clamp(round(W / scale), -128, 127)
        W_hat      = W_q * scale             # dequantised
        forward(x) = x @ W_hat.T + bias

    STE: the backward pass computes gradients as if ``W_hat == W``.

    Parameters
    ----------
    in_features, out_features, bias:
        Same as :class:`torch.nn.Linear`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self._fake_quant_weight(self.weight)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)

    @staticmethod
    def _fake_quant_weight(weight: torch.Tensor) -> torch.Tensor:
        """Return a fake-quantised copy of *weight* using STE."""
        w_max = weight.abs().max().clamp(min=1e-8)
        scale = w_max / 127.0
        w_q = torch.clamp(torch.round(weight / scale), -128.0, 127.0)
        # STE: detach the quantisation error so gradient flows through weight
        return weight + (w_q * scale - weight).detach()

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "WeightOnlyFakeQuantLinear":
        """Construct a :class:`WeightOnlyFakeQuantLinear` from an existing linear layer.

        Weight and bias data are **copied** (not shared), so the original
        *linear* is not mutated.
        """
        new = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
        new.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            new.bias.data.copy_(linear.bias.data)
        return new


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QATConfig:
    """Hyperparameters for the weight-only QAT training loop.

    Attributes
    ----------
    epochs:
        Number of full passes over the state buffer during QAT fine-tuning.
    learning_rate:
        Initial learning rate for the Adam optimizer.
    batch_size:
        Mini-batch size for each gradient update.
    max_grad_norm:
        Maximum L2-norm of gradients before clipping.  Set to ``None`` to
        disable clipping.
    val_fraction:
        Fraction of states to hold out as a validation set (0 ≤ val_fraction
        < 1).  When 0, no validation is performed.
    seed:
        Optional integer seed for reproducibility.
    loss_fn:
        Which loss function to use for the distillation signal.
        ``"kl"`` uses temperature-scaled KL divergence (Hinton et al. 2015);
        ``"mse"`` applies MSE directly on the raw Q-value logits (default,
        simpler and natural for regression-style Q-value matching).
    temperature:
        Softmax temperature for the soft (KL) distillation loss.  Ignored
        when ``loss_fn == "mse"``.
    alpha:
        Blending weight for the *soft* distillation loss.  The total loss is::

            loss = alpha * soft_loss + (1 - alpha) * hard_loss

        Set ``alpha = 1.0`` for pure soft-label distillation (default).
        Set ``alpha = 0.0`` for pure hard-label supervision.
        Ignored when ``loss_fn == "mse"`` (no hard-label term).
    dtype:
        Target weight dtype for the convert step.  ``"qint8"`` (default)
        produces ``torch.qint8`` int8 models via ``quantize_dynamic``.
    """

    epochs: int = 5
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_grad_norm: Optional[float] = 1.0
    val_fraction: float = 0.1
    seed: Optional[int] = None
    loss_fn: str = "mse"  # "mse" or "kl"
    temperature: float = 3.0
    alpha: float = 1.0
    dtype: str = "qint8"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
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
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.dtype != "qint8":
            raise ValueError("dtype must be 'qint8'")

    def torch_dtype(self) -> torch.dtype:
        """Return the ``torch.dtype`` corresponding to *self.dtype*."""
        return torch.qint8


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------


@dataclass
class QATMetrics:
    """Per-epoch training and validation metrics from :class:`QATTrainer`.

    Attributes
    ----------
    train_losses:
        Mean total loss per training epoch.
    train_soft_losses:
        Mean soft (KL/MSE) loss per training epoch.
    train_hard_losses:
        Mean hard CE loss per training epoch (empty when ``alpha == 1.0``).
    val_losses:
        Mean total loss per validation epoch (empty when
        ``val_fraction == 0``).
    action_agreements:
        Fraction of states where the student (QAT) argmax action matches
        the teacher argmax action, evaluated on the validation set after
        each epoch.
    mean_prob_similarities:
        Mean probability similarity between teacher and student soft
        distributions on the validation set.
    best_val_loss:
        Lowest validation loss seen during training.
    best_epoch:
        Epoch index (0-based) at which the best validation loss occurred.
    elapsed_seconds:
        Total wall-clock training time.
    """

    train_losses: List[float] = field(default_factory=list)
    train_soft_losses: List[float] = field(default_factory=list)
    train_hard_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    action_agreements: List[float] = field(default_factory=list)
    mean_prob_similarities: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# QAT Trainer
# ---------------------------------------------------------------------------


class QATTrainer:
    """Weight-only quantization-aware fine-tuning for student Q-networks.

    The trainer:

    1. Deep-copies the student and replaces every ``nn.Linear`` with
       :class:`WeightOnlyFakeQuantLinear` (**prepare** step).
    2. Keeps the teacher frozen and minimises the same distillation objective
       as :class:`~farm.core.decision.training.trainer_distill.DistillationTrainer`
       (**train** step).
    3. After training, applies ``torch.ao.quantization.quantize_dynamic`` to
       produce a true int8 model identical in format to the PTQ-dynamic output
       (**convert** step).

    Parameters
    ----------
    teacher:
        Frozen teacher ``nn.Module``.  Put into ``eval`` mode; gradients
        disabled during training.
    student:
        Float student ``nn.Module`` (typically a distilled checkpoint).
        **Not mutated**: :meth:`prepare` deep-copies it internally.
    config:
        :class:`QATConfig` controlling all hyperparameters.
    device:
        Target PyTorch device.  Defaults to ``torch.device("cpu")``.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[QATConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config: QATConfig = config or QATConfig()
        self.device: torch.device = device or torch.device("cpu")

        self.teacher: nn.Module = teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self._float_student: nn.Module = student.to(self.device)

        # Will be set by prepare()
        self._qat_student: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._prepared: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Insert fake-quantization nodes into the student.

        Deep-copies the float student and replaces every ``nn.Linear`` with
        :class:`WeightOnlyFakeQuantLinear`.  After this call the QAT student
        is accessible as :attr:`qat_student`.

        Calling :meth:`prepare` a second time resets to the original float
        student weights.
        """
        qat_model = copy.deepcopy(self._float_student).to(self.device)
        _replace_linear_with_fakeq(qat_model)
        qat_model.train()

        self._qat_student = qat_model
        self._optimizer = torch.optim.Adam(
            self._qat_student.parameters(), lr=self.config.learning_rate
        )
        self._prepared = True
        logger.info(
            "qat_prepared",
            n_fakeq_layers=sum(
                1
                for m in self._qat_student.modules()
                if isinstance(m, WeightOnlyFakeQuantLinear)
            ),
        )

    def train(
        self,
        states: np.ndarray,
        checkpoint_path: Optional[str] = None,
    ) -> QATMetrics:
        """Run the QAT fine-tuning loop.

        Calls :meth:`prepare` automatically if not yet prepared.

        Parameters
        ----------
        states:
            NumPy array of shape ``(N, input_dim)`` with dtype ``float32``.
            The same state distribution used for distillation should be
            reused here (e.g. from ``--states-file`` in
            ``scripts/run_distillation.py``).
        checkpoint_path:
            If given, the best QAT float student weights are saved as a
            state-dict ``.pt`` file (loadable with ``weights_only=True``)
            together with a companion ``<checkpoint_path>.json`` metadata
            file.  This is the **float QAT checkpoint** (before conversion).
            Call :meth:`convert` afterwards to get the int8 model.

        Returns
        -------
        QATMetrics
            Populated metrics object with per-epoch train/val losses,
            action agreement, and probability similarity.
        """
        if len(states) == 0:
            raise ValueError("states must be non-empty; got an array with 0 samples.")

        if not self._prepared:
            self.prepare()

        if self.config.seed is not None:
            self._set_seed(self.config.seed)

        train_states, val_states = self._split_states(states)
        train_tensor = torch.tensor(train_states, dtype=torch.float32, device=self.device)
        val_tensor = (
            torch.tensor(val_states, dtype=torch.float32, device=self.device)
            if len(val_states) > 0
            else None
        )

        metrics = QATMetrics()
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        best_train_loss: float = float("inf")

        t0 = time.perf_counter()

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
                        for k, v in self._qat_student.state_dict().items()
                    }

                logger.info(
                    "qat_epoch",
                    epoch=epoch,
                    train_loss=round(train_loss, 6),
                    val_loss=round(val_loss, 6),
                    action_agreement=round(agreement, 4),
                    mean_prob_similarity=round(prob_sim, 4),
                )
            else:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    metrics.best_val_loss = train_loss
                    metrics.best_epoch = epoch
                    best_state_dict = {
                        k: v.cpu().clone()
                        for k, v in self._qat_student.state_dict().items()
                    }

                logger.info(
                    "qat_epoch",
                    epoch=epoch,
                    train_loss=round(train_loss, 6),
                )

        metrics.elapsed_seconds = time.perf_counter() - t0

        if best_state_dict is None:
            best_state_dict = {
                k: v.cpu().clone() for k, v in self._qat_student.state_dict().items()
            }

        # Load best weights back
        self._qat_student.load_state_dict(
            {k: v.to(self.device) for k, v in best_state_dict.items()}
        )

        if checkpoint_path is not None:
            self._save_qat_checkpoint(checkpoint_path, best_state_dict, metrics)

        logger.info(
            "qat_training_complete",
            best_epoch=metrics.best_epoch,
            best_val_loss=round(metrics.best_val_loss, 6),
            elapsed_s=round(metrics.elapsed_seconds, 3),
        )
        return metrics

    def convert(self) -> nn.Module:
        """Convert the trained QAT student to a true int8 quantized model.

        Applies ``torch.ao.quantization.quantize_dynamic`` to the QAT student
        (set to eval mode) targeting ``nn.Linear`` and
        :class:`WeightOnlyFakeQuantLinear` layers.  The result is a model
        whose ``Linear`` weights are stored in int8 format, identical to the
        PTQ-dynamic output from :class:`~quantize_ptq.PostTrainingQuantizer`.

        Must be called after :meth:`train` (or at least :meth:`prepare`).

        Returns
        -------
        nn.Module
            The quantized model (ready for ``eval()`` inference).

        Raises
        ------
        RuntimeError
            If :meth:`prepare` has not been called.
        """
        if self._qat_student is None:
            raise RuntimeError(
                "QATTrainer.prepare() must be called before convert(). "
                "Call trainer.train(states) to prepare and train in one step."
            )

        self._qat_student.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            q_model = tq.quantize_dynamic(
                copy.deepcopy(self._qat_student),
                {nn.Linear, WeightOnlyFakeQuantLinear},
                dtype=self.config.torch_dtype(),
            )
        q_model.eval()
        logger.info(
            "qat_converted",
            dtype=self.config.dtype,
            n_linear=sum(1 for m in self._qat_student.modules() if isinstance(m, nn.Linear)),
        )
        return q_model

    def save_quantized(
        self,
        quantized_model: nn.Module,
        path: str,
        arch_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a converted int8 model to *path* (PTQ-compatible format).

        Saves as a full model pickle (``torch.save(model, path)``) together
        with a companion JSON metadata file at ``<path>.json``.  The format
        is identical to the output of
        :meth:`~quantize_ptq.PostTrainingQuantizer.save_checkpoint`.

        Parameters
        ----------
        quantized_model:
            The int8 ``nn.Module`` returned by :meth:`convert`.
        path:
            Destination file path (e.g. ``"student_A_qat_int8.pt"``).
        arch_kwargs:
            Optional architecture constructor arguments to record in JSON.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(quantized_model, path)
        n_linear = sum(1 for m in quantized_model.modules() if isinstance(m, nn.Linear))
        float_bytes = sum(
            p.numel() * p.element_size()
            for name, p in quantized_model.named_parameters()
            if "weight" in name
        )
        q_bytes = float_bytes // 4
        quant_result = QuantizationResult(
            mode="qat",
            dtype=self.config.dtype,
            backend=str(torch.backends.quantized.engine),
            calibration_samples=0,
            elapsed_seconds=0.0,
            linear_layers_quantized=n_linear,
            float_param_bytes=float_bytes,
            quantized_param_bytes=q_bytes,
            notes=[],
        )
        meta = {
            "quantization": quant_result.to_dict(),
            "qat": {
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "loss_fn": self.config.loss_fn,
                "dtype": self.config.dtype,
                "scope": "weight_only",
            },
            "arch_kwargs": arch_kwargs or {},
            "notes": _build_notes(),
        }
        json_path = path + ".json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("qat_quantized_saved", path=path, json_path=json_path)

    @property
    def qat_student(self) -> Optional[nn.Module]:
        """The prepared (and optionally trained) QAT student module."""
        return self._qat_student

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
        n = len(states)
        empty_suffix = (0,) + states.shape[1:]
        if n == 0:
            return states, np.empty(empty_suffix, dtype=states.dtype)
        val_fraction = self.config.val_fraction
        if val_fraction <= 0.0:
            return states, np.empty(empty_suffix, dtype=states.dtype)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val
        if n_train <= 0:
            return states, np.empty(empty_suffix, dtype=states.dtype)
        indices = np.random.permutation(n)
        return states[indices[:n_train]], states[indices[n_train:]]

    def _iter_batches(
        self, tensor: torch.Tensor
    ) -> Generator[torch.Tensor, None, None]:
        n = tensor.size(0)
        perm = torch.randperm(n, device=self.device)
        for start in range(0, n, self.config.batch_size):
            idx = perm[start : start + self.config.batch_size]
            yield tensor[idx]

    def _soft_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.loss_fn == "mse":
            return F.mse_loss(student_logits, teacher_logits)
        T = self.config.temperature
        p_t = F.softmax(teacher_logits / T, dim=-1)
        p_s = F.log_softmax(student_logits / T, dim=-1)
        return F.kl_div(p_s, p_t, reduction="batchmean", log_target=False) * (T**2)

    def _hard_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(student_logits, teacher_logits.argmax(dim=-1))

    def _compute_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        soft = self._soft_loss(teacher_logits, student_logits)
        alpha = self.config.alpha
        if alpha == 1.0 or self.config.loss_fn == "mse":
            return soft, soft, None
        hard = self._hard_loss(teacher_logits, student_logits)
        total = alpha * soft + (1.0 - alpha) * hard
        return total, soft, hard

    def _run_epoch(
        self, train_tensor: torch.Tensor
    ) -> Tuple[float, float, Optional[float]]:
        self._qat_student.train()
        total_sum = soft_sum = hard_sum = 0.0
        n_batches = 0
        has_hard = False

        for batch in self._iter_batches(train_tensor):
            with torch.no_grad():
                teacher_logits = self.teacher(batch)
            student_logits = self._qat_student(batch)
            total, soft, hard = self._compute_loss(teacher_logits, student_logits)

            self._optimizer.zero_grad()
            total.backward()
            if self.config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self._qat_student.parameters(), self.config.max_grad_norm
                )
            self._optimizer.step()

            total_sum += total.item()
            soft_sum += soft.item()
            if hard is not None:
                hard_sum += hard.item()
                has_hard = True
            n_batches += 1

        denom = max(n_batches, 1)
        mean_hard: Optional[float] = (hard_sum / denom) if has_hard else None
        return total_sum / denom, soft_sum / denom, mean_hard

    @torch.no_grad()
    def _evaluate(
        self, val_tensor: torch.Tensor
    ) -> Tuple[float, float, float]:
        self._qat_student.eval()
        self.teacher.eval()

        total_loss = 0.0
        n_batches = 0
        n_agree = 0
        n_total = 0
        prob_sim_sum = 0.0

        T = self.config.temperature

        for batch in self._iter_batches(val_tensor):
            teacher_logits = self.teacher(batch)
            student_logits = self._qat_student(batch)
            total, _, _ = self._compute_loss(teacher_logits, student_logits)
            total_loss += total.item()
            n_batches += 1

            n_agree += (teacher_logits.argmax(dim=-1) == student_logits.argmax(dim=-1)).sum().item()
            n_total += batch.size(0)

            p_t = F.softmax(teacher_logits / T, dim=-1)
            p_s = F.softmax(student_logits / T, dim=-1)
            prob_sim_sum += (1.0 - (p_t - p_s).abs().mean()).item()

        return (
            total_loss / max(n_batches, 1),
            n_agree / max(n_total, 1),
            prob_sim_sum / max(n_batches, 1),
        )

    def _save_qat_checkpoint(
        self,
        path: str,
        state_dict: Dict[str, torch.Tensor],
        metrics: QATMetrics,
    ) -> None:
        """Save QAT float student state-dict and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state_dict, path)
        logger.info("qat_float_checkpoint_saved", path=path)

        meta_path = path + ".json"
        metadata = {
            "config": {
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "max_grad_norm": self.config.max_grad_norm,
                "val_fraction": self.config.val_fraction,
                "seed": self.config.seed,
                "loss_fn": self.config.loss_fn,
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "dtype": self.config.dtype,
                "scope": "weight_only",
            },
            "metrics": metrics.to_dict(),
            "notes": _build_notes(),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, allow_nan=False)
        logger.info("qat_float_metadata_saved", path=meta_path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _replace_linear_with_fakeq(model: nn.Module) -> None:
    """Recursively replace ``nn.Linear`` with :class:`WeightOnlyFakeQuantLinear`.

    Operates **in-place** on *model*.  :class:`WeightOnlyFakeQuantLinear`
    instances are left unchanged (idempotent).
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(
            child, WeightOnlyFakeQuantLinear
        ):
            setattr(model, name, WeightOnlyFakeQuantLinear.from_linear(child))
        else:
            _replace_linear_with_fakeq(child)


def _build_notes() -> List[str]:
    return [
        "Scope: weight-only QAT. Linear weights are fake-quantized during training; "
        "activations stay in float32.",
        "LayerNorm layers are not quantized (not fuseable with Linear under fbgemm/qnnpack).",
        "Dropout layers are no-ops in eval() mode and are not quantized.",
        "Convert step applies torch.ao.quantization.quantize_dynamic (int8 weights, "
        "float32 activations). Result is identical in format to PTQ-dynamic output.",
        "int8 kernel acceleration is CPU-only; CUDA paths dequantize weights before matmul.",
    ]


def load_qat_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a converted QAT int8 model saved by :meth:`QATTrainer.save_quantized`.

    .. warning::
        This function calls ``torch.load(path, weights_only=False)`` because
        quantized ``PackedParams`` tensors cannot be loaded with
        ``weights_only=True``.  Only load checkpoints from trusted sources.

    Parameters
    ----------
    path:
        Path to the ``.pt`` file (full model pickle).
    device:
        Target device.  Defaults to ``torch.device("cpu")``.

    Returns
    -------
    model:
        The deserialised quantized ``nn.Module``, set to ``eval()`` mode.
    metadata:
        Dict parsed from the companion ``<path>.json`` file, or an empty
        dict if the JSON file is absent.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    return _load_full_model_checkpoint(
        path,
        device,
        not_found_msg=f"QAT checkpoint not found: {path}",
    )
