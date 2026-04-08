"""Post-training quantization (PTQ) for distilled student Q-networks.

This module applies post-training quantization to ``StudentQNetwork`` (and
compatible ``BaseQNetwork``) checkpoints produced by the distillation pipeline.
It supports two modes:

* **Dynamic quantization** (default, weight-only):  Converts ``nn.Linear``
  weight tensors to ``int8`` representation.  Activations remain in ``float32``
  and are dequantized on-the-fly before each matmul.  No calibration data
  required.

* **Static quantization** (activation-aware):  Inserts ``QuantStub`` /
  ``DeQuantStub`` observers, runs the model over a calibration set to collect
  activation statistics, then converts to a fully-quantized ``int8`` graph.
  Calibration uses the same state distribution as the distillation pipeline.

Architecture compatibility
--------------------------
``BaseQNetwork`` / ``StudentQNetwork`` use the following ``nn.Sequential``
layout::

    Linear → LayerNorm → ReLU → Dropout →
    Linear → LayerNorm → ReLU → Dropout →
    Linear

``LayerNorm`` is **not** fuseable with ``Linear`` under the standard
``fbgemm``/``qnnpack`` backends; it therefore remains in ``float32`` in static
mode.  ``Dropout`` is a no-op in ``eval()`` mode and is excluded from
quantization.

PyTorch version constraints
---------------------------
* Requires **PyTorch ≥ 2.0**.
* Dynamic quantization (``torch.ao.quantization.quantize_dynamic``) works on
  both CPU and CUDA host; int8 kernel dispatch is **CPU-only** (CUDA keeps
  float32 with weight packing).
* Static quantization is CPU-only; use ``backend="qnnpack"`` on ARM or
  ``backend="fbgemm"`` on x86.

Save / load format
------------------
Quantized models are persisted as full model pickles (``torch.save(model,
path)``) because ``PackedParams`` and other internal quantized tensor types
cannot be cleanly round-tripped via a plain state-dict.  Loading therefore
requires ``weights_only=False``; this is expected and documented.

A companion JSON metadata file (``<path>.json``) stores the quantization
config and architecture parameters needed to verify the checkpoint and
re-build a float reference model for comparison.

Example
-------
::

    from farm.core.decision.base_dqn import StudentQNetwork
    from farm.core.decision.training.quantize_ptq import (
        QuantizationConfig,
        PostTrainingQuantizer,
        load_quantized_checkpoint,
    )

    # Build / load your float student
    student = StudentQNetwork(input_dim=8, output_dim=4, parent_hidden_size=64)
    # ... load weights ...

    # Quantise (dynamic, weight-only)
    config = QuantizationConfig(mode="dynamic")
    quantizer = PostTrainingQuantizer(config)
    q_model, metadata = quantizer.quantize(student)

    # Save
    quantizer.save_checkpoint(q_model, "student_A_int8.pt", metadata)

    # Load back
    q_model_loaded, meta = load_quantized_checkpoint("student_A_int8.pt")
    q_model_loaded.eval()
    output = q_model_loaded(some_state_tensor)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from farm.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim: torch.ao.quantization was split from torch.quantization
# in PyTorch 1.13.  Both namespaces exist in 2.x but torch.ao is preferred.
# ---------------------------------------------------------------------------
try:
    import torch.ao.quantization as tq
except ImportError:  # pragma: no cover
    import torch.quantization as tq  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QuantizationConfig:
    """Hyperparameters for the post-training quantization pipeline.

    Attributes
    ----------
    mode:
        ``"dynamic"`` (default) performs weight-only int8 quantization without
        calibration.  ``"static"`` enables activation quantization and requires
        *calibration_states*.
    dtype:
        Target weight dtype.  Only ``"qint8"`` is supported by this PTQ
        implementation.
    backend:
        Quantization backend.  ``"qnnpack"`` is recommended for ARM / mobile;
        ``"fbgemm"`` for x86.  Dynamic quantization ignores this setting but
        static quantization uses it to set the global ``qconfig``.
    calibration_batches:
        Number of mini-batches to run through the model in static mode.
        Ignored in dynamic mode.
    calibration_batch_size:
        Size of each calibration mini-batch.  Ignored in dynamic mode.
    """

    mode: str = "dynamic"
    dtype: str = "qint8"
    backend: str = "qnnpack"
    calibration_batches: int = 10
    calibration_batch_size: int = 64

    def __post_init__(self) -> None:
        if self.mode not in ("dynamic", "static"):
            raise ValueError("mode must be 'dynamic' or 'static'")
        if self.dtype != "qint8":
            raise ValueError("dtype must be 'qint8'")
        if self.backend not in ("qnnpack", "fbgemm", "none"):
            raise ValueError("backend must be 'qnnpack', 'fbgemm', or 'none'")
        if self.mode == "static" and self.backend == "none":
            raise ValueError(
                "backend='none' is not valid for mode='static'; choose 'qnnpack' or 'fbgemm'."
            )
        if self.calibration_batches < 1:
            raise ValueError("calibration_batches must be >= 1")
        if self.calibration_batch_size < 1:
            raise ValueError("calibration_batch_size must be >= 1")

    def torch_dtype(self) -> torch.dtype:
        """Return the ``torch.dtype`` corresponding to *self.dtype*."""
        return torch.qint8


# ---------------------------------------------------------------------------
# Internal: static-quantization wrapper
# ---------------------------------------------------------------------------


class _QuantWrapper(nn.Module):
    """Wrap an arbitrary module with QuantStub / DeQuantStub.

    This allows static-quantization observers to track activation ranges
    across the full forward pass without modifying the inner module's code.

    Attributes
    ----------
    quant:
        ``QuantStub`` that dequantizes the float input to the quantized domain.
    dequant:
        ``DeQuantStub`` that dequantizes the output back to float.
    model:
        The wrapped (inner) module.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------------------------------
# Quantization report / metadata
# ---------------------------------------------------------------------------


@dataclass
class QuantizationResult:
    """Summary of a completed post-training quantization run.

    Attributes
    ----------
    mode:
        The quantization mode used (``"dynamic"`` or ``"static"``).
    dtype:
        The dtype string (``"qint8"``).
    backend:
        Backend used (``"qnnpack"``, ``"fbgemm"``, or ``"none"``).
    calibration_samples:
        Total number of calibration samples used (0 for dynamic mode).
    elapsed_seconds:
        Wall-clock time (seconds) taken by the quantization step.
    linear_layers_quantized:
        Number of ``nn.Linear`` layers that were converted.
    float_param_bytes:
        Estimated byte footprint of all tensors in the float model state dict.
    quantized_param_bytes:
        Estimated byte footprint of all tensors in the quantized model state
        dict (includes non-quantized tensors such as LayerNorm parameters).
    notes:
        Free-form notes (e.g. known limitations or fallback behaviour).
    """

    mode: str
    dtype: str
    backend: str
    calibration_samples: int
    elapsed_seconds: float
    linear_layers_quantized: int
    float_param_bytes: int
    quantized_param_bytes: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantizationResult":
        return cls(**d)


# ---------------------------------------------------------------------------
# Core quantiser
# ---------------------------------------------------------------------------


class PostTrainingQuantizer:
    """Apply post-training quantization to a float student Q-network.

    Parameters
    ----------
    config:
        :class:`QuantizationConfig` controlling the quantization strategy.

    Usage
    -----
    ::

        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, result = quantizer.quantize(student_model)
        quantizer.save_checkpoint(q_model, "student_A_int8.pt", result)
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        self.config = config or QuantizationConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def quantize(
        self,
        model: nn.Module,
        calibration_states: Optional[np.ndarray] = None,
    ) -> Tuple[nn.Module, QuantizationResult]:
        """Quantise *model* and return ``(quantized_model, result)``.

        Parameters
        ----------
        model:
            Float-precision ``nn.Module`` (typically ``StudentQNetwork`` or
            ``BaseQNetwork``) to quantise.  The original *model* is **not**
            modified in dynamic mode; a new module is returned.  In static
            mode a deep copy is wrapped and converted.
        calibration_states:
            Numpy array of shape ``(N, input_dim)`` with dtype ``float32``.
            **Required** when ``config.mode == "static"``; ignored otherwise.

        Returns
        -------
        quantized_model:
            The quantized module, ready for ``eval()`` inference.
        result:
            :class:`QuantizationResult` with summary statistics.

        Raises
        ------
        ValueError
            If ``mode == "static"`` and *calibration_states* is ``None`` or
            empty.
        """
        model.eval()

        n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        float_bytes = _estimate_tensor_bytes(model.state_dict())

        t0 = time.perf_counter()
        if self.config.mode == "dynamic":
            q_model, cal_samples = self._dynamic(model)
        else:
            if calibration_states is None or len(calibration_states) == 0:
                raise ValueError(
                    "calibration_states must be provided and non-empty for static quantization"
                )
            q_model, cal_samples = self._static(model, calibration_states)
        elapsed = time.perf_counter() - t0
        q_bytes = _estimate_tensor_bytes(q_model.state_dict())

        result = QuantizationResult(
            mode=self.config.mode,
            dtype=self.config.dtype,
            backend=self.config.backend,
            calibration_samples=cal_samples,
            elapsed_seconds=elapsed,
            linear_layers_quantized=n_linear,
            float_param_bytes=float_bytes,
            quantized_param_bytes=q_bytes,
            notes=self._build_notes(),
        )
        logger.info(
            "ptq_complete",
            mode=self.config.mode,
            linear_layers=n_linear,
            calibration_samples=cal_samples,
            elapsed_s=round(elapsed, 3),
        )
        return q_model, result

    def save_checkpoint(
        self,
        quantized_model: nn.Module,
        path: str,
        result: QuantizationResult,
        arch_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a quantized model and its metadata to *path*.

        The model is saved as a full pickle (``torch.save(model, path)``);
        this is required because quantized internal tensor types
        (``PackedParams``) cannot be round-tripped via a plain state-dict.

        A companion JSON file at ``<path>.json`` stores the
        :class:`QuantizationResult` and optional architecture kwargs so the
        checkpoint can be interpreted without running the code.

        Parameters
        ----------
        quantized_model:
            The quantized ``nn.Module`` returned by :meth:`quantize`.
        path:
            Destination file path (e.g. ``"student_A_int8.pt"``).
        result:
            :class:`QuantizationResult` metadata to persist alongside the model.
        arch_kwargs:
            Optional dict of architecture constructor arguments (e.g.
            ``{"input_dim": 8, "output_dim": 4, "parent_hidden_size": 64}``)
            recorded in the JSON for documentation purposes.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(quantized_model, path)
        meta = {
            "quantization": result.to_dict(),
            "arch_kwargs": arch_kwargs or {},
        }
        json_path = path + ".json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("ptq_checkpoint_saved", path=path, json_path=json_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dynamic(self, model: nn.Module) -> Tuple[nn.Module, int]:
        """Apply dynamic (weight-only) quantization."""
        q_model = tq.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=self.config.torch_dtype(),
        )
        return q_model, 0

    def _static(
        self, model: nn.Module, calibration_states: np.ndarray
    ) -> Tuple[nn.Module, int]:
        """Apply static (activation-aware) quantization with calibration."""
        import copy

        previous_engine = torch.backends.quantized.engine
        backend_changed = previous_engine != self.config.backend
        if backend_changed:
            torch.backends.quantized.engine = self.config.backend

        try:
            wrapped = _QuantWrapper(copy.deepcopy(model))
            wrapped.eval()
            wrapped.qconfig = tq.get_default_qconfig(self.config.backend)  # type: ignore[attr-defined]
            tq.prepare(wrapped, inplace=True)

            # Calibration
            n_states = len(calibration_states)
            bs = self.config.calibration_batch_size
            total_samples = 0
            with torch.no_grad():
                for batch_idx in range(self.config.calibration_batches):
                    start = (batch_idx * bs) % n_states
                    end = min(start + bs, n_states)
                    batch = torch.from_numpy(calibration_states[start:end])
                    wrapped(batch)
                    total_samples += end - start

            tq.convert(wrapped, inplace=True)
            return wrapped, total_samples
        finally:
            if backend_changed:
                torch.backends.quantized.engine = previous_engine

    def _build_notes(self) -> List[str]:
        notes = []
        notes.append(
            "LayerNorm layers are not fused and remain in float32 in both dynamic and static modes."
        )
        notes.append(
            "Dropout layers are no-ops in eval() mode and are not quantized."
        )
        if self.config.mode == "dynamic":
            notes.append(
                "Dynamic quantization: weights are int8; activations are computed in float32 "
                "(dequantized on-the-fly). No calibration data required."
            )
        else:
            notes.append(
                "Static quantization: both weights and activations are quantized. "
                "LayerNorm outputs may be requantized depending on backend support."
            )
        notes.append(
            "int8 kernel acceleration is CPU-only; CUDA paths dequantize weights before matmul."
        )
        return notes


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------


def _load_full_model_checkpoint(
    path: str,
    device: Optional[torch.device],
    not_found_msg: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a full-model ``.pt`` checkpoint and optional companion JSON metadata."""
    if not os.path.isfile(path):
        raise FileNotFoundError(not_found_msg)

    if device is None:
        device = torch.device("cpu")

    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()

    json_path = path + ".json"
    metadata: Dict[str, Any] = {}
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    return model, metadata


def _estimate_tensor_bytes(obj: Any) -> int:
    """Estimate bytes used by all tensors nested inside *obj*."""
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(_estimate_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(_estimate_tensor_bytes(v) for v in obj)
    return 0


def load_quantized_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a quantized model checkpoint saved by :meth:`PostTrainingQuantizer.save_checkpoint`.

    .. warning::
        This function calls ``torch.load(path, weights_only=False)`` because
        quantized ``PackedParams`` tensors cannot be loaded with
        ``weights_only=True``.  Only load checkpoints from trusted sources.

    Parameters
    ----------
    path:
        Path to the ``.pt`` file.
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
        not_found_msg=f"Quantized checkpoint not found: {path}",
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_outputs(
    float_model: nn.Module,
    quantized_model: nn.Module,
    states: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Compare Q-value outputs of a float model vs a quantized model.

    Parameters
    ----------
    float_model:
        Float-precision reference model (e.g. ``StudentQNetwork``).
    quantized_model:
        Quantized counterpart produced by :class:`PostTrainingQuantizer`.
    states:
        Numpy array of shape ``(N, input_dim)`` with dtype ``float32``.
    batch_size:
        Number of states to process in each forward pass.

    Returns
    -------
    dict with keys:

    * ``"action_agreement"`` – fraction of states where argmax actions match.
    * ``"mean_q_error"`` – mean absolute difference in Q-values (per action,
      averaged over states).
    * ``"max_q_error"`` – maximum absolute difference over all (state, action)
      pairs.
    * ``"mean_cosine_similarity"`` – mean cosine similarity between float and
      quantized Q-value vectors.
    * ``"n_states"`` – total number of states evaluated.
    """
    float_model.eval()
    quantized_model.eval()

    # Infer the device of the float model so batches are moved there before
    # the float forward pass.  Quantized models with packed int8 params must
    # stay on CPU, so they always receive a CPU tensor.
    try:
        float_device = next(float_model.parameters()).device
    except StopIteration:
        float_device = torch.device("cpu")

    all_float: List[torch.Tensor] = []
    all_quant: List[torch.Tensor] = []

    n = len(states)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            cpu_batch = torch.from_numpy(states[start : start + batch_size])
            f_out = float_model(cpu_batch.to(float_device)).float()
            q_out = quantized_model(cpu_batch)
            if q_out.is_quantized:
                q_out = q_out.dequantize()
            all_float.append(f_out.cpu())
            all_quant.append(q_out.float())

    f = torch.cat(all_float, dim=0)  # (N, output_dim)
    q = torch.cat(all_quant, dim=0)

    action_agreement = (f.argmax(dim=1) == q.argmax(dim=1)).float().mean().item()
    abs_diff = (f - q).abs()
    mean_q_error = abs_diff.mean().item()
    max_q_error = abs_diff.max().item()

    cos_sim = nn.functional.cosine_similarity(f, q, dim=1).mean().item()

    return {
        "action_agreement": action_agreement,
        "mean_q_error": mean_q_error,
        "max_q_error": max_q_error,
        "mean_cosine_similarity": cos_sim,
        "n_states": n,
    }


# ---------------------------------------------------------------------------
# Quantized validation thresholds
# ---------------------------------------------------------------------------


@dataclass
class QuantizedValidationThresholds:
    """Pass/fail thresholds for quantized-model validation.

    Defaults are deliberately more lenient than the float-student thresholds in
    :class:`~farm.core.decision.training.trainer_distill.ValidationThresholds`
    because quantization introduces a bounded amount of numerical degradation.
    Use ``report_only=True`` to disable threshold checks and always pass.

    Attributes
    ----------
    min_action_agreement:
        Minimum required top-1 action agreement between float and quantized
        model (0–1).  Default 0.75.
    max_mean_q_error:
        Maximum allowed mean absolute Q-value error.  Default 0.5.
    min_cosine_similarity:
        Minimum mean cosine similarity between float and quantized Q-value
        vectors.  Default 0.75.
    max_latency_ratio:
        Maximum allowed ``quantized_ms / float_ms`` latency ratio.  Values
        > 1 mean the quantized model is slower (possible on CPU for dynamic
        quantization with small batches).  Default 2.0.
    report_only:
        When ``True`` the :attr:`QuantizedValidationReport.passed` property
        always returns ``True`` regardless of thresholds.  Useful for
        exploratory runs.
    """

    min_action_agreement: float = 0.75
    max_mean_q_error: float = 0.5
    min_cosine_similarity: float = 0.75
    max_latency_ratio: float = 2.0
    report_only: bool = False


# ---------------------------------------------------------------------------
# Quantized validation report
# ---------------------------------------------------------------------------


@dataclass
class QuantizedValidationReport:
    """Comprehensive validation report comparing a quantized model to its float reference.

    Attributes
    ----------
    action_agreement:
        Fraction of states where float and quantized argmax actions match.
    mean_q_error:
        Mean absolute difference in Q-values per (state, action) pair.
    max_q_error:
        Maximum absolute difference over all (state, action) pairs.
    mean_cosine_similarity:
        Mean cosine similarity between float and quantized Q-value vectors.
    n_states:
        Number of states used for fidelity evaluation.
    float_inference_ms:
        Median single-sample inference latency of the float model (ms),
        warmup excluded.
    quantized_inference_ms:
        Median single-sample inference latency of the quantized model (ms),
        warmup excluded.
    latency_ratio:
        ``quantized_inference_ms / float_inference_ms``; < 1 means faster.
    float_checkpoint_bytes:
        On-disk size of the float checkpoint in bytes; ``None`` if no path
        was provided.
    quantized_checkpoint_bytes:
        On-disk size of the quantized checkpoint in bytes; ``None`` if no
        path was provided.
    size_ratio:
        ``quantized_checkpoint_bytes / float_checkpoint_bytes``; ``None`` if
        either size is unavailable.
    pytorch_version:
        ``torch.__version__`` string at validation time.
    quantization_mode:
        Quantization mode string from metadata (e.g. ``"dynamic"``,
        ``"static"``, ``"qat"``).  Falls back to ``"unknown"`` when metadata
        is absent.
    quantization_backend:
        Backend string from metadata (e.g. ``"qnnpack"``).  Falls back to
        ``"unknown"``.
    quantization_dtype:
        Dtype string from metadata (e.g. ``"qint8"``).  Falls back to
        ``"unknown"``.
    compatible:
        ``True`` when the quantized model successfully completed a forward
        pass on the validation states without raising an exception.
    thresholds:
        The :class:`QuantizedValidationThresholds` used for pass/fail.
    """

    action_agreement: float
    mean_q_error: Optional[float]
    max_q_error: Optional[float]
    mean_cosine_similarity: float
    n_states: int
    float_inference_ms: float
    quantized_inference_ms: float
    latency_ratio: float
    float_checkpoint_bytes: Optional[int]
    quantized_checkpoint_bytes: Optional[int]
    size_ratio: Optional[float]
    pytorch_version: str
    quantization_mode: str
    quantization_backend: str
    quantization_dtype: str
    compatible: bool
    thresholds: QuantizedValidationThresholds

    @property
    def passed(self) -> bool:
        """Return ``True`` if all threshold checks pass (or ``report_only`` is set)."""
        if self.thresholds.report_only:
            return True
        if not self.compatible:
            return False
        t = self.thresholds
        mean_q_error = self.mean_q_error if self.mean_q_error is not None else float("inf")
        return (
            self.action_agreement >= t.min_action_agreement
            and mean_q_error <= t.max_mean_q_error
            and self.mean_cosine_similarity >= t.min_cosine_similarity
            and self.latency_ratio <= t.max_latency_ratio
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of all report fields."""
        return {
            "fidelity": {
                "action_agreement": self.action_agreement,
                "mean_q_error": self.mean_q_error,
                "max_q_error": self.max_q_error,
                "mean_cosine_similarity": self.mean_cosine_similarity,
                "n_states": self.n_states,
            },
            "latency": {
                "float_inference_ms": self.float_inference_ms,
                "quantized_inference_ms": self.quantized_inference_ms,
                "latency_ratio": self.latency_ratio,
            },
            "size": {
                "float_checkpoint_bytes": self.float_checkpoint_bytes,
                "quantized_checkpoint_bytes": self.quantized_checkpoint_bytes,
                "size_ratio": self.size_ratio,
            },
            "compatibility": {
                "compatible": self.compatible,
                "pytorch_version": self.pytorch_version,
                "quantization_mode": self.quantization_mode,
                "quantization_backend": self.quantization_backend,
                "quantization_dtype": self.quantization_dtype,
            },
            "thresholds": {
                "min_action_agreement": self.thresholds.min_action_agreement,
                "max_mean_q_error": self.thresholds.max_mean_q_error,
                "min_cosine_similarity": self.thresholds.min_cosine_similarity,
                "max_latency_ratio": self.thresholds.max_latency_ratio,
                "report_only": self.thresholds.report_only,
            },
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Quantized validator
# ---------------------------------------------------------------------------


class QuantizedValidator:
    """Validate a quantized model against its float reference on fidelity, speed, and size.

    This validator covers all acceptance criteria for a quantized student
    checkpoint:

    * **Compatibility**: the quantized model is loaded and a forward pass runs
      without error on the validation state set.
    * **Fidelity**: action agreement, mean/max Q-error, and cosine similarity
      vs the float reference (via :func:`compare_outputs`).
    * **Latency**: per-sample median forward time with a configurable warmup
      phase excluded (``quantized_ms / float_ms``).
    * **File size**: on-disk checkpoint sizes if paths are supplied
      (``quantized_bytes / float_bytes``).
    * **Compatibility metadata**: PyTorch version, quantization mode, backend,
      and dtype recorded in the report.

    Parameters
    ----------
    float_model:
        Float-precision reference ``nn.Module`` (e.g. ``StudentQNetwork``).
    quantized_model:
        Quantized ``nn.Module`` produced by :class:`PostTrainingQuantizer` or
        :class:`~farm.core.decision.training.quantize_qat.QATTrainer`.
    thresholds:
        Optional :class:`QuantizedValidationThresholds`.  Defaults to
        conservative quantization-aware values.
    device:
        Target device.  Defaults to CPU.

    Example
    -------
    ::

        from farm.core.decision.training.quantize_ptq import (
            QuantizedValidator,
            QuantizedValidationThresholds,
            load_quantized_checkpoint,
        )

        q_model, meta = load_quantized_checkpoint("student_A_int8.pt")
        validator = QuantizedValidator(float_model, q_model)
        report = validator.validate(
            states,
            float_checkpoint_path="student_A.pt",
            quantized_checkpoint_path="student_A_int8.pt",
            quantization_metadata=meta,
        )
        print(report.passed, report.to_dict())
    """

    def __init__(
        self,
        float_model: nn.Module,
        quantized_model: nn.Module,
        thresholds: Optional[QuantizedValidationThresholds] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.float_model = float_model
        self.quantized_model = quantized_model
        self.thresholds = thresholds or QuantizedValidationThresholds()
        self.device = device or torch.device("cpu")
        self.float_model.to(self.device)
        # Quantized models with int8 packed params must stay on CPU
        # (CUDA does not support packed int8 kernels); we move them only if
        # the device is CPU to avoid a runtime error.
        if self.device.type == "cpu":
            try:
                self.quantized_model.to(self.device)
            except Exception:
                pass

    def validate(
        self,
        states: np.ndarray,
        float_checkpoint_path: Optional[str] = None,
        quantized_checkpoint_path: Optional[str] = None,
        quantization_metadata: Optional[Dict[str, Any]] = None,
        n_latency_warmup: int = 5,
        n_latency_repeats: int = 50,
        batch_size: int = 256,
    ) -> QuantizedValidationReport:
        """Run all validation checks and return a :class:`QuantizedValidationReport`.

        Parameters
        ----------
        states:
            NumPy array of shape ``(N, input_dim)`` with ``dtype=float32``.
        float_checkpoint_path:
            Optional path to the float ``.pt`` checkpoint for size comparison.
        quantized_checkpoint_path:
            Optional path to the quantized ``.pt`` checkpoint for size
            comparison.
        quantization_metadata:
            Optional dict from :func:`load_quantized_checkpoint` (the
            companion ``.json`` file).  Used to populate compatibility fields.
        n_latency_warmup:
            Forward passes to run before timing (excluded from measurements).
        n_latency_repeats:
            Number of timed single-sample forward passes; median is reported.
        batch_size:
            Batch size for the :func:`compare_outputs` forward passes.

        Returns
        -------
        QuantizedValidationReport
        """
        if len(states) == 0:
            raise ValueError("states must be non-empty; got 0 samples.")
        states_arr = np.asarray(states, dtype=np.float32)
        if states_arr.ndim != 2:
            raise ValueError(
                f"states must be a 2D array with shape (N, input_dim); got {states_arr.shape!r}"
            )

        self.float_model.eval()
        self.quantized_model.eval()

        # -- Compatibility check --
        compatible, compat_error = self._check_compatibility(states_arr)

        # -- Fidelity metrics --
        # Skip compare_outputs when incompatible to avoid a second error.
        if compatible:
            cmp = compare_outputs(
                self.float_model, self.quantized_model, states_arr, batch_size=batch_size
            )
        else:
            n = len(states_arr)
            cmp = {
                "action_agreement": 0.0,
                "mean_q_error": None,
                "max_q_error": None,
                "mean_cosine_similarity": 0.0,
                "n_states": n,
            }

        # -- Latency --
        float_ms, quant_ms = self._measure_latency(
            states_arr, n_latency_warmup, n_latency_repeats, skip_quantized=not compatible
        )
        latency_ratio = quant_ms / max(float_ms, 1e-9)

        # -- File sizes --
        float_bytes: Optional[int] = None
        quant_bytes: Optional[int] = None
        size_ratio: Optional[float] = None
        if float_checkpoint_path and os.path.isfile(float_checkpoint_path):
            float_bytes = os.path.getsize(float_checkpoint_path)
        if quantized_checkpoint_path and os.path.isfile(quantized_checkpoint_path):
            quant_bytes = os.path.getsize(quantized_checkpoint_path)
        if float_bytes is not None and quant_bytes is not None and float_bytes > 0:
            size_ratio = quant_bytes / float_bytes

        # -- Compatibility metadata --
        quant_meta = (quantization_metadata or {}).get("quantization", {})
        quant_mode = quant_meta.get("mode", "unknown")
        quant_backend = quant_meta.get("backend", "unknown")
        quant_dtype = quant_meta.get("dtype", "unknown")

        report = QuantizedValidationReport(
            action_agreement=cmp["action_agreement"],
            mean_q_error=cmp["mean_q_error"],
            max_q_error=cmp["max_q_error"],
            mean_cosine_similarity=cmp["mean_cosine_similarity"],
            n_states=cmp["n_states"],
            float_inference_ms=float_ms,
            quantized_inference_ms=quant_ms,
            latency_ratio=latency_ratio,
            float_checkpoint_bytes=float_bytes,
            quantized_checkpoint_bytes=quant_bytes,
            size_ratio=size_ratio,
            pytorch_version=torch.__version__,
            quantization_mode=quant_mode,
            quantization_backend=quant_backend,
            quantization_dtype=quant_dtype,
            compatible=compatible,
            thresholds=self.thresholds,
        )

        if not compatible:
            logger.warning(
                "quantized_validator_compat_failure",
                error=str(compat_error),
            )

        mean_q_err_log = (
            round(cmp["mean_q_error"], 6) if cmp["mean_q_error"] is not None else None
        )
        logger.info(
            "quantized_validator_complete",
            compatible=compatible,
            action_agreement=round(cmp["action_agreement"], 4),
            mean_q_error=mean_q_err_log,
            latency_ratio=round(latency_ratio, 4),
            passed=report.passed,
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_compatibility(self, states: np.ndarray) -> Tuple[bool, Optional[Exception]]:
        """Run a test forward pass to verify the quantized model is functional.

        Returns
        -------
        (compatible, error)
            ``compatible`` is ``True`` when the forward pass succeeds.
            ``error`` is the caught exception or ``None`` on success.
        """
        # Quantized models with packed int8 params are CPU-only; always probe
        # on CPU regardless of self.device to avoid a spurious device-mismatch
        # error that would mask a genuinely incompatible model.
        cpu_probe = torch.from_numpy(states[:1])
        try:
            with torch.no_grad():
                out = self.quantized_model(cpu_probe)
            if out.is_quantized:
                out = out.dequantize()
            if not torch.isfinite(out).all():
                return False, ValueError("Quantized model produced non-finite outputs.")
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, exc

    def _measure_latency(
        self,
        states: np.ndarray,
        n_warmup: int,
        n_repeats: int,
        skip_quantized: bool = False,
    ) -> Tuple[float, float]:
        """Return median single-sample latency (ms) for float and quantized models.

        Uses the first row of *states* as the probe input so that batch-size
        variation does not confound the comparison.  Warmup passes are excluded
        from timing.  When *skip_quantized* is ``True`` (e.g. the compatibility
        check failed), the quantized latency is returned as ``0.0`` to avoid a
        second error.
        """
        # Float model runs on self.device; quantized model must stay on CPU.
        float_single = torch.from_numpy(states[:1]).to(self.device)
        quant_single = torch.from_numpy(states[:1])  # always CPU for int8 kernels

        def _sync() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        def _time_model(model: nn.Module, inp: torch.Tensor) -> float:
            model.eval()
            with torch.no_grad():
                for _ in range(n_warmup):
                    model(inp)
            _sync()
            times: List[float] = []
            with torch.no_grad():
                for _ in range(n_repeats):
                    _sync()
                    t0 = time.perf_counter()
                    model(inp)
                    _sync()
                    times.append((time.perf_counter() - t0) * 1_000.0)
            return float(np.median(times)) if times else 0.0

        float_ms = _time_model(self.float_model, float_single)
        quant_ms = 0.0 if skip_quantized else _time_model(self.quantized_model, quant_single)
        return float_ms, quant_ms
