"""Crossover operators for quantized Q-network state dicts.

This module implements crossover operators that combine two
``StudentQNetwork`` / ``BaseQNetwork``-compatible state dicts (float or
quantized) into an offspring state dict. Crossover enables evolutionary or
multi-parent search strategies that mix policies **after** PTQ / QAT rather
than only mixing float weights.

Scope
-----
Operates on **paired** state dicts with **identical keys and tensor shapes**.
Offspring are always returned as **float32** state dicts, so they can be
loaded into the model class and optionally re-quantized with any PTQ / QAT
pipeline.

Quantization nuance
-------------------
Quantized tensors (e.g. ``torch.qint8`` from dynamic PTQ) are **dequantized
to float32 before crossover**.  This guarantees numerical correctness for all
three modes and avoids silent int8 arithmetic errors.  When parents carry
per-tensor scales those scales are consumed during dequantization, so the
offspring float tensor can be re-quantized later with a fresh scale policy
(e.g. ``max-abs / 127``).

Crossover modes
---------------
random
    For each parameter tensor independently flip a (possibly biased) coin
    and take the tensor from parent A or parent B.  **Deterministic** given
    the RNG seed.

layer
    Group parameters by ``nn.Sequential`` submodule (``network.0.*``,
    ``network.1.*``, …), merge **logical blocks** (Linear + following LayerNorm
    share one parent; final Linear is its own block), then alternate blocks
    between parents: even blocks from A, odd from B.

weighted
    For each aligned parameter tensor compute
    ``child = alpha * tensor_a + (1 - alpha) * tensor_b`` in float32.
    ``alpha = 1.0`` reproduces parent A; ``alpha = 0.0`` reproduces parent B.

API
---
::

    from farm.core.decision.training.crossover import crossover_quantized_state_dict

    child_sd = crossover_quantized_state_dict(
        state_dict_a,
        state_dict_b,
        mode="random",   # "random" | "layer" | "weighted"
        seed=42,
    )

    # Load into a fresh model
    model = StudentQNetwork(input_dim=8, output_dim=4, parent_hidden_size=64)
    model.load_state_dict(child_sd)

A thin file-level wrapper :func:`crossover_checkpoints` loads two ``.pt``
files, runs crossover, and saves the offspring.

:func:`initialize_child_from_crossover` is the high-level entry point: it
resolves parents (including PTQ ``*.pt`` + ``*.json`` sidecars via
:func:`~farm.core.decision.training.quantize_ptq.load_quantized_checkpoint`),
optionally infers architecture, builds a :class:`~farm.core.decision.base_dqn.BaseQNetwork`
or :class:`~farm.core.decision.base_dqn.StudentQNetwork`, and loads the
crossed float weights.  Use :class:`ChildArchitectureSpec` to override
inferred shapes.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

from farm.core.decision.training.quantize_ptq import load_quantized_checkpoint
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

CROSSOVER_MODES = ("random", "layer", "weighted")


@dataclass
class MutationConfig:
    """Configuration for Gaussian weight-noise mutation.

    Mutation adds small random noise to the parameters of a state dict,
    providing diversity and preventing evolutionary stagnation when crossover
    alone is insufficient.

    Attributes
    ----------
    noise_std:
        Standard deviation of the zero-mean Gaussian noise added to each
        selected parameter element.  ``0.0`` is a no-op (noise is still
        generated but has zero magnitude).  Typical values: ``0.001``–``0.05``.
    noise_fraction:
        Fraction of **elements** in each tensor that receive noise.
        ``1.0`` (default) mutates all weights uniformly.  Values in ``(0, 1)``
        apply a random binary mask so only a fraction of elements are perturbed.
        Must be in ``(0.0, 1.0]``.
    seed:
        Integer RNG seed for reproducibility.  ``None`` → non-reproducible.
    """

    noise_std: float = 0.01
    noise_fraction: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.noise_std < 0.0:
            raise ValueError(
                f"MutationConfig.noise_std must be >= 0.0; got {self.noise_std!r}"
            )
        if not 0.0 < self.noise_fraction <= 1.0:
            raise ValueError(
                f"MutationConfig.noise_fraction must be in (0.0, 1.0]; "
                f"got {self.noise_fraction!r}"
            )


def mutate_state_dict(
    state_dict: Dict[str, Any],
    config: MutationConfig,
) -> Dict[str, Any]:
    """Apply Gaussian weight-noise mutation to a float32 state dict.

    Returns a **new** dict; the input *state_dict* is not modified.
    Non-tensor entries (e.g. integer scalars) are deep-copied unchanged.

    Parameters
    ----------
    state_dict:
        Source state dict (float32 tensors; quantized tensors are
        dequantized to float32 before noise is added).
    config:
        :class:`MutationConfig` controlling noise scale and coverage.

    Returns
    -------
    Dict[str, Any]
        Mutated float32 state dict.

    Examples
    --------
    ::

        from farm.core.decision.training.crossover import (
            MutationConfig, mutate_state_dict
        )

        cfg = MutationConfig(noise_std=0.01, seed=42)
        mutated_sd = mutate_state_dict(model.state_dict(), cfg)
        model.load_state_dict(mutated_sd)
    """
    rng = np.random.default_rng(config.seed)
    mutated: Dict[str, Any] = {}
    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            mutated[key] = copy.deepcopy(val)
            continue
        t = _to_float(val)  # dequantize + cast to float32
        if config.noise_std == 0.0:
            mutated[key] = t
            continue
        noise = torch.from_numpy(
            rng.standard_normal(t.shape).astype("float32")
        ) * config.noise_std
        if config.noise_fraction < 1.0:
            mask = torch.from_numpy(
                (rng.random(t.shape) < config.noise_fraction).astype("float32")
            )
            noise = noise * mask
        mutated[key] = t + noise
    logger.info(
        "mutate_state_dict",
        n_params=len(state_dict),
        noise_std=config.noise_std,
        noise_fraction=config.noise_fraction,
    )
    return mutated


@dataclass
class ChildArchitectureSpec:
    """Explicit Q-network shape for :func:`initialize_child_from_crossover`.

    When provided, skips shape inference from the parent state dict (keys must
    still match the implied layout: ``network.0`` … ``network.8``).

    Attributes
    ----------
    input_dim, output_dim:
        State and action dimensions.
    hidden_size:
        Hidden width of the child: ``BaseQNetwork.hidden_size``, or the
        **student** hidden (``max(16, parent_hidden_size // 2)``) when the
        child class is :class:`~farm.core.decision.base_dqn.StudentQNetwork`.
    parent_hidden_size:
        Constructor argument for :class:`~farm.core.decision.base_dqn.StudentQNetwork`.
        When ``None``, a default is inferred from *hidden_size* (see module notes).
    """

    input_dim: int
    output_dim: int
    hidden_size: int
    parent_hidden_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float(tensor: torch.Tensor) -> torch.Tensor:
    """Return *tensor* as a ``float32`` CPU tensor.

    If *tensor* is already ``float32`` (or other float dtype) it is returned
    as-is (moved to CPU if necessary).  Quantized tensors are dequantized
    first.
    """
    if tensor.is_quantized:
        tensor = tensor.dequantize()
    return tensor.cpu().float()


def _float_state_dict_from_dynamic_quantized_module(model: nn.Module) -> Dict[str, Any]:
    """Extract a float ``state_dict`` matching ``BaseQNetwork`` parameter keys.

    Dynamic PTQ stores ``Linear`` layers as ``DynamicQuantizedLinear``; their
    :meth:`~torch.nn.Module.state_dict` uses packed weights that
    :func:`crossover_quantized_state_dict` cannot blend.  This walks the
    ``network`` :class:`~torch.nn.Sequential` and reads dequantized weights /
    biases via the layers' public API.
    """
    out: Dict[str, Any] = {}
    net = getattr(model, "network", None)
    if not isinstance(net, nn.Sequential):
        raise TypeError(
            "Quantized parent must have a Sequential `network` attribute "
            f"(got {type(net).__name__!r})"
        )
    qdynamic = getattr(torch.nn.quantized, "dynamic", None)
    qlinear_type = getattr(qdynamic, "Linear", None) if qdynamic is not None else None

    for i, layer in enumerate(net):
        prefix = f"network.{i}"
        lname = type(layer).__name__
        if isinstance(layer, nn.Linear):
            out[f"{prefix}.weight"] = layer.weight.detach().cpu().float()
            out[f"{prefix}.bias"] = layer.bias.detach().cpu().float()
        elif qlinear_type is not None and isinstance(layer, qlinear_type):
            w_t = layer.weight()
            b_t = layer.bias()
            if w_t.is_quantized:
                w_t = w_t.dequantize()
            if b_t.is_quantized:
                b_t = b_t.dequantize()
            out[f"{prefix}.weight"] = w_t.detach().cpu().float()
            out[f"{prefix}.bias"] = b_t.detach().cpu().float()
        elif "DynamicQuantizedLinear" in lname:
            w_fn = getattr(layer, "weight", None)
            b_fn = getattr(layer, "bias", None)
            if not callable(w_fn) or not callable(b_fn):
                raise TypeError(f"Unsupported quantized layer at {prefix}: {lname!r}")
            w_t = w_fn()
            b_t = b_fn()
            if w_t.is_quantized:
                w_t = w_t.dequantize()
            if b_t.is_quantized:
                b_t = b_t.dequantize()
            out[f"{prefix}.weight"] = w_t.detach().cpu().float()
            out[f"{prefix}.bias"] = b_t.detach().cpu().float()
        elif isinstance(layer, nn.LayerNorm):
            out[f"{prefix}.weight"] = layer.weight.detach().cpu().float()
            out[f"{prefix}.bias"] = layer.bias.detach().cpu().float()
    return out


def _validate_state_dicts(
    sd_a: Dict[str, Any],
    sd_b: Dict[str, Any],
) -> List[str]:
    """Validate that *sd_a* and *sd_b* are compatible for crossover.

    Returns the sorted list of keys shared by both dicts.

    Raises
    ------
    ValueError
        If the key sets differ or if any tensor pair has mismatched shapes.
    """
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        raise ValueError(
            f"State dict key mismatch.\n"
            f"  Only in A: {only_a}\n"
            f"  Only in B: {only_b}"
        )

    keys = sorted(keys_a)
    shape_errors: List[str] = []
    for k in keys:
        va = sd_a[k]
        vb = sd_b[k]
        a_is_tensor = isinstance(va, torch.Tensor)
        b_is_tensor = isinstance(vb, torch.Tensor)
        # Enforce type parity: both must be tensors or both must be non-tensors.
        if a_is_tensor != b_is_tensor:
            a_type = "Tensor" if a_is_tensor else type(va).__name__
            b_type = "Tensor" if b_is_tensor else type(vb).__name__
            raise ValueError(
                f"Type mismatch at key '{k}': "
                f"A is {a_type}, B is {b_type}. "
                "Both entries must be tensors or both must be non-tensors."
            )
        if not a_is_tensor:
            # Both are non-tensors – no shape to check.
            continue
        # Compare shapes directly to avoid unnecessary float conversion.
        if va.shape != vb.shape:
            shape_errors.append(f"  '{k}': A={tuple(va.shape)}, B={tuple(vb.shape)}")

    if shape_errors:
        raise ValueError(
            "Shape mismatch for the following parameters:\n" + "\n".join(shape_errors)
        )

    return keys


def _layer_groups(keys: List[str]) -> Dict[str, List[str]]:
    """Partition *keys* into groups by their first two name components.

    For the standard ``BaseQNetwork`` / ``StudentQNetwork`` architecture the
    keys look like ``"network.0.weight"``.  Parameters that share the first
    two dot-separated segments (e.g. ``"network.0"``) are assigned to the
    same group.

    Non-tensor keys (e.g. plain scalars) are kept in their own singleton
    groups under their full name.

    Returns
    -------
    OrderedDict-like ``{group_name: [key, ...]}`` preserving the order of
    first appearance.
    """
    groups: Dict[str, List[str]] = {}
    for k in keys:
        parts = k.split(".")
        # Use up to the first two components as the group key.
        group = ".".join(parts[:2]) if len(parts) >= 2 else k
        groups.setdefault(group, []).append(k)
    return groups


def _layer_group_block_id(group: str, fallback_idx: int) -> int:
    """Map a ``network.N``-style group name to a logical block index for layer crossover.

    :class:`~farm.core.decision.base_dqn.BaseQNetwork` uses
    ``Linear, LayerNorm, ReLU, Dropout`` per hidden stage, so learned parameters
    sit at submodule indices ``0, 1`` (first stage), ``4, 5`` (second), and
    ``8`` (output Linear).  Those indices must share the same parent assignment
    so LayerNorm scaling stays consistent with the preceding Linear.

    For groups that do not match ``network.<int>``, or unknown indices, *fallback_idx*
    preserves the previous per-group enumeration behavior.
    """
    parts = group.split(".")
    if len(parts) >= 2:
        try:
            mod_idx = int(parts[1])
        except ValueError:
            return fallback_idx
        if mod_idx in (0, 1):
            return 0
        if mod_idx in (4, 5):
            return 1
        if mod_idx == 8:
            return 2
    return fallback_idx


# ---------------------------------------------------------------------------
# Crossover mode implementations
# ---------------------------------------------------------------------------


def _random_crossover(
    sd_a: Dict[str, Any],
    sd_b: Dict[str, Any],
    keys: List[str],
    rng: np.random.Generator,
    alpha: float,
) -> Dict[str, Any]:
    """Uniform per-tensor random crossover.

    Each parameter key is independently assigned to parent A with probability
    *alpha* (default 0.5) or parent B with probability ``1 - alpha``.

    Parameters
    ----------
    sd_a, sd_b:
        Source state dicts (already validated).
    keys:
        Sorted list of keys to process.
    rng:
        Numpy random generator (deterministic given its seed).
    alpha:
        Probability of selecting from parent A.  ``alpha=1.0`` reproduces A;
        ``alpha=0.0`` reproduces B.

    Returns
    -------
    Dict[str, Any]
        Float32 offspring state dict.  Tensor entries are ``float32``; any
        non-tensor entries (e.g. integer scalars) are deep-copied from
        parent A.
    """
    child: Dict[str, Any] = {}
    choices = rng.random(len(keys))  # uniform [0, 1) for each key
    for i, k in enumerate(keys):
        va = sd_a[k]
        vb = sd_b[k]
        if isinstance(va, torch.Tensor):
            child[k] = _to_float(va) if choices[i] < alpha else _to_float(vb)
        else:
            # Non-tensor entries (e.g. int scalars): take from A by default
            child[k] = copy.deepcopy(va)
    return child


def _layer_crossover(
    sd_a: Dict[str, Any],
    sd_b: Dict[str, Any],
    keys: List[str],
) -> Dict[str, Any]:
    """Layer-group-based crossover.

    Parameters are grouped by submodule prefix (``network.N``).  Groups that
    belong to the same logical block (Linear + LayerNorm for each hidden stage,
    and the output Linear on :class:`~farm.core.decision.base_dqn.BaseQNetwork`)
    are assigned the same parent.  Even-indexed blocks come from parent A;
    odd-indexed blocks come from parent B, avoiding inconsistent feature scaling
    between a Linear and its following LayerNorm.

    Parameters
    ----------
    sd_a, sd_b:
        Source state dicts (already validated).
    keys:
        Sorted list of keys to process.

    Returns
    -------
    Dict[str, Any]
        Float32 offspring state dict.  Tensor entries are ``float32``; any
        non-tensor entries (e.g. integer scalars) are deep-copied from the
        selected parent.
    """
    groups = _layer_groups(keys)
    child: Dict[str, Any] = {}
    for fallback_idx, (group_name, group_keys) in enumerate(groups.items()):
        block_id = _layer_group_block_id(group_name, fallback_idx)
        parent = sd_a if block_id % 2 == 0 else sd_b
        for k in group_keys:
            v = parent[k]
            if isinstance(v, torch.Tensor):
                child[k] = _to_float(v)
            else:
                child[k] = copy.deepcopy(v)
    return child


def _weighted_crossover(
    sd_a: Dict[str, Any],
    sd_b: Dict[str, Any],
    keys: List[str],
    alpha: float,
) -> Dict[str, Any]:
    """Weighted-average crossover.

    For each tensor parameter:
    ``child = alpha * parent_a + (1 - alpha) * parent_b``

    Both parents are dequantized to float32 before blending.  The resulting
    offspring is always float32 and can be re-quantized via any downstream
    PTQ / QAT pipeline.

    Edge cases:
    * ``alpha = 1.0``: child is an exact copy of parent A.
    * ``alpha = 0.0``: child is an exact copy of parent B.

    Parameters
    ----------
    sd_a, sd_b:
        Source state dicts (already validated).
    keys:
        Sorted list of keys to process.
    alpha:
        Blend coefficient for parent A.

    Returns
    -------
    Dict[str, Any]
        Float32 offspring state dict.  Tensor entries are ``float32``; any
        non-tensor entries (e.g. integer scalars) are deep-copied from
        parent A.
    """
    child: Dict[str, Any] = {}
    for k in keys:
        va = sd_a[k]
        vb = sd_b[k]
        if isinstance(va, torch.Tensor):
            fa = _to_float(va)
            fb = _to_float(vb)
            child[k] = alpha * fa + (1.0 - alpha) * fb
        else:
            # Non-tensor entries: take from A by default
            child[k] = copy.deepcopy(va)
    return child


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def crossover_quantized_state_dict(
    state_dict_a: Dict[str, Any],
    state_dict_b: Dict[str, Any],
    mode: str = "random",
    rng: Optional[np.random.Generator] = None,
    alpha: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Combine two Q-network state dicts into a single offspring state dict.

    Both parents must have **identical keys and tensor shapes**.  Quantized
    tensors (``torch.qint8`` or other integer dtypes) are dequantized to
    ``float32`` before crossover.  Tensor entries in the offspring are returned
    as ``float32`` values; any non-tensor entries (e.g. integer scalars) are
    preserved and copied through unchanged.

    Parameters
    ----------
    state_dict_a:
        First parent's state dict (float or quantized).
    state_dict_b:
        Second parent's state dict (float or quantized).
    mode:
        Crossover strategy.  One of:

        ``"random"``
            Each parameter tensor is independently drawn from parent A with
            probability *alpha* or from parent B with probability
            ``1 - alpha``.  Deterministic given *rng* / *seed*.

        ``"layer"``
            Parameters are grouped by submodule, merged into logical blocks
            (each hidden Linear + its LayerNorm, then the output Linear for
            standard Q-networks).  Even-indexed blocks come from parent A,
            odd-indexed from parent B.

        ``"weighted"``
            Child tensor = ``alpha * parent_a + (1 - alpha) * parent_b``
            computed in float32.  ``alpha=1.0`` → copy of A; ``alpha=0.0``
            → copy of B.

    rng:
        Numpy ``np.random.Generator`` used by the ``"random"`` mode.  When
        ``None`` and *seed* is also ``None`` a non-reproducible RNG is
        created.  Ignored for ``"layer"`` and ``"weighted"`` modes.
    alpha:
        In ``"random"`` mode: probability of selecting from parent A per
        tensor.  In ``"weighted"`` mode: blend coefficient for parent A.
        Must be in ``[0.0, 1.0]``.  Ignored for ``"layer"`` mode.
    seed:
        Integer seed for the ``"random"`` mode RNG.  Takes precedence over a
        provided *rng* object only when *rng* is ``None``.  Ignored for other
        modes.

    Returns
    -------
    Dict[str, Any]
        Offspring state dict with float32 tensor entries; any non-tensor
        entries from the parents are deep-copied and passed through unchanged.

    Raises
    ------
    ValueError
        If *mode* is not one of the supported values, if *alpha* is outside
        ``[0, 1]``, or if the state dicts have mismatched keys / tensor
        shapes.

    Examples
    --------
    ::

        from farm.core.decision.base_dqn import StudentQNetwork
        from farm.core.decision.training.crossover import (
            crossover_quantized_state_dict,
        )

        # Float state dicts
        child_sd = crossover_quantized_state_dict(
            model_a.state_dict(),
            model_b.state_dict(),
            mode="random",
            seed=42,
        )
        model_c = StudentQNetwork(input_dim=8, output_dim=4, parent_hidden_size=64)
        model_c.load_state_dict(child_sd)
    """
    if mode not in CROSSOVER_MODES:
        raise ValueError(
            f"mode must be one of {CROSSOVER_MODES!r}, got {mode!r}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha!r}")

    keys = _validate_state_dicts(state_dict_a, state_dict_b)

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng(seed)
        child = _random_crossover(state_dict_a, state_dict_b, keys, rng, alpha)
    elif mode == "layer":
        child = _layer_crossover(state_dict_a, state_dict_b, keys)
    else:  # "weighted"
        child = _weighted_crossover(state_dict_a, state_dict_b, keys, alpha)

    logger.info(
        "crossover_complete",
        mode=mode,
        n_params=len(keys),
        alpha=alpha,
    )
    return child


def crossover_checkpoints(
    path_a: str,
    path_b: str,
    output_path: str,
    mode: str = "random",
    alpha: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Load two float state-dict checkpoints, crossover, and save offspring.

    .. note::
        This helper is designed for **float** state-dict checkpoints (e.g.
        QAT float checkpoints saved via
        :meth:`~farm.core.decision.training.quantize_qat.QATTrainer._save_qat_checkpoint`
        or a plain ``torch.save(model.state_dict(), path)``).  Full-model
        pickles (e.g. from
        :meth:`~farm.core.decision.training.quantize_ptq.PostTrainingQuantizer.save_checkpoint`)
        should be loaded with ``torch.load(path, weights_only=False)`` and
        their state dicts extracted manually before passing to
        :func:`crossover_quantized_state_dict`.

    Parameters
    ----------
    path_a:
        Path to the first parent checkpoint (``.pt`` state dict).
    path_b:
        Path to the second parent checkpoint (``.pt`` state dict).
    output_path:
        Destination path for the offspring state dict.
    mode:
        Crossover mode (``"random"``, ``"layer"``, or ``"weighted"``).
    alpha:
        Blend / selection coefficient; see :func:`crossover_quantized_state_dict`.
    seed:
        Integer RNG seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        The offspring state dict (also saved to *output_path*).  Tensor
        entries are float32; any non-tensor entries are deep-copied from the
        winning parent unchanged.
    """
    sd_a = torch.load(path_a, map_location="cpu", weights_only=True)
    sd_b = torch.load(path_b, map_location="cpu", weights_only=True)

    if not isinstance(sd_a, dict):
        raise ValueError(
            f"Checkpoint at {path_a!r} must contain a state dict, got "
            f"{type(sd_a).__name__}. Expected output from "
            "torch.save(model.state_dict(), ...) or the QAT float checkpoint saver."
        )
    if not isinstance(sd_b, dict):
        raise ValueError(
            f"Checkpoint at {path_b!r} must contain a state dict, got "
            f"{type(sd_b).__name__}. Expected output from "
            "torch.save(model.state_dict(), ...) or the QAT float checkpoint saver."
        )

    child = crossover_quantized_state_dict(
        sd_a,
        sd_b,
        mode=mode,
        alpha=alpha,
        seed=seed,
    )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(child, output_path)
    logger.info("crossover_checkpoint_saved", path=output_path, mode=mode)
    return child


# ---------------------------------------------------------------------------
# Architecture inference helpers
# ---------------------------------------------------------------------------

#: Type alias for a parent specification: an nn.Module, a filesystem path, or
#: a pre-loaded state dict.
ParentSpec = Union[nn.Module, Path, str, Dict[str, Any]]


def _read_arch_spec_from_ptq_json(pt_path: str) -> Optional[ChildArchitectureSpec]:
    """Build a :class:`ChildArchitectureSpec` from ``<pt_path>.json`` arch_kwargs.

    Used when a dynamically quantized checkpoint's :meth:`state_dict` no longer
    exposes plain ``network.*.weight`` keys, so tensor-based inference fails.
    """
    json_path = pt_path + ".json"
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    arch = data.get("arch_kwargs")
    if not isinstance(arch, dict):
        return None
    idim = arch.get("input_dim")
    odim = arch.get("output_dim")
    if idim is None or odim is None:
        return None
    idim_i, odim_i = int(idim), int(odim)
    if "hidden_size" in arch:
        return ChildArchitectureSpec(
            input_dim=idim_i,
            output_dim=odim_i,
            hidden_size=int(arch["hidden_size"]),
            parent_hidden_size=None,
        )
    if "parent_hidden_size" in arch:
        ph = int(arch["parent_hidden_size"])
        student_h = max(16, ph // 2)
        return ChildArchitectureSpec(
            input_dim=idim_i,
            output_dim=odim_i,
            hidden_size=student_h,
            parent_hidden_size=ph,
        )
    return None


def _looks_like_ptq_sidecar(json_path: str) -> bool:
    """Return True if *json_path* looks like **dynamic** PTQ metadata."""
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    q = data.get("quantization")
    if not isinstance(q, dict):
        return False
    mode = q.get("mode")
    dtype = q.get("dtype")
    # Crossover currently supports dynamic PTQ checkpoint auto-loading only.
    return mode == "dynamic" and dtype == "qint8"


def _resolve_parent(
    parent: ParentSpec,
    *,
    allow_unsafe_unpickle: bool = False,
    auto_load_ptq_checkpoints: bool = True,
) -> Dict[str, Any]:
    """Resolve *parent* to a state dict for use with :func:`crossover_quantized_state_dict`.

    The returned dict may contain float or quantized tensors; conversion to
    float32 happens later inside the crossover helpers via :func:`_to_float`.

    Accepts:
    * An ``nn.Module`` – its :meth:`~nn.Module.state_dict` is returned as-is
      (potentially on any device / dtype).
    * A ``str`` or ``pathlib.Path`` pointing to a ``.pt`` file.  If a sibling
      ``<path>.json`` looks like PTQ metadata from
      :meth:`~farm.core.decision.training.quantize_ptq.PostTrainingQuantizer.save_checkpoint`,
      the file is loaded via :func:`~farm.core.decision.training.quantize_ptq.load_quantized_checkpoint`
      (full-model unpickle; trusted checkpoints only).  Otherwise plain
      state-dict files use ``weights_only=True`` first, with optional unsafe
      fallback — see the warning below.  If the loaded object is an
      ``nn.Module``, its state dict is extracted.
    * A ``dict`` (state dict) – returned as-is without copying.

    .. warning::
        Full-model pickle loading (``weights_only=False``) can execute arbitrary
        code during unpickling and is therefore **disabled by default**.
        To permit this fallback for trusted checkpoints only, pass
        ``allow_unsafe_unpickle=True``.

    Parameters
    ----------
    parent:
        Parent specification.
    allow_unsafe_unpickle:
        When ``True``, allows fallback from ``weights_only=True`` to
        ``weights_only=False`` for full-model pickle checkpoints, including
        PTQ ``*.pt`` files loaded via :func:`load_quantized_checkpoint`.
    auto_load_ptq_checkpoints:
        When ``True`` (default), detect PTQ sidecar JSON next to ``*.pt`` paths
        and load via :func:`load_quantized_checkpoint`. This still requires
        ``allow_unsafe_unpickle=True`` because quantized full-model loading uses
        ``weights_only=False``.

    Returns
    -------
    Dict[str, Any]
        State dict (float or quantized) suitable for
        :func:`crossover_quantized_state_dict`.

    Raises
    ------
    TypeError
        If *parent* is not one of the accepted types.
    FileNotFoundError
        If *parent* is a path that does not exist.
    """
    if isinstance(parent, nn.Module):
        return parent.state_dict()

    if isinstance(parent, (str, Path)):
        path = str(parent)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Parent checkpoint not found: {path}")
        json_path = path + ".json"
        if auto_load_ptq_checkpoints and _looks_like_ptq_sidecar(json_path):
            if not allow_unsafe_unpickle:
                raise ValueError(
                    f"PTQ checkpoint at {path!r} requires full-model unpickling "
                    "(weights_only=False), which is disabled by default for safety. "
                    "Pass allow_unsafe_unpickle=True only for trusted checkpoints."
                )
            q_model, _meta = load_quantized_checkpoint(path)
            return _float_state_dict_from_dynamic_quantized_module(q_model)
        # Try weights_only=True first (plain state-dict files).  Full-model
        # pickle fallback is opt-in via allow_unsafe_unpickle=True.
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            if not allow_unsafe_unpickle:
                raise ValueError(
                    f"Checkpoint at {path!r} requires full-model unpickling "
                    "(weights_only=False), which is disabled by default for safety. "
                    "Pass allow_unsafe_unpickle=True only for trusted checkpoints."
                ) from exc
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except RuntimeError as exc:
            # Only retry for errors that specifically indicate a weights-only
            # limitation (e.g. "Weights only load failed", "GLOBAL" opcode,
            # or "unsupported global").  Re-raise any other RuntimeError so
            # unrelated failures (I/O errors, corrupt files, …) surface
            # immediately.
            _msg = str(exc).lower()
            if any(
                token in _msg
                for token in ("weights only", "unsupported global", "global", "_codecs")
            ):
                if not allow_unsafe_unpickle:
                    raise ValueError(
                        f"Checkpoint at {path!r} requires full-model unpickling "
                        "(weights_only=False), which is disabled by default for safety. "
                        "Pass allow_unsafe_unpickle=True only for trusted checkpoints."
                    ) from exc
                obj = torch.load(path, map_location="cpu", weights_only=False)
            else:
                raise
        # If it's a full nn.Module, extract its state dict.
        if isinstance(obj, nn.Module):
            return obj.state_dict()
        if isinstance(obj, dict):
            return obj
        raise TypeError(
            f"Checkpoint at {path!r} contains an unsupported object type "
            f"({type(obj).__name__!r}).  Expected an nn.Module or a state dict."
        )

    if isinstance(parent, dict):
        return parent

    raise TypeError(
        f"parent must be an nn.Module, a path (str / Path), or a state dict "
        f"(dict); got {type(parent).__name__!r}"
    )


def _infer_arch_from_state_dict(
    sd: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Infer ``(input_dim, hidden_size, output_dim)`` from a Q-network state dict.

    Supports state dicts from :class:`~farm.core.decision.base_dqn.BaseQNetwork`
    and :class:`~farm.core.decision.base_dqn.StudentQNetwork`.  Both share the
    same ``nn.Sequential`` layout::

        network.0.weight  – shape (hidden_size, input_dim)
        network.4.weight  – shape (hidden_size, hidden_size)
        network.8.weight  – shape (output_dim, hidden_size)

    Parameters
    ----------
    sd:
        State dict (float or quantized).

    Returns
    -------
    Tuple[int, int, int]
        ``(input_dim, hidden_size, output_dim)``.

    Raises
    ------
    ValueError
        If the required keys are absent or have unexpected tensor shapes.
    """
    required = ("network.0.weight", "network.4.weight", "network.8.weight")
    missing = [k for k in required if k not in sd]
    if missing:
        raise ValueError(
            "Cannot infer architecture from state dict: missing keys "
            + str(missing)
            + ".  Ensure the state dict is from a BaseQNetwork / StudentQNetwork."
        )

    def _shape(key: str) -> Tuple[int, ...]:
        t = sd[key]
        if isinstance(t, torch.Tensor):
            if t.is_quantized:
                return tuple(t.dequantize().shape)
            return tuple(t.shape)
        raise ValueError(
            f"Expected a tensor at key {key!r}, got {type(t).__name__!r}"
        )

    w0 = _shape("network.0.weight")   # (hidden_size, input_dim)
    w8 = _shape("network.8.weight")   # (output_dim, hidden_size)

    if len(w0) != 2 or len(w8) != 2:
        raise ValueError(
            f"Unexpected weight tensor ranks: network.0.weight={w0}, "
            f"network.8.weight={w8}."
        )

    hidden_size = w0[0]
    input_dim = w0[1]
    output_dim = w8[0]
    return input_dim, hidden_size, output_dim


def _infer_parent_hidden_size_for_student(student_hidden: int) -> int:
    """Best-effort inverse of ``StudentQNetwork``'s ``max(16, parent_hidden // 2)``."""
    if student_hidden <= 16:
        return 32
    return student_hidden * 2


def _select_child_network_class(
    parent_a_module: Optional[nn.Module],
    network_class: Optional[Type[nn.Module]],
    *,
    prefer_student_from_arch: bool = False,
) -> Type[nn.Module]:
    from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork

    if network_class is not None:
        if network_class is not BaseQNetwork and network_class is not StudentQNetwork:
            raise TypeError(
                "network_class must be BaseQNetwork or StudentQNetwork "
                f"(got {network_class!r})"
            )
        return network_class
    if isinstance(parent_a_module, StudentQNetwork):
        return StudentQNetwork
    if prefer_student_from_arch:
        return StudentQNetwork
    if isinstance(parent_a_module, BaseQNetwork):
        return BaseQNetwork
    return BaseQNetwork


def _build_child_network_instance(
    cls: Type[nn.Module],
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    parent_hidden_size: Optional[int],
) -> nn.Module:
    from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork

    if cls is StudentQNetwork:
        ph = (
            parent_hidden_size
            if parent_hidden_size is not None
            else _infer_parent_hidden_size_for_student(hidden_size)
        )
        inferred_student_hidden = max(16, ph // 2)
        if inferred_student_hidden != hidden_size:
            raise ValueError(
                f"parent_hidden_size={ph} implies student hidden {inferred_student_hidden}, "
                f"but weights imply hidden {hidden_size}. "
                "Pass a matching ChildArchitectureSpec.parent_hidden_size or network_class=BaseQNetwork."
            )
        return StudentQNetwork(input_dim, output_dim, parent_hidden_size=ph)
    if cls is BaseQNetwork:
        return BaseQNetwork(input_dim, output_dim, hidden_size=hidden_size)
    raise TypeError(f"Unsupported network class {cls!r}")


# ---------------------------------------------------------------------------
# Public initializer
# ---------------------------------------------------------------------------


def initialize_child_from_crossover(
    parent_a: ParentSpec,
    parent_b: ParentSpec,
    strategy: str = "random",
    *,
    rng: Optional[Union[np.random.Generator, int]] = None,
    device: Optional[Union[torch.device, str]] = None,
    allow_unsafe_unpickle: bool = False,
    auto_load_ptq_checkpoints: bool = True,
    architecture: Optional[ChildArchitectureSpec] = None,
    network_class: Optional[Type[nn.Module]] = None,
    **strategy_kwargs: Any,
) -> nn.Module:
    """Build and initialise a child ``nn.Module`` from two parents via crossover.

    This is the **single entry point** for constructing an offspring Q-network.
    It orchestrates the full pipeline:

    1. **Resolve** *parent_a* / *parent_b* to float-or-quantized state dicts
       (live models, checkpoint paths, or dicts).  PTQ full-model checkpoints
       from :meth:`~farm.core.decision.training.quantize_ptq.PostTrainingQuantizer.save_checkpoint`
       are detected when ``<path>.json`` carries PTQ metadata and loaded via
       :func:`~farm.core.decision.training.quantize_ptq.load_quantized_checkpoint`.
    2. **Infer** (or take from *architecture*) ``input_dim``, ``hidden_size``,
       ``output_dim`` from parent A's tensors.
    3. **Instantiate** a fresh :class:`~farm.core.decision.base_dqn.BaseQNetwork`
       or :class:`~farm.core.decision.base_dqn.StudentQNetwork` (when *parent_a*
       is a student, or *network_class* requests it) on CPU.
    4. **Run** :func:`crossover_quantized_state_dict` to produce a float32
       child state dict.
    5. **Load** with ``strict=True``, optionally **.to(device)**, **eval()**.

    Parameters
    ----------
    parent_a:
        First parent – :class:`torch.nn.Module`, path to ``.pt``, or state dict.
        Architecture inference reads ``network.0`` / ``network.8`` weights.
    parent_b:
        Second parent – same types; must match *parent_a* keys/shapes.
    strategy:
        ``"random"``, ``"layer"``, or ``"weighted"`` (see
        :func:`crossover_quantized_state_dict`).
    rng:
        :class:`numpy.random.Generator` or ``int`` seed for ``"random"``.
    device:
        Target device; child is built on CPU then moved.
    allow_unsafe_unpickle:
        Allow ``weights_only=False`` fallback for pickle checkpoints
        (including PTQ ``*.pt`` + ``*.json`` auto-loads). Keep ``False``
        unless checkpoints are trusted.
    auto_load_ptq_checkpoints:
        When ``True`` (default), use :func:`load_quantized_checkpoint` for
        ``*.pt`` files that have a sibling ``*.json`` with PTQ metadata
        (trusted checkpoints only — unpickles the full model).
    architecture:
        Optional :class:`ChildArchitectureSpec` to skip tensor-based inference.
    network_class:
        Force :class:`~farm.core.decision.base_dqn.BaseQNetwork` vs
        :class:`~farm.core.decision.base_dqn.StudentQNetwork`.  When ``None``,
        uses the type of *parent_a* if it is a module, else ``BaseQNetwork``.
    **strategy_kwargs:
        Forwarded to :func:`crossover_quantized_state_dict` (``alpha``,
        ``seed``, …).

    Returns
    -------
    nn.Module
        Fresh Q-network in ``eval()`` mode.

    Notes
    -----
    **Quantized parents**: Per-tensor ``qint8`` and PTQ checkpoints are
    dequantized for crossover; the child is always float weights.

    **Student parent_hidden**: When the child is a
    :class:`~farm.core.decision.base_dqn.StudentQNetwork` and
    ``parent_hidden_size`` is not supplied, it is inferred from the student
    hidden width (``hidden_size * 2`` when hidden > 16, else ``32``), which may
    not match the original teacher width if it was odd — pass
    ``ChildArchitectureSpec(..., parent_hidden_size=...)`` to override.

    Examples
    --------
    Using live models::

        from farm.core.decision.base_dqn import BaseQNetwork
        from farm.core.decision.training.crossover import (
            initialize_child_from_crossover,
        )

        parent_a = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
        parent_b = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)

        child = initialize_child_from_crossover(
            parent_a, parent_b, strategy="weighted", alpha=0.7
        )
        out = child(torch.zeros(8))  # shape (4,)

    Using checkpoint paths::

        child = initialize_child_from_crossover(
            "checkpoints/agent_a.pt",
            "checkpoints/agent_b.pt",
            strategy="random",
            rng=42,
        )
    """
    parent_a_module = parent_a if isinstance(parent_a, nn.Module) else None

    sd_a = _resolve_parent(
        parent_a,
        allow_unsafe_unpickle=allow_unsafe_unpickle,
        auto_load_ptq_checkpoints=auto_load_ptq_checkpoints,
    )
    sd_b = _resolve_parent(
        parent_b,
        allow_unsafe_unpickle=allow_unsafe_unpickle,
        auto_load_ptq_checkpoints=auto_load_ptq_checkpoints,
    )

    prefer_student_from_arch = False
    if architecture is not None:
        input_dim = architecture.input_dim
        output_dim = architecture.output_dim
        hidden_size = architecture.hidden_size
        parent_hidden_override = architecture.parent_hidden_size
        prefer_student_from_arch = parent_hidden_override is not None
    else:
        try:
            input_dim, hidden_size, output_dim = _infer_arch_from_state_dict(sd_a)
            parent_hidden_override = None
        except ValueError:
            alt_spec: Optional[ChildArchitectureSpec] = None
            if isinstance(parent_a, (str, Path)):
                alt_spec = _read_arch_spec_from_ptq_json(str(parent_a))
            if alt_spec is None:
                raise
            input_dim = alt_spec.input_dim
            output_dim = alt_spec.output_dim
            hidden_size = alt_spec.hidden_size
            parent_hidden_override = alt_spec.parent_hidden_size
            prefer_student_from_arch = parent_hidden_override is not None

    cls = _select_child_network_class(
        parent_a_module,
        network_class,
        prefer_student_from_arch=prefer_student_from_arch,
    )
    child = _build_child_network_instance(
        cls,
        input_dim,
        output_dim,
        hidden_size,
        parent_hidden_override,
    )

    resolved_rng: Optional[np.random.Generator] = None
    if isinstance(rng, int):
        resolved_rng = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        resolved_rng = rng

    child_sd = crossover_quantized_state_dict(
        sd_a,
        sd_b,
        mode=strategy,
        rng=resolved_rng,
        **strategy_kwargs,
    )

    child.load_state_dict(child_sd, strict=True)

    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        child = child.to(device)

    child.eval()

    logger.info(
        "initialize_child_from_crossover",
        strategy=strategy,
        input_dim=input_dim,
        hidden_size=hidden_size,
        output_dim=output_dim,
        child_class=cls.__name__,
        device=str(device) if device is not None else "cpu",
    )
    return child
