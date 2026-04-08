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
    Group parameters by their top-level module index (i.e. ``network.0.*``,
    ``network.4.*``, …) and alternate entire groups between parents: even
    groups from A, odd groups from B.  Keeps associated weight + bias +
    LayerNorm parameters together from the same parent.

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
state-dict files, runs crossover, and saves the offspring.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from farm.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

CROSSOVER_MODES = ("random", "layer", "weighted")


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
        # Only validate tensors (state dicts may contain non-tensor scalars)
        if not isinstance(va, torch.Tensor) or not isinstance(vb, torch.Tensor):
            continue
        fa = _to_float(va)
        fb = _to_float(vb)
        if fa.shape != fb.shape:
            shape_errors.append(f"  '{k}': A={tuple(fa.shape)}, B={tuple(fb.shape)}")

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


# ---------------------------------------------------------------------------
# Crossover mode implementations
# ---------------------------------------------------------------------------


def _random_crossover(
    sd_a: Dict[str, Any],
    sd_b: Dict[str, Any],
    keys: List[str],
    rng: np.random.Generator,
    alpha: float,
) -> Dict[str, torch.Tensor]:
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
    Float32 offspring state dict.
    """
    child: Dict[str, torch.Tensor] = {}
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
) -> Dict[str, torch.Tensor]:
    """Layer-group-based crossover.

    Parameters are grouped by their top-level module index.  Even-indexed
    groups come from parent A; odd-indexed groups come from parent B.  This
    keeps weight, bias, and associated LayerNorm parameters from the same
    parent within each block, avoiding inconsistent feature scaling.

    Parameters
    ----------
    sd_a, sd_b:
        Source state dicts (already validated).
    keys:
        Sorted list of keys to process.

    Returns
    -------
    Float32 offspring state dict.
    """
    groups = _layer_groups(keys)
    child: Dict[str, torch.Tensor] = {}
    for group_idx, (_, group_keys) in enumerate(groups.items()):
        parent = sd_a if group_idx % 2 == 0 else sd_b
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
) -> Dict[str, torch.Tensor]:
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
    Float32 offspring state dict.
    """
    child: Dict[str, torch.Tensor] = {}
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
) -> Dict[str, torch.Tensor]:
    """Combine two Q-network state dicts into a single offspring state dict.

    Both parents must have **identical keys and tensor shapes**.  Quantized
    tensors (``torch.qint8`` or other integer dtypes) are dequantized to
    ``float32`` before crossover.  The offspring is always returned as a
    ``float32`` state dict compatible with :meth:`nn.Module.load_state_dict`.

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
            Parameters are grouped by top-level module block.  Even-indexed
            blocks come from parent A, odd-indexed from parent B.  This keeps
            the weight, bias, and LayerNorm of each ``Linear`` block together
            from the same parent.

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
    Dict[str, torch.Tensor]
        Offspring state dict with float32 tensors.

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
) -> Dict[str, torch.Tensor]:
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
    Dict[str, torch.Tensor]
        The offspring float32 state dict (also saved to *output_path*).
    """
    import os

    sd_a = torch.load(path_a, map_location="cpu", weights_only=True)
    sd_b = torch.load(path_b, map_location="cpu", weights_only=True)

    child = crossover_quantized_state_dict(
        sd_a,
        sd_b,
        mode=mode,
        alpha=alpha,
        seed=seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(child, output_path)
    logger.info("crossover_checkpoint_saved", path=output_path, mode=mode)
    return child
