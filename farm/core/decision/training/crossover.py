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
state-dict files, runs crossover, and saves the offspring.
"""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def _resolve_parent(
    parent: ParentSpec,
) -> Dict[str, Any]:
    """Resolve *parent* to a state dict for use with :func:`crossover_quantized_state_dict`.

    The returned dict may contain float or quantized tensors; conversion to
    float32 happens later inside the crossover helpers via :func:`_to_float`.

    Accepts:
    * An ``nn.Module`` – its :meth:`~nn.Module.state_dict` is returned as-is
      (potentially on any device / dtype).
    * A ``str`` or ``pathlib.Path`` pointing to a ``.pt`` file.  Plain
      state-dict files (``torch.save(model.state_dict(), …)``) are loaded with
      ``weights_only=True``.  If that fails due to the file containing a
      full-model pickle, loading is retried with ``weights_only=False`` — see
      the warning below.  If the loaded object is an ``nn.Module``, its state
      dict is extracted.
    * A ``dict`` (state dict) – returned as-is without copying.

    .. warning::
        When *parent* is a path and ``weights_only=True`` loading fails, this
        function retries with ``weights_only=False``, which can execute
        arbitrary code during unpickling.  Only use trusted checkpoint files.
        To avoid this, convert the checkpoint to a plain state-dict file first:
        ``torch.save(model.state_dict(), path)`` and pass that path instead.

    Parameters
    ----------
    parent:
        Parent specification.

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
        # Try weights_only=True first (plain state-dict files).  Fall back to
        # weights_only=False only for the specific errors that indicate a
        # full-model pickle (e.g. a quantized checkpoint).
        # WARNING: weights_only=False can execute arbitrary code; only trust
        # verified checkpoint files.
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError):
            obj = torch.load(path, map_location="cpu", weights_only=False)
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
    **strategy_kwargs: Any,
) -> nn.Module:
    """Build and initialise a child ``nn.Module`` from two parents via crossover.

    This is the **single entry point** for constructing an offspring Q-network.
    It orchestrates the full pipeline:

    1. **Resolve** *parent_a* / *parent_b* to float-or-quantized state dicts
       (accepts live models, checkpoint paths, or pre-loaded state dicts).
    2. **Validate** that both parents share identical keys and tensor shapes.
    3. **Infer** the child architecture (``input_dim``, ``hidden_size``,
       ``output_dim``) from parent A's state dict.
    4. **Instantiate** a fresh :class:`~farm.core.decision.base_dqn.BaseQNetwork`
       on CPU with the inferred dimensions.
    5. **Run** the requested crossover strategy (delegating to
       :func:`crossover_quantized_state_dict`) to produce a float32 child
       state dict.
    6. **Load** the child state dict with ``strict=True``.
    7. **Move** the child to *device* (defaults to CPU) and set it to
       ``eval()`` mode.

    Parameters
    ----------
    parent_a:
        First parent – an :class:`torch.nn.Module`, a filesystem path
        (``str`` / :class:`pathlib.Path`) to a ``.pt`` checkpoint, or a
        pre-loaded state dict (``dict``).  Inputs must be state-dict
        compatible after loading/dequantization so the standard network
        parameter keys used for architecture inference (for example,
        ``network.0.weight`` and the final layer weight) are present.
        Plain full-model quantized checkpoints saved by
        :meth:`~farm.core.decision.training.quantize_ptq.PostTrainingQuantizer.save_checkpoint`
        are not supported here unless they have first been converted back
        into such a float/state-dict representation.
    parent_b:
        Second parent – same accepted types and compatibility
        requirements as *parent_a*.
    strategy:
        Crossover strategy.  One of:

        ``"random"``
            Each parameter tensor is independently drawn from parent A with
            probability *alpha* (default ``0.5``) or from parent B.
            Controlled by *rng* / the ``seed`` kwarg for reproducibility.

        ``"layer"``
            Parameters are grouped into logical blocks (Linear + LayerNorm per
            stage for standard Q-networks).  Even-indexed blocks come from
            parent A, odd-indexed from parent B.

        ``"weighted"``
            Child tensor = ``alpha * parent_a + (1 - alpha) * parent_b``
            in float32.  Requires *alpha* in *strategy_kwargs*.

    rng:
        RNG for stochastic strategies.  Accepts a
        :class:`numpy.random.Generator` or an ``int`` seed.  When ``None``
        and no ``"seed"`` key is present in *strategy_kwargs*, the RNG is
        non-reproducible.  Ignored by ``"layer"`` and ``"weighted"``
        strategies.
    device:
        Target device for the returned child.  Accepts a
        :class:`torch.device` or a device string (e.g. ``"cpu"``,
        ``"cuda:0"``).  The child is **always constructed on CPU** first,
        then moved to *device*.  Defaults to ``torch.device("cpu")``.
    **strategy_kwargs:
        Additional keyword arguments forwarded to
        :func:`crossover_quantized_state_dict`, e.g.:

        * ``alpha`` – blend / selection coefficient (float, ``[0, 1]``).
        * ``seed`` – integer seed (used when *rng* is ``None``).

    Returns
    -------
    nn.Module
        A fresh :class:`~farm.core.decision.base_dqn.BaseQNetwork` in
        ``eval()`` mode on *device* with weights initialised from the
        crossover of *parent_a* and *parent_b*.

    Raises
    ------
    TypeError
        If any parent is not an ``nn.Module``, path, or dict.
    FileNotFoundError
        If a parent path does not point to an existing file.
    ValueError
        If the parents' state dicts have mismatched keys / shapes, or if
        the architecture cannot be inferred from the state dict, or if
        *strategy* is not a recognised mode.

    Notes
    -----
    **Key alignment**: Both parents must have identical state-dict keys and
    matching tensor shapes.  Any mismatch raises a ``ValueError`` immediately.

    **Quantized parents**: Quantized tensors (``torch.qint8``) are
    dequantized to float32 before crossover, so the returned child is
    always a float-precision model suitable for inference or downstream
    training / re-quantization.

    **Determinism**: Pass *rng* as an integer seed or a seeded
    :class:`numpy.random.Generator` for reproducible ``"random"`` crossover.
    ``"layer"`` and ``"weighted"`` with a fixed *alpha* are fully
    deterministic.

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
    from farm.core.decision.base_dqn import BaseQNetwork

    # ------------------------------------------------------------------
    # 1. Resolve parents to state dicts
    # ------------------------------------------------------------------
    sd_a = _resolve_parent(parent_a)
    sd_b = _resolve_parent(parent_b)

    # ------------------------------------------------------------------
    # 2. Infer architecture from parent A (keys/shapes validated in crossover)
    # ------------------------------------------------------------------
    input_dim, hidden_size, output_dim = _infer_arch_from_state_dict(sd_a)

    # ------------------------------------------------------------------
    # 4. Instantiate a fresh child on CPU
    # ------------------------------------------------------------------
    child = BaseQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
    )

    # ------------------------------------------------------------------
    # 5. Run crossover strategy
    # ------------------------------------------------------------------
    # Normalise rng: int → Generator, keep Generator, leave None alone
    resolved_rng: Optional[np.random.Generator] = None
    if isinstance(rng, int):
        resolved_rng = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        resolved_rng = rng
    # If rng is None, crossover_quantized_state_dict may still use seed
    # kwarg from strategy_kwargs.

    child_sd = crossover_quantized_state_dict(
        sd_a,
        sd_b,
        mode=strategy,
        rng=resolved_rng,
        **strategy_kwargs,
    )

    # ------------------------------------------------------------------
    # 6. Load state dict (strict)
    # ------------------------------------------------------------------
    child.load_state_dict(child_sd, strict=True)

    # ------------------------------------------------------------------
    # 7. Move to device and set eval mode
    # ------------------------------------------------------------------
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
        device=str(device) if device is not None else "cpu",
    )
    return child
