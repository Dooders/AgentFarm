"""Qualitative and error-analysis tooling for recombined Q-networks.

This module provides helpers for **case-level** inspection of crossover children:
where the child disagrees with its parents, which states produce the largest
errors, and optionally what the internal hidden-layer activations look like.

Key exports
-----------
:class:`DisagreementRecord`
    Per-state record capturing child vs parent A/B argmax disagreements,
    per-state KL divergence, MSE, cosine similarity, and optionally raw logits.

:func:`extract_disagreements`
    Run three models over a shared state array and return one
    :class:`DisagreementRecord` per state.

:func:`worst_k_states`
    Sort records by a scalar error criterion and return the worst *k*.

:func:`export_disagreements_csv`
    Write per-state records to a CSV file.

:func:`export_disagreements_json`
    Write per-state records (and summary statistics) to a JSON file.

:func:`extract_activations`
    Memory-bounded export of one hidden-layer's post-activation outputs for a
    probe set of states (returns a NumPy array).

Typical API usage
-----------------
::

    import numpy as np
    from farm.core.decision.base_dqn import BaseQNetwork
    from farm.core.decision.training.recombination_analysis import (
        extract_disagreements,
        worst_k_states,
        export_disagreements_csv,
        export_disagreements_json,
        extract_activations,
    )

    parent_a = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    parent_b = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    child    = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    states   = np.random.randn(500, 8).astype("float32")

    records = extract_disagreements(parent_a, parent_b, child, states,
                                    include_logits=True, k_values=[1, 2, 3])
    worst   = worst_k_states(records, k=10, criterion="max_kl")
    export_disagreements_csv(records, "disagreements.csv")
    export_disagreements_json(records, "disagreements.json")

    # Hidden-layer activation export (first hidden ReLU output)
    acts = extract_activations(child, states, layer_index=2, max_states=200)
    np.save("child_activations.npy", acts)

CLI
---
See ``scripts/analyze_recombination.py`` for the command-line interface.

Related
-------
- :mod:`farm.core.decision.training.recombination_eval` — aggregate fidelity metrics
- ``scripts/analyze_recombination.py`` — CLI wrapping this module
- ``docs/howto/neural_recombination_runbook.md`` — integration guide
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from farm.utils.logging import get_logger

logger = get_logger(__name__)

#: Schema version embedded in JSON exports produced by this module.
ANALYSIS_SCHEMA_VERSION = "1.0"

#: Supported criterion names for :func:`worst_k_states`.
WORST_K_CRITERIA = frozenset(
    {
        "kl_parent_a",
        "kl_parent_b",
        "max_kl",
        "mse_parent_a",
        "mse_parent_b",
        "max_mse",
    }
)


# ---------------------------------------------------------------------------
# Per-state record
# ---------------------------------------------------------------------------


@dataclass
class DisagreementRecord:
    """Per-state disagreement information between a child and its two parents.

    Attributes
    ----------
    state_index:
        Index of this state in the original evaluation array.
    child_action:
        Argmax action selected by the child network.
    parent_a_action:
        Argmax action selected by parent A.
    parent_b_action:
        Argmax action selected by parent B.
    agrees_with_parent_a:
        ``True`` when ``child_action == parent_a_action``.
    agrees_with_parent_b:
        ``True`` when ``child_action == parent_b_action``.
    agrees_with_any_parent:
        ``True`` when the child agrees with *at least one* parent (oracle).
    parent_a_in_child_top_k:
        Mapping ``{k: bool}`` — whether parent A's argmax appears in the
        child's top-*k* actions.  Only populated when :func:`extract_disagreements`
        is called with ``k_values``.
    parent_b_in_child_top_k:
        Mapping ``{k: bool}`` — whether parent B's argmax appears in the
        child's top-*k* actions.
    kl_child_vs_parent_a:
        Per-state KL divergence KL(softmax(parent_a) || softmax(child)).
    kl_child_vs_parent_b:
        Per-state KL divergence KL(softmax(parent_b) || softmax(child)).
    mse_child_vs_parent_a:
        Per-state mean squared error between child and parent A raw logits.
    mse_child_vs_parent_b:
        Per-state mean squared error between child and parent B raw logits.
    cosine_child_vs_parent_a:
        Per-state cosine similarity between child and parent A logit vectors.
    cosine_child_vs_parent_b:
        Per-state cosine similarity between child and parent B logit vectors.
    child_logits:
        Raw Q-value logits from the child network.  ``None`` when
        ``include_logits=False`` was passed to :func:`extract_disagreements`.
    parent_a_logits:
        Raw Q-value logits from parent A.  ``None`` when ``include_logits=False``.
    parent_b_logits:
        Raw Q-value logits from parent B.  ``None`` when ``include_logits=False``.
    """

    state_index: int
    child_action: int
    parent_a_action: int
    parent_b_action: int
    agrees_with_parent_a: bool
    agrees_with_parent_b: bool
    agrees_with_any_parent: bool
    parent_a_in_child_top_k: Dict[int, bool]
    parent_b_in_child_top_k: Dict[int, bool]
    kl_child_vs_parent_a: float
    kl_child_vs_parent_b: float
    mse_child_vs_parent_a: float
    mse_child_vs_parent_b: float
    cosine_child_vs_parent_a: float
    cosine_child_vs_parent_b: float
    child_logits: Optional[List[float]] = field(default=None)
    parent_a_logits: Optional[List[float]] = field(default=None)
    parent_b_logits: Optional[List[float]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of all fields.

        ``parent_a_in_child_top_k`` and ``parent_b_in_child_top_k`` have
        their integer keys converted to strings for JSON compatibility.
        """
        d = asdict(self)
        # JSON keys must be strings
        d["parent_a_in_child_top_k"] = {
            str(k): v for k, v in self.parent_a_in_child_top_k.items()
        }
        d["parent_b_in_child_top_k"] = {
            str(k): v for k, v in self.parent_b_in_child_top_k.items()
        }
        return d

    def flat_csv_row(self, k_values: Sequence[int] = ()) -> Dict[str, Any]:
        """Return a flat dictionary suitable for a single CSV row.

        Top-k mismatch columns are emitted as ``parent_a_in_top_k_{k}`` and
        ``parent_b_in_top_k_{k}`` for each *k* in ``k_values``.  Any *k* not
        present in :attr:`parent_a_in_child_top_k` is omitted.

        Parameters
        ----------
        k_values:
            Ordered sequence of *k* values whose columns to include.  Defaults
            to the keys already stored in :attr:`parent_a_in_child_top_k`.
        """
        ks = list(k_values) if k_values else sorted(self.parent_a_in_child_top_k)
        row: Dict[str, Any] = {
            "state_index": self.state_index,
            "child_action": self.child_action,
            "parent_a_action": self.parent_a_action,
            "parent_b_action": self.parent_b_action,
            "agrees_with_parent_a": self.agrees_with_parent_a,
            "agrees_with_parent_b": self.agrees_with_parent_b,
            "agrees_with_any_parent": self.agrees_with_any_parent,
            "kl_child_vs_parent_a": self.kl_child_vs_parent_a,
            "kl_child_vs_parent_b": self.kl_child_vs_parent_b,
            "mse_child_vs_parent_a": self.mse_child_vs_parent_a,
            "mse_child_vs_parent_b": self.mse_child_vs_parent_b,
            "cosine_child_vs_parent_a": self.cosine_child_vs_parent_a,
            "cosine_child_vs_parent_b": self.cosine_child_vs_parent_b,
        }
        for k in ks:
            row[f"parent_a_in_top_k_{k}"] = self.parent_a_in_child_top_k.get(k, "")
            row[f"parent_b_in_top_k_{k}"] = self.parent_b_in_child_top_k.get(k, "")
        return row


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_disagreements(
    parent_a: nn.Module,
    parent_b: nn.Module,
    child: nn.Module,
    states: np.ndarray,
    *,
    include_logits: bool = False,
    k_values: Optional[List[int]] = None,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> List[DisagreementRecord]:
    """Run three Q-networks over *states* and return per-state disagreement records.

    Parameters
    ----------
    parent_a:
        First parent ``nn.Module``.
    parent_b:
        Second parent ``nn.Module``.
    child:
        Child ``nn.Module`` produced by crossover (and optionally fine-tuned).
    states:
        NumPy array of shape ``(N, input_dim)`` with ``dtype=float32``.
    include_logits:
        When ``True``, the raw Q-value logit vectors are stored in every
        :class:`DisagreementRecord`.  This can be large for many states.
    k_values:
        Values of *k* for top-*k* mismatch flags.  Defaults to ``[1, 2, 3]``.
        Each record will have ``parent_{a,b}_in_child_top_k[k]`` populated.
    batch_size:
        Maximum states per forward pass (memory bound).
    device:
        PyTorch device.  Defaults to CPU.

    Returns
    -------
    List[DisagreementRecord]
        One record per state, in the same order as *states*.

    Raises
    ------
    ValueError
        If *states* is not a non-empty 2-D array.
    """
    if k_values is None:
        k_values = [1, 2, 3]

    _dev = device or torch.device("cpu")
    states_arr = _validate_states_2d(states)
    n_states = states_arr.shape[0]

    for model in (parent_a, parent_b, child):
        model.to(_dev)
        model.eval()

    records: List[DisagreementRecord] = []

    for start in range(0, n_states, batch_size):
        end = min(start + batch_size, n_states)
        chunk = torch.tensor(states_arr[start:end], dtype=torch.float32, device=_dev)

        logits_a = parent_a(chunk)
        logits_b = parent_b(chunk)
        logits_c = child(chunk)

        # Ensure 2-D (handle models that squeeze single samples)
        if logits_a.dim() == 1:
            logits_a = logits_a.unsqueeze(0)
        if logits_b.dim() == 1:
            logits_b = logits_b.unsqueeze(0)
        if logits_c.dim() == 1:
            logits_c = logits_c.unsqueeze(0)

        n_act = logits_c.size(-1)
        actions_a = logits_a.argmax(dim=-1)  # (batch,)
        actions_b = logits_b.argmax(dim=-1)
        actions_c = logits_c.argmax(dim=-1)

        agrees_a = (actions_c == actions_a)
        agrees_b = (actions_c == actions_b)
        agrees_any = agrees_a | agrees_b

        # Per-state KL divergence (KL(p_ref ‖ p_child) for each parent)
        p_a = F.softmax(logits_a, dim=-1)
        p_b = F.softmax(logits_b, dim=-1)
        log_p_c = F.log_softmax(logits_c, dim=-1)
        # kl_div expects (log_input, target); reduction="none" gives per-element
        kl_a = F.kl_div(log_p_c, p_a, reduction="none").sum(dim=-1)  # (batch,)
        kl_b = F.kl_div(log_p_c, p_b, reduction="none").sum(dim=-1)

        # Per-state MSE
        diff_a = logits_c - logits_a
        diff_b = logits_c - logits_b
        mse_a = (diff_a ** 2).mean(dim=-1)  # (batch,)
        mse_b = (diff_b ** 2).mean(dim=-1)

        # Per-state cosine similarity
        cos_a = F.cosine_similarity(logits_c, logits_a, dim=-1)  # (batch,)
        cos_b = F.cosine_similarity(logits_c, logits_b, dim=-1)

        # Top-k mismatch flags: for each k, is parent's argmax in child's top-k?
        topk_c: Dict[int, torch.Tensor] = {}
        for k in k_values:
            k_clamped = min(k, n_act)
            topk_c[k] = logits_c.topk(k_clamped, dim=-1).indices  # (batch, k)

        # Materialise to CPU lists
        actions_a_list = actions_a.cpu().tolist()
        actions_b_list = actions_b.cpu().tolist()
        actions_c_list = actions_c.cpu().tolist()
        agrees_a_list = agrees_a.cpu().tolist()
        agrees_b_list = agrees_b.cpu().tolist()
        agrees_any_list = agrees_any.cpu().tolist()
        kl_a_list = kl_a.cpu().tolist()
        kl_b_list = kl_b.cpu().tolist()
        mse_a_list = mse_a.cpu().tolist()
        mse_b_list = mse_b.cpu().tolist()
        cos_a_list = cos_a.cpu().tolist()
        cos_b_list = cos_b.cpu().tolist()

        logits_a_np = logits_a.cpu().numpy() if include_logits else None
        logits_b_np = logits_b.cpu().numpy() if include_logits else None
        logits_c_np = logits_c.cpu().numpy() if include_logits else None

        for i in range(end - start):
            global_idx = start + i
            topk_a_in: Dict[int, bool] = {}
            topk_b_in: Dict[int, bool] = {}
            for k, indices_t in topk_c.items():
                row_indices = indices_t[i].cpu().tolist()
                topk_a_in[k] = int(actions_a_list[i]) in row_indices
                topk_b_in[k] = int(actions_b_list[i]) in row_indices

            record = DisagreementRecord(
                state_index=global_idx,
                child_action=int(actions_c_list[i]),
                parent_a_action=int(actions_a_list[i]),
                parent_b_action=int(actions_b_list[i]),
                agrees_with_parent_a=bool(agrees_a_list[i]),
                agrees_with_parent_b=bool(agrees_b_list[i]),
                agrees_with_any_parent=bool(agrees_any_list[i]),
                parent_a_in_child_top_k=topk_a_in,
                parent_b_in_child_top_k=topk_b_in,
                kl_child_vs_parent_a=float(kl_a_list[i]),
                kl_child_vs_parent_b=float(kl_b_list[i]),
                mse_child_vs_parent_a=float(mse_a_list[i]),
                mse_child_vs_parent_b=float(mse_b_list[i]),
                cosine_child_vs_parent_a=float(cos_a_list[i]),
                cosine_child_vs_parent_b=float(cos_b_list[i]),
                child_logits=logits_c_np[i].tolist() if logits_c_np is not None else None,
                parent_a_logits=logits_a_np[i].tolist() if logits_a_np is not None else None,
                parent_b_logits=logits_b_np[i].tolist() if logits_b_np is not None else None,
            )
            records.append(record)

    logger.info(
        "extract_disagreements_complete",
        n_states=n_states,
        n_disagree_a=sum(not r.agrees_with_parent_a for r in records),
        n_disagree_b=sum(not r.agrees_with_parent_b for r in records),
        include_logits=include_logits,
    )
    return records


# ---------------------------------------------------------------------------
# Worst-k filtering
# ---------------------------------------------------------------------------


def worst_k_states(
    records: List[DisagreementRecord],
    k: int,
    criterion: str = "max_kl",
) -> List[DisagreementRecord]:
    """Return the *k* records with the largest error according to *criterion*.

    The returned list is sorted by descending criterion value (worst first).

    Parameters
    ----------
    records:
        List of :class:`DisagreementRecord` objects (e.g. from
        :func:`extract_disagreements`).
    k:
        Number of worst states to return.  Clamped to ``len(records)`` if
        *k* is larger than the available records.
    criterion:
        Scalar metric to sort by.  Must be one of:

        ``"kl_parent_a"``
            KL divergence of child vs parent A for this state.
        ``"kl_parent_b"``
            KL divergence of child vs parent B.
        ``"max_kl"``
            Maximum of the two per-state KL values.
        ``"mse_parent_a"``
            MSE between child and parent A logits.
        ``"mse_parent_b"``
            MSE between child and parent B logits.
        ``"max_mse"``
            Maximum of the two per-state MSE values.

    Returns
    -------
    List[DisagreementRecord]
        Up to *k* records, worst first.

    Raises
    ------
    ValueError
        If *criterion* is not one of :data:`WORST_K_CRITERIA` or *k* ≤ 0.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer; got {k!r}.")
    if criterion not in WORST_K_CRITERIA:
        raise ValueError(
            f"criterion must be one of {sorted(WORST_K_CRITERIA)}; got {criterion!r}."
        )

    def _score(r: DisagreementRecord) -> float:
        if criterion == "kl_parent_a":
            return r.kl_child_vs_parent_a
        if criterion == "kl_parent_b":
            return r.kl_child_vs_parent_b
        if criterion == "max_kl":
            return max(r.kl_child_vs_parent_a, r.kl_child_vs_parent_b)
        if criterion == "mse_parent_a":
            return r.mse_child_vs_parent_a
        if criterion == "mse_parent_b":
            return r.mse_child_vs_parent_b
        # "max_mse"
        return max(r.mse_child_vs_parent_a, r.mse_child_vs_parent_b)

    sorted_records = sorted(records, key=_score, reverse=True)
    return sorted_records[: min(k, len(sorted_records))]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_disagreements_csv(
    records: List[DisagreementRecord],
    path: str,
    *,
    k_values: Optional[List[int]] = None,
) -> None:
    """Write per-state disagreement records to a CSV file.

    Each row corresponds to one state.  Columns include state index, child/parent
    actions, agreement flags, per-state KL, MSE, cosine similarity, and top-*k*
    mismatch flags.  Raw logits are **not** written to CSV (use
    :func:`export_disagreements_json` for logits).

    Parameters
    ----------
    records:
        List of :class:`DisagreementRecord` objects.
    path:
        Destination CSV file path.  Parent directories must exist.
    k_values:
        Top-*k* columns to include.  Defaults to all *k* values present in
        the first record's ``parent_a_in_child_top_k``.
    """
    if not records:
        logger.warning("export_disagreements_csv_empty", path=path)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return

    ks = list(k_values) if k_values else sorted(records[0].parent_a_in_child_top_k)
    rows = [r.flat_csv_row(ks) for r in records]
    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("export_disagreements_csv_done", path=path, n_rows=len(rows))


def export_disagreements_json(
    records: List[DisagreementRecord],
    path: str,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write per-state disagreement records to a JSON file.

    The output document has the following top-level keys:

    ``schema_version``
        :data:`ANALYSIS_SCHEMA_VERSION`.
    ``n_states``
        Total number of records.
    ``n_disagree_with_parent_a``
        Count of states where child's argmax ≠ parent A's argmax.
    ``n_disagree_with_parent_b``
        Count of states where child's argmax ≠ parent B's argmax.
    ``n_disagree_with_both``
        Count of states where child disagrees with both parents simultaneously.
    ``records``
        List of per-state dictionaries (see :meth:`DisagreementRecord.to_dict`).
    ``metadata``
        Optional extra key/value pairs passed via ``extra_metadata``.

    Parameters
    ----------
    records:
        List of :class:`DisagreementRecord` objects.
    path:
        Destination JSON file path.  Parent directories must exist.
    extra_metadata:
        Optional extra metadata merged into the ``"metadata"`` key of the
        output document.
    """
    n_disagree_a = sum(not r.agrees_with_parent_a for r in records)
    n_disagree_b = sum(not r.agrees_with_parent_b for r in records)
    n_disagree_both = sum(
        not r.agrees_with_parent_a and not r.agrees_with_parent_b for r in records
    )
    doc: Dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "n_states": len(records),
        "n_disagree_with_parent_a": n_disagree_a,
        "n_disagree_with_parent_b": n_disagree_b,
        "n_disagree_with_both": n_disagree_both,
        "records": [r.to_dict() for r in records],
        "metadata": extra_metadata or {},
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, allow_nan=False)

    logger.info("export_disagreements_json_done", path=path, n_records=len(records))


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_activations(
    model: nn.Module,
    states: np.ndarray,
    *,
    layer_index: int = 2,
    max_states: Optional[int] = None,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract hidden-layer activations for a subset of states.

    A forward hook is registered on the sub-module identified by
    *layer_index* (an index into ``list(model.modules())`` starting from the
    model itself at index 0).  For :class:`~farm.core.decision.base_dqn.BaseQNetwork`
    the indices of notable layers are:

    =========  ===========================================
    Index      Layer
    =========  ===========================================
    0          BaseQNetwork (root)
    1          network (nn.Sequential)
    2          network[0] — Linear (input → hidden)
    3          network[1] — LayerNorm
    4          network[2] — ReLU  (**first hidden ReLU**)
    5          network[3] — Dropout
    6          network[4] — Linear (hidden → hidden)
    7          network[5] — LayerNorm
    8          network[6] — ReLU  (**second hidden ReLU**)
    9          network[7] — Dropout
    10         network[8] — Linear (hidden → output)
    =========  ===========================================

    For other architectures, inspect ``list(model.modules())`` and pass the
    appropriate index.

    Parameters
    ----------
    model:
        The ``nn.Module`` to probe.  Set to eval mode; weights unchanged.
    states:
        NumPy array of shape ``(N, input_dim)`` with ``dtype=float32``.
    layer_index:
        Index of the target sub-module in ``list(model.modules())``.
        Defaults to ``2`` (the first ``nn.Linear`` in a ``BaseQNetwork``
        stored as its ``network`` sequential; override for deeper layers).
    max_states:
        Cap on the number of states to process.  ``None`` processes all.
        Use this to stay memory-bounded when *states* is large.
    batch_size:
        Maximum states per forward pass.
    device:
        PyTorch device.  Defaults to CPU.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(min(N, max_states), activation_dim)``
        containing the output of the hooked layer for each state.

    Raises
    ------
    ValueError
        If *layer_index* is out of range for *model*, or *states* is invalid.
    """
    _dev = device or torch.device("cpu")
    states_arr = _validate_states_2d(states)

    if max_states is not None and max_states > 0:
        states_arr = states_arr[:max_states]

    n_states = states_arr.shape[0]
    model.to(_dev)
    model.eval()

    # Resolve the target sub-module by flat index.
    all_modules = list(model.modules())
    if not (0 <= layer_index < len(all_modules)):
        raise ValueError(
            f"layer_index {layer_index!r} is out of range for model with "
            f"{len(all_modules)} sub-modules (indices 0–{len(all_modules) - 1})."
        )
    target_module = all_modules[layer_index]

    # Register a hook to capture activations.
    captured: List[torch.Tensor] = []

    def _hook(_module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
        captured.append(output.detach().cpu())

    handle = target_module.register_forward_hook(_hook)

    try:
        for start in range(0, n_states, batch_size):
            end = min(start + batch_size, n_states)
            chunk = torch.tensor(states_arr[start:end], dtype=torch.float32, device=_dev)
            model(chunk)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "No activations were captured; the hooked layer may not have been "
            "reached during the forward pass."
        )

    result = torch.cat(captured, dim=0)  # (n_states, activation_dim)
    acts = result.numpy().astype("float32")

    logger.info(
        "extract_activations_done",
        n_states=n_states,
        layer_index=layer_index,
        activation_shape=list(acts.shape),
    )
    return acts


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_states_2d(states: np.ndarray) -> np.ndarray:
    """Validate and coerce *states* to a float32 2-D array."""
    arr = np.asarray(states, dtype="float32")
    if arr.ndim != 2:
        raise ValueError(
            f"states must be a 2-D array of shape (N, input_dim); got shape {arr.shape!r}."
        )
    if arr.shape[0] == 0:
        raise ValueError("states must be non-empty; got 0 rows.")
    return arr
