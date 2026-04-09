"""Evaluator for comparing a crossover child against its parent networks.

This module provides the :class:`RecombinationEvaluator` and supporting
dataclasses for quantitatively assessing how well a child Q-network produced
by crossover (and optionally fine-tuned) relates to both of its parents.

For each pair — *child vs parent A* and *child vs parent B* — the evaluator
computes:

- **Top-1 / top-k action agreement**: fraction of states where both models
  select the same action (top-1) or the child's top-k includes the parent's
  argmax (top-k).
- **Output similarity**: KL divergence, MSE, MAE, and cosine similarity on
  Q-value logits.
- Optional **parent A vs parent B** baseline to show the "diversity gap".

Results are gathered into a :class:`RecombinationReport` that can be
serialised to a versioned JSON document and checked against configurable
:class:`RecombinationThresholds`.

Typical usage
-------------
::

    import numpy as np
    from farm.core.decision.base_dqn import BaseQNetwork
    from farm.core.decision.training.recombination_eval import (
        RecombinationEvaluator,
        RecombinationThresholds,
    )

    parent_a = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    parent_b = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)
    child    = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)

    states = np.random.randn(500, 8).astype("float32")

    thresholds = RecombinationThresholds(min_action_agreement=0.70)
    evaluator  = RecombinationEvaluator(parent_a, parent_b, child,
                                        thresholds=thresholds)
    report = evaluator.evaluate(states, include_parent_baseline=True)
    print(report.passed, report.to_dict())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from farm.utils.logging import get_logger

logger = get_logger(__name__)

#: Versioned JSON schema identifier embedded in every report.
#: ``1.1`` adds optional ``model_formats`` (how each role was loaded).
REPORT_SCHEMA_VERSION = "1.1"


@dataclass
class _PairwiseChunkAgg:
    """Running sums for streaming metrics over state batches (one ref/query pair)."""

    n_rows: int = 0
    n_actions: int = 0
    top1_matches: int = 0
    topk_matches: Dict[int, int] = field(default_factory=dict)
    kl_sum: float = 0.0
    mse_sum: float = 0.0
    mae_sum: float = 0.0
    cos_sum: float = 0.0


def _new_pairwise_agg(k_values: List[int]) -> _PairwiseChunkAgg:
    return _PairwiseChunkAgg(topk_matches={k: 0 for k in k_values})


@torch.no_grad()
def _update_pairwise_agg(
    agg: _PairwiseChunkAgg,
    ref_logits: torch.Tensor,
    qry_logits: torch.Tensor,
    k_values: List[int],
) -> None:
    """Accumulate metrics for one tensor batch (same semantics as :meth:`_pairwise`)."""
    batch_n = ref_logits.size(0)
    n_act = ref_logits.size(-1)
    if agg.n_actions == 0:
        agg.n_actions = n_act
    elif agg.n_actions != n_act:
        raise ValueError(
            f"Inconsistent action dimension across batches: {agg.n_actions} vs {n_act}"
        )

    ref_actions = ref_logits.argmax(dim=-1)
    qry_actions = qry_logits.argmax(dim=-1)
    agg.top1_matches += int((ref_actions == qry_actions).sum().item())

    for k in k_values:
        k_clamped = min(k, n_act)
        topk_qry = qry_logits.topk(k_clamped, dim=-1).indices
        matches = (topk_qry == ref_actions.unsqueeze(-1)).any(dim=-1)
        agg.topk_matches[k] += int(matches.sum().item())

    p_ref = F.softmax(ref_logits, dim=-1)
    log_p_qry = F.log_softmax(qry_logits, dim=-1)
    kl_batch = F.kl_div(log_p_qry, p_ref, reduction="batchmean", log_target=False)
    agg.kl_sum += float(kl_batch.item()) * batch_n
    agg.mse_sum += float(F.mse_loss(qry_logits, ref_logits, reduction="sum").item())
    agg.mae_sum += float((qry_logits - ref_logits).abs().sum().item())
    agg.cos_sum += float(F.cosine_similarity(ref_logits, qry_logits, dim=-1).sum().item())
    agg.n_rows += batch_n


def _finalize_pairwise_agg(
    agg: _PairwiseChunkAgg,
    k_values: List[int],
    *,
    label: str,
    ref_lat_ms: float,
    qry_lat_ms: float,
    apply_thresholds: bool,
    thresholds: RecombinationThresholds,
) -> PairwiseComparison:
    """Turn chunk aggregates into a :class:`PairwiseComparison` (global means)."""
    n = max(agg.n_rows, 1)
    denom_elems = max(n * agg.n_actions, 1)
    top_k_agreements = {k: agg.topk_matches[k] / n for k in k_values}
    return PairwiseComparison(
        label=label,
        action_agreement=agg.top1_matches / n,
        top_k_agreements=top_k_agreements,
        kl_divergence=agg.kl_sum / n,
        mse=agg.mse_sum / denom_elems,
        mae=agg.mae_sum / denom_elems,
        mean_cosine_similarity=agg.cos_sum / n,
        reference_inference_ms=ref_lat_ms,
        query_inference_ms=qry_lat_ms,
        thresholds=thresholds,
        apply_thresholds=apply_thresholds,
    )


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass
class RecombinationThresholds:
    """Pass/fail thresholds for child-vs-parent pairwise comparisons.

    These are intentionally more lenient than :class:`ValidationThresholds`
    used in distillation, because crossover introduces deliberate weight
    mixing that relaxes behavioural fidelity expectations.

    Attributes
    ----------
    min_action_agreement:
        Minimum required top-1 action match rate (0–1) for the child against
        *each* parent.  Both ``child_vs_parent_a`` and ``child_vs_parent_b``
        must meet this threshold for :attr:`RecombinationReport.passed` to
        be ``True``.
    max_kl_divergence:
        Maximum allowed mean KL divergence KL(parent ‖ child) on
        temperature-1 softmax distributions over Q-values.  Applied to each
        child-vs-parent comparison independently.
    max_mse:
        Maximum allowed mean squared error between parent and child raw
        Q-value logits, averaged over states and actions.  Applied
        independently to each child-vs-parent comparison.
    min_cosine_similarity:
        Minimum mean cosine similarity between parent and child Q-value
        vectors across evaluation states.  Applied independently to each
        comparison.
    report_only:
        When ``True``, :attr:`RecombinationReport.passed` always returns
        ``True`` regardless of metric values.  Use this to emit reports
        without triggering CI pass/fail gates.
    """

    min_action_agreement: float = 0.70
    max_kl_divergence: float = 1.0
    max_mse: float = 5.0
    min_cosine_similarity: float = 0.70
    report_only: bool = False


# ---------------------------------------------------------------------------
# Pairwise comparison result
# ---------------------------------------------------------------------------


@dataclass
class PairwiseComparison:
    """Metrics from one pairwise model comparison (reference vs query).

    Attributes
    ----------
    label:
        Human-readable comparison label, e.g. ``"child_vs_parent_a"``.
    action_agreement:
        Top-1 argmax action match rate.
    top_k_agreements:
        Mapping ``{k: agreement}`` for each requested k.  JSON keys are
        strings (JSON does not support integer keys).
    kl_divergence:
        Mean KL(p_reference ‖ p_query).
    mse:
        Mean squared error on raw Q-logits.
    mae:
        Mean absolute error on raw Q-logits.
    mean_cosine_similarity:
        Mean cosine similarity between Q-value vectors.
    reference_inference_ms:
        Median single-sample latency of the reference model (ms).
    query_inference_ms:
        Median single-sample latency of the query model (ms).
    thresholds:
        The :class:`RecombinationThresholds` used to evaluate this pair.
        Only child-vs-parent pairs are threshold-checked; the optional
        parent-vs-parent baseline ignores thresholds.
    apply_thresholds:
        When ``False``, :attr:`passed` always returns ``True`` (used for
        the parent-vs-parent baseline which is informational only).
    """

    label: str
    action_agreement: float
    top_k_agreements: Dict[int, float]
    kl_divergence: float
    mse: float
    mae: float
    mean_cosine_similarity: float
    reference_inference_ms: float
    query_inference_ms: float
    thresholds: RecombinationThresholds
    apply_thresholds: bool = True

    @property
    def passed(self) -> bool:
        """``True`` when all threshold checks pass (or thresholds not applied)."""
        if not self.apply_thresholds or self.thresholds.report_only:
            return True
        t = self.thresholds
        return (
            self.action_agreement >= t.min_action_agreement
            and self.kl_divergence <= t.max_kl_divergence
            and self.mse <= t.max_mse
            and self.mean_cosine_similarity >= t.min_cosine_similarity
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of all comparison fields."""
        return {
            "label": self.label,
            "action_agreement": self.action_agreement,
            "top_k_agreements": {str(k): v for k, v in self.top_k_agreements.items()},
            "kl_divergence": self.kl_divergence,
            "mse": self.mse,
            "mae": self.mae,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "reference_inference_ms": self.reference_inference_ms,
            "query_inference_ms": self.query_inference_ms,
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


@dataclass
class RecombinationReport:
    """Multi-model comparison report for a crossover child vs its parents.

    Attributes
    ----------
    schema_version:
        Versioned schema identifier for forward-compatible JSON parsing.
        Currently :data:`REPORT_SCHEMA_VERSION` (``1.1`` adds ``model_formats``).
    torch_version:
        PyTorch version string (``torch.__version__``) at evaluation time.
    n_states:
        Number of evaluation states used.
    input_dim:
        Feature dimensionality of the evaluation states.
    states_source:
        Human-readable description of the state source, e.g. a file path
        or ``"synthetic_standard_normal"``.
    comparisons:
        Ordered mapping of comparison label → :class:`PairwiseComparison`.
        Always contains ``"child_vs_parent_a"`` and ``"child_vs_parent_b"``;
        optionally ``"parent_a_vs_parent_b"`` when requested.
    thresholds:
        The :class:`RecombinationThresholds` used for pass/fail evaluation.
    child_agrees_with_parent_a:
        Fraction of states where the child's top-1 action matches parent A.
        Convenience alias for ``comparisons["child_vs_parent_a"].action_agreement``.
    child_agrees_with_parent_b:
        Convenience alias for ``comparisons["child_vs_parent_b"].action_agreement``.
    oracle_agreement:
        Fraction of states where the child agrees with *at least one* parent
        (i.e. the child matches parent A **or** parent B).  ``None`` if
        oracle states were not computed (requires raw logit data retained in
        the evaluator).
    model_paths:
        Optional mapping of model role → checkpoint path string.  Populated
        from CLI flags when applicable.
    model_formats:
        How each checkpoint was loaded, e.g. ``"float_state_dict"`` (default
        ``BaseQNetwork`` state dict) or ``"quantized_full_model"`` (pickle from
        ``load_quantized_checkpoint``).  Keys: ``parent_a``, ``parent_b``,
        ``child``.  Omitted roles imply float state dict for backward compatibility.
    """

    schema_version: str
    torch_version: str
    n_states: int
    input_dim: int
    states_source: str
    comparisons: Dict[str, PairwiseComparison]
    thresholds: RecombinationThresholds
    child_agrees_with_parent_a: float
    child_agrees_with_parent_b: float
    oracle_agreement: Optional[float]
    model_paths: Dict[str, Optional[str]] = field(default_factory=dict)
    model_formats: Dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """``True`` when *all* threshold-checked comparisons pass."""
        if self.thresholds.report_only:
            return True
        return all(c.passed for c in self.comparisons.values() if c.apply_thresholds)

    def to_dict(self) -> Dict[str, Any]:
        """Return a versioned, JSON-serialisable dictionary."""
        comparisons_dict = {
            label: cmp.to_dict() for label, cmp in self.comparisons.items()
        }
        return {
            "schema_version": self.schema_version,
            "torch_version": self.torch_version,
            "states": {
                "n_states": self.n_states,
                "input_dim": self.input_dim,
                "source": self.states_source,
            },
            "model_paths": self.model_paths,
            "model_formats": self.model_formats,
            "comparisons": comparisons_dict,
            "summary": {
                "child_agrees_with_parent_a": self.child_agrees_with_parent_a,
                "child_agrees_with_parent_b": self.child_agrees_with_parent_b,
                "oracle_agreement": self.oracle_agreement,
            },
            "thresholds": {
                "min_action_agreement": self.thresholds.min_action_agreement,
                "max_kl_divergence": self.thresholds.max_kl_divergence,
                "max_mse": self.thresholds.max_mse,
                "min_cosine_similarity": self.thresholds.min_cosine_similarity,
                "report_only": self.thresholds.report_only,
            },
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RecombinationEvaluator:
    """Evaluate a crossover child against its parent networks.

    Composes pairwise comparisons — *child vs parent A*, *child vs parent B*,
    and an optional *parent A vs parent B* baseline — using the same
    forward-pass helpers that power :class:`StudentValidator` in
    :mod:`farm.core.decision.training.trainer_distill`.

    Parameters
    ----------
    parent_a:
        First parent ``nn.Module``.  Set to ``eval`` mode; weights unchanged.
    parent_b:
        Second parent ``nn.Module``.  Set to ``eval`` mode; weights unchanged.
    child:
        Child ``nn.Module`` produced by crossover (and optionally fine-tuned).
        Set to ``eval`` mode; weights unchanged.
    thresholds:
        :class:`RecombinationThresholds` controlling pass/fail criteria.
        Defaults to :class:`RecombinationThresholds` with conservative values.
    device:
        PyTorch device.  Defaults to CPU.
    """

    def __init__(
        self,
        parent_a: nn.Module,
        parent_b: nn.Module,
        child: nn.Module,
        thresholds: Optional[RecombinationThresholds] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.parent_a: nn.Module = parent_a
        self.parent_b: nn.Module = parent_b
        self.child: nn.Module = child
        self.thresholds: RecombinationThresholds = thresholds or RecombinationThresholds()
        self.device: torch.device = device or torch.device("cpu")

        for model in (self.parent_a, self.parent_b, self.child):
            model.to(self.device)
            model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        states: np.ndarray,
        *,
        include_parent_baseline: bool = False,
        k_values: Optional[List[int]] = None,
        n_latency_warmup: int = 5,
        n_latency_repeats: int = 50,
        states_source: str = "synthetic_standard_normal",
        eval_batch_size: Optional[int] = None,
        model_paths: Optional[Dict[str, Optional[str]]] = None,
        model_formats: Optional[Dict[str, str]] = None,
    ) -> RecombinationReport:
        """Run all comparisons and return a :class:`RecombinationReport`.

        Parameters
        ----------
        states:
            NumPy array of shape ``(N, input_dim)`` – held-out evaluation
            states shared across all comparisons.
        include_parent_baseline:
            When ``True``, also compute *parent A vs parent B* metrics.  This
            informational comparison is not subject to threshold checks.
        k_values:
            Values of *k* for top-k action agreement.  Defaults to
            ``[1, 2, 3]``.
        n_latency_warmup:
            Number of forward passes run before timing starts.
        n_latency_repeats:
            Number of timed forward passes; the median is reported.
        states_source:
            Human-readable description of the state source (e.g. a file path
            or ``"synthetic_standard_normal"``).  Stored in the report.
        eval_batch_size:
            Maximum number of states per forward pass.  ``None`` or non-positive
            values evaluate the full buffer in one batch (highest memory use).
            Smaller values reduce peak memory with identical reported means.
        model_paths:
            Optional mapping of role → checkpoint path to embed in the report
            (e.g. ``{"parent_a": "checkpoints/parent_A.pt", ...}``).
        model_formats:
            Optional mapping of role → load kind for JSON consumers:
            ``"float_state_dict"`` or ``"quantized_full_model"``.

        Returns
        -------
        RecombinationReport
        """
        if k_values is None:
            k_values = [1, 2, 3]

        states_arr = _validate_states(states)
        n_states = int(states_arr.shape[0])
        input_dim = int(states_arr.shape[1])

        if eval_batch_size is not None and eval_batch_size > 0:
            chunk_size = min(n_states, int(eval_batch_size))
        else:
            chunk_size = n_states

        agg_ca = _new_pairwise_agg(k_values)
        agg_cb = _new_pairwise_agg(k_values)
        agg_ab: Optional[_PairwiseChunkAgg] = (
            _new_pairwise_agg(k_values) if include_parent_baseline else None
        )

        oracle_matches = 0
        for start in range(0, n_states, chunk_size):
            end = min(start + chunk_size, n_states)
            chunk = torch.tensor(
                states_arr[start:end], dtype=torch.float32, device=self.device
            )
            logits_a = self._forward(self.parent_a, chunk)
            logits_b = self._forward(self.parent_b, chunk)
            logits_c = self._forward(self.child, chunk)

            actions_a = logits_a.argmax(dim=-1)
            actions_b = logits_b.argmax(dim=-1)
            actions_c = logits_c.argmax(dim=-1)
            oracle_matches += int(
                ((actions_c == actions_a) | (actions_c == actions_b)).sum().item()
            )

            _update_pairwise_agg(agg_ca, logits_a, logits_c, k_values)
            _update_pairwise_agg(agg_cb, logits_b, logits_c, k_values)
            if agg_ab is not None:
                _update_pairwise_agg(agg_ab, logits_a, logits_b, k_values)

        oracle_agreement = float(oracle_matches / max(n_states, 1))

        # Latency measurements (one representative state).
        single = torch.tensor(states_arr[:1], dtype=torch.float32, device=self.device)
        lat_a = self._measure_latency(self.parent_a, single, n_latency_warmup, n_latency_repeats)
        lat_b = self._measure_latency(self.parent_b, single, n_latency_warmup, n_latency_repeats)
        lat_c = self._measure_latency(self.child, single, n_latency_warmup, n_latency_repeats)

        cmp_ca = _finalize_pairwise_agg(
            agg_ca,
            k_values,
            label="child_vs_parent_a",
            ref_lat_ms=lat_a,
            qry_lat_ms=lat_c,
            apply_thresholds=True,
            thresholds=self.thresholds,
        )
        cmp_cb = _finalize_pairwise_agg(
            agg_cb,
            k_values,
            label="child_vs_parent_b",
            ref_lat_ms=lat_b,
            qry_lat_ms=lat_c,
            apply_thresholds=True,
            thresholds=self.thresholds,
        )

        comparisons: Dict[str, PairwiseComparison] = {
            "child_vs_parent_a": cmp_ca,
            "child_vs_parent_b": cmp_cb,
        }

        if include_parent_baseline and agg_ab is not None:
            cmp_ab = _finalize_pairwise_agg(
                agg_ab,
                k_values,
                label="parent_a_vs_parent_b",
                ref_lat_ms=lat_a,
                qry_lat_ms=lat_b,
                apply_thresholds=False,
                thresholds=self.thresholds,
            )
            comparisons["parent_a_vs_parent_b"] = cmp_ab

        logger.info(
            "recombination_evaluation_complete",
            n_states=n_states,
            child_vs_a_agreement=cmp_ca.action_agreement,
            child_vs_b_agreement=cmp_cb.action_agreement,
            oracle_agreement=oracle_agreement,
            passed=all(c.passed for c in comparisons.values() if c.apply_thresholds),
        )

        return RecombinationReport(
            schema_version=REPORT_SCHEMA_VERSION,
            torch_version=torch.__version__,
            n_states=n_states,
            input_dim=input_dim,
            states_source=states_source,
            comparisons=comparisons,
            thresholds=self.thresholds,
            child_agrees_with_parent_a=cmp_ca.action_agreement,
            child_agrees_with_parent_b=cmp_cb.action_agreement,
            oracle_agreement=oracle_agreement,
            model_paths=model_paths or {},
            model_formats=dict(model_formats or {}),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """Run *model* in eval mode and return logits."""
        model.eval()
        return model(tensor)

    def _pairwise(
        self,
        *,
        label: str,
        ref_logits: torch.Tensor,
        qry_logits: torch.Tensor,
        ref_actions: torch.Tensor,
        ref_lat_ms: float,
        qry_lat_ms: float,
        k_values: List[int],
        apply_thresholds: bool,
    ) -> PairwiseComparison:
        """Compute all metrics for one (reference, query) pair."""
        # Top-1 action agreement.
        qry_actions = qry_logits.argmax(dim=-1)
        action_agreement = float((ref_actions == qry_actions).float().mean().item())

        # Top-k agreements.
        n_actions = ref_logits.size(-1)
        top_k_agreements: Dict[int, float] = {}
        for k in k_values:
            k_clamped = min(k, n_actions)
            topk_qry = qry_logits.topk(k_clamped, dim=-1).indices
            matches = (topk_qry == ref_actions.unsqueeze(-1)).any(dim=-1)
            top_k_agreements[k] = float(matches.float().mean().item())

        # Output similarity.
        p_ref = F.softmax(ref_logits, dim=-1)
        log_p_qry = F.log_softmax(qry_logits, dim=-1)
        kl = float(
            F.kl_div(log_p_qry, p_ref, reduction="batchmean", log_target=False).item()
        )
        mse = float(F.mse_loss(qry_logits, ref_logits).item())
        mae = float((qry_logits - ref_logits).abs().mean().item())
        cos_sim = float(F.cosine_similarity(ref_logits, qry_logits, dim=-1).mean().item())

        return PairwiseComparison(
            label=label,
            action_agreement=action_agreement,
            top_k_agreements=top_k_agreements,
            kl_divergence=kl,
            mse=mse,
            mae=mae,
            mean_cosine_similarity=cos_sim,
            reference_inference_ms=ref_lat_ms,
            query_inference_ms=qry_lat_ms,
            thresholds=self.thresholds,
            apply_thresholds=apply_thresholds,
        )

    def _measure_latency(
        self,
        model: nn.Module,
        single: torch.Tensor,
        n_warmup: int,
        n_repeats: int,
    ) -> float:
        """Median single-sample inference latency in milliseconds."""
        if n_repeats <= 0:
            return 0.0
        model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                model(single)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        times: List[float] = []
        with torch.no_grad():
            for _ in range(n_repeats):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                model(single)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                times.append((time.perf_counter() - t0) * 1_000.0)
        return float(np.median(times))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_states(states: np.ndarray) -> np.ndarray:
    """Validate and return a float32 2-D array of states."""
    arr = np.asarray(states, dtype="float32")
    if arr.ndim != 2:
        raise ValueError(
            f"states must be a 2D array with shape (N, input_dim); got shape {arr.shape!r}"
        )
    if arr.shape[0] == 0:
        raise ValueError("states must be non-empty; got 0 rows.")
    return arr
