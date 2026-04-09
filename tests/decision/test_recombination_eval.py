"""Tests for RecombinationEvaluator, RecombinationThresholds, PairwiseComparison,
and RecombinationReport in farm/core/decision/training/recombination_eval.py.

Covers:
- RecombinationThresholds: defaults and custom values.
- PairwiseComparison: construction, passed property, to_dict schema.
- RecombinationReport: construction, passed property, JSON serialisation.
- RecombinationEvaluator: initialisation, per-metric correctness with known
  toy weights (identical-model golden case), multi-model evaluation,
  oracle agreement, optional parent baseline, and report-only mode.
- __init__ exports.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork
from farm.core.decision.training.quantize_ptq import load_quantized_checkpoint
from farm.core.decision.training.recombination_eval import (
    REPORT_SCHEMA_VERSION,
    PairwiseComparison,
    RecombinationEvaluator,
    RecombinationReport,
    RecombinationThresholds,
    _validate_states,
)

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
HIDDEN_SIZE = 16  # Tiny hidden for fast tests
N_STATES = 50
SEED = 0


def _make_model(seed: int = SEED) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_size=HIDDEN_SIZE,
    )


def _make_states(n: int = N_STATES, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _make_thresholds(**kwargs) -> RecombinationThresholds:
    return RecombinationThresholds(**kwargs)


def _make_pairwise(**kwargs) -> PairwiseComparison:
    defaults = dict(
        label="child_vs_parent_a",
        action_agreement=0.80,
        top_k_agreements={1: 0.80, 2: 0.90, 3: 0.95},
        kl_divergence=0.5,
        mse=2.0,
        mae=0.8,
        mean_cosine_similarity=0.85,
        reference_inference_ms=0.1,
        query_inference_ms=0.15,
        thresholds=RecombinationThresholds(),
        apply_thresholds=True,
    )
    defaults.update(kwargs)
    return PairwiseComparison(**defaults)


def _make_evaluator(
    seed_a: int = 0,
    seed_b: int = 1,
    seed_c: int = 2,
    thresholds: RecombinationThresholds | None = None,
) -> tuple[RecombinationEvaluator, np.ndarray]:
    parent_a = _make_model(seed_a)
    parent_b = _make_model(seed_b)
    child = _make_model(seed_c)
    states = _make_states()
    evaluator = RecombinationEvaluator(
        parent_a, parent_b, child, thresholds=thresholds
    )
    return evaluator, states


# ---------------------------------------------------------------------------
# RecombinationThresholds
# ---------------------------------------------------------------------------


class TestRecombinationThresholds:
    def test_defaults(self):
        t = RecombinationThresholds()
        assert t.min_action_agreement == pytest.approx(0.70)
        assert t.max_kl_divergence == pytest.approx(1.0)
        assert t.max_mse == pytest.approx(5.0)
        assert t.min_cosine_similarity == pytest.approx(0.70)
        assert t.report_only is False

    def test_custom_values(self):
        t = RecombinationThresholds(
            min_action_agreement=0.90,
            max_kl_divergence=0.2,
            max_mse=1.0,
            min_cosine_similarity=0.95,
            report_only=True,
        )
        assert t.min_action_agreement == pytest.approx(0.90)
        assert t.max_kl_divergence == pytest.approx(0.2)
        assert t.max_mse == pytest.approx(1.0)
        assert t.min_cosine_similarity == pytest.approx(0.95)
        assert t.report_only is True


# ---------------------------------------------------------------------------
# PairwiseComparison
# ---------------------------------------------------------------------------


class TestPairwiseComparison:
    def test_passed_all_thresholds_met(self):
        cmp = _make_pairwise()
        # Defaults: agreement=0.80 ≥ 0.70, kl=0.5 ≤ 1.0, mse=2.0 ≤ 5.0, cos=0.85 ≥ 0.70
        assert cmp.passed is True

    def test_passed_false_low_action_agreement(self):
        t = RecombinationThresholds(min_action_agreement=0.95)
        cmp = _make_pairwise(action_agreement=0.60, thresholds=t)
        assert cmp.passed is False

    def test_passed_false_high_kl(self):
        t = RecombinationThresholds(max_kl_divergence=0.1)
        cmp = _make_pairwise(kl_divergence=0.5, thresholds=t)
        assert cmp.passed is False

    def test_passed_false_high_mse(self):
        t = RecombinationThresholds(max_mse=0.5)
        cmp = _make_pairwise(mse=2.0, thresholds=t)
        assert cmp.passed is False

    def test_passed_false_low_cosine(self):
        t = RecombinationThresholds(min_cosine_similarity=0.99)
        cmp = _make_pairwise(mean_cosine_similarity=0.50, thresholds=t)
        assert cmp.passed is False

    def test_passed_true_when_apply_thresholds_false(self):
        """Baseline pair should always pass regardless of metric values."""
        t = RecombinationThresholds(min_action_agreement=1.0)
        cmp = _make_pairwise(
            action_agreement=0.0,
            thresholds=t,
            apply_thresholds=False,
        )
        assert cmp.passed is True

    def test_passed_true_when_report_only(self):
        t = RecombinationThresholds(min_action_agreement=1.0, report_only=True)
        cmp = _make_pairwise(action_agreement=0.0, thresholds=t)
        assert cmp.passed is True

    def test_to_dict_required_keys(self):
        cmp = _make_pairwise()
        d = cmp.to_dict()
        for key in (
            "label",
            "action_agreement",
            "top_k_agreements",
            "kl_divergence",
            "mse",
            "mae",
            "mean_cosine_similarity",
            "reference_inference_ms",
            "query_inference_ms",
            "passed",
        ):
            assert key in d, f"Missing key: {key!r}"

    def test_to_dict_json_string_keys_for_top_k(self):
        cmp = _make_pairwise(top_k_agreements={1: 0.8, 2: 0.9})
        d = cmp.to_dict()
        # JSON keys must be strings
        assert "1" in d["top_k_agreements"]
        assert "2" in d["top_k_agreements"]

    def test_to_dict_json_serialisable(self):
        cmp = _make_pairwise()
        json.dumps(cmp.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# RecombinationReport
# ---------------------------------------------------------------------------


def _make_report(**kwargs) -> RecombinationReport:
    cmp_a = _make_pairwise(label="child_vs_parent_a")
    cmp_b = _make_pairwise(label="child_vs_parent_b")
    defaults = dict(
        schema_version=REPORT_SCHEMA_VERSION,
        torch_version="2.0.0",
        n_states=N_STATES,
        input_dim=INPUT_DIM,
        states_source="synthetic_standard_normal",
        comparisons={"child_vs_parent_a": cmp_a, "child_vs_parent_b": cmp_b},
        thresholds=RecombinationThresholds(),
        child_agrees_with_parent_a=0.80,
        child_agrees_with_parent_b=0.75,
        oracle_agreement=0.90,
        model_paths={},
        model_formats={},
    )
    defaults.update(kwargs)
    return RecombinationReport(**defaults)


class TestRecombinationReport:
    def test_passed_when_all_comparisons_pass(self):
        report = _make_report()
        assert report.passed is True

    def test_passed_false_when_one_comparison_fails(self):
        t = RecombinationThresholds(min_action_agreement=0.99)
        cmp_a = _make_pairwise(label="child_vs_parent_a", action_agreement=0.50, thresholds=t)
        cmp_b = _make_pairwise(label="child_vs_parent_b", action_agreement=0.50, thresholds=t)
        report = _make_report(
            comparisons={"child_vs_parent_a": cmp_a, "child_vs_parent_b": cmp_b},
            thresholds=t,
        )
        assert report.passed is False

    def test_passed_true_when_report_only(self):
        t = RecombinationThresholds(min_action_agreement=1.0, report_only=True)
        cmp_a = _make_pairwise(label="child_vs_parent_a", action_agreement=0.0, thresholds=t)
        cmp_b = _make_pairwise(label="child_vs_parent_b", action_agreement=0.0, thresholds=t)
        report = _make_report(
            comparisons={"child_vs_parent_a": cmp_a, "child_vs_parent_b": cmp_b},
            thresholds=t,
        )
        assert report.passed is True

    def test_to_dict_top_level_keys(self):
        report = _make_report()
        d = report.to_dict()
        for key in (
            "schema_version",
            "torch_version",
            "states",
            "model_paths",
            "model_formats",
            "comparisons",
            "summary",
            "thresholds",
            "passed",
        ):
            assert key in d, f"Missing top-level key: {key!r}"

    def test_to_dict_states_keys(self):
        report = _make_report()
        states = report.to_dict()["states"]
        assert "n_states" in states
        assert "input_dim" in states
        assert "source" in states

    def test_to_dict_summary_keys(self):
        report = _make_report()
        summary = report.to_dict()["summary"]
        assert "child_agrees_with_parent_a" in summary
        assert "child_agrees_with_parent_b" in summary
        assert "oracle_agreement" in summary

    def test_to_dict_thresholds_keys(self):
        report = _make_report()
        t = report.to_dict()["thresholds"]
        assert "min_action_agreement" in t
        assert "max_kl_divergence" in t
        assert "max_mse" in t
        assert "min_cosine_similarity" in t
        assert "report_only" in t

    def test_to_dict_json_serialisable(self):
        report = _make_report()
        json.dumps(report.to_dict())  # must not raise

    def test_schema_version_present(self):
        report = _make_report()
        d = report.to_dict()
        assert d["schema_version"] == REPORT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# RecombinationEvaluator – initialisation
# ---------------------------------------------------------------------------


class TestRecombinationEvaluatorInit:
    def test_default_thresholds(self):
        evaluator, _ = _make_evaluator()
        assert isinstance(evaluator.thresholds, RecombinationThresholds)

    def test_custom_thresholds(self):
        t = RecombinationThresholds(min_action_agreement=0.90)
        evaluator, _ = _make_evaluator(thresholds=t)
        assert evaluator.thresholds.min_action_agreement == pytest.approx(0.90)

    def test_default_device_is_cpu(self):
        evaluator, _ = _make_evaluator()
        assert evaluator.device.type == "cpu"

    def test_models_in_eval_mode(self):
        evaluator, _ = _make_evaluator()
        assert not evaluator.parent_a.training
        assert not evaluator.parent_b.training
        assert not evaluator.child.training


# ---------------------------------------------------------------------------
# RecombinationEvaluator – evaluate() basics
# ---------------------------------------------------------------------------


class TestRecombinationEvaluatorEvaluate:
    def test_returns_recombination_report(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert isinstance(report, RecombinationReport)

    def test_comparisons_contain_required_keys(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert "child_vs_parent_a" in report.comparisons
        assert "child_vs_parent_b" in report.comparisons

    def test_no_parent_baseline_by_default(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert "parent_a_vs_parent_b" not in report.comparisons

    def test_parent_baseline_included_when_requested(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(
            states,
            include_parent_baseline=True,
            n_latency_warmup=0,
            n_latency_repeats=3,
        )
        assert "parent_a_vs_parent_b" in report.comparisons

    def test_parent_baseline_not_threshold_checked(self):
        """Parent-vs-parent baseline should always report passed=True."""
        t = RecombinationThresholds(min_action_agreement=1.0)
        evaluator, states = _make_evaluator(thresholds=t)
        report = evaluator.evaluate(
            states,
            include_parent_baseline=True,
            n_latency_warmup=0,
            n_latency_repeats=3,
        )
        baseline = report.comparisons["parent_a_vs_parent_b"]
        assert baseline.passed is True
        assert baseline.apply_thresholds is False

    def test_n_states_matches_input(self):
        evaluator, _ = _make_evaluator()
        states = _make_states(n=37)
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.n_states == 37

    def test_input_dim_in_report(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.input_dim == INPUT_DIM

    def test_oracle_agreement_not_none(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.oracle_agreement is not None

    def test_oracle_agreement_between_zero_and_one(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert 0.0 <= report.oracle_agreement <= 1.0

    def test_schema_version_in_report(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.schema_version == REPORT_SCHEMA_VERSION

    def test_torch_version_in_report(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.torch_version == torch.__version__

    def test_model_paths_embedded(self):
        evaluator, states = _make_evaluator()
        paths = {"parent_a": "p_a.pt", "parent_b": "p_b.pt", "child": "c.pt"}
        report = evaluator.evaluate(
            states,
            n_latency_warmup=0,
            n_latency_repeats=3,
            model_paths=paths,
        )
        assert report.model_paths == paths

    def test_eval_batch_size_chunking_matches_single_batch(self):
        evaluator, states = _make_evaluator()
        r_full = evaluator.evaluate(
            states, n_latency_warmup=0, n_latency_repeats=2, eval_batch_size=None
        )
        r_chunk = evaluator.evaluate(
            states, n_latency_warmup=0, n_latency_repeats=2, eval_batch_size=11
        )
        for label in ("child_vs_parent_a", "child_vs_parent_b"):
            c1 = r_full.comparisons[label]
            c2 = r_chunk.comparisons[label]
            assert c1.action_agreement == pytest.approx(c2.action_agreement)
            assert c1.kl_divergence == pytest.approx(c2.kl_divergence, rel=1e-5, abs=1e-7)
            assert c1.mse == pytest.approx(c2.mse, rel=1e-5, abs=1e-7)
            assert c1.mean_cosine_similarity == pytest.approx(
                c2.mean_cosine_similarity, rel=1e-5, abs=1e-7
            )
        assert r_full.oracle_agreement == pytest.approx(r_chunk.oracle_agreement)

    def test_eval_batch_size_with_parent_baseline_matches(self):
        evaluator, states = _make_evaluator()
        r_full = evaluator.evaluate(
            states,
            include_parent_baseline=True,
            n_latency_warmup=0,
            n_latency_repeats=2,
            eval_batch_size=None,
        )
        r_chunk = evaluator.evaluate(
            states,
            include_parent_baseline=True,
            n_latency_warmup=0,
            n_latency_repeats=2,
            eval_batch_size=13,
        )
        b1 = r_full.comparisons["parent_a_vs_parent_b"]
        b2 = r_chunk.comparisons["parent_a_vs_parent_b"]
        assert b1.action_agreement == pytest.approx(b2.action_agreement)
        assert b1.kl_divergence == pytest.approx(b2.kl_divergence, rel=1e-5, abs=1e-7)

    def test_raises_on_empty_states(self):
        evaluator, _ = _make_evaluator()
        with pytest.raises(ValueError, match="non-empty"):
            evaluator.evaluate(
                np.empty((0, INPUT_DIM), dtype="float32"),
                n_latency_warmup=0,
                n_latency_repeats=3,
            )

    def test_raises_on_1d_states(self):
        evaluator, _ = _make_evaluator()
        with pytest.raises(ValueError, match="2D"):
            evaluator.evaluate(
                np.ones((INPUT_DIM,), dtype="float32"),
                n_latency_warmup=0,
                n_latency_repeats=3,
            )


# ---------------------------------------------------------------------------
# RecombinationEvaluator – golden case (identical models)
# ---------------------------------------------------------------------------


class TestRecombinationEvaluatorIdenticalModels:
    """When child equals parent A, action agreement with A should be 1.0."""

    def test_child_identical_to_parent_a_agreement_one(self):
        parent_a = _make_model(seed=7)
        parent_b = _make_model(seed=8)
        # Child starts identical to parent A
        child = _make_model(seed=7)

        states = _make_states(n=100, seed=5)
        evaluator = RecombinationEvaluator(parent_a, parent_b, child)
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)

        assert report.child_agrees_with_parent_a == pytest.approx(1.0), (
            f"Expected 1.0 agreement when child ≡ parent_a, "
            f"got {report.child_agrees_with_parent_a:.4f}"
        )

    def test_child_identical_to_parent_a_kl_zero(self):
        parent_a = _make_model(seed=7)
        parent_b = _make_model(seed=8)
        child = _make_model(seed=7)

        states = _make_states(n=100, seed=5)
        evaluator = RecombinationEvaluator(parent_a, parent_b, child)
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)

        cmp = report.comparisons["child_vs_parent_a"]
        assert cmp.kl_divergence == pytest.approx(0.0, abs=1e-5)
        assert cmp.mse == pytest.approx(0.0, abs=1e-5)
        assert cmp.mean_cosine_similarity == pytest.approx(1.0, abs=1e-5)

    def test_all_identical_oracle_agreement_one(self):
        model = _make_model(seed=9)
        states = _make_states(n=50, seed=6)
        evaluator = RecombinationEvaluator(model, model, model)
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)

        assert report.oracle_agreement == pytest.approx(1.0)
        assert report.child_agrees_with_parent_a == pytest.approx(1.0)
        assert report.child_agrees_with_parent_b == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RecombinationEvaluator – metrics correctness
# ---------------------------------------------------------------------------


class TestRecombinationEvaluatorMetrics:
    def test_action_agreement_between_zero_and_one(self):
        evaluator, states = _make_evaluator(seed_a=10, seed_b=11, seed_c=12)
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        for key in ("child_vs_parent_a", "child_vs_parent_b"):
            cmp = report.comparisons[key]
            assert 0.0 <= cmp.action_agreement <= 1.0

    def test_kl_divergence_non_negative(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        for cmp in report.comparisons.values():
            assert cmp.kl_divergence >= 0.0

    def test_mse_non_negative(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        for cmp in report.comparisons.values():
            assert cmp.mse >= 0.0

    def test_mae_non_negative(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        for cmp in report.comparisons.values():
            assert cmp.mae >= 0.0

    def test_cosine_similarity_in_range(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        for cmp in report.comparisons.values():
            assert -1.0 <= cmp.mean_cosine_similarity <= 1.0

    def test_top_k_agreements_present(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(
            states,
            k_values=[1, 2, 3],
            n_latency_warmup=0,
            n_latency_repeats=3,
        )
        for cmp in report.comparisons.values():
            assert len(cmp.top_k_agreements) == 3

    def test_top_1_agreement_matches_action_agreement(self):
        """Top-k with k=1 should equal top-1 action agreement."""
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(
            states,
            k_values=[1, 2],
            n_latency_warmup=0,
            n_latency_repeats=3,
        )
        for cmp in report.comparisons.values():
            assert cmp.top_k_agreements[1] == pytest.approx(
                cmp.action_agreement, abs=1e-6
            )

    def test_top_k_agreement_non_decreasing(self):
        """Top-k agreement should be non-decreasing in k."""
        evaluator, states = _make_evaluator(seed_a=3, seed_b=4, seed_c=5)
        report = evaluator.evaluate(
            states,
            k_values=[1, 2, 3],
            n_latency_warmup=0,
            n_latency_repeats=3,
        )
        for cmp in report.comparisons.values():
            agreements = [cmp.top_k_agreements[k] for k in sorted(cmp.top_k_agreements)]
            for i in range(len(agreements) - 1):
                assert agreements[i] <= agreements[i + 1] + 1e-6

    def test_latency_positive_when_measured(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=1, n_latency_repeats=5)
        for cmp in report.comparisons.values():
            assert cmp.reference_inference_ms > 0.0
            assert cmp.query_inference_ms > 0.0

    def test_oracle_agreement_gte_max_individual_agreement(self):
        """Oracle ≥ max(agree_with_a, agree_with_b) since it's the union."""
        evaluator, states = _make_evaluator(seed_a=20, seed_b=21, seed_c=22)
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        max_individual = max(
            report.child_agrees_with_parent_a, report.child_agrees_with_parent_b
        )
        assert report.oracle_agreement >= max_individual - 1e-6


# ---------------------------------------------------------------------------
# RecombinationEvaluator – report-only mode
# ---------------------------------------------------------------------------


class TestRecombinationEvaluatorReportOnly:
    def test_report_only_always_passes(self):
        t = RecombinationThresholds(min_action_agreement=1.0, report_only=True)
        evaluator, states = _make_evaluator(
            seed_a=30, seed_b=31, seed_c=32, thresholds=t
        )
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.passed is True


# ---------------------------------------------------------------------------
# RecombinationEvaluator – JSON round-trip
# ---------------------------------------------------------------------------


class TestRecombinationEvaluatorJsonRoundTrip:
    def test_to_dict_is_json_serialisable(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        serialised = json.dumps(report.to_dict())
        parsed = json.loads(serialised)
        assert "comparisons" in parsed
        assert "child_vs_parent_a" in parsed["comparisons"]
        assert "child_vs_parent_b" in parsed["comparisons"]

    def test_passed_in_dict_matches_property(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.to_dict()["passed"] == report.passed

    def test_child_agreement_summary_in_dict(self):
        evaluator, states = _make_evaluator()
        report = evaluator.evaluate(states, n_latency_warmup=0, n_latency_repeats=3)
        summary = report.to_dict()["summary"]
        assert "child_agrees_with_parent_a" in summary
        assert "child_agrees_with_parent_b" in summary
        assert "oracle_agreement" in summary


# ---------------------------------------------------------------------------
# Quantized child (full-model pickle, CPU)
# ---------------------------------------------------------------------------


class TestRecombinationEvaluatorQuantizedChild:
    def test_quantized_child_evaluation_smoke(self):
        try:
            import torch.ao.quantization as aoq
        except ImportError:  # pragma: no cover
            import torch.quantization as aoq  # type: ignore[no-redef]

        parent_a = _make_model(0)
        parent_b = _make_model(1)
        child = _make_model(2)
        child.eval()
        qchild = aoq.quantize_dynamic(copy.deepcopy(child), {nn.Linear}, dtype=torch.qint8)
        qchild.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "qchild.pt")
            torch.save(qchild, path)
            loaded, _meta = load_quantized_checkpoint(path)
        states = _make_states(24)
        evaluator = RecombinationEvaluator(
            parent_a, parent_b, loaded, device=torch.device("cpu")
        )
        report = evaluator.evaluate(
            states,
            n_latency_warmup=0,
            n_latency_repeats=2,
            model_formats={
                "parent_a": "float_state_dict",
                "parent_b": "float_state_dict",
                "child": "quantized_full_model",
            },
        )
        assert "child_vs_parent_a" in report.comparisons
        d = report.to_dict()
        assert d["model_formats"]["child"] == "quantized_full_model"
        assert 0.0 <= d["summary"]["oracle_agreement"] <= 1.0


# ---------------------------------------------------------------------------
# _validate_states helper
# ---------------------------------------------------------------------------


class TestValidateStates:
    def test_valid_2d_array(self):
        arr = np.ones((10, 8), dtype="float32")
        result = _validate_states(arr)
        assert result.shape == (10, 8)

    def test_raises_on_1d(self):
        with pytest.raises(ValueError, match="2D"):
            _validate_states(np.ones((8,), dtype="float32"))

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_states(np.empty((0, 8), dtype="float32"))

    def test_converts_to_float32(self):
        arr = np.ones((5, 4), dtype="float64")
        result = _validate_states(arr)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Module-level __init__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_imports_from_training_init(self):
        from farm.core.decision.training import (  # noqa: F401
            PairwiseComparison,
            RecombinationEvaluator,
            RecombinationReport,
            RecombinationThresholds,
            REPORT_SCHEMA_VERSION,
        )

        assert issubclass(PairwiseComparison, object)
        assert issubclass(RecombinationEvaluator, object)
        assert issubclass(RecombinationReport, object)
        assert issubclass(RecombinationThresholds, object)
        assert isinstance(REPORT_SCHEMA_VERSION, str)
        assert REPORT_SCHEMA_VERSION == "1.1"
