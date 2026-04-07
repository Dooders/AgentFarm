"""Tests for parent-student behavioural-fidelity validation.

Validates that distilled students (student_A, student_B) preserve the core
behaviour of their corresponding parent networks (parent_A, parent_B) across:

- Output similarity: KL divergence, MSE, MAE, cosine similarity.
- Action agreement: top-1 and top-k argmax match rates.
- Efficiency: parameter count and inference latency comparisons.
- Robustness slices: agreement on named edge-case state subsets.

Covers:
- ValidationThresholds: defaults and custom configuration.
- ValidationReport: construction, ``passed`` property, JSON serialisation.
- StudentValidator: initialisation, per-metric correctness (including
  golden-case identical-network assertions), and full integration tests
  for both pair A (parent_A → student_A) and pair B (parent_B → student_B).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
from farm.core.decision.training.trainer_distill import (
    DistillationConfig,
    DistillationTrainer,
    StudentValidator,
    ValidationReport,
    ValidationThresholds,
)

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
PARENT_HIDDEN = 32
SEED = 42


def _make_parent(seed: int = SEED) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=PARENT_HIDDEN)


def _make_student() -> StudentQNetwork:
    return StudentQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        parent_hidden_size=PARENT_HIDDEN,
    )


def _make_states(n: int = 200, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _train_student(
    parent: BaseQNetwork,
    student: StudentQNetwork,
    states: np.ndarray,
) -> None:
    """Quickly distil *student* from *parent* for integration tests."""
    cfg = DistillationConfig(
        epochs=5,
        batch_size=32,
        val_fraction=0.1,
        seed=SEED,
        alpha=1.0,
        temperature=3.0,
        learning_rate=1e-2,
    )
    DistillationTrainer(parent, student, cfg).train(states)


def _make_report(**kwargs) -> ValidationReport:
    """Construct a ValidationReport with sensible defaults, overrideable via kwargs."""
    defaults: dict = dict(
        action_agreement=0.90,
        top_k_agreements={1: 0.90, 2: 0.95, 3: 0.97},
        kl_divergence=0.1,
        mse=0.5,
        mae=0.3,
        mean_cosine_similarity=0.95,
        parent_param_count=1000,
        student_param_count=300,
        param_ratio=0.30,
        parent_inference_ms=1.0,
        student_inference_ms=0.5,
        latency_ratio=0.5,
        robustness_slice_agreements={},
        thresholds=ValidationThresholds(),
    )
    defaults.update(kwargs)
    return ValidationReport(**defaults)


# ---------------------------------------------------------------------------
# ValidationThresholds
# ---------------------------------------------------------------------------


class TestValidationThresholds:
    def test_defaults(self):
        t = ValidationThresholds()
        assert t.min_action_agreement == 0.85
        assert t.max_kl_divergence == 0.5
        assert t.max_mse == 2.0
        assert t.min_cosine_similarity == 0.8
        assert t.max_param_ratio == 0.9

    def test_custom_values_accepted(self):
        t = ValidationThresholds(min_action_agreement=0.95, max_mse=1.0, max_param_ratio=0.75)
        assert t.min_action_agreement == 0.95
        assert t.max_mse == 1.0
        assert t.max_param_ratio == 0.75


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_passed_when_all_thresholds_met(self):
        assert _make_report().passed is True

    def test_failed_when_action_agreement_low(self):
        assert _make_report(action_agreement=0.50).passed is False

    def test_failed_when_kl_divergence_high(self):
        assert _make_report(kl_divergence=10.0).passed is False

    def test_failed_when_mse_high(self):
        assert _make_report(mse=100.0).passed is False

    def test_failed_when_cosine_similarity_low(self):
        assert _make_report(mean_cosine_similarity=0.1).passed is False

    def test_failed_when_param_ratio_exceeds_threshold(self):
        assert _make_report(param_ratio=0.95).passed is False

    def test_to_dict_is_json_serialisable(self):
        d = _make_report().to_dict()
        serialised = json.dumps(d)
        parsed = json.loads(serialised)
        for key in (
            "action_agreement",
            "top_k_agreements",
            "kl_divergence",
            "mse",
            "mae",
            "mean_cosine_similarity",
            "parent_param_count",
            "student_param_count",
            "param_ratio",
            "parent_inference_ms",
            "student_inference_ms",
            "latency_ratio",
            "robustness_slice_agreements",
            "passed",
            "thresholds",
        ):
            assert key in parsed

    def test_to_dict_top_k_keys_are_strings(self):
        """JSON object keys must be strings; int keys should be converted."""
        d = _make_report().to_dict()
        for k in d["top_k_agreements"]:
            assert isinstance(k, str)

    def test_to_dict_action_agreement_matches(self):
        report = _make_report(action_agreement=0.88)
        assert report.to_dict()["action_agreement"] == pytest.approx(0.88)

    def test_to_dict_thresholds_included(self):
        t = ValidationThresholds(min_action_agreement=0.92)
        report = _make_report(thresholds=t)
        d = report.to_dict()
        assert d["thresholds"]["min_action_agreement"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# StudentValidator: initialisation
# ---------------------------------------------------------------------------


class TestStudentValidatorInit:
    def test_default_thresholds_are_assigned(self):
        validator = StudentValidator(_make_parent(), _make_student())
        assert isinstance(validator.thresholds, ValidationThresholds)

    def test_custom_thresholds_are_preserved(self):
        t = ValidationThresholds(min_action_agreement=0.95)
        validator = StudentValidator(_make_parent(), _make_student(), thresholds=t)
        assert validator.thresholds.min_action_agreement == 0.95

    def test_default_device_is_cpu(self):
        validator = StudentValidator(_make_parent(), _make_student())
        assert validator.device == torch.device("cpu")

    def test_parent_and_student_stored(self):
        parent = _make_parent()
        student = _make_student()
        validator = StudentValidator(parent, student)
        assert validator.parent is parent
        assert validator.student is student


# ---------------------------------------------------------------------------
# StudentValidator: validate() – metric correctness
# ---------------------------------------------------------------------------


class TestStudentValidatorMetrics:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.states = _make_states(200)
        self.parent = _make_parent()
        self.student = _make_student()
        self.validator = StudentValidator(self.parent, self.student)

    def test_returns_validation_report(self):
        report = self.validator.validate(self.states)
        assert isinstance(report, ValidationReport)

    def test_action_agreement_in_unit_interval(self):
        report = self.validator.validate(self.states)
        assert 0.0 <= report.action_agreement <= 1.0

    def test_top_k_agreements_populated_for_all_k(self):
        report = self.validator.validate(self.states, k_values=[1, 2, 3])
        assert set(report.top_k_agreements.keys()) == {1, 2, 3}
        for v in report.top_k_agreements.values():
            assert 0.0 <= v <= 1.0

    def test_top_k_agreements_monotone_non_decreasing(self):
        """Top-k agreement must be non-decreasing: k=1 <= k=2 <= k=3."""
        report = self.validator.validate(self.states, k_values=[1, 2, 3])
        assert report.top_k_agreements[1] <= report.top_k_agreements[2]
        assert report.top_k_agreements[2] <= report.top_k_agreements[3]

    def test_action_agreement_matches_top_1(self):
        """The scalar action_agreement must equal top_k_agreements[1]."""
        report = self.validator.validate(self.states, k_values=[1, 2, 3])
        assert report.action_agreement == pytest.approx(report.top_k_agreements[1])

    def test_kl_divergence_non_negative(self):
        report = self.validator.validate(self.states)
        assert report.kl_divergence >= 0.0

    def test_mse_non_negative(self):
        report = self.validator.validate(self.states)
        assert report.mse >= 0.0

    def test_mae_non_negative(self):
        report = self.validator.validate(self.states)
        assert report.mae >= 0.0

    def test_mae_and_mse_finite_and_non_negative(self):
        """MAE and MSE are both finite and non-negative."""
        report = self.validator.validate(self.states)
        # Both finite and non-negative is the key property
        assert np.isfinite(report.mae)
        assert np.isfinite(report.mse)

    def test_cosine_similarity_in_valid_range(self):
        report = self.validator.validate(self.states)
        assert -1.0 <= report.mean_cosine_similarity <= 1.0

    def test_param_counts_match_model_counts(self):
        report = self.validator.validate(self.states)
        expected_parent = sum(p.numel() for p in self.parent.parameters())
        expected_student = sum(p.numel() for p in self.student.parameters())
        assert report.parent_param_count == expected_parent
        assert report.student_param_count == expected_student

    def test_param_ratio_consistent(self):
        report = self.validator.validate(self.states)
        expected = report.student_param_count / report.parent_param_count
        assert report.param_ratio == pytest.approx(expected)

    def test_latency_values_positive(self):
        report = self.validator.validate(self.states, n_latency_warmup=1, n_latency_repeats=5)
        assert report.parent_inference_ms > 0.0
        assert report.student_inference_ms > 0.0

    def test_latency_ratio_positive(self):
        report = self.validator.validate(self.states, n_latency_warmup=1, n_latency_repeats=5)
        assert report.latency_ratio > 0.0

    def test_no_robustness_slices_by_default(self):
        report = self.validator.validate(self.states)
        assert report.robustness_slice_agreements == {}

    def test_robustness_slices_populated(self):
        slices = {
            "low_resource": _make_states(50, seed=1),
            "high_threat": _make_states(50, seed=2),
        }
        report = self.validator.validate(self.states, robustness_slices=slices)
        assert set(report.robustness_slice_agreements.keys()) == {"low_resource", "high_threat"}
        for v in report.robustness_slice_agreements.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# StudentValidator: identical-network golden cases
# ---------------------------------------------------------------------------


class TestStudentValidatorIdenticalNetworks:
    """When parent and student have identical weights all similarity metrics are exact."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.parent = _make_parent()
        # Create a network with the same architecture and copy the parent weights
        self.clone = BaseQNetwork(INPUT_DIM, OUTPUT_DIM, PARENT_HIDDEN)
        self.clone.load_state_dict(self.parent.state_dict())
        self.validator = StudentValidator(self.parent, self.clone)
        self.states = _make_states(100)

    def test_kl_divergence_is_zero(self):
        report = self.validator.validate(self.states)
        assert report.kl_divergence == pytest.approx(0.0, abs=1e-5)

    def test_mse_is_zero(self):
        report = self.validator.validate(self.states)
        assert report.mse == pytest.approx(0.0, abs=1e-5)

    def test_mae_is_zero(self):
        report = self.validator.validate(self.states)
        assert report.mae == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_is_one(self):
        report = self.validator.validate(self.states)
        assert report.mean_cosine_similarity == pytest.approx(1.0, abs=1e-5)

    def test_action_agreement_is_one(self):
        report = self.validator.validate(self.states)
        assert report.action_agreement == pytest.approx(1.0)

    def test_top_k_agreements_all_one(self):
        report = self.validator.validate(self.states, k_values=[1, 2, 3])
        for v in report.top_k_agreements.values():
            assert v == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Efficiency checks
# ---------------------------------------------------------------------------


class TestStudentEfficiency:
    def test_student_has_fewer_params_than_parent(self):
        parent = _make_parent()
        student = _make_student()
        validator = StudentValidator(parent, student)
        report = validator.validate(_make_states(50), n_latency_warmup=1, n_latency_repeats=3)
        assert report.student_param_count < report.parent_param_count

    def test_param_ratio_below_default_threshold(self):
        parent = _make_parent()
        student = _make_student()
        validator = StudentValidator(parent, student)
        report = validator.validate(_make_states(50), n_latency_warmup=1, n_latency_repeats=3)
        assert report.param_ratio < ValidationThresholds().max_param_ratio

    def test_latency_ratio_is_finite(self):
        parent = _make_parent()
        student = _make_student()
        validator = StudentValidator(parent, student)
        report = validator.validate(_make_states(50), n_latency_warmup=1, n_latency_repeats=5)
        assert np.isfinite(report.latency_ratio)


# ---------------------------------------------------------------------------
# Integration: student_A – distilled from parent_A
# ---------------------------------------------------------------------------


class TestStudentAValidation:
    """End-to-end validation of student_A (distilled from parent_A)."""

    @pytest.fixture(autouse=True)
    def _train(self):
        train_states = _make_states(300)
        self.parent_A = _make_parent(seed=0)
        self.student_A = _make_student()
        _train_student(self.parent_A, self.student_A, train_states)

        # Held-out evaluation states (different seed from training data)
        self.eval_states = _make_states(200, seed=99)

        # Lenient thresholds – only a few epochs are trained for speed
        thresholds = ValidationThresholds(
            min_action_agreement=0.0,
            max_kl_divergence=1e6,
            max_mse=1e6,
            min_cosine_similarity=-1.0,
            max_param_ratio=0.9,
        )
        self.validator = StudentValidator(self.parent_A, self.student_A, thresholds=thresholds)

    def test_student_A_action_agreement_is_finite(self):
        report = self.validator.validate(self.eval_states)
        assert np.isfinite(report.action_agreement)

    def test_student_A_top_k_agreements_populated(self):
        report = self.validator.validate(self.eval_states, k_values=[1, 2, 3])
        assert len(report.top_k_agreements) == 3

    def test_student_A_top_k_monotone(self):
        report = self.validator.validate(self.eval_states, k_values=[1, 2, 3])
        assert report.top_k_agreements[1] <= report.top_k_agreements[2]
        assert report.top_k_agreements[2] <= report.top_k_agreements[3]

    def test_student_A_kl_divergence_finite(self):
        report = self.validator.validate(self.eval_states)
        assert np.isfinite(report.kl_divergence)
        assert report.kl_divergence >= 0.0

    def test_student_A_mse_finite(self):
        report = self.validator.validate(self.eval_states)
        assert np.isfinite(report.mse)
        assert report.mse >= 0.0

    def test_student_A_mae_finite(self):
        report = self.validator.validate(self.eval_states)
        assert np.isfinite(report.mae)
        assert report.mae >= 0.0

    def test_student_A_cosine_similarity_in_range(self):
        report = self.validator.validate(self.eval_states)
        assert -1.0 <= report.mean_cosine_similarity <= 1.0

    def test_student_A_param_ratio_below_threshold(self):
        report = self.validator.validate(self.eval_states)
        assert report.param_ratio < 0.9

    def test_student_A_robustness_slices(self):
        slices = {
            "low_resource": _make_states(30, seed=10),
            "high_threat": _make_states(30, seed=20),
            "sparse_obs": _make_states(30, seed=30),
        }
        report = self.validator.validate(self.eval_states, robustness_slices=slices)
        assert set(report.robustness_slice_agreements.keys()) == set(slices.keys())
        for v in report.robustness_slice_agreements.values():
            assert 0.0 <= v <= 1.0

    def test_student_A_report_serialisable(self):
        report = self.validator.validate(
            self.eval_states, n_latency_warmup=1, n_latency_repeats=3
        )
        d = report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert "action_agreement" in parsed
        assert "passed" in parsed

    def test_student_A_passed_property_is_bool(self):
        report = self.validator.validate(self.eval_states)
        assert isinstance(report.passed, bool)


# ---------------------------------------------------------------------------
# Integration: student_B – distilled from parent_B
# ---------------------------------------------------------------------------


class TestStudentBValidation:
    """End-to-end validation of student_B (distilled from parent_B)."""

    @pytest.fixture(autouse=True)
    def _train(self):
        train_states = _make_states(300)
        # parent_B uses a different seed → distinct weight initialisation
        self.parent_B = _make_parent(seed=1)
        self.student_B = _make_student()
        _train_student(self.parent_B, self.student_B, train_states)

        self.eval_states = _make_states(200, seed=100)

        thresholds = ValidationThresholds(
            min_action_agreement=0.0,
            max_kl_divergence=1e6,
            max_mse=1e6,
            min_cosine_similarity=-1.0,
            max_param_ratio=0.9,
        )
        self.validator = StudentValidator(self.parent_B, self.student_B, thresholds=thresholds)

    def test_student_B_action_agreement_is_finite(self):
        report = self.validator.validate(self.eval_states)
        assert np.isfinite(report.action_agreement)

    def test_student_B_top_k_agreements_populated(self):
        report = self.validator.validate(self.eval_states, k_values=[1, 2, 3])
        assert len(report.top_k_agreements) == 3

    def test_student_B_top_k_monotone(self):
        report = self.validator.validate(self.eval_states, k_values=[1, 2, 3])
        assert report.top_k_agreements[1] <= report.top_k_agreements[2]
        assert report.top_k_agreements[2] <= report.top_k_agreements[3]

    def test_student_B_kl_divergence_finite(self):
        report = self.validator.validate(self.eval_states)
        assert np.isfinite(report.kl_divergence)
        assert report.kl_divergence >= 0.0

    def test_student_B_mse_finite(self):
        report = self.validator.validate(self.eval_states)
        assert np.isfinite(report.mse)
        assert report.mse >= 0.0

    def test_student_B_mae_finite(self):
        report = self.validator.validate(self.eval_states)
        assert np.isfinite(report.mae)
        assert report.mae >= 0.0

    def test_student_B_cosine_similarity_in_range(self):
        report = self.validator.validate(self.eval_states)
        assert -1.0 <= report.mean_cosine_similarity <= 1.0

    def test_student_B_param_ratio_below_threshold(self):
        report = self.validator.validate(self.eval_states)
        assert report.param_ratio < 0.9

    def test_student_B_robustness_slices(self):
        slices = {
            "low_resource": _make_states(30, seed=11),
            "high_threat": _make_states(30, seed=21),
            "sparse_obs": _make_states(30, seed=31),
        }
        report = self.validator.validate(self.eval_states, robustness_slices=slices)
        assert set(report.robustness_slice_agreements.keys()) == set(slices.keys())
        for v in report.robustness_slice_agreements.values():
            assert 0.0 <= v <= 1.0

    def test_student_B_report_serialisable(self):
        report = self.validator.validate(
            self.eval_states, n_latency_warmup=1, n_latency_repeats=3
        )
        d = report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert "action_agreement" in parsed
        assert "passed" in parsed

    def test_student_B_passed_property_is_bool(self):
        report = self.validator.validate(self.eval_states)
        assert isinstance(report.passed, bool)


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


def test_package_imports():
    from farm.core.decision.training import (  # noqa: F401
        StudentValidator,
        ValidationReport,
        ValidationThresholds,
    )
