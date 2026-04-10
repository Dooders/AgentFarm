"""Tests for the label_metrics module and the optional labels feature of StudentValidator.

Covers:
- compute_label_metrics: perfect predictions, partial errors, edge cases.
- LabelMetrics dataclass: field values, to_dict JSON serialisability.
- StudentValidator.validate() with labels= kwarg: populates label_metrics.
- Backward compatibility: no labels → label_metrics is None.
- ValidationReport.to_dict() includes label_metrics key.
- Input validation: wrong shape, mismatched length.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
from farm.core.decision.training.label_metrics import LabelMetrics, compute_label_metrics
from farm.core.decision.training.trainer_distill import (
    StudentValidator,
    ValidationReport,
    ValidationThresholds,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
PARENT_HIDDEN = 32
SEED = 7


def _make_parent(seed: int = SEED) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=PARENT_HIDDEN)


def _make_student() -> StudentQNetwork:
    return StudentQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        parent_hidden_size=PARENT_HIDDEN,
    )


def _make_states(n: int = 50, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _make_labels(n: int = 50, n_classes: int = OUTPUT_DIM, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n).astype(np.int64)


# ---------------------------------------------------------------------------
# compute_label_metrics – basic correctness
# ---------------------------------------------------------------------------


class TestComputeLabelMetrics:
    def test_perfect_predictions_accuracy_one(self):
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        assert m.accuracy == pytest.approx(1.0)

    def test_perfect_predictions_macro_f1_one(self):
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        assert m.macro_f1 == pytest.approx(1.0)

    def test_perfect_predictions_confusion_matrix_diagonal(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        cm = np.array(m.confusion_matrix)
        # All predictions correct → confusion matrix is diagonal
        assert np.all(cm == np.diag(np.diag(cm)))
        assert cm.trace() == len(labels)

    def test_accuracy_with_errors(self):
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        preds = np.array([0, 1, 1, 1, 0], dtype=np.int64)  # one error at index 2
        m = compute_label_metrics(labels, preds)
        assert m.accuracy == pytest.approx(4 / 5)

    def test_macro_f1_between_zero_and_one(self):
        labels = _make_labels(100, n_classes=4)
        preds = _make_labels(100, n_classes=4, seed=99)  # random
        m = compute_label_metrics(labels, preds)
        assert 0.0 <= m.macro_f1 <= 1.0

    def test_n_classes_inferred(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        assert m.n_classes == 3

    def test_n_classes_explicit_overrides_inference(self):
        labels = np.array([0, 1], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy(), n_classes=5)
        assert m.n_classes == 5
        assert len(m.confusion_matrix) == 5
        assert len(m.support) == 5

    def test_support_sums_to_n_samples(self):
        labels = np.array([0, 1, 2, 0, 1], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        assert sum(m.support) == len(labels)

    def test_per_class_f1_length_matches_n_classes(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        assert len(m.per_class_f1) == m.n_classes

    def test_confusion_matrix_shape(self):
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        cm = m.confusion_matrix
        assert len(cm) == m.n_classes
        for row in cm:
            assert len(row) == m.n_classes

    def test_confusion_matrix_off_diagonal_on_error(self):
        # One prediction is wrong: true=2, predicted=1
        labels = np.array([0, 1, 2], dtype=np.int64)
        preds = np.array([0, 1, 1], dtype=np.int64)
        m = compute_label_metrics(labels, preds)
        cm = np.array(m.confusion_matrix)
        # cm[2, 1] should be 1 (true 2 predicted as 1)
        assert cm[2, 1] == 1
        # cm[2, 2] should be 0
        assert cm[2, 2] == 0

    def test_all_wrong_accuracy_zero(self):
        labels = np.array([0, 0, 0], dtype=np.int64)
        preds = np.array([1, 1, 1], dtype=np.int64)
        m = compute_label_metrics(labels, preds, n_classes=2)
        assert m.accuracy == pytest.approx(0.0)

    def test_absent_class_excluded_from_macro_f1(self):
        """Class 2 absent from labels; macro-F1 should only average over classes 0 and 1."""
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        preds = np.array([0, 1, 0, 1], dtype=np.int64)
        m = compute_label_metrics(labels, preds, n_classes=3)
        # Classes 0 and 1 have perfect F1 → macro_f1 = 1.0 (class 2 excluded)
        assert m.macro_f1 == pytest.approx(1.0)
        # Class 2 should have F1 of 0.0 (excluded from average but present in list)
        assert m.per_class_f1[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_label_metrics – input validation
# ---------------------------------------------------------------------------


class TestComputeLabelMetricsValidation:
    def test_labels_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            compute_label_metrics(np.zeros((3, 2), dtype=np.int64), np.zeros(3, dtype=np.int64))

    def test_predictions_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            compute_label_metrics(np.zeros(3, dtype=np.int64), np.zeros((3, 2), dtype=np.int64))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_label_metrics(np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64))

    def test_empty_arrays_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_label_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    def test_n_classes_less_than_2_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_label_metrics(np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), n_classes=1)


# ---------------------------------------------------------------------------
# LabelMetrics.to_dict
# ---------------------------------------------------------------------------


class TestLabelMetricsToDict:
    def test_to_dict_is_json_serialisable(self):
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        serialised = json.dumps(m.to_dict())
        parsed = json.loads(serialised)
        assert "accuracy" in parsed
        assert "macro_f1" in parsed
        assert "confusion_matrix" in parsed
        assert "n_classes" in parsed
        assert "support" in parsed
        assert "per_class_f1" in parsed

    def test_to_dict_accuracy_matches(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        m = compute_label_metrics(labels, labels.copy())
        assert m.to_dict()["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# StudentValidator.validate() with labels
# ---------------------------------------------------------------------------


class TestStudentValidatorWithLabels:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.states = _make_states(100)
        self.labels = _make_labels(100)
        self.parent = _make_parent()
        self.student = _make_student()
        self.validator = StudentValidator(self.parent, self.student)

    def _validate(self, **kwargs):
        return self.validator.validate(
            self.states,
            n_latency_warmup=0,
            n_latency_repeats=0,
            **kwargs,
        )

    def test_label_metrics_none_by_default(self):
        report = self._validate()
        assert report.label_metrics is None

    def test_label_metrics_populated_when_labels_provided(self):
        report = self._validate(labels=self.labels)
        assert isinstance(report.label_metrics, LabelMetrics)

    def test_label_metrics_accuracy_in_unit_interval(self):
        report = self._validate(labels=self.labels)
        assert 0.0 <= report.label_metrics.accuracy <= 1.0

    def test_label_metrics_macro_f1_in_unit_interval(self):
        report = self._validate(labels=self.labels)
        assert 0.0 <= report.label_metrics.macro_f1 <= 1.0

    def test_label_metrics_n_classes_matches_output_dim(self):
        report = self._validate(labels=self.labels)
        # n_classes inferred from max label and max prediction
        assert report.label_metrics.n_classes >= 2

    def test_label_metrics_perfect_when_labels_equal_student_actions(self):
        """If labels exactly match what the student predicts, accuracy = 1."""
        tensor = torch.tensor(self.states, dtype=torch.float32)
        with torch.no_grad():
            self.student.eval()
            student_actions = self.student(tensor).argmax(dim=-1).numpy().astype(np.int64)
        report = self._validate(labels=student_actions)
        assert report.label_metrics.accuracy == pytest.approx(1.0)

    def test_label_metrics_confusion_matrix_is_square(self):
        report = self._validate(labels=self.labels)
        cm = report.label_metrics.confusion_matrix
        n = report.label_metrics.n_classes
        assert len(cm) == n
        for row in cm:
            assert len(row) == n

    def test_validation_report_is_not_affected_by_labels(self):
        """Core fidelity metrics must be identical with or without labels."""
        report_no_labels = self._validate()
        report_with_labels = self._validate(labels=self.labels)
        assert report_no_labels.action_agreement == pytest.approx(report_with_labels.action_agreement)
        assert report_no_labels.kl_divergence == pytest.approx(report_with_labels.kl_divergence)
        assert report_no_labels.mse == pytest.approx(report_with_labels.mse)


# ---------------------------------------------------------------------------
# StudentValidator.validate() labels – input validation
# ---------------------------------------------------------------------------


class TestStudentValidatorLabelsValidation:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.states = _make_states(20)
        self.validator = StudentValidator(_make_parent(), _make_student())

    def test_labels_wrong_shape_raises(self):
        bad_labels = np.zeros((20, 2), dtype=np.int64)
        with pytest.raises(ValueError, match="1-D"):
            self.validator.validate(self.states, labels=bad_labels)

    def test_labels_length_mismatch_raises(self):
        bad_labels = np.zeros(10, dtype=np.int64)  # 10 ≠ 20
        with pytest.raises(ValueError, match="does not match"):
            self.validator.validate(self.states, labels=bad_labels)


# ---------------------------------------------------------------------------
# ValidationReport.to_dict() with label_metrics
# ---------------------------------------------------------------------------


class TestValidationReportToDict:
    def test_to_dict_label_metrics_none_when_absent(self):
        validator = StudentValidator(_make_parent(), _make_student())
        report = validator.validate(
            _make_states(30), n_latency_warmup=0, n_latency_repeats=0
        )
        d = report.to_dict()
        assert "label_metrics" in d
        assert d["label_metrics"] is None

    def test_to_dict_label_metrics_populated_when_labels_provided(self):
        labels = _make_labels(30)
        validator = StudentValidator(_make_parent(), _make_student())
        report = validator.validate(
            _make_states(30),
            n_latency_warmup=0,
            n_latency_repeats=0,
            labels=labels,
        )
        d = report.to_dict()
        assert d["label_metrics"] is not None
        assert "accuracy" in d["label_metrics"]
        assert "macro_f1" in d["label_metrics"]
        assert "confusion_matrix" in d["label_metrics"]

    def test_to_dict_fully_json_serialisable_with_label_metrics(self):
        labels = _make_labels(30)
        validator = StudentValidator(_make_parent(), _make_student())
        report = validator.validate(
            _make_states(30),
            n_latency_warmup=0,
            n_latency_repeats=0,
            labels=labels,
        )
        json.dumps(report.to_dict(), allow_nan=False)


# ---------------------------------------------------------------------------
# __init__ exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_label_metrics_exported_from_training_package(self):
        from farm.core.decision.training import LabelMetrics, compute_label_metrics  # noqa: F401

        assert LabelMetrics is not None
        assert compute_label_metrics is not None
