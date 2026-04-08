"""Tests for the QuantizedValidator, QuantizedValidationThresholds, and
QuantizedValidationReport classes in farm/core/decision/training/quantize_ptq.py.

Covers:
- QuantizedValidationThresholds: defaults and custom values.
- QuantizedValidationReport: construction, passed property, JSON serialisation,
  to_dict schema.
- QuantizedValidator: initialisation, compatibility check, fidelity metrics,
  latency measurement, file-size fields, metadata extraction, and full
  integration with a dynamically-quantized StudentQNetwork.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import StudentQNetwork
from farm.core.decision.training.quantize_ptq import (
    PostTrainingQuantizer,
    QuantizationConfig,
    QuantizationResult,
    QuantizedValidationReport,
    QuantizedValidationThresholds,
    QuantizedValidator,
    compare_outputs,
)

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
PARENT_HIDDEN = 32  # student hidden = max(16, 16) = 16


def _make_student(seed: int = 0) -> StudentQNetwork:
    torch.manual_seed(seed)
    return StudentQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        parent_hidden_size=PARENT_HIDDEN,
    )


def _make_states(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _dynamic_quantize(student: StudentQNetwork) -> nn.Module:
    quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
    q_model, _ = quantizer.quantize(student)
    return q_model


def _make_quantization_result() -> QuantizationResult:
    return QuantizationResult(
        mode="dynamic",
        dtype="qint8",
        backend="qnnpack",
        calibration_samples=0,
        elapsed_seconds=0.01,
        linear_layers_quantized=3,
        float_param_bytes=4096,
        quantized_param_bytes=1024,
    )


def _make_report(**kwargs) -> QuantizedValidationReport:
    defaults = dict(
        action_agreement=0.95,
        mean_q_error=0.05,
        max_q_error=0.2,
        mean_cosine_similarity=0.99,
        n_states=100,
        float_inference_ms=0.1,
        quantized_inference_ms=0.15,
        latency_ratio=1.5,
        float_checkpoint_bytes=None,
        quantized_checkpoint_bytes=None,
        size_ratio=None,
        pytorch_version=torch.__version__,
        quantization_mode="dynamic",
        quantization_backend="qnnpack",
        quantization_dtype="qint8",
        compatible=True,
        thresholds=QuantizedValidationThresholds(),
    )
    defaults.update(kwargs)
    return QuantizedValidationReport(**defaults)


# ---------------------------------------------------------------------------
# QuantizedValidationThresholds
# ---------------------------------------------------------------------------


class TestQuantizedValidationThresholds:
    def test_defaults(self):
        t = QuantizedValidationThresholds()
        assert t.min_action_agreement == pytest.approx(0.75)
        assert t.max_mean_q_error == pytest.approx(0.5)
        assert t.min_cosine_similarity == pytest.approx(0.75)
        assert t.max_latency_ratio == pytest.approx(2.0)
        assert t.report_only is False

    def test_custom_values(self):
        t = QuantizedValidationThresholds(
            min_action_agreement=0.90,
            max_mean_q_error=0.1,
            min_cosine_similarity=0.95,
            max_latency_ratio=1.5,
            report_only=True,
        )
        assert t.min_action_agreement == pytest.approx(0.90)
        assert t.max_mean_q_error == pytest.approx(0.1)
        assert t.min_cosine_similarity == pytest.approx(0.95)
        assert t.max_latency_ratio == pytest.approx(1.5)
        assert t.report_only is True


# ---------------------------------------------------------------------------
# QuantizedValidationReport
# ---------------------------------------------------------------------------


class TestQuantizedValidationReport:
    def test_passed_all_thresholds_met(self):
        report = _make_report()
        assert report.passed is True

    def test_passed_false_low_action_agreement(self):
        t = QuantizedValidationThresholds(min_action_agreement=0.95)
        report = _make_report(action_agreement=0.80, thresholds=t)
        assert report.passed is False

    def test_passed_false_high_q_error(self):
        t = QuantizedValidationThresholds(max_mean_q_error=0.1)
        report = _make_report(mean_q_error=0.5, thresholds=t)
        assert report.passed is False

    def test_passed_false_low_cosine_similarity(self):
        t = QuantizedValidationThresholds(min_cosine_similarity=0.99)
        report = _make_report(mean_cosine_similarity=0.70, thresholds=t)
        assert report.passed is False

    def test_passed_false_high_latency_ratio(self):
        t = QuantizedValidationThresholds(max_latency_ratio=1.0)
        report = _make_report(latency_ratio=1.5, thresholds=t)
        assert report.passed is False

    def test_passed_false_when_not_compatible(self):
        report = _make_report(compatible=False)
        assert report.passed is False

    def test_passed_true_when_report_only_despite_failures(self):
        t = QuantizedValidationThresholds(min_action_agreement=0.99, report_only=True)
        report = _make_report(action_agreement=0.10, compatible=False, thresholds=t)
        assert report.passed is True

    def test_to_dict_has_required_sections(self):
        report = _make_report()
        d = report.to_dict()
        assert "fidelity" in d
        assert "latency" in d
        assert "size" in d
        assert "compatibility" in d
        assert "thresholds" in d
        assert "passed" in d

    def test_to_dict_fidelity_keys(self):
        report = _make_report()
        fid = report.to_dict()["fidelity"]
        assert "action_agreement" in fid
        assert "mean_q_error" in fid
        assert "max_q_error" in fid
        assert "mean_cosine_similarity" in fid
        assert "n_states" in fid

    def test_to_dict_latency_keys(self):
        report = _make_report()
        lat = report.to_dict()["latency"]
        assert "float_inference_ms" in lat
        assert "quantized_inference_ms" in lat
        assert "latency_ratio" in lat

    def test_to_dict_size_keys(self):
        report = _make_report()
        sz = report.to_dict()["size"]
        assert "float_checkpoint_bytes" in sz
        assert "quantized_checkpoint_bytes" in sz
        assert "size_ratio" in sz

    def test_to_dict_compatibility_keys(self):
        report = _make_report()
        compat = report.to_dict()["compatibility"]
        assert "compatible" in compat
        assert "pytorch_version" in compat
        assert "quantization_mode" in compat
        assert "quantization_backend" in compat
        assert "quantization_dtype" in compat

    def test_to_dict_json_serialisable(self):
        report = _make_report()
        d = report.to_dict()
        # Must not raise
        json.dumps(d)

    def test_to_dict_size_ratio_none_when_no_paths(self):
        report = _make_report()
        sz = report.to_dict()["size"]
        assert sz["float_checkpoint_bytes"] is None
        assert sz["quantized_checkpoint_bytes"] is None
        assert sz["size_ratio"] is None

    def test_to_dict_thresholds_keys(self):
        report = _make_report()
        t = report.to_dict()["thresholds"]
        assert "min_action_agreement" in t
        assert "max_mean_q_error" in t
        assert "min_cosine_similarity" in t
        assert "max_latency_ratio" in t
        assert "report_only" in t


# ---------------------------------------------------------------------------
# QuantizedValidator – initialisation
# ---------------------------------------------------------------------------


class TestQuantizedValidatorInit:
    def test_default_thresholds(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        validator = QuantizedValidator(student, q_model)
        assert isinstance(validator.thresholds, QuantizedValidationThresholds)

    def test_custom_thresholds(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        t = QuantizedValidationThresholds(min_action_agreement=0.90)
        validator = QuantizedValidator(student, q_model, thresholds=t)
        assert validator.thresholds.min_action_agreement == pytest.approx(0.90)

    def test_default_device_is_cpu(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        validator = QuantizedValidator(student, q_model)
        assert validator.device.type == "cpu"


# ---------------------------------------------------------------------------
# QuantizedValidator – validate() basics
# ---------------------------------------------------------------------------


class TestQuantizedValidatorValidate:
    def test_validate_returns_report(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        states = _make_states()
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states)
        assert isinstance(report, QuantizedValidationReport)

    def test_compatible_is_true_for_valid_model(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        states = _make_states()
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states)
        assert report.compatible is True

    def test_n_states_matches_input(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        states = _make_states(n=77)
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states)
        assert report.n_states == 77

    def test_action_agreement_high_for_dynamic_quantized(self):
        """Dynamic quantization should preserve ≥ 90% action agreement."""
        student = _make_student(seed=1)
        student.eval()
        q_model = _dynamic_quantize(student)
        states = _make_states(n=200, seed=1)
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states, n_latency_warmup=2, n_latency_repeats=5)
        assert report.action_agreement >= 0.90, (
            f"Action agreement too low: {report.action_agreement:.4f}"
        )

    def test_pytorch_version_in_report(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        states = _make_states()
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states, n_latency_warmup=1, n_latency_repeats=3)
        assert report.pytorch_version == torch.__version__

    def test_latency_values_are_positive(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        states = _make_states()
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states, n_latency_warmup=2, n_latency_repeats=5)
        assert report.float_inference_ms > 0
        assert report.quantized_inference_ms > 0
        assert report.latency_ratio > 0

    def test_raises_on_empty_states(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        validator = QuantizedValidator(student, q_model)
        with pytest.raises(ValueError, match="non-empty"):
            validator.validate(np.empty((0, INPUT_DIM), dtype="float32"))

    def test_raises_on_1d_states(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        validator = QuantizedValidator(student, q_model)
        with pytest.raises(ValueError, match="2D"):
            validator.validate(np.ones((INPUT_DIM,), dtype="float32"))

    def test_metadata_extracted_from_quantization_dict(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        states = _make_states()
        meta = {
            "quantization": {
                "mode": "dynamic",
                "backend": "qnnpack",
                "dtype": "qint8",
            }
        }
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(
            states,
            quantization_metadata=meta,
            n_latency_warmup=1,
            n_latency_repeats=3,
        )
        assert report.quantization_mode == "dynamic"
        assert report.quantization_backend == "qnnpack"
        assert report.quantization_dtype == "qint8"

    def test_metadata_falls_back_to_unknown(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        states = _make_states()
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states, n_latency_warmup=1, n_latency_repeats=3)
        assert report.quantization_mode == "unknown"
        assert report.quantization_backend == "unknown"
        assert report.quantization_dtype == "unknown"


# ---------------------------------------------------------------------------
# QuantizedValidator – file size fields
# ---------------------------------------------------------------------------


class TestQuantizedValidatorFileSize:
    def test_file_sizes_populated_when_paths_provided(self):
        student = _make_student(seed=2)
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, result = quantizer.quantize(student)

        with tempfile.TemporaryDirectory() as tmpdir:
            float_path = os.path.join(tmpdir, "student.pt")
            quant_path = os.path.join(tmpdir, "student_int8.pt")

            torch.save(student.state_dict(), float_path)
            quantizer.save_checkpoint(q_model, quant_path, result)

            states = _make_states()
            validator = QuantizedValidator(student, q_model)
            report = validator.validate(
                states,
                float_checkpoint_path=float_path,
                quantized_checkpoint_path=quant_path,
                n_latency_warmup=1,
                n_latency_repeats=3,
            )

            assert report.float_checkpoint_bytes is not None
            assert report.quantized_checkpoint_bytes is not None
            assert report.float_checkpoint_bytes > 0
            assert report.quantized_checkpoint_bytes > 0
            assert report.size_ratio is not None
            assert report.size_ratio > 0

    def test_file_sizes_none_when_paths_missing(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        states = _make_states()
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states, n_latency_warmup=1, n_latency_repeats=3)
        assert report.float_checkpoint_bytes is None
        assert report.quantized_checkpoint_bytes is None
        assert report.size_ratio is None


# ---------------------------------------------------------------------------
# QuantizedValidator – compatibility check
# ---------------------------------------------------------------------------


class TestQuantizedValidatorCompatibility:
    def test_incompatible_model_detected(self):
        """A model that raises on forward should be flagged as incompatible."""
        student = _make_student()

        class _BrokenModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("Intentional failure for testing")

        broken = _BrokenModel()
        states = _make_states()
        validator = QuantizedValidator(student, broken)
        report = validator.validate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.compatible is False

    def test_incompatible_model_fails_passed(self):
        student = _make_student()

        class _BrokenModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("Intentional failure for testing")

        broken = _BrokenModel()
        states = _make_states()
        validator = QuantizedValidator(student, broken)
        report = validator.validate(states, n_latency_warmup=0, n_latency_repeats=3)
        assert report.passed is False


# ---------------------------------------------------------------------------
# QuantizedValidator – report_only mode
# ---------------------------------------------------------------------------


class TestQuantizedValidatorReportOnly:
    def test_report_only_always_passes(self):
        student = _make_student()
        q_model = _dynamic_quantize(student)
        # Set very strict thresholds that would normally fail
        t = QuantizedValidationThresholds(
            min_action_agreement=1.0,
            max_mean_q_error=0.0,
            report_only=True,
        )
        states = _make_states()
        validator = QuantizedValidator(student, q_model, thresholds=t)
        report = validator.validate(states, n_latency_warmup=1, n_latency_repeats=3)
        assert report.passed is True


# ---------------------------------------------------------------------------
# QuantizedValidator – JSON round-trip
# ---------------------------------------------------------------------------


class TestQuantizedValidatorJsonRoundTrip:
    def test_to_dict_is_json_serialisable(self):
        student = _make_student(seed=3)
        q_model = _dynamic_quantize(student)
        states = _make_states(n=50, seed=3)
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states, n_latency_warmup=1, n_latency_repeats=3)

        d = report.to_dict()
        # Must not raise
        serialised = json.dumps(d)
        parsed = json.loads(serialised)

        assert "fidelity" in parsed
        assert "latency" in parsed
        assert "size" in parsed
        assert "compatibility" in parsed
        assert "passed" in parsed

    def test_passed_in_report_dict_matches_property(self):
        student = _make_student(seed=4)
        q_model = _dynamic_quantize(student)
        states = _make_states(n=50, seed=4)
        validator = QuantizedValidator(student, q_model)
        report = validator.validate(states, n_latency_warmup=1, n_latency_repeats=3)
        assert report.to_dict()["passed"] == report.passed


# ---------------------------------------------------------------------------
# Module-level __init__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_imports_from_training_init(self):
        from farm.core.decision.training import (  # noqa: F401
            QuantizedValidationReport,
            QuantizedValidationThresholds,
            QuantizedValidator,
        )
