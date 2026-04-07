"""Tests for farm/core/decision/training/quantize_ptq.py.

Covers:
- QuantizationConfig validation (modes, dtypes, backends).
- PostTrainingQuantizer: dynamic quantisation, forward parity, file round-trip.
- Static quantisation with calibration states.
- compare_outputs utility.
- load_quantized_checkpoint helpers (error handling, metadata).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
from farm.core.decision.training.quantize_ptq import (
    PostTrainingQuantizer,
    QuantizationConfig,
    QuantizationResult,
    compare_outputs,
    load_quantized_checkpoint,
)

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
PARENT_HIDDEN = 32  # small parent → student hidden = max(16, 16) = 16


def _make_student(seed: int = 0) -> StudentQNetwork:
    torch.manual_seed(seed)
    return StudentQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        parent_hidden_size=PARENT_HIDDEN,
    )


def _make_states(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


# ---------------------------------------------------------------------------
# QuantizationConfig tests
# ---------------------------------------------------------------------------


class TestQuantizationConfig:
    def test_defaults(self):
        cfg = QuantizationConfig()
        assert cfg.mode == "dynamic"
        assert cfg.dtype == "qint8"
        assert cfg.backend == "qnnpack"
        assert cfg.calibration_batches == 10
        assert cfg.calibration_batch_size == 64

    def test_torch_dtype_qint8(self):
        cfg = QuantizationConfig(dtype="qint8")
        assert cfg.torch_dtype() == torch.qint8

    def test_torch_dtype_quint8(self):
        cfg = QuantizationConfig(dtype="quint8")
        assert cfg.torch_dtype() == torch.quint8

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            QuantizationConfig(mode="qat")

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            QuantizationConfig(dtype="float16")

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            QuantizationConfig(backend="cuda")

    def test_invalid_calibration_batches(self):
        with pytest.raises(ValueError, match="calibration_batches"):
            QuantizationConfig(calibration_batches=0)

    def test_invalid_calibration_batch_size(self):
        with pytest.raises(ValueError, match="calibration_batch_size"):
            QuantizationConfig(calibration_batch_size=0)


# ---------------------------------------------------------------------------
# Dynamic quantisation
# ---------------------------------------------------------------------------


class TestDynamicQuantization:
    def test_returns_quantized_model_and_result(self):
        student = _make_student()
        cfg = QuantizationConfig(mode="dynamic")
        quantizer = PostTrainingQuantizer(cfg)
        q_model, result = quantizer.quantize(student)
        assert isinstance(q_model, nn.Module)
        assert isinstance(result, QuantizationResult)

    def test_result_mode_is_dynamic(self):
        student = _make_student()
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        _, result = quantizer.quantize(student)
        assert result.mode == "dynamic"

    def test_result_calibration_samples_zero(self):
        student = _make_student()
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        _, result = quantizer.quantize(student)
        assert result.calibration_samples == 0

    def test_result_linear_layers_counted(self):
        student = _make_student()
        n_linear = sum(1 for m in student.modules() if isinstance(m, nn.Linear))
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        _, result = quantizer.quantize(student)
        assert result.linear_layers_quantized == n_linear

    def test_forward_parity_action_agreement(self):
        """Quantised model should agree with float on most actions."""
        student = _make_student(seed=1)
        student.eval()
        states = _make_states(n=500, seed=1)

        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, _ = quantizer.quantize(student)

        cmp = compare_outputs(student, q_model, states)
        # Dynamic quantisation is very close to float; expect ≥ 90% agreement
        assert cmp["action_agreement"] >= 0.90, (
            f"Action agreement too low: {cmp['action_agreement']:.4f}"
        )

    def test_forward_parity_q_error(self):
        """Mean absolute Q-error should be small after dynamic quantisation."""
        student = _make_student(seed=2)
        student.eval()
        states = _make_states(n=500, seed=2)

        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, _ = quantizer.quantize(student)

        cmp = compare_outputs(student, q_model, states)
        # Expect mean error < 0.5 (Q-values are typically in [-2, 2] for these small nets)
        assert cmp["mean_q_error"] < 0.5, f"Mean Q-error too high: {cmp['mean_q_error']:.4f}"

    def test_float_model_unchanged(self):
        """Dynamic quantisation must not mutate the original model's weights."""
        student = _make_student(seed=3)
        orig_weights = {k: v.clone() for k, v in student.named_parameters()}
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        quantizer.quantize(student)
        for name, param in student.named_parameters():
            assert torch.allclose(param, orig_weights[name]), (
                f"Parameter '{name}' was mutated by dynamic quantisation"
            )

    def test_base_qnetwork_dynamic(self):
        """BaseQNetwork is also compatible with dynamic quantisation."""
        model = BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=PARENT_HIDDEN)
        model.eval()
        states = _make_states(n=100)
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, result = quantizer.quantize(model)
        cmp = compare_outputs(model, q_model, states)
        assert cmp["action_agreement"] >= 0.90


# ---------------------------------------------------------------------------
# Static quantisation
# ---------------------------------------------------------------------------


class TestStaticQuantization:
    def test_static_requires_calibration_states(self):
        student = _make_student()
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="static"))
        with pytest.raises(ValueError, match="calibration_states"):
            quantizer.quantize(student, calibration_states=None)

    def test_static_requires_non_empty_calibration(self):
        student = _make_student()
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="static"))
        with pytest.raises(ValueError, match="calibration_states"):
            quantizer.quantize(student, calibration_states=np.array([]).astype("float32"))

    def test_static_quantization_runs(self):
        """Static quantisation should run without error on a small model."""
        student = _make_student()
        states = _make_states(n=200)
        cfg = QuantizationConfig(
            mode="static",
            backend="qnnpack",
            calibration_batches=2,
            calibration_batch_size=32,
        )
        quantizer = PostTrainingQuantizer(cfg)
        q_model, result = quantizer.quantize(student, calibration_states=states)
        assert isinstance(q_model, nn.Module)
        assert result.mode == "static"
        assert result.calibration_samples > 0

    def test_static_forward_parity(self):
        """Static quantised model should have reasonable action agreement."""
        student = _make_student(seed=4)
        states = _make_states(n=300, seed=4)
        cfg = QuantizationConfig(
            mode="static",
            backend="qnnpack",
            calibration_batches=3,
            calibration_batch_size=64,
        )
        quantizer = PostTrainingQuantizer(cfg)
        q_model, _ = quantizer.quantize(student, calibration_states=states)
        cmp = compare_outputs(student, q_model, states)
        # Static can be less accurate than dynamic due to activation quantisation
        assert cmp["action_agreement"] >= 0.70, (
            f"Static action agreement too low: {cmp['action_agreement']:.4f}"
        )


# ---------------------------------------------------------------------------
# Checkpoint save / load round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    def test_save_and_load_dynamic(self):
        """A dynamic-quantised checkpoint can be saved and reloaded."""
        student = _make_student(seed=5)
        states = _make_states(n=100, seed=5)
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, result = quantizer.quantize(student)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_int8.pt")
            quantizer.save_checkpoint(q_model, path, result)

            assert os.path.isfile(path), "Checkpoint file not written"
            assert os.path.isfile(path + ".json"), "JSON metadata file not written"

            q_loaded, meta = load_quantized_checkpoint(path)
            assert isinstance(q_loaded, nn.Module)

            # Outputs must match between original quantised and reloaded model
            q_loaded.eval()
            cmp = compare_outputs(q_model, q_loaded, states)
            assert cmp["action_agreement"] == pytest.approx(1.0), (
                "Round-trip load changed actions"
            )

    def test_json_metadata_content(self):
        """JSON metadata must contain quantisation config and arch_kwargs."""
        student = _make_student()
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, result = quantizer.quantize(student)

        arch_kwargs = {"input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM, "parent_hidden_size": PARENT_HIDDEN}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_int8.pt")
            quantizer.save_checkpoint(q_model, path, result, arch_kwargs=arch_kwargs)

            with open(path + ".json") as fh:
                meta = json.load(fh)

            assert "quantization" in meta
            assert meta["quantization"]["mode"] == "dynamic"
            assert "arch_kwargs" in meta
            assert meta["arch_kwargs"]["input_dim"] == INPUT_DIM

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_quantized_checkpoint("/nonexistent/student_int8.pt")

    def test_load_without_json(self):
        """Loading a checkpoint without a companion JSON returns empty metadata."""
        student = _make_student()
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, result = quantizer.quantize(student)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_int8.pt")
            torch.save(q_model, path)  # no JSON companion
            _, meta = load_quantized_checkpoint(path)
            assert meta == {}

    def test_save_creates_output_dir(self):
        """save_checkpoint must create missing parent directories."""
        student = _make_student()
        quantizer = PostTrainingQuantizer(QuantizationConfig(mode="dynamic"))
        q_model, result = quantizer.quantize(student)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c", "student_int8.pt")
            quantizer.save_checkpoint(q_model, nested, result)
            assert os.path.isfile(nested)


# ---------------------------------------------------------------------------
# compare_outputs utility
# ---------------------------------------------------------------------------


class TestCompareOutputs:
    def test_identical_models_perfect_agreement(self):
        student = _make_student()
        student.eval()
        states = _make_states(n=100)
        # Comparing float model with itself → perfect agreement
        cmp = compare_outputs(student, student, states)
        assert cmp["action_agreement"] == pytest.approx(1.0)
        assert cmp["mean_q_error"] == pytest.approx(0.0, abs=1e-5)
        assert cmp["mean_cosine_similarity"] == pytest.approx(1.0, abs=1e-4)

    def test_n_states_reported(self):
        student = _make_student()
        states = _make_states(n=77)
        cmp = compare_outputs(student, student, states)
        assert cmp["n_states"] == 77

    def test_different_outputs_disagreement(self):
        """Two different random models should disagree on many actions."""
        m1 = _make_student(seed=10)
        m2 = _make_student(seed=99)  # different seed → different weights
        states = _make_states(n=500)
        cmp = compare_outputs(m1, m2, states)
        # With random weights, agreement should be roughly 1/output_dim = 25%
        assert cmp["action_agreement"] < 0.90, "Expected disagreement between random models"


# ---------------------------------------------------------------------------
# QuantizationResult serialisation
# ---------------------------------------------------------------------------


class TestQuantizationResult:
    def test_to_dict_round_trip(self):
        result = QuantizationResult(
            mode="dynamic",
            dtype="qint8",
            backend="qnnpack",
            calibration_samples=0,
            elapsed_seconds=0.1,
            linear_layers_quantized=3,
            float_param_bytes=4096,
            quantized_param_bytes=1024,
            notes=["note1"],
        )
        d = result.to_dict()
        restored = QuantizationResult.from_dict(d)
        assert restored.mode == result.mode
        assert restored.linear_layers_quantized == result.linear_layers_quantized
        assert restored.notes == result.notes

    def test_to_dict_json_serialisable(self):
        result = QuantizationResult(
            mode="dynamic",
            dtype="qint8",
            backend="qnnpack",
            calibration_samples=0,
            elapsed_seconds=0.5,
            linear_layers_quantized=3,
            float_param_bytes=4096,
            quantized_param_bytes=1024,
        )
        # Should not raise
        json.dumps(result.to_dict())
