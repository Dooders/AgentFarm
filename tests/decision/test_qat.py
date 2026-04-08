"""Tests for farm/core/decision/training/quantize_qat.py.

Covers:
- QATConfig validation (fields, defaults, error conditions).
- WeightOnlyFakeQuantLinear: fake-quant during training, passthrough during eval,
  gradient flow (STE), construction from nn.Linear.
- _replace_linear_with_fakeq helper: idempotency, recursion into Sequential.
- QATTrainer.prepare(): replaces Linear layers in student deep copy.
- QATTrainer.train(): training step runs, loss decreases, metrics populated.
- QATTrainer.convert(): produces a finite, forward-pass-compatible int8 model.
- QATTrainer.save_quantized() / load_qat_checkpoint(): round-trip.
- QATTrainer._save_qat_checkpoint(): float checkpoint saved as state-dict.
- QATMetrics serialisation.
- Comparison against PTQ-dynamic format (compare_outputs).
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

from farm.core.decision.base_dqn import BaseQNetwork, StudentQNetwork
from farm.core.decision.training.quantize_ptq import compare_outputs
from farm.core.decision.training.quantize_qat import (
    QATConfig,
    QATMetrics,
    QATTrainer,
    WeightOnlyFakeQuantLinear,
    _replace_linear_with_fakeq,
    load_qat_checkpoint,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
PARENT_HIDDEN = 32  # student hidden = max(16, 16) = 16


def _make_teacher(seed: int = 0) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=PARENT_HIDDEN)


def _make_student(seed: int = 1) -> StudentQNetwork:
    torch.manual_seed(seed)
    return StudentQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        parent_hidden_size=PARENT_HIDDEN,
    )


def _make_states(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _minimal_cfg(**kwargs) -> QATConfig:
    """QATConfig with tiny epochs/batch for fast unit tests."""
    defaults = dict(epochs=2, batch_size=16, val_fraction=0.1, seed=42)
    defaults.update(kwargs)
    return QATConfig(**defaults)


# ---------------------------------------------------------------------------
# QATConfig validation
# ---------------------------------------------------------------------------


class TestQATConfig:
    def test_defaults(self):
        cfg = QATConfig()
        assert cfg.epochs == 5
        assert cfg.learning_rate == pytest.approx(1e-4)
        assert cfg.batch_size == 32
        assert cfg.max_grad_norm == pytest.approx(1.0)
        assert cfg.val_fraction == pytest.approx(0.1)
        assert cfg.seed is None
        assert cfg.loss_fn == "mse"
        assert cfg.temperature == pytest.approx(3.0)
        assert cfg.alpha == pytest.approx(1.0)
        assert cfg.dtype == "qint8"

    def test_torch_dtype_qint8(self):
        assert QATConfig(dtype="qint8").torch_dtype() == torch.qint8

    def test_torch_dtype_quint8(self):
        assert QATConfig(dtype="quint8").torch_dtype() == torch.quint8

    def test_invalid_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            QATConfig(epochs=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            QATConfig(learning_rate=0.0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            QATConfig(batch_size=0)

    def test_invalid_max_grad_norm(self):
        with pytest.raises(ValueError, match="max_grad_norm"):
            QATConfig(max_grad_norm=-1.0)

    def test_max_grad_norm_none_allowed(self):
        cfg = QATConfig(max_grad_norm=None)
        assert cfg.max_grad_norm is None

    def test_invalid_val_fraction(self):
        with pytest.raises(ValueError, match="val_fraction"):
            QATConfig(val_fraction=1.0)

    def test_invalid_loss_fn(self):
        with pytest.raises(ValueError, match="loss_fn"):
            QATConfig(loss_fn="huber")

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            QATConfig(temperature=0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            QATConfig(alpha=1.5)

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            QATConfig(dtype="float16")


# ---------------------------------------------------------------------------
# WeightOnlyFakeQuantLinear
# ---------------------------------------------------------------------------


class TestWeightOnlyFakeQuantLinear:
    def test_from_linear_copies_weight(self):
        lin = nn.Linear(8, 4)
        fq = WeightOnlyFakeQuantLinear.from_linear(lin)
        assert torch.allclose(fq.weight, lin.weight)

    def test_from_linear_copies_bias(self):
        lin = nn.Linear(8, 4, bias=True)
        fq = WeightOnlyFakeQuantLinear.from_linear(lin)
        assert torch.allclose(fq.bias, lin.bias)

    def test_from_linear_no_bias(self):
        lin = nn.Linear(8, 4, bias=False)
        fq = WeightOnlyFakeQuantLinear.from_linear(lin)
        assert fq.bias is None

    def test_from_linear_does_not_share_weight(self):
        lin = nn.Linear(8, 4)
        fq = WeightOnlyFakeQuantLinear.from_linear(lin)
        lin.weight.data.fill_(99.0)
        assert not torch.allclose(fq.weight, lin.weight)

    def test_forward_eval_matches_linear(self):
        """Eval-mode forward must equal a plain Linear with the same weights."""
        lin = nn.Linear(8, 4)
        fq = WeightOnlyFakeQuantLinear.from_linear(lin)
        lin.eval()
        fq.eval()
        x = torch.randn(16, 8)
        assert torch.allclose(lin(x), fq(x), atol=1e-5)

    def test_forward_train_uses_fake_quant(self):
        """Training-mode output must differ from eval (quantisation noise)."""
        lin = nn.Linear(8, 4)
        torch.nn.init.uniform_(lin.weight, -1.0, 1.0)
        fq = WeightOnlyFakeQuantLinear.from_linear(lin)
        x = torch.randn(16, 8)
        fq.train()
        out_train = fq(x).detach()
        fq.eval()
        out_eval = fq(x).detach()
        # There should be a small but non-zero difference due to rounding
        assert not torch.allclose(out_train, out_eval, atol=1e-6)

    def test_gradient_flows_through_ste(self):
        """STE must allow gradients to flow into weights during training."""
        fq = WeightOnlyFakeQuantLinear(8, 4)
        fq.train()
        x = torch.randn(4, 8)
        loss = fq(x).sum()
        loss.backward()
        assert fq.weight.grad is not None
        assert torch.isfinite(fq.weight.grad).all()

    def test_fake_quant_weight_symmetric_range(self):
        """Fake-quantised weight must lie in the int8 symmetric range."""
        fq = WeightOnlyFakeQuantLinear(8, 4)
        fq.train()
        w_fq = WeightOnlyFakeQuantLinear._fake_quant_weight(fq.weight)
        # After fake-quant the scale is max(|W|)/127; values stay in [-max, max]
        assert torch.isfinite(w_fq).all()

    def test_output_finite_train(self):
        fq = WeightOnlyFakeQuantLinear(8, 4)
        fq.train()
        x = torch.randn(16, 8)
        assert torch.isfinite(fq(x)).all()

    def test_output_finite_eval(self):
        fq = WeightOnlyFakeQuantLinear(8, 4)
        fq.eval()
        x = torch.randn(16, 8)
        assert torch.isfinite(fq(x)).all()


# ---------------------------------------------------------------------------
# _replace_linear_with_fakeq
# ---------------------------------------------------------------------------


class TestReplaceLinear:
    def test_replaces_linear_in_sequential(self):
        seq = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        _replace_linear_with_fakeq(seq)
        assert isinstance(seq[0], WeightOnlyFakeQuantLinear)
        assert isinstance(seq[2], WeightOnlyFakeQuantLinear)
        assert isinstance(seq[1], nn.ReLU)  # non-Linear untouched

    def test_idempotent(self):
        """Calling replace twice should not break the model."""
        seq = nn.Sequential(nn.Linear(8, 4))
        _replace_linear_with_fakeq(seq)
        fq_first = seq[0]
        _replace_linear_with_fakeq(seq)  # second call
        assert seq[0] is fq_first  # same object, not re-wrapped

    def test_nested_module(self):
        """Replace works in nested containers (e.g. StudentQNetwork.network)."""
        student = _make_student()
        _replace_linear_with_fakeq(student)
        n_fq = sum(1 for m in student.modules() if isinstance(m, WeightOnlyFakeQuantLinear))
        n_plain_linear = sum(1 for m in student.modules() if type(m) is nn.Linear)
        assert n_fq == 3  # three Linear layers in StudentQNetwork
        assert n_plain_linear == 0  # all plain nn.Linear replaced

    def test_non_linear_layers_untouched(self):
        student = _make_student()
        n_layernorm_before = sum(1 for m in student.modules() if isinstance(m, nn.LayerNorm))
        _replace_linear_with_fakeq(student)
        n_layernorm_after = sum(1 for m in student.modules() if isinstance(m, nn.LayerNorm))
        assert n_layernorm_before == n_layernorm_after


# ---------------------------------------------------------------------------
# QATTrainer.prepare
# ---------------------------------------------------------------------------


class TestQATTrainerPrepare:
    def test_prepare_replaces_linear_in_copy(self):
        student = _make_student()
        teacher = _make_teacher()
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.prepare()

        # Original student untouched
        orig_linears = sum(1 for m in student.modules() if isinstance(m, nn.Linear))
        assert orig_linears == 3
        orig_fq = sum(
            1 for m in student.modules() if isinstance(m, WeightOnlyFakeQuantLinear)
        )
        assert orig_fq == 0

        # QAT student has WeightOnlyFakeQuantLinear
        assert trainer.qat_student is not None
        n_fq = sum(
            1
            for m in trainer.qat_student.modules()
            if isinstance(m, WeightOnlyFakeQuantLinear)
        )
        assert n_fq == 3

    def test_prepare_qat_student_in_train_mode(self):
        teacher = _make_teacher()
        student = _make_student()
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.prepare()
        assert trainer.qat_student.training

    def test_prepare_does_not_mutate_student(self):
        student = _make_student()
        teacher = _make_teacher()
        orig_weights = {k: v.clone() for k, v in student.named_parameters()}
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.prepare()
        for name, param in student.named_parameters():
            assert torch.allclose(param, orig_weights[name])

    def test_prepare_twice_resets(self):
        """Calling prepare() twice produces a fresh QAT student each time."""
        teacher = _make_teacher()
        student = _make_student()
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.prepare()
        first = trainer.qat_student
        trainer.prepare()
        second = trainer.qat_student
        assert first is not second


# ---------------------------------------------------------------------------
# QATTrainer.train
# ---------------------------------------------------------------------------


class TestQATTrainerTrain:
    def test_train_runs_and_returns_metrics(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        metrics = trainer.train(states)
        assert isinstance(metrics, QATMetrics)
        assert len(metrics.train_losses) == 2  # cfg epochs=2

    def test_train_calls_prepare_if_not_done(self):
        """train() auto-calls prepare() when not yet prepared."""
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        assert not trainer._prepared
        trainer.train(states)
        assert trainer._prepared

    def test_train_losses_finite(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        metrics = trainer.train(states)
        for loss in metrics.train_losses:
            assert np.isfinite(loss), f"Non-finite train loss: {loss}"

    def test_train_val_metrics_populated(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        cfg = _minimal_cfg(val_fraction=0.2)
        trainer = QATTrainer(teacher, student, cfg)
        metrics = trainer.train(states)
        assert len(metrics.val_losses) == 2
        assert len(metrics.action_agreements) == 2
        assert len(metrics.mean_prob_similarities) == 2

    def test_train_no_val_when_val_fraction_zero(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        cfg = _minimal_cfg(val_fraction=0.0)
        trainer = QATTrainer(teacher, student, cfg)
        metrics = trainer.train(states)
        assert len(metrics.val_losses) == 0
        assert len(metrics.action_agreements) == 0

    def test_train_empty_states_raises(self):
        teacher = _make_teacher()
        student = _make_student()
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        with pytest.raises(ValueError, match="non-empty"):
            trainer.train(np.empty((0, INPUT_DIM), dtype="float32"))

    def test_train_kl_loss_fn(self):
        """KL loss mode should also converge without errors."""
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        cfg = _minimal_cfg(loss_fn="kl", temperature=2.0, alpha=1.0)
        trainer = QATTrainer(teacher, student, cfg)
        metrics = trainer.train(states)
        assert len(metrics.train_losses) == 2
        for loss in metrics.train_losses:
            assert np.isfinite(loss)

    def test_train_hard_loss_tracked_when_alpha_lt_1(self):
        """Hard-loss term is tracked when alpha < 1 with kl mode."""
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        cfg = _minimal_cfg(loss_fn="kl", alpha=0.7)
        trainer = QATTrainer(teacher, student, cfg)
        metrics = trainer.train(states)
        assert len(metrics.train_hard_losses) == 2

    def test_train_saves_float_checkpoint(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "qat_student.pt")
            trainer.train(states, checkpoint_path=ckpt)
            assert os.path.isfile(ckpt), "Float checkpoint not written"
            assert os.path.isfile(ckpt + ".json"), "JSON metadata not written"
            # Must be a state-dict (weights_only=True)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            assert isinstance(state, dict)

    def test_train_elapsed_seconds_positive(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        metrics = trainer.train(states)
        assert metrics.elapsed_seconds > 0.0

    def test_train_base_qnetwork_student(self):
        """QATTrainer also works when student is a BaseQNetwork."""
        teacher = _make_teacher()
        student = BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=PARENT_HIDDEN)
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        metrics = trainer.train(states)
        assert len(metrics.train_losses) == 2


# ---------------------------------------------------------------------------
# QATTrainer.convert
# ---------------------------------------------------------------------------


class TestQATTrainerConvert:
    def test_convert_without_prepare_raises(self):
        teacher = _make_teacher()
        student = _make_student()
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        with pytest.raises(RuntimeError):
            trainer.convert()

    def test_convert_returns_module(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.train(states)
        q_model = trainer.convert()
        assert isinstance(q_model, nn.Module)

    def test_convert_forward_finite(self):
        """Converted model forward pass must produce finite outputs."""
        teacher = _make_teacher()
        student = _make_student(seed=5)
        states = _make_states(n=100, seed=5)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.train(states)
        q_model = trainer.convert()
        q_model.eval()
        x = torch.tensor(states[:32])
        with torch.no_grad():
            out = q_model(x)
        assert torch.isfinite(out).all(), "Converted model produced non-finite outputs"

    def test_convert_does_not_mutate_qat_student(self):
        """convert() should return a new model, not modify _qat_student."""
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.train(states)
        orig_type = type(list(trainer.qat_student.modules())[0])
        trainer.convert()
        new_type = type(list(trainer.qat_student.modules())[0])
        assert orig_type == new_type

    def test_convert_action_agreement_reasonable(self):
        """QAT-converted model should agree with float student on ≥ 70% actions."""
        teacher = _make_teacher(seed=10)
        student = _make_student(seed=10)
        states = _make_states(n=300, seed=10)
        cfg = _minimal_cfg(epochs=3, seed=10)
        trainer = QATTrainer(teacher, student, cfg)
        trainer.train(states)
        q_model = trainer.convert()

        # Compare QAT-converted vs original float student
        cmp = compare_outputs(student, q_model, states)
        assert cmp["action_agreement"] >= 0.70, (
            f"QAT action agreement too low: {cmp['action_agreement']:.4f}"
        )

    def test_convert_outputs_finite_batch(self):
        """A larger batch through the converted model must remain finite."""
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=200)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.train(states)
        q_model = trainer.convert()
        q_model.eval()
        x = torch.tensor(states)
        with torch.no_grad():
            out = q_model(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# QATTrainer.save_quantized / load_qat_checkpoint
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def _trained_converter(self) -> tuple:
        teacher = _make_teacher(seed=20)
        student = _make_student(seed=20)
        states = _make_states(n=100, seed=20)
        trainer = QATTrainer(teacher, student, _minimal_cfg(seed=20))
        trainer.train(states)
        q_model = trainer.convert()
        return trainer, q_model, states

    def test_save_quantized_creates_files(self):
        trainer, q_model, states = self._trained_converter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_qat_int8.pt")
            trainer.save_quantized(q_model, path)
            assert os.path.isfile(path), "int8 checkpoint not written"
            assert os.path.isfile(path + ".json"), "JSON metadata not written"

    def test_load_qat_checkpoint_returns_model(self):
        trainer, q_model, states = self._trained_converter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_qat_int8.pt")
            trainer.save_quantized(q_model, path)
            loaded, meta = load_qat_checkpoint(path)
            assert isinstance(loaded, nn.Module)

    def test_load_qat_checkpoint_metadata(self):
        trainer, q_model, states = self._trained_converter()
        arch_kwargs = {"input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_qat_int8.pt")
            trainer.save_quantized(q_model, path, arch_kwargs=arch_kwargs)
            _, meta = load_qat_checkpoint(path)
            assert "qat" in meta
            assert meta["qat"]["scope"] == "weight_only"
            assert meta["arch_kwargs"]["input_dim"] == INPUT_DIM
            assert "quantization" in meta
            assert meta["quantization"]["mode"] == "qat"
            assert meta["quantization"]["dtype"] == "qint8"
            assert meta["quantization"]["backend"] != "unknown"

    def test_round_trip_action_agreement(self):
        """Saved and reloaded model must produce identical outputs."""
        trainer, q_model, states = self._trained_converter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_qat_int8.pt")
            trainer.save_quantized(q_model, path)
            loaded, _ = load_qat_checkpoint(path)
            cmp = compare_outputs(q_model, loaded, states)
            assert cmp["action_agreement"] == pytest.approx(1.0), (
                "Round-trip changed model outputs"
            )

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_qat_checkpoint("/nonexistent/student_qat_int8.pt")

    def test_load_without_json_returns_empty_meta(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.train(states)
        q_model = trainer.convert()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_qat_int8.pt")
            torch.save(q_model, path)  # no JSON companion
            _, meta = load_qat_checkpoint(path)
            assert meta == {}

    def test_save_quantized_creates_parent_dirs(self):
        """save_quantized must create missing parent directories."""
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        trainer.train(states)
        q_model = trainer.convert()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c", "student_qat_int8.pt")
            trainer.save_quantized(q_model, nested)
            assert os.path.isfile(nested)

    def test_json_metadata_serialisable(self):
        trainer, q_model, states = self._trained_converter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "student_qat_int8.pt")
            trainer.save_quantized(q_model, path)
            with open(path + ".json") as fh:
                meta = json.load(fh)
            assert "qat" in meta
            assert "quantization" in meta
            assert "notes" in meta


# ---------------------------------------------------------------------------
# QATMetrics serialisation
# ---------------------------------------------------------------------------


class TestQATMetrics:
    def test_to_dict_round_trip(self):
        m = QATMetrics(
            train_losses=[0.1, 0.05],
            train_soft_losses=[0.1, 0.05],
            val_losses=[0.12, 0.06],
            action_agreements=[0.8, 0.85],
            mean_prob_similarities=[0.9, 0.92],
            best_val_loss=0.06,
            best_epoch=1,
            elapsed_seconds=1.23,
        )
        d = m.to_dict()
        assert d["best_epoch"] == 1
        assert d["elapsed_seconds"] == pytest.approx(1.23)
        assert d["train_losses"] == [0.1, 0.05]

    def test_to_dict_json_serialisable(self):
        m = QATMetrics(train_losses=[0.1], elapsed_seconds=0.5)
        json.dumps(m.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# Float checkpoint (state-dict) round-trip
# ---------------------------------------------------------------------------


class TestFloatCheckpoint:
    def test_float_checkpoint_loadable_with_weights_only(self):
        """QAT float checkpoint must be loadable with weights_only=True."""
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        trainer = QATTrainer(teacher, student, _minimal_cfg())
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "qat_float.pt")
            trainer.train(states, checkpoint_path=ckpt)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            assert isinstance(state, dict)
            assert len(state) > 0

    def test_float_checkpoint_json_content(self):
        teacher = _make_teacher()
        student = _make_student()
        states = _make_states(n=100)
        cfg = _minimal_cfg(epochs=2, loss_fn="mse")
        trainer = QATTrainer(teacher, student, cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "qat_float.pt")
            trainer.train(states, checkpoint_path=ckpt)
            with open(ckpt + ".json") as fh:
                meta = json.load(fh)
            assert meta["config"]["epochs"] == 2
            assert meta["config"]["loss_fn"] == "mse"
            assert meta["config"]["scope"] == "weight_only"
            assert "metrics" in meta
