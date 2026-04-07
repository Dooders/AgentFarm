"""Tests for farm/core/decision/training/trainer_distill.py.

Covers:
- DistillationConfig validation (including new temp_decay field)
- DistillationTrainer: basic training loop, KL and MSE losses, hard-loss blend,
  gradient clipping, val/train split, checkpointing, action agreement,
  temperature decay, separate soft/hard loss tracking, calibration metrics,
  and the public evaluate_agreement helper.
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
from farm.core.decision.training.trainer_distill import (
    DistillationConfig,
    DistillationMetrics,
    DistillationTrainer,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
PARENT_HIDDEN = 32


def _make_teacher() -> BaseQNetwork:
    return BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=PARENT_HIDDEN)


def _make_student() -> StudentQNetwork:
    return StudentQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        parent_hidden_size=PARENT_HIDDEN,
    )


def _make_states(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _default_cfg(**kwargs) -> DistillationConfig:
    # alpha=1.0 means pure soft-label distillation (new default semantics)
    defaults = dict(epochs=2, batch_size=16, val_fraction=0.1, seed=42, alpha=1.0)
    defaults.update(kwargs)
    return DistillationConfig(**defaults)


# ---------------------------------------------------------------------------
# DistillationConfig tests
# ---------------------------------------------------------------------------


class TestDistillationConfig:
    def test_defaults(self):
        cfg = DistillationConfig()
        assert cfg.temperature == 3.0
        assert cfg.temp_decay == 1.0
        assert cfg.alpha == 1.0
        assert cfg.learning_rate == 1e-3
        assert cfg.epochs == 10
        assert cfg.batch_size == 32
        assert cfg.max_grad_norm == 1.0
        assert cfg.val_fraction == 0.1
        assert cfg.seed is None
        assert cfg.loss_fn == "kl"

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            DistillationConfig(temperature=0.0)

    def test_invalid_temp_decay_zero(self):
        with pytest.raises(ValueError, match="temp_decay"):
            DistillationConfig(temp_decay=0.0)

    def test_invalid_temp_decay_above_one(self):
        with pytest.raises(ValueError, match="temp_decay"):
            DistillationConfig(temp_decay=1.1)

    def test_valid_temp_decay_boundary(self):
        cfg = DistillationConfig(temp_decay=1.0)
        assert cfg.temp_decay == 1.0
        cfg2 = DistillationConfig(temp_decay=0.9)
        assert cfg2.temp_decay == 0.9

    def test_invalid_alpha_high(self):
        with pytest.raises(ValueError, match="alpha"):
            DistillationConfig(alpha=1.5)

    def test_invalid_alpha_low(self):
        with pytest.raises(ValueError, match="alpha"):
            DistillationConfig(alpha=-0.1)

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="learning_rate"):
            DistillationConfig(learning_rate=0.0)

    def test_invalid_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            DistillationConfig(epochs=0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            DistillationConfig(batch_size=0)

    def test_invalid_val_fraction_high(self):
        with pytest.raises(ValueError, match="val_fraction"):
            DistillationConfig(val_fraction=1.0)

    def test_invalid_val_fraction_low(self):
        with pytest.raises(ValueError, match="val_fraction"):
            DistillationConfig(val_fraction=-0.1)

    def test_invalid_loss_fn(self):
        with pytest.raises(ValueError, match="loss_fn"):
            DistillationConfig(loss_fn="huber")

    def test_valid_boundary_alpha(self):
        cfg = DistillationConfig(alpha=0.0)
        assert cfg.alpha == 0.0
        cfg2 = DistillationConfig(alpha=1.0)
        assert cfg2.alpha == 1.0

    def test_mse_loss_fn(self):
        cfg = DistillationConfig(loss_fn="mse")
        assert cfg.loss_fn == "mse"

    def test_invalid_max_grad_norm_zero(self):
        with pytest.raises(ValueError, match="max_grad_norm"):
            DistillationConfig(max_grad_norm=0.0)

    def test_invalid_max_grad_norm_negative(self):
        with pytest.raises(ValueError, match="max_grad_norm"):
            DistillationConfig(max_grad_norm=-1.0)

    def test_none_max_grad_norm_is_valid(self):
        cfg = DistillationConfig(max_grad_norm=None)
        assert cfg.max_grad_norm is None


# ---------------------------------------------------------------------------
# DistillationTrainer: initialization
# ---------------------------------------------------------------------------


class TestDistillationTrainerInit:
    def test_teacher_frozen_after_init(self):
        teacher = _make_teacher()
        student = _make_student()
        _ = DistillationTrainer(teacher, student, _default_cfg())
        for p in teacher.parameters():
            assert not p.requires_grad

    def test_student_not_frozen(self):
        teacher = _make_teacher()
        student = _make_student()
        _ = DistillationTrainer(teacher, student, _default_cfg())
        assert any(p.requires_grad for p in student.parameters())

    def test_teacher_in_eval_mode(self):
        teacher = _make_teacher()
        student = _make_student()
        _ = DistillationTrainer(teacher, student, _default_cfg())
        assert not teacher.training

    def test_default_config_used_when_none(self):
        teacher = _make_teacher()
        student = _make_student()
        trainer = DistillationTrainer(teacher, student)
        assert isinstance(trainer.config, DistillationConfig)

    def test_default_device_is_cpu(self):
        teacher = _make_teacher()
        student = _make_student()
        trainer = DistillationTrainer(teacher, student, _default_cfg())
        assert trainer.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# DistillationTrainer: training loop
# ---------------------------------------------------------------------------


class TestDistillationTrainerTrain:
    def test_returns_distillation_metrics(self):
        trainer = DistillationTrainer(_make_teacher(), _make_student(), _default_cfg())
        metrics = trainer.train(_make_states())
        assert isinstance(metrics, DistillationMetrics)

    def test_train_losses_length_equals_epochs(self):
        cfg = _default_cfg(epochs=3)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert len(metrics.train_losses) == 3

    def test_train_soft_losses_length_equals_epochs(self):
        cfg = _default_cfg(epochs=3)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert len(metrics.train_soft_losses) == 3

    def test_train_hard_losses_populated_when_blended(self):
        """alpha < 1.0 triggers a hard CE loss component each epoch."""
        cfg = _default_cfg(epochs=3, alpha=0.5)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert len(metrics.train_hard_losses) == 3
        assert all(np.isfinite(v) for v in metrics.train_hard_losses)

    def test_train_hard_losses_empty_in_pure_soft_mode(self):
        """alpha == 1.0 → pure soft distillation → no hard loss entries."""
        cfg = _default_cfg(epochs=2, alpha=1.0)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert metrics.train_hard_losses == []

    def test_val_losses_length_equals_epochs_when_val_set(self):
        cfg = _default_cfg(epochs=3, val_fraction=0.2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states(100))
        assert len(metrics.val_losses) == 3

    def test_no_val_losses_when_val_fraction_zero(self):
        cfg = _default_cfg(val_fraction=0.0)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert metrics.val_losses == []

    def test_best_val_loss_is_finite_when_no_val_set(self):
        """When val_fraction==0, best_val_loss must be finite (not inf)."""
        cfg = _default_cfg(val_fraction=0.0)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert np.isfinite(metrics.best_val_loss)

    def test_action_agreements_populated_with_val_set(self):
        cfg = _default_cfg(epochs=2, val_fraction=0.2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states(100))
        assert len(metrics.action_agreements) == 2
        for ag in metrics.action_agreements:
            assert 0.0 <= ag <= 1.0

    def test_mean_prob_similarities_populated_with_val_set(self):
        """Calibration metric populated each epoch when a val set is present."""
        cfg = _default_cfg(epochs=2, val_fraction=0.2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states(100))
        assert len(metrics.mean_prob_similarities) == 2
        for sim in metrics.mean_prob_similarities:
            assert 0.0 <= sim <= 1.0

    def test_mean_prob_similarities_empty_without_val_set(self):
        cfg = _default_cfg(val_fraction=0.0)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert metrics.mean_prob_similarities == []

    def test_train_losses_are_finite(self):
        trainer = DistillationTrainer(_make_teacher(), _make_student(), _default_cfg())
        metrics = trainer.train(_make_states())
        assert all(np.isfinite(loss_val) for loss_val in metrics.train_losses)

    def test_val_losses_are_finite(self):
        cfg = _default_cfg(val_fraction=0.2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert all(np.isfinite(loss_val) for loss_val in metrics.val_losses)

    def test_best_val_loss_leq_all_val_losses(self):
        cfg = _default_cfg(epochs=4, val_fraction=0.2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert metrics.best_val_loss <= min(metrics.val_losses)

    def test_best_epoch_in_valid_range(self):
        cfg = _default_cfg(epochs=3, val_fraction=0.2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert 0 <= metrics.best_epoch < 3

    def test_student_weights_updated_after_training(self):
        teacher = _make_teacher()
        student = _make_student()
        initial_student_params = {k: v.clone() for k, v in student.state_dict().items()}
        trainer = DistillationTrainer(teacher, student, _default_cfg())
        trainer.train(_make_states())
        changed = any(
            not torch.equal(v, initial_student_params[k]) for k, v in student.state_dict().items()
        )
        assert changed

    def test_teacher_weights_unchanged_after_training(self):
        teacher = _make_teacher()
        student = _make_student()
        initial_teacher_params = {k: v.clone() for k, v in teacher.state_dict().items()}
        trainer = DistillationTrainer(teacher, student, _default_cfg())
        trainer.train(_make_states())
        for k, v in teacher.state_dict().items():
            assert torch.equal(v, initial_teacher_params[k])

    def test_reproducible_with_seed(self):
        states = _make_states()
        cfg1 = _default_cfg(seed=7)
        cfg2 = _default_cfg(seed=7)

        teacher1 = BaseQNetwork(INPUT_DIM, OUTPUT_DIM, PARENT_HIDDEN)
        teacher2 = BaseQNetwork(INPUT_DIM, OUTPUT_DIM, PARENT_HIDDEN)
        # Use the same teacher weights for a fair comparison
        teacher2.load_state_dict(teacher1.state_dict())

        student1 = StudentQNetwork(INPUT_DIM, OUTPUT_DIM, PARENT_HIDDEN)
        student2 = StudentQNetwork(INPUT_DIM, OUTPUT_DIM, PARENT_HIDDEN)
        student2.load_state_dict(student1.state_dict())

        DistillationTrainer(teacher1, student1, cfg1).train(states)
        DistillationTrainer(teacher2, student2, cfg2).train(states)

        for k in student1.state_dict():
            assert torch.allclose(student1.state_dict()[k], student2.state_dict()[k])


# ---------------------------------------------------------------------------
# KL vs MSE loss functions
# ---------------------------------------------------------------------------


class TestDistillationLossFunctions:
    def test_kl_loss_is_non_negative(self):
        cfg = _default_cfg(loss_fn="kl")
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert all(loss_val >= 0 for loss_val in metrics.train_losses)

    def test_mse_loss_fn(self):
        cfg = _default_cfg(loss_fn="mse", epochs=2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert len(metrics.train_losses) == 2
        assert all(np.isfinite(loss_val) for loss_val in metrics.train_losses)

    def test_mse_produces_no_hard_losses(self):
        """MSE mode has no hard CE component."""
        cfg = _default_cfg(loss_fn="mse", alpha=0.5, epochs=2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert metrics.train_hard_losses == []

    def test_alpha_blend_with_hard_loss(self):
        """alpha=0.5 → 50% soft, 50% hard; both components finite."""
        cfg = _default_cfg(alpha=0.5, epochs=2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert all(np.isfinite(loss_val) for loss_val in metrics.train_losses)
        assert all(np.isfinite(v) for v in metrics.train_hard_losses)

    def test_pure_soft_distillation_alpha_one(self):
        """alpha=1.0 → pure soft distillation; losses finite, no hard component."""
        cfg = _default_cfg(alpha=1.0, epochs=2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert all(np.isfinite(loss_val) for loss_val in metrics.train_losses)
        assert metrics.train_hard_losses == []

    def test_pure_hard_loss_alpha_zero(self):
        """alpha=0.0 → pure hard CE supervision; hard losses populated."""
        cfg = _default_cfg(alpha=0.0, epochs=2)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert all(np.isfinite(loss_val) for loss_val in metrics.train_losses)
        assert len(metrics.train_hard_losses) == 2


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestDistillationCheckpointing:
    def test_checkpoint_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "student.pt")
            trainer = DistillationTrainer(_make_teacher(), _make_student(), _default_cfg())
            trainer.train(_make_states(), checkpoint_path=ckpt)
            assert os.path.isfile(ckpt)

    def test_metadata_json_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "student.pt")
            trainer = DistillationTrainer(_make_teacher(), _make_student(), _default_cfg())
            trainer.train(_make_states(), checkpoint_path=ckpt)
            meta = ckpt + ".json"
            assert os.path.isfile(meta)

    def test_metadata_json_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "student.pt")
            cfg = _default_cfg(temperature=5.0, alpha=0.1, loss_fn="mse", temp_decay=0.9)
            trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
            trainer.train(_make_states(), checkpoint_path=ckpt)
            with open(ckpt + ".json") as fh:
                meta = json.load(fh)
            assert meta["config"]["temperature"] == 5.0
            assert meta["config"]["temp_decay"] == 0.9
            assert meta["config"]["alpha"] == 0.1
            assert meta["config"]["loss_fn"] == "mse"
            assert "metrics" in meta
            assert "train_losses" in meta["metrics"]
            assert "train_soft_losses" in meta["metrics"]
            assert "mean_prob_similarities" in meta["metrics"]

    def test_checkpoint_loadable_as_state_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "student.pt")
            student = _make_student()
            trainer = DistillationTrainer(_make_teacher(), student, _default_cfg())
            trainer.train(_make_states(), checkpoint_path=ckpt)

            loaded = torch.load(ckpt, map_location="cpu", weights_only=True)
            new_student = _make_student()
            new_student.load_state_dict(loaded)  # should not raise

    def test_metadata_json_no_infinity_when_no_val_set(self):
        """best_val_loss must be a finite JSON number even when val_fraction==0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "student.pt")
            cfg = _default_cfg(val_fraction=0.0)
            trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
            trainer.train(_make_states(), checkpoint_path=ckpt)
            # json.load would raise if the file contains 'Infinity'
            with open(ckpt + ".json") as fh:
                meta = json.load(fh)
            assert np.isfinite(meta["metrics"]["best_val_loss"])

    def test_no_checkpoint_when_path_none(self):
        """Training without a checkpoint_path should not create any files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_files = set(os.listdir(tmpdir))
            trainer = DistillationTrainer(_make_teacher(), _make_student(), _default_cfg())
            trainer.train(_make_states())
            assert set(os.listdir(tmpdir)) == original_files


# ---------------------------------------------------------------------------
# Temperature decay schedule
# ---------------------------------------------------------------------------


class TestTemperatureDecay:
    def test_no_decay_leaves_temperature_unchanged(self):
        """temp_decay=1.0 should not alter the effective temperature."""
        cfg = _default_cfg(temperature=4.0, temp_decay=1.0, epochs=3, val_fraction=0.0)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        trainer.train(_make_states())
        # After 3 epochs with decay=1.0, temperature is still 4.0 * 1.0^3 = 4.0
        assert trainer._current_temperature == pytest.approx(4.0)

    def test_decay_reduces_temperature_over_epochs(self):
        """After N epochs with temp_decay < 1, temperature should be reduced."""
        initial_temp = 4.0
        decay = 0.9
        epochs = 3
        cfg = _default_cfg(temperature=initial_temp, temp_decay=decay, epochs=epochs, val_fraction=0.0)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        trainer.train(_make_states())
        expected = initial_temp * (decay**epochs)
        assert trainer._current_temperature == pytest.approx(expected, rel=1e-5)

    def test_temperature_reset_on_second_train_call(self):
        """Calling train() twice resets the temperature to the configured value.

        Without a reset, the second run would start from the decayed value
        (4.0 * 0.5^2 = 1.0) and decay further to 1.0 * 0.5^2 = 0.25.
        With a proper reset each run starts from 4.0, so both runs end at
        4.0 * 0.5^2 = 1.0.
        """
        initial_temp = 4.0
        decay = 0.5
        epochs = 2
        expected_final = initial_temp * (decay**epochs)  # 1.0

        cfg = _default_cfg(temperature=initial_temp, temp_decay=decay, epochs=epochs, val_fraction=0.0)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        trainer.train(_make_states())
        assert trainer._current_temperature == pytest.approx(expected_final, rel=1e-5)

        # Second run: temperature is reset to initial_temp, then decays again
        trainer.train(_make_states())
        assert trainer._current_temperature == pytest.approx(expected_final, rel=1e-5)

    def test_decayed_training_produces_finite_losses(self):
        cfg = _default_cfg(temperature=5.0, temp_decay=0.8, epochs=4)
        trainer = DistillationTrainer(_make_teacher(), _make_student(), cfg)
        metrics = trainer.train(_make_states())
        assert all(np.isfinite(v) for v in metrics.train_losses)


# ---------------------------------------------------------------------------
# evaluate_agreement
# ---------------------------------------------------------------------------


class TestEvaluateAgreement:
    def test_returns_float_in_unit_interval(self):
        trainer = DistillationTrainer(_make_teacher(), _make_student(), _default_cfg())
        trainer.train(_make_states())
        ag = trainer.evaluate_agreement(_make_states(50))
        assert isinstance(ag, float)
        assert 0.0 <= ag <= 1.0

    def test_identical_networks_have_full_agreement(self):
        teacher = _make_teacher()
        # Student with same architecture and weights as teacher
        student_net = BaseQNetwork(INPUT_DIM, OUTPUT_DIM, PARENT_HIDDEN)
        student_net.load_state_dict(teacher.state_dict())
        cfg = _default_cfg(epochs=1)
        trainer = DistillationTrainer(teacher, student_net, cfg)
        # Manually copy teacher weights to student so agreement starts at 1.0
        trainer.student.load_state_dict(teacher.state_dict())
        ag = trainer.evaluate_agreement(_make_states(100))
        assert ag == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Student vs parent parameter count
# ---------------------------------------------------------------------------


class TestStudentSmallerThanParent:
    def test_student_has_fewer_params(self):
        teacher = _make_teacher()
        student = _make_student()
        parent_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        assert student_params < parent_params

    def test_student_approx_half_hidden_width(self):
        student = _make_student()
        first_linear = student.network[0]
        assert first_linear.out_features == max(16, PARENT_HIDDEN // 2)


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


def test_package_imports():
    from farm.core.decision.training import (  # noqa: F401
        DistillationConfig,
        DistillationMetrics,
        DistillationTrainer,
    )
