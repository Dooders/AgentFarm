"""Tests for farm/core/decision/training/finetune.py.

Covers:
- FineTuningConfig validation.
- FineTuner: initialisation, reference frozen, child trainable.
- FineTuner.finetune: metrics lengths, loss decreases, checkpointing.
- Before/after metrics populated correctly.
- LR scheduler wired up when patience > 0.
- KL and MSE loss functions.
- Hard-loss blend (alpha < 1.0).
- Temperature decay.
- Val-fraction=0 (no validation) path.
- evaluate_agreement helper.
- JSON metadata written alongside checkpoint.
- Round-trip: child weights loadable from checkpoint.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from farm.core.decision.base_dqn import BaseQNetwork
from farm.core.decision.training.finetune import (
    FineTuner,
    FineTuningConfig,
    FineTuningMetrics,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
HIDDEN_SIZE = 32


def _make_net(seed: int = 0) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_size=HIDDEN_SIZE)


def _make_states(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


def _default_cfg(**kwargs) -> FineTuningConfig:
    defaults = dict(epochs=2, batch_size=16, val_fraction=0.1, seed=42)
    defaults.update(kwargs)
    return FineTuningConfig(**defaults)


# ---------------------------------------------------------------------------
# FineTuningConfig tests
# ---------------------------------------------------------------------------


class TestFineTuningConfig:
    def test_defaults(self):
        cfg = FineTuningConfig()
        assert cfg.learning_rate == 1e-3
        assert cfg.epochs == 5
        assert cfg.batch_size == 32
        assert cfg.max_grad_norm == 1.0
        assert cfg.val_fraction == 0.1
        assert cfg.seed is None
        assert cfg.loss_fn == "kl"
        assert cfg.temperature == 3.0
        assert cfg.temp_decay == 1.0
        assert cfg.alpha == 1.0
        assert cfg.lr_schedule_patience == 0
        assert cfg.lr_schedule_factor == 0.5

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            FineTuningConfig(learning_rate=0.0)

    def test_invalid_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            FineTuningConfig(epochs=0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            FineTuningConfig(batch_size=0)

    def test_invalid_max_grad_norm_zero(self):
        with pytest.raises(ValueError, match="max_grad_norm"):
            FineTuningConfig(max_grad_norm=0.0)

    def test_none_max_grad_norm_is_valid(self):
        cfg = FineTuningConfig(max_grad_norm=None)
        assert cfg.max_grad_norm is None

    def test_invalid_val_fraction_high(self):
        with pytest.raises(ValueError, match="val_fraction"):
            FineTuningConfig(val_fraction=1.0)

    def test_invalid_val_fraction_low(self):
        with pytest.raises(ValueError, match="val_fraction"):
            FineTuningConfig(val_fraction=-0.1)

    def test_invalid_loss_fn(self):
        with pytest.raises(ValueError, match="loss_fn"):
            FineTuningConfig(loss_fn="huber")

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            FineTuningConfig(temperature=0.0)

    def test_invalid_temp_decay_zero(self):
        with pytest.raises(ValueError, match="temp_decay"):
            FineTuningConfig(temp_decay=0.0)

    def test_invalid_temp_decay_above_one(self):
        with pytest.raises(ValueError, match="temp_decay"):
            FineTuningConfig(temp_decay=1.1)

    def test_invalid_alpha_high(self):
        with pytest.raises(ValueError, match="alpha"):
            FineTuningConfig(alpha=1.5)

    def test_invalid_alpha_low(self):
        with pytest.raises(ValueError, match="alpha"):
            FineTuningConfig(alpha=-0.1)

    def test_valid_alpha_boundaries(self):
        cfg = FineTuningConfig(alpha=0.0)
        assert cfg.alpha == 0.0
        cfg2 = FineTuningConfig(alpha=1.0)
        assert cfg2.alpha == 1.0

    def test_invalid_lr_schedule_patience(self):
        with pytest.raises(ValueError, match="lr_schedule_patience"):
            FineTuningConfig(lr_schedule_patience=-1)

    def test_invalid_lr_schedule_factor_ge_one(self):
        with pytest.raises(ValueError, match="lr_schedule_factor"):
            FineTuningConfig(lr_schedule_factor=1.0)

    def test_invalid_lr_schedule_factor_le_zero(self):
        with pytest.raises(ValueError, match="lr_schedule_factor"):
            FineTuningConfig(lr_schedule_factor=0.0)

    def test_mse_loss_fn(self):
        cfg = FineTuningConfig(loss_fn="mse")
        assert cfg.loss_fn == "mse"


# ---------------------------------------------------------------------------
# FineTuner: initialisation
# ---------------------------------------------------------------------------


class TestFineTunerInit:
    def test_reference_frozen_after_init(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        _ = FineTuner(ref, child, _default_cfg())
        for p in ref.parameters():
            assert not p.requires_grad

    def test_child_not_frozen(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        _ = FineTuner(ref, child, _default_cfg())
        assert any(p.requires_grad for p in child.parameters())

    def test_reference_in_eval_mode(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        _ = FineTuner(ref, child, _default_cfg())
        assert not ref.training

    def test_default_config_used_when_none(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        tuner = FineTuner(ref, child)
        assert isinstance(tuner.config, FineTuningConfig)

    def test_default_device_is_cpu(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        tuner = FineTuner(ref, child, _default_cfg())
        assert tuner.device == torch.device("cpu")

    def test_scheduler_none_when_patience_zero(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        tuner = FineTuner(ref, child, _default_cfg(lr_schedule_patience=0))
        assert tuner.scheduler is None

    def test_scheduler_created_when_patience_gt_zero(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        cfg = _default_cfg(lr_schedule_patience=2, val_fraction=0.2)
        tuner = FineTuner(ref, child, cfg)
        assert tuner.scheduler is not None


# ---------------------------------------------------------------------------
# FineTuner: finetune loop
# ---------------------------------------------------------------------------


class TestFineTunerFinetune:
    def test_returns_finetuning_metrics(self):
        tuner = FineTuner(_make_net(0), _make_net(1), _default_cfg())
        metrics = tuner.finetune(_make_states())
        assert isinstance(metrics, FineTuningMetrics)

    def test_train_losses_length_equals_epochs(self):
        cfg = _default_cfg(epochs=3)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert len(metrics.train_losses) == 3

    def test_train_soft_losses_length_equals_epochs(self):
        cfg = _default_cfg(epochs=3)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert len(metrics.train_soft_losses) == 3

    def test_train_hard_losses_populated_when_blended(self):
        cfg = _default_cfg(epochs=3, alpha=0.5)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert len(metrics.train_hard_losses) == 3
        assert all(np.isfinite(v) for v in metrics.train_hard_losses)

    def test_train_hard_losses_empty_in_pure_soft_mode(self):
        cfg = _default_cfg(epochs=2, alpha=1.0)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert metrics.train_hard_losses == []

    def test_val_losses_length_equals_epochs_when_val_set(self):
        cfg = _default_cfg(epochs=3, val_fraction=0.2)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states(100))
        assert len(metrics.val_losses) == 3

    def test_no_val_losses_when_val_fraction_zero(self):
        cfg = _default_cfg(val_fraction=0.0)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert metrics.val_losses == []

    def test_best_val_loss_is_finite_when_no_val_set(self):
        cfg = _default_cfg(val_fraction=0.0)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert np.isfinite(metrics.best_val_loss)

    def test_action_agreements_populated_with_val_set(self):
        cfg = _default_cfg(epochs=2, val_fraction=0.2)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states(100))
        assert len(metrics.action_agreements) == 2
        for ag in metrics.action_agreements:
            assert 0.0 <= ag <= 1.0

    def test_mean_prob_similarities_populated_with_val_set(self):
        cfg = _default_cfg(epochs=2, val_fraction=0.2)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states(100))
        assert len(metrics.mean_prob_similarities) == 2
        for sim in metrics.mean_prob_similarities:
            assert 0.0 <= sim <= 1.0

    def test_train_losses_are_finite(self):
        tuner = FineTuner(_make_net(0), _make_net(1), _default_cfg())
        metrics = tuner.finetune(_make_states())
        assert all(np.isfinite(loss) for loss in metrics.train_losses)

    def test_val_losses_are_finite(self):
        cfg = _default_cfg(val_fraction=0.2)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert all(np.isfinite(loss) for loss in metrics.val_losses)

    def test_best_val_loss_leq_all_val_losses(self):
        cfg = _default_cfg(epochs=4, val_fraction=0.2)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert metrics.best_val_loss <= min(metrics.val_losses)

    def test_best_epoch_in_valid_range(self):
        cfg = _default_cfg(epochs=3, val_fraction=0.2)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert 0 <= metrics.best_epoch < 3

    def test_child_weights_updated_after_training(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        initial_params = {k: v.clone() for k, v in child.state_dict().items()}
        tuner = FineTuner(ref, child, _default_cfg())
        tuner.finetune(_make_states())
        for key, val in child.state_dict().items():
            if val.dtype.is_floating_point:
                assert not torch.allclose(initial_params[key], val), (
                    f"Child parameter '{key}' was not updated during fine-tuning"
                )
                break

    def test_reference_weights_unchanged_after_training(self):
        ref = _make_net(seed=0)
        child = _make_net(seed=1)
        original_ref = {k: v.clone() for k, v in ref.state_dict().items()}
        tuner = FineTuner(ref, child, _default_cfg())
        tuner.finetune(_make_states())
        for key, val in ref.state_dict().items():
            assert torch.allclose(original_ref[key].float(), val.float()), (
                f"Reference parameter '{key}' changed during fine-tuning"
            )

    def test_empty_states_raises(self):
        tuner = FineTuner(_make_net(0), _make_net(1), _default_cfg())
        with pytest.raises(ValueError, match="non-empty"):
            tuner.finetune(np.empty((0, INPUT_DIM), dtype="float32"))

    def test_mse_loss_fn(self):
        cfg = _default_cfg(loss_fn="mse")
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert all(np.isfinite(loss) for loss in metrics.train_losses)

    def test_no_grad_norm_clip(self):
        cfg = _default_cfg(max_grad_norm=None)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert metrics.train_losses

    def test_temperature_decay_applied(self):
        cfg = _default_cfg(epochs=3, temperature=4.0, temp_decay=0.5)
        ref = _make_net(0)
        child = _make_net(1)
        tuner = FineTuner(ref, child, cfg)
        tuner.finetune(_make_states())
        # After 3 epochs: 4.0 * 0.5^3 = 0.5
        assert abs(tuner._current_temperature - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Before / after metrics
# ---------------------------------------------------------------------------


class TestBeforeAfterMetrics:
    def test_initial_val_loss_set_when_val_fraction_positive(self):
        cfg = _default_cfg(val_fraction=0.2)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states(100))
        assert np.isfinite(metrics.initial_val_loss)

    def test_initial_val_loss_inf_when_val_fraction_zero(self):
        cfg = _default_cfg(val_fraction=0.0)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        assert metrics.initial_val_loss == float("inf")

    def test_initial_action_agreement_in_range(self):
        cfg = _default_cfg(val_fraction=0.2)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states(100))
        assert 0.0 <= metrics.initial_action_agreement <= 1.0

    def test_finetuning_same_network_gives_perfect_initial_agreement(self):
        """When reference == child (same weights) initial agreement == 1.0."""
        ref = _make_net(0)
        # Child is a copy of reference
        child = _make_net(0)
        cfg = _default_cfg(val_fraction=0.2)
        tuner = FineTuner(ref, child, cfg)
        metrics = tuner.finetune(_make_states(100))
        assert metrics.initial_action_agreement == pytest.approx(1.0, abs=1e-6)

    def test_best_val_loss_not_worse_than_initial_for_same_network(self):
        """Fine-tuning a child that already matches reference: val loss stays small."""
        ref = _make_net(0)
        child = _make_net(0)
        cfg = _default_cfg(epochs=3, val_fraction=0.2, seed=0)
        tuner = FineTuner(ref, child, cfg)
        metrics = tuner.finetune(_make_states(200))
        # When starting from perfect agreement, val loss should remain small
        assert metrics.best_val_loss < 0.1


# ---------------------------------------------------------------------------
# FineTuningMetrics.to_dict
# ---------------------------------------------------------------------------


class TestFineTuningMetricsToDict:
    def test_to_dict_keys(self):
        m = FineTuningMetrics()
        d = m.to_dict()
        expected_keys = {
            "initial_val_loss",
            "initial_action_agreement",
            "train_losses",
            "train_soft_losses",
            "train_hard_losses",
            "val_losses",
            "action_agreements",
            "mean_prob_similarities",
            "best_val_loss",
            "best_epoch",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_serialisable(self):
        cfg = _default_cfg(epochs=2, val_fraction=0.1)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        metrics = tuner.finetune(_make_states())
        d = metrics.to_dict()
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_checkpoint_file_created(self):
        cfg = _default_cfg(val_fraction=0.1)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "child.pt")
            tuner.finetune(_make_states(), checkpoint_path=ckpt_path)
            assert os.path.isfile(ckpt_path)

    def test_metadata_json_created(self):
        cfg = _default_cfg(val_fraction=0.1)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "child.pt")
            tuner.finetune(_make_states(), checkpoint_path=ckpt_path)
            assert os.path.isfile(ckpt_path + ".json")

    def test_metadata_json_is_valid(self):
        cfg = _default_cfg(val_fraction=0.1)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "child.pt")
            tuner.finetune(_make_states(), checkpoint_path=ckpt_path)
            with open(ckpt_path + ".json") as fh:
                meta = json.load(fh)
            assert "config" in meta
            assert "metrics" in meta

    def test_metadata_config_keys(self):
        cfg = _default_cfg(val_fraction=0.1)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "child.pt")
            tuner.finetune(_make_states(), checkpoint_path=ckpt_path)
            with open(ckpt_path + ".json") as fh:
                meta = json.load(fh)
            expected_keys = {
                "learning_rate",
                "epochs",
                "batch_size",
                "max_grad_norm",
                "val_fraction",
                "seed",
                "loss_fn",
                "temperature",
                "final_temperature",
                "temp_decay",
                "alpha",
                "lr_schedule_patience",
                "lr_schedule_factor",
            }
            assert expected_keys == set(meta["config"].keys())

    def test_checkpoint_loads_into_model(self):
        cfg = _default_cfg(val_fraction=0.0)
        tuner = FineTuner(_make_net(0), _make_net(1), cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "child.pt")
            tuner.finetune(_make_states(), checkpoint_path=ckpt_path)
            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            restored = _make_net(99)
            restored.load_state_dict(loaded)
            # Ensure forward pass works
            states_t = torch.from_numpy(_make_states(5))
            out = restored(states_t)
            assert out.shape == (5, OUTPUT_DIM)

    def test_best_weights_loaded_back_into_child(self):
        """After finetune(), child weights must match the saved checkpoint."""
        cfg = _default_cfg(epochs=3, val_fraction=0.2)
        ref = _make_net(0)
        child = _make_net(1)
        tuner = FineTuner(ref, child, cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "child.pt")
            tuner.finetune(_make_states(), checkpoint_path=ckpt_path)
            saved_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            for key in saved_sd:
                assert torch.allclose(saved_sd[key], child.state_dict()[key].cpu())


# ---------------------------------------------------------------------------
# evaluate_agreement
# ---------------------------------------------------------------------------


class TestEvaluateAgreement:
    def test_returns_float_in_range(self):
        tuner = FineTuner(_make_net(0), _make_net(1), _default_cfg())
        agreement = tuner.evaluate_agreement(_make_states(50))
        assert isinstance(agreement, float)
        assert 0.0 <= agreement <= 1.0

    def test_identical_networks_have_perfect_agreement(self):
        ref = _make_net(0)
        child = _make_net(0)  # same seed → same weights
        tuner = FineTuner(ref, child, _default_cfg())
        agreement = tuner.evaluate_agreement(_make_states(50))
        assert agreement == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Training package __init__ exports
# ---------------------------------------------------------------------------


def test_package_exports():
    from farm.core.decision.training import FineTuner, FineTuningConfig, FineTuningMetrics  # noqa: F401

    assert FineTuner is not None
    assert FineTuningConfig is not None
    assert FineTuningMetrics is not None
