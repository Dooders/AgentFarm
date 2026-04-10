"""Tests for farm.core.decision.training.holdout_utils.

Covers:
- split_replay_buffer: basic split ratios, determinism, shuffle=False,
  error handling for bad inputs.
- apply_gaussian_noise: shape preservation, zero-std copy, seeded
  reproducibility, error on negative std.
- apply_input_scaling: shape preservation, identity at scale=1.0,
  error on non-finite scale.
- make_shifted_states: factory dispatch for both shift types, error on
  unknown type.
- __init__ re-exports.
"""

from __future__ import annotations

import numpy as np
import pytest

from farm.core.decision.training.holdout_utils import (
    SHIFT_TYPES,
    apply_gaussian_noise,
    apply_input_scaling,
    make_shifted_states,
    split_replay_buffer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 8
SEED = 42


def _make_states(n: int = 200, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, INPUT_DIM)).astype("float32")


# ===========================================================================
# split_replay_buffer
# ===========================================================================


class TestSplitReplayBuffer:
    """Tests for split_replay_buffer."""

    def test_basic_split_sizes(self):
        states = _make_states(100)
        id_s, hold_s = split_replay_buffer(states, holdout_fraction=0.2, seed=SEED)
        assert id_s.shape[0] + hold_s.shape[0] == 100
        assert hold_s.shape[0] == 20
        assert id_s.shape[1] == INPUT_DIM
        assert hold_s.shape[1] == INPUT_DIM

    def test_dtype_preserved_as_float32(self):
        states = _make_states(50).astype("float64")  # non-float32 input
        id_s, hold_s = split_replay_buffer(states, holdout_fraction=0.2, seed=SEED)
        assert id_s.dtype == np.float32
        assert hold_s.dtype == np.float32

    def test_no_overlap(self):
        """ID and holdout rows must be disjoint."""
        states = _make_states(100)
        id_s, hold_s = split_replay_buffer(states, holdout_fraction=0.3, seed=SEED)
        # Convert to sets of tuple rows for equality check.
        id_rows = {tuple(row.tolist()) for row in id_s}
        hold_rows = {tuple(row.tolist()) for row in hold_s}
        assert id_rows.isdisjoint(hold_rows)

    def test_all_rows_accounted_for(self):
        """Every row in the original array must appear in exactly one split."""
        states = _make_states(80)
        id_s, hold_s = split_replay_buffer(states, holdout_fraction=0.25, seed=SEED)
        combined = np.concatenate([id_s, hold_s], axis=0)
        # Sort both to compare regardless of shuffle ordering.
        original_sorted = np.sort(states.flatten())
        combined_sorted = np.sort(combined.flatten())
        np.testing.assert_array_almost_equal(original_sorted, combined_sorted)

    def test_deterministic_with_same_seed(self):
        states = _make_states(100)
        id1, hold1 = split_replay_buffer(states, holdout_fraction=0.2, seed=7)
        id2, hold2 = split_replay_buffer(states, holdout_fraction=0.2, seed=7)
        np.testing.assert_array_equal(id1, id2)
        np.testing.assert_array_equal(hold1, hold2)

    def test_different_seeds_give_different_splits(self):
        states = _make_states(100)
        id1, _ = split_replay_buffer(states, holdout_fraction=0.2, seed=1)
        id2, _ = split_replay_buffer(states, holdout_fraction=0.2, seed=2)
        assert not np.array_equal(id1, id2)

    def test_shuffle_false_preserves_row_order(self):
        """With shuffle=False the first N-k rows go to ID, last k to holdout."""
        states = _make_states(10)
        id_s, hold_s = split_replay_buffer(states, holdout_fraction=0.3, shuffle=False)
        expected_n_hold = max(1, int(round(10 * 0.3)))  # 3
        expected_n_id = 10 - expected_n_hold
        np.testing.assert_array_equal(id_s, states[:expected_n_id])
        np.testing.assert_array_equal(hold_s, states[expected_n_id:])

    def test_holdout_fraction_boundary_low(self):
        """holdout_fraction must be strictly > 0."""
        states = _make_states(50)
        with pytest.raises(ValueError, match="holdout_fraction"):
            split_replay_buffer(states, holdout_fraction=0.0, seed=SEED)

    def test_holdout_fraction_boundary_high(self):
        """holdout_fraction must be strictly < 1."""
        states = _make_states(50)
        with pytest.raises(ValueError, match="holdout_fraction"):
            split_replay_buffer(states, holdout_fraction=1.0, seed=SEED)

    def test_holdout_fraction_too_large_for_data(self):
        """A very large fraction leaves 0 training rows."""
        states = _make_states(1)
        with pytest.raises(ValueError, match="holdout set"):
            split_replay_buffer(states, holdout_fraction=0.99, seed=SEED)

    def test_1d_states_raises(self):
        bad = np.ones(10, dtype="float32")
        with pytest.raises(ValueError, match="2-D"):
            split_replay_buffer(bad)

    def test_empty_states_raises(self):
        bad = np.empty((0, INPUT_DIM), dtype="float32")
        with pytest.raises(ValueError, match="non-empty"):
            split_replay_buffer(bad)


# ===========================================================================
# apply_gaussian_noise
# ===========================================================================


class TestApplyGaussianNoise:
    """Tests for apply_gaussian_noise."""

    def test_shape_preserved(self):
        states = _make_states(50)
        out = apply_gaussian_noise(states, std=0.1, seed=SEED)
        assert out.shape == states.shape

    def test_dtype_float32(self):
        states = _make_states(50)
        out = apply_gaussian_noise(states, std=0.1, seed=SEED)
        assert out.dtype == np.float32

    def test_zero_std_returns_copy(self):
        states = _make_states(50)
        out = apply_gaussian_noise(states, std=0.0, seed=SEED)
        np.testing.assert_array_equal(out, states)
        assert out is not states  # must be a copy

    def test_nonzero_std_modifies_values(self):
        states = _make_states(50)
        out = apply_gaussian_noise(states, std=1.0, seed=SEED)
        assert not np.array_equal(out, states)

    def test_seeded_reproducibility(self):
        states = _make_states(50)
        out1 = apply_gaussian_noise(states, std=0.5, seed=99)
        out2 = apply_gaussian_noise(states, std=0.5, seed=99)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        states = _make_states(50)
        out1 = apply_gaussian_noise(states, std=0.5, seed=1)
        out2 = apply_gaussian_noise(states, std=0.5, seed=2)
        assert not np.array_equal(out1, out2)

    def test_negative_std_raises(self):
        states = _make_states(10)
        with pytest.raises(ValueError, match="non-negative"):
            apply_gaussian_noise(states, std=-0.1)

    def test_1d_states_raises(self):
        bad = np.ones(10, dtype="float32")
        with pytest.raises(ValueError, match="2-D"):
            apply_gaussian_noise(bad, std=0.1)

    def test_empty_states_raises(self):
        bad = np.empty((0, INPUT_DIM), dtype="float32")
        with pytest.raises(ValueError, match="non-empty"):
            apply_gaussian_noise(bad, std=0.1)


# ===========================================================================
# apply_input_scaling
# ===========================================================================


class TestApplyInputScaling:
    """Tests for apply_input_scaling."""

    def test_shape_preserved(self):
        states = _make_states(50)
        out = apply_input_scaling(states, scale_factor=2.0)
        assert out.shape == states.shape

    def test_dtype_float32(self):
        states = _make_states(50)
        out = apply_input_scaling(states, scale_factor=3.0)
        assert out.dtype == np.float32

    def test_identity_at_scale_one(self):
        states = _make_states(50)
        out = apply_input_scaling(states, scale_factor=1.0)
        np.testing.assert_array_almost_equal(out, states)

    def test_values_scaled_correctly(self):
        states = _make_states(50)
        out = apply_input_scaling(states, scale_factor=2.0)
        np.testing.assert_array_almost_equal(out, states * 2.0)

    def test_zero_scale_gives_zeros(self):
        states = _make_states(50)
        out = apply_input_scaling(states, scale_factor=0.0)
        np.testing.assert_array_equal(out, np.zeros_like(states))

    def test_non_finite_scale_raises(self):
        states = _make_states(10)
        with pytest.raises(ValueError, match="finite"):
            apply_input_scaling(states, scale_factor=float("inf"))

    def test_nan_scale_raises(self):
        states = _make_states(10)
        with pytest.raises(ValueError, match="finite"):
            apply_input_scaling(states, scale_factor=float("nan"))

    def test_1d_states_raises(self):
        bad = np.ones(10, dtype="float32")
        with pytest.raises(ValueError, match="2-D"):
            apply_input_scaling(bad, scale_factor=2.0)

    def test_empty_states_raises(self):
        bad = np.empty((0, INPUT_DIM), dtype="float32")
        with pytest.raises(ValueError, match="non-empty"):
            apply_input_scaling(bad, scale_factor=2.0)


# ===========================================================================
# make_shifted_states
# ===========================================================================


class TestMakeShiftedStates:
    """Tests for make_shifted_states factory."""

    def test_gaussian_noise_dispatch(self):
        states = _make_states(50)
        out = make_shifted_states(states, "gaussian_noise", std=0.1, seed=SEED)
        expected = apply_gaussian_noise(states, std=0.1, seed=SEED)
        np.testing.assert_array_equal(out, expected)

    def test_input_scaling_dispatch(self):
        states = _make_states(50)
        out = make_shifted_states(states, "input_scaling", scale_factor=2.0)
        expected = apply_input_scaling(states, scale_factor=2.0)
        np.testing.assert_array_equal(out, expected)

    def test_unknown_shift_type_raises(self):
        states = _make_states(10)
        with pytest.raises(ValueError, match="Unknown shift_type"):
            make_shifted_states(states, "random_projection")

    def test_shift_types_constant(self):
        assert "gaussian_noise" in SHIFT_TYPES
        assert "input_scaling" in SHIFT_TYPES


# ===========================================================================
# __init__ re-exports
# ===========================================================================


def test_init_exports():
    """Ensure public helpers are accessible from the training package."""
    from farm.core.decision.training import (  # noqa: F401
        SHIFT_TYPES,
        apply_gaussian_noise,
        apply_input_scaling,
        make_shifted_states,
        split_replay_buffer,
    )
