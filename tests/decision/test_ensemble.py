"""Tests for farm/core/decision/algorithms/ensemble.py.

Covers RandomForestActionSelector, NaiveBayesActionSelector,
KNNActionSelector, and GradientBoostActionSelector (skipped when neither
xgboost nor lightgbm is available).
"""

import numpy as np
import pytest

from farm.core.decision.algorithms.ensemble import (
    GradientBoostActionSelector,
    KNNActionSelector,
    NaiveBayesActionSelector,
    RandomForestActionSelector,
    _align_proba,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

NUM_ACTIONS = 3
RNG = np.random.default_rng(42)

# Simple linearly-separable dataset: action i for states near i*10
_STATES = np.vstack([
    RNG.normal(loc=i * 10, scale=0.5, size=(20, 4))
    for i in range(NUM_ACTIONS)
]).astype(float)
_ACTIONS = np.repeat(np.arange(NUM_ACTIONS), 20)


# ---------------------------------------------------------------------------
# _align_proba utility
# ---------------------------------------------------------------------------

class TestAlignProba:
    def test_perfect_alignment(self):
        classes = np.array([0, 1, 2])
        proba_row = np.array([0.2, 0.5, 0.3])
        result = _align_proba(3, classes, proba_row)
        np.testing.assert_allclose(result, proba_row)

    def test_missing_classes_filled_with_zero(self):
        # Only classes 0 and 2 observed
        classes = np.array([0, 2])
        proba_row = np.array([0.4, 0.6])
        result = _align_proba(3, classes, proba_row)
        assert result[1] == pytest.approx(0.0, abs=1e-6)
        assert result.sum() == pytest.approx(1.0)

    def test_uniform_distribution_when_all_zero(self):
        classes = np.array([0, 1])
        proba_row = np.array([0.0, 0.0])
        result = _align_proba(3, classes, proba_row)
        np.testing.assert_allclose(result, np.full(3, 1.0 / 3))

    def test_out_of_range_class_ignored(self):
        classes = np.array([0, 5])  # 5 is out of range for num_actions=3
        proba_row = np.array([0.7, 0.3])
        result = _align_proba(3, classes, proba_row)
        assert result.shape == (3,)
        assert result.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RandomForestActionSelector
# ---------------------------------------------------------------------------

class TestRandomForestActionSelector:
    def test_unfitted_returns_uniform(self):
        algo = RandomForestActionSelector(num_actions=NUM_ACTIONS, random_state=0)
        probs = algo.predict_proba(np.zeros(4))
        np.testing.assert_allclose(probs, np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS))

    def test_train_sets_fitted_flag(self):
        algo = RandomForestActionSelector(num_actions=NUM_ACTIONS, n_estimators=10, random_state=0)
        algo.train(_STATES, _ACTIONS)
        assert algo._fitted

    def test_predict_proba_after_training(self):
        algo = RandomForestActionSelector(num_actions=NUM_ACTIONS, n_estimators=10, random_state=0)
        algo.train(_STATES, _ACTIONS)
        probs = algo.predict_proba(np.array([0.0, 0.0, 0.0, 0.0]))
        assert probs.shape == (NUM_ACTIONS,)
        assert probs.sum() == pytest.approx(1.0)
        assert all(p >= 0 for p in probs)

    def test_select_action_returns_valid_index(self):
        algo = RandomForestActionSelector(num_actions=NUM_ACTIONS, n_estimators=10, random_state=0)
        algo.train(_STATES, _ACTIONS)
        action = algo.select_action(np.array([0.0, 0.0, 0.0, 0.0]))
        assert 0 <= action < NUM_ACTIONS

    def test_num_actions_stored(self):
        algo = RandomForestActionSelector(num_actions=5, random_state=0)
        assert algo.num_actions == 5

    def test_extra_kwargs_ignored(self):
        # Should not raise
        algo = RandomForestActionSelector(num_actions=2, n_estimators=5, random_state=1, unused_param="x")
        assert algo.num_actions == 2


# ---------------------------------------------------------------------------
# NaiveBayesActionSelector
# ---------------------------------------------------------------------------

class TestNaiveBayesActionSelector:
    def test_unfitted_returns_uniform(self):
        algo = NaiveBayesActionSelector(num_actions=NUM_ACTIONS)
        probs = algo.predict_proba(np.zeros(4))
        np.testing.assert_allclose(probs, np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS))

    def test_train_and_predict(self):
        algo = NaiveBayesActionSelector(num_actions=NUM_ACTIONS)
        algo.train(_STATES, _ACTIONS)
        probs = algo.predict_proba(np.array([0.0, 0.0, 0.0, 0.0]))
        assert probs.shape == (NUM_ACTIONS,)
        assert probs.sum() == pytest.approx(1.0)

    def test_select_action_valid(self):
        algo = NaiveBayesActionSelector(num_actions=NUM_ACTIONS)
        algo.train(_STATES, _ACTIONS)
        action = algo.select_action(np.zeros(4))
        assert 0 <= action < NUM_ACTIONS


# ---------------------------------------------------------------------------
# KNNActionSelector
# ---------------------------------------------------------------------------

class TestKNNActionSelector:
    def test_unfitted_returns_uniform(self):
        algo = KNNActionSelector(num_actions=NUM_ACTIONS)
        probs = algo.predict_proba(np.zeros(4))
        np.testing.assert_allclose(probs, np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS))

    def test_train_and_predict(self):
        algo = KNNActionSelector(num_actions=NUM_ACTIONS, n_neighbors=3)
        algo.train(_STATES, _ACTIONS)
        probs = algo.predict_proba(np.zeros(4))
        assert probs.shape == (NUM_ACTIONS,)
        assert probs.sum() == pytest.approx(1.0)

    def test_select_action_valid(self):
        algo = KNNActionSelector(num_actions=NUM_ACTIONS, n_neighbors=3)
        algo.train(_STATES, _ACTIONS)
        action = algo.select_action(np.zeros(4))
        assert 0 <= action < NUM_ACTIONS

    def test_custom_n_neighbors(self):
        algo = KNNActionSelector(num_actions=2, n_neighbors=1)
        assert algo.model.n_neighbors == 1


# ---------------------------------------------------------------------------
# GradientBoostActionSelector (xgboost/lightgbm optional)
# ---------------------------------------------------------------------------

def _gradient_boost_available():
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        pass
    return False


@pytest.mark.skipif(not _gradient_boost_available(), reason="Neither xgboost nor lightgbm installed")
class TestGradientBoostActionSelector:
    def test_instantiation(self):
        algo = GradientBoostActionSelector(num_actions=NUM_ACTIONS, random_state=0)
        assert algo.num_actions == NUM_ACTIONS
        assert algo._backend in ("xgboost", "lightgbm")

    def test_unfitted_returns_uniform(self):
        algo = GradientBoostActionSelector(num_actions=NUM_ACTIONS, random_state=0)
        probs = algo.predict_proba(np.zeros(4))
        np.testing.assert_allclose(probs, np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS))

    def test_train_and_predict(self):
        algo = GradientBoostActionSelector(num_actions=NUM_ACTIONS, random_state=0)
        algo.train(_STATES, _ACTIONS)
        probs = algo.predict_proba(np.zeros(4))
        assert probs.shape == (NUM_ACTIONS,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_select_action_valid(self):
        algo = GradientBoostActionSelector(num_actions=NUM_ACTIONS, random_state=0)
        algo.train(_STATES, _ACTIONS)
        action = algo.select_action(np.zeros(4))
        assert 0 <= action < NUM_ACTIONS


class TestGradientBoostImportError:
    """When neither backend is available, instantiation should raise ImportError."""

    def test_raises_import_error_when_no_backend(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name.startswith("xgboost") or name.startswith("lightgbm"):
                raise ImportError(f"Mocked unavailable: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)

        with pytest.raises(ImportError):
            GradientBoostActionSelector(num_actions=3)
