"""Performance-characterization tests for crossover strategies.

These tests measure wall-clock time and inference quality
(Q-value error, action agreement) for the three crossover strategies.
They are marked ``@pytest.mark.slow`` and are excluded from the default
``pytest`` run (see ``pytest.ini``).  To regenerate the documented
numbers in ``docs/design/crossover_strategies.md`` run::

    pytest tests/decision/test_crossover_performance.py -m slow -v -s

or use the convenience script::

    python scripts/benchmark_crossover.py

Methodology
-----------
- Two :class:`~farm.core.decision.base_dqn.BaseQNetwork` parents with
  ``input_dim=8``, ``output_dim=4``, ``hidden_size=64`` are created
  from fixed seeds (0 and 1) so results are reproducible.
- A fixed state batch of 256 rows drawn from ``np.random.default_rng(42)``
  is used for quality metrics.
- Crossover time is measured as the wall time for
  :func:`crossover_quantized_state_dict` plus
  :meth:`nn.Module.load_state_dict`, averaged over ``N_REPEATS=10`` runs.
- Quality metrics compare the child's Q-values against parent A's
  Q-values (used as the float reference):

  - **mean_q_error**: mean absolute error of child Q-values vs parent A.
  - **max_q_error**: maximum absolute error.
  - **action_agreement**: fraction of states where ``argmax`` matches
    parent A (higher = child more similar to parent A).

All tests print a summary row matching the table in the design note.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork
from farm.core.decision.training.crossover import (
    crossover_quantized_state_dict,
    initialize_child_from_crossover,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INPUT_DIM = 8
OUTPUT_DIM = 4
HIDDEN_SIZE = 64
N_STATES = 256
N_REPEATS = 10
SEED_A = 0
SEED_B = 1
STATE_SEED = 42


def _make_model(seed: int) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_size=HIDDEN_SIZE,
    )


def _make_states() -> torch.Tensor:
    rng = np.random.default_rng(STATE_SEED)
    return torch.from_numpy(
        rng.standard_normal((N_STATES, INPUT_DIM)).astype("float32")
    )


def _q_values(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(states)


def _quality_metrics(
    child_q: torch.Tensor,
    ref_q: torch.Tensor,
) -> Dict[str, float]:
    """Compute quality metrics relative to a float reference (parent A)."""
    diff = (child_q - ref_q).abs()
    mean_err = diff.mean().item()
    max_err = diff.max().item()
    agree = (child_q.argmax(dim=1) == ref_q.argmax(dim=1)).float().mean().item()
    return {
        "mean_q_error": mean_err,
        "max_q_error": max_err,
        "action_agreement": agree,
    }


def _time_crossover(
    sd_a: dict,
    sd_b: dict,
    mode: str,
    n_repeats: int,
    **kwargs,
) -> float:
    """Return mean wall-clock seconds for crossover + load_state_dict."""
    model = BaseQNetwork(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_size=HIDDEN_SIZE,
    )
    times: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        child_sd = crossover_quantized_state_dict(sd_a, sd_b, mode=mode, **kwargs)
        model.load_state_dict(child_sd)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Correctness smoke checks (fast, included in default run)
# ---------------------------------------------------------------------------


class TestCrossoverSmoke:
    """Quick sanity assertions for all three strategies.

    These run under the default ``pytest`` invocation (no slow marker) and
    serve as regression guards that any new strategy change must not break.
    """

    def test_random_output_shape(self):
        pa = _make_model(SEED_A)
        pb = _make_model(SEED_B)
        child = initialize_child_from_crossover(pa, pb, strategy="random", rng=0)
        out = child(_make_states())
        assert out.shape == (N_STATES, OUTPUT_DIM)
        assert torch.isfinite(out).all()

    def test_layer_output_shape(self):
        pa = _make_model(SEED_A)
        pb = _make_model(SEED_B)
        child = initialize_child_from_crossover(pa, pb, strategy="layer")
        out = child(_make_states())
        assert out.shape == (N_STATES, OUTPUT_DIM)
        assert torch.isfinite(out).all()

    def test_weighted_output_shape(self):
        pa = _make_model(SEED_A)
        pb = _make_model(SEED_B)
        child = initialize_child_from_crossover(pa, pb, strategy="weighted", alpha=0.5)
        out = child(_make_states())
        assert out.shape == (N_STATES, OUTPUT_DIM)
        assert torch.isfinite(out).all()

    def test_identical_parents_all_strategies(self):
        """Child from identical parents must equal either parent for all modes."""
        pa = _make_model(SEED_A)
        states = _make_states()
        ref_q = _q_values(pa, states)

        for mode in ("random", "layer", "weighted"):
            kwargs = {"seed": 0} if mode == "random" else {}
            child_sd = crossover_quantized_state_dict(
                pa.state_dict(), pa.state_dict(), mode=mode, **kwargs
            )
            child = _make_model(99)
            child.load_state_dict(child_sd)
            child_q = _q_values(child, states)
            assert torch.allclose(child_q, ref_q, atol=1e-5), (
                f"mode={mode!r}: child from identical parents differs from parent"
            )

    def test_weighted_alpha_boundary_quality(self):
        """alpha=1.0 → child==A; alpha=0.0 → child==B."""
        pa = _make_model(SEED_A)
        pb = _make_model(SEED_B)
        states = _make_states()
        ref_a = _q_values(pa, states)
        ref_b = _q_values(pb, states)

        child_a_sd = crossover_quantized_state_dict(
            pa.state_dict(), pb.state_dict(), mode="weighted", alpha=1.0
        )
        child_b_sd = crossover_quantized_state_dict(
            pa.state_dict(), pb.state_dict(), mode="weighted", alpha=0.0
        )
        child_a = _make_model(99)
        child_b = _make_model(99)
        child_a.load_state_dict(child_a_sd)
        child_b.load_state_dict(child_b_sd)
        qa = _q_values(child_a, states)
        qb = _q_values(child_b, states)
        assert torch.allclose(qa, ref_a, atol=1e-5)
        assert torch.allclose(qb, ref_b, atol=1e-5)


# ---------------------------------------------------------------------------
# Performance benchmarks (slow – excluded from default run)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCrossoverPerformance:
    """Wall-clock and quality benchmarks for each crossover strategy.

    Run with::

        pytest tests/decision/test_crossover_performance.py -m slow -v -s

    Results are printed to stdout and can be compared with the table in
    ``docs/design/crossover_strategies.md``.

    Setup
    -----
    - Parents: ``BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64)``
      seeded at 0 (parent A) and 1 (parent B).
    - States: 256 synthetic states from ``np.random.default_rng(42)``.
    - Reference for quality metrics: parent A's Q-values.
    - Timing: mean of 10 repetitions (crossover + load_state_dict).
    """

    @pytest.fixture(scope="class")
    def parents(self):
        pa = _make_model(SEED_A)
        pb = _make_model(SEED_B)
        return pa, pb

    @pytest.fixture(scope="class")
    def states(self):
        return _make_states()

    @pytest.fixture(scope="class")
    def ref_q(self, parents, states):
        pa, _ = parents
        return _q_values(pa, states)

    def _run_benchmark(self, mode, sd_a, sd_b, states, ref_q, **kwargs):
        mean_time = _time_crossover(sd_a, sd_b, mode, N_REPEATS, **kwargs)

        child_sd = crossover_quantized_state_dict(sd_a, sd_b, mode=mode, **kwargs)
        child = _make_model(99)
        child.load_state_dict(child_sd)
        child_q = _q_values(child, states)
        metrics = _quality_metrics(child_q, ref_q)

        print(
            f"\n[{mode}] mean_time={mean_time*1000:.2f}ms "
            f"mean_q_error={metrics['mean_q_error']:.4f} "
            f"max_q_error={metrics['max_q_error']:.4f} "
            f"action_agreement={metrics['action_agreement']:.3f}"
        )
        return mean_time, metrics

    def test_random_performance(self, parents, states, ref_q):
        pa, pb = parents
        sd_a, sd_b = pa.state_dict(), pb.state_dict()
        mean_time, metrics = self._run_benchmark(
            "random", sd_a, sd_b, states, ref_q, seed=42
        )
        # Sanity: operation must complete in < 5 seconds (very generous)
        assert mean_time < 5.0, f"random crossover too slow: {mean_time:.3f}s"
        # Outputs must be finite
        assert metrics["mean_q_error"] >= 0
        assert 0.0 <= metrics["action_agreement"] <= 1.0

    def test_layer_performance(self, parents, states, ref_q):
        pa, pb = parents
        sd_a, sd_b = pa.state_dict(), pb.state_dict()
        mean_time, metrics = self._run_benchmark(
            "layer", sd_a, sd_b, states, ref_q
        )
        assert mean_time < 5.0, f"layer crossover too slow: {mean_time:.3f}s"
        assert metrics["mean_q_error"] >= 0
        assert 0.0 <= metrics["action_agreement"] <= 1.0

    def test_weighted_performance(self, parents, states, ref_q):
        pa, pb = parents
        sd_a, sd_b = pa.state_dict(), pb.state_dict()
        mean_time, metrics = self._run_benchmark(
            "weighted", sd_a, sd_b, states, ref_q, alpha=0.5
        )
        assert mean_time < 5.0, f"weighted crossover too slow: {mean_time:.3f}s"
        assert metrics["mean_q_error"] >= 0
        assert 0.0 <= metrics["action_agreement"] <= 1.0

    def test_weighted_quality_interpolates(self, parents, states, ref_q):
        """Verify that weighted crossover at alpha=0.5 sits between parents."""
        pa, pb = parents
        sd_a, sd_b = pa.state_dict(), pb.state_dict()
        ref_b = _q_values(pb, states)

        child_sd = crossover_quantized_state_dict(
            sd_a, sd_b, mode="weighted", alpha=0.5
        )
        child = _make_model(99)
        child.load_state_dict(child_sd)
        child_q = _q_values(child, states)

        err_to_a = (child_q - ref_q).abs().mean().item()
        err_to_b = (child_q - ref_b).abs().mean().item()
        print(
            f"\n[weighted α=0.5] err_to_A={err_to_a:.4f} err_to_B={err_to_b:.4f}"
        )
        # At α=0.5 the child should be closer to both parents than either
        # parent is to the other (i.e. it truly interpolates)
        err_a_to_b = (ref_q - ref_b).abs().mean().item()
        assert err_to_a <= err_a_to_b + 1e-4
        assert err_to_b <= err_a_to_b + 1e-4

    def test_random_diversity_across_seeds(self, parents, states):
        """Random crossover with different seeds produces diverse Q-values."""
        pa, pb = parents
        sd_a, sd_b = pa.state_dict(), pb.state_dict()

        q_values = []
        for seed in range(5):
            child_sd = crossover_quantized_state_dict(
                sd_a, sd_b, mode="random", seed=seed
            )
            child = _make_model(99)
            child.load_state_dict(child_sd)
            q_values.append(_q_values(child, states))

        # At least some pairs of children must differ
        diffs = []
        for i in range(len(q_values)):
            for j in range(i + 1, len(q_values)):
                diffs.append((q_values[i] - q_values[j]).abs().mean().item())
        assert max(diffs) > 1e-6, "Different seeds produced identical Q-values"
        print(f"\n[random diversity] max pairwise Q diff={max(diffs):.4f}")

    def test_strategy_comparison_summary(self, parents, states, ref_q):
        """Print a summary table of all three strategies for documentation."""
        pa, pb = parents
        sd_a, sd_b = pa.state_dict(), pb.state_dict()

        rows = []
        for mode, kwargs in [
            ("random", {"seed": 42}),
            ("layer", {}),
            ("weighted", {"alpha": 0.5}),
        ]:
            mean_time = _time_crossover(sd_a, sd_b, mode, N_REPEATS, **kwargs)
            child_sd = crossover_quantized_state_dict(sd_a, sd_b, mode=mode, **kwargs)
            child = _make_model(99)
            child.load_state_dict(child_sd)
            child_q = _q_values(child, states)
            m = _quality_metrics(child_q, ref_q)
            rows.append((mode, mean_time, m))

        print("\n\nCrossover Strategy Comparison")
        print("=" * 72)
        print(
            f"{'Strategy':<12} {'Time (ms)':>10} {'Mean Q Err':>12} "
            f"{'Max Q Err':>10} {'Act. Agree':>12}"
        )
        print("-" * 72)
        for mode, t, m in rows:
            print(
                f"{mode:<12} {t*1000:>10.2f} {m['mean_q_error']:>12.4f} "
                f"{m['max_q_error']:>10.4f} {m['action_agreement']:>12.3f}"
            )
        print("=" * 72)
        print(
            f"Setup: input_dim={INPUT_DIM}, hidden_size={HIDDEN_SIZE}, "
            f"output_dim={OUTPUT_DIM}, n_states={N_STATES}, "
            f"n_repeats={N_REPEATS}, seed_A={SEED_A}, seed_B={SEED_B}"
        )
