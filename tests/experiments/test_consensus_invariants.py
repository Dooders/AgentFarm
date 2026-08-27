"""Invariant tests for the political consensus experiment."""

import numpy as np
import pandas as pd
import pytest

from farm.experiments.consensus.allocation import allocate
from farm.experiments.consensus.experiment import ExperimentConfig, run_trials, summarize
from farm.experiments.consensus.metrics import ALLOCATION_COLUMNS
from farm.experiments.consensus.paradigms import CONSTRAINED_PARADIGM
from farm.experiments.consensus.population import generate_candidates, generate_population

pytestmark = pytest.mark.unit

SMALL_CONFIG = ExperimentConfig(
    trials=30,
    voters=160,
    candidates=6,
    population="two_cluster",
    seed=7,
    include_constrained=True,
    lambda_cap=0.25,
)


@pytest.fixture(scope="module")
def trials() -> pd.DataFrame:
    return run_trials(SMALL_CONFIG)


def test_allocations_sum_to_one_and_nonnegative(trials: pd.DataFrame) -> None:
    alloc = trials[list(ALLOCATION_COLUMNS)].to_numpy()
    np.testing.assert_allclose(alloc.sum(axis=1), 1.0, atol=1e-9)
    assert (alloc >= 0.0).all()


def test_utilities_finite(trials: pd.DataFrame) -> None:
    welfare = trials[["total_welfare", "supporter_welfare", "loser_welfare", "gap"]].to_numpy()
    assert np.isfinite(welfare).all()


def test_party_loser_share_near_half_under_two_equal_clusters(trials: pd.DataFrame) -> None:
    party_share = trials.loc[trials["paradigm"] == "party", "loser_share"]
    assert 0.40 <= party_share.mean() <= 0.60
    assert party_share.between(0.30, 0.70).all()


def test_identical_seed_and_params_give_identical_summary(trials: pd.DataFrame) -> None:
    rerun = summarize(run_trials(SMALL_CONFIG))
    pd.testing.assert_frame_equal(summarize(trials), rerun)


def test_constrained_individual_never_exceeds_lambda_cap(trials: pd.DataFrame) -> None:
    constrained = trials[trials["paradigm"] == CONSTRAINED_PARADIGM]
    assert not constrained.empty
    assert (constrained["lambda_effective"] <= SMALL_CONFIG.lambda_cap + 1e-12).all()
    expected = np.minimum(constrained["lambda_winner"], SMALL_CONFIG.lambda_cap)
    np.testing.assert_allclose(constrained["lambda_effective"], expected)


def test_zero_supporters_falls_back_to_everyone_direction() -> None:
    rng = np.random.default_rng(0)
    population = generate_population(rng, n_voters=50, population_type="two_cluster")
    candidates = generate_candidates(rng, n_candidates=4, population=population)
    no_supporters = np.zeros(population.n_voters, dtype=bool)
    all_supporters = np.ones(population.n_voters, dtype=bool)
    platform = candidates.platforms[0]
    # With zero supporters the allocation must ignore lambda entirely.
    np.testing.assert_allclose(
        allocate(population.benefits, no_supporters, platform, lam=1.0),
        allocate(population.benefits, all_supporters, platform, lam=0.0),
    )


def test_animation_renders_mp4(tmp_path) -> None:
    from matplotlib.animation import FFMpegWriter

    if not FFMpegWriter.isAvailable():
        pytest.skip("ffmpeg not available")
    from farm.experiments.consensus.animate import render_animation

    out = render_animation(tmp_path / "anim.mp4", voters=60, n_candidates=4, seed=1, fps=2)
    assert out.exists() and out.stat().st_size > 0


def test_benefits_are_normalized_rows() -> None:
    rng = np.random.default_rng(3)
    for population_type in ("one_cluster", "two_cluster", "three_cluster", "rural_town"):
        population = generate_population(rng, n_voters=80, population_type=population_type)
        np.testing.assert_allclose(population.benefits.sum(axis=1), 1.0, atol=1e-9)
        assert (population.benefits >= 0.0).all()
