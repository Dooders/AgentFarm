"""Invariant tests for the political consensus experiment."""

from pathlib import Path

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


def test_overview_numbers_load_from_committed_results() -> None:
    pytest.importorskip("manim")
    from farm.experiments.consensus.overview_video import load_numbers

    default_dir, correlated_dir = Path("results/consensus"), Path("results/consensus_lambda_correlated")
    if not (default_dir / "summary.csv").exists() or not (correlated_dir / "summary.csv").exists():
        pytest.skip("committed result artifacts not present")
    numbers = load_numbers(default_dir, correlated_dir)
    assert set(numbers.loser_welfare) >= {"party", "individual", "score", "latent_match"}
    assert all(np.isfinite(v) for v in numbers.loser_welfare.values())


def test_benefits_are_normalized_rows() -> None:
    rng = np.random.default_rng(3)
    for population_type in ("one_cluster", "two_cluster", "three_cluster", "rural_town"):
        population = generate_population(rng, n_voters=80, population_type=population_type)
        np.testing.assert_allclose(population.benefits.sum(axis=1), 1.0, atol=1e-9)
        assert (population.benefits >= 0.0).all()


def test_portable_command_replaces_out_dir() -> None:
    from run_experiment import _portable_command

    assert (
        _portable_command(["--seed", "0", "--out", "/tmp/x"])
        == "python run_experiment.py --seed 0 --out {run_dir}"
    )
    assert (
        _portable_command(["--out=/tmp/x", "--seed", "0"])
        == "python run_experiment.py --out={run_dir} --seed 0"
    )
    assert _portable_command(["--seed", "0"]).endswith("--seed 0 --out {run_dir}")


def test_outputs_are_bitwise_reproducible_and_derivations_verify(tmp_path) -> None:
    """Two runs with the same seed produce byte-identical artifacts, and the
    derived artifacts (summary, allocation means, report) recompute exactly
    from trials.csv."""
    import subprocess
    import sys

    from run_experiment import _verify_report

    repo = Path(__file__).resolve().parents[2]
    args = [
        sys.executable,
        str(repo / "run_experiment.py"),
        "--trials", "6", "--voters", "60", "--candidates", "4",
        "--population", "two_cluster", "--seed", "3",
    ]
    a, b = tmp_path / "a", tmp_path / "b"
    subprocess.run(args + ["--out", str(a)], check=True, cwd=repo, capture_output=True)
    subprocess.run(args + ["--out", str(b)], check=True, cwd=repo, capture_output=True)

    artifacts = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert artifacts == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    for rel in artifacts:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), f"{rel} differs between identical runs"

    assert _verify_report(a) == 0

    # Tampering with a derived artifact is caught.
    summary = a / "summary.csv"
    summary.write_text(summary.read_text().replace("0.", "1.", 1))
    assert _verify_report(a) == 1
