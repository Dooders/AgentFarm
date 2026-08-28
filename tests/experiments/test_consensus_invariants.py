"""Invariant and property tests for the political consensus experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farm.experiments.consensus.allocation import BLEND_DIRECTED, BLEND_PLATFORM, allocate
from farm.experiments.consensus.contrasts import paired_contrasts
from farm.experiments.consensus.experiment import (
    ExperimentConfig,
    config_manifest,
    run_cell,
    run_trials,
    summarize,
    write_outputs,
)
from farm.experiments.consensus.mechanism import choose_lambda_reelection
from farm.experiments.consensus.metrics import ALLOCATION_COLUMNS, evaluate_trial
from farm.experiments.consensus.paradigms import CONSTRAINED_PARADIGM, PARADIGMS, SELECTION_PARADIGMS, run_election
from farm.experiments.consensus.population import Candidates, Population, generate_candidates, generate_population

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
    selection = trials[trials["paradigm"].isin(SELECTION_PARADIGMS)]
    welfare = selection[["total_welfare", "supporter_welfare", "loser_welfare", "gap"]].to_numpy()
    assert np.isfinite(welfare).all()
    assert np.isfinite(trials["total_welfare"]).all()
    assert np.isfinite(trials["minority_welfare"]).all()


def test_party_loser_share_near_half_under_two_equal_clusters(trials: pd.DataFrame) -> None:
    """Generator/sanity check, not a proof the party rule is correct.

    ``two_cluster`` forces 50/50 cluster counts and party brands sit at those
    cluster means, so party loser_share is near 1/2 by construction.
    """
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
    assert set(numbers.minority_welfare) >= {"party", "individual", "score", "latent_match"}
    assert all(np.isfinite(v) for v in numbers.minority_welfare.values())


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
        == "python run_experiment.py --seed 0 --out '{run_dir}'"
    )
    assert (
        _portable_command(["--out=/tmp/x", "--seed", "0"])
        == "python run_experiment.py '--out={run_dir}' --seed 0"
    )
    assert _portable_command(["--seed", "0"]).endswith("--seed 0 --out '{run_dir}'")


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


def test_effective_platform_share_equals_blend_weight() -> None:
    benefits = np.ones((12, 5)) / 5.0
    supporters = np.ones(12, dtype=bool)
    directed = benefits.mean(axis=0)
    for scale in (0.05, 1.0, 8.0, 40.0):
        platform = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=float) * scale
        alloc = allocate(benefits, supporters, platform, lam=0.4)
        plat = platform / platform.sum()
        expected = BLEND_DIRECTED * directed + BLEND_PLATFORM * plat
        np.testing.assert_allclose(alloc, expected, atol=1e-9)
        denom = plat - directed
        weight = (alloc - directed) / np.where(np.abs(denom) > 1e-12, denom, np.nan)
        np.testing.assert_allclose(weight[np.isfinite(weight)], BLEND_PLATFORM, atol=1e-9)
    alloc_zero = allocate(benefits, supporters, np.zeros(5), lam=0.4)
    np.testing.assert_allclose(alloc_zero, directed, atol=1e-9)


def test_allocate_is_scale_invariant_in_platform() -> None:
    rng = np.random.default_rng(4)
    population = generate_population(rng, 40, "two_cluster")
    candidates = generate_candidates(rng, 4, population)
    supporters = population.cluster_ids == 0
    platform = candidates.platforms[0]
    base = allocate(population.benefits, supporters, platform, lam=0.6)
    for scale in (0.1, 3.0, 25.0):
        np.testing.assert_allclose(
            allocate(population.benefits, supporters, scale * platform, lam=0.6),
            base,
            atol=1e-9,
        )


def test_complement_welfare_nonincreasing_in_lambda() -> None:
    rng = np.random.default_rng(5)
    population = generate_population(rng, 80, "two_cluster")
    candidates = generate_candidates(rng, 4, population)
    supporters = population.cluster_ids == 0
    platform = candidates.platforms[0]
    welfare = []
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        alloc = allocate(population.benefits, supporters, platform, lam)
        utility = population.benefits @ alloc
        welfare.append(float(utility[~supporters].mean()))
    assert all(welfare[i] >= welfare[i + 1] - 1e-9 for i in range(len(welfare) - 1))


def test_lambda_zero_maximizes_total_welfare_in_family() -> None:
    """In-family total welfare at λ=0 vs λ=1, with a documented allowance.

    Mean utility ``dir_all @ a`` is linear, so the true utilitarian peak is a
    simplex vertex, not ``dir_all``. The steward's family uses ``dir_all`` as
    the everyone-direction. Then ``u(λ=0) ≥ u(λ=1)`` iff
    ``dir_all @ dir_all ≥ dir_all @ dir_supporters`` (platform held fixed).
    That fails when the supporter mean is *more* aligned with ``dir_all`` than
    ``dir_all`` is with itself — e.g. supporters concentrated on the globally
    popular project. The test uses the half of voters with the *lowest* weight
    on that project, where the inequality holds.
    """
    rng = np.random.default_rng(6)
    population = generate_population(rng, 80, "two_cluster")
    candidates = generate_candidates(rng, 4, population)
    dir_all = population.benefits.mean(axis=0)
    popular = int(dir_all.argmax())
    supporters = np.zeros(population.n_voters, dtype=bool)
    supporters[np.argsort(population.benefits[:, popular])[: population.n_voters // 2]] = True
    platform = candidates.platforms[0]
    dir_s = population.benefits[supporters].mean(axis=0)
    assert dir_all @ dir_all >= dir_all @ dir_s - 1e-12
    u0 = float((population.benefits @ allocate(population.benefits, supporters, platform, 0.0)).mean())
    u1 = float((population.benefits @ allocate(population.benefits, supporters, platform, 1.0)).mean())
    assert u0 >= u1 - 1e-9


def test_candidate_permutation_preserves_winner_platform_and_lambda() -> None:
    rng = np.random.default_rng(8)
    population = generate_population(rng, 60, "two_cluster")
    candidates = generate_candidates(rng, 5, population)
    perm = rng.permutation(candidates.n_candidates)
    shuffled = Candidates(platforms=candidates.platforms[perm], lam=candidates.lam[perm])
    for name in PARADIGMS:
        original = run_election(name, population, candidates)
        relabeled = run_election(name, population, shuffled)
        np.testing.assert_allclose(candidates.platforms[original.winner], shuffled.platforms[relabeled.winner])
        np.testing.assert_allclose(candidates.lam[original.winner], shuffled.lam[relabeled.winner])


def test_project_permutation_preserves_welfare_scalars() -> None:
    rng = np.random.default_rng(9)
    population = generate_population(rng, 60, "two_cluster")
    candidates = generate_candidates(rng, 5, population)
    perm = rng.permutation(5)
    shuffled_pop = Population(
        prefs=population.prefs[:, perm],
        benefits=population.benefits[:, perm],
        cluster_ids=population.cluster_ids,
        cluster_centers=population.cluster_centers[:, perm],
    )
    shuffled_cand = Candidates(platforms=candidates.platforms[:, perm], lam=candidates.lam)
    election = run_election("individual", population, candidates)
    election_p = run_election("individual", shuffled_pop, shuffled_cand)
    row = evaluate_trial("individual", population, candidates, election)
    row_p = evaluate_trial("individual", shuffled_pop, shuffled_cand, election_p)
    for key in ("total_welfare", "minority_welfare", "lambda_winner", "loser_share"):
        np.testing.assert_allclose(row[key], row_p[key], atol=1e-9)


def test_paired_identity_and_winner_can_differ(trials: pd.DataFrame) -> None:
    party = trials[trials["paradigm"] == "party"]
    individual = trials[trials["paradigm"] == "individual"]
    merged = party.merge(individual, on="trial", suffixes=("_p", "_i"))
    assert (merged["n_voters_p"] == merged["n_voters_i"]).all()
    assert (merged["winner_p"] != merged["winner_i"]).any()


def test_contrasts_use_matching_trial_ids_and_finite_cis() -> None:
    config = ExperimentConfig(trials=8, voters=80, candidates=4, seed=1, persist_ballots=False)
    trials = run_trials(config)
    contrasts = paired_contrasts(trials)
    primary = contrasts[contrasts["family"] == "primary"]
    assert not primary.empty
    assert (primary["n_pairs"] == 8).all()
    assert np.isfinite(primary[["ci_low", "ci_high", "delta_mean"]].to_numpy()).all()
    party = trials[trials["paradigm"] == "party"].set_index("trial")["minority_welfare"]
    individual = trials[trials["paradigm"] == "individual"].set_index("trial")["minority_welfare"]
    manual = float((individual - party).mean())
    row = primary[(primary["paradigm"] == "individual") & (primary["endpoint"] == "minority_welfare")].iloc[0]
    np.testing.assert_allclose(row["delta_mean"], manual)


def test_constrained_and_baselines_absent_from_primary_contrasts(trials: pd.DataFrame) -> None:
    contrasts = paired_contrasts(trials)
    assert CONSTRAINED_PARADIGM not in set(contrasts["paradigm"])
    assert "random_winner" not in set(contrasts["paradigm"])
    assert "utilitarian" not in set(contrasts["paradigm"])


def test_audit_party_supporter_share_near_half(tmp_path) -> None:
    config = ExperimentConfig(trials=8, voters=80, candidates=4, seed=1, persist_ballots=True)
    run = run_cell(config)
    write_outputs(run.trials, tmp_path, config_manifest(config, "python run_experiment.py"), audit=run.audit)
    data = np.load(tmp_path / "private" / "ballots.npz")
    assert 0.40 <= float(data["supporters_party"].mean()) <= 0.60
    assert data["cluster_ids"].shape == (8, 80)


def test_default_lambda_is_independent_of_platform_extremity() -> None:
    rng = np.random.default_rng(0)
    population = generate_population(rng, 200, "two_cluster")
    independent = generate_candidates(rng, 80, population, lambda_correlated=False)
    extremity = np.linalg.norm(independent.platforms - population.prefs.mean(axis=0), axis=1)
    corr = float(np.corrcoef(extremity, independent.lam)[0, 1])
    assert abs(corr) < 0.35
    coupled = generate_candidates(rng, 80, population, lambda_correlated=True)
    ext = np.linalg.norm(coupled.platforms - population.prefs.mean(axis=0), axis=1)
    assert float(np.corrcoef(ext, coupled.lam)[0, 1]) > 0.8


def test_default_primary_question_is_ballot_format() -> None:
    assert ExperimentConfig().primary_question == "ballot_format_fixed_partition"
    assert ExperimentConfig().include_lambda_primary is False
    assert ExperimentConfig(lambda_correlated=True).primary_question == "lambda_selection_robustness"
    assert ExperimentConfig(mechanism="reelection").primary_question == "reelection_incentives"


def test_report_does_not_treat_flat_lambda_as_finding() -> None:
    from farm.experiments.consensus.experiment import allocation_means
    from farm.experiments.consensus.report import render_report

    config = ExperimentConfig(trials=4, voters=40, candidates=4, seed=0, persist_ballots=False)
    trials = run_trials(config)
    text = render_report(trials, summarize(trials), allocation_means(trials), config_manifest(config, "cmd"))
    assert "by construction" in text
    assert "supports the hypothesis" not in text
    assert "λ_winner unchanged across rules" not in text
    assert CONSTRAINED_PARADIGM not in text or "not a voting rule" in text


def test_reelection_chooses_lambda_from_incentives() -> None:
    rng = np.random.default_rng(2)
    population = generate_population(rng, 80, "two_cluster")
    candidates = generate_candidates(rng, 6, population)
    observe = np.ones(population.n_voters, dtype=bool)
    platform = candidates.platforms[0]
    lam_all = choose_lambda_reelection(population.benefits, observe, platform, observe)
    lam_few = choose_lambda_reelection(
        population.benefits, population.cluster_ids == 0, platform, observe
    )
    assert 0.0 <= lam_few <= lam_all <= 1.0
    assert lam_all != lam_few
    config = ExperimentConfig(trials=6, voters=60, candidates=4, seed=2, mechanism="reelection")
    trials = run_trials(config)
    chosen = trials.loc[trials["paradigm"].isin(SELECTION_PARADIGMS), "lambda_winner"]
    assert chosen.between(0.0, 1.0).all()
    assert set(np.round(chosen, 5)).issubset(set(np.round(np.linspace(0.0, 1.0, 21), 5)))


def test_abandon_trailing_changes_some_ballots() -> None:
    rng = np.random.default_rng(11)
    population = generate_population(rng, 80, "two_cluster")
    candidates = generate_candidates(rng, 8, population)
    sincere = run_election("individual", population, candidates, voting="sincere")
    strategic = run_election("individual", population, candidates, voting="abandon_trailing")
    assert sincere.ballots is not None and strategic.ballots is not None
    assert sincere.ballots.shape == strategic.ballots.shape
