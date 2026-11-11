"""Cooperation experiment: config, world, simulation, metrics, analysis and pipeline."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from farm.core.cooperation import Motive
from farm.experiments.cooperation.agents import income_stream
from farm.experiments.cooperation.analysis import ContrastResult, analyze, holm, paired_test, write_analysis
from farm.experiments.cooperation.config import (
    BOND_KIND,
    CONDITIONS,
    EXPLORATORY_CONDITION_ORDER,
    PRIMARY_CONDITION_ORDER,
    SECONDARY_CONDITION_ORDER,
    Condition,
    DecisionConfig,
    RunConfig,
    WorldConfig,
    grid_cell_name,
    pair_bond,
)
from farm.experiments.cooperation.experiment import MatrixConfig, load_outputs, run_matrix
from farm.experiments.cooperation.metrics import (
    founder_pairs,
    kaplan_meier,
    restricted_mean_survival,
    summarize_run,
    survival_at,
    welfare_summary,
)
from farm.experiments.cooperation.report import write_report
from farm.experiments.cooperation.simulation import CooperationSimulation, run_simulation
from farm.experiments.cooperation.world import IncomeWorld

pytestmark = pytest.mark.unit

SMALL_WORLD = WorldConfig(initial_population=16, max_population=40, carrying_capacity=24, ticks=80)


def _cfg(name="preset_marriage", seed=3, cooperative=True, world=SMALL_WORLD):
    return RunConfig(condition=CONDITIONS[name], seed=seed, world=world, cooperative=cooperative)


# --------------------------------------------------------------------- config
def test_condition_catalogue_is_complete():
    assert len(PRIMARY_CONDITION_ORDER) == 27
    assert len(SECONDARY_CONDITION_ORDER) == 7
    assert len(EXPLORATORY_CONDITION_ORDER) == 5
    cell = CONDITIONS[grid_cell_name(0.5, 0.3, 1.0)]
    assert (cell.bond.scope, cell.bond.exit_cost, cell.bond.recognition) == (0.5, 0.3, 1.0)
    assert all(c.bond.kind == BOND_KIND for c in CONDITIONS.values())


def test_condition_relabels_bond_kind_and_validates_nesting():
    cond = Condition(name="x", family="f", bond=pair_bond(0.5, 0.0, 0.0).with_changes(kind="other"))
    assert cond.bond.kind == BOND_KIND
    with pytest.raises(ValueError):
        Condition(name="x", family="f", bond=pair_bond(0.5, 0.0, 0.0), nesting="clan")


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_population": 200},
        {"shock_prob": 1.0},
        {"drought_length": 500},
        {"offspring_cost": 1.0},
        {"max_energy": 10.0},
        {"carrying_capacity": 0},
        {"trait_heritability": 1.5},
    ],
)
def test_world_config_rejects_bad_values(changes):
    with pytest.raises(ValueError):
        replace(WorldConfig(), **changes)


def test_decision_config_motive_mix():
    probs = DecisionConfig(motive_mix=(("economic", 3.0), ("emotional", 1.0))).motive_probabilities()
    assert probs[Motive.ECONOMIC] == pytest.approx(0.75)
    assert sum(probs.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        DecisionConfig(motive_mix=(("greed", 1.0),))
    with pytest.raises(ValueError):
        DecisionConfig(review_interval=0)


# ---------------------------------------------------------------------- world
def test_drought_share_and_crowding():
    world = IncomeWorld(WorldConfig(), seed=1)
    share = np.mean([world.is_drought(t) for t in range(1200)])
    assert share == pytest.approx(20 / 120)
    assert world.crowding(10) == 1.0
    assert world.crowding(140) == pytest.approx(0.5)


def test_income_streams_are_keyed_by_seed_and_slot():
    a = income_stream(5, True, 3).random(10)
    b = income_stream(5, True, 3).random(10)
    c = income_stream(5, False, 3).random(10)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ----------------------------------------------------------------- simulation
def test_simulation_is_deterministic():
    first = run_simulation(_cfg())
    second = run_simulation(_cfg())
    pd.testing.assert_frame_equal(first.agents, second.agents)
    pd.testing.assert_frame_equal(first.units, second.units)
    assert first.counters == second.counters


def test_solo_twin_shares_founders_and_shock_sequences():
    coop = CooperationSimulation(_cfg())
    solo = CooperationSimulation(_cfg(cooperative=False))
    for slot in range(SMALL_WORLD.initial_population):
        a, b = coop.agents[f"f{slot}"], solo.agents[f"f{slot}"]
        assert (a.skill, a.diligence, a.motive) == (b.skill, b.diligence, b.motive)
        assert np.array_equal(a.rng.random(5), b.rng.random(5))
    assert coop.world.phase == solo.world.phase


def test_motive_condition_changes_motives_but_not_other_traits():
    base = CooperationSimulation(_cfg("preset_marriage"))
    pure = CooperationSimulation(_cfg("marriage_hedonist"))
    assert all(a.motive is Motive.HEDONIST for a in pure.agents.values())
    for slot in range(SMALL_WORLD.initial_population):
        assert base.agents[f"f{slot}"].skill == pure.agents[f"f{slot}"].skill


def test_cooperative_run_forms_units_and_solo_run_does_not():
    coop = run_simulation(_cfg())
    solo = run_simulation(_cfg(cooperative=False))
    assert coop.counters.get("matches_chosen", 0) > 0
    assert len(coop.units) > 0 and set(coop.units["kind"]) == {BOND_KIND}
    assert solo.units.empty and solo.event_counts == {}
    assert coop.timeseries["coverage"].max() > 0
    assert solo.timeseries["coverage"].max() == 0


def test_nesting_conditions_build_families_and_arranged_matches():
    lineage = run_simulation(_cfg("marriage_lineage", world=replace(SMALL_WORLD, ticks=150)))
    assert lineage.counters.get("families_formed", 0) > 0
    assert "family" in set(lineage.units["kind"])
    arranged = run_simulation(_cfg("marriage_arranged", world=replace(SMALL_WORLD, ticks=200)))
    assert arranged.counters.get("matches_arranged", 0) > 0
    assert arranged.units.loc[arranged.units["kind"] == BOND_KIND, "arranged"].any()


def test_energy_never_exceeds_storage_limit():
    sim = CooperationSimulation(_cfg())
    for _ in range(40):
        sim.step()
        assert all(a.energy <= SMALL_WORLD.max_energy + 1e-9 for a in sim.alive_agents())


# -------------------------------------------------------------------- metrics
def test_kaplan_meier_and_rmst():
    _times, surv = kaplan_meier([1, 2, 3], [True, True, True])
    assert surv.tolist() == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])
    assert restricted_mean_survival([1, 2, 3], [True, True, True], tau=3) == pytest.approx(2.0)
    # A censored unit keeps the curve up: one of two observed to fail at 2.
    assert survival_at([2, 5], [True, False], 3) == pytest.approx(0.5)
    assert restricted_mean_survival([2, 5], [True, False], tau=4) == pytest.approx(2 + 0.5 * 2)
    assert np.isnan(restricted_mean_survival([], [], tau=4))


def test_founder_pairs_and_welfare_summary():
    coop = run_simulation(_cfg())
    solo = run_simulation(_cfg(cooperative=False))
    pairs = founder_pairs(coop, solo)
    assert len(pairs) == SMALL_WORLD.initial_population
    assert (pairs["d_lifespan"] == pairs["lifespan"] - pairs["lifespan_solo"]).all()
    summary = welfare_summary(pairs)
    assert -SMALL_WORLD.ticks <= summary["welfare_d_lifespan"] <= SMALL_WORLD.ticks
    with pytest.raises(ValueError):
        founder_pairs(coop, run_simulation(_cfg(seed=4, cooperative=False)))
    row = summarize_run(coop)
    assert {"bond_rmst", "trap_rate", "coverage_mean"} <= set(row)
    assert "bond_rmst" not in summarize_run(solo)


# ------------------------------------------------------------------- analysis
def test_holm_correction():
    tests = [
        ContrastResult(k, "", "m", 10, 0.0, 0.0, 0.0, 0, p)
        for k, p in (("a", 0.01), ("b", 0.04), ("c", 0.03), ("d", 0.2))
    ]
    holm(tests, alpha=0.05)
    by = {t.key: t for t in tests}
    assert by["a"].significant and by["a"].p_holm == pytest.approx(0.04)
    assert not by["c"].significant and by["c"].p_holm == pytest.approx(0.09)
    assert not by["b"].significant and by["b"].p_holm == pytest.approx(0.09)
    assert by["d"].p_holm == pytest.approx(0.2)


def test_paired_test_detects_a_consistent_shift():
    diff = pd.Series(np.linspace(0.5, 1.5, 20))
    result = paired_test("x", "shift", "m", diff)
    assert result.positive_seeds == 20 and result.p_value < 0.001
    assert result.ci_low > 0
    zero = paired_test("z", "none", "m", pd.Series(np.zeros(20)), alternative="two-sided")
    assert zero.p_value == 1.0


# ------------------------------------------------------------------- pipeline
def test_tiny_matrix_runs_analyses_reports_and_resumes(tmp_path):
    names = (grid_cell_name(0.5, 0.0, 0.0), grid_cell_name(0.5, 0.6, 1.0), "preset_marriage")
    matrix = MatrixConfig(seeds=(1, 2), condition_names=names, world=SMALL_WORLD, workers=1)
    outputs = run_matrix(matrix, tmp_path)
    assert set(outputs.runs["condition"]) == set(names)
    assert len(outputs.runs) == 6 and len(outputs.solo_runs) == 2
    assert (outputs.founders.groupby(["condition", "seed"]).size() == SMALL_WORLD.initial_population).all()
    result = analyze(outputs)
    assert result.confirmatory == [] and result.notes
    write_analysis(result, tmp_path)
    report = write_report(outputs, result, tmp_path)
    assert report.exists() and (tmp_path / "figures").exists()
    reloaded = load_outputs(tmp_path)
    pd.testing.assert_frame_equal(
        reloaded.runs.sort_values(["condition", "seed"]).reset_index(drop=True),
        outputs.runs.sort_values(["condition", "seed"]).reset_index(drop=True),
    )
    progress = []
    run_matrix(matrix, tmp_path, progress=progress.append)
    assert any("3 cells resumed" in line for line in progress)
    with pytest.raises(ValueError):
        run_matrix(replace(matrix, seeds=(1, 2, 3)), tmp_path)


def test_resume_rejects_changed_condition_definitions(tmp_path, monkeypatch):
    import farm.experiments.cooperation.experiment as experiment

    name = grid_cell_name(0.5, 0.0, 0.0)
    matrix = MatrixConfig(seeds=(1,), condition_names=(name,), world=SMALL_WORLD, workers=1)
    run_matrix(matrix, tmp_path)
    original = experiment.CONDITIONS[name]
    changed = replace(original, bond=original.bond.with_changes(scope=0.9))
    monkeypatch.setitem(experiment.CONDITIONS, name, changed)
    with pytest.raises(ValueError, match="condition definitions changed"):
        run_matrix(matrix, tmp_path)


def test_a_fresh_subset_run_does_not_mix_in_stale_cells(tmp_path):
    first = (grid_cell_name(0.5, 0.0, 0.0), "preset_marriage")
    run_matrix(MatrixConfig(seeds=(1,), condition_names=first, world=SMALL_WORLD, workers=1), tmp_path)
    second = MatrixConfig(seeds=(2,), condition_names=("preset_marriage",), world=SMALL_WORLD, workers=1)
    outputs = run_matrix(second, tmp_path, resume=False)
    assert set(outputs.runs["condition"]) == {"preset_marriage"}
    reloaded = load_outputs(tmp_path)  # --analyze-only path
    assert set(reloaded.runs["condition"]) == {"preset_marriage"}
    assert set(reloaded.runs["seed"]) == {2}


def test_preset_only_run_still_writes_a_report(tmp_path):
    matrix = MatrixConfig(seeds=(1, 2), condition_names=("preset_marriage",), world=SMALL_WORLD, workers=1)
    outputs = run_matrix(matrix, tmp_path)
    report = write_report(outputs, analyze(outputs), tmp_path)
    assert report.exists()
    assert not (tmp_path / "figures" / "grid_bond_rmst.png").exists()


def test_only_an_agents_own_family_counts_as_belonging():
    from farm.core.cooperation import CooperationRegistry, preset
    from farm.experiments.cooperation.decisions import parent_is_own

    reg = CooperationRegistry(rng=np.random.default_rng(0))
    kin_family = reg.form(preset("family"), ["kin"], step=0)
    arranged = reg.form(preset("marriage"), ["kin", "outsider"], step=5, parent_id=kin_family.unit_id)
    assert parent_is_own("kin", arranged, kin_family)
    assert not parent_is_own("outsider", arranged, kin_family)
    couple = reg.form(preset("marriage"), ["x", "y"], step=0)
    anchored = reg.form(preset("family"), [], step=3, children=[couple.unit_id])
    assert parent_is_own("x", couple, anchored) and parent_is_own("y", couple, anchored)
