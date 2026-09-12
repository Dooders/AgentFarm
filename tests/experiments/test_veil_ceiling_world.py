"""Unit tests for the Veil Ceiling world, monitoring channel and configuration."""

import numpy as np
import pytest

from farm.experiments.veil_ceiling.config import (
    CONDITIONS,
    COVERAGE_PRIMARY,
    FIDELITY_SWEEP,
    PENALTY_PRIMARY,
    PENALTY_ROBUSTNESS,
    PRIMARY_CONDITION_ORDER,
    AnalysisThresholds,
    LearnerConfig,
    MonitoringConfig,
    RunConfig,
    WorldConfig,
    robustness_conditions,
)
from farm.experiments.veil_ceiling.monitoring import Monitoring
from farm.experiments.veil_ceiling.world import ResourceField

pytestmark = pytest.mark.unit


# ── configuration ─────────────────────────────────────────────────────────
def test_pre_registered_conditions_are_present() -> None:
    assert set(PRIMARY_CONDITION_ORDER) == set(CONDITIONS)
    assert CONDITIONS["C0"].monitoring.coverage == 0.0
    assert CONDITIONS["C0"].monitoring.fidelity is None
    assert CONDITIONS["C1"].monitoring == MonitoringConfig(coverage=COVERAGE_PRIMARY, fidelity=0.0)
    assert CONDITIONS["C2"].monitoring == MonitoringConfig(coverage=COVERAGE_PRIMARY, fidelity=1.0)
    assert CONDITIONS["C4"].monitoring.decorrelated is True
    assert CONDITIONS["C4"].monitoring.fidelity == 1.0
    sweep = [CONDITIONS[f"C3_f{f:g}"].monitoring.fidelity for f in FIDELITY_SWEEP]
    assert sweep == list(FIDELITY_SWEEP)
    assert all(CONDITIONS[f"C3_f{f:g}"].family == "C3" for f in FIDELITY_SWEEP)


def test_expected_penalty_matched_between_c1_and_c2() -> None:
    c1, c2 = CONDITIONS["C1"].monitoring, CONDITIONS["C2"].monitoring
    assert c1.expected_penalty_per_defection == c2.expected_penalty_per_defection == COVERAGE_PRIMARY * PENALTY_PRIMARY
    assert CONDITIONS["C0"].monitoring.expected_penalty_per_defection == 0.0


def test_robustness_conditions_rename_and_rescale_penalty() -> None:
    robust = robustness_conditions()
    assert set(robust) == {f"{base}_p{p:g}" for p in PENALTY_ROBUSTNESS for base in ("C1", "C2")}
    for p in PENALTY_ROBUSTNESS:
        assert robust[f"C1_p{p:g}"].monitoring.penalty == p
        assert robust[f"C2_p{p:g}"].family == "C2"
        assert robust[f"C2_p{p:g}"].monitoring.fidelity == 1.0


def test_flip_probability_follows_binary_symmetric_channel() -> None:
    assert MonitoringConfig(coverage=0.5, fidelity=1.0).flip_probability == 0.0
    assert MonitoringConfig(coverage=0.5, fidelity=0.0).flip_probability == 0.5
    assert MonitoringConfig(coverage=0.5, fidelity=0.5).flip_probability == pytest.approx(0.25)
    assert MonitoringConfig(coverage=0.0, fidelity=None).flip_probability == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coverage": 1.5},
        {"coverage": 0.5, "fidelity": 2.0},
        {"coverage": 0.5, "penalty": -1.0},
        {"coverage": 0.5, "fidelity": None, "decorrelated": True},
        {"coverage": 0.5, "epoch_ticks": 0},
    ],
)
def test_monitoring_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MonitoringConfig(**kwargs)


def test_run_config_validation() -> None:
    with pytest.raises(ValueError):
        RunConfig(condition=CONDITIONS["C0"], seed=1, inheritance_mode="genetic")
    with pytest.raises(ValueError):
        RunConfig(condition=CONDITIONS["C0"], seed=1, train_ticks=100, window_ticks=30)
    cfg = RunConfig(condition=CONDITIONS["C2"], seed=7, inheritance_mode="lamarckian", train_ticks=100, eval_ticks=20)
    assert cfg.cell_id == "C2__lamarckian"
    assert cfg.run_id == "C2__lamarckian__s7"
    assert cfg.total_ticks == 120


def test_learner_and_threshold_validation() -> None:
    with pytest.raises(ValueError):
        LearnerConfig(gamma=1.0)
    with pytest.raises(ValueError):
        LearnerConfig(epsilon_start=0.1, epsilon_min=0.2)
    with pytest.raises(ValueError):
        AnalysisThresholds(auc_high=0.5, auc_collapse=0.7)
    with pytest.raises(ValueError):
        AnalysisThresholds(baseline_bounds=(0.9, 0.1))


def test_world_regions_and_heldout_band() -> None:
    w = WorldConfig()
    assert w.n_cells == w.width * w.height
    assert w.is_heldout(w.heldout_min_x, 0) and not w.is_heldout(w.heldout_min_x - 1, 0)
    regions = {w.region_of(x, y) for x in range(w.heldout_min_x) for y in range(w.height)}
    assert regions == {0, 1, 2, 3}
    assert w.region_of(w.width - 1, w.height - 1) == 4


# ── resource field ────────────────────────────────────────────────────────
@pytest.fixture
def field() -> ResourceField:
    return ResourceField(WorldConfig(), np.random.default_rng(3))


def test_gather_never_crosses_regeneration_threshold(field: ResourceField) -> None:
    cfg = field.cfg
    node = 0
    field.amount[node] = cfg.regen_threshold + 1.0
    taken = field.gather(node)
    assert taken == pytest.approx(1.0)
    assert field.amount[node] == pytest.approx(cfg.regen_threshold)
    assert field.gather(node) == 0.0
    assert field.amount[node] == pytest.approx(cfg.regen_threshold)


def test_over_harvest_draws_below_threshold_and_suppresses_regen(field: ResourceField) -> None:
    cfg = field.cfg
    node = 0
    field.amount[node] = cfg.regen_threshold
    taken = field.over_harvest(node)
    assert taken == pytest.approx(min(cfg.gather_amount, cfg.regen_threshold))
    assert field.amount[node] < cfg.regen_threshold
    depleted = field.amount[node]
    healthy = 1
    field.amount[healthy] = cfg.regen_threshold + 0.5
    before_healthy = field.amount[healthy]
    field.regenerate()
    growth_depleted = field.amount[node] - depleted
    growth_healthy = field.amount[healthy] - before_healthy
    assert growth_depleted >= 0.0
    assert growth_healthy > growth_depleted


def test_regeneration_saturates_at_node_max(field: ResourceField) -> None:
    field.amount[:] = field.cfg.node_max_amount
    field.regenerate()
    assert np.all(field.amount <= field.cfg.node_max_amount)


def test_defection_opportunity_is_individually_profitable(field: ResourceField) -> None:
    cfg = field.cfg
    node = 0
    field.amount[node] = cfg.node_max_amount
    assert not field.is_defection_opportunity(node)
    field.amount[node] = cfg.regen_threshold + 0.5
    assert field.is_defection_opportunity(node)
    assert field.over_harvest_yield(node) - cfg.over_harvest_effort > field.gather_yield(node)
    field.amount[node] = 0.0
    assert not field.is_defection_opportunity(node)
    assert not field.is_defection_opportunity(-1)


def test_field_layout_is_seed_deterministic() -> None:
    a = ResourceField(WorldConfig(), np.random.default_rng(11))
    b = ResourceField(WorldConfig(), np.random.default_rng(11))
    c = ResourceField(WorldConfig(), np.random.default_rng(12))
    assert np.array_equal(a.xs, b.xs) and np.array_equal(a.ys, b.ys)
    assert not (np.array_equal(a.xs, c.xs) and np.array_equal(a.ys, c.ys))


# ── monitoring ────────────────────────────────────────────────────────────
def _monitoring(cfg: MonitoringConfig, seed: int = 5) -> Monitoring:
    streams = np.random.SeedSequence(seed).spawn(3)
    return Monitoring(WorldConfig(), cfg, *(np.random.default_rng(s) for s in streams))


def test_coverage_is_exact_over_training_cells_only() -> None:
    mon = _monitoring(MonitoringConfig(coverage=0.5, fidelity=1.0))
    assert mon.maybe_resample(0, include_heldout=False)
    world = mon.world
    xs = np.arange(world.n_cells) % world.width
    assert mon.true_mask[xs >= world.heldout_min_x].sum() == 0
    assert mon.true_mask.sum() == round(0.5 * (xs < world.heldout_min_x).sum())
    assert mon.coverage_realised() == pytest.approx(0.5)


def test_masks_resample_only_at_epoch_boundaries_and_on_phase_change() -> None:
    mon = _monitoring(MonitoringConfig(coverage=0.5, fidelity=1.0, epoch_ticks=25))
    mon.maybe_resample(0, include_heldout=False)
    first = mon.true_mask.copy()
    assert not mon.maybe_resample(1, include_heldout=False)
    assert np.array_equal(first, mon.true_mask)
    assert mon.maybe_resample(25, include_heldout=False)
    assert not np.array_equal(first, mon.true_mask)
    assert mon.maybe_resample(26, include_heldout=True)
    world = mon.world
    assert mon.true_mask[np.arange(world.n_cells) % world.width >= world.heldout_min_x].any()


def test_cue_fidelity_extremes() -> None:
    perfect = _monitoring(MonitoringConfig(coverage=0.5, fidelity=1.0))
    perfect.maybe_resample(0, include_heldout=False)
    cells = perfect.training_cells
    assert all(perfect.cue(int(c)) == int(perfect.true_mask[c]) for c in cells)

    blind = _monitoring(MonitoringConfig(coverage=0.5, fidelity=0.0))
    blind.maybe_resample(0, include_heldout=False)
    cues = np.array([blind.cue(int(c)) for c in cells for _ in range(20)])
    truth = np.array([int(blind.true_mask[c]) for c in cells for _ in range(20)])
    agreement = np.mean(cues == truth)
    assert abs(agreement - 0.5) < 0.05

    silent = _monitoring(MonitoringConfig(coverage=0.0, fidelity=None))
    silent.maybe_resample(0, include_heldout=False)
    assert all(silent.cue(int(c)) == 0 for c in cells)


def test_intermediate_fidelity_matches_channel_accuracy() -> None:
    mon = _monitoring(MonitoringConfig(coverage=0.5, fidelity=0.6))
    mon.maybe_resample(0, include_heldout=False)
    cells = mon.training_cells
    cues = np.array([mon.cue(int(c)) for c in cells for _ in range(30)])
    truth = np.array([int(mon.true_mask[c]) for c in cells for _ in range(30)])
    expected_accuracy = 1.0 - mon.cfg.flip_probability
    assert abs(np.mean(cues == truth) - expected_accuracy) < 0.03


def test_decorrelated_cue_tracks_decoy_not_true_mask() -> None:
    mon = _monitoring(MonitoringConfig(coverage=0.5, fidelity=1.0, decorrelated=True))
    mon.maybe_resample(0, include_heldout=False)
    cells = mon.training_cells
    cues = np.array([mon.cue(int(c)) for c in cells])
    assert np.array_equal(cues, mon.decoy_mask[cells].astype(int))
    assert not np.array_equal(cues, mon.true_mask[cells].astype(int))
    assert mon.decoy_mask.sum() == mon.true_mask.sum()


def test_true_mask_is_shared_across_conditions_with_same_seed() -> None:
    c1 = _monitoring(CONDITIONS["C1"].monitoring, seed=9)
    c2 = _monitoring(CONDITIONS["C2"].monitoring, seed=9)
    c4 = _monitoring(CONDITIONS["C4"].monitoring, seed=9)
    for mon in (c1, c2, c4):
        mon.maybe_resample(0, include_heldout=False)
    assert np.array_equal(c1.true_mask, c2.true_mask)
    assert np.array_equal(c2.true_mask, c4.true_mask)
