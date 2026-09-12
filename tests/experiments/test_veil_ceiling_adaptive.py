"""Tests for adaptive monitoring: reallocation of a fixed coverage budget."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farm.experiments.veil_ceiling.adaptive import (
    ADAPTIVE_REPORT_FILENAME,
    analyze_adaptive,
    compute_adaptive_contrasts,
    write_adaptive,
)
from farm.experiments.veil_ceiling.config import (
    ADAPTIVE_CONDITION_ORDER,
    ADAPTIVE_MATRIX_CONDITION_ORDER,
    CONDITIONS,
    MONITOR_POLICY_ADAPTIVE_BLIND,
    MONITOR_POLICY_ADAPTIVE_CELLS,
    MONITOR_POLICY_ADAPTIVE_MOVE,
    AnalysisThresholds,
    MonitoringConfig,
    RunConfig,
    WorldConfig,
)
from farm.experiments.veil_ceiling.experiment import MatrixConfig, run_matrix
from farm.experiments.veil_ceiling.monitoring import RESIDUAL_DEFECTIONS, RESIDUAL_MOVES, Monitoring
from farm.experiments.veil_ceiling.simulation import run_simulation

pytestmark = pytest.mark.unit


def _monitoring(cfg: MonitoringConfig, seed: int = 5) -> Monitoring:
    streams = np.random.SeedSequence(seed).spawn(3)
    return Monitoring(WorldConfig(), cfg, *(np.random.default_rng(s) for s in streams))


def test_adaptive_conditions_are_registered() -> None:
    assert len(ADAPTIVE_CONDITION_ORDER) == 9
    for name in ADAPTIVE_CONDITION_ORDER:
        cond = CONDITIONS[name]
        assert cond.family == "A"
        assert cond.monitoring.coverage == CONDITIONS["C2"].monitoring.coverage
        assert cond.monitoring.penalty == CONDITIONS["C2"].monitoring.penalty
        assert cond.monitoring.policy in (
            MONITOR_POLICY_ADAPTIVE_CELLS,
            MONITOR_POLICY_ADAPTIVE_MOVE,
            MONITOR_POLICY_ADAPTIVE_BLIND,
        )
    assert "C2" in ADAPTIVE_MATRIX_CONDITION_ORDER
    assert CONDITIONS["C2"].monitoring.policy == "static"


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        MonitoringConfig(coverage=0.5, fidelity=1.0, policy="oracle")
    with pytest.raises(ValueError):
        MonitoringConfig(coverage=0.0, fidelity=1.0, policy=MONITOR_POLICY_ADAPTIVE_CELLS)


def test_first_epoch_matches_static_true_mask() -> None:
    static = _monitoring(CONDITIONS["C2"].monitoring, seed=11)
    adaptive = _monitoring(CONDITIONS["A_f1_cells"].monitoring, seed=11)
    static.maybe_resample(0, include_heldout=False)
    adaptive.maybe_resample(0, include_heldout=False)
    assert np.array_equal(static.true_mask, adaptive.true_mask)
    assert np.array_equal(static.decoy_mask, adaptive.decoy_mask)


def test_adaptive_draw_overweights_high_residual_cells() -> None:
    cfg = MonitoringConfig(coverage=0.5, fidelity=1.0, policy=MONITOR_POLICY_ADAPTIVE_CELLS)
    mon = _monitoring(cfg, seed=3)
    mon.maybe_resample(0, include_heldout=False)
    hot = int(mon.training_cells[0])
    cold = int(mon.training_cells[1])
    for _ in range(50):
        mon.record_residual(hot, RESIDUAL_DEFECTIONS)
    weights = mon._weights(mon.training_cells)
    by_cell = dict(zip(mon.training_cells.tolist(), weights.tolist()))
    assert by_cell[hot] > 10 * by_cell[cold]
    mon.maybe_resample(25, include_heldout=False)
    assert mon.n_adaptive_draws == 1
    assert mon.coverage_realised() == pytest.approx(0.5)
    assert mon.last_kl > 0.0
    assert 0.0 <= mon.last_weight_gini <= 1.0


def test_adaptive_move_ignores_defection_residual() -> None:
    cfg = MonitoringConfig(coverage=0.5, fidelity=1.0, policy=MONITOR_POLICY_ADAPTIVE_MOVE)
    mon = _monitoring(cfg, seed=4)
    mon.maybe_resample(0, include_heldout=False)
    cold = int(mon.training_cells[0])
    hot = int(mon.training_cells[1])
    for _ in range(40):
        mon.record_residual(cold, RESIDUAL_DEFECTIONS)
        mon.record_residual(hot, RESIDUAL_MOVES)
    weights = mon._weights(mon.training_cells)
    by_cell = dict(zip(mon.training_cells.tolist(), weights.tolist()))
    assert by_cell[hot] > 10 * by_cell[cold]
    mon.maybe_resample(25, include_heldout=False)
    assert mon.true_mask[hot] or by_cell[hot] > by_cell[cold]


def test_coverage_stays_exact_after_adaptive_epochs() -> None:
    mon = _monitoring(CONDITIONS["A_f1_cells"].monitoring, seed=8)
    mon.maybe_resample(0, include_heldout=False)
    for cell in mon.training_cells[:10]:
        mon.record_residual(int(cell), RESIDUAL_DEFECTIONS)
    mon.maybe_resample(25, include_heldout=False)
    world = mon.world
    xs = np.arange(world.n_cells) % world.width
    assert mon.true_mask[xs >= world.heldout_min_x].sum() == 0
    assert mon.true_mask.sum() == round(0.5 * (xs < world.heldout_min_x).sum())


def test_simulation_adaptive_run_records_telemetry() -> None:
    cfg = RunConfig(
        condition=CONDITIONS["A_f1_cells"],
        seed=2,
        inheritance_mode="lamarckian",
        train_ticks=100,
        eval_ticks=50,
        window_ticks=50,
    )
    result = run_simulation(cfg)
    assert result.summary["monitor_policy"] == MONITOR_POLICY_ADAPTIVE_CELLS
    assert result.summary["adaptive_draws"] > 0
    assert result.summary["expected_penalty_per_defection"] == pytest.approx(3.0)
    assert "mask_overlap" in result.windows.columns
    assert result.windows["coverage_realised"].dropna().between(0.49, 0.51).all()


def test_static_run_does_not_count_adaptive_draws() -> None:
    cfg = RunConfig(
        condition=CONDITIONS["C2"],
        seed=2,
        train_ticks=100,
        eval_ticks=0,
        window_ticks=50,
    )
    result = run_simulation(cfg)
    assert result.summary["monitor_policy"] == "static"
    assert result.summary["adaptive_draws"] == 0


TINY = MatrixConfig(
    seeds=(1, 2),
    condition_names=("C2", "A_f1_cells", "A_f1_blind"),
    include_robustness=False,
    train_ticks=100,
    eval_ticks=50,
    window_ticks=50,
    workers=1,
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("veil_adaptive")
    return out, run_matrix(TINY, out, progress=None)


def test_adaptive_matrix_and_analysis(tiny) -> None:
    out, outputs = tiny
    assert len(outputs.runs) == 12
    assert set(outputs.runs["monitor_policy"]) >= {
        "static",
        MONITOR_POLICY_ADAPTIVE_CELLS,
        MONITOR_POLICY_ADAPTIVE_BLIND,
    }
    result = analyze_adaptive(outputs, AnalysisThresholds(bootstrap_reps=40))
    assert not result.contrasts.empty
    assert {"adaptive-static", "cells-blind"} <= set(result.contrasts["contrast"])
    report = write_adaptive(result, outputs, out)
    assert report == Path(out) / ADAPTIVE_REPORT_FILENAME
    text = report.read_text(encoding="utf-8")
    assert "Adaptive monitor" in text
    assert (Path(out) / "figures" / "adaptive_delta.png").exists()
    assert (Path(out) / "analysis" / "adaptive" / "contrasts.csv").exists()
    reloaded = pd.read_csv(Path(out) / "analysis" / "adaptive" / "contrasts.csv")
    assert len(reloaded) == len(result.contrasts)


def test_compute_adaptive_contrasts_requires_family_a() -> None:
    empty = compute_adaptive_contrasts(pd.DataFrame({"family": ["C2"], "inheritance_mode": ["baldwinian"]}))
    assert empty.empty
