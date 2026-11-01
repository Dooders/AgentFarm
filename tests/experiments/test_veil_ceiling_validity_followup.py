"""Tests for the validity follow-up (feature ablation, honest-calibrated evaluator, profiles)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farm.experiments.veil_ceiling.config import AnalysisThresholds
from farm.experiments.veil_ceiling.experiment import MatrixConfig, run_matrix
from farm.experiments.veil_ceiling.metrics import OBSERVED_FEATURES, agent_validity_table, predictive_validity
from farm.experiments.veil_ceiling.validity_followup import (
    CALIBRATION_CONDITIONS,
    FEATURE_SETS,
    FOLLOWUP_DIRNAME,
    FOLLOWUP_REPORT_FILENAME,
    cross_condition_validity,
    feature_ablation,
    feature_profiles,
    run_followup,
    write_followup,
)

pytestmark = pytest.mark.unit

FAST = AnalysisThresholds(bootstrap_reps=30)
TINY = MatrixConfig(
    seeds=(1, 2),
    condition_names=("C1", "C2", "C4"),
    include_robustness=False,
    train_ticks=100,
    eval_ticks=0,
    window_ticks=50,
    workers=1,
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("veil_followup")
    return out, run_matrix(TINY, out, progress=None)


def test_feature_sets_are_valid_subsets_of_observed_features() -> None:
    for name, features in FEATURE_SETS.items():
        assert features, name
        assert set(features) <= set(OBSERVED_FEATURES), name
        assert len(set(features)) == len(features), name
    assert FEATURE_SETS["full"] == tuple(OBSERVED_FEATURES)
    assert FEATURE_SETS["defect_rate"] == ("obs_defect_rate",)


def test_predictive_validity_respects_feature_subset(tiny) -> None:
    _, outputs = tiny
    table = agent_validity_table(outputs.agents_by_cell["C1__baldwinian"], FAST)
    full = predictive_validity(table, FAST, np.random.default_rng(0))
    single = predictive_validity(table, FAST, np.random.default_rng(0), features=("obs_log_ticks",))
    assert full.n_agents == single.n_agents
    assert single.auc_rank == full.auc_rank
    assert single.auc_classifier != full.auc_classifier


def test_feature_ablation_covers_every_cell_and_set(tiny) -> None:
    _, outputs = tiny
    ablation = feature_ablation(outputs, FAST)
    assert len(ablation) == len(outputs.cells()) * len(FEATURE_SETS)
    assert set(ablation["feature_set"]) == set(FEATURE_SETS)
    full = ablation[ablation["feature_set"] == "full"]
    assert (full["n_features"] == len(OBSERVED_FEATURES)).all()
    finite = ablation.dropna(subset=["auc"])
    assert ((finite["auc_lo"] <= finite["auc"] + 1e-9) & (finite["auc"] <= finite["auc_hi"] + 1e-9)).all()
    assert finite["auc"].between(0.0, 1.0).all()


def test_cross_condition_validity_excludes_self_and_bounds_auc(tiny) -> None:
    _, outputs = tiny
    cross = cross_condition_validity(outputs, FAST)
    assert set(cross["calibration_condition"]) == set(CALIBRATION_CONDITIONS)
    assert not (cross["condition"] == cross["calibration_condition"]).any()
    for mode in ("baldwinian", "lamarckian"):
        sub = cross[cross["inheritance_mode"] == mode]
        assert set(sub["condition"]) <= {"C1", "C2", "C4"}
        assert (sub["inheritance_mode"] == mode).all()
    assert cross["auc"].between(0.0, 1.0).all()


def test_cross_condition_validity_skips_single_class_calibration_target(tiny) -> None:
    _, outputs = tiny
    modified = dict(outputs.agents_by_cell)
    for cell in ("C1__baldwinian", "C1__lamarckian"):
        agents = modified[cell].copy()
        for c in (0, 1):
            agents[f"opportunities__train__train__m0__c{c}"] = 30.0
            agents[f"opportunities__train__train__m1__c{c}"] = 30.0
            agents[f"defections__train__train__m0__c{c}"] = 0.0
        modified[cell] = agents
    degenerate = outputs.__class__(
        runs=outputs.runs,
        windows=outputs.windows,
        agents_by_cell=modified,
        train_ticks=outputs.train_ticks,
    )
    cross = cross_condition_validity(degenerate, FAST, calibration_conditions=("C1",))
    assert cross.empty


def test_feature_profiles_and_coefficients(tiny) -> None:
    _, outputs = tiny
    profiles, coefficients = feature_profiles(outputs, FAST)
    assert set(profiles["condition"]) <= {"C1", "C2", "C4"}
    for _, group in profiles.groupby("cell_id"):
        assert group["share"].sum() == pytest.approx(1.0)
    for f in OBSERVED_FEATURES:
        assert f"mean_{f}" in profiles.columns
        assert f"coef_{f}" in coefficients.columns
    assert 0 < len(coefficients) <= len(outputs.cells())


def test_run_and_write_followup(tiny) -> None:
    out, outputs = tiny
    result = run_followup(outputs, FAST)
    report = write_followup(result, outputs, out, FAST)
    assert report == Path(out) / FOLLOWUP_REPORT_FILENAME
    text = report.read_text(encoding="utf-8")
    for heading in ("Feature ablation", "Honest-calibrated evaluator", "Which way does each channel leak", "Summary"):
        assert heading in text
    table_dir = Path(out) / "analysis" / FOLLOWUP_DIRNAME
    assert {p.name for p in table_dir.iterdir()} == {f"{name}.csv" for name in result.tables()}
    figures = Path(out) / "figures"
    assert (figures / "validity_ablation.png").exists() and (figures / "cross_condition_validity.png").exists()
    reloaded = pd.read_csv(table_dir / "feature_ablation.csv")
    assert len(reloaded) == len(result.ablation)
