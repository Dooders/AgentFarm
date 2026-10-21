"""Tests for ``scripts/calibrate_intrinsic_cooperation.py``."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip("scipy")

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.cooperation.genes import COOPERATION_GENE_NAMES
from farm.runners.intrinsic_cooperation_experiment import DEFAULT_ARMS, HYPOTHESES, TRACKED_ACTIONS
from scripts.calibrate_intrinsic_cooperation import (
    GATES,
    REPORT_FILE,
    REPORT_JSON,
    build_report,
    main,
    mde_paired,
    n_for_power,
    power_paired,
    render_markdown,
)

pytestmark = pytest.mark.unit

_SCALARS = {
    "mean_population": 40.0,
    "final_population": 42.0,
    "peak_population": 55.0,
    "total_births": 30.0,
    "total_deaths": 20.0,
    "mean_coverage": 0.0,
    "final_coverage": 0.0,
    "mean_units_active": 0.0,
    "units_formed": 0.0,
    "mean_unit_lifetime": 0.0,
    "bond_share_late": 0.01,
    "policy_bond_prob_late": 0.05,
    "acting_bond_prob_late": 0.04,
    "bond_rank_late": 6.0,
    "policy_gather_prob_late": 0.5,
    "policy_gather_lift_late": 0.375,
    "gather_share_over_prior_late": 0.3,
    "policy_peak_late": 0.6,
    "mean_lifespan_at_death": 80.0,
    "starvation_deaths": 10.0,
    "combat_deaths": 2.0,
    "max_generation": 4.0,
    "change_expected_exit_payment": 0.0,
    "forfeits_per_enforced_exit": float("nan"),
}


def _arm(arm: str, seed: int) -> dict:
    rng = np.random.default_rng(seed * 7 + len(arm))
    out = dict(_SCALARS)
    out["mean_population"] += rng.normal(0, 1.0)
    if arm.startswith("cooperation"):
        out.update(mean_coverage=0.3, final_coverage=0.25, units_formed=40.0, mean_unit_lifetime=30.0)
        out["mean_population"] += 6.0
        out["mean_lifespan_at_death"] += 10.0
        out["forfeits_per_enforced_exit"] = 1.2
        out["change_expected_exit_payment"] = 0.1 + rng.normal(0, 0.02)
    out["mean_action_share"] = {a: (0.1 if a in ("bond", "leave") else 0.8 / 8) for a in TRACKED_ACTIONS}
    out["gene_mean_end"] = dict.fromkeys(COOPERATION_GENE_NAMES, 0.5)
    out["gene_mean_change"] = {
        g: (0.05 if arm == "cooperation" and g == "coop_exit_cost" else 0.0) + rng.normal(0, 0.01)
        for g in COOPERATION_GENE_NAMES
    }
    return out


def _summary(seeds=(1, 2, 3), cap=60) -> dict:
    return {
        "config": {
            "num_steps": 400,
            "seeds": list(seeds),
            "seed": seeds[0],
            "num_replicates": len(seeds),
            "arms": list(DEFAULT_ARMS),
            "max_population": cap,
            "alpha": 0.05,
            "selection_pressure": "medium",
            "policy_temperature": 0.1,
            "git_commit": "abcdef1234567890",
        },
        "replicates": [{"index": i, "seed": s, **{a: _arm(a, s) for a in DEFAULT_ARMS}} for i, s in enumerate(seeds)],
        "aggregate": None,
        "hypotheses": None,
    }


def test_power_model_is_monotone_and_consistent():
    assert power_paired(0.0, 20, 0.05, "greater") == pytest.approx(0.05, abs=0.01)
    assert power_paired(0.5, 20, 0.05, "greater") > power_paired(0.3, 20, 0.05, "greater")
    assert power_paired(0.5, 40, 0.05, "greater") > power_paired(0.5, 20, 0.05, "greater")
    assert power_paired(0.5, 20, 0.05, "two-sided") < power_paired(0.5, 20, 0.05, "greater")
    mde = mde_paired(20, 0.05, "greater")
    assert power_paired(mde, 20, 0.05, "greater") == pytest.approx(0.8, abs=0.01)
    n = n_for_power(0.8, 0.05, "two-sided")
    assert n is not None and power_paired(0.8, n, 0.05, "two-sided") >= 0.8 > power_paired(
        0.8, n - 1, 0.05, "two-sided"
    )
    assert n_for_power(0.0, 0.05, "greater") is None


def test_report_covers_gates_learning_drift_and_every_hypothesis():
    report = build_report(_summary(), target_n=20)
    assert set(report["gates"]) == set(GATES)
    assert report["gates"]["learning_L1"]["passed"] and report["gates"]["learning_L2"]["passed"]
    assert report["gates"]["units_form"]["passed"] and report["gates"]["generations"]["passed"]
    assert report["gates"]["not_pinned"]["passed"]  # 46 / 60 < 0.95
    assert report["gates"]["exit_cost_bounded"]["passed"]
    assert set(report["learning"]) == {"L1", "L2"} and len(report["learning"]["L1"]["per_replicate"]) == 3
    assert "change_coop_exit_cost" in report["drift"] and "change_expected_exit_payment" in report["drift"]
    assert report["drift"]["change_coop_exit_cost"]["paired_diff"]["mean"] > 0.03
    assert set(report["hypotheses"]) == set(HYPOTHESES)
    s3 = report["hypotheses"]["S3"]
    assert s3["pilot_n"] == 3 and s3["mean_diff"] > 0 and 0 <= s3["power_at_target_n"] <= 1
    assert report["alpha_strict"] == pytest.approx(0.05 / len(HYPOTHESES))
    assert set(report["regime"]) == set(DEFAULT_ARMS)
    assert report["regime"]["cooperation"]["mean_coverage"]["mean"] == pytest.approx(0.3)


def test_gates_fail_when_the_regime_is_wrong():
    summary = _summary(cap=40)  # every arm sits above 0.95 x cap
    for rep in summary["replicates"]:
        rep["baseline"]["policy_gather_lift_late"] = -0.01
        rep["cooperation"]["max_generation"] = 1
    report = build_report(summary)
    assert report["gates"]["not_pinned"]["passed"] is False
    assert report["gates"]["learning_L1"]["passed"] is False
    assert report["gates"]["generations"]["passed"] is False
    markdown = render_markdown(report)
    assert "| not_pinned | FAIL |" in markdown and "| learning_L2 | pass |" in markdown


def test_main_writes_markdown_and_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "intrinsic_cooperation_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_summary(), handle)
        assert main([path, "--target-n", "12"]) == 0
        assert os.path.exists(os.path.join(tmp, REPORT_FILE))
        with open(os.path.join(tmp, REPORT_JSON), encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["target_n"] == 12 and "S4_exit_cost" in report["hypotheses"]
        with open(os.path.join(tmp, REPORT_FILE), encoding="utf-8") as handle:
            text = handle.read()
        assert "## Gates" in text and "| S5 |" in text and "`abcdef1234`" in text


def test_main_refuses_a_single_replicate():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_summary(seeds=(1,)), handle)
        with pytest.raises(SystemExit, match="two replicates"):
            main([path])
