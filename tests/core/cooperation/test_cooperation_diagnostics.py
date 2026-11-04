"""Failure-mode classification in :mod:`farm.core.cooperation.diagnostics`."""

import math

import numpy as np
import pytest

from farm.core.cooperation import (
    CooperationRegistry,
    CooperationSettings,
    DiagnosticThresholds,
    Entry,
    FailureMode,
    UnitEconomics,
    UnitGoals,
    diagnose_all,
    diagnose_unit,
    mode_prevalence,
)

pytestmark = pytest.mark.unit

T = DiagnosticThresholds(min_steps=5, window=10, fragility_lifetime=8, trap_steps=3)


def _registry(**econ):
    defaults = {"synergy": 0.2, "payout_rate": 1.0, "aid_threshold": 0.0, "aid_amount": 0.0}
    defaults.update(econ)
    return CooperationRegistry(
        economics=UnitEconomics(**defaults),
        unit_goals=UnitGoals(reserve_target=0.0),
        rng=np.random.default_rng(0),
    )


def _pair(**kw):
    base = {"kind": "pair", "min_members": 2, "max_members": 2, "scope": 0.5}
    base.update(kw)
    return CooperationSettings(**base)


def _run(reg, incomes, steps, start=1):
    for step in range(start, start + steps):
        for agent_id, income in incomes.items():
            reg.contribute(agent_id, income, step)
        reg.settle(step)
    return start + steps


def test_symmetric_equal_pair_is_healthy():
    reg = _registry()
    unit = reg.form(_pair(), ["a", "b"], step=0)
    end = _run(reg, {"a": 1.0, "b": 1.0}, 20)
    diagnosis = diagnose_unit(unit, end, T)
    assert diagnosis.primary is FailureMode.HEALTHY
    assert diagnosis.modes == [FailureMode.HEALTHY]


def test_young_active_unit_is_indeterminate():
    reg = _registry()
    unit = reg.form(_pair(), ["a", "b"], step=0)
    end = _run(reg, {"a": 1.0, "b": 1.0}, 2)
    assert diagnose_unit(unit, end, T).primary is FailureMode.INDETERMINATE


def test_early_dissolution_is_fragility():
    reg = _registry()
    unit = reg.form(_pair(), ["a", "b"], step=0)
    end = _run(reg, {"a": 1.0, "b": 1.0}, 3)
    reg.leave(unit.unit_id, "a", step=end)
    assert diagnose_unit(unit, end + 1, T).primary is FailureMode.FRAGILITY


def test_rank_weighted_payouts_with_equal_inputs_is_capture():
    reg = _registry()
    unit = reg.form(_pair(symmetry=0.0), ["head", "junior"], step=0)
    end = _run(reg, {"head": 1.0, "junior": 1.0}, 20)
    diagnosis = diagnose_unit(unit, end, T)
    assert FailureMode.CAPTURE in diagnosis.modes
    assert diagnosis.evidence["capture_share"] == pytest.approx(0.8)


def test_symmetric_redistribution_from_rich_to_poor_is_not_capture():
    reg = _registry()
    unit = reg.form(_pair(), ["rich", "poor"], step=0)
    end = _run(reg, {"rich": 4.0, "poor": 0.5}, 20)
    assert FailureMode.CAPTURE not in diagnose_unit(unit, end, T).modes


def test_tiny_scope_is_hollow():
    reg = _registry()
    unit = reg.form(_pair(scope=0.01), ["a", "b"], step=0)
    end = _run(reg, {"a": 1.0, "b": 1.0}, 20)
    assert diagnose_unit(unit, end, T).primary is FailureMode.HOLLOW


def test_recorded_price_blocked_exit_desire_is_trap():
    reg = _registry()
    unit = reg.form(_pair(), ["a", "b"], step=0)
    end = _run(reg, {"a": 1.0, "b": 1.0}, 10)
    for _ in range(3):
        reg.record_exit_desire(unit.unit_id, "a", blocked_by_price=True)
    assert diagnose_unit(unit, end, T).primary is FailureMode.TRAP


def test_parent_veto_and_unconsented_arrangement_are_override():
    reg = _registry()
    arranged = reg.form(_pair(), ["a", "b"], step=0, arranged=True, consents={"a": False, "b": True})
    end = _run(reg, {"a": 1.0, "b": 1.0}, 10)
    assert diagnose_unit(arranged, end, T).primary is FailureMode.OVERRIDE

    couple = reg.form(_pair(), ["c", "d"], step=end)
    reg.form(
        CooperationSettings(kind="family", min_members=1, max_members=None, recognition=1.0, entry=Entry.BORN),
        [],
        step=end,
        children=[couple.unit_id],
    )
    end2 = _run(reg, {"c": 1.0, "d": 1.0}, 10, start=end + 1)
    assert reg.leave(couple.unit_id, "c", end2).vetoed
    vetoed = [e.unit_id for e in reg.events if e.kind == "exit_vetoed"]
    assert FailureMode.OVERRIDE in diagnose_unit(couple, end2, T, vetoed).modes
    assert FailureMode.OVERRIDE not in diagnose_unit(couple, end2, T).modes


def test_heavy_parent_dues_make_members_net_losers_which_is_drift():
    reg = _registry(synergy=0.1)
    couple = reg.form(_pair(scope=0.8), ["a", "b"], step=0)
    family = reg.form(
        CooperationSettings(kind="family", min_members=1, max_members=None, scope=0.9, entry=Entry.BORN),
        ["elder"],
        step=0,
        children=[couple.unit_id],
    )
    end = _run(reg, {"a": 2.0, "b": 2.0, "elder": 0.0}, 20)
    diagnosis = diagnose_unit(couple, end, T)
    assert FailureMode.DRIFT in diagnosis.modes
    assert diagnosis.evidence["window_member_net"] < 0
    assert family.members["elder"].net > 0


def test_mode_prevalence_excludes_indeterminate_and_sums_primaries():
    reg = _registry()
    healthy = reg.form(_pair(), ["a", "b"], step=0)
    hollow = reg.form(_pair(scope=0.01), ["c", "d"], step=0)
    end = _run(reg, {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}, 20)
    young = reg.form(_pair(), ["e", "f"], step=end)
    diagnoses = diagnose_all([healthy, hollow, young], end + 1, T)
    prevalence = mode_prevalence(diagnoses)
    assert prevalence["n_classified"] == 2
    assert prevalence["healthy"] == pytest.approx(0.5)
    assert prevalence["hollow"] == pytest.approx(0.5)
    primaries = sum(v for k, v in prevalence.items() if k.startswith("primary_"))
    assert primaries == pytest.approx(1.0)
    assert math.isnan(mode_prevalence([], kind="pair")["healthy"])


def test_thresholds_are_validated():
    with pytest.raises(ValueError):
        DiagnosticThresholds(min_steps=0)
    with pytest.raises(ValueError):
        DiagnosticThresholds(capture_share=1.5)
