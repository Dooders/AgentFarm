"""Ledger priors, multi-step exit desire and death-ended units."""

import numpy as np
import pytest

from farm.core.cooperation import (
    CooperationRegistry,
    CooperationSettings,
    DiagnosticThresholds,
    FailureMode,
    UnitEconomics,
    UnitGoals,
    diagnose_unit,
)

pytestmark = pytest.mark.unit


def _registry(synergy=0.0):
    return CooperationRegistry(
        economics=UnitEconomics(
            synergy=synergy, payout_rate=1.0, aid_threshold=0.0, aid_amount=0.0, benefit_ema_alpha=0.5
        ),
        unit_goals=UnitGoals(reserve_target=0.0),
        rng=np.random.default_rng(0),
    )


PAIR = CooperationSettings(kind="pair", min_members=2, max_members=2, scope=1.0)


def test_priors_seed_the_running_benefit_estimate():
    reg = _registry()
    unit = reg.form(PAIR, ["a", "b"], step=0, priors={"a": 0.4})
    assert unit.members["a"].benefit_ema == pytest.approx(0.4)
    assert unit.members["b"].benefit_prior is None
    reg.contribute("a", 2.0, 1)
    reg.contribute("b", 0.0, 1)
    reg.settle(1)
    # a: received 1.0, contributed 2.0 -> step net -1.0; EMA 0.5 * 0.4 + 0.5 * -1.0
    assert unit.members["a"].benefit_ema == pytest.approx(-0.3)
    # b had no prior, so its estimate is its first step net.
    assert unit.members["b"].benefit_ema == pytest.approx(1.0)


def test_join_accepts_a_prior():
    reg = _registry()
    trio = CooperationSettings(kind="trio", max_members=3)
    unit = reg.form(trio, ["a", "b"], step=0)
    assert reg.join(unit.unit_id, "c", step=1, prior=-0.2)
    assert unit.members["c"].benefit_ema == pytest.approx(-0.2)


def test_exit_desire_can_be_recorded_for_a_review_interval():
    reg = _registry()
    unit = reg.form(PAIR, ["a", "b"], step=0)
    reg.record_exit_desire(unit.unit_id, "a", blocked_by_price=True, steps=10)
    assert unit.members["a"].trapped_steps == 10
    with pytest.raises(ValueError):
        reg.record_exit_desire(unit.unit_id, "a", blocked_by_price=True, steps=0)


def test_young_unit_ended_by_death_is_not_fragility():
    thresholds = DiagnosticThresholds(min_steps=5, fragility_lifetime=30)
    reg = _registry(synergy=0.2)
    died_young = reg.form(PAIR, ["a", "b"], step=0)
    for step in range(1, 4):
        reg.contribute("a", 1.0, step)
        reg.contribute("b", 1.0, step)
        reg.settle(step)
    reg.remove_agent("a", step=3, reason="starvation")
    assert died_young.dissolution_reason == "member_starvation"
    assert diagnose_unit(died_young, 4, thresholds).primary is FailureMode.INDETERMINATE

    died_later = reg.form(PAIR, ["c", "d"], step=0)
    for step in range(1, 12):
        reg.contribute("c", 1.0, step)
        reg.contribute("d", 1.0, step)
        reg.settle(step)
    reg.remove_agent("c", step=11, reason="age")
    diagnosis = diagnose_unit(died_later, 12, thresholds)
    assert FailureMode.FRAGILITY not in diagnosis.modes
    assert diagnosis.primary is FailureMode.HEALTHY

    left_young = reg.form(PAIR, ["e", "f"], step=0)
    reg.leave(left_young.unit_id, "e", step=2)
    assert diagnose_unit(left_young, 3, thresholds).primary is FailureMode.FRAGILITY
