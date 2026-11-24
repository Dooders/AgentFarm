"""Rule enforcement and pool accounting in :class:`CooperationRegistry`."""

import numpy as np
import pytest

from farm.core.cooperation import (
    CooperationRegistry,
    CooperationSettings,
    Entry,
    UnitEconomics,
    UnitGoals,
    preset,
)

pytestmark = pytest.mark.unit


def _registry(seed=0, **econ):
    defaults = {"synergy": 0.0, "payout_rate": 1.0, "aid_threshold": 0.0, "aid_amount": 0.0}
    defaults.update(econ)
    return CooperationRegistry(
        economics=UnitEconomics(**defaults),
        unit_goals=UnitGoals(reserve_target=0.0),
        rng=np.random.default_rng(seed),
    )


def _pair(**kw):
    base = {"kind": "pair", "min_members": 2, "max_members": 2, "scope": 0.5}
    base.update(kw)
    return CooperationSettings(**base)


# ------------------------------------------------------------------ formation
def test_form_enforces_member_bounds_and_exclusivity():
    reg = _registry()
    with pytest.raises(ValueError):
        reg.form(_pair(), ["a"], step=0)
    with pytest.raises(ValueError):
        reg.form(_pair(), ["a", "b", "c"], step=0)
    reg.form(_pair(), ["a", "b"], step=0)
    with pytest.raises(ValueError, match="exclusivity_limit"):
        reg.form(_pair(), ["a", "c"], step=1)
    assert reg.can_join("c", _pair()) == (True, "ok")
    assert reg.partners_of("a") == {"b"}


def test_non_exclusive_units_allow_several_memberships_up_to_limit():
    reg = _registry()
    friends = CooperationSettings(kind="friendship", max_members=None, exclusive=False, nonexclusive_limit=2)
    reg.form(friends, ["a", "b"], step=0)
    reg.form(friends, ["a", "c"], step=0)
    assert reg.can_join("a", friends) == (False, "exclusivity_limit")
    assert len(reg.units_of("a", "friendship")) == 2


def test_join_respects_capacity():
    reg = _registry()
    trio = CooperationSettings(kind="trio", max_members=3)
    unit = reg.form(trio, ["a", "b"], step=0)
    assert reg.join(unit.unit_id, "c", step=1)
    assert not reg.join(unit.unit_id, "d", step=1)
    assert reg.can_join("d", unit) == (False, "unit_full")


# ---------------------------------------------------------------------- flows
def test_contribute_pools_scope_and_splits_across_memberships():
    reg = _registry()
    friends = CooperationSettings(kind="friendship", max_members=None, exclusive=False, scope=0.4)
    u1 = reg.form(friends, ["a", "b"], step=0)
    u2 = reg.form(friends, ["a", "c"], step=0)
    pooled = reg.contribute("a", 10.0, step=1)
    assert pooled == pytest.approx(4.0)
    assert u1.pool == pytest.approx(2.0) and u2.pool == pytest.approx(2.0)
    assert reg.contribute("z", 10.0, step=1) == 0.0
    assert reg.contribute("b", -5.0, step=1) == 0.0


def test_settle_conserves_resources_without_synergy():
    reg = _registry(payout_rate=0.6)
    unit = reg.form(_pair(scope=0.5), ["a", "b"], step=0)
    total_in = 0.0
    total_out = 0.0
    for step in range(1, 20):
        total_in += reg.contribute("a", 3.0, step) + reg.contribute("b", 1.0, step)
        total_out += sum(reg.settle(step).values())
    assert total_in == pytest.approx(total_out + unit.pool)


def test_symmetric_pair_shares_payouts_equally_and_creates_half_synergy():
    reg = _registry(synergy=0.2)
    unit = reg.form(_pair(scope=1.0), ["a", "b"], step=0)
    reg.contribute("a", 3.0, 1)
    reg.contribute("b", 1.0, 1)
    pay = reg.settle(1)
    # inflow 4.0, pair surplus 0.2 * 4 * 0.5 = 0.4, full payout split evenly.
    assert pay["a"] == pytest.approx(2.2)
    assert pay["b"] == pytest.approx(2.2)
    assert unit.totals["surplus"] == pytest.approx(0.4)
    assert unit.members["b"].net == pytest.approx(1.2)


def test_asymmetric_unit_pays_head_eighty_percent():
    reg = _registry()
    reg.form(_pair(scope=1.0, symmetry=0.0), ["head", "junior"], step=0)
    reg.contribute("head", 1.0, 1)
    reg.contribute("junior", 1.0, 1)
    pay = reg.settle(1)
    assert pay["head"] == pytest.approx(1.6)
    assert pay["junior"] == pytest.approx(0.4)


def test_reserve_target_is_held_back_and_aid_draws_on_it():
    reg = CooperationRegistry(
        economics=UnitEconomics(synergy=0.0, payout_rate=1.0, aid_threshold=2.0, aid_amount=1.0),
        unit_goals=UnitGoals(reserve_target=1.0),
        rng=np.random.default_rng(0),
    )
    unit = reg.form(_pair(scope=1.0), ["a", "b"], step=0)
    reg.contribute("a", 5.0, 1)
    reg.contribute("b", 0.0, 1)
    pay = reg.settle(1, holdings={"a": 10.0, "b": 0.5})
    # b is needy: aid = min(1.0, 2.0 - 0.5, pool) = 1.0; then pool 4.0 - reserve 2.0 = 2.0 paid out.
    assert unit.totals["aid"] == pytest.approx(1.0)
    assert pay["a"] == pytest.approx(1.0)
    assert pay["b"] == pytest.approx(2.0)
    assert unit.pool == pytest.approx(2.0)


def test_lone_participant_creates_no_surplus():
    assert UnitEconomics(synergy=1.0).surplus(10.0, 1) == 0.0
    assert UnitEconomics(synergy=1.0).surplus(10.0, 4) == pytest.approx(7.5)


# ---------------------------------------------------------------- enforcement
def test_unrecognised_exit_cost_is_never_collected():
    reg = _registry()
    unit = reg.form(_pair(exit_cost=1.0, recognition=0.0), ["a", "b"], step=0)
    result = reg.leave(unit.unit_id, "a", step=1, holdings=10.0)
    assert result.left and not result.enforced and result.paid == 0.0
    assert result.dissolved and not unit.is_active


def test_recognised_exit_cost_is_forfeited_to_remaining_members():
    reg = _registry()
    unit = reg.form(_pair(exit_cost=0.5, recognition=1.0), ["a", "b"], step=0)
    result = reg.leave(unit.unit_id, "a", step=1, holdings=10.0)
    assert result.enforced and result.paid == pytest.approx(5.0)
    assert result.payouts == {"b": pytest.approx(5.0)}
    assert not reg.is_member("a") and not reg.is_member("b")


def test_offers_are_blocked_in_proportion_to_recognition():
    reg = _registry()
    reg.form(_pair(recognition=1.0), ["a", "b"], step=0)
    reg.form(_pair(kind="loose", recognition=0.0), ["c", "d"], step=0)
    assert reg.offer_blocked("a", "pair", step=1)
    assert not reg.offer_blocked("c", "loose", step=1)
    assert not reg.offer_blocked("zed", "pair", step=1)
    assert reg.event_counts["offer_blocked"] == 1


def test_parent_vetoes_only_exits_that_would_dissolve_the_child():
    reg = _registry()
    couple = reg.form(_pair(), ["a", "b"], step=0)
    family = reg.form(
        CooperationSettings(kind="family", min_members=1, max_members=None, recognition=1.0, entry=Entry.BORN),
        [],
        step=0,
        children=[couple.unit_id],
    )
    assert reg.depth(couple) == 1 and reg.root_of(couple) is family
    assert reg.veto_probability(couple.unit_id, "a") == 1.0
    result = reg.leave(couple.unit_id, "a", step=1)
    assert result.vetoed and not result.left and reg.is_member("a", couple.unit_id)
    trio = reg.form(CooperationSettings(kind="trio", max_members=3), ["x", "y", "z"], step=0, parent_id=family.unit_id)
    assert reg.veto_probability(trio.unit_id, "x") == 0.0


def test_attach_rejects_cycles_and_second_parents():
    reg = _registry()
    a = reg.form(CooperationSettings(kind="k1", min_members=1, max_members=None), ["a"], step=0)
    b = reg.form(CooperationSettings(kind="k2", min_members=1, max_members=None), ["b"], step=0)
    reg.attach(a.unit_id, b.unit_id, step=1)
    with pytest.raises(ValueError):
        reg.attach(b.unit_id, a.unit_id, step=1)
    c = reg.form(CooperationSettings(kind="k3", min_members=1, max_members=None), ["c"], step=0)
    with pytest.raises(ValueError):
        reg.attach(a.unit_id, c.unit_id, step=1)


def test_nesting_sends_dues_up_and_payouts_down():
    reg = _registry(payout_rate=1.0)
    couple = reg.form(_pair(scope=1.0), ["a", "b"], step=0)
    family = reg.form(
        CooperationSettings(kind="family", min_members=1, max_members=None, scope=0.5, entry=Entry.BORN),
        ["kid"],
        step=0,
        children=[couple.unit_id],
    )
    reg.contribute("a", 2.0, 1)
    reg.contribute("b", 2.0, 1)
    reg.contribute("kid", 0.0, 1)
    pay = reg.settle(1)
    # Couple inflow 4 -> 2 up to the family; family pays 2 split couple/kid (1 each);
    # couple then pays 2 + 1 = 3 split a/b.
    assert couple.totals["upward"] == pytest.approx(2.0)
    assert family.totals["inflow"] == pytest.approx(2.0)
    assert couple.totals["downward"] == pytest.approx(1.0)
    assert pay["kid"] == pytest.approx(1.0)
    assert pay["a"] == pytest.approx(1.5) and pay["b"] == pytest.approx(1.5)


def test_birth_joins_the_household_family_not_the_parents_birth_family():
    reg = _registry()
    family_settings = CooperationSettings(kind="family", min_members=1, max_members=None, entry=Entry.BORN)
    birth_family = reg.form(family_settings, ["parent_sibling"], step=0)
    assert reg.join(birth_family.unit_id, "a", step=0)
    couple = reg.form(_pair(), ["a", "b"], step=1)
    # Before a household exists, a child joins the parent's own BORN unit.
    assert reg.register_birth("a", "c1", step=2) == [birth_family.unit_id]
    household = reg.form(
        CooperationSettings(kind="household", min_members=1, max_members=None, entry=Entry.BORN),
        [],
        step=3,
        children=[couple.unit_id],
    )
    assert reg.register_birth("b", "c2", step=4) == [household.unit_id]
    assert reg.register_birth("a", "c3", step=4) == [household.unit_id]
    assert reg.register_birth("loner", "c4", step=4) == []


def test_removing_a_dead_agent_dissolves_without_exit_cost_or_payout_to_the_dead():
    reg = _registry()
    unit = reg.form(_pair(exit_cost=1.0, recognition=1.0, scope=1.0), ["a", "b"], step=0)
    unit.pool = 4.0
    payouts = reg.remove_agent("a", step=1)
    assert payouts == {"b": pytest.approx(4.0)}
    assert not unit.is_active and unit.dissolution_reason == "member_death"
    assert not reg.is_member("a") and not reg.is_member("b")


def test_record_exit_desire_counts_trapped_steps():
    reg = _registry()
    unit = reg.form(_pair(), ["a", "b"], step=0)
    reg.record_exit_desire(unit.unit_id, "a", blocked_by_price=True)
    reg.record_exit_desire(unit.unit_id, "a", blocked_by_price=False)
    assert unit.members["a"].exit_desire_steps == 2
    assert unit.members["a"].trapped_steps == 1


def test_same_seed_gives_same_enforcement_draws():
    def run(seed):
        reg = _registry(seed)
        draws = []
        for i in range(30):
            unit = reg.form(_pair(exit_cost=0.5, recognition=0.5), [f"a{i}", f"b{i}"], step=i)
            draws.append(reg.leave(unit.unit_id, f"a{i}", step=i, holdings=1.0).enforced)
        return draws

    assert run(7) == run(7)
    assert any(run(7)) and not all(run(7))


def test_summary_and_presets_work_together():
    reg = _registry()
    reg.form(preset("marriage"), ["a", "b"], step=0)
    reg.form(preset("friendship"), ["a", "c"], step=0)
    summary = reg.summary(current_step=10)
    assert summary["units_formed"] == 2 and summary["agents_in_units"] == 3
    assert reg.membership_counts() == {"marriage": 2, "friendship": 2}
