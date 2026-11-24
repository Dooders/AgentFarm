"""Settings, presets and goal valuation for :mod:`farm.core.cooperation`."""

import math

import pytest

from farm.core.cooperation import (
    MOTIVE_SETTING_PREFERENCES,
    MOTIVES,
    PRESETS,
    CooperationSettings,
    Entry,
    MemberGoals,
    MembershipAppraisal,
    Motive,
    UnitGoals,
    motive_fit,
    preset,
    rank_weights,
    value_of_membership,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "field_name, value",
    [("scope", -0.1), ("scope", 1.1), ("exit_cost", 2.0), ("recognition", -1.0), ("symmetry", 1.5)],
)
def test_unit_interval_settings_are_validated(field_name, value):
    with pytest.raises(ValueError):
        CooperationSettings(**{field_name: value})


def test_member_bounds_are_validated():
    with pytest.raises(ValueError):
        CooperationSettings(min_members=0)
    with pytest.raises(ValueError):
        CooperationSettings(min_members=3, max_members=2)
    assert CooperationSettings(min_members=2, max_members=None).has_capacity(10_000)


def test_entry_strings_are_coerced_and_round_trip():
    settings = CooperationSettings(kind="x", entry="born")
    assert settings.entry is Entry.BORN
    assert CooperationSettings.from_dict(settings.to_dict()) == settings


def test_exclusivity_sets_membership_limit():
    assert CooperationSettings(exclusive=True).max_memberships == 1
    assert CooperationSettings(exclusive=False, nonexclusive_limit=4).max_memberships == 4


def test_expected_exit_payment_is_enforcement_weighted():
    settings = CooperationSettings(exit_cost=0.5, recognition=0.4)
    assert settings.expected_exit_payment(10.0) == pytest.approx(2.0)
    assert settings.expected_exit_payment(-3.0) == 0.0


@pytest.mark.parametrize("n", [1, 2, 3, 7])
@pytest.mark.parametrize("symmetry", [0.0, 0.3, 1.0])
def test_rank_weights_sum_to_one_and_decay_with_rank(n, symmetry):
    weights = rank_weights(n, symmetry)
    assert len(weights) == n
    assert sum(weights) == pytest.approx(1.0)
    assert all(a >= b - 1e-12 for a, b in zip(weights, weights[1:]))
    if symmetry == 1.0:
        assert all(w == pytest.approx(1.0 / n) for w in weights)


def test_fully_asymmetric_pair_splits_eighty_twenty():
    assert rank_weights(2, 0.0) == pytest.approx((0.8, 0.2))


def test_presets_cover_the_framing_table():
    assert set(PRESETS) == {"marriage", "cohabitation", "friendship", "business_partnership", "family", "tribe"}
    marriage = preset("marriage")
    assert (marriage.max_members, marriage.exclusive, marriage.entry) == (2, True, Entry.CHOSEN)
    # Cohabitation differs from marriage mainly on recognition.
    assert preset("cohabitation").recognition < marriage.recognition
    # Family is entered by birth, not choice.
    assert preset("family").entry is Entry.BORN
    assert preset("marriage", scope=0.2).scope == 0.2
    with pytest.raises(KeyError):
        preset("guild")


def test_member_goals_normalise_and_report_dominant_motive():
    goals = MemberGoals.mixture({Motive.ECONOMIC: 3.0, Motive.EMOTIONAL: 1.0})
    assert sum(goals.motives.values()) == pytest.approx(1.0)
    assert goals.weight(Motive.ECONOMIC) == pytest.approx(0.75)
    assert goals.dominant is Motive.ECONOMIC
    assert MemberGoals.single(Motive.HEDONIST).dominant is Motive.HEDONIST
    with pytest.raises(ValueError):
        MemberGoals({Motive.ECONOMIC: -1.0})
    with pytest.raises(ValueError):
        MemberGoals({Motive.ECONOMIC: 0.0})


def test_default_goals_are_uniform_over_motives():
    goals = MemberGoals()
    for motive in MOTIVES:
        assert goals.weight(motive) == pytest.approx(1.0 / len(MOTIVES))


@pytest.mark.parametrize("motive", list(Motive))
def test_motive_fit_follows_preference_directions(motive):
    prefs = MOTIVE_SETTING_PREFERENCES[motive]
    high = {"scope": 1.0, "exit_cost": 1.0, "recognition": 1.0, "symmetry": 1.0}
    low = {"scope": 0.0, "exit_cost": 0.0, "recognition": 0.0, "symmetry": 0.0}
    favoured = CooperationSettings(
        **{k: (high[k] if prefs.get(k, 0) >= 0 else low[k]) for k in high},
        exclusive=prefs.get("exclusivity", 1) >= 0,
    )
    disfavoured = CooperationSettings(
        **{k: (low[k] if prefs.get(k, 0) >= 0 else high[k]) for k in high},
        exclusive=prefs.get("exclusivity", 1) < 0,
    )
    assert motive_fit(motive, favoured) == pytest.approx(1.0)
    assert motive_fit(motive, disfavoured) == pytest.approx(-1.0)


def test_hedonists_prefer_low_exit_cost_and_religious_prefer_high():
    cheap_exit = CooperationSettings(exit_cost=0.0, recognition=0.5)
    vows = CooperationSettings(exit_cost=1.0, recognition=0.5)
    hedonist = MemberGoals.single(Motive.HEDONIST)
    religious = MemberGoals.single(Motive.RELIGIOUS)
    assert hedonist.settings_fit(cheap_exit) > hedonist.settings_fit(vows)
    assert religious.settings_fit(vows) > religious.settings_fit(cheap_exit)


def test_value_of_membership_is_monotone_in_flows_and_affinity():
    goals = MemberGoals.mixture({Motive.ECONOMIC: 1.0, Motive.EMOTIONAL: 1.0})
    settings = preset("marriage")
    base = value_of_membership(goals, settings, MembershipAppraisal())
    richer = value_of_membership(goals, settings, MembershipAppraisal(net_transfer=0.5))
    safer = value_of_membership(goals, settings, MembershipAppraisal(insurance=0.5))
    fonder = value_of_membership(goals, settings, MembershipAppraisal(compatibility=0.8))
    assert richer > base and safer > base and fonder > base
    # A pure hedonist does not value affinity at all.
    hedonist = MemberGoals.single(Motive.HEDONIST)
    assert value_of_membership(hedonist, settings, MembershipAppraisal(compatibility=1.0)) == pytest.approx(
        value_of_membership(hedonist, settings, MembershipAppraisal())
    )


def test_unit_goals_utility_rewards_persistence_reserve_and_size():
    goals = UnitGoals(persistence=1.0, reserve=0.5, growth=0.25, reserve_target=2.0)
    assert goals.utility(pool=0.0, n_participants=2, active=False) == 0.0
    empty = goals.utility(pool=0.0, n_participants=2)
    full = goals.utility(pool=4.0, n_participants=2)
    assert full - empty == pytest.approx(0.5)
    assert goals.utility(pool=100.0, n_participants=2) == pytest.approx(full)
    # Same reserve coverage, larger unit: the growth term raises utility.
    assert goals.utility(pool=8.0, n_participants=4) > goals.utility(pool=4.0, n_participants=2)
    assert math.isfinite(goals.utility(pool=1.0, n_participants=1))
    with pytest.raises(ValueError):
        UnitGoals(reserve=-1.0)
