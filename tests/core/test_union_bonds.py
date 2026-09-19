"""Port invariants for exclusive pair-bonds (Layer C / PR 1014)."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from farm.core.hyperparameter_chromosome import default_hyperparameter_chromosome
from farm.core.union_bonds import (
    UNION_GENE_NAMES,
    UnionPolicy,
    dissolve,
    ensure_bond_state,
    equalize_with_partner,
    form_bond,
    freeze_learning_genes,
    gather_synergy_multiplier,
    gene_value,
    get_union_policy,
    is_mature_bond,
    mature_bond_partner,
    on_agent_terminate,
    partner_of,
    policy_for_arm,
    snapshot_union_metrics,
    tick_union_bonds,
    try_bond_action,
    try_leave_action,
    union_enabled,
)


def _agent(agent_id: str, energy: float = 5.0, **kwargs):
    chromosome = default_hyperparameter_chromosome()
    agent = SimpleNamespace(
        agent_id=agent_id,
        alive=True,
        resource_level=energy,
        position=(0.0, 0.0),
        partner_id=None,
        pair_age=0,
        bond_strength=0.0,
        role="none",
        hyperparameter_chromosome=chromosome,
        environment=None,
    )
    for key, value in kwargs.items():
        setattr(agent, key, value)
    return agent


def _paired_env(first, second, policy: UnionPolicy):
    env = SimpleNamespace(
        union_policy=policy,
        alive_agent_objects=[first, second],
        _agent_objects={first.agent_id: first, second.agent_id: second},
    )
    first.environment = env
    second.environment = env
    return env


@pytest.mark.unit
def test_union_genes_are_on_default_chromosome():
    chromosome = default_hyperparameter_chromosome()
    for name in UNION_GENE_NAMES:
        gene = chromosome.get_gene(name)
        assert gene is not None
        assert gene.min_value == 0.0
        assert gene.max_value == 1.0


@pytest.mark.unit
def test_bond_is_symmetric_and_charges_bonding_cost():
    policy = UnionPolicy(enabled=True, pairing_mode="optional", bonding_cost=0.8)
    first = _agent("a", 5.0)
    second = _agent("b", 5.0)
    _paired_env(first, second, policy)

    assert form_bond(first, second, policy) is True
    assert first.partner_id == "b"
    assert second.partner_id == "a"
    assert first.resource_level == pytest.approx(4.6)
    assert second.resource_level == pytest.approx(4.6)
    assert {first.role, second.role} == {"gather", "guard"}


@pytest.mark.unit
def test_bond_refuses_when_either_cannot_cover_cost():
    policy = UnionPolicy(enabled=True, pairing_mode="optional", bonding_cost=0.8)
    rich = _agent("rich", 5.0)
    poor = _agent("poor", 0.2)
    _paired_env(rich, poor, policy)
    assert form_bond(rich, poor, policy) is False
    assert rich.partner_id is None
    assert poor.partner_id is None
    assert rich.resource_level == 5.0


@pytest.mark.unit
def test_leave_clears_both_sides_and_zeros_strength():
    policy = UnionPolicy(enabled=True, pairing_mode="optional")
    left = _agent("left", 4.0, partner_id="right", pair_age=9, bond_strength=0.7)
    right = _agent("right", 4.0, partner_id="left", pair_age=9, bond_strength=0.7)
    _paired_env(left, right, policy)

    duration = dissolve(left, reason="leave", policy=policy)
    assert duration == 9
    assert left.partner_id is None
    assert right.partner_id is None
    assert left.bond_strength == 0.0
    assert right.bond_strength == 0.0
    assert left.role == "none"


@pytest.mark.unit
def test_synergy_is_one_during_courtship_and_when_partner_dead():
    policy = UnionPolicy(enabled=True, pairing_mode="optional", courtship_steps=12)
    living = _agent("a", 4.0, partner_id="b", pair_age=3, bond_strength=0.9, position=(0.0, 0.0))
    partner = _agent("b", 4.0, partner_id="a", pair_age=3, bond_strength=0.9, position=(0.2, 0.0))
    _paired_env(living, partner, policy)

    assert gather_synergy_multiplier(living) == 1.0
    living.pair_age = 12
    partner.pair_age = 12
    assert gather_synergy_multiplier(living) > 1.0
    partner.alive = False
    assert gather_synergy_multiplier(living) == 1.0


@pytest.mark.unit
def test_synergy_is_one_when_partners_are_not_colocated():
    policy = UnionPolicy(enabled=True, pairing_mode="optional", courtship_steps=0)
    first = _agent("a", 4.0, partner_id="b", pair_age=20, bond_strength=0.9, position=(0.0, 0.0))
    second = _agent("b", 4.0, partner_id="a", pair_age=20, bond_strength=0.9, position=(8.0, 0.0))
    _paired_env(first, second, policy)
    assert gather_synergy_multiplier(first) == 1.0


@pytest.mark.unit
def test_tick_pulls_partners_together():
    policy = UnionPolicy(enabled=True, pairing_mode="optional", courtship_steps=12, pair_pull=0.95)
    first = _agent("a", 4.0, partner_id="b", pair_age=2, bond_strength=0.4, position=(0.0, 0.0))
    second = _agent("b", 4.0, partner_id="a", pair_age=2, bond_strength=0.4, position=(6.0, 0.0))
    env = _paired_env(first, second, policy)
    env.width = 24.0
    env.height = 24.0
    tick_union_bonds(env)
    assert first.position[0] > 0.0
    assert second.position[0] < 6.0
    assert abs(second.position[0] - first.position[0]) < 6.0


@pytest.mark.unit
def test_solo_arm_never_sets_partner_id():
    policy = policy_for_arm("solo_only")
    first = _agent("a", 5.0)
    second = _agent("b", 5.0)
    _paired_env(first, second, policy)
    assert policy.pairing_enabled is False
    assert form_bond(first, second, policy) is False
    assert first.partner_id is None
    assert second.partner_id is None


@pytest.mark.unit
def test_freeze_learning_genes_keeps_union_loci_evolvable():
    frozen = freeze_learning_genes(default_hyperparameter_chromosome())
    assert frozen.get_gene("learning_rate").evolvable is False
    assert frozen.get_gene("pair_commitment").evolvable is True
    assert frozen.get_gene("share_weight").evolvable is True
    assert frozen.get_gene("reward_share_bonus").evolvable is True


@pytest.mark.unit
def test_policy_for_arm_sets_pairing_and_share_rules():
    solo = policy_for_arm("solo_only")
    promiscuous = policy_for_arm("promiscuous")
    optional = policy_for_arm("optional_union")
    forced = policy_for_arm("forced_union")
    assert solo.pairing_mode == "off" and solo.residual_share is True
    assert promiscuous.pairing_mode == "off" and promiscuous.suppress_promiscuous_share is False
    assert optional.pairing_mode == "optional" and optional.pairing_enabled is True
    assert forced.pairing_mode == "forced" and forced.forced_pair_rate == 0.85
    with pytest.raises(ValueError, match="unknown union arm"):
        policy_for_arm("not_an_arm")
    with pytest.raises(ValueError, match="pairing_mode"):
        UnionPolicy(pairing_mode="monogamy")


@pytest.mark.unit
def test_policy_and_gene_lookups_fall_back_cleanly():
    policy = UnionPolicy(enabled=True, pairing_mode="optional")
    env = SimpleNamespace(union_policy=policy, config=SimpleNamespace(union_enabled=False))
    agent = _agent("a")
    agent.environment = env
    agent.hyperparameter_chromosome = None
    assert get_union_policy(env) is policy
    assert get_union_policy(agent) is policy
    assert get_union_policy(None) is None
    assert union_enabled(env) is True
    assert union_enabled(SimpleNamespace(config=SimpleNamespace(union_enabled=True))) is True
    assert union_enabled(SimpleNamespace(environment=SimpleNamespace(config=SimpleNamespace(union_enabled=True)))) is True
    assert gene_value(agent, "pair_commitment", 0.42) == 0.42
    bare = SimpleNamespace()
    ensure_bond_state(bare)
    assert bare.partner_id is None
    assert bare.pair_age == 0
    assert bare.bond_strength == 0.0
    assert bare.role == "none"


@pytest.mark.unit
def test_partner_lookup_uses_get_agent_then_alive_list():
    policy = UnionPolicy(enabled=True, pairing_mode="optional")
    left = _agent("left", partner_id="right", pair_age=15, bond_strength=0.6)
    right = _agent("right", partner_id="left", pair_age=15, bond_strength=0.6)
    env = SimpleNamespace(
        union_policy=policy,
        alive_agent_objects=[left, right],
        get_agent=lambda agent_id: right if agent_id == "right" else left,
    )
    left.environment = env
    right.environment = env
    assert partner_of(left) is right
    assert is_mature_bond(left) is True
    assert mature_bond_partner(left) is right

    listed = _agent("listed", partner_id="peer")
    peer = _agent("peer", partner_id="listed")
    listed.environment = SimpleNamespace(union_policy=policy, alive_agent_objects=[listed, peer])
    assert partner_of(listed) is peer
    assert partner_of(_agent("orphan", partner_id="gone")) is None
    assert is_mature_bond(_agent("solo")) is False
    assert mature_bond_partner(_agent("solo")) is None


@pytest.mark.unit
def test_form_bond_refuses_same_dead_or_already_locked_agents():
    policy = UnionPolicy(enabled=True, pairing_mode="optional", bonding_cost=0.8)
    first = _agent("a", 5.0)
    twin = first
    second = _agent("b", 5.0, alive=False)
    third = _agent("c", 5.0, partner_id="someone")
    env = _paired_env(first, second, policy)
    env.alive_agent_objects = [first, second, third]
    env._agent_objects["c"] = third
    third.environment = env
    assert form_bond(first, twin, policy) is False
    assert form_bond(first, second, policy) is False
    second.alive = True
    assert form_bond(first, third, policy) is False
    specialist = _agent("spec", 5.0)
    specialist.hyperparameter_chromosome = specialist.hyperparameter_chromosome.with_overrides({"specialize": 0.1})
    other = _agent("oth", 5.0)
    other.hyperparameter_chromosome = other.hyperparameter_chromosome.with_overrides({"specialize": 0.9})
    _paired_env(specialist, other, policy)
    assert form_bond(specialist, other, policy) is True
    assert specialist.role == "guard"
    assert other.role == "gather"


@pytest.mark.unit
def test_widow_pays_grief_and_leave_action_clears_the_pair():
    policy = UnionPolicy(enabled=True, pairing_mode="optional", grief_scale=0.5, exit_tax=2.0)
    left = _agent("left", 4.0, partner_id="right", pair_age=8, bond_strength=0.8)
    right = _agent("right", 4.0, partner_id="left", pair_age=8, bond_strength=0.8)
    right.hyperparameter_chromosome = right.hyperparameter_chromosome.with_overrides({"fidelity": 1.0})
    _paired_env(left, right, policy)

    on_agent_terminate(left)
    assert right.partner_id is None
    assert right.resource_level < 4.0

    solo = _agent("solo", 3.0, partner_id="ghost")
    on_agent_terminate(solo)
    assert solo.partner_id is None

    again_left = _agent("L", 5.0, partner_id="R", pair_age=6, bond_strength=0.5)
    again_right = _agent("R", 5.0, partner_id="L", pair_age=6, bond_strength=0.5)
    _paired_env(again_left, again_right, policy)
    result = try_leave_action(again_left)
    assert result["success"] is True
    assert again_left.partner_id is None
    assert again_right.partner_id is None
    assert try_leave_action(_agent("unpaired"))["success"] is False


@pytest.mark.unit
def test_try_bond_action_and_equalize_and_snapshot():
    policy = UnionPolicy(
        enabled=True,
        pairing_mode="forced",
        bonding_cost=0.4,
        social_range=3.0,
        equalize_rate=0.5,
        courtship_steps=0,
    )
    first = _agent("a", 6.0, position=(0.0, 0.0))
    second = _agent("b", 2.0, position=(1.0, 0.0))
    env = _paired_env(first, second, policy)
    env.union_rng = random.Random(0)

    assert try_bond_action(first)["success"] is True
    assert first.partner_id == "b"
    assert try_bond_action(first)["success"] is False
    shared = equalize_with_partner(first)
    assert shared is not None
    assert first.resource_level != 6.0 - 0.2
    snap = snapshot_union_metrics(env)
    assert snap["paired_frac"] == 1.0
    assert snap["synergy_index"] is None or snap["synergy_index"] > 0.0
    assert snap["n_alive"] == 2

    lonely = _agent("solo", 3.0)
    lonely_env = SimpleNamespace(union_policy=policy, alive_agent_objects=[lonely], _agent_objects={})
    lonely.environment = lonely_env
    assert try_bond_action(lonely)["success"] is False
    assert equalize_with_partner(lonely) is None
    empty = snapshot_union_metrics(SimpleNamespace(alive_agent_objects=[]))
    assert empty["paired_frac"] == 0.0


@pytest.mark.unit
def test_tick_forms_pairs_exits_on_accident_and_marks_dirty():
    policy = UnionPolicy(
        enabled=True,
        pairing_mode="optional",
        bonding_cost=0.2,
        social_range=4.0,
        accident_divorce=1.0,
        implicit_leave=True,
        courtship_steps=0,
    )
    first = _agent("a", 8.0, position=(0.0, 0.0))
    second = _agent("b", 8.0, position=(0.4, 0.0))
    env = _paired_env(first, second, policy)
    env.width = 24.0
    env.height = 24.0

    class _ZeroRng:
        def random(self):
            return 0.0

        def shuffle(self, seq):
            return None

    env.union_rng = _ZeroRng()
    dirty = {"n": 0}

    def _mark():
        dirty["n"] += 1

    env.mark_positions_dirty = _mark
    tick_union_bonds(env)
    assert first.partner_id == "b"
    assert policy.bond_events >= 1
    tick_union_bonds(env)
    assert policy.leave_events >= 1
    assert dirty["n"] >= 1

    wrap_left = _agent("w1", 5.0, partner_id="w2", pair_age=2, bond_strength=0.5, position=(0.2, 0.0))
    wrap_right = _agent("w2", 5.0, partner_id="w1", pair_age=2, bond_strength=0.5, position=(23.8, 0.0))
    wrap_policy = UnionPolicy(enabled=True, pairing_mode="optional", accident_divorce=0.0, implicit_leave=False)
    wrap_env = _paired_env(wrap_left, wrap_right, wrap_policy)
    wrap_env.width = 24.0
    wrap_env.height = 24.0
    tick_union_bonds(wrap_env)
    assert wrap_left.pair_age == 3
    assert wrap_left.bond_strength > 0.5

    dead_partner = _agent("alive", 4.0, partner_id="dead")
    ghost_env = SimpleNamespace(
        union_policy=UnionPolicy(enabled=True, pairing_mode="optional"),
        alive_agent_objects=[dead_partner],
        _agent_objects={},
    )
    dead_partner.environment = ghost_env
    tick_union_bonds(ghost_env)
    assert dead_partner.partner_id is None

    off = SimpleNamespace(union_policy=UnionPolicy(enabled=False), alive_agent_objects=[])
    tick_union_bonds(off)

    class _BadChromosome:
        def get_value(self, name):
            raise TypeError("unreadable")

    assert gene_value(SimpleNamespace(hyperparameter_chromosome=_BadChromosome()), "pair_commitment", 0.3) == 0.3
    assert gather_synergy_multiplier(_agent("off")) == 1.0
    disabled = _agent("d", 5.0)
    disabled.environment = SimpleNamespace(union_policy=UnionPolicy(enabled=True, pairing_mode="off"))
    assert try_bond_action(disabled)["success"] is False
    no_env = _agent("n", 5.0)
    no_env.union_policy = UnionPolicy(enabled=True, pairing_mode="optional")
    no_env.environment = None
    assert try_bond_action(no_env)["error"] == "No environment"

    reject_policy = UnionPolicy(enabled=True, pairing_mode="optional", social_range=5.0, bonding_cost=0.2)
    seeker = _agent("s", 8.0, position=(0.0, 0.0))
    target = _agent("t", 8.0, position=(0.5, 0.0))
    reject_env = _paired_env(seeker, target, reject_policy)

    class _OneRng:
        def random(self):
            return 1.0

    reject_env.union_rng = _OneRng()
    assert try_bond_action(seeker)["error"] == "Pairing probability rejected the bond"

    poor = _agent("p", 0.05, position=(0.0, 0.0))
    rich = _agent("r", 8.0, position=(0.4, 0.0))
    cost_policy = UnionPolicy(enabled=True, pairing_mode="forced", bonding_cost=2.0, social_range=5.0)
    cost_env = _paired_env(poor, rich, cost_policy)
    cost_env.union_rng = _ZeroRng()
    assert try_bond_action(poor)["error"] == "Bond refused (cost or lock)"

    widow = _agent("w1", 4.0, partner_id="w2", pair_age=4)
    spouse = _agent("w2", 4.0, partner_id="w1", pair_age=4)
    widow_env = SimpleNamespace(alive_agent_objects=[widow, spouse], _agent_objects={"w1": widow, "w2": spouse})
    widow.environment = widow_env
    spouse.environment = widow_env
    assert try_leave_action(widow)["success"] is True
    assert spouse.partner_id is None

    no_pair_policy = UnionPolicy(enabled=True, pairing_mode="optional", implicit_pairing=False)
    idle_a = _agent("i1", 5.0)
    idle_b = _agent("i2", 5.0)
    idle_env = _paired_env(idle_a, idle_b, no_pair_policy)
    tick_union_bonds(idle_env)
    assert idle_a.partner_id is None

    leave_policy = UnionPolicy(
        enabled=True,
        pairing_mode="optional",
        accident_divorce=0.0,
        implicit_leave=True,
        implicit_pairing=False,
    )
    leaver = _agent("lv", 1.0, partner_id="st", pair_age=4, bond_strength=0.1)
    stayer = _agent("st", 1.0, partner_id="lv", pair_age=4, bond_strength=0.1)
    leaver.hyperparameter_chromosome = leaver.hyperparameter_chromosome.with_overrides({"fidelity": 0.0})
    leave_env = _paired_env(leaver, stayer, leave_policy)
    leave_env.union_rng = _ZeroRng()
    tick_union_bonds(leave_env)
    assert leave_policy.leave_events >= 1

    same = _agent("same_a", 4.0, partner_id="same_b", pair_age=3, bond_strength=0.4, position=(2.0, 2.0))
    twin = _agent("same_b", 4.0, partner_id="same_a", pair_age=3, bond_strength=0.4, position=(2.0, 2.0))
    same_env = _paired_env(same, twin, UnionPolicy(enabled=True, pairing_mode="optional", implicit_leave=False, accident_divorce=0.0))
    same_env.width = 0.0
    same_env.height = 0.0
    tick_union_bonds(same_env)
    assert same.pair_age == 4

    mixed_solo = _agent("ms", 10.0)
    mixed_a = _agent("ma", 20.0, partner_id="mb", pair_age=5, bond_strength=0.5)
    mixed_b = _agent("mb", 12.0, partner_id="ma", pair_age=5, bond_strength=0.5)
    mixed_env = SimpleNamespace(
        union_policy=UnionPolicy(enabled=True, pairing_mode="optional"),
        alive_agent_objects=[mixed_solo, mixed_a, mixed_b],
        _agent_objects={"ms": mixed_solo, "ma": mixed_a, "mb": mixed_b},
    )
    mixed_solo.environment = mixed_env
    mixed_a.environment = mixed_env
    mixed_b.environment = mixed_env
    mixed_snap = snapshot_union_metrics(mixed_env)
    assert mixed_snap["synergy_index"] is not None
    assert mixed_snap["paired_frac"] == pytest.approx(2.0 / 3.0)
    assert is_mature_bond(_agent("ghost", partner_id="missing")) is False


@pytest.mark.unit
def test_wrapped_neighbors_pair_and_receive_synergy():
    policy = UnionPolicy(
        enabled=True,
        pairing_mode="forced",
        forced_pair_rate=1.0,
        bonding_cost=0.0,
        social_range=2.0,
        courtship_steps=0,
        accident_divorce=0.0,
        implicit_leave=False,
        colocation_radius=1.75,
    )
    first = _agent("a", 5.0, position=(0.2, 0.0))
    second = _agent("b", 5.0, position=(23.8, 0.0))
    env = _paired_env(first, second, policy)
    env.width = 24.0
    env.height = 24.0
    env.union_rng = random.Random(0)

    tick_union_bonds(env)
    assert first.partner_id == "b"
    assert second.partner_id == "a"
    first.pair_age = 1
    second.pair_age = 1
    first.bond_strength = 0.9
    second.bond_strength = 0.9
    assert gather_synergy_multiplier(first) > 1.0


@pytest.mark.unit
def test_pairing_rng_is_seeded_from_environment_seed():
    policy = UnionPolicy(
        enabled=True,
        pairing_mode="optional",
        bonding_cost=0.0,
        social_range=5.0,
        accident_divorce=0.0,
        implicit_leave=False,
    )

    def _bond_once(seed: int) -> bool:
        first = _agent("a", 5.0, position=(0.0, 0.0))
        second = _agent("b", 5.0, position=(0.4, 0.0))
        env = _paired_env(first, second, policy)
        env.seed_value = seed
        return try_bond_action(first)["success"]

    outcomes = [_bond_once(123) for _ in range(8)]
    assert all(outcome == outcomes[0] for outcome in outcomes)

    seeded = _paired_env(_agent("s1", 5.0, position=(0.0, 0.0)), _agent("s2", 5.0, position=(0.4, 0.0)), policy)
    seeded.seed_value = 7
    tick_union_bonds(seeded)
    assert isinstance(seeded.union_rng, random.Random)

    reject_policy = UnionPolicy(enabled=True, pairing_mode="optional", social_range=5.0, bonding_cost=0.0)
    seeker = _agent("s", 8.0, position=(0.0, 0.0))
    target = _agent("t", 8.0, position=(0.5, 0.0))
    reject_env = _paired_env(seeker, target, reject_policy)

    class _OneRng:
        def random(self):
            return 1.0

    reject_env.intrinsic_evolution_rng = _OneRng()
    assert try_bond_action(seeker)["error"] == "Pairing probability rejected the bond"
