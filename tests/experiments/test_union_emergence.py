"""Invariants for the standalone union-emergence arena (PR 1014)."""

from __future__ import annotations

import pytest

from experiments.union_emergence.union_intrinsic_evolution import (
    Agent,
    RunSpec,
    UnionArena,
    WorldParams,
    v2_worlds,
)


@pytest.mark.unit
def test_bond_is_symmetric_and_charges_bonding_cost():
    world = WorldParams(name="unit", bonding_cost=0.8)
    arena = UnionArena(RunSpec(world=world, arm="optional_union", seed=0, steps=1))
    first = Agent(
        agent_id=9001,
        lineage=9001,
        x=1.0,
        y=1.0,
        energy=5.0,
        pair_commitment=0.8,
        fidelity=0.5,
        specialize=0.7,
        share_weight=0.1,
        attack_weight=0.1,
    )
    second = Agent(
        agent_id=9002,
        lineage=9002,
        x=1.1,
        y=1.0,
        energy=5.0,
        pair_commitment=0.7,
        fidelity=0.4,
        specialize=0.3,
        share_weight=0.1,
        attack_weight=0.1,
    )
    arena.agents[first.agent_id] = first
    arena.agents[second.agent_id] = second

    assert arena.form_bond(first, second) is True
    assert first.partner_id == second.agent_id
    assert second.partner_id == first.agent_id
    assert first.energy == pytest.approx(4.6)
    assert second.energy == pytest.approx(4.6)
    assert {first.role, second.role} == {"gather", "guard"}
    assert first.role != second.role


@pytest.mark.unit
def test_bond_refuses_when_either_cannot_cover_cost():
    world = WorldParams(name="unit", bonding_cost=0.8)
    arena = UnionArena(RunSpec(world=world, arm="optional_union", seed=1, steps=1))
    rich = Agent(1, 1, 0.0, 0.0, 5.0, 0.9, 0.5, 0.5, 0.1, 0.1)
    poor = Agent(2, 2, 0.1, 0.0, 0.2, 0.9, 0.5, 0.5, 0.1, 0.1)
    assert arena.form_bond(rich, poor) is False
    assert rich.partner_id is None
    assert poor.partner_id is None
    assert rich.energy == 5.0
    assert poor.energy == 0.2


@pytest.mark.unit
def test_leave_clears_both_sides_and_zeros_strength():
    world = WorldParams(name="unit")
    arena = UnionArena(RunSpec(world=world, arm="optional_union", seed=2, steps=1))
    left = Agent(10, 10, 0.0, 0.0, 4.0, 0.6, 0.5, 0.5, 0.1, 0.1, partner_id=11, pair_age=9, bond_strength=0.7)
    right = Agent(11, 11, 0.2, 0.0, 4.0, 0.6, 0.5, 0.5, 0.1, 0.1, partner_id=10, pair_age=9, bond_strength=0.7)
    arena.agents[10] = left
    arena.agents[11] = right

    arena.dissolve(left, reason="leave")
    assert left.partner_id is None
    assert right.partner_id is None
    assert left.bond_strength == 0.0
    assert right.bond_strength == 0.0
    assert left.role == "none"
    assert right.role == "none"
    assert arena.closed_pairs[-1].duration == 9


@pytest.mark.unit
def test_synergy_is_one_during_courtship_and_when_partner_dead():
    world = WorldParams(name="unit", courtship_steps=12, binary_lock=False)
    arena = UnionArena(RunSpec(world=world, arm="optional_union", seed=3, steps=1))
    living = Agent(
        20,
        20,
        5.0,
        5.0,
        4.0,
        0.8,
        0.5,
        1.0,
        0.1,
        0.1,
        partner_id=21,
        pair_age=3,
        bond_strength=0.9,
    )
    partner = Agent(
        21,
        21,
        5.1,
        5.0,
        4.0,
        0.8,
        0.5,
        1.0,
        0.1,
        0.1,
        partner_id=20,
        pair_age=3,
        bond_strength=0.9,
    )
    arena.agents[20] = living
    arena.agents[21] = partner

    assert arena.synergy_multiplier(living) == 1.0

    living.pair_age = 12
    partner.pair_age = 12
    assert arena.synergy_multiplier(living) > 1.0

    partner.alive = False
    assert arena.synergy_multiplier(living) == 1.0


@pytest.mark.unit
def test_solo_arm_never_sets_partner_id():
    world = WorldParams(name="unit", initial_pop=20, max_pop=40)
    arena = UnionArena(RunSpec(world=world, arm="solo_only", seed=4, steps=40))
    arena.run()
    assert all(agent.partner_id is None for agent in arena.agents.values())
    assert arena.telemetry.bond_events == 0


@pytest.mark.unit
def test_mature_pair_reproduces_once_per_step():
    world = WorldParams(
        name="unit",
        initial_pop=0,
        max_pop=10,
        courtship_steps=12,
        offspring_cost=1.7,
        reproduce_threshold=3.6,
    )
    arena = UnionArena(RunSpec(world=world, arm="optional_union", seed=5, steps=1))
    first = Agent(1, 1, 0.0, 0.0, 5.0, 0.6, 0.5, 0.5, 0.1, 0.1, partner_id=2, pair_age=12)
    second = Agent(2, 2, 0.1, 0.0, 5.0, 0.6, 0.5, 0.5, 0.1, 0.1, partner_id=1, pair_age=12)
    arena.agents[1] = first
    arena.agents[2] = second
    arena.next_id = 3

    reproducers = list(arena._alive())
    for agent in reproducers:
        arena._reproduce(agent, arena._alive())

    children = [agent for agent in arena.agents.values() if agent.agent_id not in (1, 2)]
    assert len(children) == 1
    assert arena.telemetry.reproduce_events == 1
    assert first.energy == pytest.approx(4.15)
    assert second.energy == pytest.approx(4.15)
    assert children[0].lineage == 2


@pytest.mark.unit
def test_failed_bond_roll_is_not_retried_by_nearest_neighbor():
    world = WorldParams(name="unit", initial_pop=0, max_pop=10, bonding_cost=0.8)
    arena = UnionArena(RunSpec(world=world, arm="optional_union", seed=6, steps=1))
    first = Agent(1, 1, 1.0, 1.0, 5.0, 0.5, 0.5, 0.5, 0.1, 0.1)
    second = Agent(2, 2, 1.1, 1.0, 5.0, 0.5, 0.5, 0.5, 0.1, 0.1)
    arena.agents[1] = first
    arena.agents[2] = second
    rolls = [0.9, 0.05]
    arena.rng.shuffle = lambda seq: seq.sort(key=lambda agent: agent.agent_id)
    arena.rng.random = lambda: rolls.pop(0)

    arena._maybe_bond(arena._alive())

    assert first.partner_id is None
    assert second.partner_id is None
    assert arena.telemetry.bond_events == 0
    assert rolls == [0.05]


@pytest.mark.unit
def test_v2_worlds_match_pre_register_knobs():
    worlds = {world.name: world for world in v2_worlds()}
    assert worlds["baseline"].bonding_cost == 0.8
    assert worlds["baseline"].courtship_steps == 12
    assert worlds["baseline"].exit_tax == 1.1
    assert worlds["baseline"].social_range == 3.2
    assert worlds["cheap_exit_cheap_bond"].bonding_cost == 0.2
    assert worlds["cheap_exit_cheap_bond"].exit_tax == 0.2
    assert worlds["no_courtship"].courtship_steps == 0
    assert worlds["wide_neighborhood"].social_range == 8.0
