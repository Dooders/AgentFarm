"""Port invariants for exclusive pair-bonds (Layer C / PR 1014)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from farm.core.hyperparameter_chromosome import default_hyperparameter_chromosome
from farm.core.union_bonds import (
    UNION_GENE_NAMES,
    UnionPolicy,
    dissolve,
    form_bond,
    freeze_learning_genes,
    gather_synergy_multiplier,
    policy_for_arm,
    tick_union_bonds,
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
