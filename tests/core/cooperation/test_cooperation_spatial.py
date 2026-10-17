"""Cooperation in the spatial world, tested without the engine (no torch).

The :class:`CooperationService` only needs agents with an id, a position, a
resource level, a chromosome and an action list, plus a spatial query; the
fakes below provide exactly that.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from farm.core.cooperation import (
    COOPERATION_ACTION_NAMES,
    CooperationActionType,
    CooperationConfig,
    CooperationService,
    cooperation_service_of,
)
from farm.core.cooperation.actions import bond_action, filter_cooperation_actions, leave_action
from farm.core.cooperation.genes import (
    COOPERATION_GENE_NAMES,
    MOTIVE_GENE_NAMES,
    blended_settings,
    cooperation_genes,
    extend_chromosome,
    gene_values,
    goals_from_values,
    has_cooperation_genes,
    motive_similarity,
    settings_from_values,
)
from farm.core.cooperation.goals import Motive
from farm.core.cooperation.service import SUMMARY_FILE, TIMESERIES_FILE, UNITS_FILE
from farm.core.hyperparameter_chromosome import (
    crossover_chromosomes,
    default_hyperparameter_chromosome,
    mutate_chromosome,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------- fakes
@dataclass
class FakeAction:
    name: str
    weight: float


class FakeSpatial:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def get_nearby(self, position, radius, kinds):
        out = [a for a in self.world.agents.values() if a.alive and math.dist(a.position, position) <= radius]
        return {"agents": out}


@dataclass
class FakeState:
    parent_ids: list = field(default_factory=list)


class FakeAgent:
    def __init__(self, world, agent_id, position, resources=10.0, values=None, parent=None):
        self.agent_id = agent_id
        self.position = tuple(position)
        self.resource_level = float(resources)
        self.alive = True
        self.environment = world
        self.spatial_service = world.spatial
        chromosome = default_hyperparameter_chromosome()
        if values is not None:
            chromosome = extend_chromosome(chromosome, values)
        self.hyperparameter_chromosome = chromosome
        self.state = FakeState([parent] if parent else [])
        base = [("move", 0.4), ("gather", 0.3), ("share", 0.2), ("pass", 0.1)]
        self.actions = [FakeAction(n, w) for n, w in base] + [FakeAction("bond", 0.1), FakeAction("leave", 0.05)]


class FakeWorld:
    def __init__(self, config: CooperationConfig, seed: int = 0) -> None:
        self.time = 0
        self.agents: dict[str, FakeAgent] = {}
        self.spatial = FakeSpatial(self)
        self.cooperation = CooperationService(config, environment=self, seed=seed)

    def add(self, agent_id, position, resources=10.0, values=None, parent=None) -> FakeAgent:
        agent = FakeAgent(self, agent_id, position, resources, values, parent)
        self.agents[agent_id] = agent
        self.cooperation.on_agent_added(agent)
        return agent

    def kill(self, agent_id) -> None:
        agent = self.agents.pop(agent_id)
        agent.alive = False
        self.cooperation.on_agent_removed(agent)

    def end_step(self) -> None:
        self.cooperation.settle()
        self.time += 1


PAIR_VALUES = {
    "coop_scope": 0.5,
    "coop_exit_cost": 0.5,
    "coop_recognition": 1.0,
    "coop_symmetry": 1.0,
    "coop_exclusivity": 1.0,
    "coop_max_members": 2.0,
}


def _config(**overrides) -> CooperationConfig:
    base = {
        "enabled": True,
        "randomize_founder_genes": False,
        "synergy": 0.0,
        "payout_rate": 1.0,
        "aid_threshold": 0.0,
        "aid_amount": 0.0,
        "reserve_target": 0.0,
        "bond_range": 5.0,
    }
    base.update(overrides)
    return CooperationConfig(**base)


def _bonded_pair(config=None, values=None, resources=10.0):
    world = FakeWorld(config or _config())
    v = dict(PAIR_VALUES, **(values or {}))
    a = world.add("a", (0, 0), resources, values=v)
    b = world.add("b", (1, 0), resources, values=v)
    first = world.cooperation.bond(a)
    second = world.cooperation.bond(b)
    assert not first["success"] and second["success"]
    return world, a, b


# -------------------------------------------------------------------- config
def test_config_is_off_by_default_and_round_trips():
    cfg = CooperationConfig()
    assert not cfg.enabled and not cfg.active
    assert CooperationConfig(enabled=True, mode="placebo").active is False
    custom = CooperationConfig(
        enabled=True,
        fixed_settings={"scope": 0.7, "exit_cost": 0.5, "recognition": 0.9},
        founder_gene_values={"coop_scope": 0.2},
    )
    again = CooperationConfig.from_dict(json.loads(json.dumps(custom.to_dict())))
    assert again == custom
    assert again.unit_settings().scope == 0.7 and again.unit_settings().kind == "bond"
    assert CooperationConfig.from_dict(None) == CooperationConfig()


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "sometimes"},
        {"terms": "auction"},
        {"bond_range": -1.0},
        {"offer_window": 0},
        {"horizon": 0.0},
        {"synergy": -0.1},
        {"fixed_settings": {"scope": 2.0}},
    ],
)
def test_config_rejects_invalid_values(bad):
    with pytest.raises((ValueError, TypeError)):
        CooperationConfig(**bad)


def test_config_rejects_unknown_fields_and_genes():
    with pytest.raises(KeyError):
        CooperationConfig.from_dict({"enabled": True, "telepathy": True})
    with pytest.raises(KeyError):
        CooperationConfig(founder_gene_values={"coop_charisma": 1.0})


# --------------------------------------------------------------------- genes
def test_extend_chromosome_appends_loci_once():
    base = default_hyperparameter_chromosome()
    extended = extend_chromosome(base, {"coop_scope": 0.9})
    assert has_cooperation_genes(extended) and not has_cooperation_genes(base)
    assert len(extended.genes) == len(base.genes) + len(COOPERATION_GENE_NAMES)
    assert extended.genes[: len(base.genes)] == base.genes
    assert extended.get_value("coop_scope") == 0.9
    assert extend_chromosome(extended, {"coop_scope": 0.1}) is extended
    assert all(extended.get_gene(n).evolvable for n in COOPERATION_GENE_NAMES)
    with pytest.raises(KeyError):
        cooperation_genes({"nope": 1.0})


def test_extended_chromosomes_mutate_and_cross_over():
    a = extend_chromosome(default_hyperparameter_chromosome(), {"coop_exit_cost": 0.0})
    b = extend_chromosome(default_hyperparameter_chromosome(), {"coop_exit_cost": 1.0})
    child = crossover_chromosomes(a, b, mode="uniform", rng=random.Random(3))
    assert has_cooperation_genes(child)
    assert child.get_value("coop_exit_cost") in (0.0, 1.0)
    mutated = mutate_chromosome(child, mutation_rate=1.0, mutation_scale=0.2, rng=random.Random(4))
    for name in COOPERATION_GENE_NAMES:
        gene = mutated.get_gene(name)
        assert gene.min_value <= gene.value <= gene.max_value


def test_settings_and_goals_from_gene_values():
    values = gene_values(extend_chromosome(default_hyperparameter_chromosome()))
    s = settings_from_values({**values, "coop_exclusivity": 0.49, "coop_max_members": 3.6})
    assert not s.exclusive and s.max_members == 4 and s.kind == "bond"
    assert settings_from_values({**values, "coop_exclusivity": 0.5}).exclusive
    mixed = blended_settings(
        [{**values, "coop_scope": 0.2, "coop_exclusivity": 1.0}, {**values, "coop_scope": 0.6, "coop_exclusivity": 0.2}]
    )
    assert mixed.scope == pytest.approx(0.4) and mixed.exclusive  # mean exclusivity 0.6
    zero = dict.fromkeys(MOTIVE_GENE_NAMES, 0.0)
    assert goals_from_values(zero).weight(Motive.HEDONIST) == pytest.approx(0.2)
    hedonist = goals_from_values({**zero, "motive_hedonist": 1.0})
    emotional = goals_from_values({**zero, "motive_emotional": 1.0})
    assert motive_similarity(hedonist, hedonist) == pytest.approx(1.0)
    assert motive_similarity(hedonist, emotional) == pytest.approx(-1.0)


# ------------------------------------------------------- service: lifecycle
def test_cooperation_service_lookup_is_strict():
    assert cooperation_service_of(None) is None
    assert cooperation_service_of(Mock()) is None
    world = FakeWorld(_config())
    assert cooperation_service_of(world) is world.cooperation
    actions = [FakeAction("move", 1.0), FakeAction("bond", 1.0), FakeAction("leave", 1.0)]
    assert [a.name for a in filter_cooperation_actions(actions, Mock())] == ["move"]
    assert len(filter_cooperation_actions(actions, world)) == 3
    assert [int(x) for x in CooperationActionType] == [8, 9]
    assert COOPERATION_ACTION_NAMES == ("bond", "leave")


def test_actions_without_cooperation_fail_cleanly():
    agent = Mock()
    agent.environment = Mock()
    assert bond_action(agent)["success"] is False
    assert leave_action(agent)["success"] is False


def test_founders_get_loci_and_offspring_keep_inherited_ones():
    world = FakeWorld(_config(randomize_founder_genes=True), seed=11)
    a = world.add("a", (0, 0))
    b = world.add("b", (1, 0))
    assert has_cooperation_genes(a.hyperparameter_chromosome)
    va, vb = gene_values(a.hyperparameter_chromosome), gene_values(b.hyperparameter_chromosome)
    assert va != vb  # independently drawn
    again = FakeWorld(_config(randomize_founder_genes=True), seed=11).add("a", (0, 0))
    assert gene_values(again.hyperparameter_chromosome) == va  # seeded
    child = FakeAgent(world, "c", (0, 0), parent="a")
    child.hyperparameter_chromosome = a.hyperparameter_chromosome
    world.agents["c"] = child
    world.cooperation.on_agent_added(child)
    assert child.hyperparameter_chromosome is a.hyperparameter_chromosome
    assert world.cooperation.founders_seen == 2


def test_fixed_genes_when_evolution_is_off():
    world = FakeWorld(_config(evolve_genes=False))
    a = world.add("a", (0, 0))
    assert not any(a.hyperparameter_chromosome.get_gene(n).evolvable for n in COOPERATION_GENE_NAMES)


def test_action_weights_follow_propensity_genes_idempotently():
    world = FakeWorld(_config())
    a = world.add("a", (0, 0), values={"bond_weight": 0.5, "leave_weight": 0.25})
    weights = {x.name: x.weight for x in a.actions}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["bond"] == pytest.approx(0.5 / 1.75)
    assert weights["leave"] == pytest.approx(0.25 / 1.75)
    assert weights["move"] / weights["gather"] == pytest.approx(0.4 / 0.3)
    world.cooperation.refresh_action_weights(a)
    assert {x.name: x.weight for x in a.actions} == pytest.approx(weights)


# ------------------------------------------------------- service: formation
def test_two_open_agents_in_range_form_a_unit_with_blended_terms():
    world = FakeWorld(_config())
    a = world.add("a", (0, 0), values={**PAIR_VALUES, "coop_scope": 0.2})
    b = world.add("b", (3, 4), values={**PAIR_VALUES, "coop_scope": 0.6})
    first = world.cooperation.bond(a)
    assert first["success"] is False and world.cooperation.is_open("a")
    second = world.cooperation.bond(b)
    assert second["success"] is True
    assert second["details"]["distance"] == pytest.approx(5.0)
    unit = world.cooperation.registry.get(second["details"]["unit_id"])
    assert set(unit.members) == {"a", "b"}
    assert unit.settings.scope == pytest.approx(0.4)
    assert list(unit.members) == ["a", "b"]  # the standing offer ranks first
    assert not world.cooperation.is_open("a") and not world.cooperation.is_open("b")
    assert world.cooperation.is_partner("a", "b")
    assert world.cooperation.counters["units_formed"] == 1


def test_initiator_terms_and_fixed_settings():
    world = FakeWorld(_config(terms="initiator"))
    a = world.add("a", (0, 0), values={**PAIR_VALUES, "coop_scope": 0.2})
    b = world.add("b", (1, 0), values={**PAIR_VALUES, "coop_scope": 0.6})
    world.cooperation.bond(a)
    unit_id = world.cooperation.bond(b)["details"]["unit_id"]
    assert world.cooperation.registry.get(unit_id).settings.scope == pytest.approx(0.2)

    fixed = FakeWorld(_config(fixed_settings={"scope": 0.9, "exit_cost": 0.0, "recognition": 0.0}))
    c = fixed.add("c", (0, 0), values=PAIR_VALUES)
    d = fixed.add("d", (1, 0), values=PAIR_VALUES)
    fixed.cooperation.bond(c)
    unit_id = fixed.cooperation.bond(d)["details"]["unit_id"]
    assert fixed.cooperation.registry.get(unit_id).settings.scope == pytest.approx(0.9)


def test_offers_need_range_and_expire():
    world = FakeWorld(_config(offer_window=3))
    a = world.add("a", (0, 0), values=PAIR_VALUES)
    b = world.add("b", (10, 0), values=PAIR_VALUES)
    world.cooperation.bond(a)
    assert world.cooperation.bond(b)["success"] is False  # out of range
    c = world.add("c", (1, 0), values=PAIR_VALUES)
    for _ in range(4):
        world.end_step()
    assert not world.cooperation.is_open("a")
    assert world.cooperation.bond(c)["success"] is False


def test_placebo_never_forms_units_but_keeps_genes_and_weights():
    world = FakeWorld(_config(mode="placebo"))
    a = world.add("a", (0, 0), values=PAIR_VALUES)
    b = world.add("b", (1, 0), values=PAIR_VALUES)
    assert world.cooperation.bond(a)["error"] == "placebo"
    assert world.cooperation.bond(b)["error"] == "placebo"
    assert world.cooperation.leave(a)["error"] == "placebo"
    assert not world.cooperation.registry.units
    assert world.cooperation.on_income(a, 3.0) == 0.0
    assert has_cooperation_genes(a.hyperparameter_chromosome)
    assert world.cooperation.partner_ids("a") == set()
    world.end_step()
    assert world.cooperation.counters["bond_placebo"] == 2


def test_a_newcomer_joins_a_group_with_room():
    values = {**PAIR_VALUES, "coop_exclusivity": 0.0, "coop_max_members": 3.0}
    world = FakeWorld(_config())
    a = world.add("a", (0, 0), values=values)
    b = world.add("b", (1, 0), values=values)
    c = world.add("c", (0, 1), values=values)
    world.cooperation.bond(a)
    unit_id = world.cooperation.bond(b)["details"]["unit_id"]
    world.cooperation.bond(a)  # the group recruits
    joined = world.cooperation.bond(c)
    assert joined["success"] and joined["details"]["how"] == "join_other"
    assert joined["details"]["unit_id"] == unit_id
    assert set(world.cooperation.registry.get(unit_id).members) == {"a", "b", "c"}


def test_recognition_blocks_switching_out_of_an_exclusive_unit():
    world, a, _b = _bonded_pair(values={"coop_recognition": 1.0})
    c = world.add("c", (0, 1), values=PAIR_VALUES)
    world.cooperation.bond(c)
    result = world.cooperation.bond(a)  # a wants to switch to c
    assert result["success"] is False
    assert result["details"]["reasons"] == {"offer_blocked": 1}
    assert world.cooperation.is_partner("a", "b")


def test_unrecognised_exclusive_unit_lets_a_member_switch_and_pay_nothing():
    world, a, _b = _bonded_pair(values={"coop_recognition": 0.0, "coop_exit_cost": 1.0})
    c = world.add("c", (0, 1), values={**PAIR_VALUES, "coop_recognition": 0.0, "coop_exit_cost": 1.0})
    world.cooperation.bond(c)
    result = world.cooperation.bond(a)
    assert result["success"] and result["details"]["switched"] == ["a"]
    assert world.cooperation.is_partner("a", "c") and not world.cooperation.is_partner("a", "b")
    assert a.resource_level == pytest.approx(10.0)
    assert not world.cooperation.registry.is_member("b")


def test_switching_can_be_disabled():
    world, a, _b = _bonded_pair(config=_config(allow_switching=False), values={"coop_recognition": 0.0})
    c = world.add("c", (0, 1), values=PAIR_VALUES)
    world.cooperation.bond(c)
    assert world.cooperation.bond(a)["success"] is False
    assert world.cooperation.is_partner("a", "b")


# ----------------------------------------------------------- service: flows
def test_members_pool_gathered_income_and_receive_payouts_as_pending_reward():
    world, a, b = _bonded_pair()
    assert world.cooperation.on_income(a, 4.0) == pytest.approx(2.0)  # scope 0.5
    outsider = world.add("z", (50, 50), values=PAIR_VALUES)
    assert world.cooperation.on_income(outsider, 4.0) == 0.0
    a.resource_level -= 2.0  # the gather action keeps only income - pooled
    world.end_step()
    # synergy 0, payout rate 1, no reserve: the pool of 2 splits 1 / 1
    assert a.resource_level == pytest.approx(9.0) and b.resource_level == pytest.approx(11.0)
    income, _reward = world.cooperation.consume_pending("b")
    assert income == pytest.approx(1.0)
    assert world.cooperation.consume_pending("b") == (0.0, 0.0)


def test_membership_reward_follows_motive_fit():
    zero = dict.fromkeys(MOTIVE_GENE_NAMES, 0.0)
    # Religious agents like recognition and exit costs; hedonists dislike exit costs.
    world, _a, _b = _bonded_pair(values={**zero, "motive_religious": 1.0})
    world.end_step()
    assert world.cooperation.consume_pending("a")[1] > 0.0
    world2, c, _ = _bonded_pair(values={**zero, "motive_hedonist": 1.0, "coop_symmetry": 1.0})
    world2.end_step()
    assert world2.cooperation.consume_pending(c.agent_id)[1] == pytest.approx(2.0 * -0.5 / 20.0)


def test_enforced_exit_forfeits_holdings_to_the_deserted_partner():
    world, a, b = _bonded_pair(values={"coop_exit_cost": 0.5, "coop_recognition": 1.0})
    result = world.cooperation.leave(a)
    assert result["success"] and result["details"]["enforced"] and result["details"]["dissolved"]
    assert a.resource_level == pytest.approx(5.0)
    assert b.resource_level == pytest.approx(15.0)
    assert world.cooperation.consume_pending("b")[0] == pytest.approx(5.0)
    assert world.cooperation.leave(a)["error"] == "not_a_member"


def test_unenforced_exit_is_free():
    world, a, _b = _bonded_pair(values={"coop_exit_cost": 1.0, "coop_recognition": 0.0})
    assert world.cooperation.leave(a)["details"]["paid"] == 0.0
    assert a.resource_level == pytest.approx(10.0)


def test_aid_reaches_a_member_below_the_threshold():
    cfg = _config(aid_threshold=2.0, aid_amount=1.0, payout_rate=0.0)
    world, a, b = _bonded_pair(config=cfg)
    world.cooperation.on_income(b, 6.0)
    a.resource_level = 0.5
    world.end_step()
    assert a.resource_level == pytest.approx(1.5)
    assert world.cooperation.consume_pending("a")[0] == pytest.approx(1.0)


def test_death_dissolves_the_unit_and_pays_the_survivor():
    world, a, b = _bonded_pair()
    world.cooperation.on_income(a, 4.0)
    unit = world.cooperation.registry.units_of("b")[0]
    world.kill("a")
    assert not unit.is_active and unit.dissolution_reason == "member_death"
    assert b.resource_level == pytest.approx(12.0)
    assert world.cooperation.agent("a") is None


def test_dead_members_unknown_to_the_service_are_purged_at_settlement():
    world, a, _b = _bonded_pair()
    a.alive = False  # died without on_agent_removed
    world.end_step()
    assert not world.cooperation.registry.is_member("b")
    assert world.cooperation.agent("a") is None
    assert "a" not in world.cooperation._income_ema
    # A corpse can no longer be treated as a parent.
    world.add("c", (0, 0), values=PAIR_VALUES, parent="a")
    assert world.cooperation.counters["born_in"] == 0


def test_a_vetoed_switch_leaves_every_existing_unit_intact():
    free_exit = {**PAIR_VALUES, "coop_recognition": 0.0, "coop_exit_cost": 0.0}
    cfg = _config(family_on_birth=True, family_settings={"recognition": 1.0})
    world = FakeWorld(cfg)
    a = world.add("a", (0, 0), values=free_exit)
    b = world.add("b", (1, 0), values=free_exit)
    world.cooperation.bond(a)
    world.cooperation.bond(b)
    world.add("kid", (0, 0), values=free_exit, parent="a")  # the a-b bond now sits in a family
    d = world.add("d", (0, 2), values=free_exit)
    e = world.add("e", (0, 3), values=free_exit)
    world.cooperation.bond(d)
    world.cooperation.bond(e)
    world.cooperation.bond(d)  # d is open again, ready to switch
    before = {u.unit_id for u in world.cooperation.registry.active_units()}
    result = world.cooperation.bond(a)  # needs d to leave d-e and a to leave a-b; the family vetoes a
    assert result["success"] is False
    assert result["details"]["reasons"] == {"switch_vetoed": 1}
    assert {u.unit_id for u in world.cooperation.registry.active_units()} == before
    assert world.cooperation.is_partner("a", "b") and world.cooperation.is_partner("d", "e")


def test_family_on_birth_nests_the_parents_unit():
    world, _a, _b = _bonded_pair(config=_config(family_on_birth=True))
    child = world.add("c", (0, 0), values=PAIR_VALUES, parent="a")
    pair = next(u for u in world.cooperation.registry.units_of("a") if u.kind == "bond")
    family = world.cooperation.registry.units_of("c")[0]
    assert family.kind == "family" and pair.parent_id == family.unit_id
    sibling = world.add("d", (0, 0), values=PAIR_VALUES, parent="b")
    assert family.unit_id in {u.unit_id for u in world.cooperation.registry.units_of("d")}
    assert world.cooperation.is_partner("c", "d")
    del child, sibling


def test_partner_is_preferred_coparent():
    world, a, b = _bonded_pair()
    z = world.add("z", (0.5, 0), values=PAIR_VALUES)
    assert world.cooperation.preferred_coparent(a, [z, b]) is b
    assert world.cooperation.preferred_coparent(z, [a, b]) is None


# ---------------------------------------------------- service: invariants
def _random_world(seed: int, steps: int = 300, n: int = 24, **cfg):
    rng = random.Random(seed)
    config = _config(randomize_founder_genes=True, synergy=0.2, payout_rate=0.8, reserve_target=2.0, **cfg)
    world = FakeWorld(config, seed=seed)
    for i in range(n):
        world.add(f"a{i}", (rng.uniform(0, 20), rng.uniform(0, 20)), resources=10.0)
    gathered = 0.0
    for _ in range(steps):
        for agent in list(world.agents.values()):
            r = rng.random()
            if r < 0.4:
                income = rng.uniform(0.0, 3.0)
                gathered += income
                pooled = world.cooperation.on_income(agent, income)
                agent.resource_level += income - pooled
            elif r < 0.55:
                world.cooperation.bond(agent)
            elif r < 0.6:
                world.cooperation.leave(agent)
            else:
                x, y = agent.position
                agent.position = (x + rng.uniform(-1, 1), y + rng.uniform(-1, 1))
        world.end_step()
    return world, gathered


def test_resources_are_conserved_up_to_synergy():
    world, gathered = _random_world(seed=5)
    reg = world.cooperation.registry
    held = sum(a.resource_level for a in world.agents.values())
    pools = sum(u.pool for u in reg.units.values())
    surplus = sum(u.totals["surplus"] for u in reg.units.values())
    assert held + pools == pytest.approx(24 * 10.0 + gathered + surplus, rel=1e-9)
    assert world.cooperation.counters["units_formed"] > 0
    assert world.cooperation.counters["exits_leave"] > 0


def test_unit_bookkeeping_stays_consistent():
    world, _ = _random_world(seed=9, steps=200, founder_gene_values={})
    reg = world.cooperation.registry
    for agent_id in world.agents:
        held = reg.units_of(agent_id, "bond")
        if any(u.settings.exclusive for u in held):
            assert len(held) == 1
        for unit in held:
            assert agent_id in unit.members
    for unit in reg.active_units():
        assert unit.n_participants >= unit.settings.min_members
        assert unit.settings.max_members is None or unit.n_members <= unit.settings.max_members


def test_telemetry_and_outputs(tmp_path):
    world, _ = _random_world(seed=3, steps=60, n=12, output_dir=str(tmp_path), record_interval=2)
    service = world.cooperation
    rows = service.timeseries
    assert len(rows) == 30 and rows[-1]["step"] == 58
    assert {"coverage", "units_active", "mean_coop_exit_cost", "unit_scope", "dominant_hedonist"} <= set(rows[0])
    assert all(0.0 <= r["coverage"] <= 1.0 for r in rows)
    service.close()
    service.close()  # idempotent
    summary = json.loads((tmp_path / SUMMARY_FILE).read_text())
    assert summary["units_formed"] == service.counters["units_formed"]
    assert summary["action_indices"] == {"bond": 8, "leave": 9}
    assert (tmp_path / TIMESERIES_FILE).read_text().startswith("step,")
    units_csv = (tmp_path / UNITS_FILE).read_text().splitlines()
    assert units_csv[0].startswith("unit_id,") and len(units_csv) == 1 + len(service.registry.units)
