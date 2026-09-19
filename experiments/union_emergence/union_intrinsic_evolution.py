#!/usr/bin/env python3
"""Standalone union-emergence arena (PR 1014 / v2 threshold grid).

Exclusive pair-bonds versus promiscuous share under implicit selection.
Not wired into ``farm.runners`` or the chromosome; see SCOPE.md.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ── Frozen defaults (devlog 2026-09-18) ─────────────────────────────────────

DEFAULT_BONDING_COST = 0.8
DEFAULT_COURTSHIP_STEPS = 12
DEFAULT_EXIT_TAX = 1.1
DEFAULT_SOCIAL_RANGE = 3.2
DEFAULT_ACCIDENT_DIVORCE = 0.01
DEFAULT_MAX_POP = 80
DEFAULT_BOND_GAIN = 0.11
DEFAULT_BOND_DECAY = 0.015
DEFAULT_SYNERGY_K = 0.95
DEFAULT_FORCED_PAIR_RATE = 0.85
DEFAULT_EQUALIZE_RATE = 0.10
DEFAULT_GRIEF_SCALE = 0.18
DEFAULT_RESIDUAL_SHARE = 0.03
DEFAULT_PROMISCUOUS_SHARE = 0.18
COLOCATION_RADIUS = 1.75
GATHER_RADIUS = 1.55
PAIR_PULL = 0.95
INITIAL_BOND_STRENGTH = 0.18

ARMS = ("solo_only", "promiscuous", "optional_union", "forced_union")
UNION_ARMS = ("optional_union", "forced_union")


@dataclass(frozen=True)
class WorldParams:
    """Literature knobs for one v2 (or v1) world cell."""

    name: str = "baseline"
    bonding_cost: float = DEFAULT_BONDING_COST
    courtship_steps: int = DEFAULT_COURTSHIP_STEPS
    exit_tax: float = DEFAULT_EXIT_TAX
    social_range: float = DEFAULT_SOCIAL_RANGE
    accident_divorce: float = DEFAULT_ACCIDENT_DIVORCE
    max_pop: int = DEFAULT_MAX_POP
    bond_gain: float = DEFAULT_BOND_GAIN
    bond_decay: float = DEFAULT_BOND_DECAY
    synergy_k: float = DEFAULT_SYNERGY_K
    binary_lock: bool = False
    world_size: float = 22.0
    n_resources: int = 52
    initial_pop: int = 48
    metabolism: float = 0.08
    gather_base: float = 0.38
    resource_regen: float = 0.24
    resource_cap: float = 3.4
    offspring_cost: float = 1.7
    reproduce_threshold: float = 3.6
    mutation_scale: float = 0.08


@dataclass(frozen=True)
class RunSpec:
    """One cell of the grid: world × arm × seed × horizon."""

    world: WorldParams
    arm: str
    seed: int
    steps: int


def v2_worlds() -> Tuple[WorldParams, ...]:
    """Four worlds in the compact threshold grid."""
    baseline = WorldParams(name="baseline")
    return (
        baseline,
        replace(
            baseline,
            name="cheap_exit_cheap_bond",
            bonding_cost=0.2,
            exit_tax=0.2,
        ),
        replace(baseline, name="no_courtship", courtship_steps=0),
        replace(baseline, name="wide_neighborhood", social_range=8.0),
    )


def v1_world() -> WorldParams:
    """v1 pilot: instant synergy, no bonding cost, binary lock."""
    return WorldParams(
        name="v1_binary_lock",
        bonding_cost=0.0,
        courtship_steps=0,
        exit_tax=0.4,
        binary_lock=True,
        accident_divorce=0.008,
    )


def clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _torus_delta(a: float, b: float, size: float) -> float:
    raw = a - b
    half = size * 0.5
    if raw > half:
        raw -= size
    elif raw < -half:
        raw += size
    return raw


def torus_distance(ax: float, ay: float, bx: float, by: float, size: float) -> float:
    dx = _torus_delta(ax, bx, size)
    dy = _torus_delta(ay, by, size)
    return math.hypot(dx, dy)


@dataclass
class Agent:
    agent_id: int
    lineage: int
    x: float
    y: float
    energy: float
    pair_commitment: float
    fidelity: float
    specialize: float
    share_weight: float
    attack_weight: float
    partner_id: Optional[int] = None
    pair_age: int = 0
    bond_strength: float = 0.0
    role: str = "none"
    alive: bool = True
    stress: float = 0.0


@dataclass
class ResourcePatch:
    x: float
    y: float
    amount: float


@dataclass
class PairRecord:
    duration: int
    strength: float


@dataclass
class StepTelemetry:
    paired_energy: List[float] = field(default_factory=list)
    solo_energy: List[float] = field(default_factory=list)
    extractions: List[float] = field(default_factory=list)
    strengths: List[float] = field(default_factory=list)
    paired_count: int = 0
    alive_count: int = 0
    share_events: int = 0
    bond_events: int = 0
    leave_events: int = 0
    gather_events: int = 0
    attack_events: int = 0
    reproduce_events: int = 0


class UnionArena:
    """Implicit-selection 2-D energy arena with exclusive pair-bonds."""

    def __init__(self, spec: RunSpec) -> None:
        if spec.arm not in ARMS:
            raise ValueError(f"unknown arm {spec.arm!r}; expected one of {ARMS}")
        self.spec = spec
        self.world = spec.world
        self.arm = spec.arm
        self.rng = random.Random(spec.seed)
        self.agents: Dict[int, Agent] = {}
        self.resources: List[ResourcePatch] = []
        self.next_id = 0
        self.closed_pairs: List[PairRecord] = []
        self.telemetry = StepTelemetry()
        self._gene_start: Dict[str, float] = {}
        self._spawn_world()

    def _spawn_world(self) -> None:
        size = self.world.world_size
        for _ in range(self.world.n_resources):
            self.resources.append(
                ResourcePatch(
                    x=self.rng.random() * size,
                    y=self.rng.random() * size,
                    amount=self.rng.uniform(1.0, self.world.resource_cap),
                )
            )
        for _ in range(self.world.initial_pop):
            self._birth(parent=None, mate=None, x=None, y=None)
        self._gene_start = self._mean_genes()

    def _birth(
        self,
        parent: Optional[Agent],
        mate: Optional[Agent],
        x: Optional[float],
        y: Optional[float],
    ) -> Agent:
        size = self.world.world_size
        agent_id = self.next_id
        self.next_id += 1
        if parent is None:
            lineage = agent_id
            px = self.rng.random() * size if x is None else x
            py = self.rng.random() * size if y is None else y
            child = Agent(
                agent_id=agent_id,
                lineage=lineage,
                x=px,
                y=py,
                energy=self.rng.uniform(4.0, 6.2),
                pair_commitment=clip01(self.rng.gauss(0.52, 0.16)),
                fidelity=clip01(self.rng.gauss(0.48, 0.16)),
                specialize=clip01(self.rng.gauss(0.50, 0.16)),
                share_weight=max(0.0, min(2.0, self.rng.gauss(0.15, 0.08))),
                attack_weight=max(0.0, min(2.0, self.rng.gauss(0.10, 0.06))),
            )
        else:
            donor = mate if mate is not None else parent
            scale = self.world.mutation_scale

            def inherit(a: float, b: float) -> float:
                base = a if self.rng.random() < 0.5 else b
                return base + self.rng.gauss(0.0, scale)

            child = Agent(
                agent_id=agent_id,
                lineage=parent.lineage,
                x=(parent.x + self.rng.uniform(-0.4, 0.4)) % size,
                y=(parent.y + self.rng.uniform(-0.4, 0.4)) % size,
                energy=self.world.offspring_cost * 0.55,
                pair_commitment=clip01(inherit(parent.pair_commitment, donor.pair_commitment)),
                fidelity=clip01(inherit(parent.fidelity, donor.fidelity)),
                specialize=clip01(inherit(parent.specialize, donor.specialize)),
                share_weight=max(0.0, min(2.0, inherit(parent.share_weight, donor.share_weight))),
                attack_weight=max(0.0, min(2.0, inherit(parent.attack_weight, donor.attack_weight))),
            )
        self.agents[agent_id] = child
        return child

    def _alive(self) -> List[Agent]:
        return [agent for agent in self.agents.values() if agent.alive]

    def _mean_genes(self) -> Dict[str, float]:
        living = self._alive()
        if not living:
            return {
                "pair_commitment": 0.0,
                "fidelity": 0.0,
                "specialize": 0.0,
                "share_weight": 0.0,
                "attack_weight": 0.0,
            }
        return {
            "pair_commitment": statistics.fmean(a.pair_commitment for a in living),
            "fidelity": statistics.fmean(a.fidelity for a in living),
            "specialize": statistics.fmean(a.specialize for a in living),
            "share_weight": statistics.fmean(a.share_weight for a in living),
            "attack_weight": statistics.fmean(a.attack_weight for a in living),
        }

    def partner_of(self, agent: Agent) -> Optional[Agent]:
        if agent.partner_id is None:
            return None
        other = self.agents.get(agent.partner_id)
        if other is None or not other.alive:
            return None
        return other

    def are_colocated(self, a: Agent, b: Agent) -> bool:
        return torus_distance(a.x, a.y, b.x, b.y, self.world.world_size) <= COLOCATION_RADIUS

    def pairing_enabled(self) -> bool:
        return self.arm in UNION_ARMS

    def synergy_multiplier(self, agent: Agent) -> float:
        """Gather multiplier. 1.0 during courtship or if the partner is gone."""
        other = self.partner_of(agent)
        if other is None:
            return 1.0
        if agent.pair_age < self.world.courtship_steps:
            return 1.0
        if not self.are_colocated(agent, other):
            return 1.0
        mean_spec = 0.5 * (agent.specialize + other.specialize)
        strength = 1.0 if self.world.binary_lock else agent.bond_strength
        return 1.0 + self.world.synergy_k * strength * mean_spec

    def form_bond(self, a: Agent, b: Agent) -> bool:
        """Symmetric lock. Charges bonding_cost split. False if refused."""
        if a.agent_id == b.agent_id:
            return False
        if a.partner_id is not None or b.partner_id is not None:
            return False
        if not a.alive or not b.alive:
            return False
        share = self.world.bonding_cost * 0.5
        if a.energy < share or b.energy < share:
            return False
        a.energy -= share
        b.energy -= share
        a.partner_id = b.agent_id
        b.partner_id = a.agent_id
        a.pair_age = 0
        b.pair_age = 0
        if self.world.binary_lock:
            a.bond_strength = 1.0
            b.bond_strength = 1.0
        else:
            a.bond_strength = INITIAL_BOND_STRENGTH
            b.bond_strength = INITIAL_BOND_STRENGTH
        if a.specialize >= b.specialize:
            a.role, b.role = "gather", "guard"
        else:
            a.role, b.role = "guard", "gather"
        self.telemetry.bond_events += 1
        return True

    def dissolve(self, a: Agent, reason: str = "leave") -> None:
        """Clear both sides and zero strength. ``reason`` is leave/accident/death."""
        other = None
        if a.partner_id is not None:
            other = self.agents.get(a.partner_id)
        duration = a.pair_age
        strength = a.bond_strength
        if duration > 0:
            self.closed_pairs.append(PairRecord(duration=duration, strength=strength))
        for agent in (a, other):
            if agent is None:
                continue
            agent.partner_id = None
            agent.pair_age = 0
            agent.bond_strength = 0.0
            agent.role = "none"
        if reason in ("leave", "accident"):
            self.telemetry.leave_events += 1

    def _neighbors(self, agent: Agent, radius: float, living: Sequence[Agent]) -> List[Agent]:
        found: List[Agent] = []
        size = self.world.world_size
        for other in living:
            if other.agent_id == agent.agent_id or not other.alive:
                continue
            if torus_distance(agent.x, agent.y, other.x, other.y, size) <= radius:
                found.append(other)
        return found

    def _nearest_resource(self, agent: Agent) -> Optional[ResourcePatch]:
        best: Optional[ResourcePatch] = None
        best_d = float("inf")
        size = self.world.world_size
        for patch in self.resources:
            dist = torus_distance(agent.x, agent.y, patch.x, patch.y, size)
            if dist < best_d:
                best_d = dist
                best = patch
        return best

    def _step_resources(self) -> None:
        for patch in self.resources:
            patch.amount = min(self.world.resource_cap, patch.amount + self.world.resource_regen)

    def _update_bonds(self, living: Sequence[Agent]) -> None:
        seen = set()
        for agent in living:
            other = self.partner_of(agent)
            if other is None:
                if agent.partner_id is not None:
                    self.dissolve(agent, reason="death")
                continue
            pair_key = tuple(sorted((agent.agent_id, other.agent_id)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            agent.pair_age += 1
            other.pair_age = agent.pair_age
            if self.world.binary_lock:
                agent.bond_strength = 1.0
                other.bond_strength = 1.0
                continue
            if self.are_colocated(agent, other):
                gain = self.world.bond_gain * min(agent.pair_commitment, other.pair_commitment)
                new_strength = clip01(agent.bond_strength + gain - self.world.bond_decay)
            else:
                new_strength = clip01(agent.bond_strength - self.world.bond_decay)
            agent.bond_strength = new_strength
            other.bond_strength = new_strength

    def _maybe_exit(self, living: Sequence[Agent]) -> None:
        if not self.pairing_enabled():
            return
        processed = set()
        for agent in living:
            other = self.partner_of(agent)
            if other is None:
                continue
            pair_key = tuple(sorted((agent.agent_id, other.agent_id)))
            if pair_key in processed:
                continue
            processed.add(pair_key)
            if self.rng.random() < self.world.accident_divorce:
                tax = 0.25 * self.world.exit_tax * 0.5
                agent.energy = max(0.0, agent.energy - tax)
                other.energy = max(0.0, other.energy - tax)
                self.dissolve(agent, reason="accident")
                continue
            stress = 0.5 * (agent.stress + other.stress)
            strength = agent.bond_strength
            for actor, peer in ((agent, other), (other, agent)):
                leave_p = (
                    (1.0 - actor.fidelity)
                    * (0.04 + stress)
                    * (1.0 - 0.5 * strength)
                )
                if self.rng.random() < leave_p:
                    tax = self.world.exit_tax * actor.fidelity * 0.5
                    actor.energy = max(0.0, actor.energy - tax)
                    peer.energy = max(0.0, peer.energy - 0.25 * tax)
                    self.dissolve(actor, reason="leave")
                    break

    def _maybe_bond(self, living: Sequence[Agent]) -> None:
        if not self.pairing_enabled():
            return
        singles = [agent for agent in living if agent.partner_id is None]
        self.rng.shuffle(singles)
        claimed = set()
        attempted_pairs = set()
        for agent in singles:
            if agent.agent_id in claimed or agent.partner_id is not None:
                continue
            candidates = [
                other
                for other in self._neighbors(agent, self.world.social_range, singles)
                if other.agent_id not in claimed and other.partner_id is None
            ]
            if not candidates:
                continue
            other = min(
                candidates,
                key=lambda cand: torus_distance(
                    agent.x, agent.y, cand.x, cand.y, self.world.world_size
                ),
            )
            pair_key = tuple(sorted((agent.agent_id, other.agent_id)))
            if pair_key in attempted_pairs:
                continue
            attempted_pairs.add(pair_key)
            if self.arm == "forced_union":
                pair_p = DEFAULT_FORCED_PAIR_RATE
            else:
                pair_p = agent.pair_commitment * other.pair_commitment
            if self.rng.random() < pair_p and self.form_bond(agent, other):
                claimed.add(agent.agent_id)
                claimed.add(other.agent_id)

    def _move(self, agent: Agent, living: Sequence[Agent]) -> None:
        size = self.world.world_size
        other = self.partner_of(agent)
        if other is not None:
            dx = _torus_delta(other.x, agent.x, size)
            dy = _torus_delta(other.y, agent.y, size)
            dist = math.hypot(dx, dy) or 1.0
            pull = PAIR_PULL + 0.25 * agent.bond_strength
            agent.x = (agent.x + pull * dx / dist) % size
            agent.y = (agent.y + pull * dy / dist) % size
            return
        patch = self._nearest_resource(agent)
        if patch is not None:
            dx = _torus_delta(patch.x, agent.x, size)
            dy = _torus_delta(patch.y, agent.y, size)
            dist = math.hypot(dx, dy) or 1.0
            step = 0.85
            agent.x = (agent.x + step * dx / dist + self.rng.uniform(-0.15, 0.15)) % size
            agent.y = (agent.y + step * dy / dist + self.rng.uniform(-0.15, 0.15)) % size
        else:
            agent.x = (agent.x + self.rng.uniform(-1.0, 1.0)) % size
            agent.y = (agent.y + self.rng.uniform(-1.0, 1.0)) % size

    def _gather(self, agent: Agent) -> None:
        patch = self._nearest_resource(agent)
        if patch is None:
            return
        dist = torus_distance(agent.x, agent.y, patch.x, patch.y, self.world.world_size)
        if dist > GATHER_RADIUS:
            return
        role_bonus = 1.18 if agent.role == "gather" else 1.0
        taken = min(patch.amount, self.world.gather_base * self.synergy_multiplier(agent) * role_bonus)
        patch.amount -= taken
        agent.energy += taken
        self.telemetry.gather_events += 1

    def _share_or_equalize(self, agent: Agent, living: Sequence[Agent]) -> None:
        other = self.partner_of(agent)
        if other is not None:
            if agent.agent_id < other.agent_id:
                return
            poorer, richer = (agent, other) if agent.energy < other.energy else (other, agent)
            gap = richer.energy - poorer.energy
            transfer = DEFAULT_EQUALIZE_RATE * gap
            richer.energy -= transfer
            poorer.energy += transfer
            return
        if self.arm == "solo_only":
            rate = DEFAULT_RESIDUAL_SHARE * agent.share_weight
        elif self.arm == "promiscuous":
            rate = DEFAULT_PROMISCUOUS_SHARE * agent.share_weight
        else:
            return
        if rate <= 0.0:
            return
        neighbors = self._neighbors(agent, self.world.social_range, living)
        if not neighbors:
            return
        target = min(neighbors, key=lambda cand: cand.energy)
        if target.energy >= agent.energy:
            return
        gift = min(agent.energy * 0.25, rate)
        agent.energy -= gift
        target.energy += gift
        self.telemetry.share_events += 1

    def _attack(self, agent: Agent, living: Sequence[Agent]) -> None:
        if agent.energy > 2.4 or agent.attack_weight < 0.12:
            return
        victims = self._neighbors(agent, 1.6, living)
        if not victims:
            return
        victim = self.rng.choice(victims)
        if self.partner_of(agent) is victim:
            return
        guard = 0.45 if victim.role == "guard" else 1.0
        steal = 0.18 * agent.attack_weight * guard
        steal = min(steal, victim.energy)
        victim.energy -= steal
        agent.energy += steal * 0.7
        self.telemetry.attack_events += 1

    def _reproduce(self, agent: Agent, living: Sequence[Agent]) -> None:
        if len(living) >= self.world.max_pop:
            return
        if agent.energy < self.world.reproduce_threshold:
            return
        other = self.partner_of(agent)
        mature = (
            other is not None
            and agent.pair_age >= self.world.courtship_steps
            and other.alive
        )
        cost = self.world.offspring_cost
        if mature:
            if agent.agent_id < other.agent_id:
                return
            each = 0.5 * cost
            if agent.energy < each or other.energy < each:
                return
            agent.energy -= each
            other.energy -= each
            mate = other
        else:
            if agent.energy < cost:
                return
            agent.energy -= cost
            mate = None
        self._birth(parent=agent, mate=mate, x=agent.x, y=agent.y)
        self.telemetry.reproduce_events += 1

    def _deaths(self, living: Sequence[Agent]) -> None:
        for agent in living:
            agent.energy -= self.world.metabolism
            hunger = max(0.0, 2.2 - agent.energy)
            agent.stress = clip01(0.55 * agent.stress + 0.25 * min(1.0, hunger / 2.2))
            if agent.energy <= 0.0:
                other = self.partner_of(agent)
                if other is not None:
                    other.energy = max(0.0, other.energy - DEFAULT_GRIEF_SCALE * other.fidelity)
                    self.dissolve(agent, reason="death")
                agent.alive = False
                agent.partner_id = None
                agent.bond_strength = 0.0

    def _record_metrics(self, living: Sequence[Agent]) -> None:
        self.telemetry.alive_count += len(living)
        paired = 0
        seen = set()
        for agent in living:
            other = self.partner_of(agent)
            if other is None:
                self.telemetry.solo_energy.append(agent.energy)
                continue
            self.telemetry.paired_energy.append(agent.energy)
            paired += 1
            pair_key = tuple(sorted((agent.agent_id, other.agent_id)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            extraction = abs(agent.energy - other.energy) * abs(
                agent.pair_commitment - other.pair_commitment
            )
            self.telemetry.extractions.append(extraction)
            self.telemetry.strengths.append(agent.bond_strength)
        self.telemetry.paired_count += paired

    def step(self) -> None:
        self._step_resources()
        living = self._alive()
        self._update_bonds(living)
        self._maybe_exit(living)
        living = self._alive()
        self._maybe_bond(living)
        living = self._alive()
        for agent in living:
            self._move(agent, living)
        for agent in living:
            self._gather(agent)
        for agent in living:
            self._share_or_equalize(agent, living)
        for agent in living:
            self._attack(agent, living)
        # Snapshot living before births so reproduction does not iterate a growing dict.
        reproducers = list(self._alive())
        for agent in reproducers:
            self._reproduce(agent, self._alive())
        self._deaths(self._alive())
        self._record_metrics(self._alive())

    def run(self) -> Dict[str, object]:
        for _ in range(self.spec.steps):
            self.step()
        return self.summarize()

    def summarize(self) -> Dict[str, object]:
        living = self._alive()
        genes = self._mean_genes()
        paired_e = self.telemetry.paired_energy
        solo_e = self.telemetry.solo_energy
        if paired_e and solo_e and statistics.fmean(solo_e) > 1e-9:
            synergy = statistics.fmean(paired_e) / statistics.fmean(solo_e)
        else:
            synergy = None
        durations = [record.duration for record in self.closed_pairs]
        # Include still-open pairs so short runs are not empty.
        for agent in living:
            if agent.partner_id is not None and agent.agent_id < agent.partner_id:
                durations.append(agent.pair_age)
        steps = max(1, self.spec.steps)
        paired_frac = self.telemetry.paired_count / max(1, self.telemetry.alive_count)
        return {
            "world": self.world.name,
            "arm": self.arm,
            "seed": self.spec.seed,
            "steps": self.spec.steps,
            "synergy_index": synergy,
            "paired_frac": paired_frac,
            "mean_pair_duration": statistics.fmean(durations) if durations else None,
            "mean_bond_strength": (
                statistics.fmean(self.telemetry.strengths) if self.telemetry.strengths else None
            ),
            "mean_extraction": (
                statistics.fmean(self.telemetry.extractions) if self.telemetry.extractions else None
            ),
            "lineages_alive": len({agent.lineage for agent in living}),
            "final_pop": len(living),
            "mean_energy": statistics.fmean(a.energy for a in living) if living else 0.0,
            "delta_pair_commitment": genes["pair_commitment"] - self._gene_start["pair_commitment"],
            "delta_fidelity": genes["fidelity"] - self._gene_start["fidelity"],
            "delta_specialize": genes["specialize"] - self._gene_start["specialize"],
            "delta_share_weight": genes["share_weight"] - self._gene_start["share_weight"],
            "delta_attack_weight": genes["attack_weight"] - self._gene_start["attack_weight"],
            "final_genes": genes,
            "start_genes": self._gene_start,
            "actions": {
                "share": self.telemetry.share_events / steps,
                "bond": self.telemetry.bond_events / steps,
                "leave": self.telemetry.leave_events / steps,
                "gather": self.telemetry.gather_events / steps,
                "attack": self.telemetry.attack_events / steps,
                "reproduce": self.telemetry.reproduce_events / steps,
            },
        }


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return statistics.fmean(present)


def _std_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if len(present) < 2:
        return 0.0 if present else None
    return statistics.pstdev(present)


def aggregate_runs(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Mean/std over seeds for one (world, arm) cell."""
    keys = (
        "synergy_index",
        "paired_frac",
        "mean_pair_duration",
        "mean_bond_strength",
        "mean_extraction",
        "lineages_alive",
        "final_pop",
        "mean_energy",
        "delta_pair_commitment",
        "delta_fidelity",
        "delta_specialize",
        "delta_share_weight",
        "delta_attack_weight",
    )
    summary: Dict[str, object] = {
        "world": runs[0]["world"],
        "arm": runs[0]["arm"],
        "n_seeds": len(runs),
        "steps": runs[0]["steps"],
        "seeds": [run["seed"] for run in runs],
    }
    for key in keys:
        values = [run.get(key) for run in runs]
        summary[key] = _mean_or_none(values)  # type: ignore[arg-type]
        summary[f"{key}_std"] = _std_or_none(values)  # type: ignore[arg-type]
    return summary


def evaluate_win_conditions(cells: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """First-glance / v2 checks against the pre-register."""
    by_key = {(cell["world"], cell["arm"]): cell for cell in cells}

    def get(world: str, arm: str, metric: str) -> Optional[float]:
        cell = by_key.get((world, arm))
        if cell is None:
            return None
        value = cell.get(metric)
        return float(value) if isinstance(value, (int, float)) else None

    optional_syn = get("baseline", "optional_union", "synergy_index")
    forced_syn = get("baseline", "forced_union", "synergy_index")
    cheap_syn = get("cheap_exit_cheap_bond", "optional_union", "synergy_index")
    noco_syn = get("no_courtship", "optional_union", "synergy_index")
    opt_extract = get("baseline", "optional_union", "mean_extraction")
    forced_extract = get("baseline", "forced_union", "mean_extraction")
    opt_d_commit = get("baseline", "optional_union", "delta_pair_commitment")
    opt_d_fid = get("baseline", "optional_union", "delta_fidelity")
    wide_d_fid = get("wide_neighborhood", "optional_union", "delta_fidelity")
    cheap_dur = get("cheap_exit_cheap_bond", "optional_union", "mean_pair_duration")
    base_dur = get("baseline", "optional_union", "mean_pair_duration")

    checks = {
        "baseline_optional_synergy_gt_1": optional_syn is not None and optional_syn > 1.0,
        "baseline_forced_synergy_gt_1": forced_syn is not None and forced_syn > 1.0,
        "no_courtship_inflates_optional_synergy": (
            optional_syn is not None and noco_syn is not None and noco_syn >= optional_syn
        ),
        "cheap_exit_drops_or_holds_optional_synergy": (
            optional_syn is not None and cheap_syn is not None and cheap_syn <= optional_syn + 0.02
        ),
        "forced_extraction_gt_optional": (
            opt_extract is not None and forced_extract is not None and forced_extract > opt_extract
        ),
        "optional_commitment_does_not_climb": opt_d_commit is not None and opt_d_commit <= 0.03,
        "optional_fidelity_rises_on_baseline": opt_d_fid is not None and opt_d_fid > 0.0,
        "wide_neighborhood_flattens_fidelity": (
            opt_d_fid is not None and wide_d_fid is not None and wide_d_fid <= opt_d_fid + 0.01
        ),
        "cheap_exit_does_not_lengthen_bonds": (
            base_dur is not None and cheap_dur is not None and cheap_dur <= base_dur + 1.0
        ),
    }
    return {
        "checks": checks,
        "passed": sum(1 for ok in checks.values() if ok),
        "total": len(checks),
        "headline": {
            "baseline_optional_synergy": optional_syn,
            "baseline_forced_synergy": forced_syn,
            "no_courtship_optional_synergy": noco_syn,
            "cheap_exit_optional_synergy": cheap_syn,
            "baseline_optional_extraction": opt_extract,
            "baseline_forced_extraction": forced_extract,
            "baseline_optional_delta_commitment": opt_d_commit,
            "baseline_optional_delta_fidelity": opt_d_fid,
        },
    }


def run_grid(specs: Sequence[RunSpec]) -> Dict[str, object]:
    raw: List[Dict[str, object]] = []
    for spec in specs:
        raw.append(UnionArena(spec).run())
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in raw:
        grouped.setdefault((str(row["world"]), str(row["arm"])), []).append(row)
    cells = [aggregate_runs(group) for group in grouped.values()]
    return {
        "n_runs": len(raw),
        "cells": cells,
        "runs": raw,
        "win_conditions": evaluate_win_conditions(cells),
    }


def grid_specs(mode: str, seeds: Optional[Sequence[int]], steps: Optional[int]) -> List[RunSpec]:
    if mode == "v1":
        world = v1_world()
        used_seeds = list(seeds) if seeds else list(range(8))
        horizon = steps if steps is not None else 900
        return [
            RunSpec(world=world, arm=arm, seed=seed, steps=horizon)
            for arm in ARMS
            for seed in used_seeds
        ]
    worlds = v2_worlds()
    if mode == "first_glance":
        used_seeds = list(seeds) if seeds else [0, 1]
        horizon = steps if steps is not None else 220
    elif mode == "v2":
        used_seeds = list(seeds) if seeds else list(range(4))
        horizon = steps if steps is not None else 550
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return [
        RunSpec(world=world, arm=arm, seed=seed, steps=horizon)
        for world in worlds
        for arm in ARMS
        for seed in used_seeds
    ]


def render_table(payload: Dict[str, object]) -> str:
    cells: Sequence[Dict[str, object]] = payload["cells"]  # type: ignore[assignment]
    lines = [
        "| world | arm | synergy | energy | paired | duration | Δ commit | Δ fidelity | extraction | lineages |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    def fmt(cell: Dict[str, object], key: str, digits: int = 3) -> str:
        value = cell.get(key)
        if value is None:
            return "—"
        return f"{float(value):.{digits}f}"

    for cell in cells:
        lines.append(
            "| {world} | {arm} | {syn} | {en} | {pf} | {dur} | {dc} | {df} | {ex} | {lin} |".format(
                world=cell["world"],
                arm=cell["arm"],
                syn=fmt(cell, "synergy_index"),
                en=fmt(cell, "mean_energy", 1),
                pf=fmt(cell, "paired_frac"),
                dur=fmt(cell, "mean_pair_duration", 1),
                dc=fmt(cell, "delta_pair_commitment"),
                df=fmt(cell, "delta_fidelity"),
                ex=fmt(cell, "mean_extraction"),
                lin=fmt(cell, "lineages_alive", 1),
            )
        )
    wins = payload.get("win_conditions", {})
    checks = wins.get("checks", {}) if isinstance(wins, dict) else {}
    lines.append("")
    lines.append(
        "Win-condition direction checks: "
        f"{wins.get('passed', 0)}/{wins.get('total', 0)}."
    )
    for name, ok in checks.items():
        mark = "PASS" if ok else "FAIL"
        lines.append(f"- {mark}: {name}")
    return "\n".join(lines)


def write_outputs(
    payload: Dict[str, object],
    output_dir: Path,
    mode: str,
    sandbox_dir: Optional[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "v1":
        summary_name = "v1_pilot_summary.json"
        report_name = "v1_pilot.md"
    elif mode == "first_glance":
        summary_name = "first_glance_summary.json"
        report_name = "first_glance_grid.md"
    else:
        summary_name = "compact_threshold_summary.json"
        report_name = "v2_grid.md"
    summary_path = output_dir / summary_name
    report_path = output_dir / report_name
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    table = render_table(payload)
    report_path.write_text(
        f"# Union emergence ({mode})\n\n{table}\n",
        encoding="utf-8",
    )
    if sandbox_dir is not None:
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        (sandbox_dir / summary_name).write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("first_glance", "v2", "v1"),
        default="v2",
        help="first_glance = 2×220 v2 grid; v2 = 4×550; v1 = 8×900 binary-lock pilot",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override horizon")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Override seed list")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/union_emergence"),
        help="Where to write JSON/markdown summaries",
    )
    parser.add_argument(
        "--sandbox-dir",
        type=Path,
        default=None,
        help="Optional extra copy path (union_experiment/ in the original sandbox)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    specs = grid_specs(args.mode, args.seeds, args.steps)
    payload = run_grid(specs)
    payload["mode"] = args.mode
    payload["spec"] = {
        "n_specs": len(specs),
        "arms": list(ARMS),
        "worlds": sorted({spec.world.name for spec in specs}),
        "seeds": sorted({spec.seed for spec in specs}),
        "steps": specs[0].steps if specs else None,
    }
    path = write_outputs(payload, args.output_dir, args.mode, args.sandbox_dir)
    print(render_table(payload))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
