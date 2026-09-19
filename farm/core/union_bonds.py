"""Exclusive pair-bond mechanics for the union-emergence AgentFarm port.

Runtime state (``partner_id``, ``pair_age``, ``bond_strength``, ``role``) lives
on the agent. Heritable willingness lives on the chromosome
(``pair_commitment``, ``fidelity``, ``specialize``). Pairing can be implicit
(literature knobs) or chosen via the ``bond`` / ``leave`` actions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from farm.core.hyperparameter_chromosome import (
    HyperparameterChromosome,
    HyperparameterGene,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)

UNION_GENE_NAMES: Tuple[str, ...] = ("pair_commitment", "fidelity", "specialize")
UNION_ACTION_NAMES: Tuple[str, ...] = ("bond", "leave")
PAIRING_MODES: Tuple[str, ...] = ("off", "optional", "forced")

# Chromosome A (learning) loci stay frozen for the Layer C experiment.
LEARNING_GENE_NAMES: Tuple[str, ...] = (
    "learning_rate",
    "gamma",
    "epsilon_decay",
    "memory_size",
    "batch_size",
    "tau",
    "dqn_hidden_size",
    "target_update_freq",
    "rl_train_freq",
    "per_alpha",
    "per_beta_start",
    "per_beta_end",
    "ensemble_size",
)


@dataclass
class UnionPolicy:
    """Literature knobs + arm mode for one union-emergence run."""

    enabled: bool = False
    pairing_mode: str = "off"
    implicit_pairing: bool = True
    bonding_cost: float = 0.8
    courtship_steps: int = 12
    exit_tax: float = 1.1
    social_range: float = 3.2
    accident_divorce: float = 0.01
    bond_gain: float = 0.11
    bond_decay: float = 0.015
    synergy_k: float = 0.95
    forced_pair_rate: float = 0.85
    equalize_rate: float = 0.10
    grief_scale: float = 0.18
    residual_share: bool = False
    suppress_promiscuous_share: bool = True
    initial_bond_strength: float = 0.18
    colocation_radius: float = 1.75
    pair_pull: float = 0.95
    implicit_leave: bool = True
    leave_events: int = 0
    bond_events: int = 0

    def __post_init__(self) -> None:
        if self.pairing_mode not in PAIRING_MODES:
            raise ValueError(f"pairing_mode must be one of {PAIRING_MODES}; got {self.pairing_mode!r}")

    @property
    def pairing_enabled(self) -> bool:
        return self.enabled and self.pairing_mode in ("optional", "forced")


def policy_for_arm(arm: str, base: Optional[UnionPolicy] = None) -> UnionPolicy:
    """Return a policy copy with pairing/share rules for one experiment arm."""
    policy = replace(base or UnionPolicy(enabled=True))
    policy.enabled = True
    if arm == "solo_only":
        policy.pairing_mode = "off"
        policy.residual_share = True
        policy.suppress_promiscuous_share = True
    elif arm == "promiscuous":
        policy.pairing_mode = "off"
        policy.residual_share = False
        policy.suppress_promiscuous_share = False
    elif arm == "optional_union":
        policy.pairing_mode = "optional"
        policy.residual_share = False
        policy.suppress_promiscuous_share = True
    elif arm == "forced_union":
        policy.pairing_mode = "forced"
        policy.residual_share = False
        policy.suppress_promiscuous_share = True
    else:
        raise ValueError(f"unknown union arm {arm!r}")
    return policy


def get_union_policy(owner: Any) -> Optional[UnionPolicy]:
    """Read ``union_policy`` from an environment or agent."""
    if owner is None:
        return None
    policy = getattr(owner, "union_policy", None)
    if isinstance(policy, UnionPolicy):
        return policy
    env = getattr(owner, "environment", None)
    policy = getattr(env, "union_policy", None) if env is not None else None
    return policy if isinstance(policy, UnionPolicy) else None


def union_enabled(owner: Any) -> bool:
    policy = get_union_policy(owner)
    if policy is not None:
        return policy.enabled
    config = getattr(owner, "config", None)
    if config is None and hasattr(owner, "environment"):
        config = getattr(getattr(owner, "environment", None), "config", None)
    return bool(getattr(config, "union_enabled", False))


def gene_value(agent: Any, name: str, default: float) -> float:
    chromosome = getattr(agent, "hyperparameter_chromosome", None)
    if chromosome is None:
        return default
    try:
        return float(chromosome.get_value(name))
    except (KeyError, TypeError, ValueError):
        return default


def ensure_bond_state(agent: Any) -> None:
    """Initialize union fields on an agent that predates this port."""
    if not hasattr(agent, "partner_id"):
        agent.partner_id = None
    if not hasattr(agent, "pair_age"):
        agent.pair_age = 0
    if not hasattr(agent, "bond_strength"):
        agent.bond_strength = 0.0
    if not hasattr(agent, "role"):
        agent.role = "none"


def partner_of(agent: Any) -> Optional[Any]:
    ensure_bond_state(agent)
    partner_id = getattr(agent, "partner_id", None)
    if partner_id is None:
        return None
    env = getattr(agent, "environment", None)
    if env is None:
        return None
    other = None
    lookup = getattr(env, "get_agent", None)
    if callable(lookup):
        other = lookup(partner_id)
    if other is None:
        objects = getattr(env, "_agent_objects", None)
        if isinstance(objects, dict):
            other = objects.get(partner_id)
    if other is None:
        for candidate in getattr(env, "alive_agent_objects", []):
            if getattr(candidate, "agent_id", None) == partner_id:
                other = candidate
                break
    if other is None or not getattr(other, "alive", False):
        return None
    return other


def are_colocated(agent: Any, other: Any, radius: float) -> bool:
    return _distance(agent, other) <= radius


def _distance(agent: Any, other: Any) -> float:
    pos_a = getattr(agent, "position", (0.0, 0.0))
    pos_b = getattr(other, "position", (0.0, 0.0))
    return math.dist(pos_a, pos_b)


def _world_size(agent: Any) -> Tuple[float, float]:
    env = getattr(agent, "environment", None)
    width = float(getattr(env, "width", 0.0) or 0.0)
    height = float(getattr(env, "height", 0.0) or 0.0)
    return width, height


def _shortest_delta(delta: float, size: float) -> float:
    if size <= 0.0:
        return delta
    half = size * 0.5
    if delta > half:
        return delta - size
    if delta < -half:
        return delta + size
    return delta


def _assign_position(agent: Any, position: Tuple[float, float]) -> None:
    agent.position = position


def _pull_pair_together(agent: Any, other: Any, policy: UnionPolicy) -> None:
    """Move both partners toward each other so courtship/synergy can fire."""
    width, height = _world_size(agent)
    ax, ay = getattr(agent, "position", (0.0, 0.0))
    bx, by = getattr(other, "position", (0.0, 0.0))
    dx = _shortest_delta(bx - ax, width)
    dy = _shortest_delta(by - ay, height)
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return
    strength = float(getattr(agent, "bond_strength", 0.0))
    step = min(dist * 0.5, policy.pair_pull + 0.25 * strength)
    ux, uy = dx / dist, dy / dist

    def _step(x: float, y: float, sign: float) -> Tuple[float, float]:
        nx = x + sign * step * ux
        ny = y + sign * step * uy
        if width > 0.0:
            nx %= width
        if height > 0.0:
            ny %= height
        return (nx, ny)

    _assign_position(agent, _step(ax, ay, 1.0))
    _assign_position(other, _step(bx, by, -1.0))


def is_mature_bond(agent: Any, policy: Optional[UnionPolicy] = None) -> bool:
    policy = policy or get_union_policy(agent)
    if policy is None or not policy.pairing_enabled:
        return False
    other = partner_of(agent)
    if other is None:
        return False
    return int(getattr(agent, "pair_age", 0)) >= int(policy.courtship_steps)


def mature_bond_partner(agent: Any) -> Optional[Any]:
    if not is_mature_bond(agent):
        return None
    return partner_of(agent)


def gather_synergy_multiplier(agent: Any) -> float:
    """1.0 during courtship, if unpaired, or if the partner is gone."""
    policy = get_union_policy(agent)
    if policy is None or not policy.pairing_enabled:
        return 1.0
    other = partner_of(agent)
    if other is None:
        return 1.0
    if int(getattr(agent, "pair_age", 0)) < int(policy.courtship_steps):
        return 1.0
    if not are_colocated(agent, other, policy.colocation_radius):
        return 1.0
    mean_spec = 0.5 * (gene_value(agent, "specialize", 0.5) + gene_value(other, "specialize", 0.5))
    strength = float(getattr(agent, "bond_strength", 0.0))
    return 1.0 + policy.synergy_k * strength * mean_spec


def form_bond(first: Any, second: Any, policy: Optional[UnionPolicy] = None) -> bool:
    """Symmetric lock. Charges ``bonding_cost`` split. False if refused."""
    policy = policy or get_union_policy(first) or get_union_policy(second)
    if policy is None or not policy.pairing_enabled:
        return False
    ensure_bond_state(first)
    ensure_bond_state(second)
    if first is second:
        return False
    if not getattr(first, "alive", False) or not getattr(second, "alive", False):
        return False
    if getattr(first, "partner_id", None) is not None or getattr(second, "partner_id", None) is not None:
        return False
    share = policy.bonding_cost * 0.5
    if float(first.resource_level) < share or float(second.resource_level) < share:
        return False
    first.resource_level = float(first.resource_level) - share
    second.resource_level = float(second.resource_level) - share
    first.partner_id = second.agent_id
    second.partner_id = first.agent_id
    first.pair_age = 0
    second.pair_age = 0
    first.bond_strength = policy.initial_bond_strength
    second.bond_strength = policy.initial_bond_strength
    if gene_value(first, "specialize", 0.5) >= gene_value(second, "specialize", 0.5):
        first.role, second.role = "gather", "guard"
    else:
        first.role, second.role = "guard", "gather"
    policy.bond_events += 1
    return True


def dissolve(agent: Any, reason: str = "leave", policy: Optional[UnionPolicy] = None) -> Optional[int]:
    """Clear both sides and zero strength. Returns the ended pair's age."""
    policy = policy or get_union_policy(agent)
    ensure_bond_state(agent)
    other = partner_of(agent)
    duration = int(getattr(agent, "pair_age", 0))
    for member in (agent, other):
        if member is None:
            continue
        ensure_bond_state(member)
        member.partner_id = None
        member.pair_age = 0
        member.bond_strength = 0.0
        member.role = "none"
    if policy is not None and reason in ("leave", "accident"):
        policy.leave_events += 1
    return duration if duration > 0 else None


def on_agent_terminate(agent: Any) -> None:
    """Apply grief to the widow and clear the bond."""
    policy = get_union_policy(agent)
    other = partner_of(agent)
    if other is None:
        ensure_bond_state(agent)
        agent.partner_id = None
        agent.bond_strength = 0.0
        return
    if policy is not None:
        grief = policy.grief_scale * gene_value(other, "fidelity", 0.5)
        other.resource_level = max(0.0, float(other.resource_level) - grief)
    dissolve(agent, reason="death", policy=policy)


def _pairing_probability(agent: Any, other: Any, policy: UnionPolicy) -> float:
    if policy.pairing_mode == "forced":
        return policy.forced_pair_rate
    return gene_value(agent, "pair_commitment", 0.5) * gene_value(other, "pair_commitment", 0.5)


def _neighbors(agent: Any, radius: float, living: Sequence[Any]) -> List[Any]:
    found: List[Any] = []
    for other in living:
        if other is agent or not getattr(other, "alive", False):
            continue
        if _distance(agent, other) <= radius:
            found.append(other)
    return found


def _update_live_bonds(living: Sequence[Any], policy: UnionPolicy) -> None:
    seen = set()
    for agent in living:
        other = partner_of(agent)
        if other is None:
            if getattr(agent, "partner_id", None) is not None:
                dissolve(agent, reason="death", policy=policy)
            continue
        pair_key = tuple(sorted((str(agent.agent_id), str(other.agent_id))))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        agent.pair_age = int(getattr(agent, "pair_age", 0)) + 1
        other.pair_age = agent.pair_age
        if are_colocated(agent, other, policy.colocation_radius):
            gain = policy.bond_gain * min(
                gene_value(agent, "pair_commitment", 0.5),
                gene_value(other, "pair_commitment", 0.5),
            )
            strength = min(1.0, max(0.0, float(agent.bond_strength) + gain - policy.bond_decay))
        else:
            strength = min(1.0, max(0.0, float(agent.bond_strength) - policy.bond_decay))
        agent.bond_strength = strength
        other.bond_strength = strength
        _pull_pair_together(agent, other, policy)


def _maybe_exit(living: Sequence[Any], policy: UnionPolicy, rng: random.Random) -> None:
    processed = set()
    for agent in living:
        other = partner_of(agent)
        if other is None:
            continue
        pair_key = tuple(sorted((str(agent.agent_id), str(other.agent_id))))
        if pair_key in processed:
            continue
        processed.add(pair_key)
        if rng.random() < policy.accident_divorce:
            tax = 0.25 * policy.exit_tax * 0.5
            agent.resource_level = max(0.0, float(agent.resource_level) - tax)
            other.resource_level = max(0.0, float(other.resource_level) - tax)
            dissolve(agent, reason="accident", policy=policy)
            continue
        if not policy.implicit_leave:
            continue
        stress_a = max(0.0, min(1.0, (2.2 - float(agent.resource_level)) / 2.2))
        stress_b = max(0.0, min(1.0, (2.2 - float(other.resource_level)) / 2.2))
        stress = 0.5 * (stress_a + stress_b)
        strength = float(getattr(agent, "bond_strength", 0.0))
        for actor, peer in ((agent, other), (other, agent)):
            leave_p = (
                (1.0 - gene_value(actor, "fidelity", 0.5))
                * (0.04 + stress)
                * (1.0 - 0.5 * strength)
            )
            if rng.random() < leave_p:
                tax = policy.exit_tax * gene_value(actor, "fidelity", 0.5) * 0.5
                actor.resource_level = max(0.0, float(actor.resource_level) - tax)
                peer.resource_level = max(0.0, float(peer.resource_level) - 0.25 * tax)
                dissolve(actor, reason="leave", policy=policy)
                break


def _maybe_bond(living: Sequence[Any], policy: UnionPolicy, rng: random.Random) -> None:
    if not policy.pairing_enabled or not policy.implicit_pairing:
        return
    singles = [agent for agent in living if getattr(agent, "partner_id", None) is None]
    rng.shuffle(singles)
    claimed = set()
    for agent in singles:
        agent_id = getattr(agent, "agent_id", None)
        if agent_id in claimed or getattr(agent, "partner_id", None) is not None:
            continue
        candidates = [
            other
            for other in _neighbors(agent, policy.social_range, singles)
            if getattr(other, "agent_id", None) not in claimed
            and getattr(other, "partner_id", None) is None
        ]
        if not candidates:
            continue
        other = min(candidates, key=lambda cand: _distance(agent, cand))
        if rng.random() < _pairing_probability(agent, other, policy):
            if form_bond(agent, other, policy):
                claimed.add(agent.agent_id)
                claimed.add(other.agent_id)


def tick_union_bonds(environment: Any) -> None:
    """Per-step strength, accidental divorce, implicit leave, implicit pairing."""
    policy = get_union_policy(environment)
    if policy is None or not policy.enabled:
        return
    living = [agent for agent in getattr(environment, "alive_agent_objects", []) if getattr(agent, "alive", False)]
    for agent in living:
        ensure_bond_state(agent)
    rng = getattr(environment, "union_rng", None) or getattr(environment, "intrinsic_evolution_rng", None)
    if rng is None:
        rng = random.Random()
        environment.union_rng = rng
    _update_live_bonds(living, policy)
    if policy.pairing_enabled:
        _maybe_exit(living, policy, rng)
        living = [agent for agent in living if getattr(agent, "alive", False)]
        _maybe_bond(living, policy, rng)
    mark_dirty = getattr(environment, "mark_positions_dirty", None)
    if callable(mark_dirty):
        mark_dirty()


def try_bond_action(agent: Any) -> Dict[str, Any]:
    """Agent-chosen bond: nearest eligible single in ``social_range``."""
    policy = get_union_policy(agent)
    if policy is None or not policy.pairing_enabled:
        return {"success": False, "error": "Pairing is disabled", "details": {}}
    ensure_bond_state(agent)
    if getattr(agent, "partner_id", None) is not None:
        return {"success": False, "error": "Agent is already bonded", "details": {}}
    env = getattr(agent, "environment", None)
    if env is None:
        return {"success": False, "error": "No environment", "details": {}}
    living = [other for other in env.alive_agent_objects if other is not agent and getattr(other, "alive", False)]
    candidates = [
        other
        for other in _neighbors(agent, policy.social_range, living)
        if getattr(other, "partner_id", None) is None
    ]
    if not candidates:
        return {
            "success": False,
            "error": "No eligible partner in range",
            "details": {"social_range": policy.social_range},
        }
    other = min(candidates, key=lambda cand: _distance(agent, cand))
    rng = getattr(env, "union_rng", None) or random
    if rng.random() >= _pairing_probability(agent, other, policy):
        return {
            "success": False,
            "error": "Pairing probability rejected the bond",
            "details": {"target_id": other.agent_id},
        }
    if not form_bond(agent, other, policy):
        return {
            "success": False,
            "error": "Bond refused (cost or lock)",
            "details": {"target_id": other.agent_id, "bonding_cost": policy.bonding_cost},
        }
    return {
        "success": True,
        "error": None,
        "details": {"target_id": other.agent_id, "bonding_cost": policy.bonding_cost},
    }


def try_leave_action(agent: Any) -> Dict[str, Any]:
    """Agent-chosen leave: exit tax scaled by fidelity, then clear both sides."""
    policy = get_union_policy(agent)
    ensure_bond_state(agent)
    other = partner_of(agent)
    if other is None:
        return {"success": False, "error": "Agent is not bonded", "details": {}}
    if policy is None:
        dissolve(agent, reason="leave")
        return {"success": True, "error": None, "details": {"target_id": other.agent_id}}
    tax = policy.exit_tax * gene_value(agent, "fidelity", 0.5) * 0.5
    agent.resource_level = max(0.0, float(agent.resource_level) - tax)
    other.resource_level = max(0.0, float(other.resource_level) - 0.25 * tax)
    duration = dissolve(agent, reason="leave", policy=policy)
    return {
        "success": True,
        "error": None,
        "details": {"target_id": other.agent_id, "exit_tax": tax, "pair_duration": duration},
    }


def equalize_with_partner(agent: Any) -> Optional[Dict[str, Any]]:
    """Loose equalization toward the poorer partner. None if not bonded."""
    policy = get_union_policy(agent)
    other = partner_of(agent)
    if other is None or policy is None:
        return None
    poorer, richer = (agent, other) if float(agent.resource_level) < float(other.resource_level) else (other, agent)
    gap = float(richer.resource_level) - float(poorer.resource_level)
    transfer = policy.equalize_rate * gap
    richer.resource_level = float(richer.resource_level) - transfer
    poorer.resource_level = float(poorer.resource_level) + transfer
    return {
        "success": True,
        "error": None,
        "details": {
            "target_id": other.agent_id,
            "amount_shared": transfer,
            "equalize": True,
        },
    }


def freeze_learning_genes(chromosome: HyperparameterChromosome) -> HyperparameterChromosome:
    """Mark Chromosome A loci as not evolvable; keep union/share/goal loci open."""
    frozen = set(LEARNING_GENE_NAMES)
    genes: List[HyperparameterGene] = []
    for gene in chromosome.genes:
        if gene.name in frozen and gene.evolvable:
            genes.append(
                HyperparameterGene(
                    name=gene.name,
                    value_type=gene.value_type,
                    value=gene.value,
                    min_value=gene.min_value,
                    max_value=gene.max_value,
                    default=gene.default,
                    evolvable=False,
                    mutation_scale=gene.mutation_scale,
                    mutation_probability=gene.mutation_probability,
                    mutation_strategy=gene.mutation_strategy,
                )
            )
        else:
            genes.append(gene)
    return HyperparameterChromosome(genes=tuple(genes))


def snapshot_union_metrics(environment: Any) -> Dict[str, Any]:
    """One-step snapshot used by the Layer C runner."""
    living = [agent for agent in getattr(environment, "alive_agent_objects", []) if getattr(agent, "alive", False)]
    paired_energy: List[float] = []
    solo_energy: List[float] = []
    extractions: List[float] = []
    strengths: List[float] = []
    durations: List[int] = []
    gene_sums = {name: 0.0 for name in UNION_GENE_NAMES + ("share_weight",)}
    seen = set()
    for agent in living:
        ensure_bond_state(agent)
        energy = float(getattr(agent, "resource_level", 0.0))
        other = partner_of(agent)
        if other is None:
            solo_energy.append(energy)
        else:
            paired_energy.append(energy)
            pair_key = tuple(sorted((str(agent.agent_id), str(other.agent_id))))
            if pair_key not in seen:
                seen.add(pair_key)
                extractions.append(
                    abs(energy - float(other.resource_level))
                    * abs(gene_value(agent, "pair_commitment", 0.5) - gene_value(other, "pair_commitment", 0.5))
                )
                strengths.append(float(getattr(agent, "bond_strength", 0.0)))
                durations.append(int(getattr(agent, "pair_age", 0)))
        for name in gene_sums:
            gene_sums[name] += gene_value(agent, name, 0.0)
    n = max(1, len(living))
    paired_n = len(paired_energy)
    solo_mean = sum(solo_energy) / len(solo_energy) if solo_energy else None
    paired_mean = sum(paired_energy) / paired_n if paired_energy else None
    synergy = None
    if paired_mean is not None and solo_mean is not None and solo_mean > 1e-9:
        synergy = paired_mean / solo_mean
    policy = get_union_policy(environment)
    return {
        "n_alive": len(living),
        "paired_frac": paired_n / n if living else 0.0,
        "synergy_index": synergy,
        "mean_energy": (sum(paired_energy) + sum(solo_energy)) / n if living else 0.0,
        "mean_extraction": sum(extractions) / len(extractions) if extractions else None,
        "mean_bond_strength": sum(strengths) / len(strengths) if strengths else None,
        "mean_pair_age": sum(durations) / len(durations) if durations else None,
        "leave_events": getattr(policy, "leave_events", 0) if policy else 0,
        "bond_events": getattr(policy, "bond_events", 0) if policy else 0,
        "gene_means": {name: gene_sums[name] / n for name in gene_sums},
    }
