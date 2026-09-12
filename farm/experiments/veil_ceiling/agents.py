"""Agent state, per-agent event ledger, and the inheritance adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

from farm.experiments.veil_ceiling.learner import MLPQLearner


class VeilAction(IntEnum):
    MOVE = 0
    GATHER = 1
    OVER_HARVEST = 2
    PASS = 3


N_ACTIONS = len(VeilAction)

# Observation vector layout. Position is deliberately absent so the only
# spatial information available to the learner is the coarse region id.
FEATURE_NAMES = (
    "energy",
    "node_in_range",
    "node_amount",
    "node_above_threshold",
    "gather_yield",
    "over_harvest_yield",
    "healthy_node_distance",
    "crowding",
    "cue",
    "region_0",
    "region_1",
    "region_2",
    "region_3",
    "region_heldout",
    "age",
)
N_FEATURES = len(FEATURE_NAMES)
CUE_FEATURE_INDEX = FEATURE_NAMES.index("cue")
REGION_FEATURE_OFFSET = FEATURE_NAMES.index("region_0")

# Ledger axes: [stat][phase][region_kind][monitored][cue]
LEDGER_STATS = (
    "ticks",
    "opportunities",
    "defections",
    "gathers",
    "moves",
    "passes",
    "penalties",
    "energy_sum",
)
PHASES = ("train", "eval")
REGION_KINDS = ("train", "heldout")


@dataclass
class AgentLedger:
    """Counts of what an agent did, split by phase, region, monitor state and cue."""

    counts: np.ndarray = field(default_factory=lambda: np.zeros((len(LEDGER_STATS), 2, 2, 2, 2), dtype=float))

    def record(
        self,
        *,
        phase: int,
        region_kind: int,
        monitored: int,
        cue: int,
        opportunity: bool,
        action: int,
        defected: bool,
        gathered: bool,
        penalised: bool,
        energy: float,
    ) -> None:
        c = self.counts[:, phase, region_kind, monitored, cue]
        c[0] += 1
        if opportunity:
            c[1] += 1
        if defected:
            c[2] += 1
        elif gathered:
            c[3] += 1
        elif action == VeilAction.MOVE:
            c[4] += 1
        elif action == VeilAction.PASS:
            c[5] += 1
        if penalised:
            c[6] += 1
        c[7] += energy

    def flatten(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for si, stat in enumerate(LEDGER_STATS):
            for pi, phase in enumerate(PHASES):
                for ri, region in enumerate(REGION_KINDS):
                    for m in (0, 1):
                        for cue in (0, 1):
                            out[f"{stat}__{phase}__{region}__m{m}__c{cue}"] = float(self.counts[si, pi, ri, m, cue])
        return out


@dataclass
class VeilAgent:
    """One spatially embedded learning agent."""

    agent_id: int
    x: int
    y: int
    energy: float
    learner: MLPQLearner
    generation: int
    born_tick: int
    parent_id: int | None = None
    age: int = 0
    alive: bool = True
    death_tick: int | None = None
    offspring: int = 0
    energy_gained: float = 0.0
    penalties_paid: float = 0.0
    ledger: AgentLedger = field(default_factory=AgentLedger)
    pending_state: np.ndarray | None = None
    pending_action: int | None = None
    pending_reward: float = 0.0

    def __post_init__(self) -> None:
        # Adapter so ``apply_lamarckian_policy_warmstart`` finds the learner at
        # ``agent.behavior.decision_module.algorithm`` like a core AgentFarm agent.
        self.behavior = SimpleNamespace(decision_module=SimpleNamespace(algorithm=self.learner))

    @property
    def resource_level(self) -> float:
        return self.energy

    def summary_row(self, current_tick: int) -> dict[str, float]:
        lifespan = (self.death_tick if self.death_tick is not None else current_tick) - self.born_tick
        row: dict[str, float] = {
            "agent_id": self.agent_id,
            "parent_id": -1 if self.parent_id is None else self.parent_id,
            "generation": self.generation,
            "born_tick": self.born_tick,
            "death_tick": -1 if self.death_tick is None else self.death_tick,
            "alive_at_end": int(self.alive),
            "lifespan": max(lifespan, 0),
            "offspring": self.offspring,
            "energy_gained": self.energy_gained,
            "penalties_paid": self.penalties_paid,
            "final_energy": self.energy,
        }
        row.update(self.ledger.flatten())
        return row


def summarise_agents(agents: list[VeilAgent], current_tick: int) -> list[dict[str, float]]:
    return [a.summary_row(current_tick) for a in agents]
