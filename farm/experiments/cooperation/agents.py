"""Agents of the cooperation experiment and their per-agent random streams.

Each agent owns a ``numpy.random.Generator`` used *only* for its income
shocks, seeded from ``(seed, stream, slot)``. A founder's shock sequence is
therefore identical in a cooperative run and in its solo twin, which is what
makes the per-founder welfare comparison a paired one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from farm.core.cooperation import MemberGoals, Motive

FOUNDER_STREAM: int = 1
BIRTH_STREAM: int = 2


def income_stream(seed: int, founder: bool, slot: int) -> np.random.Generator:
    """Independent shock stream for one agent slot."""
    return np.random.default_rng([int(seed), FOUNDER_STREAM if founder else BIRTH_STREAM, int(slot)])


def unit_vector(rng: np.random.Generator, dims: int) -> np.ndarray:
    v = rng.normal(size=dims)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else np.eye(dims)[0]


@dataclass
class CoopAgent:
    """One agent: traits, energy, private goals and running statistics."""

    agent_id: str
    slot: int
    founder: bool
    skill: float
    diligence: float
    motive: Motive
    affinity: np.ndarray
    energy: float
    born_tick: int
    rng: np.random.Generator = field(repr=False)
    parent_id: str | None = None
    goals: MemberGoals = field(init=False, repr=False)
    alive: bool = True
    death_tick: int | None = None
    death_cause: str | None = None
    income_ema: float = 0.0
    offspring: int = 0
    ticks_alive: int = 0
    energy_sum: float = 0.0
    effort_sum: float = 0.0
    member_ticks: int = 0
    gross_income: float = 0.0
    pooled: float = 0.0
    received: float = 0.0
    exit_paid: float = 0.0
    last_effort: float = 1.0

    def __post_init__(self) -> None:
        self.goals = MemberGoals.single(self.motive)

    def age(self, tick: int) -> int:
        return int(tick) - self.born_tick

    def lifespan(self, end_tick: int) -> int:
        end = self.death_tick if self.death_tick is not None else end_tick
        return max(0, int(end) - self.born_tick)

    def compatibility(self, other: CoopAgent) -> float:
        """Cosine similarity of affinity vectors, in ``[-1, 1]``."""
        return float(np.clip(np.dot(self.affinity, other.affinity), -1.0, 1.0))

    def observe_income(self, gross: float, alpha: float) -> None:
        self.income_ema = (1.0 - alpha) * self.income_ema + alpha * gross

    def record_tick(self, effort: float, in_unit: bool) -> None:
        self.ticks_alive += 1
        self.energy_sum += self.energy
        self.effort_sum += effort
        self.last_effort = effort
        if in_unit:
            self.member_ticks += 1

    def to_record(self, end_tick: int) -> dict:
        lifespan = self.lifespan(end_tick)
        return {
            "agent_id": self.agent_id,
            "slot": self.slot,
            "founder": self.founder,
            "parent_id": self.parent_id,
            "motive": self.motive.value,
            "skill": self.skill,
            "diligence": self.diligence,
            "born_tick": self.born_tick,
            "death_tick": self.death_tick,
            "death_cause": self.death_cause,
            "censored": self.death_tick is None,
            "lifespan": lifespan,
            "offspring": self.offspring,
            "mean_energy": self.energy_sum / self.ticks_alive if self.ticks_alive else float("nan"),
            "mean_effort": self.effort_sum / self.ticks_alive if self.ticks_alive else float("nan"),
            "member_share": self.member_ticks / self.ticks_alive if self.ticks_alive else 0.0,
            "gross_income": self.gross_income,
            "pooled": self.pooled,
            "received": self.received,
            "exit_paid": self.exit_paid,
            "final_energy": self.energy,
        }
