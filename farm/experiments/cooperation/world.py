"""Income, costs and shocks for the cooperation experiment.

Two kinds of risk, because pooling helps against only one of them:

* **Idiosyncratic shocks** -- each tick an agent earns nothing with
  probability ``shock_prob``, drawn from its own stream. A partner's income
  can carry it through (risk pooling).
* **Droughts** -- every agent's income is multiplied by ``drought_factor``
  for ``drought_length`` ticks every ``drought_period`` ticks. Everyone is
  hit at once, so pooling cannot smooth it; only a reserve can.

Crowding: above ``carrying_capacity`` living agents, income is scaled by
``carrying_capacity / N`` (a fixed resource flow shared by more mouths).

The drought phase is drawn from the run seed and is shared by a cooperative
run and its solo twin.
"""

from __future__ import annotations

import numpy as np

from farm.experiments.cooperation.agents import CoopAgent
from farm.experiments.cooperation.config import WorldConfig

DROUGHT_STREAM: int = 6


class IncomeWorld:
    """Deterministic drought schedule plus per-agent shock draws."""

    def __init__(self, config: WorldConfig, seed: int) -> None:
        self.config = config
        rng = np.random.default_rng([int(seed), DROUGHT_STREAM])
        self.phase = int(rng.integers(config.drought_period))

    def is_drought(self, tick: int) -> bool:
        c = self.config
        if c.drought_length <= 0:
            return False
        return (int(tick) + self.phase) % c.drought_period >= c.drought_period - c.drought_length

    def multiplier(self, tick: int) -> float:
        return self.config.drought_factor if self.is_drought(tick) else 1.0

    def crowding(self, n_alive: int) -> float:
        """Income scaling from population pressure, in ``(0, 1]``."""
        k = self.config.carrying_capacity
        return 1.0 if n_alive <= k else k / float(n_alive)

    def draw_income(self, agent: CoopAgent, effort: float, tick: int, crowding: float = 1.0) -> float:
        """Gross income for one tick. Always consumes exactly one draw."""
        shocked = agent.rng.random() < self.config.shock_prob
        if shocked:
            return 0.0
        return agent.skill * effort * self.multiplier(tick) * crowding

    def cost(self, effort: float) -> float:
        """Energy spent this tick: metabolism plus quadratic effort cost."""
        c = self.config
        return c.metabolism + 0.5 * c.effort_cost * effort * effort

    def expected_multiplier(self) -> float:
        """Long-run mean of shock and drought scaling (for appraisals)."""
        c = self.config
        drought_share = c.drought_length / c.drought_period if c.drought_period else 0.0
        return (1.0 - c.shock_prob) * (1.0 - drought_share * (1.0 - c.drought_factor))
