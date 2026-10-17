"""How agents appraise units, accept offers, leave, and choose effort.

Agents value a membership with
:func:`farm.core.cooperation.goals.value_of_membership`; this module builds
the :class:`~farm.core.cooperation.goals.MembershipAppraisal` inputs.

* **Prospective unit** -- steady-state expectations from income *signals*
  (a partner's income EMA seen through multiplicative log-normal noise):
  pooled inflow ``P = scope * sum(y)``, surplus from
  :meth:`~farm.core.cooperation.unit.UnitEconomics.surplus`, own share by
  rank weight, minus own contribution; parent dues reduce what comes back.
* **Current unit** -- the realised running net benefit from the member's
  ledger (payouts plus aid minus contributions), so members learn from what
  the unit actually did for them rather than from what they expected.
* **Insurance** -- expected support on the agent's own shock ticks:
  ``shock_prob * scope * share * sum(partner income)``.

Effort: with marginal retention ``m`` (the fraction of an extra unit of own
income that comes back), the privately optimal effort under a quadratic cost
is ``m * skill / effort_cost``; diligence blends it toward full effort. Broad
scope therefore invites shirking by low-diligence members -- the free-rider
cost of pooling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from farm.core.cooperation import (
    CooperationRegistry,
    CooperationSettings,
    CooperativeUnit,
    MembershipAppraisal,
    UnitEconomics,
    ValuationScales,
    rank_weights,
    value_of_membership,
)
from farm.experiments.cooperation.agents import CoopAgent
from farm.experiments.cooperation.config import DecisionConfig, WorldConfig


def scales_for(decisions: DecisionConfig) -> ValuationScales:
    return ValuationScales(
        horizon=decisions.horizon,
        fit_scale=decisions.fit_scale,
        affinity_scale=decisions.affinity_scale,
    )


def income_signal(observer_rng: np.random.Generator, target: CoopAgent, noise: float) -> float:
    """What an outsider believes about ``target``'s income."""
    if noise <= 0.0:
        return target.income_ema
    return target.income_ema * float(np.exp(observer_rng.normal(0.0, noise)))


def expected_flows(
    y_self: float,
    y_others: Sequence[float],
    settings: CooperationSettings,
    economics: UnitEconomics,
    self_rank: int,
    shock_prob: float,
    parent_scope: float = 0.0,
    reserve_target: float = 0.0,
    horizon: float = 1.0,
) -> tuple[float, float, float]:
    """Steady-state ``(net_transfer, insurance, gross_flow)`` for one member.

    ``gross_flow`` is pooled inflow plus surplus, before any parent dues.
    The member's share of building the unit's reserve
    (``reserve_target * n``) is amortised over ``horizon`` steps and charged
    against the net transfer: a new unit pays little out until its reserve
    is full, and a forward-looking agent expects that.
    """
    n = 1 + len(y_others)
    weights = rank_weights(n, settings.symmetry)
    share = weights[min(self_rank, n - 1)]
    pooled = settings.scope * (y_self + sum(y_others))
    gross = pooled + economics.surplus(pooled, n)
    returned = gross * (1.0 - parent_scope)
    reserve_charge = share * reserve_target * n / horizon if horizon > 0 else 0.0
    net = share * returned - settings.scope * y_self - reserve_charge
    insurance = shock_prob * settings.scope * share * sum(y_others) * (1.0 - parent_scope)
    return net, insurance, gross


def marginal_retention(agent_id: str, registry: CooperationRegistry | None, economics: UnitEconomics) -> float:
    """Fraction of an extra unit of income that comes back to the agent."""
    if registry is None:
        return 1.0
    units = registry.units_of(agent_id)
    if not units:
        return 1.0
    m = 1.0
    per_unit = 1.0 / len(units)
    for unit in units:
        n = unit.n_participants
        share = unit.weights().get(agent_id, 0.0)
        parent_scope = 0.0
        if unit.parent_id is not None and unit.parent_id in registry.units:
            parent = registry.units[unit.parent_id]
            if parent.is_active:
                parent_scope = parent.settings.scope
        back = share * (1.0 + economics.synergy * (1.0 - 1.0 / n)) * (1.0 - parent_scope)
        m += unit.settings.scope * per_unit * (back - 1.0)
    return max(0.0, m)


def choose_effort(agent: CoopAgent, retention: float, world: WorldConfig) -> float:
    rational = min(1.0, max(world.min_effort, retention * agent.skill / world.effort_cost))
    return agent.diligence + (1.0 - agent.diligence) * rational


def noisy_accept(value: float, threshold: float, noise: float, rng: np.random.Generator) -> bool:
    """Logistic-noise acceptance: ``value + eps > threshold``."""
    eps = float(rng.logistic(0.0, noise)) if noise > 0.0 else 0.0
    return value + eps > threshold


def appraise_prospect(
    agent: CoopAgent,
    partners: Sequence[CoopAgent],
    settings: CooperationSettings,
    economics: UnitEconomics,
    decisions: DecisionConfig,
    world: WorldConfig,
    rng: np.random.Generator,
    self_rank: int,
    parent_scope: float = 0.0,
    parent_is_own: bool = False,
    reserve_target: float = 0.0,
) -> tuple[float, float]:
    """``(value, expected_net)`` to ``agent`` of a unit of itself and ``partners``.

    ``expected_net`` seeds the member's running benefit estimate if the unit
    forms (see :class:`~farm.core.cooperation.unit.MemberLedger`).
    """
    signals = [income_signal(rng, p, decisions.income_signal_noise) for p in partners]
    net, insurance, gross = expected_flows(
        agent.income_ema,
        signals,
        settings,
        economics,
        self_rank,
        world.shock_prob,
        parent_scope,
        reserve_target,
        decisions.horizon,
    )
    compat = float(np.mean([agent.compatibility(p) for p in partners])) if partners else 0.0
    parent_gain = parent_scope * gross if parent_is_own else 0.0
    appraisal = MembershipAppraisal(
        net_transfer=net, insurance=insurance, compatibility=compat, parent_gain=parent_gain
    )
    return value_of_membership(agent.goals, settings, appraisal, scales_for(decisions)), net


def appraise_current(
    agent: CoopAgent,
    unit: CooperativeUnit,
    agents: dict[str, CoopAgent],
    registry: CooperationRegistry,
    decisions: DecisionConfig,
    world: WorldConfig,
) -> float:
    """Value to ``agent`` of staying in ``unit``, from realised flows."""
    ledger = unit.members[agent.agent_id]
    partners = [agents[pid] for pid in unit.members if pid != agent.agent_id and pid in agents]
    share = unit.weights().get(agent.agent_id, 0.0)
    parent_scope = 0.0
    parent_gain = 0.0
    if unit.parent_id is not None and unit.parent_id in registry.units:
        parent = registry.units[unit.parent_id]
        if parent.is_active:
            parent_scope = parent.settings.scope
            recent = unit.history[-10:]
            parent_gain = float(np.mean([r.upward for r in recent])) if recent else 0.0
    insurance = (
        world.shock_prob * unit.settings.scope * share * sum(p.income_ema for p in partners) * (1.0 - parent_scope)
    )
    compat = float(np.mean([agent.compatibility(p) for p in partners])) if partners else 0.0
    appraisal = MembershipAppraisal(
        net_transfer=ledger.benefit_ema,
        insurance=insurance,
        compatibility=compat,
        parent_gain=parent_gain,
    )
    return value_of_membership(agent.goals, unit.settings, appraisal, scales_for(decisions))
