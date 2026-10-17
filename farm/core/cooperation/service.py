"""Cooperation in the spatial AgentFarm world.

:class:`CooperationService` connects the institution-agnostic
:class:`~farm.core.cooperation.registry.CooperationRegistry` to the
environment, the agents' brains and their chromosomes. The environment
creates one when ``SimulationConfig.cooperation.enabled`` is true and calls
its hooks; nothing here imports torch, so it can be tested on its own.

What the agent controls
    Two learned actions. ``bond`` opens the agent to a unit for
    ``offer_window`` steps and looks for another open agent within
    ``bond_range``; two open agents near each other form a unit, or one joins
    the other's unit when it has room. ``leave`` exits the agent's least
    rewarding unit. Whether and when to take either is up to the agent's
    policy; their innate prior weights are the ``bond_weight`` and
    ``leave_weight`` genes.

What the institution enforces
    The settings of the unit, taken from its founders' heritable settings
    loci (``terms="mean"`` averages them): members pool ``scope`` of every
    unit of resource they gather, the pool earns a synergy surplus, keeps a
    reserve, pays emergency aid to members below the aid threshold and pays
    out the rest by rank weights. Leaving forfeits ``exit_cost`` of the
    leaver's resources when enforced (probability ``recognition``);
    recognition also blocks outside offers to members of exclusive units.

What the agent feels
    Pooling shows up as a smaller resource gain on ``gather``. Payouts, aid
    and dissolution shares are added to the agent's resources between its
    actions; the reward's resource term spans everything since the agent's
    previous action, so they count once, at the agent's own
    ``reward_resource_weight``. Members
    also receive a per-step membership reward from their motive loci: how
    well the unit's settings fit their motives, and how like-minded their
    partners are.

What is recorded
    A per-step time series, every unit with its failure-mode diagnosis, and
    a summary, written to ``config.output_dir`` when the run ends.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from farm.core.cooperation.actions import COOPERATION_ACTION_NAMES, CooperationActionType
from farm.core.cooperation.config import CooperationConfig
from farm.core.cooperation.diagnostics import DEFAULT_THRESHOLDS, diagnose_all, mode_prevalence
from farm.core.cooperation.genes import (
    COOPERATION_GENE_NAMES,
    MOTIVE_GENE_NAMES,
    blended_settings,
    extend_chromosome,
    gene_values,
    goals_from_values,
    has_cooperation_genes,
    motive_similarity,
    sample_gene_values,
    settings_from_values,
)
from farm.core.cooperation.goals import MemberGoals, MembershipAppraisal, UnitGoals, value_of_membership
from farm.core.cooperation.registry import CooperationRegistry
from farm.core.cooperation.settings import CooperationSettings, Entry, rank_weights
from farm.core.cooperation.unit import CooperativeUnit

#: Smoothing of each agent's gathered-income estimate (per step).
INCOME_EMA_ALPHA: float = 0.05

#: Offsets that keep the service's random streams apart from the world's.
_NP_STREAM = 0xC0
_PY_STREAM = 0xC1

TIMESERIES_FILE = "cooperation_timeseries.csv"
UNITS_FILE = "cooperation_units.csv"
SUMMARY_FILE = "cooperation_summary.json"
EVENTS_FILE = "cooperation_events.jsonl"


def _parent_ids(agent: Any) -> list[str]:
    state = getattr(agent, "state", None)
    ids = getattr(state, "parent_ids", None)
    if ids is None:
        ids = getattr(agent, "parent_ids", None)
    return [str(i) for i in (ids or [])]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


class _AgentGenes:
    """Cached phenotype of one chromosome (refreshed when it is replaced)."""

    __slots__ = ("affinity_weight", "chromosome", "goals", "offered", "values")

    def __init__(self, chromosome: Any, values: dict[str, float], offered: CooperationSettings, goals: MemberGoals):
        self.chromosome = chromosome
        self.values = values
        self.offered = offered
        self.goals = goals
        self.affinity_weight = goals.component_weights()["affinity"]


class CooperationService:
    """Cooperation hooks for one :class:`~farm.core.environment.Environment`."""

    def __init__(self, config: CooperationConfig, environment: Any = None, seed: int | None = None) -> None:
        if not isinstance(config, CooperationConfig):
            raise TypeError("config must be a CooperationConfig")
        self.config = config
        self.environment = environment
        base_seed = config.seed if config.seed is not None else seed
        self._np_rng = np.random.default_rng(None if base_seed is None else [int(base_seed), _NP_STREAM])
        self._py_rng = random.Random(None if base_seed is None else int(base_seed) * 1_000_003 + _PY_STREAM)
        self.registry = CooperationRegistry(
            economics=config.economics(),
            unit_goals=UnitGoals(reserve_target=config.reserve_target),
            rng=self._np_rng,
            keep_events=config.keep_events,
        )
        self._fixed_settings = config.unit_settings()
        self._family_settings = config.family_unit_settings()
        self._scales = config.scales()
        self._agents: dict[str, Any] = {}
        self._genes: dict[str, _AgentGenes] = {}
        self._open_until: dict[str, int] = {}
        self._income_ema: dict[str, float] = {}
        self._income_step: dict[str, float] = defaultdict(float)
        self._pending_income: dict[str, float] = defaultdict(float)
        self._pending_reward: dict[str, float] = defaultdict(float)
        self.counters: Counter = Counter()
        self._step_counters: Counter = Counter()
        self.timeseries: list[dict[str, Any]] = []
        self.founders_seen = 0
        self._closed = False

    # ------------------------------------------------------------- basics
    @property
    def active(self) -> bool:
        """Units form and pool (false in placebo mode)."""
        return self.config.active

    @property
    def kind(self) -> str:
        return self.config.kind

    def current_step(self) -> int:
        return int(getattr(self.environment, "time", 0) or 0)

    def _count(self, key: str, amount: float = 1) -> None:
        self.counters[key] += amount
        self._step_counters[key] += amount

    def agent(self, agent_id: str) -> Any:
        return self._agents.get(agent_id)

    def _alive(self, agent: Any) -> bool:
        return agent is not None and bool(getattr(agent, "alive", True))

    # --------------------------------------------------------------- genes
    def genes_of(self, agent: Any) -> _AgentGenes:
        """Settings offered and motives of ``agent``, from its current chromosome."""
        chromosome = getattr(agent, "hyperparameter_chromosome", None)
        cached = self._genes.get(agent.agent_id)
        if cached is not None and cached.chromosome is chromosome:
            return cached
        values = gene_values(chromosome)
        offered = self._fixed_settings or settings_from_values(values, self.kind)
        genes = _AgentGenes(chromosome, values, offered, goals_from_values(values))
        self._genes[agent.agent_id] = genes
        return genes

    def _founder_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        if self.config.randomize_founder_genes:
            values.update(sample_gene_values(self._py_rng))
        values.update({k: float(v) for k, v in self.config.founder_gene_values.items()})
        return values

    def _ensure_genes(self, agent: Any) -> None:
        chromosome = getattr(agent, "hyperparameter_chromosome", None)
        if chromosome is None or has_cooperation_genes(chromosome):
            return
        extended = extend_chromosome(chromosome, self._founder_values())
        if not self.config.evolve_genes:
            names = set(COOPERATION_GENE_NAMES)
            extended = type(extended)(
                genes=tuple(dataclasses.replace(g, evolvable=False) if g.name in names else g for g in extended.genes)
            )
        agent.hyperparameter_chromosome = extended
        self.founders_seen += 1

    def refresh_action_weights(self, agent: Any) -> None:
        """Set the ``bond``/``leave`` prior weights from the agent's propensity genes.

        The weights are expressed relative to the agent's other actions so
        that repeated refreshes are idempotent: with propensities ``b`` and
        ``l`` the normalised weights are ``b / (1 + b + l)`` and
        ``l / (1 + b + l)``.
        """
        actions = getattr(agent, "actions", None)
        if not actions:
            return
        coop = {a.name: a for a in actions if a.name in COOPERATION_ACTION_NAMES}
        if not coop:
            return
        values = self.genes_of(agent).values
        base_total = sum(max(0.0, a.weight) for a in actions if a.name not in COOPERATION_ACTION_NAMES)
        if base_total <= 0.0:
            base_total = 1.0
        for name, action in coop.items():
            action.weight = max(0.0, float(values[f"{name}_weight"])) * base_total
        total = sum(max(0.0, a.weight) for a in actions)
        if total > 0.0:
            for a in actions:
                a.weight = max(0.0, a.weight) / total

    # ---------------------------------------------------------- lifecycle
    def on_agent_added(self, agent: Any) -> None:
        """Register an agent: give founders their loci, apply propensities, enrol births."""
        agent_id = str(agent.agent_id)
        self._ensure_genes(agent)
        self._agents[agent_id] = agent
        self._income_ema.setdefault(agent_id, 0.0)
        self.refresh_action_weights(agent)
        if not self.active:
            return
        parents = [p for p in _parent_ids(agent) if p in self._agents]
        if parents:
            self._enrol_birth(parents[0], agent_id)

    def _enrol_birth(self, parent_id: str, child_id: str) -> None:
        step = self.current_step()
        if self.config.family_on_birth:
            for unit in self.registry.units_of(parent_id):
                if unit.settings.entry is Entry.BORN or unit.parent_id is not None:
                    continue
                if self.registry.nearest_born_ancestor(unit) is not None:
                    continue
                self.registry.form(self._family_settings, [child_id], step, children=[unit.unit_id])
                self._count("families_formed")
                self._count("born_in")
                return
        joined = self.registry.register_birth(parent_id, child_id, step)
        if joined:
            self._count("born_in", len(joined))

    def on_agent_removed(self, agent: Any, reason: str = "death") -> None:
        """Drop a dead agent from every unit; survivors split any dissolved pool."""
        self._drop(str(agent.agent_id), reason)

    def _drop(self, agent_id: str, reason: str = "death") -> None:
        """Forget ``agent_id`` everywhere: units, offers, pending credit, caches."""
        self._open_until.pop(agent_id, None)
        self._pending_income.pop(agent_id, None)
        self._pending_reward.pop(agent_id, None)
        self._income_step.pop(agent_id, None)
        self._agents.pop(agent_id, None)
        if self.active and self.registry.is_member(agent_id):
            payouts = self.registry.remove_agent(agent_id, self.current_step(), reason=reason)
            self._count("member_deaths")
            self._credit(payouts, "dissolution")
        self._genes.pop(agent_id, None)
        self._income_ema.pop(agent_id, None)

    # ---------------------------------------------------------------- flows
    def on_income(self, agent: Any, amount: float) -> float:
        """Pool the unit's share of ``amount`` gathered by ``agent``; return it."""
        agent_id = str(agent.agent_id)
        amount = float(amount)
        if amount > 0.0:
            self._income_step[agent_id] += amount
        if not self.active or amount <= 0.0 or not self.registry.is_member(agent_id):
            return 0.0
        pooled = float(self.registry.contribute(agent_id, amount, self.current_step()))
        self._count("pooled", pooled)
        return pooled

    def _credit(self, payouts: Mapping[str, float], source: str) -> float:
        credited = 0.0
        for agent_id, amount in payouts.items():
            if amount <= 0.0:
                continue
            agent = self._agents.get(agent_id)
            if not self._alive(agent):
                self._count("unclaimed", amount)
                continue
            agent.resource_level = float(agent.resource_level) + float(amount)
            self._pending_income[agent_id] += float(amount)
            credited += float(amount)
        if credited:
            self._count(f"credited_{source}", credited)
        return credited

    def consume_pending(self, agent_id: str) -> tuple[float, float]:
        """``(income, membership_reward)`` accrued since the agent's last reward.

        ``income`` (payouts, aid, dissolution shares) is already part of the
        agent's resource change over that interval; it is returned so the
        reward can remove it when ``credit_payouts`` is off.
        """
        return (
            float(self._pending_income.pop(agent_id, 0.0)),
            float(self._pending_reward.pop(agent_id, 0.0)),
        )

    def settle(self) -> None:
        """End-of-step pool dynamics, membership rewards, diagnostics and telemetry."""
        step = self.current_step()
        self._update_income_estimates()
        self._purge_dead()
        if self.active:
            holdings = {
                aid: float(self._agents[aid].resource_level)
                for aid in self.registry.members_in_any_unit()
                if aid in self._agents
            }
            payouts = self.registry.settle(step, holdings)
            self._credit(payouts, "payout")
            self._membership_rewards()
            if step % self.config.review_interval == 0:
                self._review(step)
        if step % self.config.record_interval == 0:
            self.timeseries.append(self._telemetry_row(step))
        self._step_counters.clear()
        expired = [aid for aid, until in self._open_until.items() if until < step]
        for aid in expired:
            del self._open_until[aid]

    def _purge_dead(self) -> None:
        """Forget agents whose death never reached :meth:`on_agent_removed`."""
        dead = {aid for aid, agent in self._agents.items() if not self._alive(agent)}
        dead |= {aid for aid in self.registry.members_in_any_unit() if aid not in self._agents}
        for agent_id in sorted(dead):
            self._drop(agent_id, "death")

    def _update_income_estimates(self) -> None:
        a = INCOME_EMA_ALPHA
        for agent_id in self._agents:
            income = self._income_step.get(agent_id, 0.0)
            self._income_ema[agent_id] = (1.0 - a) * self._income_ema.get(agent_id, 0.0) + a * income
        self._income_step.clear()

    def _like_mindedness(self, agent_id: str, others: Iterable[str]) -> float:
        me = self._agents.get(agent_id)
        if me is None:
            return 0.0
        goals = self.genes_of(me).goals
        sims = [motive_similarity(goals, self.genes_of(self._agents[o]).goals) for o in others if o in self._agents]
        return float(np.mean(sims)) if sims else 0.0

    def _membership_rewards(self) -> None:
        scale = self.config.membership_reward_scale
        if scale <= 0.0:
            return
        h = self._scales.horizon
        for unit in self.registry.active_units():
            members = list(unit.members)
            for agent_id in members:
                agent = self._agents.get(agent_id)
                if agent is None:
                    continue
                genes = self.genes_of(agent)
                fit = genes.goals.settings_fit(unit.settings)
                like = self._like_mindedness(agent_id, [m for m in members if m != agent_id])
                reward = (
                    scale
                    * (self._scales.fit_scale * fit + self._scales.affinity_scale * genes.affinity_weight * like)
                    / h
                )
                self._pending_reward[agent_id] += reward
                self._count("membership_reward", reward)

    def _review(self, step: int) -> None:
        """Diagnostic only: would each member's own motives want out, and is it priced in?

        Members' actual exits are decided by their policies. This records, in
        the units' ledgers, when a member's motive-weighted valuation of its
        realised flows is negative (exit desire) and when the enforced exit
        payment exceeds that loss (trapped), so the failure-mode diagnostics
        of the non-spatial model apply unchanged.
        """
        interval = self.config.review_interval
        for unit in self.registry.active_units():
            for agent_id, ledger in list(unit.members.items()):
                agent = self._agents.get(agent_id)
                if agent is None or ledger.steps < interval:
                    continue
                genes = self.genes_of(agent)
                like = self._like_mindedness(agent_id, [m for m in unit.members if m != agent_id])
                value = value_of_membership(
                    genes.goals,
                    unit.settings,
                    MembershipAppraisal(net_transfer=ledger.benefit_ema, compatibility=like),
                    self._scales,
                )
                if value >= 0.0:
                    continue
                price = unit.settings.expected_exit_payment(float(agent.resource_level))
                self.registry.record_exit_desire(
                    unit.unit_id, agent_id, blocked_by_price=value >= -price, steps=interval
                )

    # ------------------------------------------------------------ actions
    def is_open(self, agent_id: str, step: int | None = None) -> bool:
        step = self.current_step() if step is None else step
        return self._open_until.get(agent_id, -1) >= step

    def _candidates(self, agent: Any) -> list[tuple[Any, float]]:
        spatial = getattr(agent, "spatial_service", None)
        if spatial is None:
            return []
        try:
            nearby = spatial.get_nearby(agent.position, self.config.bond_range, ["agents"]).get("agents", [])
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return []
        partners = self.registry.partners_of(agent.agent_id)
        out: list[tuple[Any, float]] = []
        step = self.current_step()
        for other in nearby:
            other_id = getattr(other, "agent_id", None)
            if other_id is None or other_id == agent.agent_id or other_id in partners:
                continue
            if not self._alive(other) or other_id not in self._agents:
                continue
            if not self.is_open(other_id, step):
                continue
            out.append((other, _distance(agent.position, other.position)))
        out.sort(key=lambda pair: (pair[1], str(pair[0].agent_id)))
        return out[: self.config.max_candidates]

    def _units_to_vacate(self, agent_id: str, target: Any) -> list[CooperativeUnit] | None:
        """Units ``agent_id`` must leave to enter ``target``; ``None`` if impossible."""
        ok, reason = self.registry.can_join(agent_id, target)
        if ok:
            return []
        if reason != "exclusivity_limit":
            return None
        settings = target.settings if isinstance(target, CooperativeUnit) else target
        held = [u for u in self.registry.units_of(agent_id) if u.kind == settings.kind]
        limit = settings.max_memberships
        if any(u.settings.exclusive for u in held):
            exclusive = [u for u in held if u.settings.exclusive]
            rest = [u for u in held if not u.settings.exclusive]
            vacate = list(exclusive)
            if len(rest) >= limit:
                vacate += self._worst_units(agent_id, rest, len(rest) - limit + 1)
            return vacate
        return self._worst_units(agent_id, held, len(held) - limit + 1)

    @staticmethod
    def _worst_units(agent_id: str, units: Sequence[CooperativeUnit], k: int) -> list[CooperativeUnit]:
        ordered = sorted(units, key=lambda u: (u.members[agent_id].benefit_ema, u.unit_id))
        return ordered[: max(0, k)]

    def _expected_net(
        self, agent_id: str, partner_ids: Sequence[str], settings: CooperationSettings, rank: int
    ) -> float:
        n = 1 + len(partner_ids)
        weights = rank_weights(n, settings.symmetry)
        share = weights[min(rank, n - 1)]
        y_self = self._income_ema.get(agent_id, 0.0)
        pooled = settings.scope * (y_self + sum(self._income_ema.get(p, 0.0) for p in partner_ids))
        gross = pooled + self.registry.economics.surplus(pooled, n)
        return share * gross - settings.scope * y_self

    def _can_join_after(self, agent_id: str, target: Any, vacate: Sequence[CooperativeUnit]) -> bool:
        """Whether ``agent_id`` could enter ``target`` once it has left ``vacate``."""
        if isinstance(target, CooperativeUnit):
            if not target.is_active or agent_id in target.members or not target.has_capacity():
                return False
            settings = target.settings
        else:
            settings = target
        gone = {u.unit_id for u in vacate}
        held = [u for u in self.registry.units_of(agent_id) if u.kind == settings.kind and u.unit_id not in gone]
        limit = 1 if any(u.settings.exclusive for u in held) else settings.max_memberships
        return len(held) < limit

    def _leave_unit(
        self, agent: Any, unit: CooperativeUnit, step: int, reason: str, *, veto_checked: bool = False
    ) -> Any:
        holdings = max(0.0, float(agent.resource_level))
        result = self.registry.leave(unit.unit_id, agent.agent_id, step, holdings, veto_checked=veto_checked)
        if result.vetoed:
            self._count("exit_vetoed")
            return result
        if result.paid > 0.0:
            agent.resource_level = max(0.0, float(agent.resource_level) - result.paid)
            self._count("forfeits", result.paid)
        if result.enforced:
            self._count("exits_enforced")
        self._count(f"exits_{reason}")
        if result.dissolved:
            self._count("dissolved_by_exit")
        self._credit(result.payouts, "dissolution")
        return result

    def _plan(self, agent: Any, other: Any) -> tuple[str, Any, list[tuple[Any, list[CooperativeUnit]]]] | None:
        """How ``agent`` and the open ``other`` would cooperate, or ``None``.

        Returns ``(how, target, movers)``: ``how`` is ``"join_other"`` (agent
        joins other's unit), ``"join_own"`` (other joins agent's unit) or
        ``"form"`` (a new unit; ``target`` is its settings). ``movers`` lists
        who must first leave which units.
        """
        a_id, o_id = agent.agent_id, other.agent_id
        for unit in self.registry.units_of(o_id, self.kind):
            if unit.has_capacity() and a_id not in unit.members:
                vacate = self._units_to_vacate(a_id, unit)
                if vacate is not None and (not vacate or self.config.allow_switching):
                    return "join_other", unit, [(agent, vacate)] if vacate else []
        for unit in self.registry.units_of(a_id, self.kind):
            if unit.has_capacity() and o_id not in unit.members:
                vacate = self._units_to_vacate(o_id, unit)
                if vacate is not None and (not vacate or self.config.allow_switching):
                    return "join_own", unit, [(other, vacate)] if vacate else []
        settings = self._terms(other, agent)
        movers = []
        for party in (other, agent):
            vacate = self._units_to_vacate(party.agent_id, settings)
            if vacate is None:
                return None
            if vacate:
                if not self.config.allow_switching:
                    return None
                movers.append((party, vacate))
        return "form", settings, movers

    def _terms(self, initiator: Any, responder: Any) -> CooperationSettings:
        if self._fixed_settings is not None:
            return self._fixed_settings
        if self.config.terms == "initiator":
            return self.genes_of(initiator).offered
        return blended_settings([self.genes_of(initiator).values, self.genes_of(responder).values], self.kind)

    def bond(self, agent: Any) -> dict[str, Any]:
        """The ``bond`` action (see the module docstring)."""
        step = self.current_step()
        agent_id = agent.agent_id
        self._count("bond_attempts")
        if not self.active:
            self._count("bond_placebo")
            return {"success": False, "error": "placebo", "details": {"placebo": True}}
        self._open_until[agent_id] = step + self.config.offer_window
        candidates = self._candidates(agent)
        if not candidates:
            return {
                "success": False,
                "error": "no_open_partner_in_range",
                "details": {"open_until": self._open_until[agent_id]},
            }
        reasons: Counter = Counter()
        for other, dist in candidates:
            outcome = self._try_match(agent, other, step)
            if outcome.get("success"):
                outcome["details"]["distance"] = dist
                return outcome
            reasons[outcome.get("error", "failed")] += 1
        return {
            "success": False,
            "error": "no_match",
            "details": {"reasons": dict(reasons), "open_until": self._open_until.get(agent_id)},
        }

    def _try_match(self, agent: Any, other: Any, step: int) -> dict[str, Any]:
        plan = self._plan(agent, other)
        if plan is None:
            self._count("bond_infeasible")
            return {"success": False, "error": "infeasible", "details": {}}
        how, target, movers = plan
        # Everything that can stop the match is decided before anyone leaves
        # a unit, so a failed bond never breaks up existing units.
        for party, _ in movers:
            if self.registry.offer_blocked(party.agent_id, self.kind, step):
                self._count("offers_blocked")
                return {"success": False, "error": "offer_blocked", "details": {"blocked": party.agent_id}}
        for party, units in movers:
            if not self._can_join_after(party.agent_id, target, units):
                self._count("bond_infeasible")
                return {"success": False, "error": "infeasible", "details": {}}
        for party, units in movers:
            for unit in units:
                if self.registry.draw_veto(unit.unit_id, party.agent_id, step):
                    self._count("exit_vetoed")
                    return {"success": False, "error": "switch_vetoed", "details": {"unit_id": unit.unit_id}}
        for party, units in movers:
            for unit in units:
                self._leave_unit(party, unit, step, reason="switch", veto_checked=True)
        a_id, o_id = agent.agent_id, other.agent_id
        if how == "form":
            settings: CooperationSettings = target
            for party in (other, agent):
                ok, _ = self.registry.can_join(party.agent_id, settings)
                if not ok:
                    self._count("bond_infeasible")
                    return {"success": False, "error": "infeasible_after_switch", "details": {}}
            priors = {
                o_id: self._expected_net(o_id, [a_id], settings, 0),
                a_id: self._expected_net(a_id, [o_id], settings, 1),
            }
            unit = self.registry.form(settings, [o_id, a_id], step, priors=priors)
            self._count("units_formed")
        else:
            unit = target
            joiner = agent if how == "join_other" else other
            others = [m for m in unit.members if m != joiner.agent_id]
            prior = self._expected_net(joiner.agent_id, others, unit.settings, len(unit.members))
            if not self.registry.join(unit.unit_id, joiner.agent_id, step, prior=prior):
                self._count("bond_infeasible")
                return {"success": False, "error": "join_refused", "details": {}}
            self._count("units_joined")
        self._open_until.pop(a_id, None)
        self._open_until.pop(o_id, None)
        self._count("bond_success")
        return {
            "success": True,
            "error": None,
            "details": {
                "unit_id": unit.unit_id,
                "how": how,
                "target_id": o_id,
                "members": list(unit.members),
                "settings": unit.settings.to_dict(),
                "switched": [p.agent_id for p, _ in movers],
            },
        }

    def leave(self, agent: Any) -> dict[str, Any]:
        """The ``leave`` action: exit the agent's lowest-benefit unit."""
        step = self.current_step()
        agent_id = agent.agent_id
        self._count("leave_attempts")
        self._open_until.pop(agent_id, None)
        if not self.active:
            self._count("leave_placebo")
            return {"success": False, "error": "placebo", "details": {"placebo": True}}
        units = self.registry.units_of(agent_id)
        if not units:
            return {"success": False, "error": "not_a_member", "details": {}}
        unit = self._worst_units(agent_id, units, 1)[0]
        result = self._leave_unit(agent, unit, step, reason="leave")
        if result.vetoed:
            return {"success": False, "error": "exit_vetoed", "details": {"unit_id": unit.unit_id}}
        return {
            "success": True,
            "error": None,
            "details": {
                "unit_id": unit.unit_id,
                "paid": result.paid,
                "enforced": result.enforced,
                "dissolved": result.dissolved,
            },
        }

    # --------------------------------------------------------- perception
    def partner_ids(self, agent_id: str) -> set[str]:
        """Everyone who shares an active unit with ``agent_id``."""
        if not self.active:
            return set()
        return self.registry.partners_of(agent_id)

    def is_partner(self, agent_id: str, other_id: str) -> bool:
        return other_id in self.partner_ids(agent_id)

    def is_member(self, agent_id: str) -> bool:
        return self.active and self.registry.is_member(agent_id)

    def preferred_coparent(self, agent: Any, candidates: Sequence[Any]) -> Any:
        """The nearest partner among ``candidates`` (``None`` when there is none)."""
        if not (self.active and self.config.partner_coparent):
            return None
        partners = self.partner_ids(agent.agent_id)
        pool = [c for c in candidates if getattr(c, "agent_id", None) in partners]
        if not pool:
            return None
        return min(pool, key=lambda c: (_distance(agent.position, c.position), str(c.agent_id)))

    # ----------------------------------------------------------- telemetry
    def _telemetry_row(self, step: int) -> dict[str, Any]:
        alive = [a for a in self._agents.values() if self._alive(a)]
        n_alive = len(alive)
        members = self.registry.members_in_any_unit() if self.active else set()
        units = self.registry.active_units() if self.active else []
        bonds = [u for u in units if u.kind == self.kind]
        row: dict[str, Any] = {
            "step": step,
            "alive": n_alive,
            "members": len(members),
            "coverage": len(members) / n_alive if n_alive else 0.0,
            "units_active": len(bonds),
            "families_active": sum(1 for u in units if u.kind != self.kind),
            "mean_unit_size": float(np.mean([u.n_members for u in bonds])) if bonds else 0.0,
            "open_offers": sum(1 for until in self._open_until.values() if until >= step),
        }
        for key in (
            "bond_attempts",
            "bond_success",
            "units_formed",
            "units_joined",
            "leave_attempts",
            "exits_leave",
            "exits_switch",
            "exits_enforced",
            "exit_vetoed",
            "offers_blocked",
            "dissolved_by_exit",
            "member_deaths",
            "born_in",
            "pooled",
            "forfeits",
            "credited_payout",
            "credited_dissolution",
            "membership_reward",
        ):
            row[key] = float(self._step_counters.get(key, 0.0))
        for name in COOPERATION_GENE_NAMES:
            vals = [self.genes_of(a).values[name] for a in alive]
            row[f"mean_{name}"] = float(np.mean(vals)) if vals else float("nan")
        for name in ("scope", "exit_cost", "recognition", "symmetry"):
            vals = [getattr(u.settings, name) for u in bonds]
            row[f"unit_{name}"] = float(np.mean(vals)) if vals else float("nan")
        row["unit_exclusive"] = float(np.mean([u.settings.exclusive for u in bonds])) if bonds else float("nan")
        row["unit_max_members"] = (
            float(np.mean([u.settings.max_members or 0 for u in bonds])) if bonds else float("nan")
        )
        if alive:
            dominant = Counter(self.genes_of(a).goals.dominant.value for a in alive)
            for gene in MOTIVE_GENE_NAMES:
                motive = gene.split("_", 1)[1]
                row[f"dominant_{motive}"] = dominant.get(motive, 0) / n_alive
        return row

    def unit_records(self, step: int | None = None) -> list[dict[str, Any]]:
        """One flat row per unit, with its diagnosis."""
        step = self.current_step() if step is None else step
        units = list(self.registry.units.values())
        diagnoses = {d.unit_id: d for d in diagnose_all(units, step, DEFAULT_THRESHOLDS, self.registry.vetoed_unit_ids)}
        rows = []
        for unit in units:
            d = diagnoses[unit.unit_id]
            ledgers = {**unit.departed, **unit.members}
            founders = sorted(ledgers.values(), key=lambda lg: lg.rank_order)[:2]
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "kind": unit.kind,
                    "founded_step": unit.founded_step,
                    "dissolved_step": unit.dissolved_step,
                    "dissolution_reason": unit.dissolution_reason,
                    "lifetime": unit.lifetime(step),
                    "active": unit.is_active,
                    "censored": unit.is_active,
                    "max_size": len(ledgers),
                    "parent_id": unit.parent_id,
                    **{f"setting_{k}": v for k, v in unit.settings.to_dict().items() if k != "kind"},
                    "total_inflow": unit.totals["inflow"],
                    "total_surplus": unit.totals["surplus"],
                    "total_payout": unit.totals["payout"],
                    "total_aid": unit.totals["aid"],
                    "total_forfeits": unit.totals["forfeits"],
                    "exit_desire_steps": sum(lg.exit_desire_steps for lg in ledgers.values()),
                    "trapped_steps": sum(lg.trapped_steps for lg in ledgers.values()),
                    "member_steps": sum(lg.steps for lg in ledgers.values()),
                    "founder_ids": ";".join(lg.participant_id for lg in founders),
                    "primary_mode": d.primary.value,
                    "modes": ";".join(m.value for m in d.modes),
                }
            )
        return rows

    def summary(self, step: int | None = None) -> dict[str, Any]:
        step = self.current_step() if step is None else step
        units = list(self.registry.units.values())
        diagnoses = diagnose_all(units, step, DEFAULT_THRESHOLDS, self.registry.vetoed_unit_ids)
        bonds = [u for u in units if u.kind == self.kind]
        coverage = [r["coverage"] for r in self.timeseries]
        return {
            "config": self.config.to_dict(),
            "final_step": step,
            "founders_seen": self.founders_seen,
            "counters": {k: float(v) for k, v in self.counters.items()},
            "registry": self.registry.summary(step),
            "units_formed": len(bonds),
            "mean_unit_lifetime": float(np.mean([u.lifetime(step) for u in bonds])) if bonds else 0.0,
            "mean_coverage": float(np.mean(coverage)) if coverage else 0.0,
            "final_coverage": coverage[-1] if coverage else 0.0,
            "failure_modes": mode_prevalence(diagnoses, self.kind),
            "action_indices": {a.name.lower(): int(a) for a in CooperationActionType},
        }

    def close(self, output_dir: str | None = None) -> None:
        """Write telemetry (once). Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        out = output_dir or self.config.output_dir
        if not out:
            return
        os.makedirs(out, exist_ok=True)
        step = self.current_step()
        _write_csv(os.path.join(out, TIMESERIES_FILE), self.timeseries)
        _write_csv(os.path.join(out, UNITS_FILE), self.unit_records(step))
        with open(os.path.join(out, SUMMARY_FILE), "w", encoding="utf-8") as handle:
            json.dump(self.summary(step), handle, indent=2, default=str)
        if self.config.keep_events and self.registry.events:
            with open(os.path.join(out, EVENTS_FILE), "w", encoding="utf-8") as handle:
                handle.writelines(json.dumps(event.to_dict(), default=str) + "\n" for event in self.registry.events)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


__all__ = [
    "EVENTS_FILE",
    "INCOME_EMA_ALPHA",
    "SUMMARY_FILE",
    "TIMESERIES_FILE",
    "UNITS_FILE",
    "CooperationService",
]
