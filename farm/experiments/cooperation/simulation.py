"""Tick loop for the cooperation experiment.

Order of a tick:

1. **Work** -- each living agent picks effort from its marginal retention
   (and current crowding), draws income from its own shock stream, pools
   ``scope`` of it into its units and pays metabolism plus effort cost.
2. **Settle** -- the registry creates surplus, moves dues up and payouts
   down, and pays emergency aid to members below the aid threshold.
3. **Deaths** -- starvation (energy <= 0) or old age; units lose the member
   without an exit cost and may dissolve.
4. **Exits** -- members past ``min_tenure`` compare the value of staying
   with the expected exit payment: leave, or stay trapped.
5. **Matching** -- arranged matches (when families arrange), then random
   encounters: offers, switches and new units.
6. **Births** -- above the reproduction threshold and below the population
   cap; in nesting conditions a reproducing couple anchors a family unit and
   the child is born into it.

A *solo twin* (``RunConfig.cooperative = False``) runs steps 1, 3 and 6 with
the same seed: founders get identical traits and identical shock sequences,
so per-founder outcomes can be compared pairwise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from farm.core.cooperation import CooperationRegistry, Motive, diagnose_all
from farm.experiments.cooperation.agents import CoopAgent, income_stream, unit_vector
from farm.experiments.cooperation.config import BOND_KIND, FAMILY_SETTINGS, THRESHOLDS, RunConfig
from farm.experiments.cooperation.decisions import (
    appraise_current,
    appraise_prospect,
    choose_effort,
    income_signal,
    marginal_retention,
    noisy_accept,
)
from farm.experiments.cooperation.world import IncomeWorld

TRAIT_STREAM: int = 0
DECISION_STREAM: int = 3
REGISTRY_STREAM: int = 4
POPULATION_STREAM: int = 5


@dataclass
class RunResult:
    """Raw outputs of one run."""

    config: RunConfig
    agents: pd.DataFrame
    units: pd.DataFrame
    timeseries: pd.DataFrame
    counters: dict
    event_counts: dict


class CooperationSimulation:
    """One cooperative run or its solo twin."""

    def __init__(self, config: RunConfig) -> None:
        self.cfg = config
        self.world_cfg = config.world
        self.decisions = config.effective_decisions()
        self.condition = config.condition
        self.bond = config.condition.bond
        self.world = IncomeWorld(config.world, config.seed)
        seed = int(config.seed)
        self.trait_rng = np.random.default_rng([seed, TRAIT_STREAM])
        self.decision_rng = np.random.default_rng([seed, DECISION_STREAM])
        self.pop_rng = np.random.default_rng([seed, POPULATION_STREAM])
        self.registry: CooperationRegistry | None = None
        if config.cooperative:
            self.registry = CooperationRegistry(
                economics=config.economics,
                unit_goals=config.unit_goals,
                rng=np.random.default_rng([seed, REGISTRY_STREAM]),
                keep_events=False,
            )
        self.agents: dict[str, CoopAgent] = {}
        self.tick = 0
        self.next_birth_slot = 0
        self.counters: Counter = Counter()
        self.rows: list[dict] = []
        self._effort: dict[str, float] = {}
        self._motive_probs = self.decisions.motive_probabilities()
        self._create_founders()

    # --------------------------------------------------------------- creation
    def _draw_motive(self, u: float) -> Motive:
        cumulative = 0.0
        last = Motive.ECONOMIC
        for motive, p in self._motive_probs.items():
            if p <= 0.0:
                continue
            cumulative += p
            last = motive
            if u < cumulative:
                return motive
        return last

    def _create_founders(self) -> None:
        w = self.world_cfg
        n = w.initial_population
        rng = self.trait_rng
        skills = w.skill_median * np.exp(rng.normal(0.0, w.skill_sigma, size=n))
        diligences = rng.beta(self.decisions.diligence_a, self.decisions.diligence_b, size=n)
        motive_u = rng.random(size=n)
        affinities = [unit_vector(rng, w.affinity_dims) for _ in range(n)]
        expected = self.world.expected_multiplier()
        for i in range(n):
            agent = CoopAgent(
                agent_id=f"f{i}",
                slot=i,
                founder=True,
                skill=float(skills[i]),
                diligence=float(diligences[i]),
                motive=self._draw_motive(float(motive_u[i])),
                affinity=affinities[i],
                energy=w.initial_energy,
                born_tick=0,
                rng=income_stream(self.cfg.seed, True, i),
            )
            agent.income_ema = agent.skill * expected
            self.agents[agent.agent_id] = agent

    def _spawn(self, parent: CoopAgent, tick: int) -> CoopAgent:
        w = self.world_cfg
        rng = self.pop_rng
        slot = self.next_birth_slot
        self.next_birth_slot += 1
        h = w.trait_heritability
        # Fixed number of draws per birth, whatever the parameters.
        z_skill = rng.normal(0.0, w.skill_sigma)
        fresh_diligence = rng.beta(self.decisions.diligence_a, self.decisions.diligence_b)
        u_inherit = rng.random()
        u_motive = rng.random()
        noise = rng.normal(0.0, w.affinity_mutation_sigma, size=w.affinity_dims)
        parent_log = np.log(parent.skill / w.skill_median)
        skill = w.skill_median * np.exp(h * parent_log + np.sqrt(1.0 - h * h) * z_skill)
        diligence = h * parent.diligence + (1.0 - h) * fresh_diligence
        motive = parent.motive if u_inherit < w.motive_heritability else self._draw_motive(float(u_motive))
        affinity = parent.affinity + noise
        norm = float(np.linalg.norm(affinity))
        affinity = affinity / norm if norm > 0 else parent.affinity.copy()
        child = CoopAgent(
            agent_id=f"b{slot}",
            slot=slot,
            founder=False,
            skill=float(skill),
            diligence=float(np.clip(diligence, 0.0, 1.0)),
            motive=motive,
            affinity=affinity,
            energy=w.offspring_energy,
            born_tick=tick + 1,
            rng=income_stream(self.cfg.seed, False, slot),
            parent_id=parent.agent_id,
        )
        child.income_ema = child.skill * self.world.expected_multiplier()
        parent.energy -= w.offspring_cost
        parent.offspring += 1
        self.agents[child.agent_id] = child
        return child

    # ---------------------------------------------------------------- helpers
    def alive_agents(self) -> list[CoopAgent]:
        return [a for a in self.agents.values() if a.alive]

    def _credit(self, payouts: dict[str, float]) -> None:
        for agent_id, amount in payouts.items():
            agent = self.agents.get(agent_id)
            if agent is not None and agent.alive:
                agent.energy += amount
                agent.received += amount

    def _is_adult(self, agent: CoopAgent, tick: int) -> bool:
        return agent.age(tick) >= self.world_cfg.maturity_age

    def _in_family(self, agent: CoopAgent) -> bool:
        return bool(self.registry and self.registry.units_of(agent.agent_id, FAMILY_SETTINGS.kind))

    # ------------------------------------------------------------------- tick
    def step(self) -> None:
        t = self.tick
        reg = self.registry
        alive = self.alive_agents()
        econ = self.cfg.economics
        alpha = self.decisions.income_ema_alpha

        crowding = self.world.crowding(len(alive))
        for agent in alive:
            retention = marginal_retention(agent.agent_id, reg, econ)
            effort = choose_effort(agent, retention * crowding, self.world_cfg)
            gross = self.world.draw_income(agent, effort, t, crowding)
            pooled = reg.contribute(agent.agent_id, gross, t) if reg is not None else 0.0
            agent.energy += gross - pooled - self.world.cost(effort)
            agent.gross_income += gross
            agent.pooled += pooled
            agent.observe_income(gross, alpha)
            self._effort[agent.agent_id] = effort

        if reg is not None:
            payouts = reg.settle(t, holdings={a.agent_id: a.energy for a in alive})
            self._credit(payouts)
        cap = self.world_cfg.max_energy
        for agent in alive:
            if agent.energy > cap:
                self.counters["energy_spilled"] += agent.energy - cap
                agent.energy = cap

        deaths = 0
        for agent in alive:
            in_bond = bool(reg and reg.units_of(agent.agent_id, BOND_KIND))
            agent.record_tick(self._effort.get(agent.agent_id, 0.0), in_bond)
        for agent in alive:
            cause = None
            if agent.energy <= 0.0:
                cause = "starvation"
            elif agent.age(t) + 1 >= self.world_cfg.max_age:
                cause = "age"
            if cause is None:
                continue
            agent.alive = False
            agent.death_tick = t + 1
            agent.death_cause = cause
            deaths += 1
            self.counters[f"deaths_{cause}"] += 1
            if reg is not None:
                self._credit(reg.remove_agent(agent.agent_id, t, reason=cause))

        if reg is not None:
            self._exit_phase(t)
            if self.condition.nesting == "arranged":
                self._arranged_phase(t)
            self._encounter_phase(t)

        births = self._birth_phase(t)
        self._record_row(t, births, deaths)
        self.tick += 1

    def _record_row(self, t: int, births: int, deaths: int) -> None:
        alive = self.alive_agents()
        reg = self.registry
        n = len(alive)
        in_bond = sum(1 for a in alive if reg and reg.units_of(a.agent_id, BOND_KIND)) if reg else 0
        self.rows.append(
            {
                "tick": t,
                "population": n,
                "drought": self.world.is_drought(t),
                "mean_energy": float(np.mean([a.energy for a in alive])) if alive else float("nan"),
                "coverage": in_bond / n if n else 0.0,
                "bonds_active": len(reg.active_units(BOND_KIND)) if reg else 0,
                "families_active": len(reg.active_units(FAMILY_SETTINGS.kind)) if reg else 0,
                "mean_effort": float(np.mean([self._effort.get(a.agent_id, 0.0) for a in alive])) if alive else 0.0,
                "births": births,
                "deaths": deaths,
            }
        )

    # ------------------------------------------------------------------ exits
    def _exit_phase(self, t: int) -> None:
        reg = self.registry
        assert reg is not None
        d = self.decisions
        for agent in self.alive_agents():
            for unit in reg.units_of(agent.agent_id):
                if not unit.is_active or agent.agent_id not in unit.members:
                    continue
                ledger = unit.members[agent.agent_id]
                if ledger.steps < d.min_tenure or (ledger.steps - d.min_tenure) % d.review_interval:
                    continue
                value = appraise_current(agent, unit, self.agents, reg, d, self.world_cfg)
                value += float(self.decision_rng.logistic(0.0, d.decision_noise)) if d.decision_noise > 0 else 0.0
                payment = unit.settings.expected_exit_payment(agent.energy)
                if value >= 0.0:
                    continue
                if value >= -payment:
                    reg.record_exit_desire(unit.unit_id, agent.agent_id, blocked_by_price=True, steps=d.review_interval)
                    continue
                result = reg.leave(unit.unit_id, agent.agent_id, t, holdings=agent.energy)
                if result.vetoed:
                    reg.record_exit_desire(
                        unit.unit_id, agent.agent_id, blocked_by_price=False, steps=d.review_interval
                    )
                    self.counters[f"exits_vetoed_{unit.kind}"] += 1
                    continue
                agent.energy -= result.paid
                agent.exit_paid += result.paid
                self._credit(result.payouts)
                self.counters[f"exits_{unit.kind}"] += 1
                if result.enforced:
                    self.counters[f"exits_enforced_{unit.kind}"] += 1

    # --------------------------------------------------------------- matching
    def _arranged_phase(self, t: int) -> None:
        reg = self.registry
        assert reg is not None
        d = self.decisions
        adults_free = [
            a for a in self.alive_agents() if self._is_adult(a, t) and reg.can_join(a.agent_id, self.bond)[0]
        ]
        for family in sorted(reg.active_units(FAMILY_SETTINGS.kind), key=lambda u: u.unit_id):
            for _ in range(d.arranged_attempts_per_tick):
                kin = [
                    a for a in adults_free if a.agent_id in family.members and reg.can_join(a.agent_id, self.bond)[0]
                ]
                if not kin:
                    break
                a = kin[int(self.decision_rng.integers(len(kin)))]
                outsiders = [
                    b
                    for b in adults_free
                    if b.agent_id not in family.members and b.alive and reg.can_join(b.agent_id, self.bond)[0]
                ]
                if not outsiders:
                    break
                signals = [income_signal(self.decision_rng, b, d.income_signal_noise) for b in outsiders]
                b = outsiders[int(np.argmax(signals))]
                parent_scope = family.settings.scope
                reserve = self.cfg.unit_goals.reserve_target
                v_a, net_a = appraise_prospect(
                    a,
                    [b],
                    self.bond,
                    self.cfg.economics,
                    d,
                    self.world_cfg,
                    self.decision_rng,
                    0,
                    parent_scope,
                    True,
                    reserve,
                )
                v_b, net_b = appraise_prospect(
                    b,
                    [a],
                    self.bond,
                    self.cfg.economics,
                    d,
                    self.world_cfg,
                    self.decision_rng,
                    1,
                    parent_scope,
                    False,
                    reserve,
                )
                consents = {a.agent_id: v_a > d.accept_threshold, b.agent_id: v_b > d.accept_threshold}
                reg.form(
                    self.bond,
                    [a.agent_id, b.agent_id],
                    t,
                    parent_id=family.unit_id,
                    arranged=True,
                    consents=consents,
                    priors={a.agent_id: net_a, b.agent_id: net_b},
                )
                self.counters["matches_arranged"] += 1
                self.counters["matches_arranged_unconsented"] += sum(1 for ok in consents.values() if not ok)

    def _encounter_phase(self, t: int) -> None:
        reg = self.registry
        assert reg is not None
        d = self.decisions
        if d.encounters_per_tick <= 0:
            return
        adults = [a for a in self.alive_agents() if self._is_adult(a, t)]
        if len(adults) < 2:
            return
        arranged = self.condition.nesting == "arranged"
        seekers = [a for a in adults if reg.can_join(a.agent_id, self.bond)[0] or self._host_unit(a) is not None]
        if arranged:
            seekers = [a for a in seekers if not self._in_family(a)]
        order = self.decision_rng.permutation(len(seekers))
        for idx in order:
            a = seekers[int(idx)]
            for _ in range(d.encounters_per_tick):
                b = adults[int(self.decision_rng.integers(len(adults)))]
                if b.agent_id == a.agent_id:
                    continue
                self._try_pair(a, b, t)

    def _host_unit(self, agent: CoopAgent):
        """A unit of the agent's with room for one more (multi-member kinds)."""
        reg = self.registry
        if reg is None or (self.bond.max_members is not None and self.bond.max_members <= 2):
            return None
        for unit in reg.units_of(agent.agent_id, BOND_KIND):
            if unit.has_capacity():
                return unit
        return None

    def _try_pair(self, a: CoopAgent, b: CoopAgent, t: int) -> None:
        reg = self.registry
        assert reg is not None
        d = self.decisions
        econ = self.cfg.economics
        settings = self.bond
        if not (a.alive and b.alive):
            return
        if b.agent_id in reg.partners_of(a.agent_id, BOND_KIND):
            return
        if self.condition.nesting == "arranged" and self._in_family(b):
            return
        host = self._host_unit(a)
        if host is None and not reg.can_join(a.agent_id, settings)[0]:
            return
        target = host if host is not None else settings
        b_ok, reason = reg.can_join(b.agent_id, target)
        b_current = None
        if not b_ok:
            if reason != "exclusivity_limit" or not settings.exclusive:
                return
            held = reg.units_of(b.agent_id, BOND_KIND)
            if not held:
                return
            if held[0].members[b.agent_id].steps < d.min_tenure:
                return
            if reg.offer_blocked(b.agent_id, BOND_KIND, t):
                self.counters["offers_blocked"] += 1
                return
            b_current = held[0]

        reserve = self.cfg.unit_goals.reserve_target
        net_a = None
        if host is not None:
            members = [self.agents[pid] for pid in host.members if pid in self.agents]
            others_for_a = [m for m in members if m.agent_id != a.agent_id] + [b]
            rank_a = sorted(host.members.values(), key=lambda lg: lg.rank_order).index(host.members[a.agent_id])
            v_a_with, _ = appraise_prospect(
                a, others_for_a, settings, econ, d, self.world_cfg, self.decision_rng, rank_a, reserve_target=0.0
            )
            v_a_now = appraise_current(a, host, self.agents, reg, d, self.world_cfg)
            a_accepts = noisy_accept(v_a_with - v_a_now, 0.0, d.decision_noise, self.decision_rng)
            v_b_new, net_b = appraise_prospect(
                b,
                members,
                settings,
                econ,
                d,
                self.world_cfg,
                self.decision_rng,
                host.n_participants,
                reserve_target=reserve,
            )
        else:
            v_a, net_a = appraise_prospect(
                a, [b], settings, econ, d, self.world_cfg, self.decision_rng, 0, reserve_target=reserve
            )
            a_accepts = noisy_accept(v_a, d.accept_threshold, d.decision_noise, self.decision_rng)
            v_b_new, net_b = appraise_prospect(
                b, [a], settings, econ, d, self.world_cfg, self.decision_rng, 1, reserve_target=reserve
            )
        if not a_accepts:
            return
        if b_current is not None:
            v_b_stay = appraise_current(b, b_current, self.agents, reg, d, self.world_cfg)
            pay = b_current.settings.expected_exit_payment(b.energy)
            b_accepts = noisy_accept(v_b_new - pay - v_b_stay, d.switch_premium, d.decision_noise, self.decision_rng)
        else:
            b_accepts = noisy_accept(v_b_new, d.accept_threshold, d.decision_noise, self.decision_rng)
        if not b_accepts:
            return
        if b_current is not None:
            result = reg.leave(b_current.unit_id, b.agent_id, t, holdings=b.energy)
            if not result.left:
                reg.record_exit_desire(b_current.unit_id, b.agent_id, blocked_by_price=False)
                self.counters[f"exits_vetoed_{b_current.kind}"] += 1
                return
            b.energy -= result.paid
            b.exit_paid += result.paid
            self._credit(result.payouts)
            self.counters["switches"] += 1
            if result.enforced:
                self.counters[f"exits_enforced_{b_current.kind}"] += 1
        if host is not None:
            if reg.join(host.unit_id, b.agent_id, t, prior=net_b):
                self.counters["joins"] += 1
        else:
            reg.form(settings, [a.agent_id, b.agent_id], t, priors={a.agent_id: net_a, b.agent_id: net_b})
            self.counters["matches_chosen"] += 1

    # ----------------------------------------------------------------- births
    def _birth_phase(self, t: int) -> int:
        w = self.world_cfg
        alive = self.alive_agents()
        population = len(alive)
        births = 0
        order = self.pop_rng.permutation(len(alive))
        for idx in order:
            parent = alive[int(idx)]
            if not parent.alive or parent.energy < w.reproduction_threshold or not self._is_adult(parent, t):
                continue
            if population >= w.max_population:
                self.counters["births_blocked_cap"] += 1
                continue
            child = self._spawn(parent, t)
            population += 1
            births += 1
            self.counters["births"] += 1
            if self.registry is not None and self.condition.nesting != "none":
                self._ensure_household(parent, t)
                self.registry.register_birth(parent.agent_id, child.agent_id, t)
        return births

    def _ensure_household(self, parent: CoopAgent, t: int) -> None:
        reg = self.registry
        assert reg is not None
        for bond in reg.units_of(parent.agent_id, BOND_KIND):
            if bond.is_active and bond.parent_id is None:
                reg.form(FAMILY_SETTINGS, [], t, children=[bond.unit_id])
                self.counters["families_formed"] += 1

    # -------------------------------------------------------------------- run
    def run(self) -> RunResult:
        for _ in range(self.world_cfg.ticks):
            if not self.alive_agents():
                break
            self.step()
        return self.result()

    def result(self) -> RunResult:
        end = self.tick
        agents = pd.DataFrame([a.to_record(end) for a in self.agents.values()])
        units = self._units_frame(end)
        timeseries = pd.DataFrame(self.rows)
        event_counts = dict(self.registry.event_counts) if self.registry else {}
        return RunResult(
            config=self.cfg,
            agents=agents,
            units=units,
            timeseries=timeseries,
            counters=dict(self.counters),
            event_counts=event_counts,
        )

    def _units_frame(self, end: int) -> pd.DataFrame:
        columns = [
            "unit_id",
            "kind",
            "founded_step",
            "dissolved_step",
            "dissolution_reason",
            "lifetime",
            "censored",
            "arranged",
            "has_parent",
            "n_members_ever",
            "inflow",
            "gross_income",
            "surplus",
            "upward",
            "downward",
            "aid",
            "payout",
            "forfeits",
            "trapped_steps",
            "member_steps",
            "primary_mode",
            "modes",
        ]
        reg = self.registry
        if reg is None or not reg.units:
            return pd.DataFrame(columns=columns)
        units = list(reg.units.values())
        diagnoses = {d.unit_id: d for d in diagnose_all(units, end, THRESHOLDS, reg.vetoed_unit_ids)}
        rows = []
        for unit in units:
            ledgers = {**unit.departed, **unit.members}
            agent_ledgers = [lg for lg in ledgers.values() if not lg.is_unit]
            diag = diagnoses[unit.unit_id]
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "kind": unit.kind,
                    "founded_step": unit.founded_step,
                    "dissolved_step": unit.dissolved_step,
                    "dissolution_reason": unit.dissolution_reason,
                    "lifetime": unit.lifetime(end),
                    "censored": unit.is_active,
                    "arranged": unit.arranged,
                    "has_parent": unit.parent_id is not None,
                    "n_members_ever": len(agent_ledgers),
                    **{
                        k: unit.totals[k]
                        for k in (
                            "inflow",
                            "gross_income",
                            "surplus",
                            "upward",
                            "downward",
                            "aid",
                            "payout",
                            "forfeits",
                        )
                    },
                    "trapped_steps": sum(lg.trapped_steps for lg in agent_ledgers),
                    "member_steps": sum(lg.steps for lg in agent_ledgers),
                    "primary_mode": diag.primary.value,
                    "modes": "|".join(m.value for m in diag.modes),
                }
            )
        return pd.DataFrame(rows, columns=columns)


def run_simulation(config: RunConfig) -> RunResult:
    """Run one cooperative simulation or solo twin."""
    return CooperationSimulation(config).run()
