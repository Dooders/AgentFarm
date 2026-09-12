"""Single seeded run of the veil-ceiling world.

The run has two phases. During ``train_ticks`` agents learn, and monitoring
is confined to the training region. During ``eval_ticks`` learning is frozen
and monitoring is extended to the held-out band so transfer of any cue
conditioning can be measured in a region never paired with enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from farm.core.inheritance_telemetry import InheritanceTelemetry
from farm.core.policy_inheritance import apply_lamarckian_policy_warmstart
from farm.experiments.veil_ceiling.agents import (
    CUE_FEATURE_INDEX,
    N_ACTIONS,
    N_FEATURES,
    REGION_FEATURE_OFFSET,
    VeilAction,
    VeilAgent,
    summarise_agents,
)
from farm.experiments.veil_ceiling.config import RunConfig
from farm.experiments.veil_ceiling.learner import MLPQLearner
from farm.experiments.veil_ceiling.monitoring import (
    RESIDUAL_DEFECTIONS,
    RESIDUAL_MOVES,
    RESIDUAL_OCCUPANCY,
    Monitoring,
)
from farm.experiments.veil_ceiling.world import ResourceField

_STREAM_WORLD, _STREAM_MONITOR, _STREAM_DECOY, _STREAM_CUE, _STREAM_DYNAMICS, _STREAM_POLICY = range(6)
_WINDOW_STATS = ("opportunities", "defections", "penalties")


@dataclass
class ActionOutcome:
    """What one executed action did, for ledger bookkeeping."""

    opportunity: bool = False
    defected: bool = False
    gathered: bool = False
    penalised: bool = False


@dataclass
class RunResult:
    """Outputs of one run: per-agent ledgers, per-window series, run summary."""

    config: RunConfig
    agents: pd.DataFrame
    windows: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


class VeilSimulation:
    """Owns the world, the agents and the monitoring for one run."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        streams = np.random.SeedSequence(cfg.seed).spawn(6)
        self.world_rng = np.random.default_rng(streams[_STREAM_WORLD])
        self.dynamics_rng = np.random.default_rng(streams[_STREAM_DYNAMICS])
        self._policy_seed_seq = streams[_STREAM_POLICY]
        self.field = ResourceField(cfg.world, self.world_rng)
        self.monitoring = Monitoring(
            cfg.world,
            cfg.condition.monitoring,
            true_rng=np.random.default_rng(streams[_STREAM_MONITOR]),
            decoy_rng=np.random.default_rng(streams[_STREAM_DECOY]),
            cue_rng=np.random.default_rng(streams[_STREAM_CUE]),
        )
        self.telemetry = InheritanceTelemetry()
        self.agents: list[VeilAgent] = []
        self._next_agent_id = 0
        self.tick = 0
        self.births = 0
        self.deaths = 0
        self._window_counts = np.zeros((len(_WINDOW_STATS), 2, 2, 2))
        self._window_births = 0
        self._window_deaths = 0
        self._window_rows: list[dict[str, Any]] = []
        self._epoch_overlaps: list[float] = []
        self._epoch_kl: list[float] = []
        self._epoch_gini: list[float] = []
        self._spawn_initial_population()

    # ── setup ─────────────────────────────────────────────────────────────
    def _new_learner(self) -> MLPQLearner:
        # ``spawn`` is stateful, so the k-th agent created in any two runs
        # that share a seed receives the same policy stream.
        seq = self._policy_seed_seq.spawn(1)[0]
        return MLPQLearner(N_FEATURES, N_ACTIONS, self.cfg.learner, np.random.default_rng(seq))

    def _spawn_initial_population(self) -> None:
        w = self.cfg.world
        xs = self.world_rng.integers(0, w.width, size=w.initial_population)
        ys = self.world_rng.integers(0, w.height, size=w.initial_population)
        for i in range(w.initial_population):
            self._add_agent(int(xs[i]), int(ys[i]), w.initial_energy, generation=0, parent=None)

    def _add_agent(self, x: int, y: int, energy: float, generation: int, parent: VeilAgent | None) -> VeilAgent:
        learner = self._new_learner()
        learner.learning_enabled = self.tick < self.cfg.train_ticks
        agent = VeilAgent(
            agent_id=self._next_agent_id,
            x=x,
            y=y,
            energy=energy,
            learner=learner,
            generation=generation,
            born_tick=self.tick,
            parent_id=None if parent is None else parent.agent_id,
        )
        self._next_agent_id += 1
        self.agents.append(agent)
        if parent is not None and self.cfg.inheritance_mode == "lamarckian":
            reason = apply_lamarckian_policy_warmstart(parent, agent)
            if reason is None:
                self.telemetry.record_applied()
            else:
                self.telemetry.record_skipped(reason)
        return agent

    # ── observation ───────────────────────────────────────────────────────
    @property
    def in_eval(self) -> bool:
        return self.tick >= self.cfg.train_ticks

    def observe(self, agent: VeilAgent, cue: int, crowding: int) -> np.ndarray:
        w = self.cfg.world
        f = np.zeros(N_FEATURES)
        f[0] = min(agent.energy / 20.0, 2.0)
        node = self.field.best_node_in_range(agent.x, agent.y)
        if node >= 0:
            stock = float(self.field.amount[node])
            f[1] = 1.0
            f[2] = stock / w.node_max_amount
            f[3] = 1.0 if stock > w.regen_threshold else 0.0
            f[4] = self.field.gather_yield(node) / w.gather_amount
            f[5] = self.field.over_harvest_yield(node) / w.gather_amount
        _, dist = self.field.nearest_healthy_node(agent.x, agent.y)
        f[6] = 1.0 if not np.isfinite(dist) else min(dist / max(w.width, w.height), 1.0)
        f[7] = min(crowding / 8.0, 1.0)
        f[CUE_FEATURE_INDEX] = float(cue)
        f[REGION_FEATURE_OFFSET + w.region_of(agent.x, agent.y)] = 1.0
        f[N_FEATURES - 1] = agent.age / w.max_age
        return f

    def _crowding(self) -> np.ndarray:
        """Number of other alive agents within harvest range of each agent's cell."""
        w = self.cfg.world
        occupancy = np.zeros((w.height, w.width), dtype=np.int64)
        for a in self.agents:
            if a.alive:
                occupancy[a.y, a.x] += 1
        r = w.harvest_range
        padded = np.pad(occupancy, r)
        window = sum(
            padded[dy : dy + w.height, dx : dx + w.width] for dy in range(2 * r + 1) for dx in range(2 * r + 1)
        )
        return window - 1

    # ── actions ───────────────────────────────────────────────────────────
    def _move(self, agent: VeilAgent) -> None:
        w = self.cfg.world
        here = self.field.best_node_in_range(agent.x, agent.y)
        at_healthy = here >= 0 and self.field.amount[here] > w.regen_threshold
        target, _ = self.field.nearest_healthy_node(agent.x, agent.y)
        if at_healthy or target < 0:
            dx, dy = self.dynamics_rng.choice([-1, 0, 1], size=2)
        else:
            tx, ty = self.field.xs[target], self.field.ys[target]
            dx, dy = int(np.sign(tx - agent.x)), int(np.sign(ty - agent.y))
        agent.x = int(min(max(agent.x + dx, 0), w.width - 1))
        agent.y = int(min(max(agent.y + dy, 0), w.height - 1))
        agent.energy -= w.move_cost

    def _act(self, agent: VeilAgent, action: int, monitored: bool) -> ActionOutcome:
        """Execute ``action`` and classify what happened."""
        w = self.cfg.world
        node = self.field.best_node_in_range(agent.x, agent.y)
        outcome = ActionOutcome(opportunity=self.field.is_defection_opportunity(node))
        if action == VeilAction.MOVE:
            self._move(agent)
        elif action == VeilAction.GATHER:
            if node >= 0:
                taken = self.field.gather(node)
                agent.energy += taken
                agent.energy_gained += taken
                outcome.gathered = taken > 0.0
        elif action == VeilAction.OVER_HARVEST and node >= 0:
            agent.energy -= w.over_harvest_effort
            taken = self.field.over_harvest(node)
            agent.energy += taken
            agent.energy_gained += taken
            if taken > 0.0 and outcome.opportunity:
                # Only a threshold-crossing draw is a defection; at a rich
                # node the action is an ordinary (if effortful) gather.
                outcome.defected = True
                if monitored:
                    penalty = self.cfg.condition.monitoring.penalty
                    agent.energy -= penalty
                    agent.penalties_paid += penalty
                    outcome.penalised = True
            elif taken > 0.0:
                outcome.gathered = True
        return outcome

    # ── main loop ─────────────────────────────────────────────────────────
    def step(self) -> None:
        cfg = self.cfg
        w = cfg.world
        resampled = self.monitoring.maybe_resample(self.tick, include_heldout=self.in_eval)
        if resampled and np.isfinite(self.monitoring.last_overlap):
            self._epoch_overlaps.append(self.monitoring.last_overlap)
            self._epoch_kl.append(self.monitoring.last_kl)
            self._epoch_gini.append(self.monitoring.last_weight_gini)
        self.field.regenerate()
        crowding = self._crowding()
        phase = 1 if self.in_eval else 0
        for agent in self.agents:
            if not agent.alive:
                continue
            cell = self.field.cell_index(agent.x, agent.y)
            monitored = self.monitoring.is_monitored(cell)
            cue = self.monitoring.cue(cell)
            region_kind = 1 if w.is_heldout(agent.x, agent.y) else 0
            state = self.observe(agent, cue, int(crowding[agent.y, agent.x]))
            if agent.pending_state is not None:
                agent.learner.observe(agent.pending_state, agent.pending_action, agent.pending_reward, state, False)
            energy_before = agent.energy
            action = agent.learner.select_action(state)
            outcome = self._act(agent, action, monitored)
            agent.energy -= w.base_consumption
            agent.age += 1
            reward = agent.energy - energy_before
            agent.ledger.record(
                phase=phase,
                region_kind=region_kind,
                monitored=int(monitored),
                cue=cue,
                opportunity=outcome.opportunity,
                action=action,
                defected=outcome.defected,
                gathered=outcome.gathered,
                penalised=outcome.penalised,
                energy=agent.energy,
            )
            wc = self._window_counts[:, region_kind, int(monitored), cue]
            if outcome.opportunity:
                wc[0] += 1
            if outcome.defected:
                wc[1] += 1
            if outcome.penalised:
                wc[2] += 1
            if monitored and not self.in_eval:
                if outcome.defected:
                    self.monitoring.record_residual(cell, RESIDUAL_DEFECTIONS)
                if action == VeilAction.MOVE:
                    self.monitoring.record_residual(cell, RESIDUAL_MOVES)
                self.monitoring.record_residual(cell, RESIDUAL_OCCUPANCY)
            if agent.energy <= 0.0 or agent.age >= w.max_age:
                agent.alive = False
                agent.death_tick = self.tick
                agent.learner.observe(state, action, reward, np.zeros(N_FEATURES), True)
                agent.pending_state = None
                self.deaths += 1
                self._window_deaths += 1
            else:
                agent.pending_state = state
                agent.pending_action = action
                agent.pending_reward = reward
        self._reproduce()
        self.tick += 1
        if self.tick % cfg.window_ticks == 0 or self.tick == cfg.total_ticks:
            self._close_window()

    def _reproduce(self) -> None:
        w = self.cfg.world
        alive = sum(1 for a in self.agents if a.alive)
        for agent in list(self.agents):
            if alive >= w.max_population:
                break
            if not agent.alive or agent.energy < w.reproduction_threshold:
                continue
            if self.dynamics_rng.random() >= w.reproduction_chance:
                continue
            agent.energy -= w.offspring_cost
            agent.offspring += 1
            dx, dy = self.dynamics_rng.choice([-1, 0, 1], size=2)
            x = int(min(max(agent.x + dx, 0), w.width - 1))
            y = int(min(max(agent.y + dy, 0), w.height - 1))
            self._add_agent(x, y, w.offspring_energy, generation=agent.generation + 1, parent=agent)
            alive += 1
            self.births += 1
            self._window_births += 1

    def _close_window(self) -> None:
        alive = [a for a in self.agents if a.alive]
        row: dict[str, Any] = {
            "window_end_tick": self.tick,
            "phase": "eval" if self.tick > self.cfg.train_ticks else "train",
            "population": len(alive),
            "mean_generation": float(np.mean([a.generation for a in alive])) if alive else float("nan"),
            "births": self._window_births,
            "deaths": self._window_deaths,
            "total_stock": self.field.total_stock(),
            "suppressed_fraction": self.field.suppressed_fraction(),
            "coverage_realised": self.monitoring.coverage_realised(),
            "monitor_policy": self.cfg.condition.monitoring.policy,
            "mask_overlap": self.monitoring.last_overlap,
            "weight_kl": self.monitoring.last_kl,
            "weight_gini": self.monitoring.last_weight_gini,
        }
        for si, stat in enumerate(_WINDOW_STATS):
            for ri, region in enumerate(("train", "heldout")):
                for m in (0, 1):
                    for cue in (0, 1):
                        row[f"{stat}__{region}__m{m}__c{cue}"] = float(self._window_counts[si, ri, m, cue])
        self._window_rows.append(row)
        self._window_counts[:] = 0.0
        self._window_births = 0
        self._window_deaths = 0

    def freeze_learning(self) -> None:
        for agent in self.agents:
            agent.learner.learning_enabled = False

    def run(self) -> RunResult:
        cfg = self.cfg
        while self.tick < cfg.total_ticks:
            if self.tick == cfg.train_ticks:
                self.freeze_learning()
            self.step()
            if not any(a.alive for a in self.agents):
                if self.tick % cfg.window_ticks:
                    self._close_window()
                break
        return self._result()

    def _result(self) -> RunResult:
        cfg = self.cfg
        agents_df = pd.DataFrame(summarise_agents(self.agents, self.tick))
        windows_df = pd.DataFrame(self._window_rows)
        # Enforcement is matched on training-region defections during the
        # training phase; the held-out band is never monitored while learning.
        train_defections = float(sum(a.ledger.counts[2, 0, 0].sum() for a in self.agents))
        train_penalties = float(sum(a.ledger.counts[6, 0, 0].sum() for a in self.agents))
        summary: dict[str, Any] = {
            "run_id": cfg.run_id,
            "cell_id": cfg.cell_id,
            "condition": cfg.condition.name,
            "family": cfg.condition.family,
            "inheritance_mode": cfg.inheritance_mode,
            "seed": cfg.seed,
            "coverage": cfg.condition.monitoring.coverage,
            "fidelity": -1.0 if cfg.condition.monitoring.fidelity is None else cfg.condition.monitoring.fidelity,
            "penalty": cfg.condition.monitoring.penalty,
            "decorrelated": int(cfg.condition.monitoring.decorrelated),
            "monitor_policy": cfg.condition.monitoring.policy,
            "expected_penalty_per_defection": cfg.condition.monitoring.expected_penalty_per_defection,
            "realised_enforcement": (train_penalties / train_defections) if train_defections > 0 else float("nan"),
            "adaptive_draws": self.monitoring.n_adaptive_draws,
            "mean_mask_overlap": float(np.mean(self._epoch_overlaps)) if self._epoch_overlaps else float("nan"),
            "mean_weight_kl": float(np.mean(self._epoch_kl)) if self._epoch_kl else 0.0,
            "mean_weight_gini": float(np.mean(self._epoch_gini)) if self._epoch_gini else 0.0,
            "ticks_completed": self.tick,
            "extinct": int(not any(a.alive for a in self.agents)),
            "final_population": int(sum(1 for a in self.agents if a.alive)),
            "total_agents": len(self.agents),
            "births": self.births,
            "deaths": self.deaths,
            "warmstart_applied": self.telemetry.warmstart_applied,
            "warmstart_skipped": self.telemetry.warmstart_skipped,
        }
        return RunResult(config=cfg, agents=agents_df, windows=windows_df, summary=summary)


def run_simulation(cfg: RunConfig) -> RunResult:
    """Run one seeded configuration to completion."""
    return VeilSimulation(cfg).run()
