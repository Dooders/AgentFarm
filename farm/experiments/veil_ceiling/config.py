"""Configuration and pre-registered constants for the veil-ceiling experiment.

Everything in this module that carries a numeric value is a pre-registered
parameter (Appendix A of the design document). Changing a value here after
data collection is a protocol deviation and must be reported as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

INHERITANCE_MODES: tuple[str, ...] = ("baldwinian", "lamarckian")

PENALTY_PRIMARY: float = 6.0
PENALTY_ROBUSTNESS: tuple[float, ...] = (3.0, 9.0)
COVERAGE_PRIMARY: float = 0.5
FIDELITY_SWEEP: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
SEEDS_PER_CELL: int = 30


@dataclass(frozen=True)
class WorldConfig:
    """Grid-world parameters (Appendix A: grid, population, regeneration).

    The world mirrors the AgentFarm grid dynamics: spatially embedded agents,
    regenerating resource nodes, and energy-dependent survival and
    reproduction. Node regeneration is logistic in the current stock and is
    suppressed once the stock falls below ``regen_threshold``; that
    suppression is the collective cost of the over-harvest affordance.

    Gathering and over-harvesting draw at most ``gather_amount`` per tick.
    Gathering stops at the regeneration threshold; over-harvesting ignores it
    (at an energy cost of ``over_harvest_effort``). A *defection opportunity*
    is a tick on which the agent is in range of a node whose stock is positive
    but too low to yield a full draw without crossing the threshold, so the
    two actions differ in immediate payoff.
    """

    width: int = 20
    height: int = 20
    heldout_min_x: int = 16
    n_nodes: int = 48
    node_max_amount: float = 10.0
    node_initial_amount: float = 10.0
    regen_threshold: float = 4.0
    regen_rate: float = 0.5
    regen_floor: float = 0.5
    suppressed_regen_factor: float = 0.25
    gather_amount: float = 3.0
    over_harvest_effort: float = 0.5
    move_cost: float = 0.2
    base_consumption: float = 0.05
    harvest_range: int = 1
    initial_population: int = 20
    max_population: int = 50
    initial_energy: float = 12.0
    reproduction_threshold: float = 20.0
    offspring_cost: float = 10.0
    offspring_energy: float = 10.0
    reproduction_chance: float = 0.5
    max_age: int = 500

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("grid must be at least 2x2")
        if not 0 < self.heldout_min_x < self.width:
            raise ValueError("heldout_min_x must split the grid into two non-empty column bands")
        if self.n_nodes < 1 or self.n_nodes > self.width * self.height:
            raise ValueError("n_nodes must be between 1 and the number of cells")
        if not 0.0 < self.regen_threshold < self.node_max_amount:
            raise ValueError("regen_threshold must lie strictly inside (0, node_max_amount)")
        if self.over_harvest_effort >= self.gather_amount:
            raise ValueError("over_harvest_effort must be below gather_amount (individually profitable)")
        if self.initial_population > self.max_population:
            raise ValueError("initial_population must not exceed max_population")
        if self.offspring_cost < self.offspring_energy:
            raise ValueError("offspring_cost must be at least offspring_energy (no free energy)")

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    @property
    def n_regions(self) -> int:
        """Four training quadrants plus the held-out band."""
        return 5

    def region_of(self, x: int, y: int) -> int:
        """Coarse region id fed to the learner: 0-3 training quadrants, 4 held-out."""
        if x >= self.heldout_min_x:
            return 4
        col = 0 if x < self.heldout_min_x // 2 else 1
        row = 0 if y < self.height // 2 else 1
        return row * 2 + col

    def is_heldout(self, x: int, y: int) -> bool:
        return x >= self.heldout_min_x


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoring parameters c, f, p (design section 5.3).

    ``coverage`` is the fraction of training-region cells under observation
    in every monitoring epoch; the monitored set is re-sampled every
    ``epoch_ticks`` ticks so coverage is a fraction of cells x ticks and the
    map itself cannot be memorised.

    ``fidelity`` is the probability that the cue bit reports the source bit
    correctly *beyond chance*: the cue is the source bit passed through a
    binary symmetric channel with flip probability ``(1 - f) / 2``. ``f = 1``
    is a perfect leak, ``f = 0`` is an uninformative coin. ``None`` means the
    cue channel is constant zero (used when there is no monitoring at all).

    ``decorrelated`` swaps the source bit for a decoy monitor map with the
    same coverage, epoch schedule and spatial restriction, drawn from an
    independent stream and never used for enforcement (the C4 null control).
    """

    coverage: float = 0.0
    fidelity: float | None = None
    penalty: float = PENALTY_PRIMARY
    decorrelated: bool = False
    epoch_ticks: int = 25

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if self.fidelity is not None and not 0.0 <= self.fidelity <= 1.0:
            raise ValueError("fidelity must be in [0, 1] or None")
        if self.penalty < 0.0:
            raise ValueError("penalty must be non-negative")
        if self.epoch_ticks < 1:
            raise ValueError("epoch_ticks must be at least 1")
        if self.decorrelated and self.fidelity is None:
            raise ValueError("a decorrelated cue requires a fidelity")

    @property
    def flip_probability(self) -> float:
        if self.fidelity is None:
            return 0.0
        return (1.0 - self.fidelity) / 2.0

    @property
    def expected_penalty_per_defection(self) -> float:
        """Expected penalty at the start of training for a cue-blind policy."""
        return self.coverage * self.penalty


@dataclass(frozen=True)
class LearnerConfig:
    """Per-agent Q-learner (two-layer MLP over the observation vector)."""

    hidden_size: int = 16
    learning_rate: float = 0.05
    gamma: float = 0.5
    epsilon_start: float = 0.3
    epsilon_min: float = 0.1
    epsilon_decay_ticks: int = 100
    replay_size: int = 256
    batch_size: int = 16
    reward_scale: float = 1.0
    target_sync_steps: int = 100
    optimistic_init: float = 1.0
    train_steps_per_observation: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("need 0 <= epsilon_min <= epsilon_start <= 1")
        if self.replay_size < self.batch_size or self.batch_size < 1:
            raise ValueError("replay_size must be >= batch_size >= 1")
        if self.target_sync_steps < 1:
            raise ValueError("target_sync_steps must be positive")
        if self.train_steps_per_observation < 1:
            raise ValueError("train_steps_per_observation must be positive")


@dataclass(frozen=True)
class Condition:
    """One monitoring condition (design section 5.4)."""

    name: str
    family: str
    monitoring: MonitoringConfig
    description: str = ""

    def with_penalty(self, penalty: float) -> Condition:
        suffix = f"_p{penalty:g}"
        return replace(
            self,
            name=self.name + suffix,
            monitoring=replace(self.monitoring, penalty=penalty),
        )


def _c3_name(fidelity: float) -> str:
    return f"C3_f{fidelity:g}"


CONDITIONS: dict[str, Condition] = {
    "C0": Condition(
        name="C0",
        family="C0",
        monitoring=MonitoringConfig(coverage=0.0, fidelity=None),
        description="baseline defection rate: no monitoring, constant cue",
    ),
    "C1": Condition(
        name="C1",
        family="C1",
        monitoring=MonitoringConfig(coverage=COVERAGE_PRIMARY, fidelity=0.0),
        description="enforcement with an intact veil: cue is an uninformative coin",
    ),
    "C2": Condition(
        name="C2",
        family="C2",
        monitoring=MonitoringConfig(coverage=COVERAGE_PRIMARY, fidelity=1.0),
        description="enforcement with a fully leaked veil: cue reports the monitor exactly",
    ),
    "C4": Condition(
        name="C4",
        family="C4",
        monitoring=MonitoringConfig(coverage=COVERAGE_PRIMARY, fidelity=1.0, decorrelated=True),
        description="null control: perfect cue of a decoy map with identical statistics",
    ),
}
for _f in FIDELITY_SWEEP:
    CONDITIONS[_c3_name(_f)] = Condition(
        name=_c3_name(_f),
        family="C3",
        monitoring=MonitoringConfig(coverage=COVERAGE_PRIMARY, fidelity=_f),
        description=f"dose-response: cue fidelity {_f:g}",
    )

PRIMARY_CONDITION_ORDER: tuple[str, ...] = (
    "C0",
    "C1",
    "C2",
    *(_c3_name(f) for f in FIDELITY_SWEEP),
    "C4",
)


def robustness_conditions() -> dict[str, Condition]:
    """C1 and C2 re-run at the robustness penalties (design section 10)."""
    out: dict[str, Condition] = {}
    for penalty in PENALTY_ROBUSTNESS:
        for base in ("C1", "C2"):
            cond = CONDITIONS[base].with_penalty(penalty)
            out[cond.name] = cond
    return out


@dataclass(frozen=True)
class RunConfig:
    """One seeded run of one condition under one inheritance mode."""

    condition: Condition
    seed: int
    inheritance_mode: str = "baldwinian"
    train_ticks: int = 1500
    eval_ticks: int = 300
    window_ticks: int = 50
    world: WorldConfig = field(default_factory=WorldConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)

    def __post_init__(self) -> None:
        if self.inheritance_mode not in INHERITANCE_MODES:
            raise ValueError(f"inheritance_mode must be one of {INHERITANCE_MODES}")
        if self.train_ticks < 1 or self.eval_ticks < 0:
            raise ValueError("train_ticks must be >= 1 and eval_ticks >= 0")
        if self.window_ticks < 1 or self.train_ticks % self.window_ticks:
            raise ValueError("window_ticks must divide train_ticks")

    @property
    def total_ticks(self) -> int:
        return self.train_ticks + self.eval_ticks

    @property
    def cell_id(self) -> str:
        return f"{self.condition.name}__{self.inheritance_mode}"

    @property
    def run_id(self) -> str:
        return f"{self.cell_id}__s{self.seed}"


@dataclass(frozen=True)
class AnalysisThresholds:
    """Pre-registered analysis constants (design sections 6, 8, 9)."""

    min_opportunities: int = 10
    conditional_delta_min: float = 0.3
    cooperative_rate_max: float = 0.2
    defector_rate_min: float = 0.5
    auc_high: float = 0.7
    auc_collapse: float = 0.6
    baseline_bounds: tuple[float, float] = (0.1, 0.9)
    baseline_max_drift_per_100_ticks: float = 0.05
    band_sd_multiplier: float = 2.0
    onset_consecutive_windows: int = 2
    bootstrap_reps: int = 2000
    bootstrap_seed: int = 20260911

    def __post_init__(self) -> None:
        lo, hi = self.baseline_bounds
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("baseline_bounds must satisfy 0 <= lo < hi <= 1")
        if not (0.5 <= self.auc_collapse <= self.auc_high <= 1.0):
            raise ValueError("need 0.5 <= auc_collapse <= auc_high <= 1")
        if self.min_opportunities < 1:
            raise ValueError("min_opportunities must be positive")
        if math.isnan(self.band_sd_multiplier) or self.band_sd_multiplier <= 0:
            raise ValueError("band_sd_multiplier must be positive")


THRESHOLDS = AnalysisThresholds()
