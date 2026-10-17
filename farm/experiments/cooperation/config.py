"""Parameters, conditions and pre-registered constants for the cooperation experiment.

Numeric values in this module are the experiment's registered parameters
(see ``docs/research/experiments/cooperation/Design.md``). Changing one after
data collection is a protocol deviation and must be reported as such.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from farm.core.cooperation import (
    CooperationSettings,
    DiagnosticThresholds,
    Entry,
    Motive,
    UnitEconomics,
    UnitGoals,
    preset,
)

SEEDS_PER_CELL: int = 20

#: The primary grid: scope x exit cost x recognition for an exclusive pair bond.
PRIMARY_SCOPE: tuple[float, ...] = (0.2, 0.5, 0.8)
PRIMARY_EXIT_COST: tuple[float, ...] = (0.0, 0.3, 0.6)
PRIMARY_RECOGNITION: tuple[float, ...] = (0.0, 0.5, 1.0)

NESTING_MODES: tuple[str, ...] = ("none", "lineage", "arranged")

BOND_KIND: str = "bond"


@dataclass(frozen=True)
class WorldConfig:
    """Non-spatial foraging world with idiosyncratic and aggregate shocks.

    Each tick an agent earns ``skill * effort`` scaled by its own shock draw
    (zero income with probability ``shock_prob``) and by the shared drought
    schedule, pays ``metabolism + effort_cost * effort**2 / 2`` and dies at
    zero energy or at ``max_age``. Energy above ``max_energy`` is lost
    (storage limit).

    Offspring traits: log-skill and diligence regress toward fresh draws
    from the founding distributions with ``trait_heritability`` (0 = fresh
    draw, the default, which keeps the trait distribution stationary so
    institutions are compared in the same world rather than in worlds that
    evolve apart); a child keeps its parent's motive with probability
    ``motive_heritability`` and otherwise draws from the motive mix;
    affinity is the parent's plus noise, so kin resemble each other.

    Resources are finite: when more than ``carrying_capacity`` agents are
    alive, every agent's income is scaled by ``carrying_capacity / N``. The
    population therefore settles where the marginal agent breaks even, below
    the hard ``max_population`` cap, instead of growing until births are
    blocked. Because a cooperative run can keep more agents alive than its
    solo twin, the twin comparison measures the *total* effect of an
    institution, crowding feedback included.
    """

    initial_population: int = 60
    max_population: int = 120
    carrying_capacity: int = 70
    ticks: int = 600
    skill_median: float = 1.0
    skill_sigma: float = 0.35
    shock_prob: float = 0.15
    drought_period: int = 120
    drought_length: int = 20
    drought_factor: float = 0.4
    metabolism: float = 0.2
    effort_cost: float = 1.0
    min_effort: float = 0.2
    initial_energy: float = 8.0
    max_energy: float = 40.0
    reproduction_threshold: float = 22.0
    offspring_cost: float = 12.0
    offspring_energy: float = 8.0
    maturity_age: int = 20
    max_age: int = 300
    trait_heritability: float = 0.0
    motive_heritability: float = 0.0
    affinity_mutation_sigma: float = 0.15
    affinity_dims: int = 3

    def __post_init__(self) -> None:
        if self.initial_population < 2 or self.initial_population > self.max_population:
            raise ValueError("initial_population must be in [2, max_population]")
        if self.carrying_capacity < 1:
            raise ValueError("carrying_capacity must be positive")
        if self.ticks < 1:
            raise ValueError("ticks must be positive")
        if not 0.0 <= self.shock_prob < 1.0:
            raise ValueError("shock_prob must lie in [0, 1)")
        if not 0.0 <= self.drought_factor <= 1.0:
            raise ValueError("drought_factor must lie in [0, 1]")
        if self.drought_length > self.drought_period:
            raise ValueError("drought_length must not exceed drought_period")
        if not 0.0 < self.min_effort <= 1.0:
            raise ValueError("min_effort must lie in (0, 1]")
        if self.offspring_cost < self.offspring_energy:
            raise ValueError("offspring_cost must be at least offspring_energy (no free energy)")
        if self.reproduction_threshold <= self.offspring_cost:
            raise ValueError("reproduction_threshold must exceed offspring_cost")
        if self.max_energy <= self.reproduction_threshold:
            raise ValueError("max_energy must exceed reproduction_threshold")
        if not (0.0 <= self.trait_heritability <= 1.0 and 0.0 <= self.motive_heritability <= 1.0):
            raise ValueError("heritabilities must lie in [0, 1]")
        if self.effort_cost <= 0.0 or self.metabolism < 0.0:
            raise ValueError("effort_cost must be positive and metabolism non-negative")


@dataclass(frozen=True)
class DecisionConfig:
    """How agents appraise, accept, leave and work.

    ``horizon``, ``fit_scale`` and ``affinity_scale`` feed
    :class:`~farm.core.cooperation.goals.ValuationScales`. Accept and leave
    decisions add logistic noise with scale ``decision_noise``. Members
    review each membership every ``review_interval`` steps once past
    ``min_tenure`` (reconsidering every step would let decision noise alone
    dissolve units whose value is near zero).
    """

    encounters_per_tick: int = 1
    horizon: float = 20.0
    fit_scale: float = 1.0
    affinity_scale: float = 2.0
    accept_threshold: float = 0.5
    switch_premium: float = 1.0
    decision_noise: float = 0.5
    min_tenure: int = 10
    review_interval: int = 10
    income_signal_noise: float = 0.2
    income_ema_alpha: float = 0.05
    diligence_a: float = 2.0
    diligence_b: float = 2.0
    motive_mix: tuple[tuple[str, float], ...] = tuple((m.value, 1.0) for m in Motive)
    arranged_attempts_per_tick: int = 1

    def __post_init__(self) -> None:
        if self.encounters_per_tick < 0:
            raise ValueError("encounters_per_tick must be non-negative")
        if self.decision_noise < 0.0 or self.horizon <= 0.0:
            raise ValueError("decision_noise must be non-negative and horizon positive")
        if self.min_tenure < 0:
            raise ValueError("min_tenure must be non-negative")
        if self.review_interval < 1:
            raise ValueError("review_interval must be at least 1")
        if not 0.0 < self.income_ema_alpha <= 1.0:
            raise ValueError("income_ema_alpha must lie in (0, 1]")
        weights = [w for _, w in self.motive_mix]
        if not weights or min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError("motive_mix must have non-negative weights with a positive sum")
        for name, _ in self.motive_mix:
            Motive(name)

    def motive_probabilities(self) -> dict[Motive, float]:
        total = sum(w for _, w in self.motive_mix)
        out = {m: 0.0 for m in Motive}
        for name, w in self.motive_mix:
            out[Motive(name)] += w / total
        return out


def pair_bond(scope: float, exit_cost: float, recognition: float, **overrides: Any) -> CooperationSettings:
    """The primary-grid institution: an exclusive, chosen, symmetric pair."""
    base = CooperationSettings(
        kind=BOND_KIND,
        min_members=2,
        max_members=2,
        scope=scope,
        exit_cost=exit_cost,
        recognition=recognition,
        exclusive=True,
        entry=Entry.CHOSEN,
        symmetry=1.0,
    )
    return base.with_changes(**overrides) if overrides else base


@dataclass(frozen=True)
class Condition:
    """One cell of the matrix: the bond institution plus nesting and population."""

    name: str
    family: str
    bond: CooperationSettings
    nesting: str = "none"
    motive_mix: tuple[tuple[str, float], ...] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.nesting not in NESTING_MODES:
            raise ValueError(f"nesting must be one of {NESTING_MODES}")
        if self.bond.kind != BOND_KIND:
            object.__setattr__(self, "bond", self.bond.with_changes(kind=BOND_KIND))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "bond": self.bond.to_dict(),
            "nesting": self.nesting,
            "motive_mix": [list(x) for x in self.motive_mix] if self.motive_mix else None,
            "description": self.description,
        }


def grid_cell_name(scope: float, exit_cost: float, recognition: float) -> str:
    return f"S{scope:.1f}_E{exit_cost:.1f}_R{recognition:.1f}"


def _primary_conditions() -> dict[str, Condition]:
    out: dict[str, Condition] = {}
    for recognition in PRIMARY_RECOGNITION:
        for exit_cost in PRIMARY_EXIT_COST:
            for scope in PRIMARY_SCOPE:
                name = grid_cell_name(scope, exit_cost, recognition)
                out[name] = Condition(
                    name=name,
                    family="grid",
                    bond=pair_bond(scope, exit_cost, recognition),
                    description=f"pair bond: scope {scope}, exit cost {exit_cost}, recognition {recognition}",
                )
    return out


def _preset_conditions() -> dict[str, Condition]:
    out: dict[str, Condition] = {}
    for name in ("marriage", "cohabitation", "friendship", "business_partnership"):
        out[f"preset_{name}"] = Condition(
            name=f"preset_{name}",
            family="preset",
            bond=preset(name),
            description=f"bond configured as the {name.replace('_', ' ')} preset",
        )
    return out


def _structure_conditions() -> dict[str, Condition]:
    marriage = preset("marriage")
    return {
        "marriage_ranked": Condition(
            name="marriage_ranked",
            family="symmetry",
            bond=marriage.with_changes(symmetry=0.0),
            description="marriage preset with fully rank-ordered payouts (symmetry 0)",
        ),
        "marriage_lineage": Condition(
            name="marriage_lineage",
            family="nesting",
            bond=marriage,
            nesting="lineage",
            description="marriage preset; couples that reproduce anchor a BORN-entry family unit",
        ),
        "marriage_arranged": Condition(
            name="marriage_arranged",
            family="nesting",
            bond=marriage.with_changes(entry=Entry.ARRANGED),
            nesting="arranged",
            description="marriage preset; families arrange matches for adult offspring",
        ),
    }


def _motive_conditions() -> dict[str, Condition]:
    marriage = preset("marriage")
    return {
        f"marriage_{m.value}": Condition(
            name=f"marriage_{m.value}",
            family="motive",
            bond=marriage,
            motive_mix=((m.value, 1.0),),
            description=f"marriage preset with an all-{m.value} population",
        )
        for m in Motive
    }


CONDITIONS: dict[str, Condition] = {
    **_primary_conditions(),
    **_preset_conditions(),
    **_structure_conditions(),
    **_motive_conditions(),
}

PRIMARY_CONDITION_ORDER: tuple[str, ...] = tuple(_primary_conditions())
SECONDARY_CONDITION_ORDER: tuple[str, ...] = tuple(_preset_conditions()) + tuple(_structure_conditions())
EXPLORATORY_CONDITION_ORDER: tuple[str, ...] = tuple(_motive_conditions())

#: Family-level unit used by the nesting conditions.
FAMILY_SETTINGS: CooperationSettings = preset("family")


@dataclass(frozen=True)
class RunConfig:
    """Everything one simulation needs."""

    condition: Condition
    seed: int
    world: WorldConfig = field(default_factory=WorldConfig)
    decisions: DecisionConfig = field(default_factory=DecisionConfig)
    economics: UnitEconomics = field(default_factory=UnitEconomics)
    unit_goals: UnitGoals = field(default_factory=UnitGoals)
    cooperative: bool = True

    @property
    def run_id(self) -> str:
        tag = self.condition.name if self.cooperative else "solo"
        return f"{tag}__s{self.seed}"

    def solo_twin(self) -> RunConfig:
        """The matched no-cooperation run (same seed, world and decision rules)."""
        return replace(self, cooperative=False)

    def effective_decisions(self) -> DecisionConfig:
        if self.condition.motive_mix:
            return replace(self.decisions, motive_mix=self.condition.motive_mix)
        return self.decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition.to_dict(),
            "seed": self.seed,
            "world": asdict(self.world),
            "decisions": asdict(self.effective_decisions()),
            "economics": asdict(self.economics),
            "unit_goals": asdict(self.unit_goals),
            "cooperative": self.cooperative,
        }


#: Diagnostic thresholds used for failure-mode prevalence (registered).
THRESHOLDS: DiagnosticThresholds = DiagnosticThresholds(
    min_steps=20,
    window=50,
    capture_share=0.6,
    capture_gap=0.25,
    hollow_throughput=0.05,
    fragility_lifetime=30,
    trap_steps=15,
    loss_tolerance=0.02,
)

#: Significance level for the confirmatory family (Holm-corrected).
ALPHA: float = 0.05

#: Unit survival is evaluated at this many ticks after formation.
SURVIVAL_HORIZON: int = 100
