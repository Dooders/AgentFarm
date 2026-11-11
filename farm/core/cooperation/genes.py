"""Heritable cooperation loci for agents in the spatial world (Chromosome D).

In the spatial AgentFarm world an agent's attitude to cooperation is part of
its :class:`~farm.core.hyperparameter_chromosome.HyperparameterChromosome`,
so it is inherited, mutated and crossed over like every other gene. Three
groups of loci are added when cooperation is switched on:

* **Settings** -- the terms of the institution an agent offers when it
  founds a unit: ``coop_scope``, ``coop_exit_cost``, ``coop_recognition``,
  ``coop_symmetry``, ``coop_exclusivity`` (exclusive when ``>= 0.5``) and
  ``coop_max_members`` (rounded to an integer).
* **Motives** -- ``motive_political`` ... ``motive_religious``: the weights of
  the agent's :class:`~farm.core.cooperation.goals.MemberGoals` mixture,
  normalised on use. They decide which settings the agent finds rewarding and
  how much it values like-minded partners.
* **Propensities** -- ``bond_weight`` and ``leave_weight``: the innate prior
  weights of the ``bond`` and ``leave`` actions, which the learned policy
  then reshapes. They are bounded at 0.3 (against other action weights that
  sum to 1) so the prior cannot do the policy's work: at the bound an agent
  bonds on at most a fifth of its steps before learning anything.

The loci are appended to a chromosome only in runs with cooperation enabled
(:func:`extend_chromosome`), so runs without cooperation keep exactly the
historical chromosome and random-number stream.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from farm.core.cooperation.goals import MOTIVES, MemberGoals, Motive
from farm.core.cooperation.settings import CooperationSettings, Entry
from farm.core.hyperparameter_chromosome import (
    GeneValueType,
    HyperparameterChromosome,
    HyperparameterGene,
)

#: Settings loci, in the order they appear on the chromosome.
SETTING_GENE_NAMES: tuple[str, ...] = (
    "coop_scope",
    "coop_exit_cost",
    "coop_recognition",
    "coop_symmetry",
    "coop_exclusivity",
    "coop_max_members",
)

#: Motive loci, one per :class:`~farm.core.cooperation.goals.Motive`.
MOTIVE_GENE_NAMES: tuple[str, ...] = tuple(f"motive_{m.value}" for m in MOTIVES)

#: Innate action-prior loci for the two cooperation actions.
PROPENSITY_GENE_NAMES: tuple[str, ...] = ("bond_weight", "leave_weight")

#: Every cooperation locus, in chromosome order.
COOPERATION_GENE_NAMES: tuple[str, ...] = SETTING_GENE_NAMES + MOTIVE_GENE_NAMES + PROPENSITY_GENE_NAMES

#: ``coop_exclusivity`` at or above this value makes the offered unit exclusive.
EXCLUSIVITY_CUTOFF: float = 0.5

#: Bounds and defaults: ``name -> (default, min, max)``.
GENE_BOUNDS: dict[str, tuple[float, float, float]] = {
    "coop_scope": (0.5, 0.0, 1.0),
    "coop_exit_cost": (0.3, 0.0, 1.0),
    "coop_recognition": (0.5, 0.0, 1.0),
    "coop_symmetry": (1.0, 0.0, 1.0),
    "coop_exclusivity": (1.0, 0.0, 1.0),
    "coop_max_members": (2.0, 2.0, 6.0),
    **{name: (0.5, 0.0, 1.0) for name in MOTIVE_GENE_NAMES},
    "bond_weight": (0.05, 0.0, 0.3),
    "leave_weight": (0.03, 0.0, 0.3),
}


def cooperation_genes(values: Mapping[str, float] | None = None) -> tuple[HyperparameterGene, ...]:
    """The cooperation loci with default (or ``values``) settings, all evolvable."""
    values = dict(values or {})
    unknown = set(values) - set(COOPERATION_GENE_NAMES)
    if unknown:
        raise KeyError(f"unknown cooperation gene(s): {', '.join(sorted(unknown))}")
    genes = []
    for name in COOPERATION_GENE_NAMES:
        default, lo, hi = GENE_BOUNDS[name]
        genes.append(
            HyperparameterGene(
                name=name,
                value_type=GeneValueType.REAL,
                value=float(values.get(name, default)),
                min_value=lo,
                max_value=hi,
                default=default,
                evolvable=True,
            )
        )
    return tuple(genes)


def has_cooperation_genes(chromosome: HyperparameterChromosome | None) -> bool:
    if chromosome is None:
        return False
    names = {gene.name for gene in chromosome.genes}
    return all(name in names for name in COOPERATION_GENE_NAMES)


def extend_chromosome(
    chromosome: HyperparameterChromosome,
    values: Mapping[str, float] | None = None,
) -> HyperparameterChromosome:
    """Append any missing cooperation loci to ``chromosome``.

    Loci already present are left untouched (``values`` only fills missing
    ones), so applying this to an offspring that inherited the loci is a
    no-op.
    """
    present = {gene.name for gene in chromosome.genes}
    missing = [gene for gene in cooperation_genes(values) if gene.name not in present]
    if not missing:
        return chromosome
    return HyperparameterChromosome(genes=tuple(chromosome.genes) + tuple(missing))


def sample_gene_values(
    rng: random.Random,
    names: Sequence[str] = COOPERATION_GENE_NAMES,
) -> dict[str, float]:
    """Draw each named locus uniformly within its bounds."""
    out: dict[str, float] = {}
    for name in names:
        _, lo, hi = GENE_BOUNDS[name]
        out[name] = rng.uniform(lo, hi)
    return out


def gene_values(chromosome: HyperparameterChromosome | None) -> dict[str, float]:
    """Cooperation gene values, falling back to defaults for missing loci."""
    out: dict[str, float] = {}
    for name in COOPERATION_GENE_NAMES:
        gene = chromosome.get_gene(name) if chromosome is not None else None
        out[name] = float(gene.value) if gene is not None else GENE_BOUNDS[name][0]
    return out


def _clip(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(value)))


def settings_from_values(values: Mapping[str, float], kind: str = "bond") -> CooperationSettings:
    """The institution an agent with these gene values offers."""
    max_members = round(_clip(values["coop_max_members"], 2.0, 6.0))
    return CooperationSettings(
        kind=kind,
        min_members=2,
        max_members=max_members,
        scope=_clip(values["coop_scope"], 0.0, 1.0),
        exit_cost=_clip(values["coop_exit_cost"], 0.0, 1.0),
        recognition=_clip(values["coop_recognition"], 0.0, 1.0),
        exclusive=float(values["coop_exclusivity"]) >= EXCLUSIVITY_CUTOFF,
        entry=Entry.CHOSEN,
        symmetry=_clip(values["coop_symmetry"], 0.0, 1.0),
    )


def blended_settings(founder_values: Sequence[Mapping[str, float]], kind: str = "bond") -> CooperationSettings:
    """Terms of a unit founded by several agents: the mean of their settings loci."""
    if not founder_values:
        raise ValueError("at least one founder is required")
    mean = {name: sum(float(v[name]) for v in founder_values) / len(founder_values) for name in SETTING_GENE_NAMES}
    return settings_from_values(mean, kind)


def goals_from_values(values: Mapping[str, float]) -> MemberGoals:
    """The agent's motive mixture; uniform when every motive locus is zero."""
    weights = {m: max(0.0, float(values[f"motive_{m.value}"])) for m in MOTIVES}
    if sum(weights.values()) <= 0.0:
        weights = {m: 1.0 for m in MOTIVES}
    return MemberGoals(weights)


def motive_similarity(a: MemberGoals, b: MemberGoals) -> float:
    """Like-mindedness of two motive mixtures, in ``[-1, 1]`` (``1 - L1 distance``)."""
    distance = sum(abs(a.weight(m) - b.weight(m)) for m in MOTIVES)
    return 1.0 - distance


def dominant_motive(values: Mapping[str, float]) -> Motive:
    return goals_from_values(values).dominant


__all__ = [
    "COOPERATION_GENE_NAMES",
    "EXCLUSIVITY_CUTOFF",
    "GENE_BOUNDS",
    "MOTIVE_GENE_NAMES",
    "PROPENSITY_GENE_NAMES",
    "SETTING_GENE_NAMES",
    "blended_settings",
    "cooperation_genes",
    "dominant_motive",
    "extend_chromosome",
    "gene_values",
    "goals_from_values",
    "has_cooperation_genes",
    "motive_similarity",
    "sample_gene_values",
    "settings_from_values",
]
