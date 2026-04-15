"""Typed chromosome model for evolvable agent hyperparameters.

This module complements :mod:`farm.core.genome`:
- ``Genome`` captures action weights and module state dicts.
- ``HyperparameterChromosome`` captures bounded, typed learning hyperparameters.

The schema is intentionally small and explicit so additional loci can be added
without changing the existing genome serialization abstraction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GeneValueType(str, Enum):
    """Supported gene value kinds.

    Notes:
        Discrete and binary types are intentionally not implemented yet to keep
        the initial schema narrow and well-validated for real-valued parameters.
    """

    REAL = "real"


@dataclass(frozen=True)
class HyperparameterGene:
    """Single typed and bounded hyperparameter locus."""

    name: str
    value_type: GeneValueType
    value: float
    min_value: float
    max_value: float
    default: float
    evolvable: bool = True

    def __post_init__(self) -> None:
        self._validate_name()
        self._validate_real_bounds()
        self._validate_value(self.default, field_name="default")
        self._validate_value(self.value, field_name="value")

    def _validate_name(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Gene name must be a non-empty string.")

    def _validate_real_bounds(self) -> None:
        if self.value_type is not GeneValueType.REAL:
            raise ValueError(f"Unsupported gene value type: {self.value_type}.")
        if self.min_value > self.max_value:
            raise ValueError(
                f"Gene '{self.name}' has invalid bounds: min_value > max_value."
            )

    def _validate_value(self, value: float, field_name: str) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError(
                f"Gene '{self.name}' {field_name} must be numeric for real-valued genes."
            )
        numeric_value = float(value)
        if numeric_value < self.min_value or numeric_value > self.max_value:
            raise ValueError(
                f"Gene '{self.name}' {field_name}={numeric_value} outside bounds "
                f"[{self.min_value}, {self.max_value}]."
            )

    def with_value(self, value: float) -> "HyperparameterGene":
        """Return a copy of this gene with a validated replacement value."""
        return HyperparameterGene(
            name=self.name,
            value_type=self.value_type,
            value=value,
            min_value=self.min_value,
            max_value=self.max_value,
            default=self.default,
            evolvable=self.evolvable,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gene data to a plain dictionary."""
        return {
            "name": self.name,
            "value_type": self.value_type.value,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default": self.default,
            "evolvable": self.evolvable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterGene":
        """Deserialize a gene from dictionary form."""
        return cls(
            name=data["name"],
            value_type=GeneValueType(data["value_type"]),
            value=float(data["value"]),
            min_value=float(data["min_value"]),
            max_value=float(data["max_value"]),
            default=float(data["default"]),
            evolvable=bool(data.get("evolvable", True)),
        )


@dataclass(frozen=True)
class HyperparameterChromosome:
    """Ordered set of hyperparameter loci with strong invariants."""

    genes: Tuple[HyperparameterGene, ...]

    def __post_init__(self) -> None:
        names = [gene.name for gene in self.genes]
        if len(set(names)) != len(names):
            raise ValueError("Chromosome cannot contain duplicate gene names.")

    def get_gene(self, name: str) -> Optional[HyperparameterGene]:
        """Return a gene by name when present."""
        for gene in self.genes:
            if gene.name == name:
                return gene
        return None

    def get_value(self, name: str) -> float:
        """Get the current value for a named gene."""
        gene = self.get_gene(name)
        if gene is None:
            raise KeyError(f"Unknown gene: {name}")
        return gene.value

    def evolvable_gene_names(self) -> Tuple[str, ...]:
        """Return names of loci marked as evolvable."""
        return tuple(gene.name for gene in self.genes if gene.evolvable)

    def fixed_gene_names(self) -> Tuple[str, ...]:
        """Return names of loci that remain globally fixed."""
        return tuple(gene.name for gene in self.genes if not gene.evolvable)

    def with_overrides(self, overrides: Dict[str, float]) -> "HyperparameterChromosome":
        """Return a new chromosome with validated value overrides."""
        updated_genes: List[HyperparameterGene] = []
        for gene in self.genes:
            if gene.name in overrides:
                updated_genes.append(gene.with_value(overrides[gene.name]))
            else:
                updated_genes.append(gene)

        unknown_names = set(overrides) - {gene.name for gene in self.genes}
        if unknown_names:
            unknown = ", ".join(sorted(unknown_names))
            raise KeyError(f"Unknown gene override(s): {unknown}")

        return HyperparameterChromosome(genes=tuple(updated_genes))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chromosome to dictionary form."""
        return {"genes": [gene.to_dict() for gene in self.genes]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterChromosome":
        """Deserialize chromosome from dictionary form."""
        genes = tuple(HyperparameterGene.from_dict(raw_gene) for raw_gene in data["genes"])
        return cls(genes=genes)


# Default hyperparameter loci.
# learning_rate is enabled for evolution now; others are placeholders that
# document the extension path while remaining fixed in global config.
DEFAULT_HYPERPARAMETER_GENES: Tuple[HyperparameterGene, ...] = (
    HyperparameterGene(
        name="learning_rate",
        value_type=GeneValueType.REAL,
        value=0.001,
        min_value=1e-6,
        max_value=1.0,
        default=0.001,
        evolvable=True,
    ),
    HyperparameterGene(
        name="epsilon_decay",
        value_type=GeneValueType.REAL,
        value=0.995,
        min_value=0.9,
        max_value=0.9999,
        default=0.995,
        evolvable=False,
    ),
    HyperparameterGene(
        name="memory_size",
        value_type=GeneValueType.REAL,
        value=2000.0,
        min_value=128.0,
        max_value=50000.0,
        default=2000.0,
        evolvable=False,
    ),
)


def default_hyperparameter_chromosome() -> HyperparameterChromosome:
    """Return default chromosome with baseline hyperparameter loci."""
    return HyperparameterChromosome(genes=DEFAULT_HYPERPARAMETER_GENES)


def hyperparameter_evolution_registry() -> Dict[str, bool]:
    """Return map of hyperparameters and whether they are evolvable."""
    chromosome = default_hyperparameter_chromosome()
    return {gene.name: gene.evolvable for gene in chromosome.genes}


def chromosome_from_values(
    values: Optional[Dict[str, float]] = None,
) -> HyperparameterChromosome:
    """Build a chromosome from defaults with optional value overrides."""
    chromosome = default_hyperparameter_chromosome()
    if not values:
        return chromosome
    return chromosome.with_overrides(values)


def chromosome_from_learning_config(learning_config: Any) -> HyperparameterChromosome:
    """Build a chromosome from any object with matching learning attributes."""
    overrides: Dict[str, float] = {}
    for gene in DEFAULT_HYPERPARAMETER_GENES:
        if hasattr(learning_config, gene.name):
            overrides[gene.name] = getattr(learning_config, gene.name)
    return chromosome_from_values(overrides)


def mutate_chromosome(
    chromosome: HyperparameterChromosome,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.2,
) -> HyperparameterChromosome:
    """Mutate evolvable genes by bounded multiplicative perturbation."""
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")
    if mutation_scale < 0.0:
        raise ValueError("mutation_scale must be non-negative.")

    updated_genes: List[HyperparameterGene] = []
    for gene in chromosome.genes:
        if not gene.evolvable or random.random() >= mutation_rate:
            updated_genes.append(gene)
            continue

        delta = random.uniform(-mutation_scale, mutation_scale)
        raw_value = gene.value * (1.0 + delta)
        bounded_value = max(gene.min_value, min(gene.max_value, raw_value))
        updated_genes.append(gene.with_value(bounded_value))

    return HyperparameterChromosome(genes=tuple(updated_genes))


def apply_chromosome_to_learning_config(
    learning_config: Any,
    chromosome: HyperparameterChromosome,
) -> Any:
    """Return a copy of learning config with chromosome values applied.

    Supports Pydantic v2 models via ``model_copy(update=...)`` and falls back to
    in-place attribute assignment for plain objects.
    """
    updates: Dict[str, Any] = {}
    for gene in chromosome.genes:
        if hasattr(learning_config, gene.name):
            current_value = getattr(learning_config, gene.name)
            if isinstance(current_value, int):
                updates[gene.name] = int(round(gene.value))
            else:
                updates[gene.name] = gene.value

    model_copy = getattr(learning_config, "model_copy", None)
    if callable(model_copy):
        return learning_config.model_copy(update=updates)

    for key, value in updates.items():
        setattr(learning_config, key, value)
    return learning_config
