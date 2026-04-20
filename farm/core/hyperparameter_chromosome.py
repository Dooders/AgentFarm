"""Typed chromosome model for evolvable agent hyperparameters.

This module complements :mod:`farm.core.genome`:
- ``Genome`` captures action weights and module state dicts.
- ``HyperparameterChromosome`` captures bounded, typed learning hyperparameters.

The schema is intentionally small and explicit so additional loci can be added
without changing the existing genome serialization abstraction.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


def validate_non_negative_mapping(label: str, mapping: Mapping[str, float]) -> None:
    """Raise ``ValueError`` if any value in ``mapping`` is negative, NaN, or infinite.

    Shared between ``mutate_chromosome``'s per-gene multiplier kwargs and
    :class:`farm.runners.adaptive_mutation.AdaptiveMutationConfig` so both
    code paths reject the same set of inputs with the same message format.
    """
    for name, value in mapping.items():
        if value < 0.0 or math.isnan(value) or math.isinf(value):
            raise ValueError(f"{label}['{name}'] must be a non-negative finite number.")


class GeneValueType(Enum):
    """Supported gene value kinds.

    Notes:
        Discrete and binary types are intentionally not implemented yet to keep
        the initial schema narrow and well-validated for real-valued parameters.
    """

    REAL = "real"


class GeneEncodingScale(str, Enum):
    """Supported scale transforms for real-valued genes."""

    LINEAR = "linear"
    LOG = "log"


class CrossoverMode(str, Enum):
    """Supported crossover operators for chromosome vectors.

    - ``SINGLE_POINT``: one random pivot splits the gene vector between parents.
    - ``UNIFORM``: each gene is independently drawn from either parent.
    - ``BLEND``: each evolvable gene is interpolated between parent values using a
      random blend factor drawn from ``[−blend_alpha, 1 + blend_alpha]`` (BLX-α).
      Values are clamped to gene bounds after blending.
    - ``MULTI_POINT``: ``num_crossover_points`` random pivots divide the gene vector
      into alternating segments inherited from each parent.
    """

    SINGLE_POINT = "single_point"
    UNIFORM = "uniform"
    BLEND = "blend"
    MULTI_POINT = "multi_point"


class MutationMode(str, Enum):
    """Supported mutation operators for real-valued genes."""

    GAUSSIAN = "gaussian"
    MULTIPLICATIVE = "multiplicative"


class BoundaryMode(str, Enum):
    """Strategies for handling out-of-bound values after mutation.

    - ``CLAMP`` (default): hard-clamp the mutated value to ``[min_value, max_value]``.
      Simple and safe, but can cause boundary collapse when many mutations push
      genes against a wall and they stay there.
    - ``REFLECT``: bounce the mutated value back from the boundary.  If the raw
      value overshoots by *d*, the reflected result is *d* inside the boundary.
      This preserves the bounded invariant while avoiding absorbing edge states.
    - ``INTERIOR_BIASED``: clamp first like ``CLAMP``, then nudge any value
      landing *exactly* on a boundary inward by a small random amount drawn
      uniformly from ``(0, interior_bias_fraction * span]``.  This prevents
      repeated exact-boundary hits without changing behavior for values that
      land strictly inside the range.  The ``interior_bias_fraction`` parameter
      of :func:`mutate_chromosome` controls the nudge magnitude (default
      ``1e-3``).
    """

    CLAMP = "clamp"
    REFLECT = "reflect"
    INTERIOR_BIASED = "interior_biased"


@dataclass(frozen=True)
class BoundaryPenaltyConfig:
    """Configuration for soft fitness penalties applied near gene boundaries.

    When ``enabled`` is ``True``, :func:`compute_boundary_penalty` returns a
    positive float that the caller **subtracts** from the raw fitness score.
    This discourages prolonged occupation of boundary values without completely
    forbidding them.

    Attributes:
        enabled: Whether soft boundary penalties are active.  Default ``False``
            so existing code is unaffected.
        penalty_strength: Maximum penalty applied to a single gene sitting
            exactly on a boundary.  Summed over all evolvable genes.  Default
            ``0.01``.
        near_boundary_threshold: Fraction of the gene's range within which the
            penalty ramps linearly from zero (at the inner edge) to
            ``penalty_strength`` (at the boundary itself).  Must be in
            ``(0, 0.5]``.  Default ``0.05`` (5 % of range on each side).
    """

    enabled: bool = False
    penalty_strength: float = 0.01
    near_boundary_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.penalty_strength < 0.0:
            raise ValueError("penalty_strength must be non-negative.")
        if not 0.0 < self.near_boundary_threshold <= 0.5:
            raise ValueError("near_boundary_threshold must be in (0, 0.5].")


@dataclass(frozen=True)
class GeneEncodingSpec:
    """Encoding settings for converting gene values to stored representations."""

    scale: GeneEncodingScale = GeneEncodingScale.LINEAR
    bit_width: Optional[int] = None

    def __post_init__(self) -> None:
        if self.bit_width is not None and self.bit_width <= 0:
            raise ValueError("bit_width must be positive when provided.")


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
    mutation_scale: float = 0.2
    mutation_probability: float = 0.1
    mutation_strategy: MutationMode = MutationMode.GAUSSIAN

    def __post_init__(self) -> None:
        self._validate_name()
        self._validate_real_bounds()
        self._validate_value(self.default, field_name="default")
        self._validate_value(self.value, field_name="value")
        self._validate_mutation_controls()

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

    def _validate_mutation_controls(self) -> None:
        if self.mutation_scale < 0.0:
            raise ValueError("mutation_scale must be non-negative.")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0 and 1.")

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
            mutation_scale=self.mutation_scale,
            mutation_probability=self.mutation_probability,
            mutation_strategy=self.mutation_strategy,
        )

    def normalize(
        self,
        value: Optional[float] = None,
        *,
        scale: GeneEncodingScale = GeneEncodingScale.LINEAR,
    ) -> float:
        """Normalize a real value to [0, 1] using the requested scale."""
        raw_value = self.value if value is None else float(value)
        self._validate_value(raw_value, field_name="value")
        return _normalize_real_value(raw_value, self.min_value, self.max_value, scale)

    def denormalize(
        self,
        normalized_value: float,
        *,
        scale: GeneEncodingScale = GeneEncodingScale.LINEAR,
    ) -> float:
        """Decode a normalized [0, 1] scalar into the gene's value range."""
        if not 0.0 <= normalized_value <= 1.0:
            raise ValueError("normalized_value must be within [0, 1].")
        return _denormalize_real_value(normalized_value, self.min_value, self.max_value, scale)

    def encode(
        self,
        value: Optional[float] = None,
        *,
        encoding: Optional[GeneEncodingSpec] = None,
    ) -> Union[float, int]:
        """Encode a gene value as normalized float or quantized integer."""
        resolved_encoding = encoding or default_encoding_spec_for_gene(self.name)
        normalized = self.normalize(value=value, scale=resolved_encoding.scale)
        if resolved_encoding.bit_width is None:
            return normalized

        max_bucket = _quantization_max_bucket(resolved_encoding.bit_width)
        return int(round(normalized * max_bucket))

    def decode(
        self,
        encoded_value: Union[float, int],
        *,
        encoding: Optional[GeneEncodingSpec] = None,
    ) -> float:
        """Decode a normalized float or quantized integer into real value."""
        resolved_encoding = encoding or default_encoding_spec_for_gene(self.name)
        normalized_value: float
        if resolved_encoding.bit_width is None:
            normalized_value = float(encoded_value)
        else:
            if not isinstance(encoded_value, (int, float)):
                raise TypeError("Quantized encoded value must be numeric.")
            if isinstance(encoded_value, float) and not encoded_value.is_integer():
                raise ValueError("Quantized encoded value must be an integer bucket.")
            bucket = int(encoded_value)
            max_bucket = _quantization_max_bucket(resolved_encoding.bit_width)
            if bucket < 0 or bucket > max_bucket:
                raise ValueError(
                    f"Quantized encoded value must be within [0, {max_bucket}], got {bucket}."
                )
            normalized_value = bucket / max_bucket

        return self.denormalize(normalized_value, scale=resolved_encoding.scale)

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
            "mutation_scale": self.mutation_scale,
            "mutation_probability": self.mutation_probability,
            "mutation_strategy": self.mutation_strategy.value,
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
            mutation_scale=float(data.get("mutation_scale", 0.2)),
            mutation_probability=float(data.get("mutation_probability", 0.1)),
            mutation_strategy=MutationMode(data.get("mutation_strategy", MutationMode.GAUSSIAN.value)),
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


# Default encoding policy by gene name.
# learning_rate uses log-space 8-bit quantization so equal bucket changes map to
# multiplicative LR changes, which is typically more meaningful than linear shifts.
# epsilon_decay and gamma use linear 8-bit quantization; their ranges are bounded
# and do not span multiple orders of magnitude in a way that requires log space.
DEFAULT_GENE_ENCODINGS: Dict[str, GeneEncodingSpec] = {
    "learning_rate": GeneEncodingSpec(scale=GeneEncodingScale.LOG, bit_width=8),
    "epsilon_decay": GeneEncodingSpec(scale=GeneEncodingScale.LINEAR, bit_width=8),
    "gamma": GeneEncodingSpec(scale=GeneEncodingScale.LINEAR, bit_width=8),
}

# Smallest positive IEEE-754 binary64; mirrors DecisionConfig allowing any (0, 1].
_EPSILON_DECAY_GENE_MIN = math.ldexp(1.0, -1074)


# Default hyperparameter loci.
# learning_rate, gamma, and epsilon_decay are enabled for evolution; memory_size
# remains a fixed placeholder documenting the extension path while keeping integer
# rounding concerns separate from the continuous-gene evolution phase.
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
        name="gamma",
        value_type=GeneValueType.REAL,
        value=0.99,
        min_value=0.0,
        max_value=1.0,
        default=0.99,
        evolvable=True,
    ),
    HyperparameterGene(
        name="epsilon_decay",
        value_type=GeneValueType.REAL,
        value=0.995,
        min_value=_EPSILON_DECAY_GENE_MIN,
        max_value=1.0,
        default=0.995,
        evolvable=True,
    ),
    HyperparameterGene(
        name="memory_size",
        value_type=GeneValueType.REAL,
        value=10000.0,
        min_value=1.0,
        max_value=1_000_000.0,
        default=10000.0,
        evolvable=False,
    ),
)


def default_hyperparameter_chromosome() -> HyperparameterChromosome:
    """Return default chromosome with baseline hyperparameter loci."""
    return HyperparameterChromosome(genes=DEFAULT_HYPERPARAMETER_GENES)


def default_encoding_spec_for_gene(gene_name: str) -> GeneEncodingSpec:
    """Return default encoding settings for a gene name."""
    return DEFAULT_GENE_ENCODINGS.get(gene_name, GeneEncodingSpec())


def _quantization_max_bucket(bit_width: int) -> int:
    return (1 << bit_width) - 1


def _normalize_real_value(
    value: float,
    min_value: float,
    max_value: float,
    scale: GeneEncodingScale,
) -> float:
    if min_value == max_value:
        return 0.0

    if scale is GeneEncodingScale.LINEAR:
        return (value - min_value) / (max_value - min_value)

    if min_value <= 0.0 or max_value <= 0.0 or value <= 0.0:
        raise ValueError("Log scale encoding requires strictly positive bounds and value.")

    min_log = math.log10(min_value)
    max_log = math.log10(max_value)
    value_log = math.log10(value)
    return (value_log - min_log) / (max_log - min_log)


def _denormalize_real_value(
    normalized_value: float,
    min_value: float,
    max_value: float,
    scale: GeneEncodingScale,
) -> float:
    if min_value == max_value:
        return min_value

    if scale is GeneEncodingScale.LINEAR:
        return min_value + normalized_value * (max_value - min_value)

    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError("Log scale decoding requires strictly positive bounds.")

    min_log = math.log10(min_value)
    max_log = math.log10(max_value)
    decoded_log = min_log + normalized_value * (max_log - min_log)
    return math.pow(10.0, decoded_log)


def _resolve_encoding_spec(
    gene_name: str,
    encoding_specs: Optional[Mapping[str, GeneEncodingSpec]],
) -> GeneEncodingSpec:
    if encoding_specs and gene_name in encoding_specs:
        return encoding_specs[gene_name]
    return default_encoding_spec_for_gene(gene_name)


def _selected_genes(
    chromosome: HyperparameterChromosome,
    include_fixed: bool,
) -> Tuple[HyperparameterGene, ...]:
    if include_fixed:
        return chromosome.genes
    return tuple(gene for gene in chromosome.genes if gene.evolvable)


def encode_chromosome(
    chromosome: HyperparameterChromosome,
    *,
    include_fixed: bool = False,
    encoding_specs: Optional[Mapping[str, GeneEncodingSpec]] = None,
) -> Dict[str, Union[int, float]]:
    """Encode genes by name to normalized floats or quantized integers."""
    return {
        gene.name: gene.encode(encoding=_resolve_encoding_spec(gene.name, encoding_specs))
        for gene in _selected_genes(chromosome, include_fixed=include_fixed)
    }


def decode_chromosome(
    encoded_values: Mapping[str, Union[int, float]],
    *,
    template: Optional[HyperparameterChromosome] = None,
    encoding_specs: Optional[Mapping[str, GeneEncodingSpec]] = None,
) -> HyperparameterChromosome:
    """Decode named encoded values into a chromosome instance."""
    chromosome_template = template or default_hyperparameter_chromosome()
    genes_by_name = {gene.name: gene for gene in chromosome_template.genes}

    unknown_names = set(encoded_values) - set(genes_by_name)
    if unknown_names:
        unknown = ", ".join(sorted(unknown_names))
        raise KeyError(f"Unknown encoded gene value(s): {unknown}")

    overrides: Dict[str, float] = {}
    for gene_name, encoded_value in encoded_values.items():
        gene = genes_by_name[gene_name]
        overrides[gene_name] = gene.decode(
            encoded_value,
            encoding=_resolve_encoding_spec(gene_name, encoding_specs),
        )
    return chromosome_template.with_overrides(overrides)


def encode_chromosome_vector(
    chromosome: HyperparameterChromosome,
    *,
    include_fixed: bool = False,
    encoding_specs: Optional[Mapping[str, GeneEncodingSpec]] = None,
) -> Tuple[Union[int, float], ...]:
    """Encode selected genes as an ordered vector."""
    return tuple(
        gene.encode(encoding=_resolve_encoding_spec(gene.name, encoding_specs))
        for gene in _selected_genes(chromosome, include_fixed=include_fixed)
    )


def decode_chromosome_vector(
    encoded_values: Sequence[Union[int, float]],
    *,
    template: Optional[HyperparameterChromosome] = None,
    include_fixed: bool = False,
    encoding_specs: Optional[Mapping[str, GeneEncodingSpec]] = None,
) -> HyperparameterChromosome:
    """Decode an ordered vector into a chromosome using template gene order."""
    chromosome_template = template or default_hyperparameter_chromosome()
    genes = _selected_genes(chromosome_template, include_fixed=include_fixed)
    if len(encoded_values) != len(genes):
        raise ValueError(
            f"encoded_values length {len(encoded_values)} does not match expected {len(genes)}."
        )

    overrides = {
        gene.name: gene.decode(
            encoded_value,
            encoding=_resolve_encoding_spec(gene.name, encoding_specs),
        )
        for gene, encoded_value in zip(genes, encoded_values)
    }
    return chromosome_template.with_overrides(overrides)


def hyperparameter_evolution_registry() -> Dict[str, bool]:
    """Return map of hyperparameters and whether they are evolvable."""
    chromosome = default_hyperparameter_chromosome()
    return {gene.name: gene.evolvable for gene in chromosome.genes}


def default_hyperparameter_registry() -> Dict[str, bool]:
    """Alias for ``hyperparameter_evolution_registry`` with shorter name."""
    return hyperparameter_evolution_registry()


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


def _apply_boundary(
    raw_value: float,
    min_value: float,
    max_value: float,
    mode: BoundaryMode,
    *,
    rng: Optional[random.Random] = None,
    interior_bias_fraction: float = 1e-3,
) -> float:
    """Apply a boundary strategy to a raw (possibly out-of-range) gene value.

    Args:
        raw_value: The value produced by a mutation operator before bounding.
        min_value: Gene's lower bound (inclusive).
        max_value: Gene's upper bound (inclusive).
        mode: The :class:`BoundaryMode` strategy to apply.
        rng: Optional :class:`random.Random` instance used by
            ``INTERIOR_BIASED`` mode.  Falls back to the module-level
            ``random`` singleton when ``None``.
        interior_bias_fraction: Fraction of the gene span used as the upper
            bound of the inward nudge in ``INTERIOR_BIASED`` mode.  Must be
            non-negative.  Only used when ``mode`` is
            :attr:`BoundaryMode.INTERIOR_BIASED`.

    Returns:
        A float guaranteed to lie within ``[min_value, max_value]``.
    """
    if mode is BoundaryMode.CLAMP:
        return max(min_value, min(max_value, raw_value))

    if mode is BoundaryMode.INTERIOR_BIASED:
        clamped = max(min_value, min(max_value, raw_value))
        span = max_value - min_value
        if span == 0.0 or interior_bias_fraction <= 0.0:
            return clamped
        resolved_rng = rng or random
        nudge_max = min(interior_bias_fraction * span, span)
        if clamped == min_value:
            return min_value + resolved_rng.uniform(0.0, nudge_max)
        if clamped == max_value:
            return max_value - resolved_rng.uniform(0.0, nudge_max)
        return clamped

    # REFLECT: bounce the value back from each boundary.
    # The folded space has period = 2 * span; the first half maps straight,
    # the second half maps in reverse (the "bounce").
    span = max_value - min_value
    if span == 0.0:
        return min_value
    period = 2.0 * span
    offset = raw_value - min_value
    # Python's % operator always returns a non-negative result when the divisor
    # is positive, so no negative-remainder correction is needed.
    mod = offset % period
    if mod <= span:
        return min_value + mod
    return max_value - (mod - span)


def mutate_chromosome(
    chromosome: HyperparameterChromosome,
    *,
    mutation_rate: Optional[float] = None,
    mutation_scale: Optional[float] = None,
    mutation_mode: Optional[Union[MutationMode, str]] = None,
    boundary_mode: Union[BoundaryMode, str] = BoundaryMode.CLAMP,
    interior_bias_fraction: float = 1e-3,
    per_gene_rate_multipliers: Optional[Mapping[str, float]] = None,
    per_gene_scale_multipliers: Optional[Mapping[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> HyperparameterChromosome:
    """Mutate evolvable genes using bounded real-valued perturbations.

    Args:
        chromosome: Source chromosome to mutate.
        mutation_rate: Probability of mutating each evolvable gene.  Overrides
            per-gene ``mutation_probability`` when provided.  Must be in
            ``[0, 1]``.
        mutation_scale: Perturbation scale.  Overrides per-gene
            ``mutation_scale`` when provided.  Must be non-negative.
        mutation_mode: Perturbation operator to use (``gaussian`` or
            ``multiplicative``).  Overrides per-gene ``mutation_strategy``
            when provided.
        boundary_mode: How to handle raw values that exceed gene bounds after
            mutation.  ``"clamp"`` (default) reproduces the original behavior;
            ``"reflect"`` bounces the value back off the boundary so edge states
            are not absorbing; ``"interior_biased"`` clamps first and then
            nudges values sitting exactly on a boundary inward by a small
            random amount.  See :class:`BoundaryMode`.
        interior_bias_fraction: Fraction of the gene span used as the upper
            bound of the inward nudge when ``boundary_mode`` is
            ``"interior_biased"``.  Must be non-negative.  Default ``1e-3``
            (0.1 % of gene range).  Ignored for other boundary modes.
        per_gene_rate_multipliers: Optional mapping of gene name to a
            non-negative multiplier applied to the resolved per-gene mutation
            probability.  After multiplication, the probability is clamped to
            ``[0, 1]``.  Genes absent from the mapping keep their resolved
            probability unchanged.  Supports per-gene adaptive mutation.
        per_gene_scale_multipliers: Optional mapping of gene name to a
            non-negative multiplier applied to the resolved per-gene mutation
            scale.  Genes absent from the mapping keep their resolved scale
            unchanged.  Supports per-gene adaptive mutation.
        rng: Optional :class:`random.Random` instance for deterministic tests.

    Notes:
        - ``gaussian``: additive Gaussian perturbation where sigma is
          ``mutation_scale * (max_value - min_value)``.
        - ``multiplicative``: legacy multiplicative perturbation using a
          uniform delta in ``[-mutation_scale, mutation_scale]``.
        - Boundary handling is controlled by ``boundary_mode`` (default:
          ``BoundaryMode.CLAMP``).
    """
    if mutation_rate is not None and not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")
    if mutation_scale is not None and mutation_scale < 0.0:
        raise ValueError("mutation_scale must be non-negative.")
    if interior_bias_fraction < 0.0:
        raise ValueError("interior_bias_fraction must be non-negative.")
    if per_gene_rate_multipliers:
        validate_non_negative_mapping("per_gene_rate_multipliers", per_gene_rate_multipliers)
    if per_gene_scale_multipliers:
        validate_non_negative_mapping("per_gene_scale_multipliers", per_gene_scale_multipliers)
    resolved_mode_override = MutationMode(mutation_mode) if mutation_mode is not None else None
    resolved_boundary_mode = BoundaryMode(boundary_mode)
    resolved_rng = rng or random
    updated_genes: List[HyperparameterGene] = []
    for gene in chromosome.genes:
        resolved_probability = mutation_rate if mutation_rate is not None else gene.mutation_probability
        resolved_scale = mutation_scale if mutation_scale is not None else gene.mutation_scale
        resolved_mode = resolved_mode_override if resolved_mode_override is not None else gene.mutation_strategy
        if per_gene_rate_multipliers and gene.name in per_gene_rate_multipliers:
            resolved_probability = max(
                0.0,
                min(1.0, resolved_probability * per_gene_rate_multipliers[gene.name]),
            )
        if per_gene_scale_multipliers and gene.name in per_gene_scale_multipliers:
            resolved_scale = max(0.0, resolved_scale * per_gene_scale_multipliers[gene.name])
        if not gene.evolvable or resolved_rng.random() >= resolved_probability:
            updated_genes.append(gene)
            continue

        if resolved_mode is MutationMode.GAUSSIAN:
            span = gene.max_value - gene.min_value
            sigma = span * resolved_scale
            raw_value = gene.value + resolved_rng.gauss(0.0, sigma)
        else:
            delta = resolved_rng.uniform(-resolved_scale, resolved_scale)
            raw_value = gene.value * (1.0 + delta)

        bounded_value = _apply_boundary(
            raw_value,
            gene.min_value,
            gene.max_value,
            resolved_boundary_mode,
            rng=resolved_rng,
            interior_bias_fraction=interior_bias_fraction,
        )
        updated_genes.append(gene.with_value(bounded_value))

    return HyperparameterChromosome(genes=tuple(updated_genes))


def compute_boundary_penalty(
    chromosome: HyperparameterChromosome,
    config: Optional[BoundaryPenaltyConfig] = None,
) -> float:
    """Compute a soft fitness penalty for genes sitting near their bounds.

    The returned value is intended to be **subtracted** from the caller's raw
    fitness score.  A gene resting exactly on a boundary incurs the full
    ``config.penalty_strength``; a gene at the inner edge of the threshold zone
    (distance from boundary == ``near_boundary_threshold``) incurs zero penalty.
    The ramp is linear between those two points.  The total penalty is the sum
    over all evolvable genes.

    When ``config.enabled`` is ``False`` (the default), this function returns
    ``0.0`` immediately so callers can include the call unconditionally.

    Args:
        chromosome: The chromosome to evaluate.
        config: Penalty settings.  Defaults to :class:`BoundaryPenaltyConfig`
            with ``enabled=False``.

    Returns:
        Non-negative float representing the total penalty (0.0 when disabled).

    Example::

        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.02)
        penalty = compute_boundary_penalty(chromosome, cfg)
        adjusted_fitness = raw_fitness - penalty
    """
    resolved_config = config if config is not None else BoundaryPenaltyConfig()
    if not resolved_config.enabled:
        return 0.0

    total_penalty = 0.0
    threshold = resolved_config.near_boundary_threshold
    strength = resolved_config.penalty_strength

    for gene in chromosome.genes:
        if not gene.evolvable:
            continue
        span = gene.max_value - gene.min_value
        if span == 0.0:
            continue
        normalized = (gene.value - gene.min_value) / span
        distance_from_boundary = min(normalized, 1.0 - normalized)
        if distance_from_boundary < threshold:
            fraction = 1.0 - distance_from_boundary / threshold
            total_penalty += strength * fraction

    return total_penalty


def crossover_chromosomes(
    parent_a: HyperparameterChromosome,
    parent_b: HyperparameterChromosome,
    *,
    mode: Union[CrossoverMode, str] = CrossoverMode.SINGLE_POINT,
    include_fixed: bool = False,
    uniform_parent_b_probability: float = 0.5,
    blend_alpha: float = 0.5,
    num_crossover_points: int = 2,
    rng: Optional[random.Random] = None,
) -> HyperparameterChromosome:
    """Create a child chromosome by crossing two parent gene vectors.

    Args:
        parent_a: First parent chromosome.
        parent_b: Second parent chromosome (must be schema-compatible with *parent_a*).
        mode: Crossover operator to apply.  One of :class:`CrossoverMode`.
        include_fixed: When ``True``, non-evolvable genes participate in crossover
            alongside evolvable ones.
        uniform_parent_b_probability: Probability of inheriting each gene from
            *parent_b* under ``UNIFORM`` mode.  Must be in ``[0, 1]``.
        blend_alpha: Extent parameter for ``BLEND`` (BLX-α) crossover.  The random
            blend factor is drawn uniformly from ``[−blend_alpha, 1 + blend_alpha]``,
            allowing offspring to explore slightly beyond the interval spanned by the
            parents.  ``0.0`` restricts the child strictly between the two parent
            values; ``0.5`` is a common heuristic.  Must be non-negative.
        num_crossover_points: Number of pivot points used by ``MULTI_POINT``
            crossover.  Must be at least 1.  Values larger than the number of
            selected genes minus 1 are clamped automatically.
        rng: Optional seeded :class:`random.Random` instance for reproducible
            results.  Falls back to the module-level ``random`` singleton when
            ``None``.

    Returns:
        A new :class:`HyperparameterChromosome` whose genes are derived from the
        selected crossover strategy.

    Notes:
        This operator intentionally lives outside ``Genome`` to keep action-set
        genome evolution separate from hyperparameter chromosome evolution.
    """
    if not 0.0 <= uniform_parent_b_probability <= 1.0:
        raise ValueError("uniform_parent_b_probability must be between 0 and 1.")

    _validate_compatible_chromosomes(parent_a, parent_b)
    resolved_mode = CrossoverMode(mode)
    if resolved_mode is CrossoverMode.BLEND and blend_alpha < 0.0:
        raise ValueError("blend_alpha must be non-negative.")
    if resolved_mode is CrossoverMode.MULTI_POINT and num_crossover_points < 1:
        raise ValueError("num_crossover_points must be at least 1.")
    resolved_rng = rng or random

    selected_indices = [
        idx
        for idx, gene in enumerate(parent_a.genes)
        if include_fixed or gene.evolvable
    ]
    if not selected_indices:
        return HyperparameterChromosome(genes=tuple(parent_a.genes))

    child_genes = list(parent_a.genes)
    if resolved_mode is CrossoverMode.SINGLE_POINT:
        if len(selected_indices) == 1:
            selected_parent = parent_b if resolved_rng.random() < 0.5 else parent_a
            child_genes[selected_indices[0]] = selected_parent.genes[selected_indices[0]]
        else:
            pivot = resolved_rng.randint(1, len(selected_indices) - 1)
            for selected_position, gene_idx in enumerate(selected_indices):
                source = parent_a if selected_position < pivot else parent_b
                child_genes[gene_idx] = source.genes[gene_idx]
    elif resolved_mode is CrossoverMode.BLEND:
        for gene_idx in selected_indices:
            gene_a = parent_a.genes[gene_idx]
            gene_b = parent_b.genes[gene_idx]
            lo = min(gene_a.value, gene_b.value)
            hi = max(gene_a.value, gene_b.value)
            span = hi - lo
            lower_bound = lo - blend_alpha * span
            upper_bound = hi + blend_alpha * span
            raw_value = resolved_rng.uniform(lower_bound, upper_bound)
            clamped = max(gene_a.min_value, min(gene_a.max_value, raw_value))
            child_genes[gene_idx] = gene_a.with_value(clamped)
    elif resolved_mode is CrossoverMode.MULTI_POINT:
        n = len(selected_indices)
        effective_points = min(num_crossover_points, n - 1) if n > 1 else 0
        if effective_points == 0:
            # Only one gene is selected; no pivot is possible, so pick randomly
            # between parents (mirroring SINGLE_POINT behaviour for n=1).
            selected_parent = parent_b if resolved_rng.random() < 0.5 else parent_a
            child_genes[selected_indices[0]] = selected_parent.genes[selected_indices[0]]
        else:
            pivot_positions = sorted(
                resolved_rng.sample(range(1, n), effective_points)
            )
            # Alternate segments: even-indexed segments from parent_a, odd from parent_b
            segment = 0
            for selected_position, gene_idx in enumerate(selected_indices):
                if segment < len(pivot_positions) and selected_position >= pivot_positions[segment]:
                    segment += 1
                source = parent_a if segment % 2 == 0 else parent_b
                child_genes[gene_idx] = source.genes[gene_idx]
    else:
        for gene_idx in selected_indices:
            if resolved_rng.random() < uniform_parent_b_probability:
                child_genes[gene_idx] = parent_b.genes[gene_idx]

    return HyperparameterChromosome(genes=tuple(child_genes))


def _validate_compatible_chromosomes(
    parent_a: HyperparameterChromosome,
    parent_b: HyperparameterChromosome,
) -> None:
    if len(parent_a.genes) != len(parent_b.genes):
        raise ValueError("Chromosomes must have the same number of genes for crossover.")

    for index, (gene_a, gene_b) in enumerate(zip(parent_a.genes, parent_b.genes)):
        if gene_a.name != gene_b.name:
            raise ValueError(
                f"Chromosome gene mismatch at index {index}: "
                f"'{gene_a.name}' != '{gene_b.name}'."
            )
        if (
            gene_a.value_type != gene_b.value_type
            or gene_a.min_value != gene_b.min_value
            or gene_a.max_value != gene_b.max_value
            or gene_a.evolvable != gene_b.evolvable
        ):
            raise ValueError(
                f"Chromosome gene '{gene_a.name}' has incompatible schema across parents."
            )


def apply_chromosome_to_learning_config(
    learning_config: Any,
    chromosome: HyperparameterChromosome,
) -> Any:
    """Return a copy of learning config with chromosome values applied.

    Supports Pydantic v2 models via ``model_copy(update=...)`` and falls back to
    a deep copy with attributes updated for plain objects, ensuring the original
    is never mutated regardless of config type.

    Integration note:
        ``Genome.from_agent`` captures action weights and module state only, not this
        hyperparameter chromosome. Apply chromosome values to decision config before
        constructing offspring so module initialization sees the evolved settings.
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

    copied = copy.deepcopy(learning_config)
    for key, value in updates.items():
        setattr(copied, key, value)
    return copied
