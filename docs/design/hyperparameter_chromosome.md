# Hyperparameter Chromosome Design

This document explains the typed hyperparameter chromosome model, how it is currently wired into agent reproduction, and how to extend it for additional genes and mutation strategies.

## Why this exists

`farm/core/genome.py` already serializes:
- action weights (`action_set`)
- module state dicts (`module_states`)
- agent metadata (`agent_type`, resources, health)

That is useful for full agent snapshots, but it does not provide a small typed representation of *tunable hyperparameters* with explicit bounds and validation rules.

`farm/core/hyperparameter_chromosome.py` adds that explicit model in parallel.

## Core model

The chromosome model is intentionally narrow and strict:

- `GeneValueType`
  - currently supports `real` (with extension points for discrete/binary later)
- `GeneEncodingScale` *(new)*
  - `LINEAR` — uniform spacing across the gene's numeric range
  - `LOG` — log₁₀ spacing; equal bucket steps correspond to multiplicative changes (requires strictly positive bounds and value)
- `GeneEncodingSpec` *(new)*
  - frozen dataclass: `scale` (`GeneEncodingScale`) + optional `bit_width`
  - when `bit_width` is set, the normalized float is quantized to an integer in `[0, 2^bit_width − 1]`
  - `bit_width` must be positive (validated at construction)
- `HyperparameterGene`
  - name, type, value, min/max bounds, default, and `evolvable` flag
  - per-gene mutation controls:
    - `mutation_scale` (default `0.2`)
    - `mutation_probability` (default `0.1`)
    - `mutation_strategy` (`MutationMode`, default `GAUSSIAN`)
  - validates:
    - non-empty name
    - valid min/max range
    - in-range numeric value and default
    - valid mutation controls (non-negative scale, probability in `[0, 1]`)
  - **new methods** (real-valued genes only):
    - `normalize(value, *, scale)` → `float` in `[0, 1]`
    - `denormalize(normalized_value, *, scale)` → `float` in gene range
    - `encode(value, *, encoding)` → `float` or `int` (uses `default_encoding_spec_for_gene` if omitted)
    - `decode(encoded_value, *, encoding)` → `float`
- `HyperparameterChromosome`
  - ordered tuple of `HyperparameterGene`
  - enforces unique gene names
  - supports:
    - name lookup (`get_gene`, `get_value`)
    - evolvable/fixed partitioning
    - validated overrides (`with_overrides`)
    - serialization (`to_dict`, `from_dict`)

## Default gene registry

The default registry is defined in `DEFAULT_HYPERPARAMETER_GENES`:

- `learning_rate` (evolvable) — range `[1e-6, 1.0]`, encoded with **log-scale 8-bit quantization** by default so that equal bucket steps map to multiplicative LR changes
- `gamma` (evolvable) — discount factor, range `[0.0, 1.0]`, default `0.99`, encoded with **linear 8-bit quantization**
- `epsilon_decay` (evolvable) — exploration decay rate, range `(0, 1.0]`, default `0.995`, encoded with **linear 8-bit quantization**
- `memory_size` (fixed placeholder) — integer-rounding concerns are kept separate from the continuous-gene evolution phase

Default encoding policies per gene name are stored in `DEFAULT_GENE_ENCODINGS`:

```python
DEFAULT_GENE_ENCODINGS = {
    "learning_rate": GeneEncodingSpec(scale=GeneEncodingScale.LOG, bit_width=8),
    "epsilon_decay": GeneEncodingSpec(scale=GeneEncodingScale.LINEAR, bit_width=8),
    "gamma": GeneEncodingSpec(scale=GeneEncodingScale.LINEAR, bit_width=8),
}
```

Genes not listed in `DEFAULT_GENE_ENCODINGS` fall back to `GeneEncodingSpec()` (linear scale, no quantization).

Helpers:

- `default_hyperparameter_chromosome()`
- `hyperparameter_evolution_registry()`
- `default_hyperparameter_registry()` *(alias for the default evolution registry)*
- `chromosome_from_values()`
- `chromosome_from_learning_config()`
- `default_encoding_spec_for_gene(gene_name)` — look up the default `GeneEncodingSpec` for a gene name

## Runtime wiring in reproduction

`farm/core/agent/core.py` now uses the chromosome as part of offspring creation.

On agent init:
- `self.hyperparameter_chromosome` is created from `self.config.decision`.

On `AgentCore.reproduce()`:
1. Use the parent's stored chromosome (`self.hyperparameter_chromosome`) as the
   source of inheritable hyperparameters.
   - If that attribute is missing, derive it from `self.config.decision` and
     sync it back onto the parent.
2. Mutate evolvable genes via `mutate_chromosome(...)`.
3. Deep-copy parent config.
4. Apply chromosome values to child decision config via `apply_chromosome_to_learning_config(...)`.
5. Deduct the reproduction resource cost (`offspring_cost`) from the parent agent.
   - If the resource component reports that the deduction failed (insufficient resources), `reproduce()` returns `False` immediately without creating offspring.
   - If offspring creation subsequently raises an exception after the cost has
     been deducted, reproduction attempts a partial-add rollback when needed and
     then refunds by calling `resource_comp.add(offspring_cost)`.
   - Refund is only suppressed when rollback fails *and* the offspring still
     appears present in environment tracking structures (unresolved state).
6. Create offspring with the child config.
7. Store the resulting chromosome on the offspring.

This keeps:
- existing action/module-state genome behavior intact
- hyperparameter evolution explicit and typed

## Mutation behavior

`mutate_chromosome(chromosome, mutation_rate=None, mutation_scale=None, mutation_mode=None, boundary_mode="clamp")`:

- only mutates genes where `evolvable=True`
- resolves mutation settings per gene by default:
  - probability from `gene.mutation_probability`
  - scale from `gene.mutation_scale`
  - strategy from `gene.mutation_strategy`
- optional global arguments override per-gene values when provided
- supports two real-valued mutation operators:
  - `gaussian` (default):
    - `new_value = old_value + Normal(0, mutation_scale * (max_value - min_value))`
  - `multiplicative` (legacy mode):
    - `new_value = old_value * (1 + uniform(-scale, scale))`
- boundary handling is controlled by `boundary_mode` (see below)

## Boundary handling

When a mutation produces a raw value outside `[min_value, max_value]`, the
`boundary_mode` argument to `mutate_chromosome` determines what happens.

### `BoundaryMode.CLAMP` (default)

The raw value is hard-clamped:

```python
bounded = max(min_value, min(max_value, raw_value))
```

Simple and safe, but can cause **boundary collapse**: repeated mutations push a
gene to a wall and it becomes "absorbed" there, eliminating diversity.

### `BoundaryMode.REFLECT`

The raw value is folded back from the boundary like a billiard ball:

- overshoot by *d* above `max_value` → result is `max_value − d` (bounced back)
- works symmetrically for `min_value`
- multiple reflections handled correctly via modular arithmetic

This avoids absorbing edge states while still keeping the gene inside
`[min_value, max_value]`.

```python
mutated = mutate_chromosome(chromosome, mutation_rate=0.1, boundary_mode="reflect")
```

**Recommended default**: `CLAMP` for stability in early experiments;
`REFLECT` when you observe boundary collapse (genes sticking at min/max).

### Soft boundary penalties

`compute_boundary_penalty(chromosome, config)` returns a non-negative float
that should be **subtracted** from the raw fitness score.

```python
from farm.core.hyperparameter_chromosome import BoundaryPenaltyConfig, compute_boundary_penalty

cfg = BoundaryPenaltyConfig(
    enabled=True,
    penalty_strength=0.01,       # max penalty per gene
    near_boundary_threshold=0.05, # 5% of range on each side
)
adjusted_fitness = raw_fitness - compute_boundary_penalty(chromosome, cfg)
```

Penalty ramps linearly:

| Gene position (normalized) | Penalty fraction |
|---|---|
| exactly on boundary (0.0 or 1.0) | 1.0 × `penalty_strength` |
| `near_boundary_threshold` inside boundary | 0.0 |

The total penalty is summed over all `evolvable=True` genes.  Fixed genes never
contribute.  The function returns `0.0` immediately when `enabled=False`
(default) so callers can include the call unconditionally.

**`BoundaryPenaltyConfig` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Whether to compute a penalty at all |
| `penalty_strength` | `float` | `0.01` | Maximum per-gene penalty |
| `near_boundary_threshold` | `float` | `0.05` | Fraction of gene range to consider "near boundary" |

**Recommended defaults**: start with `penalty_strength=0.01` and
`near_boundary_threshold=0.05`.  Increase `penalty_strength` if boundary
collapse persists; widen `near_boundary_threshold` to push genes further from
the walls.

## Gene encoding and decoding

Encoding converts a gene's float value into a compact representation for storage, transmission, or evolutionary search.  Decoding is the inverse operation.

### Encoding specs

```python
from farm.core.hyperparameter_chromosome import GeneEncodingScale, GeneEncodingSpec

# Linear scale, no quantization (floating-point normalized to [0, 1])
spec_linear = GeneEncodingSpec()

# Log scale, 8-bit quantized integer in [0, 255]
spec_log_8bit = GeneEncodingSpec(scale=GeneEncodingScale.LOG, bit_width=8)
```

- **`scale=LINEAR`** — `(value - min) / (max - min)`
- **`scale=LOG`** — `(log10(value) - log10(min)) / (log10(max) - log10(min))`.  Bounds and value must be strictly positive.
- **`bit_width`** — when set, the normalized float is rounded to `round(normalized * (2**bit_width - 1))`, yielding an integer bucket.

### Gene-level encode/decode

```python
gene = HyperparameterGene("learning_rate", ..., min_value=1e-6, max_value=1.0, value=1e-3)

# Encode using the default policy for this gene name (log + 8-bit)
bucket = gene.encode()          # e.g., 102 (integer 0..255)
lr     = gene.decode(bucket)    # ~1e-3

# Override with an explicit spec
spec   = GeneEncodingSpec(scale=GeneEncodingScale.LINEAR)
norm   = gene.encode(encoding=spec)   # float in [0, 1]
lr     = gene.decode(norm, encoding=spec)
```

### Chromosome-level helpers

Four module-level functions work on full chromosomes:

| Function | Returns | Description |
|----------|---------|-------------|
| `encode_chromosome(chromosome, *, include_fixed, encoding_specs)` | `Dict[str, int \| float]` | Encode evolvable genes by name |
| `decode_chromosome(encoded_values, *, template, encoding_specs)` | `HyperparameterChromosome` | Decode named values back to a chromosome |
| `encode_chromosome_vector(chromosome, *, include_fixed, encoding_specs)` | `Tuple[int \| float, …]` | Encode as an ordered vector |
| `decode_chromosome_vector(encoded_values, *, template, include_fixed, encoding_specs)` | `HyperparameterChromosome` | Decode an ordered vector using template gene order |

`encoding_specs` is an optional `Mapping[str, GeneEncodingSpec]` that overrides per-gene defaults.  If omitted, `DEFAULT_GENE_ENCODINGS` is used.

```python
from farm.core.hyperparameter_chromosome import (
    encode_chromosome, decode_chromosome,
    encode_chromosome_vector, decode_chromosome_vector,
)

chrom = default_hyperparameter_chromosome()

# Round-trip via dict (three evolvable genes: learning_rate, gamma, epsilon_decay)
encoded = encode_chromosome(chrom)          # {"learning_rate": 102, "gamma": 252, "epsilon_decay": 253}
restored = decode_chromosome(encoded, template=chrom)

# Round-trip via vector (preserves gene order, length == number of evolvable genes)
vec = encode_chromosome_vector(chrom)       # (102, 252, 253)
restored_vec = decode_chromosome_vector(vec, template=chrom)
```

## How to add a new gene

### 1) Add gene to the default registry

In `farm/core/hyperparameter_chromosome.py`, append to `DEFAULT_HYPERPARAMETER_GENES`:

```python
HyperparameterGene(
    name="gamma",
    value_type=GeneValueType.REAL,
    value=0.95,
    min_value=0.0,
    max_value=1.0,
    default=0.95,
    evolvable=True,
)
```

Guidelines:
- keep defaults aligned with `DecisionConfig` defaults
- choose conservative bounds first, then broaden based on empirical results
- mark as fixed (`evolvable=False`) until ready for live evolution

### 2) Register an encoding policy (optional)

If the default linear/no-quantization encoding is not appropriate, add an entry to `DEFAULT_GENE_ENCODINGS`:

```python
DEFAULT_GENE_ENCODINGS["gamma"] = GeneEncodingSpec(scale=GeneEncodingScale.LINEAR, bit_width=8)
```

Use log-scale for parameters that span multiple orders of magnitude (e.g., learning rates, weight-decay coefficients).  Omit the entry to fall back to continuous linear encoding.

### 3) Ensure config compatibility

`apply_chromosome_to_learning_config(...)` applies values only for fields that exist on the target config object.

For `DecisionConfig` fields:
- no extra wiring is needed beyond adding the gene

For non-decision hyperparameters:
- either add a corresponding field on decision config
- or extend application logic for another config target

### 4) Add tests

At minimum add tests in `tests/test_hyperparameter_chromosome.py` for:
- bound validation
- serialization round-trip
- mutation behavior
- config projection
- encode/decode round-trip (including edge values and quantization boundaries)

If runtime flow changes, add or update integration tests similar to:
- `tests/test_agent_reproduction_hyperparameters.py`

## Crossover strategies

`crossover_chromosomes(parent_a, parent_b, *, mode, ...)` supports four operators selectable via `CrossoverMode`:

| Mode | String key | Description |
|------|-----------|-------------|
| `SINGLE_POINT` | `"single_point"` | One random pivot; genes before the pivot from parent A, the rest from parent B. |
| `UNIFORM` | `"uniform"` | Each gene independently drawn from parent B with probability `uniform_parent_b_probability` (default 0.5). |
| `BLEND` | `"blend"` | BLX-α: each gene value is sampled uniformly from `[lo − α·span, hi + α·span]` and clamped to gene bounds. Controls recombination range beyond the parents' interval. Set `blend_alpha=0.0` for a convex combination. |
| `MULTI_POINT` | `"multi_point"` | `num_crossover_points` random pivots divide the gene vector into alternating segments from each parent. Useful for longer gene vectors. |

`EvolutionExperimentConfig` exposes:
- `crossover_mode` — selects the operator (default `CrossoverMode.UNIFORM`)
- `blend_alpha` — BLX-α extent (default `0.5`; must be ≥ 0)
- `num_crossover_points` — pivot count for multi-point (default `2`; must be ≥ 1)

All modes are deterministic when an explicit `rng=random.Random(seed)` is passed.



The current mutation strategy is simple and intentionally local.

To adapt:

- change the default mutation operator (for example, gaussian ↔ multiplicative,
  or implement log-space mutation for selected genes)
- add per-gene mutation scales
- add schedule-based mutation rate by generation
- support crossover across two parent chromosomes
- switch `boundary_mode` to `"reflect"` to avoid boundary collapse
- enable `BoundaryPenaltyConfig` to add a soft fitness signal near walls

Recommended approach:
- keep `HyperparameterGene` and `HyperparameterChromosome` validation unchanged
- prefer per-gene mutation controls first; add new strategy functions only when needed
- keep tests deterministic by patching randomness in unit tests

## Relationship to `Genome`

Use both abstractions together:

- `Genome` for action weights + module state snapshots and crossover/mutation utilities in that space
- `HyperparameterChromosome` for typed, bounded hyperparameter evolution

They are parallel tracks today; a future unification step can compose both into a higher-level evolutionary payload if needed.

## Current limitations

- only real-valued genes are implemented
- runtime integration currently mutates during `AgentCore.reproduce()` only
- mutation rate is currently a constant in `AgentCore` (`DEFAULT_HYPERPARAMETER_MUTATION_RATE`)
- log-scale encoding requires strictly positive gene bounds; genes with `min_value ≤ 0` must use `GeneEncodingScale.LINEAR`

The schema now supports three continuously evolvable genes (`learning_rate`, `gamma`, `epsilon_decay`).  `memory_size` remains fixed pending integer-gene support.

## Discrete-gene roadmap

`memory_size` and other integer/discrete parameters require rounding during config projection (`int(round(gene.value))`).  The planned migration path is:

1. **Integer rounding guard** — validate that after rounding, the projected integer falls within the gene's declared bounds.  This can be added to `apply_chromosome_to_learning_config` without schema changes.
2. **Dedicated `GeneValueType.INTEGER` variant** — extend the `GeneValueType` enum and add a validation branch in `HyperparameterGene.__post_init__` that enforces integer-valued `min_value`, `max_value`, `default`, and `value`.  Encode/decode methods already round when `bit_width` is set, so no encoding changes are needed.
3. **Enable `memory_size`** — once the integer guard is in place, flip `evolvable=True` for `memory_size` and add coverage to the chromosome and evolution-experiment test suites.
4. **Binary/categorical genes** — implement `GeneValueType.BINARY` or `GeneValueType.CATEGORICAL` when needed; keep separate from the real/integer path to preserve existing validation.

Until step 1 is validated in integration tests, `memory_size` stays fixed to avoid undetected rounding drift.

## Evolution experiment outputs

`farm/runners/evolution_experiment.py` persists two machine-readable artifacts when `output_dir` is set:

- `evolution_generation_summaries.json`
  - per-generation fitness aggregates (`best_fitness`, `mean_fitness`, `min_fitness`)
  - per-gene statistics (`mean`, `median`, `std`, `min`, `max`)
  - best candidate chromosome values for that generation
- `evolution_lineage.json`
  - one row per evaluated candidate with lineage (`parent_ids`) and fitness metadata

Use `scripts/plot_hyperparameter_evolution.py` to produce a convergence chart from the summaries JSON.

## Crossover strategy comparison runs

To compare crossover operators directly, run:

```bash
python scripts/compare_evolution_crossover_strategies.py \
  --environment testing \
  --generations 3 \
  --population-size 6 \
  --steps-per-candidate 50 \
  --crossover-modes uniform,blend,multi_point,single_point \
  --seeds 42,43,44 \
  --output-json experiments/evolution/crossover_strategy_comparison.json
```

The report contains:

- `mode_summaries`
  - per-mode aggregate stats for `final_best_fitness`, `final_mean_fitness`, and `final_diversity`
  - summary fields include `mean`, `stdev`, `min`, and `max`
- `runs`
  - one row per `(mode, seed)` with raw final-generation fitness and diversity values
- `config`
  - full run configuration for reproducibility

Use this artifact to compare crossover strategy impact on convergence quality (fitness) and population spread (diversity) across repeated seeds.
