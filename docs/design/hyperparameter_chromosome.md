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
- `epsilon_decay` (fixed placeholder)
- `memory_size` (fixed placeholder)

Default encoding policies per gene name are stored in `DEFAULT_GENE_ENCODINGS`:

```python
DEFAULT_GENE_ENCODINGS = {
    "learning_rate": GeneEncodingSpec(scale=GeneEncodingScale.LOG, bit_width=8),
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

`mutate_chromosome(chromosome, mutation_rate=None, mutation_scale=None, mutation_mode=None)`:

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
- clamps to `[min_value, max_value]`

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

# Round-trip via dict
encoded = encode_chromosome(chrom)          # {"learning_rate": 102}
restored = decode_chromosome(encoded, template=chrom)

# Round-trip via vector (preserves gene order)
vec = encode_chromosome_vector(chrom)       # (102,)
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

## How to adapt mutation strategy

The current mutation strategy is simple and intentionally local.

To adapt:

- change the default mutation operator (for example, gaussian ↔ multiplicative,
  or implement log-space mutation for selected genes)
- add per-gene mutation scales
- add schedule-based mutation rate by generation
- support crossover across two parent chromosomes

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

These are deliberate for a small, verifiable first increment.

## Evolution experiment outputs

`farm/runners/evolution_experiment.py` persists two machine-readable artifacts when `output_dir` is set:

- `evolution_generation_summaries.json`
  - per-generation fitness aggregates (`best_fitness`, `mean_fitness`, `min_fitness`)
  - per-gene statistics (`mean`, `median`, `std`, `min`, `max`)
  - best candidate chromosome values for that generation
- `evolution_lineage.json`
  - one row per evaluated candidate with lineage (`parent_ids`) and fitness metadata

Use `scripts/plot_hyperparameter_evolution.py` to produce a convergence chart from the summaries JSON.
