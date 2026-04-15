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
- `HyperparameterGene`
  - name, type, value, min/max bounds, default, and `evolvable` flag
  - validates:
    - non-empty name
    - valid min/max range
    - in-range numeric value and default
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

- `learning_rate` (evolvable)
- `epsilon_decay` (fixed placeholder)
- `memory_size` (fixed placeholder)

Helpers:

- `default_hyperparameter_chromosome()`
- `hyperparameter_evolution_registry()`
- `chromosome_from_values()`
- `chromosome_from_learning_config()`

## Runtime wiring in reproduction

`farm/core/agent/core.py` now uses the chromosome as part of offspring creation.

On agent init:
- `self.hyperparameter_chromosome` is created from `self.config.decision`.

On `AgentCore.reproduce()`:
1. Build a chromosome from the parent decision config.
2. Mutate evolvable genes via `mutate_chromosome(...)`.
3. Deep-copy parent config.
4. Apply chromosome values to child decision config via `apply_chromosome_to_learning_config(...)`.
5. Create offspring with the child config.
6. Store the resulting chromosome on the offspring.

This keeps:
- existing action/module-state genome behavior intact
- hyperparameter evolution explicit and typed

## Mutation behavior

`mutate_chromosome(chromosome, mutation_rate=0.1, mutation_scale=0.2)`:

- only mutates genes where `evolvable=True`
- applies multiplicative perturbation for real-valued genes:
  - `new_value = old_value * (1 + uniform(-scale, scale))`
- clamps to `[min_value, max_value]`

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

### 2) Ensure config compatibility

`apply_chromosome_to_learning_config(...)` applies values only for fields that exist on the target config object.

For `DecisionConfig` fields:
- no extra wiring is needed beyond adding the gene

For non-decision hyperparameters:
- either add a corresponding field on decision config
- or extend application logic for another config target

### 3) Add tests

At minimum add tests in `tests/test_hyperparameter_chromosome.py` for:
- bound validation
- serialization round-trip
- mutation behavior
- config projection

If runtime flow changes, add or update integration tests similar to:
- `tests/test_agent_reproduction_hyperparameters.py`

## How to adapt mutation strategy

The current mutation strategy is simple and intentionally local.

To adapt:

- replace multiplicative mutation with additive/log-space strategy
- add per-gene mutation scales
- add schedule-based mutation rate by generation
- support crossover across two parent chromosomes

Recommended approach:
- keep `HyperparameterGene` and `HyperparameterChromosome` validation unchanged
- implement strategy changes behind `mutate_chromosome(...)` or a new strategy function
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

These are deliberate for a small, verifiable first increment.
