# Genetic System — Landing Page

This document is the **single starting point** for understanding everything genetics-related in
AgentFarm: agent inheritance, hyperparameter chromosomes, lineage IDs, the outer-loop evolution
runner, neural Q-network crossover, persistence, metrics, analysis, and the embedding research
scaffold.  It is written to be read top-to-bottom as a design overview, and it cross-links to every
file and follow-up doc you would need to dig deeper.

> **Audience:** new contributors, researchers extending the evolutionary side of AgentFarm, and
> reviewers asking *“where does the genetic stuff live?”*.

---

## 1. What “genetics” means in AgentFarm

The codebase uses the word *genetic* in several distinct senses.  Treat them as **separate but
interlocking subsystems**:

| Subsystem | Representation | Where it lives | Purpose |
|---|---|---|---|
| **Lineage identifier** | `genome_id` string + `generation` int + `parent_ids` list | `farm/utils/identity.py`, `farm/database/data_types.py`, `farm/core/state.py` | Track who descended from whom across one simulation. |
| **Hyperparameter chromosome** | Typed bounded `HyperparameterGene` tuple | `farm/core/hyperparameter_chromosome.py` | The *real* inheritable, mutable substrate evolved during reproduction. |
| **`Genome` dict (legacy/utility)** | Python `dict` of action weights + module state | `farm/core/genome.py` | Full agent snapshot + selection helpers reused by the outer evolution runner. |
| **Outer-loop evolution experiment** | Population of chromosomes evaluated by full simulations | `farm/runners/evolution_experiment.py` | Multi-generation hyperparameter search across simulations. |
| **Neural-network crossover** | Q-network weights | `farm/core/decision/training/crossover.py` and friends | Recombine and fine-tune trained parent Q-networks into a child network. |
| **Genome embeddings (research)** | Neural embedding from `(generation, parent hash, trait hash)` | `farm/genome_embeddings/` | Research scaffold for embedding lineage into vector space. |
| **Genetics analysis module** | Pandas DataFrames built from DB or evolution artifacts | `farm/analysis/genetics/` | Lineage / chromosome statistics and plots. |

The **primary** in-simulation inheritance path is **chromosome + `genome_id`**.  Everything else is
either an analysis layer over those, or a research / outer-loop tool on top of them.

---

## 2. Design intent

### 2.1 Goals

1. **Track ancestry losslessly** without a relational `parent_id` foreign key.  Encode parents
   directly in the agent’s `genome_id` so that even raw SQL dumps remain self-describing.
2. **Evolve a small, typed, bounded set of hyperparameters** during reproduction with safe
   defaults, instead of evolving the whole agent state.  This keeps the search space interpretable
   and preserves reproducibility.
3. **Keep the simulation loop and the outer evolution loop orthogonal.**  A single simulation can
   run with or without evolutionary pressure.  The outer `EvolutionExperiment` runner treats
   simulations as black-box fitness functions.
4. **Make every evolved locus observable.**  All mutation operators, boundary modes, and
   per-generation stats are designed to land in JSON artifacts and SQL columns so analyses can be
   replayed and compared.

### 2.2 Non-goals

- AgentFarm does **not** evolve full neural-network architectures during simulation reproduction.
  Network *weight* recombination happens only in the offline crossover-and-finetune pipeline
  (Section 6).
- There is **no global gene pool** outside of an in-flight simulation.  Cross-simulation transfer
  is the job of `EvolutionExperiment`.
- Sexual reproduction with two parents is *representable* in `genome_id` but the in-world
  `reproduce` action is currently asexual (one parent).  Two-parent mating is reserved for
  outer-loop crossover.

### 2.3 Architectural choices worth knowing

- **Dual representation.**  `Genome` (`farm/core/genome.py`) is a fat snapshot dict; the typed
  `HyperparameterChromosome` (`farm/core/hyperparameter_chromosome.py`) is the evolvable substrate.
  They are **parallel tracks** by design — see “Relationship to `Genome`” in
  [`hyperparameter_chromosome.md`](hyperparameter_chromosome.md).
- **Lineage as string encoding.**  `genome_id` carries parent IDs and a per-base ordinal counter,
  parsed via `GenomeId.from_string` (`farm/database/data_types.py`).  No separate parent FK.
- **Frozen state objects.**  `AgentState` is a frozen Pydantic model; reproduction updates use
  `model_copy(update=...)`.  Genes are validated dataclasses with `with_value` style mutation.
- **Selection lives with `Genome`.**  Tournament and roulette helpers are static methods on
  `Genome` so the outer experiment runner can reuse them without a separate selection module.
- **Bounded, typed genes.**  Every gene declares `min_value`, `max_value`, `default`, `evolvable`,
  and per-gene mutation knobs.  Validation runs at construction.
- **Boundary-aware mutation.**  Three boundary modes (`CLAMP`, `REFLECT`, `INTERIOR_BIASED`) plus
  optional soft penalties exist specifically to combat boundary collapse on `learning_rate`.

---

## 3. Lineage: `genome_id`, `generation`, `parent_ids`

### 3.1 Format

`Identity.genome_id(parent_ids, ...)` produces a string in this canonical shape:

| Case | `parent_ids` | `genome_id` example |
|---|---|---|
| Initial agent (no parents) | `[]` | `::1`, `::2`, … |
| Cloning (one parent) | `["agent_a"]` | `agent_a:1`, `agent_a:2`, … |
| Sexual (two parents) | `["agent_a", "agent_b"]` | `agent_a:agent_b:1`, … |

The trailing integer is a per-base ordinal counter.  It guarantees uniqueness across siblings
sharing the same parent set.

Defined in `farm/utils/identity.py`:

- `Identity.genome_id(parent_ids, existing_genome_checker=None)` — allocates the next counter,
  consulting both an in-memory registry and (optionally) a DB existence callback to avoid
  collisions across runs that share a database.
- `_genome_id_registry: dict[str, int]` — per-`Identity` counter map keyed by base string.

Parsed back via `GenomeId.from_string(...)` in `farm/database/data_types.py` and surfaced to the
analysis layer through `parse_parent_ids(...)` in `farm/analysis/genetics/utils.py`.

### 3.2 Where it gets assigned

`Environment.add_agent` (`farm/core/environment.py`) is the single assignment point:

1. If the incoming agent already has a `genome_id`, leave it alone.
2. Otherwise read `agent.state.parent_ids`, call `identity.genome_id(...)`, and write the result
   back onto `state` and into the DB row.
3. A per-environment `_genome_id_cache` short-circuits redundant DB existence checks.

### 3.3 Generation counter

`generation` is an explicit `int` on both `AgentState` and `AgentModel`:

- Initial agents: `generation = 0`.
- Offspring: `generation = parent.generation + 1` (set in `AgentCore.reproduce()`).
- `farm/database/validation.py::validate_generation_monotonicity` checks that offspring generations
  are exactly parent + 1 (looking up the first parent via `GenomeId.from_string`).

---

## 4. Hyperparameter chromosome (the evolved substrate)

This is the core inheritance unit and is documented in detail at
[`docs/design/hyperparameter_chromosome.md`](hyperparameter_chromosome.md).  This section is a
high-level summary; the linked doc is the reference.

### 4.1 Schema

`farm/core/hyperparameter_chromosome.py` defines:

- `GeneValueType` — currently only `REAL` is implemented; integer / categorical roadmap is in
  [`evolvable_loci_roadmap.md`](evolvable_loci_roadmap.md).
- `GeneEncodingScale` (`LINEAR`, `LOG`) and `GeneEncodingSpec(scale, bit_width)` — bit-quantized
  encoding for compact storage and search.
- `HyperparameterGene` — `name`, `value`, `min_value`, `max_value`, `default`, `evolvable`,
  `mutation_scale`, `mutation_probability`, `mutation_strategy`.
- `HyperparameterChromosome` — ordered tuple of `HyperparameterGene` with unique-name enforcement,
  partitioning into evolvable/fixed, validated overrides, and `to_dict` / `from_dict`.
- Operators: `mutate_chromosome(...)`, `crossover_chromosomes(...)`,
  `apply_chromosome_to_learning_config(...)`, `chromosome_from_learning_config(...)`.
- Encoding helpers: `encode_chromosome`, `decode_chromosome`,
  `encode_chromosome_vector`, `decode_chromosome_vector`.

### 4.2 Default gene registry

`DEFAULT_HYPERPARAMETER_GENES` ships with three evolvable real genes plus one fixed placeholder:

| Gene | Range | Default | Encoding | Status |
|---|---|---|---|---|
| `learning_rate` | `[1e-6, 1.0]` | from `DecisionConfig` | log, 8-bit | evolvable |
| `gamma` | `[0.0, 1.0]` | `0.99` | linear, 8-bit | evolvable |
| `epsilon_decay` | `(0, 1.0]` | `0.995` | linear, 8-bit | evolvable |
| `memory_size` | integer | — | — | fixed (pending integer-gene support) |

### 4.3 Mutation operators

`mutate_chromosome(...)`:

- Mutates only `evolvable=True` genes.
- Resolves `mutation_probability`, `mutation_scale`, `mutation_strategy` per gene; global args
  override per-gene values.
- Two real-valued operators: `GAUSSIAN` (`new = old + Normal(0, scale·span)`) and
  `MULTIPLICATIVE` (`new = old · (1 + uniform(-scale, scale))`).
- Boundary modes: `CLAMP` (default), `REFLECT`, `INTERIOR_BIASED` (with `interior_bias_fraction`).
- Optional `BoundaryPenaltyConfig` adds a non-negative penalty subtracted from raw fitness.

### 4.4 Crossover operators

`crossover_chromosomes(parent_a, parent_b, *, mode, ...)`:

| Mode | Description |
|---|---|
| `SINGLE_POINT` | One pivot; left from A, right from B. |
| `UNIFORM` (default) | Each gene independently from B with `uniform_parent_b_probability`. |
| `BLEND` | BLX-α; sample uniformly from `[lo − α·span, hi + α·span]`, clamp to bounds. |
| `MULTI_POINT` | `num_crossover_points` pivots; alternating segments from A and B. |

All operators are deterministic when an explicit `rng=random.Random(seed)` is passed.

---

## 5. The in-simulation reproduction lifecycle

```
                ┌────────────────────────────────────────────────────────┐
                │                  per-step environment loop             │
                └────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                         farm/core/action.py::reproduce_action
                  ┌──────────────────────────────────────────────────┐
                  │ 1. resource gate (min_reproduction_resources +   │
                  │    offspring_cost)                               │
                  │ 2. roll reproduction_chance                      │
                  │ 3. call agent.reproduce()                        │
                  └──────────────────────────────────────────────────┘
                                          │
                                          ▼
                         farm/core/agent/core.py::AgentCore.reproduce
                  ┌──────────────────────────────────────────────────┐
                  │ a. deduct offspring_cost (with rollback/refund)  │
                  │ b. parent_chromosome = self.hyperparameter_      │
                  │    chromosome (or rebuild from decision config)  │
                  │ c. child_chromosome = mutate_chromosome(         │
                  │       deepcopy(parent_chromosome),               │
                  │       mutation_rate=DEFAULT_HYPERPARAMETER_      │
                  │       MUTATION_RATE)        # 0.1                │
                  │ d. child_config = deepcopy(self.config)          │
                  │ e. child_config.decision = apply_chromosome_to_  │
                  │       learning_config(child_config.decision,     │
                  │                       child_chromosome)          │
                  │ f. AgentFactory.create_learning_agent(...)       │
                  │ g. offspring.generation = self.generation + 1    │
                  │ h. offspring.state.parent_ids = [self.agent_id]  │
                  │ i. environment.add_agent(offspring,              │
                  │                          flush_immediately=True) │
                  └──────────────────────────────────────────────────┘
                                          │
                                          ▼
                         farm/core/environment.py::Environment.add_agent
                  ┌──────────────────────────────────────────────────┐
                  │ assigns genome_id from state.parent_ids via      │
                  │ Identity.genome_id(...) and persists the row     │
                  └──────────────────────────────────────────────────┘
```

Key constants and config knobs:

- `DEFAULT_HYPERPARAMETER_MUTATION_RATE = 0.1` (`farm/core/agent/core.py`).
- `min_reproduction_resources` (default `8`), `offspring_cost` (default `5` in
  `reproduce_action`, `3` in `AgentBehaviorConfig`), `reproduction_chance` (default `0.5`),
  `offspring_initial_resources` (component default `10.0`, sim config default `5`),
  `reproduction_success_bonus` (default `0.15`).

> **Naming caveat.**  `AgentBehaviorConfig` and the `reproduce_action` defaults differ.  When in
> doubt, trace `getattr(agent.config, "<name>", <default>)` in `farm/core/action.py`.

---

## 6. Outer-loop evolution experiment

`farm/runners/evolution_experiment.py` (~700 lines) treats a simulation as a fitness function and
evolves a population of `HyperparameterChromosome`s over multiple generations.

### 6.1 Components

- `EvolutionExperimentConfig` — generations, population size, steps per candidate, mutation
  knobs (rate, scale, mode, boundary mode, interior bias), crossover knobs (mode, blend α,
  num points), selection method, tournament size, elitism count, fitness metric, adaptive
  mutation, and convergence criteria.
- `EvolutionFitnessMetric` — `FINAL_POPULATION` (default), `TOTAL_BIRTHS`, `FINAL_RESOURCES`.
- `EvolutionSelectionMethod` — `TOURNAMENT` or `ROULETTE`; both delegate to `Genome.*_selection`.
- `ConvergenceCriteria` — fitness-plateau and diversity-collapse checks with `early_stop`.
- `AdaptiveMutationController` (`farm/runners/adaptive_mutation.py`) — adjusts mutation rate /
  scale based on diversity collapse signals.

### 6.2 Per-generation loop

1. **Evaluate** each candidate by running `farm.core.simulation.run_simulation` with the
   chromosome projected onto `SimulationConfig.decision`.
2. **Score** each candidate using the chosen fitness metric, optionally subtracting
   `compute_boundary_penalty(...)`.
3. **Select** parents via tournament or roulette (delegated to `Genome.tournament_selection` /
   `Genome.roulette_selection`).
4. **Crossover** parent chromosomes per `crossover_mode`.
5. **Mutate** offspring via `mutate_chromosome` with `boundary_mode` and adaptive rates applied.
6. **Elitism**: copy top `elitism_count` chromosomes unchanged into the next generation.
7. **Check convergence** (fitness plateau, diversity collapse).  Stop early if `early_stop=True`.

### 6.3 Persisted artifacts

When `output_dir` is set, three JSON files are written:

- `evolution_generation_summaries.json` — per-generation `best_fitness`, `mean_fitness`,
  `min_fitness`, `gene_statistics` (mean/median/std/min/max + `at_min_count`, `at_max_count`,
  `boundary_fraction`), `boundary_occupancy`, `best_chromosome`.
- `evolution_lineage.json` — one row per evaluated candidate with `candidate_id`, `generation`,
  `fitness`, gene values, `parent_ids`, and metadata.
- `evolution_metadata.json` — full config snapshot for reproducibility.

### 6.4 CLIs and presets

- `scripts/run_evolution_experiment.py` — primary entry point.
- `scripts/evolution_experiment_cli.py` — argparse + presets such as `stable_hyper_evo`.
- `scripts/compare_evolution_crossover_strategies.py` — sweep across crossover operators.
- `scripts/plot_hyperparameter_evolution.py` — quick convergence chart from the summaries JSON.
- `scripts/run_multi_gen_search.py` — multi-generation search emitting its own `lineage.json`.

---

## 7. Neural Q-network crossover (separate pipeline)

This pipeline lives under `farm/core/decision/training/` and is **not** triggered by the in-world
`reproduce` action.  It recombines two trained parent Q-networks into a child network and
optionally fine-tunes it.

| File | Role |
|---|---|
| `crossover.py` | `initialize_child_from_crossover(...)` and weight-level recombination strategies. |
| `crossover_search.py` | Sweeps over crossover + fine-tune regimes. |
| `finetune.py` | Child fine-tuning; reads YAML section `crossover_child_finetune`. |
| `recombination_eval.py` | Evaluation harness comparing child vs parents. |
| `recombination_analysis.py`, `recombination_stats.py` | Metrics aggregation. |

Configuration block in `farm/config/default.yaml`:

```yaml
crossover_child_finetune:
  learning_rate: ...
  epochs: ...
  batch_size: ...
  max_grad_norm: ...
  val_fraction: ...
  seed: ...
  loss_fn: ...
  temperature: ...
  alpha: ...
  optimizer: ...
  early_stopping_patience: ...
```

Deeper reading: [`distill_quantize_crossover_finetune.md`](distill_quantize_crossover_finetune.md),
[`crossover_strategies.md`](crossover_strategies.md), and
[`crossover_search_space.md`](crossover_search_space.md).

---

## 8. Persistence (database & file artifacts)

### 8.1 SQLite via SQLAlchemy (`farm/database/models.py`)

`agents` table — only the columns relevant to genetics:

| Column | Type | Purpose |
|---|---|---|
| `genome_id` | `String(64)` | Encodes parents and per-base ordinal. |
| `generation` | `Integer` | Generation counter. |
| `action_weights` | `JSON` | Snapshot of action weights at registration (not the full `Genome` dict). |

`simulation_steps` table — population-level metrics:

| Column | Description |
|---|---|
| `current_max_generation` | Highest live generation that step. |
| `genetic_diversity` | `len(distinct base genome_ids) / total_agents` over **living** agents. |
| `dominant_genome_ratio` | `max(base genome_id count) / total_agents`. |

`AgentState` rows do **not** store `genome_id` or `generation`; those live on the `agents` row.
There is **no** separate `parent_id` foreign key — parents are recovered by parsing `genome_id`.

### 8.2 File artifacts

- `evolution_generation_summaries.json`, `evolution_lineage.json`, `evolution_metadata.json`
  (Section 6.3).
- `lineage.json` from `scripts/run_multi_gen_search.py`.
- Crossover sweeps (Section 7) produce their own JSON reports.

---

## 9. Metrics

`farm/core/metrics_tracker.py` computes two genetics metrics every step:

```python
def get_base_genome_id(genome_id: str) -> str:
    # Strip the trailing :counter to group siblings with shared lineage.
    ...

genetic_diversity   = len(genome_counts) / total_agents
dominant_genome_ratio = max(genome_counts.values()) / total_agents
```

These are not allele-level metrics; they are **lineage-grouping** metrics computed from
`genome_id` strings of currently living agents.

The evolution experiment additionally tracks **gene-level diversity** via
`compute_normalized_diversity(...)` in `farm/runners/adaptive_mutation.py`.

---

## 10. Analysis layer

`farm/analysis/genetics/` is a registered analysis module (`genetics`) that supports two data
sources:

1. **Simulation database** — `build_agent_genetics_dataframe(...)` loads per-agent action weights,
   generation, and lineage.
2. **Evolution-experiment artifacts** — `build_evolution_experiment_dataframe(...)` loads
   per-candidate chromosome values, fitness, and parent IDs from the JSON files in Section 6.3.

| File | Role |
|---|---|
| `__init__.py` | Public API surface (`parse_parent_ids`, builders, `analyze_genetics`, module). |
| `module.py` | `GeneticsModule(BaseAnalysisModule)` registration with the analysis service. |
| `compute.py` | DataFrame builders for both data sources. |
| `data.py` | `process_genetics_data` data processor. |
| `analyze.py` | `analyze_genetics(...)` orchestrator. |
| `plot.py` | `plot_generation_distribution`, `plot_fitness_over_generations`. |
| `utils.py` | `parse_parent_ids` (delegates to `GenomeId.from_string`). |

Other analysis modules consume genetics data through `parse_parent_ids` /
`GenomeId.from_string`:

- `farm/analysis/significant_events/compute.py` — birth events.
- `farm/analysis/social_behavior/compute.py` — `compute_reproduction_social_patterns`.
- `farm/analysis/genesis/compute.py`, `dominance/data.py`, `advantage/compute.py` — parent
  parsing from offspring `genome_id`.

Charts: `farm/charts/chart_simulation.py` plots `genetic_diversity` and
`dominant_genome_ratio`; `farm/charts/chart_agents.py` and `chart_analyzer.py` use the
`base_genome_id` helper for lineage-size plots.

---

## 11. Genome embeddings (research scaffold)

`farm/genome_embeddings/` is a **research scaffold**, not a production pipeline.  It contains a
neural encoder (`encoder.py::GenomeEncoder`) that maps `(generation, parent hash bits, trait
hash bits)` to a fixed-dimension embedding, plus matching `dataset.py`, `training.py`,
`utils.py`, and `visualization.py` helpers.  Treat it as an experiment seed; nothing in the core
loop depends on it.

---

## 12. Configuration knobs

### 12.1 In-simulation reproduction

In `farm/config/config.py`:

| Field | Default | Notes |
|---|---|---|
| `AgentBehaviorConfig.offspring_cost` | `3` | Resource cost paid by parent. |
| `AgentBehaviorConfig.min_reproduction_resources` | `8` | Minimum resources required to attempt. |
| `AgentBehaviorConfig.offspring_initial_resources` | `5` | Resources granted to offspring. |
| `ActionRewardConfig.reproduction_success_bonus` | `0.15` | Reward shaping. |

In `farm/core/agent/config/component_configs.py`:

- `ReproductionConfig.offspring_initial_resources` (component default `10.0`); merged from
  simulation config in `AgentComponentConfig.from_simulation_config`.

In `farm/core/action.py::reproduce_action`:

- `getattr(agent.config, "reproduction_chance", 0.5)`.  Note this attribute is read off whichever
  config object the agent has — verify the type if you override it.

In `farm/core/agent/core.py`:

- `DEFAULT_HYPERPARAMETER_MUTATION_RATE = 0.1`.

### 12.2 Outer-loop evolution

`EvolutionExperimentConfig` defaults (in `farm/runners/evolution_experiment.py`):

| Field | Default |
|---|---|
| `num_generations` | `3` |
| `population_size` | `6` |
| `num_steps_per_candidate` | `50` |
| `mutation_rate` | `0.25` |
| `mutation_scale` | `0.2` |
| `mutation_mode` | `GAUSSIAN` |
| `boundary_mode` | `CLAMP` |
| `interior_bias_fraction` | `1e-3` |
| `crossover_mode` | `UNIFORM` |
| `blend_alpha` | `0.5` |
| `num_crossover_points` | `2` |
| `selection_method` | `TOURNAMENT` |
| `tournament_size` | `3` |
| `elitism_count` | `1` |
| `fitness_metric` | `FINAL_POPULATION` |

CLI defaults (e.g. `--generations` in `scripts/evolution_experiment_cli.py`) may differ — those
are user-facing presets.

### 12.3 Neural crossover fine-tune

`crossover_child_finetune` block in `farm/config/default.yaml` (Section 7).

### 12.4 Resource “evolution” (not agent genetics)

`ResourcesConfig.mutation_rate` (default `0.01`) and `mutation_interval` (default `50`) drive the
`EvolutionaryRegenerator` resource subsystem.  These have nothing to do with agent inheritance,
despite the names.

---

## 13. Tests

| Path | Coverage |
|---|---|
| `tests/test_genome.py` | `Genome` dict utilities, selection. |
| `tests/test_genome_id_generation.py` | `Identity.genome_id` end-to-end with `Environment`. |
| `tests/test_identity.py` | Identity helpers. |
| `tests/test_hyperparameter_chromosome.py` | Chromosome mutation, crossover, encoding bounds. |
| `tests/test_agent_reproduction_hyperparameters.py` | Reproduction wires chromosome correctly. |
| `tests/agent/components/test_reproduction.py` | Reproduction component, generation counter. |
| `tests/analysis/test_genetics.py` | `parse_parent_ids` + DataFrame builders. |
| `tests/runners/test_evolution_experiment.py`, `tests/runners/test_evolution_regression.py` | Outer-loop runner + JSON schema. |
| `tests/test_run_evolution_experiment_cli.py` | CLI surface. |
| `tests/test_compare_evolution_crossover_strategies.py` | Crossover strategy comparisons. |
| `tests/database/test_validation.py` | Generation monotonicity. |

---

## 14. Reading order

If you are **new** and want to understand the genetic system, read these in order:

1. **This page** — for the map.
2. [`hyperparameter_chromosome.md`](hyperparameter_chromosome.md) — the canonical design doc for
   the evolved substrate (gene schema, mutation, crossover, boundary handling, encoding).
3. [`evolvable_loci_roadmap.md`](evolvable_loci_roadmap.md) — where the chromosome model is
   headed (integer / categorical / hierarchical / meta-evolution phases).
4. [`../experiments/hyperparameter_evolution_convergence.md`](../experiments/hyperparameter_evolution_convergence.md)
   — empirical convergence study + lineage JSON walkthrough.
5. [`../experiments/multi_seed_cohort.md`](../experiments/multi_seed_cohort.md) — cohort-level
   evolution flags and chromosome-occupancy metrics.
6. [`../devlog/2026-04-17-dna-hyperparameter-evolution.md`](../devlog/2026-04-17-dna-hyperparameter-evolution.md)
   — devlog framing the “DNA-style” hyperparameter evolution work.
7. [`distill_quantize_crossover_finetune.md`](distill_quantize_crossover_finetune.md),
   [`crossover_strategies.md`](crossover_strategies.md),
   [`crossover_search_space.md`](crossover_search_space.md) — neural Q-network crossover (separate
   pipeline).
8. [`../features/ai_machine_learning.md`](../features/ai_machine_learning.md) — feature-level
   description of the genome system and embeddings (note: some snippets are illustrative; trust
   the modules over the prose).

---

## 15. File index

### Core

- `farm/core/genome.py` — `Genome` dict + selection helpers.
- `farm/core/hyperparameter_chromosome.py` — typed chromosome, mutation, crossover, encoding,
  boundary penalties.
- `farm/core/agent/core.py` — `AgentCore.reproduce()` lifecycle.
- `farm/core/agent/components/reproduction.py` — resource-cost component template.
- `farm/core/agent/config/component_configs.py` — `ReproductionConfig` defaults and wiring.
- `farm/core/action.py` — `reproduce_action`.
- `farm/core/environment.py` — `Environment.add_agent` (genome_id assignment).
- `farm/core/state.py` — `AgentState` (`generation`, `parent_ids`, `genome_id`).
- `farm/core/metrics_tracker.py` — `genetic_diversity`, `dominant_genome_ratio`.
- `farm/utils/identity.py` — `Identity.genome_id` and registry.

### Outer-loop evolution

- `farm/runners/evolution_experiment.py`
- `farm/runners/adaptive_mutation.py`

### Neural crossover

- `farm/core/decision/training/crossover.py`
- `farm/core/decision/training/crossover_search.py`
- `farm/core/decision/training/finetune.py`
- `farm/core/decision/training/recombination_eval.py`
- `farm/core/decision/training/recombination_analysis.py`
- `farm/core/decision/training/recombination_stats.py`

### Persistence

- `farm/database/models.py` — `AgentModel`, `SimulationStepModel` columns.
- `farm/database/data_types.py` — `GenomeId` parser.
- `farm/database/data_logging.py` — agent batch logging path.
- `farm/database/validation.py` — `validate_generation_monotonicity`.
- `farm/database/repositories/agent_repository.py` — `get_children`, `genome_id LIKE` lookups.
- `farm/database/repositories/population_repository.py` — `unique_genomes`.

### Analysis & charts

- `farm/analysis/genetics/{__init__,module,compute,data,analyze,plot,utils}.py`
- `farm/analysis/significant_events/compute.py`
- `farm/analysis/social_behavior/compute.py`
- `farm/analysis/genesis/compute.py`
- `farm/analysis/dominance/data.py`
- `farm/analysis/advantage/compute.py`
- `farm/charts/chart_simulation.py`, `chart_agents.py`, `chart_analyzer.py`

### Research scaffold

- `farm/genome_embeddings/{encoder,dataset,training,utils,visualization}.py`

### Scripts

- `scripts/run_evolution_experiment.py`
- `scripts/evolution_experiment_cli.py`
- `scripts/compare_evolution_crossover_strategies.py`
- `scripts/plot_hyperparameter_evolution.py`
- `scripts/run_multi_gen_search.py`
- `scripts/run_cohort_experiment.py`
- `scripts/run_crossover_search.py`, `benchmark_crossover.py`,
  `aggregate_crossover_stats.py`, `finetune_child.py`

### Existing docs

- `docs/design/hyperparameter_chromosome.md`
- `docs/design/evolvable_loci_roadmap.md`
- `docs/design/distill_quantize_crossover_finetune.md`
- `docs/design/crossover_strategies.md`
- `docs/design/crossover_search_space.md`
- `docs/experiments/hyperparameter_evolution_convergence.md`
- `docs/experiments/multi_seed_cohort.md`
- `docs/devlog/2026-04-17-dna-hyperparameter-evolution.md`
- `docs/features/ai_machine_learning.md`

---

## 16. Caveats and known inconsistencies

- **`Genome` dict vs chromosome.**  The in-world reproduction path mutates the
  `HyperparameterChromosome`, **not** the `Genome` dict.  `Genome.mutate` and `Genome.crossover`
  are utilities reused by tests and tooling but are not part of the per-step lifecycle.
- **Two-parent reproduction.**  `Identity.genome_id` and `GenomeId` support two parents.  The
  in-world `AgentCore.reproduce()` currently sets only one parent in `state.parent_ids`.  Sexual
  reproduction would require a new code path that populates two IDs before `add_agent`.
- **`reproduction_chance` lookup.**  `reproduce_action` uses `getattr(agent.config,
  "reproduction_chance", 0.5)`.  `AgentBehaviorConfig` does not define this attribute today; the
  effective default is `0.5` unless your config object exposes it.
- **Outdated `agent_based_modeling_analysis.md` snippet.**  That doc shows `offspring.genome =
  mutate(agent.genome)`, which does not match the current implementation.  Treat it as
  illustrative.
- **Resource “mutation”.**  `ResourcesConfig.mutation_rate` / `mutation_interval` belong to the
  resource regenerator, not agent genetics.
- **Memory size.**  `memory_size` is `evolvable=False` until integer-gene support lands
  ([`evolvable_loci_roadmap.md`](evolvable_loci_roadmap.md), Phase 2).
