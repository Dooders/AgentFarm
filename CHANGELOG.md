# Changelog

All notable changes to this project are documented in this file.

This repository uses an automated Copilot-assisted workflow to draft changelog updates after merges into `main`.

## Format

Each entry should include:

- what changed
- why it matters
- user/developer impact

Suggested grouping:

- Added
- Changed
- Fixed
- Docs
- Performance

## Entries

### 2026-04-24

#### Added

- **Allele-frequency and selection-pressure analytics** in `farm.analysis.genetics` — added allele-frequency timeseries and selection-pressure summaries, plus diversity-timeseries fallback from generation summaries when per-candidate evaluations are unavailable. ([#776](https://github.com/Dooders/AgentFarm/pull/776))
- **Phylogenetics analysis module** (`farm.analysis.phylogenetics`) — builds lineage trees/DAGs from simulation or evolution lineage data, computes summary statistics, exports JSON/Newick, registers with the analysis framework, and adds matplotlib visualization support. ([#780](https://github.com/Dooders/AgentFarm/pull/780))
- **Fitness landscape analysis** in the genetics module — added gene-fitness correlations, pairwise epistasis analysis, marginal-effect plots, and 2D landscape visualizations under a new `fitness_landscape` function group. ([#782](https://github.com/Dooders/AgentFarm/pull/782))
- **Population genetics tooling** — added a seeded Wright-Fisher neutral drift simulator plus F_ST, migration-count, and gene-flow timeseries analysis under a new `population_genetics` function group. ([#784](https://github.com/Dooders/AgentFarm/pull/784))

#### Docs

- Published a new devlog post on evolving hyperparameter genomes for foraging learning agents and refreshed docs/devlog index links. ([#776](https://github.com/Dooders/AgentFarm/pull/776))
- Redesigned the GitHub Pages documentation site with custom Jekyll layouts, shared navigation, branded CSS/assets, a docs landing page, devlog cards, local `github-pages` Gemfile, and a custom 404 page. ([#779](https://github.com/Dooders/AgentFarm/pull/779))

---

### 2026-04-23

#### Added

- **Genotypic diversity metrics** for `farm.analysis.genetics` — added continuous/categorical locus diversity, population diversity summaries, and evolution diversity timeseries APIs with expanded edge-case coverage. ([#768](https://github.com/Dooders/AgentFarm/pull/768))
- **Intrinsic evolution experiments** — added `IntrinsicEvolutionExperiment`, `IntrinsicEvolutionPolicy`, simulation hooks, and `GeneTrajectoryLogger` so per-agent hyperparameter chromosomes can evolve in-situ during simulations. ([#774](https://github.com/Dooders/AgentFarm/pull/774))

#### Changed

- `AgentCore.reproduce()` now preserves chromosomes when no intrinsic evolution policy is attached, keeping outer-loop experiments deterministic unless in-situ evolution is explicitly enabled. ([#774](https://github.com/Dooders/AgentFarm/pull/774))
- Shared gene-statistics computation between outer-loop and intrinsic evolution runners so trajectory/snapshot artifacts report comparable chromosome statistics. ([#774](https://github.com/Dooders/AgentFarm/pull/774))

---

### 2026-04-21

#### Added

- **Interior-biased mutation** (`BoundaryMode.INTERIOR_BIASED`) — clamps overshooting genes then nudges exact-boundary values inward by `interior_bias_fraction`, helping avoid boundary collapse while preserving tunable mutation behavior. ([#753](https://github.com/Dooders/AgentFarm/pull/753))
- **Boundary occupancy telemetry** — per-generation summaries now include per-gene `at_min_count`, `at_max_count`, `boundary_fraction`, and a top-level `boundary_occupancy` map for diagnosing boundary collapse. ([#753](https://github.com/Dooders/AgentFarm/pull/753))
- **Multi-seed cohort runner** — added `CohortRunner`, `run_cohort_experiment.py`, cohort manifests, JSON/CSV aggregate artifacts, cross-seed convergence/fitness summaries, and documentation for statistically robust evolution experiments. ([#754](https://github.com/Dooders/AgentFarm/pull/754))
- **Evolution regression CI** — added an `evolution_regression` pytest marker, threshold-based optimizer regression tests, and a scheduled/path-filtered GitHub Actions workflow. ([#755](https://github.com/Dooders/AgentFarm/pull/755))
- **Genetics analysis module** — added a registered built-in module for genome/chromosome/lineage analysis, plotting support, and high-level genetics summaries. ([#765](https://github.com/Dooders/AgentFarm/pull/765))

#### Changed

- Centralized parent-ID parsing through `parse_parent_ids` and updated reproduction-related analysis modules to use the shared helper for consistent genome lineage handling. ([#765](https://github.com/Dooders/AgentFarm/pull/765))

---

### 2026-04-20

#### Added

- **Evolution convergence criteria** — added `ConvergenceCriteria`/`ConvergenceReason`, optional early stopping for fitness plateau or diversity collapse, and persisted `evolution_metadata.json` convergence status. ([#747](https://github.com/Dooders/AgentFarm/pull/747))
- **`stable_hyper_evo` preset** and CLI convergence flags — added two-pass preset parsing, a pre-run `run_manifest.json`, and configurable convergence options for reproducible evolution experiments. ([#747](https://github.com/Dooders/AgentFarm/pull/747))
- **Adaptive mutation defaults and damping** — added built-in per-gene rate/scale multiplier defaults, `use_default_per_gene_multipliers`, and `max_step_multiplier` to limit per-generation mutation multiplier swings. ([#749](https://github.com/Dooders/AgentFarm/pull/749))

#### Changed

- Adaptive mutation telemetry now records `last_fitness_delta` / `best_fitness_delta` so generation summaries show the improvement that triggered each adaptation event. ([#749](https://github.com/Dooders/AgentFarm/pull/749))

---

### 2026-04-19

#### Added

- **Expanded evolvable loci** — promoted `gamma` and `epsilon_decay` into the hyperparameter chromosome with linear 8-bit encoding, mutation/crossover coverage, config projection, and documentation for future discrete/integer gene support. ([#719](https://github.com/Dooders/AgentFarm/pull/719))
- **Prioritized Experience Replay** — added `PrioritizedReplayBuffer`, configurable PER settings on decision/RL wrappers, Tianshou sampling with indices/importance weights, priority updates from TD errors, diagnostics, and documentation. ([#744](https://github.com/Dooders/AgentFarm/pull/744))

#### Changed

- Seeded resource regeneration now uses a vectorized, hash-based NumPy mask keyed by resource identity, position, timestep, and seed, making decisions reproducible and stable regardless of resource order. ([#723](https://github.com/Dooders/AgentFarm/pull/723))
- Simulation database logging now uses `flush_if_needed()` and `needs_flush` to avoid empty or premature buffer flushes, including support for sharded loggers. ([#725](https://github.com/Dooders/AgentFarm/pull/725))
- Core simulation bookkeeping now maintains incremental alive-agent sets, per-step resource-consumption deltas, batch-friendly spatial index updates, and single-pass metrics aggregation for better large-population performance. ([#730](https://github.com/Dooders/AgentFarm/pull/730))
- `AgentCore` and `Environment` gained observation caching, ordered O(1) agent membership/removal, in-memory genome-ID caching, and step/cumulative perception profiling APIs. ([#735](https://github.com/Dooders/AgentFarm/pull/735))

---

### 2026-04-18

#### Added

- **Adaptive mutation controller** — added `AdaptiveMutationConfig`, `AdaptiveMutationController`, normalized diversity tracking, per-gene mutation multipliers, CLI flags, and generation-summary telemetry for dynamic mutation rate/scale adjustment. ([#714](https://github.com/Dooders/AgentFarm/pull/714))
- **Reflective boundary mutation** (`BoundaryMode.REFLECT`) for `mutate_chromosome` — mutated gene values that overshoot a bound now bounce back instead of sticking at the wall, preventing boundary collapse and preserving population diversity. ([#708](https://github.com/Dooders/AgentFarm/pull/708), [#709](https://github.com/Dooders/AgentFarm/pull/709))
- **Soft boundary penalties** (`BoundaryPenaltyConfig` + `compute_boundary_penalty`) — an opt-in, linearly-ramped fitness penalty that discourages genes from crowding near their min/max limits without hard constraints. Subtract the result from raw fitness to apply. ([#708](https://github.com/Dooders/AgentFarm/pull/708))
- **BLX-alpha blend crossover** (`CrossoverMode.BLEND`) — a new gene-level crossover operator that samples each offspring gene from a uniform range extended by `blend_alpha` beyond the parent interval, allowing exploration beyond the parents' convex hull. ([#717](https://github.com/Dooders/AgentFarm/pull/717))
- **Multi-point crossover** (`CrossoverMode.MULTI_POINT`) — a new segment-based crossover operator that splits selected genes into alternating segments from each parent at `num_crossover_points` random pivot positions. ([#717](https://github.com/Dooders/AgentFarm/pull/717))
- **Crossover config knobs** (`blend_alpha`, `num_crossover_points`) — exposed in `EvolutionExperimentConfig` and the evolution CLI so users can tune the new operators per-experiment without code changes. ([#717](https://github.com/Dooders/AgentFarm/pull/717))
- **Crossover strategy comparison runner** (`scripts/compare_evolution_crossover_strategies.py`) — executes repeated evolution runs across crossover modes and writes `crossover_strategy_comparison.json` with per-mode fitness/diversity aggregates and per-seed raw results. ([#717](https://github.com/Dooders/AgentFarm/pull/717))

#### Changed

- Removed redundant boundary handling parameters from `EvolutionExperimentConfig`; boundary settings are now defined exclusively through `BoundaryPenaltyConfig` to avoid duplication. ([#709](https://github.com/Dooders/AgentFarm/pull/709))
- Improved `copilot-changelog-after-merge.yml` workflow based on review feedback. ([#707](https://github.com/Dooders/AgentFarm/pull/707))

#### Fixed

- **Prototype pollution** in Config Explorer editor (`farm/editor`): hardened `deepMerge` and `deepClone` to skip dangerous keys (`__proto__`, `constructor`, `prototype`) and only recurse into own-property destinations. Addresses code scanning alert #5. ([#711](https://github.com/Dooders/AgentFarm/pull/711))
- Added explicit `permissions: contents: read` to `tests.yml` workflow so all test jobs run with least-privilege `GITHUB_TOKEN` scopes. Addresses code scanning alert #3. ([#712](https://github.com/Dooders/AgentFarm/pull/712))
- Added explicit `permissions: contents: read` to `deterministic-simulation.yml` workflow. Addresses code scanning alert #1. ([#713](https://github.com/Dooders/AgentFarm/pull/713))

---

### 2026-04-17

#### Added

- **`HyperparameterChromosome`** — a typed, bounded, evolvable hyperparameter model with encoding/decoding, mutation (Gaussian), and crossover utilities. Each gene can be individually marked evolvable or fixed, with per-gene min/max bounds. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- **`EvolutionExperiment`** — a generation-based hyperparameter evolution runner supporting tournament and roulette selection strategies, lineage tracking, and persisted `evolution_generation_summaries.json` / `evolution_lineage.json` artifacts. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- `Genome.tournament_selection` and `Genome.roulette_selection` helpers with optional seeding and index-returning. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- CLI and plotting scripts for hyperparameter evolution runs. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Design document (`docs/design/hyperparameter_chromosome.md`) and experiment note (`docs/experiments/hyperparameter_evolution_convergence.md`) covering the new chromosome model, mutation strategies, and evolution convergence analysis. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Established GitHub Pages publishing for repository docs and devlog. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Added a dedicated devlog section and an initial post covering DNA-style hyperparameter evolution design and outcomes. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Introduced an automated workflow to request Copilot-authored changelog updates after merges to `main`. ([#707](https://github.com/Dooders/AgentFarm/pull/707))

#### Changed

- `AgentCore.reproduce()` now derives offspring configs from a mutated `HyperparameterChromosome`, storing the resulting chromosome on the child agent. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- Agent component config construction maps fields from the `learning` config section into the agent's decision config, enabling more flexible agent initialization. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- `EnvironmentalFactorsConfig` now raises `ValueError` for any normalized factor outside `[0.0, 1.0]` (or `tolerance_width <= 0`) instead of silently clamping, making misconfigured simulations fail fast and loudly. ([#690](https://github.com/Dooders/AgentFarm/pull/690))

#### Fixed

- Reproduction error handling hardened: a failed resource-cost deduction now aborts the reproduction cleanly rather than leaving partial state. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- `Environment.rollback_partial_agent_add` now compensates DB side effects by deleting the `AgentModel` row via a transaction instead of writing a spurious death record, and also purges associated metrics/DB buffers. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
- `DataLogger` now requires a truthy `simulation_id` and raises `ValueError` when it is missing or empty. ([#690](https://github.com/Dooders/AgentFarm/pull/690))

#### Security

- Clarified usage and security implications of the `allow_unsafe_unpickle` flag for checkpoint loading in crossover/fine-tune pipelines; explicit opt-in is now documented and recorded in audit reports. ([#690](https://github.com/Dooders/AgentFarm/pull/690))
