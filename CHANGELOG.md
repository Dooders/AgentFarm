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

### 2026-06-16

#### Docs

- **Inherited payload design** — added `docs/design/inherited_payload_design.md` (Issue #848) re-deriving what offspring should inherit from a parent's learned decision module after the Baldwinian vs Lamarckian A/B and early-life fitness follow-ups showed full policy-weight copy is a null and action priors already cross via Chromosome A; updated the hyperparameter-genome devlog open questions and clarified non-default Lamarckian wording. ([#905](https://github.com/Dooders/AgentFarm/pull/905))

---

### 2026-06-09

#### Added

- **Per-agent evolvable RL goals (Chromosome C)** — added nine `reward_*` loci to `HyperparameterChromosome`, rewired `AgentCore._calculate_reward` to read heritable goal weights (with caching), and introduced `IntrinsicGoalsExperiment` plus `scripts/run_intrinsic_goals_experiment.py` to run uniform-vs-unique goal-diversity arms with per-step action-mix and goal-gene telemetry; documented in `docs/experiments/intrinsic_evolution/intrinsic_goals.md`. Default chromosomes reproduce the historical reward formula unchanged. ([#891](https://github.com/Dooders/AgentFarm/pull/891), [#899](https://github.com/Dooders/AgentFarm/pull/899))
- **Determinism regression expansion** — hardened the determinism harness (`tests/test_deterministic.py`) to compare intermediate snapshots and DB contents without masking production seeding, added `tests/test_cross_process_determinism.py` for cross-process reproducibility, and wired both into the deterministic-simulation CI workflow. ([#898](https://github.com/Dooders/AgentFarm/pull/898), [#900](https://github.com/Dooders/AgentFarm/pull/900))
- **Test-suite coverage pass** — added property-based and utility tests, expanded coverage for `farm/research/analysis` and database model/utilities, fixed false-positive tests, and updated `.coveragerc` / `pytest.ini` / CI coverage settings. ([#896](https://github.com/Dooders/AgentFarm/pull/896), [#897](https://github.com/Dooders/AgentFarm/pull/897))

#### Fixed

- Snapshot determinism checks now require every requested step to be present (not just a subset), and early-terminated simulations compare only the overlapping step range instead of failing on missing later snapshots. ([#898](https://github.com/Dooders/AgentFarm/pull/898), [#900](https://github.com/Dooders/AgentFarm/pull/900))
- Action-mix telemetry no longer counts inactive agents or stale `last_action_name` values, fixing skewed aggregate plots in the intrinsic-goals experiment. ([#899](https://github.com/Dooders/AgentFarm/pull/899))
- Component tests no longer infinite-loop when probing genome IDs against a bare `Mock` database object. ([#898](https://github.com/Dooders/AgentFarm/pull/898))

#### Docs

- Devlog `2026-06-09-every-agent-a-different-goal.md` introduces the goal-diversity experiment and its first readouts. ([#891](https://github.com/Dooders/AgentFarm/pull/891))
- `docs/deterministic_simulations.md` now points at `tests/test_deterministic.py` and documents the cross-process determinism pytest entry point. ([#898](https://github.com/Dooders/AgentFarm/pull/898))

---

### 2026-06-05

#### Added

- **Early-life offspring fitness analysis** — added `scripts/analyze_early_life_fitness.py` and `scripts/plot_early_life_fitness.py` to quantify newborn-level fitness under Baldwinian vs Lamarckian inheritance (paired-seed deltas, startup transients, reward/action breakdowns) with unit tests and a devlog entry on measuring at the wrong level. ([#890](https://github.com/Dooders/AgentFarm/pull/890))

---

### 2026-06-01

#### Docs

- Refreshed the Baldwinian vs Lamarckian A/B devlog (`2026-05-21-baldwinian-vs-lamarckian-ab-harness.md`) with post-fix aggregate results, warm-start coverage context, and updated index links. ([#889](https://github.com/Dooders/AgentFarm/pull/889))

---

### 2026-05-31

#### Changed

- `compare_inheritance_arms.py` now reports warm-start coverage metrics (applied vs skipped counts and skip-reason breakdowns) so Lamarckian A/B readouts distinguish "inheritance disabled" from "inheritance attempted but incompatible." ([#888](https://github.com/Dooders/AgentFarm/pull/888))

---

### 2026-05-22

#### Added

- **Baldwinian vs Lamarckian inheritance A/B** — added `farm/core/policy_inheritance.py`, `InheritanceTelemetry`, `scripts/run_inheritance_mode_ab.py`, and `scripts/compare_inheritance_arms.py` to run matched intrinsic-evolution sweeps under Baldwinian (fresh policy) vs Lamarckian (parent policy warm-start) arms with paired-seed delta summaries, heatmaps, and lineage metrics; documented in `docs/experiments/intrinsic_evolution/inheritance_mode_ab.md`. ([#886](https://github.com/Dooders/AgentFarm/pull/886))
- **`farm/analysis/lineage_metrics.py`** — shared helper for churn/cluster-count extraction used by inheritance and crossover comparison scripts. ([#887](https://github.com/Dooders/AgentFarm/pull/887))

#### Changed

- **Decision-path refactor** — rewired `TianshouWrapper` action selection through explicit `_policy_q_values` + masked-softmax composition with optional action weights, fixing a defect where learned policy weights did not influence executed actions (invalidating pre-fix inheritance A/B aggregates); `DecisionModule.decide_action` now multiplicatively combines policy probabilities with per-action weights. ([#886](https://github.com/Dooders/AgentFarm/pull/886))
- Structured logging only attaches the JSON traceback processor when JSON output is enabled, keeping console rendering clean in non-JSON environments. ([#887](https://github.com/Dooders/AgentFarm/pull/887))

#### Fixed

- `compare_inheritance_arms` paired-delta heatmap handles null `mean_delta` entries in JSON summaries without crashing. ([#887](https://github.com/Dooders/AgentFarm/pull/887))
- `compare_inheritance_arms` robust paired deltas now require `n >= 2` seeds. ([#886](https://github.com/Dooders/AgentFarm/pull/886))

#### Docs

- Devlog `2026-05-21-baldwinian-vs-lamarckian-ab-harness.md` and backfilled CHANGELOG entries through 2026-05-20. ([#886](https://github.com/Dooders/AgentFarm/pull/886))

---

### 2026-05-20

#### Docs

- **Gene flow vs resource buffer devlog** — published `2026-05-18-gene-flow-and-the-buffer.md` reporting the crossover-enabled rerun, showing that gene flow robustly compresses speciation in conservative profiles, does not robustly collapse buffered trajectories, and leaves balanced profiles noisy; added the new speciation-trajectory figure and refreshed cross-links from prior devlogs, `crossover_rerun.md`, and `docs/devlog/index.md`. ([#882](https://github.com/Dooders/AgentFarm/pull/882))

---

### 2026-05-19

#### Docs

- Refreshed `AGENTS.md`: removed duplicate rows from the *Services overview* table and the duplicate Node-version gotcha, dropped a stale claim about pre-existing `tests/analysis/test_dominance.py` failures, updated the Ruff pre-existing warning count from ~136 to ~163, and noted that the pre-provisioned Cursor Cloud `venv/` uses Python 3.12. ([#883](https://github.com/Dooders/AgentFarm/pull/883))

---

### 2026-05-17

#### Fixed

- **DQN learning stack overhaul** — wired the epsilon-greedy schedule through the Tianshou DQN wrapper so exploration actually decays per action, plumbed `dqn_hidden_size` into the Q-network, corrected the YAML→`DecisionConfig` mapping so `learning.memory_size` / `batch_size` reach both legacy and current RL modules, and auto-scaled the deferred RL training throttle so each alive agent gets one gradient step per environment step by default (median `policy.learn()` calls jumped from 2 to 18 over a 100-step run, with measurable lifespan and weight-change gains). ([#878](https://github.com/Dooders/AgentFarm/pull/878), [#881](https://github.com/Dooders/AgentFarm/pull/881))
- `AgentCore.step` now logs decide/execute failures with context instead of silently swallowing them, making future RL regressions easier to diagnose. ([#881](https://github.com/Dooders/AgentFarm/pull/881))

#### Changed

- Default `max_learning_updates_per_step` is now `0`, which auto-scales the per-step training budget to the alive-agent count; the prior hard cap of 4 is no longer the default. ([#878](https://github.com/Dooders/AgentFarm/pull/878))

#### Added

- New `scripts/diagnose_dqn_learning.py` instruments the DQN learning loop (per-agent `policy.learn()` counts, `|Δw|₂`, live epsilon) plus reward-trend, late-vs-early decision-quality, and `--legacy` A/B baseline diagnostics. ([#878](https://github.com/Dooders/AgentFarm/pull/878), [#881](https://github.com/Dooders/AgentFarm/pull/881))
- Tests covering epsilon wiring, hidden-size plumbing, replay knob mapping, and the auto-scaled deferred training throttle. ([#878](https://github.com/Dooders/AgentFarm/pull/878))

#### Docs

- Devlog `2026-05-16-is-the-dqn-actually-learning.md` and refreshed deep-Q-learning / training-flow notes walk through the investigation, the five fixes, and the resulting decision-quality numbers. ([#878](https://github.com/Dooders/AgentFarm/pull/878), [#881](https://github.com/Dooders/AgentFarm/pull/881))

---

### 2026-05-14

#### Added

- **Crossover rerun experiment** — new `run_crossover_rerun.py` orchestrator re-runs the stable-profile seed sweep across multiple crossover arms, `compare_crossover_arms.py` performs paired-by-seed baseline-vs-treatment comparisons (including lineage cluster-count/churn extraction from `cluster_lineage.jsonl`) with robustness-style verdicting and plots, and `run_stable_profile_seed_sweep.py` gained crossover / co-parent CLI flags plus manifest/dry-run reporting. Documented in `docs/experiments/crossover_rerun.md`. ([#870](https://github.com/Dooders/AgentFarm/pull/870))
- **Long-horizon balanced-profile experiment** — `scripts/run_balanced_long_horizon_experiment.py` runs long-horizon intrinsic-evolution sweeps on the balanced profile with disk-backed databases, resume, and dry-run modes; mirrors the stable-profile sweep interface for consistent analysis. ([#869](https://github.com/Dooders/AgentFarm/pull/869))

#### Changed

- `scripts/run_stable_profile_seed_sweep.py` gained a `--disk-database` option (recommended for long-horizon runs) and a `_maybe_resume_skip` helper that skips already-completed runs when `--resume` is set. ([#869](https://github.com/Dooders/AgentFarm/pull/869))

#### Fixed

- `compare_crossover_arms` now bases its collapse verdict only on robust metrics, removing spurious flips from noisy seed-level signals. ([#870](https://github.com/Dooders/AgentFarm/pull/870))

---

### 2026-05-13

#### Docs

- Devlog `2026-05-12-seed-sweep-reality-check.md` presents the 6-seed-per-profile replication of the resource-buffer comparison, showing that previously reported `learning_rate` and `ensemble_size` direction-flips were single-seed artifacts while speciation divergence remains robust; updated the earlier `2026-05-04` post to link to the replication and clarify which claims did not survive multi-seed testing, added a "dynamic tipping region" glossary entry, and refreshed the devlog index. ([#868](https://github.com/Dooders/AgentFarm/pull/868))

#### Changed

- Bumped several `farm/editor` Babel dependencies (`@babel/code-frame`, `generator`, `helper-module-imports/transforms`, `helper-plugin-utils`, `helper-validator-identifier`, `parser`, `plugin-transform-modules-systemjs`) for compatibility and security. ([#868](https://github.com/Dooders/AgentFarm/pull/868))

---

### 2026-05-12

#### Added

- **Stable-profile seed-sweep infrastructure** — `scripts/run_stable_profile_seed_sweep.py` runs `IntrinsicEvolutionExperiment` for every `(profile, seed)` pair with the fixed comparison settings and writes a `sweep_manifest.json`; `scripts/analyze_stable_profile_seed_sweep.py` aggregates the JSONL artifacts into per-profile mean/variance/95% t-CI summaries for speciation index, trajectory slope, and gene shifts, classifies each as robustly convergent, robustly direction-flipping, or seed-sensitive, and emits Markdown + plots. Backed by 44 unit/integration tests. ([#863](https://github.com/Dooders/AgentFarm/pull/863))

#### Docs

- Docs site gained a responsive "On this page" TOC sidebar (auto-generated heading anchors, active-section highlighting, smooth scroll) plus a full-screen Mermaid diagram viewer with `svg-pan-zoom` (zoom/pan/reset, keyboard + backdrop close) and refreshed layout CSS. ([#866](https://github.com/Dooders/AgentFarm/pull/866))

---

### 2026-05-11

#### Added

- **Ecology visualization script** for intrinsic evolution analysis, plus a foraging-grid animated GIF (`foraging-grid.gif`) and the standalone `scripts/make_foraging_gif.py` renderer (loads per-step agent/resource state from the simulation SQLite DB, overlays birth/death markers, optional sparkline timeline) embedded in the foraging devlog. ([#862](https://github.com/Dooders/AgentFarm/pull/862), [#865](https://github.com/Dooders/AgentFarm/pull/865))

#### Docs

- **Experiments catalog** — rewrote `docs/experiments.md` as a curated catalog of currently defined experiments (Intrinsic Evolution, Hyperparameter Evolution Convergence, Multi-Seed Cohort, Memory Agent, One of a Kind, Rabbit's Foot) with status, runner/CLI references, and links to detailed docs, and split the generic `ExperimentRunner` how-to into a new `docs/experiment_runner.md`. Renamed `Design & Considerations.md` → `DesignConsiderations.md` so catalog links resolve cleanly. ([#864](https://github.com/Dooders/AgentFarm/pull/864), [#865](https://github.com/Dooders/AgentFarm/pull/865))

---

### 2026-05-10

#### Changed

- Bumped `@babel/plugin-transform-modules-systemjs` from 7.27.1 to 7.29.4 in `farm/editor`. ([#861](https://github.com/Dooders/AgentFarm/pull/861))

---

### 2026-05-09

#### Added

- **Verified spatial benchmark artifacts** — added `--verified` mode to `comprehensive_spatial_benchmark.py` (deterministic RNG, optional `PYTHONHASHSEED=0`) that writes committed `benchmarks/results/spatial_benchmark_verified.json` and `SPATIAL_BENCHMARK_VERIFIED.md` with host metadata, batch-vs-immediate microbenchmarks, and a new simulation-style interleaved `step_workload_benchmark`. Backed by a unit test asserting artifact existence and schema; `.gitignore` keeps other benchmark outputs local. ([#858](https://github.com/Dooders/AgentFarm/pull/858), [#860](https://github.com/Dooders/AgentFarm/pull/860))

#### Docs

- `spatial_indexing_performance.md` and `FEATURES.md` now align with the real APIs (`SpatialIndexConfig`, current `get_nearby` / `get_nearest` shapes, named-index querying, SciPy KD-tree note), and `AGENTS.md` documents the verified-artifact regeneration command. ([#858](https://github.com/Dooders/AgentFarm/pull/858))

---

### 2026-05-08

#### Docs

- Added a README pointer near the active-development note directing readers who want a mature, widely adopted ABM or RL stack toward Mesa and the common RL ecosystems instead of expecting AgentFarm to fill that niche. ([#856](https://github.com/Dooders/AgentFarm/pull/856), [#857](https://github.com/Dooders/AgentFarm/pull/857))
- Intrinsic-evolution devlog gained speciation-index and gene-trajectory figures, the phylogenetic/intrinsic lineage plotting moved to a tidy hierarchical layout (leaf-centered x positions, larger qualitative palette, elbow edges, optional lineage bands), and docs table styling was upgraded for readability. ([#859](https://github.com/Dooders/AgentFarm/pull/859))

#### Changed

- Tightened Mermaid initialization from `securityLevel: "loose"` to `"strict"` on the docs site. ([#857](https://github.com/Dooders/AgentFarm/pull/857))

---

### 2026-05-07

#### Docs

- Added Mermaid.js v10 (CDN-loaded) plus `mermaid-init.js` to the docs site so fenced `language-mermaid` blocks render as responsive, horizontally scrollable diagrams. ([#855](https://github.com/Dooders/AgentFarm/pull/855))
- Refreshed the docs visual system: swapped the site logo to `logo.png`, refined section/feature/quickstart/post card styling, and added detailed Rouge syntax-highlighting rules plus new CSS variables for code tokens (keywords, strings, numbers, functions, operators, comments). ([#852](https://github.com/Dooders/AgentFarm/pull/852))
- Added a global project-status notice banner under the docs site header, with new `.site-notice` styling in `agentfarm.css`. ([#854](https://github.com/Dooders/AgentFarm/pull/854))
- Rewrote `docs/deep_q_learning.md` and related AI/ML pages to separate the active DQN path (`DecisionModule` + `DQNWrapper`) from the legacy `BaseDQNModule`, with concrete `DecisionConfig` examples (replay/PER, training flow, target sync, action masking, replay diagnostics); swapped TD3 references to DDPG, and updated spatial-indexing docs to use `SpatialIndexConfig` with current `get_nearby` / `get_nearest` patterns. ([#853](https://github.com/Dooders/AgentFarm/pull/853))

---

### 2026-05-06

#### Changed

- **Deterministic simulation CI workflow** now triggers on both `main` and `dev` (plus manual `workflow_dispatch`), pins `actions/setup-python@v5` with pip caching keyed off dependency files, sets `PYTHONHASHSEED` / `PYTHONUNBUFFERED`, splits checks into targeted component / regression / CLI-smoke / extended-CLI pytest steps with per-step timeouts, and uploads `simulations/` artifacts on failure. ([#842](https://github.com/Dooders/AgentFarm/pull/842))

#### Docs

- Backfilled CHANGELOG entries for 2026-04-25 through 2026-05-05 and polished the 2026-05-04 resource-buffer devlog (YAML front matter fix, table formatting, GitHub-issue links for follow-ups, number/spacing cleanups); replaced the gene list in the hyperparameter-genome devlog with a chromosome anatomy figure and reframed several open-questions/next-steps as linked issues. ([#851](https://github.com/Dooders/AgentFarm/pull/851))

---

### 2026-05-05

#### Added

- **Deferred batch RL training scheduler** — added per-step deferred batching so agent learning updates can be grouped and applied more efficiently during simulation steps. ([#836](https://github.com/Dooders/AgentFarm/pull/836))
- **Deterministic simulation workflow expansion** — broadened deterministic CI coverage to better exercise regression paths and reproducibility checks. ([#839](https://github.com/Dooders/AgentFarm/pull/839))
- **Stage 0 step-loop profiling baseline** — introduced baseline cProfile artifacts and reporting for the 30x30 / 30-agent scenario to ground later performance work. ([#834](https://github.com/Dooders/AgentFarm/pull/834))

#### Changed

- Simulation config validation now enforces non-negative `max_learning_updates_per_step`, making invalid scheduler settings fail fast at config load time.
- Test automation now includes explicit deterministic pytest execution and a nightly heavy-tests workflow to improve continuous regression coverage.

#### Docs

- GitHub Pages docs received a minimalist, content-first redesign with updated layout styling and navigation polish. ([#837](https://github.com/Dooders/AgentFarm/pull/837))

---

### 2026-05-04

#### Added

- **Intrinsic evolution resource-buffer analysis** — added comparison/reporting support to evaluate buffered-resource behavior in intrinsic evolution experiments.

#### Changed

- Refined runtime logging levels across core environment/decision paths to reduce noise while keeping actionable diagnostics.
- Removed obsolete intrinsic-evolution artifact files tied to superseded memory-size experiments to keep experiment outputs focused.

#### Fixed

- Fixed duplicated `estimation_step` keyword handling in `DQNPolicy` and streamlined action-materialization to avoid redundant work.

---

### 2026-05-03

#### Added

- **Tier 1 evolvable gene wiring** — enabled and fully wired chromosome A/B Tier 1 loci into live simulation behavior, with supporting tests/docs updates. ([#832](https://github.com/Dooders/AgentFarm/pull/832))

#### Docs

- Expanded glossary cross-linking and terminology explanations to improve discoverability of key AgentFarm concepts.

#### Fixed

- Anchored `.gitignore` handling for `experiments/` to repository root, preventing accidental overmatching in nested paths.

---

### 2026-05-01

#### Changed

- Refactored memory-mapped storage configuration/implementation to simplify setup and improve maintainability.
- Enhanced environmental and temporal grid handling with lazy allocation to reduce unnecessary upfront allocations.
- Consolidated duplicated zero-padding window logic and removed an unused `VISIBILITY` world-layer path.

#### Fixed

- Applied cleanup passes for repeated empty-except and unused-import findings raised by automated review checks.

---

### 2026-04-30

#### Added

- **GPU-accelerated spatial computations** with CPU fallback paths — introduced GPU kernel support for spatial-index operations while preserving CPU execution compatibility. ([#827](https://github.com/Dooders/AgentFarm/pull/827))

#### Changed

- Applied post-review refinements to GPU kernel behavior, `SpatialIndex` docs, initial-conditions validation messaging, and related tests/examples.

#### Fixed

- `run_simulation` no longer rebuilds initial-diversity config from `--seed` alone, preventing unintentional config mutation.

---

### 2026-04-29

#### Added

- **Speciation quality/cadence upgrades** — added stability-score-aware quality bundling and support for separate clustering cadence in `GeneTrajectoryLogger` snapshots. ([#819](https://github.com/Dooders/AgentFarm/pull/819))
- **Cross-type pollination option** — introduced `allow_cross_type_pollination` for co-parent selection policy control. ([#821](https://github.com/Dooders/AgentFarm/pull/821))
- **Initial genotype diversity seeding** — implemented platform-wide initial-diversity seeding hooks and expanded intrinsic runner handling for initial-conditions setup. ([#825](https://github.com/Dooders/AgentFarm/pull/825))

#### Fixed

- Corrected speciation index derivation/cache behavior and tightened edge-case handling around zero-chromosome snapshots and cluster-matching quality calculations.

#### Docs

- Updated intrinsic-evolution and snapshot-field documentation to clarify new `speciation_quality`/initial-conditions behavior.

---

### 2026-04-28

#### Added

- **Auto-tuned DBSCAN clustering parameters** — added data-driven `eps`/`min_samples` suggestion and auto-tuning support for speciation clustering. ([#812](https://github.com/Dooders/AgentFarm/pull/812))
- **Improved cluster matching mechanics** — upgraded matching beyond greedy centroid pairing, with stronger validation around distance gates/thresholds. ([#814](https://github.com/Dooders/AgentFarm/pull/814))
- **Richer speciation quality metrics** — added multi-metric quality bundling beyond silhouette-only scoring for more informative speciation diagnostics. ([#816](https://github.com/Dooders/AgentFarm/pull/816))

#### Fixed

- Hardened Hungarian/gating and percentile-validation edge cases in speciation matching, plus supporting comments/tests for numeric-stability behavior.

---

### 2026-04-27

#### Added

- **Optional feature scaling before clustering** — added pre-clustering scaling controls for speciation workflows to stabilize distance-sensitive clustering runs. ([#807](https://github.com/Dooders/AgentFarm/pull/807))

#### Changed

- Improved speciation clustering validation and defaults to make clustering behavior more robust across datasets.

#### Docs

- Added intrinsic evolution experiment documentation updates with accompanying data artifact references.

---

### 2026-04-26

#### Added

- **Intrinsic evolution experiment tooling** — added runner CLI flows, experiment outputs, and analysis scripts for repeated intrinsic evolution study workflows. ([#797](https://github.com/Dooders/AgentFarm/pull/797))

#### Changed

- Improved plotting and lineage handling in speciation analysis outputs for clearer trajectory interpretation.
- Applied review-driven hardening around docs and projection/validation behaviors in intrinsic evolution flows.

#### Fixed

- `GeneTrajectoryLogger` speciation now works without an output directory and correctly resets lineage tracking after extinction events.

---

### 2026-04-25

#### Added

- **Speciation/niche detection over chromosome trajectories** for intrinsic evolution experiments, including clustering-based lineage interpretation and niche-oriented analysis hooks. ([#794](https://github.com/Dooders/AgentFarm/pull/794))

#### Fixed

- Applied automated cleanup for repeated empty-except and unused-import issues detected in review automation during the speciation integration cycle.

---

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
