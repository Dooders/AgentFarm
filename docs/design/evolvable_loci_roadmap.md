# Evolvable Loci Roadmap

This design note proposes a forward roadmap for expanding evolvable loci in AgentFarm's hyperparameter chromosome model.

It is aligned to project goals:

- improve research value for complex adaptive systems
- maintain reproducible, experiment-grade outputs
- scale evolution across more parameters without losing stability

## Current State

The current chromosome model supports real-valued loci and already evolves:

- `learning_rate`
- `gamma`
- `epsilon_decay`

`memory_size` exists as a fixed placeholder. Mutation, crossover, and per-generation statistics are in place, and convergence artifacts now report multi-gene stats.

## Roadmap Goals

1. Expand expressive search space while preserving bounded, typed validation.
2. Keep outputs interpretable for research comparisons and post-hoc analysis.
3. Prevent regressions in reproducibility, runtime cost, and boundary behavior.
4. Enable mixed-gene evolution (continuous + integer + categorical) in a staged way.

## Guiding Principles

- Keep schema strict: add types only when invariants and tests are ready.
- Prefer incremental rollout with per-phase exit criteria.
- Add loci by hypothesis, not by volume: each new gene should map to a research question.
- Keep telemetry first-class: every evolved locus must appear in summary artifacts.

## Locus Selection Framework

Add a locus only when it passes all checks:

- Has a clear mechanistic effect on adaptation or emergent behavior.
- Has safe numeric/categorical bounds and stable defaults.
- Can be projected to runtime config without ambiguous coercion.
- Can be measured with existing metrics or a planned metric addition.

## Phase Plan

### Phase 1: Continuous Gene Expansion (near-term)

Scope:

- Add 3-6 additional continuous decision-policy loci.
- Recommended first candidates:
  - `tau` (target network update smoothing)
  - `gradient_clip_norm`
  - `batch_size` as continuous proxy only if represented as derived integer later (otherwise defer to Phase 2)
  - optional action module learning rates (`move_learning_rate`, `attack_learning_rate`, etc.) behind feature flag

Implementation targets:

- Extend default chromosome registry and encoding specs.
- Use log-scale encoding for order-of-magnitude parameters.
- Add per-gene mutation scale defaults to reduce early collapse.

Exit criteria:

- New loci appear in `gene_statistics` and `best_chromosome`.
- No loss of determinism for seeded runs.
- Regression tests cover serialize/encode/decode/mutation/crossover/projection for each new locus.

### Phase 2: Integer Gene Type (short-term)

Scope:

- Introduce `GeneValueType.INTEGER`.
- Migrate `memory_size` from fixed to evolvable.
- Add 1-2 more integer loci after `memory_size` validation.

Implementation targets:

- Enforce integer bounds/default/value validation in schema.
- Enforce projection-time integer bound checks after rounding.
- Ensure mutation operators for integer genes use discrete-safe steps.

Exit criteria:

- `memory_size` evolves end-to-end in experiment runs.
- Integer loci never violate config bounds in lineage or runtime config.
- Mixed real+integer crossover is deterministic under seed.

### Phase 3: Categorical/Binary Gene Support (mid-term)

Scope:

- Add discrete categorical support for strategy toggles with small cardinality.
- Example candidates:
  - boundary strategy (`clamp` vs `reflect`)
  - selection strategy (`tournament` vs `roulette`) in controlled meta-runs
  - crossover operator family (`uniform`, `blend`, `multi_point`)

Implementation targets:

- Introduce explicit categorical encoding map per gene.
- Add mutation as category flip/sampled transition matrix, not numeric perturbation.
- Ensure summaries report category frequencies per generation.

Exit criteria:

- Categorical loci can co-evolve with real/integer loci.
- Summaries remain machine-readable and comparable across runs.
- No runtime failures from invalid categorical projection.

### Phase 4: Hierarchical and Module-Specific Loci (mid-to-long term)

Scope:

- Evolve module-scoped hyperparameters separately from global policy parameters.
- Candidate groups:
  - movement module exploration profile
  - attack/share module learning dynamics
  - reproduction thresholds or costs where biologically meaningful

Implementation targets:

- Add namespace-aware loci (for example `decision.gamma`, `attack.learning_rate`).
- Add compatibility guardrails so absent modules cannot receive locus updates.
- Add weighted fitness decomposition to avoid overfitting one module.

Exit criteria:

- Multi-module loci evolve without schema collisions.
- Cross-module effects are observable in analysis pipelines.
- Run-time overhead remains acceptable for baseline experiment sizes.

### Phase 5: Meta-Evolution and Curriculum Coupling (long-term)

Scope:

- Evolve adaptation schedules, not only fixed scalar values.
- Candidate meta-loci:
  - mutation-rate schedule parameters
  - exploration decay schedule parameters
  - curriculum phase transition thresholds

Implementation targets:

- Represent schedule genes as compact parameterized functions.
- Persist schedule state and values in summaries for reproducibility.
- Add replay tooling to reconstruct schedule behavior from lineage artifacts.

Exit criteria:

- Meta-loci improve convergence or robustness on at least one benchmark family.
- Replay and analysis tooling can fully reconstruct evolved schedules.

## Cross-Cutting Workstreams

### Testing

- Add a mixed-loci test matrix:
  - real only
  - real + integer
  - real + integer + categorical
- Keep deterministic seeded tests for mutation and crossover edge cases.

### Telemetry and Analysis

- Maintain required summary fields for every evolvable locus.
- Add per-locus diversity and boundary-pressure indicators.
- Add categorical distribution summaries once categorical genes are introduced.

### Runtime and Performance

- Track per-generation overhead as locus count increases.
- Set practical max-locus guardrails for smoke and convergence suites.
- Add profiling checkpoints when moving from each phase.

## Suggested Milestones

- M1: Phase 1 complete (expanded continuous set in production artifacts)
- M2: Phase 2 complete (`memory_size` evolvable with integer-safe invariants)
- M3: Phase 3 complete (first mixed-type convergence experiments)
- M4: Phase 4 complete (module-scoped evolution enabled)
- M5: Phase 5 pilot (meta-loci on one benchmark track)

## Risks and Mitigations

- Search-space explosion
  - Mitigation: phase gates, per-gene mutation scaling, adaptive mutation defaults.
- Boundary collapse in larger loci sets
  - Mitigation: reflective boundaries and calibrated soft penalties.
- Reduced interpretability
  - Mitigation: mandatory per-locus telemetry and run manifests.
- Regression in reproducibility
  - Mitigation: seeded test coverage and deterministic artifact checks in CI.

## Immediate Next Steps

1. Approve Phase 1 candidate loci list.
2. Implement Phase 1 schema/test updates.
3. Run convergence suite and compare against current regenerated baseline artifacts.
4. Open Phase 2 implementation branch for integer gene type and `memory_size` activation.
