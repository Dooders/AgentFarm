# Evolution Surface Catalog: Candidate Genes & Chromosomes

A proposal-style catalog of new genes and chromosome groupings that can be added
to AgentFarm's existing evolutionary framework to grow the evolution surface.
Each tier below is independent and can be adopted in isolation.

## Framework grounding

These recommendations are anchored to what is currently wired up in the codebase:

- `farm/core/hyperparameter_chromosome.py` — `HyperparameterGene` +
  `HyperparameterChromosome`. Currently 4 default genes:
  `learning_rate`, `gamma`, `epsilon_decay`, `memory_size`.
  Only `GeneValueType.REAL` is implemented; supports linear/log scale and
  optional bit-width quantization. Mutation supports gaussian / multiplicative,
  with `clamp` / `reflect` / `interior_biased` boundary modes. Crossover
  supports single-point, uniform, BLEND (BLX-α), and multi-point.
- `farm/core/genome.py` — `Genome` carries `action_set` (action weights) and
  `module_states` (state dicts). Selection (tournament / roulette) lives here.
- `farm/core/initial_diversity.py` — seeding modes `none` /
  `independent_mutation` / `unique` / `min_distance`.
- `farm/runners/intrinsic_evolution_experiment.py` — in-situ evolution:
  chromosome inheritance + crossover + density-dependent reproduction cost.
- `farm/core/agent/config/component_configs.py` + `farm/core/decision/config.py`
  — the master `AgentComponentConfig` and `DecisionConfig` exposing every knob
  that *could* become a locus.
- `apply_chromosome_to_learning_config` currently writes back into
  `agent.config.decision` only, by attribute-name match.

---

## Tier 1 — drop-in additions (no framework changes) — **IMPLEMENTED**

These extend the existing **learning chromosome** by adding `HyperparameterGene`
entries to `DEFAULT_HYPERPARAMETER_GENES`. Names already match `DecisionConfig`
attributes, so `apply_chromosome_to_learning_config` will project them
automatically.

> **Status:** All Chromosome A and B genes below are now part of
> `DEFAULT_HYPERPARAMETER_GENES` in `farm/core/hyperparameter_chromosome.py`
> (31 evolvable loci total).  Encodings are registered in
> `DEFAULT_GENE_ENCODINGS`; integer-typed `DecisionConfig` fields are
> projected via the existing rounding-with-bounds-check path.

### Runtime-effect audit

All 31 default chromosome loci now have a runtime consumer.  Each gene
flows into ``DecisionConfig`` via
:func:`apply_chromosome_to_learning_config` and is read by exactly one
identified pipeline path, summarized below.

#### Chromosome A — learning / RL hyperparameters

| Gene | Consumer | Notes |
|------|----------|-------|
| `learning_rate` | `BaseDQNModule.__init__`, Tianshou builders | Adam optimizer LR |
| `gamma` | `BaseDQNModule`, all Tianshou wrappers | discount factor |
| `epsilon_start` | `BaseDQNModule`, Tianshou DQN (`eps_train`), fallback | exploration init |
| `epsilon_min` | `BaseDQNModule`, Tianshou DQN (`eps_test`, `eps_train_final`) | exploration floor |
| `epsilon_decay` | `BaseDQNModule` | decay multiplier |
| `memory_size` | `BaseDQNModule.memory` | replay buffer size |
| `tau` | `BaseDQNModule.tau` | soft target update |
| `batch_size` | `BaseDQNModule.train` | gradient noise |
| `target_update_freq` | Tianshou `DQNPolicy(target_update_freq=…)` (explicit kwarg) | hard target sync cadence |
| `dqn_hidden_size` | `BaseDQNModule.q_network`, target net | network capacity |
| `rl_train_freq` | All Tianshou wrappers (`train_freq=…`) | training cadence |
| `per_alpha` | All Tianshou wrappers' replay buffer | PER prioritization |
| `per_beta_start` | All Tianshou wrappers' replay buffer | IS-correction warmup |
| `per_beta_end` | All Tianshou wrappers' replay buffer | IS-correction final |
| `ensemble_size` | `RandomForestActionSelector.n_estimators` (auto-injected when `algorithm_type ∈ {"random_forest", "gradient_boost"}`) | tree-ensemble width |

#### Chromosome B — action-policy priors

Two-stage pipeline:

1. **Base weights** — `move_weight`, `gather_weight`, `share_weight`,
   `attack_weight`, `reproduce_weight` are bridged into
   ``core.actions[i].weight`` by
   ``AgentCore._customize_action_weights`` (at construction) and
   ``AgentCore.refresh_action_weights_from_decision_config`` (after
   chromosome re-application).  Per-agent gene values win over the
   environment-level ``agent_parameters`` block; default-valued genes
   fall through unchanged.
2. **State-aware re-weighting** — ``LearningAgentBehavior.decide_action``
   calls ``compute_action_weights`` from
   ``farm/core/decision/action_weight_policy.py`` to scale the base
   weights by the eight multipliers gated by the three thresholds:
   - `move_mult_no_resources` (no resources nearby)
   - `gather_mult_low_resources` (resource ratio < 0.5)
   - `share_mult_wealthy` (resource ratio ≥ 0.7)
   - `share_mult_poor` (resource ratio < 0.3)
   - `attack_mult_desperate` (starvation risk ≥ `attack_starvation_threshold`)
   - `attack_mult_stable` (health ratio ≥ 1 − `attack_defense_threshold`)
   - `reproduce_mult_wealthy` (resource ratio ≥ `reproduce_resource_threshold`)
   - `reproduce_mult_poor` (resource ratio < `reproduce_resource_threshold`)

The scaled vector is then passed as `action_weights` to
``DecisionModule.decide_action``, which already supports per-action
biasing of both Q-value-based exploitation and weighted random
exploration.

### Chromosome A — Learning / RL hyperparameters

Extends `DEFAULT_HYPERPARAMETER_GENES`.

| Gene | Range | Default | Scale | Notes |
|---|---|---|---|---|
| `learning_rate` *(existing)* | 1e-6 – 1.0 | 0.001 | log | already there |
| `gamma` *(existing)* | 0.0 – 1.0 | 0.99 | linear | already there |
| `epsilon_decay` *(existing)* | ~0 – 1.0 | 0.995 | linear | already there |
| `memory_size` *(existing)* | 1 – 1e6 | 10000 | linear | already projected to int |
| `epsilon_start` | 0.0 – 1.0 | 1.0 | linear | exploration aggressiveness |
| `epsilon_min` | 0.0 – 0.5 | 0.01 | log | floor for exploration |
| `tau` | 1e-4 – 0.5 | 0.005 | log | soft-update speed; major behaviour driver |
| `batch_size` | 8 – 1024 | 32 | log | int-projected; affects gradient noise |
| `target_update_freq` | 1 – 5000 | 100 | log | int-projected; complementary to `tau` |
| `dqn_hidden_size` | 8 – 512 | 64 | log | int-projected; capacity vs. overfit |
| `rl_train_freq` | 1 – 64 | 4 | log | how often to update |
| `per_alpha` | 0.0 – 1.0 | 0.6 | linear | only meaningful when `replay_strategy="prioritized"` |
| `per_beta_start` | 0.0 – 1.0 | 0.4 | linear | IS-correction warmup |
| `per_beta_end` | 0.0 – 1.0 | 1.0 | linear | IS-correction final |
| `ensemble_size` | 1 – 16 | 1 | linear | int-projected |

### Chromosome B — Action-policy priors

Live on `DecisionConfig`. These are *behavioural priors* that bias the policy
before learning kicks in — they are extremely high-leverage selection targets.

| Gene | Range | Default |
|---|---|---|
| `move_weight` | 0.0 – 2.0 | 0.30 |
| `gather_weight` | 0.0 – 2.0 | 0.30 |
| `share_weight` | 0.0 – 2.0 | 0.15 |
| `attack_weight` | 0.0 – 2.0 | 0.10 |
| `reproduce_weight` | 0.0 – 2.0 | 0.15 |
| `move_mult_no_resources` | 0.5 – 3.0 | 1.5 |
| `gather_mult_low_resources` | 0.5 – 3.0 | 1.5 |
| `share_mult_wealthy` | 0.5 – 3.0 | 1.3 |
| `share_mult_poor` | 0.0 – 1.5 | 0.5 |
| `attack_mult_desperate` | 0.5 – 3.0 | 1.4 |
| `attack_mult_stable` | 0.0 – 1.5 | 0.6 |
| `reproduce_mult_wealthy` | 0.5 – 3.0 | 1.4 |
| `reproduce_mult_poor` | 0.0 – 1.5 | 0.3 |
| `attack_starvation_threshold` | 0.0 – 1.0 | 0.5 |
| `attack_defense_threshold` | 0.0 – 1.0 | 0.3 |
| `reproduce_resource_threshold` | 0.0 – 1.0 | 0.7 |

The action-prior genes are the single largest evolution-surface win available
without touching the framework. They are the dial between "aggressor",
"cooperator", "hoarder", and "explorer" niches.

---

## Tier 2 — needs a small extension to `apply_chromosome_*`

To evolve traits living on the *other* component configs (`combat`, `resource`,
`movement`, `perception`, `reproduction`, `reward`, `communication`),
generalize the projection step to walk `AgentComponentConfig` by namespaced
gene names like `combat.starting_health`.

Concretely: in `apply_chromosome_to_learning_config` (rename to
`apply_chromosome_to_agent_config`), if a gene name contains `.`, split into
`(component, attr)` and update `getattr(agent_config, component)` instead of
`agent_config.decision`. The frozen component dataclasses already work with
`dataclasses.replace`, so the SRP/OCP split is clean.

### Chromosome C — Combat / physiology (`combat.*`)

| Gene | Range | Default | Scale |
|---|---|---|---|
| `combat.starting_health` | 20 – 400 | 100 | log |
| `combat.base_attack_strength` | 0 – 50 | 10 | linear |
| `combat.base_defense_strength` | 0 – 30 | 5 | linear |
| `combat.defense_damage_reduction` | 0.0 – 0.95 | 0.5 | linear |
| `combat.defense_timer_duration` | 1 – 20 | 3 | linear (int) |

### Chromosome D — Metabolism (`resource.*`)

| Gene | Range | Default | Scale |
|---|---|---|---|
| `resource.base_consumption_rate` | 0.1 – 5.0 | 1.0 | log |
| `resource.starvation_threshold` | 1 – 1000 | 100 | log (int) |

### Chromosome E — Reproduction (`reproduction.*`)

| Gene | Range | Default | Scale |
|---|---|---|---|
| `reproduction.offspring_cost` | 1.0 – 100.0 | 5.0 | log |
| `reproduction.offspring_initial_resources` | 0.0 – 100.0 | 10.0 | log |

These pair well with the runner's existing `ReproductionPressureConfig`: agents
that lower their own cost too far will be punished by density terms, producing
real selection.

### Chromosome F — Movement & perception

| Gene | Range | Default | Scale |
|---|---|---|---|
| `movement.max_movement` | 0.5 – 32.0 | 8.0 | log |
| `movement.perception_radius` | 1 – 20 | 5 | linear (int) |
| `perception.perception_radius` | 1 – 20 | 5 | linear (int) |

Wider perception is "metabolically" costly only if you couple it to
`resource.base_consumption_rate` in a custom reward-shaping rule.

### Chromosome G — Communication

| Gene | Range | Default | Scale |
|---|---|---|---|
| `communication.communication_range` | 0.0 – 200.0 | 50.0 | linear |
| `communication.broadcast_cost` | 0.0 – 5.0 | 0.0 | linear |
| `communication.reward_per_message` | -0.1 – 0.1 | 0.01 | linear |
| `communication.inbox_capacity` | 1 – 200 | 20 | log (int) |

This is one of the more interesting evolution surfaces — lets "talkative" vs.
"silent" lineages emerge.

### Chromosome H — Reward shaping (meta-evolution)

Risky but powerful: evolvable reward weights produce drift away from the
designer's reward signal, which is exactly what you want if you're studying
intrinsic motivation.

| Gene | Range | Default |
|---|---|---|
| `reward.resource_reward_scale` | 0.0 – 5.0 | 1.0 |
| `reward.health_reward_scale` | 0.0 – 5.0 | 0.5 |
| `reward.survival_bonus` | 0.0 – 1.0 | 0.1 |
| `reward.death_penalty` | -50.0 – 0.0 | -10.0 |
| `reward.age_bonus` | 0.0 – 0.1 | 0.01 |
| `reward.combat_success_bonus` | 0.0 – 10.0 | 2.0 |
| `reward.reproduction_bonus` | 0.0 – 20.0 | 5.0 |
| `reward.cooperation_bonus` | 0.0 – 10.0 | 1.0 |

> If you do this, freeze the *evaluation* fitness used by
> `evolution_experiment.py` to a fixed external metric (population persistence,
> diversity, etc.) so you don't reward Goodharting. Keep these genes off by
> default and behind an explicit opt-in flag.

---

## Tier 3 — needs new gene **value types**

Today only `GeneValueType.REAL` is implemented. The doc string explicitly says
discrete/binary are reserved. Adding them unlocks a class of architectural /
topological genes.

### Suggested new value types

- `INTEGER` — bounded int with native rounding (today you fake it via REAL +
  projection in `apply_chromosome_to_learning_config`; a real type lets you
  express the contract on the gene itself and validate).
- `BINARY` — 0/1; trivial as a single quantization-1 REAL but cleaner to read.
- `CATEGORICAL` — fixed list of choice strings/ints, with a Hamming-style
  mutation operator and per-allele mutation probability.

Crossover already works for these (single-point / uniform / multi-point); only
BLEND would need a guard to no-op for non-real types.

### Chromosome I — Architecture (categorical/integer)

| Gene | Type | Domain |
|---|---|---|
| `decision.algorithm_type` | CATEGORICAL | `dqn`, `ddqn`, `ppo`, `sac`, `a2c`, `mlp`, `random_forest`, `gradient_boost`, `knn` |
| `decision.replay_strategy` | CATEGORICAL | `uniform`, `prioritized` |
| `decision.use_exploration_bonus` | BINARY | 0 / 1 |
| `decision.dqn_hidden_size` | INTEGER | 8 – 512 |
| `decision.ensemble_size` | INTEGER | 1 – 16 |
| `decision.feature_engineering[*]` | BINARY × N | one bit per available flag |

### Chromosome J — Action repertoire (binary mask)

The current `Genome.action_set` evolves *weights*; turning **presence** into
binary genes lets agents lose actions entirely.

| Gene | Type | Notes |
|---|---|---|
| `action.attack_enabled` | BINARY | gates whether `attack` is in the action list |
| `action.share_enabled` | BINARY | gates `share` |
| `action.reproduce_enabled` | BINARY | gates `reproduce` |
| `action.defend_enabled` | BINARY | gates `defend` |
| `action.communicate_enabled` | BINARY | gates communicate, when communication is on |

Pair with weight genes for full topology + magnitude evolution.

---

## Tier 4 — multi-chromosome organism (small structural change)

`HyperparameterChromosome` is currently a flat tuple. To express *karyotype*
(multiple chromosomes that recombine independently), introduce a
`ChromosomeBundle` dataclass:

```
ChromosomeBundle
├── learning:        HyperparameterChromosome  # Chromosomes A + B
├── physiology:      HyperparameterChromosome  # C + D + E
├── sensorimotor:    HyperparameterChromosome  # F + G
├── reward_shaping:  HyperparameterChromosome  # H (optional)
├── architecture:    HyperparameterChromosome  # I (needs Tier 3)
└── repertoire:      HyperparameterChromosome  # J (needs Tier 3)
```

Benefits this gives you for free:

- Per-chromosome mutation rate / scale (the genes you really want to drift fast
  vs. slow live in different chromosomes).
- Linkage groups: all combat genes recombine together via single-point
  crossover but not with learning genes.
- Independent assortment: the existing `crossover_chromosomes` runs once per
  chromosome.
- Cleaner `apply_*`: each chromosome's genes target one component config;
  namespace prefixes become unnecessary.

---

## Recommended rollout order

Go in this order; each step is independent and testable:

1. **Tier 1 / Chromosome A + B** — pure additions to
   `DEFAULT_HYPERPARAMETER_GENES`. Highest impact / lowest risk. Action-prior
   genes (B) are likely the single biggest qualitative win.
2. **Generalize `apply_chromosome_to_learning_config`** to
   `apply_chromosome_to_agent_config` with namespaced gene names. Add
   Chromosomes C–G.
3. **Add `GeneValueType.INTEGER` and `BINARY` (then `CATEGORICAL`)** plus
   matching mutation operators (uniform-int, bit-flip, allele-swap). Unlocks
   Chromosomes I and J.
4. **Introduce `ChromosomeBundle`** so chromosomes mutate / cross independently
   and `IntrinsicEvolutionPolicy` can take per-chromosome rates.
5. **Reward-shaping chromosome (H)** last, behind an opt-in flag, with an
   external fitness metric to prevent Goodharting.
