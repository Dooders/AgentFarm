# AgentFarm Simulation Engine: Architecture, Performance and Scientific Viability Review

Repository: `AgentFarm`, branch `dev`, commit `952c6b14`
Review environment: 4 x Intel Xeon vCPU, 15 GB RAM, no GPU, Linux 6.12, Python 3.12.3, torch 2.15.0 (CPU), tianshou (as pinned in `requirements-lock.txt`)
All measurements in this document were produced during this review on the above machine. Raw numbers are in [`scaling_results.jsonl`](scaling_results.jsonl) and [`spatial_benchmark_verified_rerun.md`](spatial_benchmark_verified_rerun.md) next to this file. The scaling harness is committed as `scripts/engine_review/engine_baseline.py`; the learning-signal and movement checks are encoded as strict-xfail tests in `tests/integration/test_engine_semantics.py` so that fixing a finding forces the corresponding marker to be removed. The findings are tracked as GitHub issues generated from [`issues.yaml`](issues.yaml) by `scripts/engine_review/bootstrap_github_project.py`; the committed throughput baseline is `baseline_952c6b14.json`.

---

## 1. Executive Summary

AgentFarm is an ambitious, broadly scoped platform: a PettingZoo-compatible environment, a component-based agent core, a per-agent DQN stack, an evolvable hyperparameter chromosome, a SQLite metrics pipeline and a family of experiment runners. The surface area is large and much of the engineering (component architecture, deterministic identity service, buffered DB logging, benchmark harness, verified spatial artifacts) is genuinely good. However, the engine as it exists at `952c6b14` is **not currently in a state where simulation results can support scientific claims**, because several core mechanics are silently broken in the default execution path. These are not performance issues; they are correctness failures that invalidate the semantics of the environment.

The five findings that matter most:

1. **Agents never move.** `move_action` in `farm/core/action.py` requires `agent.validation_service` and `agent.update_position`, neither of which exists on `AgentCore`. The `hasattr` guard makes the action fail quietly every time with `"Invalid position"`. Verified: 0 of 28 agents changed position over 200 steps of the development config. Every result produced since the Aug-Oct 2025 component refactors has been a simulation of stationary agents that gather from whatever resources happen to be within hard-coded range of their spawn point.
2. **The RESOURCES observation channel is effectively blind.** `PerceptionComponent._create_world_layers` builds layers for `PerceptionConfig.perception_radius = 5` (11 x 11) while the observation space is built from `ObservationConfig.R = 6` (13 x 13). `WorldLayerHandler.process` in `farm/core/channels.py` interprets the shape mismatch as "this is a world-sized grid" and crops at world coordinates. Verified: 10 non-zero resource cells vs 58 ground-truth cells at step 0, then zero for the rest of the run. The DQN is therefore learning from a 2197-dimensional input in which the single most decision-relevant channel carries no information.
3. **Config is largely decorative for actions.** `gathering_range`, `attack_range`, `share_range`, `max_gather_amount`, `share_amount`, `min_keep_amount`, `max_movement` and `curriculum_phases` are read via `getattr(agent.config, ..., <literal>)` where `agent.config` is an `AgentComponentConfig` that does not define those attributes. YAML values in `farm/config/default.yaml` are ignored; the hard-coded fallbacks (gather 30, attack 20, share 30, max_gather 10) are what actually run. On a 50 x 50 world a gathering range of 30 means agents can gather from most of the map without moving, which is why finding 1 was never noticed.
4. **The RL signal is malformed.** Terminal transitions are never stored (31 deaths observed, 0 transitions with `done=True`), so `reward_death_penalty` is a dead gene and the Q-function never sees termination. Base metabolic consumption happens in `ResourceComponent.on_step_start`, before the reward's pre-snapshot, so the agent's own survival cost is excluded from `_calculate_reward`. Combat and share metrics are never recorded because `hasattr(agent, "metrics_service")` is false. Every action is logged twice to `agent_actions` (66,584 rows for 33,292 agent-steps).
5. **The experiment layer does not deliver what the research questions require.** `ExperimentRunner` has no per-iteration seeding (all iterations replay the same trajectory), applies sweep parameters with flat `setattr` on a nested config (no-op), and looks for `simulation.db` while the engine writes `simulation_{id}.db`, so its analysis step is always skipped. Default `run_simulation.py` runs no chromosome mutation at all, so "evolution" only happens when an `IntrinsicEvolutionPolicy` is explicitly attached. The dedicated research runners (`cohort_runner.py`, `intrinsic_goals_experiment.py`) handle seeds correctly, but no central statistics module exists; bootstrap/t-test/Holm helpers are duplicated across three experiment folders.

Performance is a secondary concern relative to the above, but the profile is clear: **DQN training is 52-61% of wall time**, per-agent networks cost ~17 MB of RSS each (replay buffer of full observation tensors plus ~1.1M parameters plus Adam state), the engine runs at a flat ~3.8 ms per agent-step with learning (1.8 ms without), scales linearly in population with no super-linear terms up to 300 agents, and perception radius dominates everything else (R=10 is 2.5x slower than R=3). SQLite, spatial indexing and the metrics tracker are *not* bottlenecks; several benchmark claims in the README and `benchmarks/` are either unreproducible or measure the wrong thing.

Determinism, by contrast, holds: two runs with the same seed produced identical state over 120 steps, and the committed verified spatial benchmark numbers reproduce within noise.

Bottom line: fix the five correctness items first, then re-run every experiment whose conclusions depended on agent movement, resource perception or terminal rewards. Until then, published devlog findings about foraging, cooperation or intrinsic-goal evolution should be treated as unvalidated.

---

## 2. Architecture & Design Review

### 2.1 Simulation loop and step cycle

Files: `farm/core/simulation.py` (`run_simulation`, `_run_deferred_learning_updates`), `farm/core/environment.py` (`Environment.step`, `Environment.update`).

The loop is simple and readable: for each step, iterate `environment.agent_objects` in insertion order, call `agent.act()`, then `environment.update()`, then run deferred learning updates round-robin. Observations:

- **Duplicated loop body.** `run_simulation` contains two near-identical loops for the resume and fresh-start paths. This is a maintenance hazard: the deferred learning scheduler and the memory-monitor hooks must be kept in sync by hand.
- **Fixed sequential ordering.** Agents always act in insertion order. Earlier agents get first pick at contested resources every step. There is no random permutation and no simultaneous-update mode. For ABM work this is a well known source of ordering artefacts (Comer 2014; Caron-Lormier et al. 2008) and should at minimum be a configurable scheduler.
- **`Environment` is a god class.** It implements the PettingZoo AECEnv API, the batch-mode loop, resource management, metrics, spatial index lifecycle, agent registry, DB handoff and observation-space construction. Both `step()` (AEC) and `update()` (batch) exist and partially overlap. SRP is violated badly enough that it is hard to reason about which path a given test exercises.
- **`init_random_seeds` is only called when `seed=` is passed to `run_simulation`,** which the CLI does not do; seeding actually happens inside `Environment.__init__` from `config.seed`. It works, but two seeding mechanisms with different triggers is confusing and brittle. `run_simulation.py` also re-execs with `PYTHONHASHSEED=0`, which is correct but surprising.

### 2.2 Agent lifecycle, resources and termination

Files: `farm/core/agent/core.py`, `farm/core/agent/components/resource.py`, `farm/core/agent/components/reproduction.py`, `farm/core/environment.py` (`add_agent`, `remove_agent`).

- `AgentCore.step()` orders component hooks as `on_step_start` -> decide -> `_execute_action` -> reward -> `behavior.update`. Because `ResourceComponent.on_step_start` applies base consumption and starvation checks *before* the reward pre-snapshot, the metabolic cost and death-by-starvation are invisible to the reward, and a starving agent calls `core.terminate()` before any transition is recorded. The agent's final experience is lost and `done=True` is never emitted (see Section 3.5).
- `Environment.add_agent` attempts dependency injection with `if hasattr(agent, "validation_service") and agent.validation_service is None:`. `AgentCore` has no such attribute, so the guard never fires. This single line is the root cause of the movement failure.
- `reproduce()` constructs a new `AgentFactory` per call and, with no policy attached, `_derive_child_chromosome` returns `deepcopy(parent_chromosome)`. Children are clones; there is no mutation in the default path (contradicting `docs/design/hyperparameter_chromosome.md`, which still describes mutation in `reproduce`).
- `remove_agent` is correct but relies on the caller to have already written the final state; combined with the above, death bookkeeping is split across three places.

### 2.3 Spatial indexing

Files: `farm/core/spatial/index.py`, `farm/core/spatial/*`.

The design is sound: a cKDTree per entity type, dirty flags, md5 position hashing to detect changes, and alternative quadtree / spatial-hash backends. Measured behaviour (Section 3.4) shows it is not a bottleneck even when movement is forced on. Two design notes: the md5 hash of all positions on every `get_nearby` when dirty is O(n) work that a simple version counter would replace, and the batch-update path is slower than immediate updates in the verified benchmark, so it should not be the default.

### 2.4 Observation channels and perception

Files: `farm/core/observations.py`, `farm/core/channels.py`, `farm/core/agent/components/perception.py`, `farm/core/agent/config/component_configs.py`.

13 channels (SELF_HP, ALLIES_HP, ENEMIES_HP, RESOURCES, OBSTACLES, TERRAIN_COST, VISIBILITY, KNOWN_EMPTY, DAMAGE_HEAT, TRAILS, ALLY_SIGNAL, GOAL, LANDMARKS) over a (2R+1)^2 egocentric window. The channel registry and handler abstraction are well designed. Problems:

- The radius mismatch (`PerceptionConfig.perception_radius = 5` vs `ObservationConfig.R = 6`) described in the Executive Summary. `AgentComponentConfig.from_simulation_config` does not propagate `observation.R` into `PerceptionConfig`, and `Environment._setup_observation_space` defaults to `ObservationConfig()` when the simulation config has none. Two sources of truth, no assertion that they agree.
- DAMAGE_HEAT, TRAILS, ALLY_SIGNAL and GOAL are declared and allocated but never populated by any component (`# TODO` markers in `perception.py`). They cost conv input bandwidth and memory for nothing.
- `perceive_world` + `update_known_empty` + `_build_dense_tensor` together account for ~53 s of the 292 s baseline profile (18% of total, 52% of the non-training agent step). Most of this is Python-level per-cell work in `apply_to_dense` (150,953 calls, 6.7 s tottime).

### 2.5 Action space

File: `farm/core/action.py`.

Eight actions (move, gather, attack, share, defend, pass, reproduce, communicate) in a registry; `move` picks a random direction (so the policy has no directional control: it decides *whether* to move, not *where*), and gather/attack/share auto-target the nearest entity. This makes the effective action space far coarser than the 8 logits suggest; combined with the config fallbacks, "gather" within radius 30 on a 50 x 50 map is nearly always available. The duck-typed `hasattr(agent, "metrics_service")` checks mean combat and sharing are never recorded in the DB.

### 2.6 BaseAgent / DecisionModule / DQN stack

Files: `farm/core/decision/decision.py`, `farm/core/decision/algorithms/tianshou.py`, `farm/core/decision/algorithms/rl_base.py`, `farm/core/agent/behaviors/learning.py`.

- One `AdaptiveQNet` per agent: Conv2d(13->32->64->64) on 13 x 13, flatten to 10,816, Linear 96 -> 48 -> 8. ~1.1M parameters, a private Adam optimizer and a private `PrioritizedReplayBuffer`. There is no parameter sharing option and no vectorised batch inference across agents; each agent runs its own forward pass per step (33,292 `predict_proba` calls in the baseline).
- The custom `PrioritizedReplayBuffer` in `rl_base.py` stores a list of Python dicts each holding full `state` and `next_state` tensors (13 x 13 x 13 float32 = 8.8 KB each, so ~17.6 KB per transition), and samples with `np.random.choice` over the full priority vector: O(n) per sample rather than the O(log n) sum-tree PER was designed around. Tianshou ships a `PrioritizedReplayBuffer` with a segment tree that is not used.
- `DecisionModule.update()` calls `log_agent_action` a second time with module metadata, producing duplicate `agent_actions` rows.
- `train_if_ready` trains every 4th call (`train_freq = 4`); 7,451 `train_on_batch` calls consumed 179 s of the 292 s profile.
- **Algorithm selection has dead and invalid branches.** `DecisionConfig` validates `"ddqn"` as an allowed `algorithm_type` ("alias for dqn with DDQN features") but `DecisionModule._initialize_algorithm` has no `ddqn` branch, so it silently falls through to `FallbackAlgorithm` (random policy). The live `dqn` path is vanilla one-step DQN via Tianshou `DQNPolicy`; Double DQN exists only in the orphaned `BaseDQNModule` (`base_dqn.py`), which `DecisionModule` never uses. `DecisionConfig.tau` (soft target) is unused on the Tianshou path (hard `target_update_freq=100`). `ppo` and `a2c` are trained from the same off-policy replay buffer as DQN, which is not a valid training regime for on-policy methods; `sac`/`ddpg` continuous actors are projected onto discrete logits. README still describes a Stable-Baselines3 backend; `stable_baselines.py` does not exist.
- **Three reward systems disagree.** `AgentCore._calculate_reward` is what the learner sees. `RewardComponent` (`components/reward.py`) computes a different formula (`resource_delta + 0.5*health_delta + 0.1`) into `total_reward`, and `pass_action`/`defend_action` add `ActionRewardConfig` bonuses (0.01 / 0.02) directly to `total_reward`. The `total_reward` column in `agent_states` therefore does not equal the sum of rewards the policy was trained on, so any analysis of "reward over time" from the DB is measuring a different quantity from the learning signal.
- **Two sets of hyperparameter defaults.** `DecisionConfig` defaults are `gamma=0.99`, `memory_size=10000`, `dqn_hidden_size=64`; `farm/config/config.py` `LearningConfig` defaults are `gamma=0.95`, `memory_size=2000`, `dqn_hidden_size=24`, and `AgentComponentConfig.from_simulation_config` deliberately maps the latter over the former. Runs started from a `SimulationConfig` therefore use 0.95/2000/24, which is what the profiling in Section 3 measured. Documentation quoting the `DecisionConfig` numbers is wrong for every real run.
- **Device handling.** Tianshou `_get_default_config` hard-codes `"device": "cpu"` and ignores `SimulationConfig.device`; `DeviceConfig.cpu_threads` defaults to 1 and `torch.set_num_threads(1)` is applied. On this 4-core host the learner was single-threaded by configuration, not by necessity.

### 2.7 Genome / hyperparameter chromosome

Files: `farm/core/hyperparameter_chromosome.py`, `farm/runners/intrinsic_evolution_experiment.py`.

40 real-valued genes: learning rate, gamma, epsilon schedule, `memory_size` in [1, 1e6], batch size, hidden sizes, PER alpha/beta, per-action weights and multipliers, and `reward_*` weights that define the agent's intrinsic objective. The chromosome is well structured and the Baldwinian / Lamarckian / P2-P4 warm-start variants are a thoughtful design. Issues:

- `memory_size` up to 1e6 with 17.6 KB transitions is 17.6 GB per agent. On this machine any lineage that drifts upward in `memory_size` will OOM the host; the gene bounds are not tied to the observation footprint.
- `reward_death_penalty` is unreachable (Section 3.5). `reward_action_bonus` is added for every non-pass action, including failed moves, so the gene rewards attempting rather than achieving.
- Mutation and crossover only run when an `IntrinsicEvolutionPolicy` is attached. The default simulation has *no* evolutionary operator, yet `MetricsTracker.genetic_diversity` (lineage count / N) is reported as though evolution were happening.

### 2.8 Memory systems

Redis-backed agent memory (`farm/memory/`) is optional and well isolated. It is not in the default path and I did not profile it; the `redis_memory` benchmark spec exists. Nothing here blocks the findings above.

### 2.9 Metrics, logging and analysis

Files: `farm/core/metrics_tracker.py`, `farm/database/*`, `farm/analysis/*`.

SQLAlchemy logging with an in-memory option is measurably cheap at the scales tested (Section 3.4). `MetricsTracker` is O(n) per step. Problems are mostly semantic rather than performance: duplicate action rows, missing combat/share rows, and `genetic_diversity` reporting lineage counts as diversity. Two structural notes from reading `farm/database/data_logging.py`: `DataLogger.log_step` bulk-inserts every alive agent's state, every resource and the step row inside an immediate `_execute_in_transaction` call, so the `buffer_size=1000` / `commit_interval=30s` design only applies to actions and health incidents; per-step state persistence is synchronous and O(agents + resources) every step. It did not show up in the profile here because the development config is small and in-memory, but it will scale linearly with population on disk. The legacy `DataCollector._estimate_agent_territory` is O(n^2) pairwise distance and should not be used for large populations. `simulation_steps` deliberately omits combat counters ("derive from actions"), which is impossible while combat actions are not recorded.

### 2.10 Experiment runner, sweeps, multi-seed

Files: `farm/runners/experiment_runner.py`, `farm/runners/cohort_runner.py`, `farm/runners/intrinsic_goals_experiment.py`, `farm/runners/parallel_experiment_runner.py`.

`ExperimentRunner` is the generic entry point and is the weakest component: no seed increment per iteration, flat `setattr(config, key, value)` on a nested `SimulationConfig` (silently creates an unused attribute), flat `getattr(base_config, "use_in_memory_db")` (always the default), and a `simulation.db` vs `simulation_{id}.db` path mismatch that skips analysis. `cohort_runner.py` and `intrinsic_goals_experiment.py` are considerably better: explicit seed lists, `seed = base + index`, paired arms. `parallel_experiment_runner.py` also passes seeds. The same flat-config anti-pattern appears in `benchmarks/implementations/memory_db_benchmark.py`.

### 2.11 Cross-cutting design debt: duck typing by `getattr`/`hasattr`

This is the single theme behind findings 1-4. The codebase routinely writes `getattr(obj, "attr", default)` or `if hasattr(obj, "x"):` to tolerate multiple object shapes. Every one of the correctness failures above is a case where the "tolerant" branch silently became the only branch after a refactor and no test caught it. This pattern defeats both type checking and fail-fast behaviour and should be replaced with explicit protocols (`typing.Protocol`) and constructor-time validation.

---

## 3. Profiling & Performance Results

### 3.1 Baseline: development config, 1000 steps, cProfile

Command: `python run_simulation.py --environment development --steps 1000 --profile` (50 x 50 world, 10/10/10 agents, `max_population 50`, in-memory DB, seed 1234567890). Full stats: `simulations/profile_stats.txt`.

| Metric | Value |
|---|---|
| Wall time | 293.0 s (295.97 s incl. interpreter; 283.2 s user, 8.7 s sys) |
| Steps | 1000 |
| Agent-steps | 33,379 |
| ms per step | 293 |
| ms per agent-step (profiled) | 8.8 |
| Peak RSS | 4,050 MB |
| Function calls | 94.98M (92.46M primitive) |

Cumulative time breakdown (profiled; cProfile adds overhead to Python-heavy paths):

| Component | Cum. time | Share of total | Key function |
|---|---|---|---|
| Deferred DQN training | 179.1 s | 61% | `simulation._run_deferred_learning_updates` -> `tianshou.train_on_batch` (7,451 calls, 24 ms each) |
| of which backward pass | 62.8 s | 21% | `torch._C._EngineBase.run_backward` |
| of which Adam step | 45.8 s | 16% | `adam._single_tensor_adam` (8.1 s tottime; 17.9 s in `Tensor.sqrt`) |
| Agent acting | 102.5 s | 35% | `AgentCore.step` |
| of which observation build | 53.6 s | 18% | `perception.get_observation_tensor` -> `observations.perceive_world` (46.4 s) |
| of which policy inference | 23.5 s | 8% | `tianshou.predict_proba` -> `_policy_q_values` |
| of which action execution | 41.4 s | 14% | `AgentCore._execute_action` (includes decision + logging) |
| Conv2d forward (train + inference) | 55.5 s | 19% | `torch.conv2d`, 144,936 calls |

Observations:

- 7,451 train calls / 33,380 `train_if_ready` calls = 22%, consistent with `train_frequency = 4` plus warm-up.
- The Adam step spends 17.9 s in `sqrt` alone because tianshou's `_single_tensor_adam` path is used (foreach/fused paths are not enabled on CPU by default). This is 6% of total runtime.
- Observation construction is dominated by Python per-cell loops (`apply_to_dense`: 150,953 calls, 6.7 s tottime; `channels.process`: 196,749 calls).
- This run was partly contended by a concurrent process (load average 4 on 4 cores) during roughly the first third; the scaling runs below were executed sequentially and are the better absolute numbers.

### 3.2 Benchmarks shipped in the repository

`python -m benchmarks.run_benchmarks --list` returns `redis_memory`, `pragma_profile`, `observation_flow`, `memory_db` and the profiler-wrapper experiments. Results of the specs I ran:

| Spec | Result | Assessment |
|---|---|---|
| `observation_baseline.yaml` (small) | mean step 34.2 ms, p95 34.8 ms, total 0.68 s; spatial query 35 ms total, bilinear 11 ms | Works. Confirms spatial queries are cheap relative to tensor build. |
| `memory_db_baseline.yaml` (small) | disk 7.50 s, in-memory 5.33 s, in-memory+persist 5.22 s; reported "1.41x" | **Invalid.** The benchmark sets `self.config.use_in_memory_db`, `system_agents`, `width` as flat attributes on a nested `SimulationConfig`; they are ignored, so all three arms run the same disk-backed configuration. The 1.41x is JIT/warm-up noise. The README's "33.6% faster with in-memory DB" claim has no supporting measurement. |
| `spatial_comprehensive_baseline.yaml` | Fails: references an experiment name not registered in `benchmarks/run_benchmarks.py` | Spec rot. |
| `comprehensive_spatial_benchmark.py --verified` | Reproduces the committed `benchmarks/results/SPATIAL_BENCHMARK_VERIFIED.md` within noise (see [`spatial_benchmark_verified_rerun.md`](spatial_benchmark_verified_rerun.md)) | Works and is honest. It shows AgentFarm's KD-tree wrapper builds 7x slower than raw SciPy (2.80 ms vs 0.38 ms at 2000 entities) and that batch updates are slower than immediate updates. The quadtree is 60x slower to build than SciPy. |

### 3.3 Scaling: population

All runs: 50 x 50 world (100 x 100 for 300-agent runs), 200 steps (100 for 300 agents), in-memory DB, R=6, learning on unless noted. `ms_per_agent_step` = wall / sum of live agents per step. Peak RSS via `resource.getrusage(RUSAGE_CHILDREN)`.

| Label | Start pop | Mean pop | Wall (s) | ms/step | ms/agent-step | Peak RSS (MB) | KD rebuilds |
|---|---|---|---|---|---|---|---|
| pop_15 | 15 | 15.0 | 14.0 | 56.7 | 3.78 | 1,419 | 1 |
| pop_30 | 30 | 30.0 | 25.8 | 114.9 | 3.83 | 1,787 | 1 |
| pop_60 | 60 | 52.8 | 43.1 | 200.4 | 3.79 | 2,567 | 33 |
| pop_120 | 120 | 68.0 | 53.2 | 248.6 | 3.66 | 4,242 | 87 |
| pop_30_nolearn | 30 | 30.0 | 13.8 | 55.1 | 1.84 | 1,287 | 1 |
| pop_120_nolearn | 120 | 67.2 | 29.8 | 132.1 | 1.97 | 2,138 | 94 |
| pop_300_nolearn (100x100, 100 steps) | 300 | 184.4 | 42.2 | 368.6 | 2.00 | 3,652 | 173 |

Conclusions:

- **Per-agent cost is flat** at ~3.8 ms with learning and ~1.9 ms without, from 15 to 300 agents. There is no super-linear term in the engine at these scales; the simulation is embarrassingly linear in population.
- **Learning roughly doubles cost.** The 1.8-2.0 ms non-learning floor is observation construction + action execution + logging, which is pure Python and is the real long-term ceiling.
- **Memory is the binding constraint, not CPU.** With learning: ~26.9 MB of peak RSS per starting agent (slope from pop_15 to pop_120), of which ~17 MB is attributable to the learning stack (pop_30 vs pop_30_nolearn: 500 MB / 30 agents). 120 agents already peaks at 4.2 GB; 500 agents would exceed this 15 GB host. Non-learning memory is ~9.5 MB/agent (observation buffers, per-agent torch tensors).
- **Populations collapse quickly.** `max_population` 120 fell to 30 within 200 steps with no movement; the 300-agent runs lost 80% of agents in 100 steps. This is a direct consequence of findings 1-3: agents cannot forage beyond spawn range.

### 3.4 Scaling: perception radius, DB backend, spatial index

| Label | R | Learning | Wall (s) | ms/agent-step | Peak RSS (MB) |
|---|---|---|---|---|---|
| R3 | 3 (7x7) | on | 17.7 | 2.51 | 1,293 |
| pop_30 | 6 (13x13) | on | 25.8 | 3.83 | 1,787 |
| R10 | 10 (21x21) | on | 40.8 | 6.30 | 2,895 |
| R3_nolearn | 3 | off | 12.2 | 1.59 | 1,118 |
| pop_30_nolearn | 6 | off | 13.8 | 1.84 | 1,287 |
| R10_nolearn | 10 | off | 16.3 | 2.20 | 1,678 |

Perception radius is the dominant tunable: R=10 costs 2.5x R=3 with learning (conv input grows 9x, from 49 to 441 cells per channel; the flatten layer scales accordingly and so does Adam state). Memory follows the same curve.

| Label | Variant | Wall (s) | ms/agent-step | KD rebuilds |
|---|---|---|---|---|
| pop_30 | in-memory DB | 25.80 | 3.83 | 1 |
| pop_30_diskdb | disk SQLite | 25.73 | 3.83 | 1 |
| pop_30_nolearn | move broken (default) | 13.79 | 1.84 | 1 |
| pop_30_fixmove_nolearn | move patched to work | 13.99 | 1.85 | 1,751 |
| pop_120_nolearn | move broken | 29.81 | 1.97 | 94 |
| pop_120_fixmove_nolearn | move patched | 32.15 | 2.08 | 4,308 |
| pop_300_nolearn | move broken | 42.24 | 2.00 | 173 |
| pop_300_fixmove_nolearn | move patched | 43.54 | 2.10 | 6,312 |

- **Disk vs in-memory SQLite: no measurable difference** at this logging volume. The buffered writer already hides disk latency.
- **Spatial index is not a bottleneck.** Forcing movement on (monkey-patching `validation_service`/`update_position` for the test) increased KD-tree rebuilds from 1 to 6,312 at 300 agents and cost only 5% more wall time (2.00 -> 2.10 ms/agent-step). The per-move dirty marking and md5 hashing are cheap relative to observation construction.

### 3.5 Learning-signal verification

Instrumenting `TianshouWrapper.store_experience` over a 200-step development run (now encoded in `tests/integration/test_engine_semantics.py`):

- Deaths observed via `remove_agent`: 31. Transitions stored with `done=True`: **0**.
- Every stored transition has `done=False`; `behavior.update(..., done=not self.alive, ...)` is only reached by agents that survived the step, and starvation terminates inside `on_step_start` before the step body runs.
- Movement: 0 of 28 agents changed position in 200 steps; every `move` result was `success=False, error="Invalid position..."`, `validation_service_available=False`.
- RESOURCES channel: 10 non-zero cells at step 0 vs 58 ground-truth resources within R=6 of the sampled agent; 0 non-zero cells at all later steps.
- `agent_actions` rows: 66,584 for 33,292 agent-steps (exactly 2x).

### 3.6 Determinism

Two 120-step development runs (15 agents) in separate processes with the same seed produced identical agent positions, resource levels, health and action sequences at every step. The `PYTHONHASHSEED=0` re-exec, `cpu_thread_bootstrap.pin_cpu_math_threads`, `Environment.__init__` seeding and the deterministic `Identity` service work. Residual risks not exercised by this test: `torch.backends.cudnn.deterministic` is commented out in `simulation.py` (irrelevant on CPU, relevant on GPU); the observation `"overwrite"` point-reduction mode is documented as order-dependent in `observations.py`; unseeded runs fall back to `uuid4` IDs and a `time.time()`-based DB path; and the `parallel_experiment_runner` path was not tested for cross-process determinism.

### 3.7 Hot-spot summary

In priority order of achievable savings on the default path:

1. **Per-agent DQN training (52-61%).** Largest lever: parameter sharing or a shared replay buffer with agent-ID conditioning, vectorised batch inference across agents, `torch.optim.Adam(foreach=True)` or `fused=True`, a smaller conv stack or a stride-2 first layer, and training every N steps per *population* rather than per agent.
2. **Observation construction (18%, ~52% of the non-learning floor).** Vectorise `perceive_world`/`apply_to_dense` with NumPy slicing; drop the four never-populated channels (31% of the tensor); cache the static OBSTACLES/TERRAIN layers.
3. **Replay buffer memory (~17 MB/agent).** Store uint8 or float16 observations, store `next_state` by index rather than copy, and switch to tianshou's segment-tree PER.
4. **Adam `sqrt` (6%).** Use the foreach/fused implementation.
5. **Duplicate logging (~2-3%).** Remove the second `log_agent_action` in `DecisionModule.update`.

Spatial indexing, SQLite and `MetricsTracker` do not merit optimisation effort now.

---

## 4. Scientific Viability Assessment

### 4.1 Reproducibility and provenance

- **Seeds:** Deterministic given a seed (Section 3.6). `ExperimentRunner` does not vary seeds across iterations, so its "N iterations" are N copies of one trajectory; `cohort_runner` and `intrinsic_goals_experiment` do this correctly.
- **Config versioning:** `simulations/config.json` is written alongside each DB and the verified spatial benchmark records git revision, platform and `PYTHONHASHSEED`. Simulation DBs do not record git revision or package versions; add both.
- **Code-level provenance is undermined by the silent config fallbacks.** A recorded YAML with `gathering_range: 5` did not produce a run with gathering range 5. Any provenance record from the current code is unreliable for action parameters.

### 4.2 Validity threats specific to the current engine

| Threat | Source | Effect on inference |
|---|---|---|
| No movement | `move_action` / `add_agent` injection | Spatial dynamics, foraging, territory, migration and encounter-rate hypotheses are untestable. Population collapse is an artefact of spawn position. |
| Blind RESOURCES channel | R mismatch | Any claim that agents "learn to find resources" is false: they cannot perceive them. Learning curves reflect reward noise plus the SELF_HP/ALLIES/ENEMIES channels. |
| Missing terminal transitions | step ordering | Q-targets bootstrap through death; `reward_death_penalty` gene has zero selection gradient; survival-weight genes are confounded with episode length. |
| Reward excludes metabolic cost | `on_step_start` ordering | The agent's objective does not include the cost of living; "survival" is only rewarded via the constant `reward_survival_weight`. |
| Ordering bias | insertion-order scheduler | Systematic advantage to early-spawned agents and their lineages; confounds any heritability analysis. |
| Clone reproduction by default | `_derive_child_chromosome` | "Genetic diversity" metric is lineage count; no variation means no evolution. Reports of trait drift in default runs are impossible by construction. |
| Missing combat/share metrics | `metrics_service` absent | Cooperation and conflict analyses have no data. |
| Duplicate action rows | `DecisionModule.update` | Any action-frequency statistic is doubled; action-distribution comparisons are still valid ratios but counts are wrong. |

### 4.3 Statistical practice

- **Multi-seed variance:** The paired-arm design in `intrinsic_goals_experiment.py` (same seed across arms) is good practice, and the recorded experiments did use reasonable replication: the veil-ceiling study ran 13 conditions x 2 inheritance modes x 30 seed-matched pairs (780 runs), the intrinsic-goals study 20 paired seeds, the consensus study 250 x 400 x 8, and the 2026-05-12 "seed-sweep reality check" devlog explicitly found that single-seed effects "mostly die" at 6 seeds. The defaults, however, are low (`IntrinsicGoalsExperimentConfig.num_replicates = 1`, cohort example `seeds=[1, 2, 3]`), so a new experiment built from the templates starts without replication. Report per-seed outcomes, not just means.
- **Multiple comparisons and effect sizes:** rigour is concentrated in a few folders rather than shared. `farm/experiments/consensus/contrasts.py` has paired deltas, t-CIs, Wilcoxon, Holm adjustment and paired Cohen's d; `veil_ceiling/analysis.py` has percentile bootstrap CIs (2000 reps); the intrinsic-goals runner has `ttest_rel` and Cohen's dz; `analysis/advantage` has `ttest_ind` + Cohen's d; `recombination_stats.py` has its own bootstrap and Welch tests; `CohortRunner` reports mean/std only; `research/util.calculate_statistics` uses a 1.96*SE normal approximation. There is no shared `farm/stats` module, no Cliff's delta or other robust effect size, and no project-wide convention for pre-registering primary endpoints.
- **Controls:** Random-policy and no-learning baselines are not run automatically by any runner. Given that the RESOURCES channel is blind, the current learned policies may be statistically indistinguishable from random; this must be checked before any learning claim.
- **Burn-in and autocorrelation:** Step-level metrics are reported without burn-in exclusion or autocorrelation-aware CIs; consecutive steps are heavily correlated.

### 4.4 Alignment with stated research aims

The devlog and `docs/research/experiments-catalog.md` frame AgentFarm around emergent cooperation, intrinsic-goal evolution and population dynamics under resource pressure. The architecture (chromosome with reward genes, Baldwinian/Lamarckian inheritance, paired-arm runners) is well matched to those aims. The current implementation is not: the three mechanisms those questions depend on (movement, resource perception, death-as-terminal) are inoperative. The experimental *scaffolding* is viable; the *engine state* is not.

The devlog itself corroborates this. The 2026-05-16 entry "Is the DQN learning?" fixed four bugs (a global train throttle, epsilon never passed to `set_eps`, YAML knobs dropped, a dead hidden-size parameter) and still found that late-vs-early decision quality "only barely improves", attributing the remainder to environment signal-to-noise. Findings 1-4 of this review are a concrete explanation for that result: a learner that cannot see resources, cannot move, and never observes a terminal state has almost nothing to learn from. The 2026-06-04 and 2026-07 inheritance-ladder entries report warm-start (Lamarckian) policies *losing* to cold starts across 90-run A/Bs with "zero robust positives"; inherited Q-functions trained on this malformed signal would be expected to transfer noise, so those negative results should also be re-run after the fixes rather than taken as evidence against inheritance per se.

### 4.5 Comparison to field practice

- **ABM (Mesa, NetLogo, Repast):** random or staged activation schedulers are standard; AgentFarm has none. Mesa's `RandomActivation` is the minimum bar.
- **Evolutionary computation:** mutation and crossover operators are always on in the default evolutionary loop; diversity metrics are computed on genotype distance, not lineage count.
- **MARL (PettingZoo/RLlib/MARLlib):** parameter sharing is the default for homogeneous agents; per-agent independent DQN with private buffers is the most expensive and least sample-efficient configuration and is normally reserved as a baseline. Terminal-state handling and `done` flags are validated by environment API tests (`pettingzoo.test.api_test`); running it here would likely surface the `done` issue.
- **Benchmarking:** committed, regenerable benchmark artefacts with git revision (the verified spatial benchmark) are above-average practice. The `memory_db` benchmark and README claims are below it.

### 4.6 Documentation drift

`docs/design/hyperparameter_chromosome.md` describes mutation in `reproduce` (not true by default). `docs/concepts/agents-and-decisions.md` is an experimental proposal mislabelled as a concept doc. `farm/config/default.yaml` carries per-module DQN knobs (`move_*`, `attack_*`, ...) and `curriculum_phases` that nothing reads. README describes a Stable-Baselines3/TD3 backend that does not exist and performance claims that were never measured. `DecisionConfig` docstrings advertise `ddqn` as a Double-DQN alias that actually yields a random policy. `CurriculumConfig` defaults to a three-phase curriculum while `DecisionConfig.curriculum_phases` defaults to `None`, and nothing wires one to the other. `research_models.py` docstrings reference a `LearningExperience` table that does not exist.

---

## 5. Prioritized Recommendations

### 5.1 High impact, low effort (do before any further experiments)

1. **Fix movement.** Give `AgentCore` a real `validation_service` dependency and an `update_position` method (or make `move_action` call `agent.movement.move_to(...)` on the `MovementComponent`). Replace the `hasattr` guard in `Environment.add_agent` with explicit injection. Add an integration test that asserts at least one agent position changes in a 50-step real `Environment` run.
2. **Unify perception radius.** Derive `PerceptionConfig.perception_radius` from `ObservationConfig.R` in `AgentComponentConfig.from_simulation_config`, and assert equality in `WorldLayerHandler.process` instead of falling back to world-grid semantics. Add a test asserting the RESOURCES channel is non-zero when a resource is within R.
3. **Plumb action parameters.** Add `gathering_range`, `attack_range`, `share_range`, `max_gather_amount`, `share_amount`, `min_keep_amount`, `max_movement` to `AgentComponentConfig` (or a new `ActionConfig`), populate from YAML, and delete the `getattr(..., literal)` fallbacks so misconfiguration raises.
4. **Store terminal transitions.** Move base consumption and starvation checks after the reward pre-snapshot (or snapshot before `on_step_start`), and ensure a `done=True` transition is pushed in `terminate()` for any agent that dies mid-step. Include metabolic cost in `_calculate_reward`.
5. **Remove the duplicate `log_agent_action`** in `DecisionModule.update`.
6. **Inject `metrics_service`** into agents so combat and share metrics are recorded, or remove the dead code paths.
7. **Fix `ExperimentRunner`:** per-iteration seed (`base_seed + i`), nested config updates (use the existing `SimulationConfig` update helpers), correct DB filename.
8. **Fix or delete the `memory_db` benchmark** and remove the README "33.6%" claim until a real measurement exists. Register or delete `spatial_comprehensive_baseline.yaml`.
9. **Add random-policy and no-learning control arms** to `cohort_runner` and `intrinsic_goals_experiment` by default.
10. **Tie `memory_size` gene bounds to the observation footprint** (e.g. cap at 20k for R=6) to prevent OOM during evolution.
11. **Make `ddqn` raise or route to a real Double-DQN implementation**; refuse `ppo`/`a2c` on the replay-buffer path (or implement a proper on-policy collector) rather than silently training them off-policy.
12. **Collapse the three reward paths into one**: delete `RewardComponent`'s parallel formula and the `total_reward +=` side effects in `pass_action`/`defend_action`, and make `agent_states.total_reward` the cumulative sum of the rewards actually passed to `behavior.update`.
13. **Pick one set of DQN defaults.** Either delete `LearningConfig.gamma/memory_size/dqn_hidden_size` or delete the `DecisionConfig` defaults they override; document the effective values (currently 0.95 / 2000 / 24).
14. **Respect `SimulationConfig.device` and `cpu_threads`** in the Tianshou wrapper; on multi-core CPU hosts set `cpu_threads` to the physical core count by default.

### 5.2 Medium-term (next iteration of the engine)

1. **Replace `getattr`/`hasattr` duck typing with `typing.Protocol` interfaces** for agent services (`ValidationService`, `MetricsService`, `SpatialService`) and validate at construction. Run mypy in CI.
2. **Configurable scheduler**: random permutation per step (default), staged, and insertion-order (for backward-compat tests only).
3. **Vectorise observation construction**: NumPy slice-based layer extraction, pre-allocated per-agent buffers, drop unpopulated channels until their producers exist.
4. **Learning stack efficiency**: `Adam(foreach=True)`; optional parameter sharing across agents of the same type with agent-ID embedding; batched inference (stack all agents' observations, one forward pass per step); switch to tianshou's segment-tree PER; store observations as uint8/float16 with next-state by index.
5. **Central `farm/stats` module**: bootstrap CI, paired/Welch tests, Holm/BH correction, Cohen's d / Cliff's delta, autocorrelation-aware step-level CIs. Make all three experiment folders import from it.
6. **Provenance in the DB**: git revision, dirty flag, package versions, resolved effective config (post-fallback), hardware.
7. **PettingZoo API test** (`pettingzoo.test.api_test`) in CI against `Environment` to catch `done`/observation-space regressions.
8. **Split `Environment`**: separate the AEC adapter from the batch simulation core; move resource and metrics lifecycle into their own services.
9. **Default-on mutation** in `reproduce` with a configurable rate, and a genotype-distance diversity metric.
10. **Make per-step state persistence actually buffered.** `DataLogger.log_step` should append to `_step_buffer` and flush on the same size/interval policy as actions; add a `state_log_interval` so large runs can persist agent/resource states every k steps instead of every step. Retire `DataCollector._estimate_agent_territory` or replace it with a spatial-index query.

### 5.3 Longer-term

1. **Scale path**: with ~2 ms/agent-step of pure-Python floor, 1000-agent runs of 10k steps are ~6 hours single-core. Options: multiprocessing over agents per step (observations are independent given a frozen world snapshot), or moving perception and action resolution to a NumPy/JAX vectorised world representation.
2. **Memory path**: shared replay and shared networks are prerequisites for populations above ~400 on commodity hardware.
3. **Pre-registration and experiment manifests**: declare primary endpoint, seeds, arms and analysis before running; have the runner write the manifest into the results directory.
4. **Benchmark CI gate**: run the verified spatial benchmark and a 200-step engine benchmark on every PR and fail on >10% regression in ms/agent-step or peak RSS.
5. **Documentation hygiene**: delete dead config keys, correct `hyperparameter_chromosome.md`, move `agents-and-decisions.md` to `docs/design/proposals/`.

---

## 6. Open Questions

1. **Which published devlog findings depend on movement or resource perception?** Every result since the component refactor (Aug-Oct 2025) needs to be classified as "unaffected", "needs rerun" or "retracted". The experiments catalog should carry this status.
2. **Is per-agent independent DQN a deliberate research choice or an accident of history?** If the research question is about individual learning and heritable priors, independent networks are defensible but the sample-efficiency cost should be stated. If it is about population-level emergence, parameter sharing would be both faster and closer to field practice.
3. **What is the intended semantics of `move`?** Random-direction movement makes the policy unable to steer. Is the design to add directional actions (8 or 4 move actions) or to keep a single "explore" primitive?
4. **Should metabolic cost be part of the agent's reward?** The current (accidental) exclusion may or may not match the intended intrinsic-goal framing; it should be an explicit decision in `RewardConfig`.
5. **What are the target scales?** Recommendations differ sharply for "50 agents, 10k steps, 20 seeds" (fix correctness, modest optimisation) versus "1000 agents, 100k steps" (requires shared networks, vectorised perception and multiprocessing).
6. **Is Lamarckian inheritance (copying trained weights) intended to be a treatment or a control?** Its interaction with the missing terminal transitions means inherited Q-functions have never seen a death; this should be evaluated once finding 4 is fixed.
7. **Why does `genetic_diversity` use lineage count?** If the intent is to track lineages, rename it; if it is to track diversity, compute genotype distance.
8. **What is the ownership model for the benchmarks directory?** Several specs and the README have drifted from the code; a single owner and a CI gate would prevent recurrence.

---

## Appendix A: Evidence index

| Claim | Evidence |
|---|---|
| Movement never succeeds | `farm/core/action.py` `move_action` lines ~953-990; `farm/core/environment.py` `add_agent` `hasattr(agent, "validation_service")` guard; measured 0/28 moved, all `validation_service_available=False` (`tests/integration/test_engine_semantics.py::test_agents_move`) |
| RESOURCES channel blind | `farm/core/agent/config/component_configs.py` `PerceptionConfig.perception_radius=5`; `farm/core/environment.py` `_setup_observation_space` -> `ObservationConfig()` R=6; `farm/core/channels.py` `WorldLayerHandler.process` shape branch; measured 10 vs 58 non-zero cells, then 0 |
| Config fallbacks | `farm/core/action.py` `getattr(agent.config, "gathering_range", 30)` and siblings; `AgentComponentConfig` lacks these fields |
| No terminal transitions | `farm/core/agent/components/resource.py` `on_step_start`; `farm/core/agent/core.py` `step()` ordering; 31 deaths / 0 `done=True` |
| Duplicate action logging | `farm/core/decision/decision.py` `update()`; 66,584 rows / 33,292 agent-steps |
| No default mutation | `farm/core/agent/core.py` `_derive_child_chromosome` -> `deepcopy` |
| ExperimentRunner defects | `farm/runners/experiment_runner.py`: no seed increment, flat `setattr`, `simulation.db` path |
| Profile numbers | `python run_simulation.py --environment development --steps 1000 --profile` -> `simulations/profile_stats.txt` (top-50 cumulative table reproduced in Section 3.1) |
| Scaling numbers | [`scaling_results.jsonl`](scaling_results.jsonl) (15 configurations); regenerate with `scripts/engine_review/engine_baseline.py` |
| Benchmark results | `python -m benchmarks.run_benchmarks` with the `observation_baseline` / `memory_db_baseline` specs; [`spatial_benchmark_verified_rerun.md`](spatial_benchmark_verified_rerun.md); `benchmarks/implementations/memory_db_benchmark.py` flat-config writes |
| Determinism | Two same-seed subprocess runs, identical per-step SHA-256 digests over 120 steps (`tests/integration/test_engine_semantics.py::test_same_seed_is_deterministic`) |
| `ddqn` falls to fallback | `farm/core/decision/config.py` allows `"ddqn"`; `farm/core/decision/decision.py` `_initialize_algorithm` branches only on `ppo`/`sac`/`dqn`/`a2c`/`ddpg`/`fallback` |
| Dual defaults | `farm/config/config.py` `LearningConfig` (0.95 / 2000 / 24) mapped over `DecisionConfig` in `component_configs.py` `from_simulation_config` |
| Three reward paths | `farm/core/agent/core.py` `_calculate_reward`; `farm/core/agent/components/reward.py`; `pass_action`/`defend_action` in `action.py` |
| Synchronous per-step persistence | `farm/database/data_logging.py` `log_step` calls `_execute_in_transaction` directly; `_step_buffer` size check after it is unreachable for that data |
| Devlog corroboration | `docs/research/devlog/` entries 2026-05-12 (seed sweep), 2026-05-16 (is the DQN learning), 2026-06-04 and 2026-07-08/09 (inheritance ladder) |
