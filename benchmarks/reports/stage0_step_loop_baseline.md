# Stage 0 — Step-loop baseline (30×30 grid / 30 agents)

This report establishes a numerical baseline for the simulation step loop on the
small grid that "feels slow" interactively, so any future optimization (Python
refactor, Numba/Cython, Rust extension, JAX rewrite) can be measured against
it.

Reproduce with:

```bash
source venv/bin/activate
PYTHONHASHSEED=0 python -m scripts.profile_step_loop --steps 100 --warmup-steps 5
PYTHONHASHSEED=0 python -m scripts.profile_step_loop --steps 100 --warmup-steps 5 --no-train \
    --out simulations/profile_step_loop_notrain.prof
snakeviz simulations/profile_step_loop.prof
```

Hardware: same VM the agent runs on (Linux, Python 3.12, CPU only — no CUDA in
this environment). Single in-memory SQLite database, default development
profile, seed `1234567890`, 5 warm-up steps excluded from the profile.

---

## Wall-clock summary

| Scenario | wall (s) | steps/sec | ms / step | alive at end |
|---|---:|---:|---:|---:|
| **with training** (`training_frequency=4`, default) | 37.24 | **2.69** | **372** | 53 |
| **no training** (`should_train()` patched to `False`) | 14.997 | **6.67** | **150** | 55 |

Both runs start with 30 agents on a 30×30 grid; population grows to ~50 by step
100 because reproduction is enabled by default.

**The training path more than doubles wall-clock per step.** Just from this
ratio, **~60 % of the wall on the small-grid default run is RL training**, not
anything that a language port could speed up — Tianshou DQN already lives in
PyTorch C++/CUDA.

---

## Top-20 by cumulative time — with training

```
rank  ncalls    tottime   %tt    cumtime    percall  function
   1    4119     0.003   0.0%    35.039   0.008507  farm/core/agent/core.py:478(act)
   2    4118     0.073   0.2%    35.034   0.008508  farm/core/agent/core.py:429(step)
   3    4118     0.050   0.1%    28.407   0.006898  farm/core/agent/core.py:537(_execute_action)
   4       2     0.004   0.0%    27.754  13.876760  farm/core/simulation.py:275(run_simulation)
   5    4118     0.008   0.0%    21.354   0.005185  farm/core/agent/behaviors/learning.py:129(update)
   6    4118     0.048   0.1%    21.298   0.005172  farm/core/decision/decision.py:875(update)
   7     690     0.023   0.1%    20.923   0.030323  farm/core/decision/algorithms/tianshou.py:1262(train)
   8     690     0.053   0.1%    20.898   0.030287  farm/core/decision/algorithms/tianshou.py:1104(train_on_batch)
   9     690     0.050   0.1%    17.876   0.025908  tianshou/policy/modelfree/dqn.py:167(learn)
  10     690     0.017   0.0%    10.813   0.015671  torch/optim/optimizer.py:512(wrapper)
  ... (Adam optimizer + backward stack continues)
  16    8236     0.023   0.1%     9.717   0.001180  farm/core/agent/core.py:492(_create_observation)
  17    8104     0.069   0.2%     9.685   0.001195  farm/core/agent/components/perception.py:374(get_observation_tensor)
  18    8104     0.282   0.8%     7.828   0.000966  farm/core/observations.py:1517(perceive_world)
```

## Top-20 by self (`tottime`) — with training

```
rank  ncalls   tottime   %tt    function
   1     690    4.408  11.8%   <method 'run_backward' of 'torch._C._EngineBase'>
   2    8280    3.879  10.4%   <method 'sqrt' of 'torch._C.TensorBase'>
   3     690    2.793   7.5%   torch/optim/adam.py:347(_single_tensor_adam)
   4    4140    2.206   5.9%   <built-in method torch._C._nn.linear>
   5    4299    2.196   5.9%   <built-in method torch.conv2d>
   6     636    1.853   5.0%   <method 'uniform_' of 'torch._C.TensorBase'>
   7    8280    1.108   3.0%   <method 'lerp_' of 'torch._C.TensorBase'>
   8   23642    1.068   2.9%   farm/core/observations.py:144(apply_to_dense)
   9   29003    0.975   2.6%   farm/utils/spatial.py:13(bilinear_distribute_value)
  10    8280    0.685   1.8%   <method 'mul_' of 'torch._C.TensorBase'>
  11    8280    0.674   1.8%   <method 'addcdiv_' of 'torch._C.TensorBase'>
  12   17954    0.626   1.7%   <built-in method torch.zeros_like>
  13    8280    0.623   1.7%   <method 'addcmul_' of 'torch._C.TensorBase'>
  14     636    0.514   1.4%   <method 'clone' of 'torch._C.TensorBase'>
  15   32416    0.513   1.4%   farm/core/observations.py:1112(_store_sparse_grid)
  16   74937    0.449   1.2%   <built-in method torch.tensor>
  17   22382    0.437   1.2%   farm/core/spatial/index.py:957(get_nearby)
  18   16208    0.314   0.8%   farm/core/observations.py:1212(_build_dense_tensor)
  19    8104    0.282   0.8%   farm/core/observations.py:1517(perceive_world)
  20   64832    0.265   0.7%   <method 'sum' of 'torch._C.TensorBase'>
```

**Read of with-train profile**

- Lines 1–7 (Adam + backward + linear/conv2d/lerp_) sum to **~46 % of self time**
  spent inside torch C++ for the **DQN training step** (`tianshou.train` →
  `_single_tensor_adam`). 690 train calls in 100 steps ≈ 6.9 trains/step
  (`training_frequency=4`, ~28 alive learners on average).
- The next cluster (`apply_to_dense`, `bilinear_distribute_value`,
  `_store_sparse_grid`, `_build_dense_tensor`, `get_nearby`, `perceive_world`)
  is **pure Python in `farm.core.observations` / `farm.utils.spatial`** and is
  the perception system. Together that's **~9 % of self time**.
- DB I/O, spatial index updates, environment update, and policy forward all
  individually account for **<1 %** of self time.

---

## Top-20 by cumulative time — no training (inference only)

```
rank  ncalls    tottime   %tt    cumtime    percall  function
   1    4352     0.005   0.0%    13.579   0.003120  farm/core/agent/core.py:478(act)
   2    4351     0.066   0.4%    13.573   0.003120  farm/core/agent/core.py:429(step)
   3    8702     0.020   0.1%     9.377   0.001078  farm/core/agent/core.py:492(_create_observation)
   4    8563     0.065   0.4%     9.350   0.001092  farm/core/agent/components/perception.py:374(get_observation_tensor)
   5    8563     0.273   1.8%     7.555   0.000882  farm/core/observations.py:1517(perceive_world)
   6    4351     0.047   0.3%     7.285   0.001674  farm/core/agent/core.py:537(_execute_action)
   7       2     0.001   0.0%     5.547   2.773478  farm/core/simulation.py:275(run_simulation)
   8    8563     0.248   1.7%     2.303   0.000269  farm/core/observations.py:1481(update_known_empty)
   9   17126     0.013   0.1%     2.059   0.000120  farm/core/observations.py:1634(tensor)
  11   17126     0.305   2.0%     2.045   0.000119  farm/core/observations.py:1212(_build_dense_tensor)
  18    4351     0.008   0.1%     1.682   0.000387  farm/core/action.py:414(execute)
```

## Top-20 by self (`tottime`) — no training

```
rank   ncalls   tottime   %tt    function
   1      660    1.462   9.8%   <method 'uniform_' of 'torch._C.TensorBase'>   (init noise)
   2    25125    1.019   6.8%   farm/core/observations.py:144(apply_to_dense)
   3    30864    0.957   6.4%   farm/utils/spatial.py:13(bilinear_distribute_value)
   4    34252    0.493   3.3%   farm/core/observations.py:1112(_store_sparse_grid)
   5    23624    0.416   2.8%   farm/core/spatial/index.py:957(get_nearby)
   6    17126    0.305   2.0%   farm/core/observations.py:1212(_build_dense_tensor)
   7     8563    0.273   1.8%   farm/core/observations.py:1517(perceive_world)
   8      660    0.268   1.8%   <method 'clone' of 'torch._C.TensorBase'>
   9    68504    0.253   1.7%   <method 'sum' of 'torch._C.TensorBase'>
  10     8563    0.248   1.7%   farm/core/observations.py:1481(update_known_empty)
  11  1003429    0.246   1.6%   <built-in method builtins.getattr>
  12    34252    0.240   1.6%   farm/core/observations.py:1192(_decay_sparse_channel)
  13    74802    0.237   1.6%   <built-in method torch.tensor>
  14    25689    0.229   1.5%   farm/core/observations.py:471(crop_local)
  15    49868    0.205   1.4%   <built-in method torch.cat>
  16     4351    0.199   1.3%   tianshou/policy/modelfree/dqn.py:122(forward)   (policy forward)
  17     8563    0.181   1.2%   farm/core/observations.py:1342(_compute_entities_from_spatial_index)
  18     8563    0.181   1.2%   farm/core/observations.py:742(make_disk_mask)
  19  1090096    0.178   1.2%   <method 'get' of 'dict' objects>
  20   506743    0.173   1.2%   <built-in method builtins.hasattr>
```

---

## Per-callsite breakdown (cumulative)

| Callsite | with-train (cum s) | no-train (cum s) | notes |
|---|---:|---:|---|
| `agent.step` (full per-agent tick) | 35.03 | 13.57 | aggregate over 4118 / 4351 calls |
| `_create_observation` (called **twice** per step — pre & post) | 9.72 | 9.38 | almost identical regardless of training |
| `perception.get_observation_tensor` | 9.69 | 9.35 | |
| `observations.perceive_world` | 7.83 | 7.56 | dominant inner work |
| `decision.update` (after-action hook) | **21.30** | 0.27 | ~all training |
| `tianshou.train` / `train_on_batch` | **20.92** | 0 | 690 calls in with-train |
| `behaviors.learning.update` | 21.35 | 0.33 | wraps `decision.update` |
| `_execute_action` (action dispatch + 2nd obs + DB log) | 28.41 | 7.29 | most of the no-train cost is the **second** `_create_observation` for `next_state` |
| `decide_action` (policy forward, single agent) | 0.70 | 0.65 | tiny |
| `environment.update` (post-tick env update) | 0.34 | 0.26 | tiny |
| `metrics_tracker.update_metrics` | 0.28 | 0.21 | tiny |
| `resource_manager.update_resources` | 0.013 | 0.010 | negligible |
| `spatial_index.update` (direct calls) | 0.019 | 0.018 | negligible |
| `data_logger.flush_if_needed` | 0.005 | 0.000 | negligible |

(`run_simulation` cumtime denominator behaves oddly across the warm-up + main
runs because cProfile sums them differently; treat the column as relative
weight, not as a fraction of wall.)

---

## Bucketed time by module (`tottime`)

| Module | with-train | no-train |
|---|---:|---:|
| `torch/optim/*` | **2.92 s (7.9 %)** | 0.00 s |
| `torch/nn/modules/*` | 0.28 s | 0.08 s |
| `tianshou/*` | 0.30 s | 0.20 s |
| `farm/core/observations.py` | 3.83 s (10.3 %) | **3.68 s (24.5 %)** |
| `farm/core/spatial/index.py` | 0.48 s | 0.45 s |
| `farm/core/agent/core.py` | 0.41 s | 0.39 s |
| `farm/core/decision/decision.py` | 0.27 s | 0.26 s |
| `farm/core/agent/components/perception.py` | 0.27 s | 0.25 s |
| `farm/core/decision/algorithms/tianshou.py` | 0.17 s | 0.07 s |
| `farm/core/action.py` | 0.12 s | 0.11 s |
| `farm/database/data_logging.py` | 0.05 s | 0.05 s |
| `farm/core/resource_manager.py` | 0.01 s | 0.01 s |
| `farm/core/environment.py` | 0.01 s | 0.01 s |
| `farm/database/database.py` | 0.00 s | 0.00 s |
| **other** (mostly torch C/C++ kernels) | 28.0 s | 9.4 s |

The ≈75 % "other" is overwhelmingly **torch built-in methods** (`run_backward`,
`sqrt`, `linear`, `conv2d`, `uniform_`, `lerp_`, `mul_`, `addcdiv_`,
`addcmul_`, `tensor`, `cat`, `sum`, `zeros_like`). Those are torch C++/ATen
kernels — not Python — and are not amenable to a language port.

---

## What this baseline tells us about porting

1. **RL training (`tianshou.train` → Adam → autograd backward) is the single
   biggest cost on the default 30×30/30‑agent run: ~21 s of 37 s wall ≈ 57 %.**
   Per‑agent training is invoked ~7×/step on average. **No language port
   helps here** — it's already in `torch._C`. The wins are:
   - Lower `learning.training_frequency` (currently 4).
   - Train less often or batch training across agents instead of per‑agent
     calls (`farm/core/decision/decision.py` ~973 invokes
     `algorithm.train(batch=None)` once per agent per N steps; this is 30
     small training calls when one batched call would do).

2. **Perception is the dominant Python‑level cost in both modes**, and the only
   real candidate for a language port:

   - `observations.apply_to_dense`, `_store_sparse_grid`, `_build_dense_tensor`,
     `update_known_empty`, `crop_local`, `make_disk_mask`,
     `_compute_entities_from_spatial_index`, `_decay_sparse_channel`
   - `farm/utils/spatial.py:bilinear_distribute_value`
   - `farm/core/spatial/index.py:get_nearby` (already a thin wrapper over
     scipy `cKDTree`, but called 23 k+ times)

   Together: **~6.4 s of 15 s wall in no-train (≈42 %)**, and similar
   absolute cost in with‑train. **A Numba/Cython/Rust port of the perception
   inner loop is the highest‑leverage native acceleration.**

3. **`_create_observation` is called twice per agent per step** — once for
   `state_tensor` at the top of `AgentCore.step`, then again for
   `next_state_tensor` at the bottom of `_execute_action`
   (`farm/core/agent/core.py` ~459 and ~633). That's **8 563 / 4 351 ≈ 1.97×
   per step** in the no‑train profile, and explains why `_execute_action`
   cumtime (7.3 s) is so close to perception cumtime (9.4 s) — most of
   `_execute_action`'s cost is the *second* observation. **Caching
   `next_state` (or skipping it when the agent did `pass`) is a free 30–40 %
   win** on the no‑train path with no native code.

4. **Policy forward is tiny** (`decide_action` 0.65 s, `dqn.forward` 0.20 s
   self) — ~4 % of wall. Batching forwards across agents would shave a few
   percent but isn't where the time is. If we ever port to a native kernel,
   we should not bother native‑porting the policy call.

5. **Spatial index, environment update, resource regen, metrics, DB flushes
   are all <2 % each.** None of these are worth porting at this scale. The
   reports in `benchmarks/reports/0.1.0/` saw spatial queries dominate at
   *large* scales (~69 % of perception time at 200+ agents); at 30 agents
   the KD‑tree is essentially free and the Python overhead around it is what
   costs.

---

## Revised recommendation in light of the data

The Stage 1 / Stage 2 plan from the previous turn still stands, but **the
ordering changes** because the small‑grid bottleneck mix is:

> **~57 % training, ~25 % perception (Python), ~10 % other torch ops, ~5 %
> policy forward, <3 % everything else.**

So:

1. **First (no native code, biggest win on the small grid):**
   - Lower `training_frequency` from 4 → 16 (or implement a single shared
     batched training step across agents per N global steps). Expected: ~40 %
     wall reduction on default settings.
   - Cache `next_state` reuse in `AgentCore._execute_action` (don't rebuild
     observation when nothing observation‑relevant changed). Expected: ~20 %
     wall reduction on the inference path.
   - Batch `decide_action` across agents into a single `policy(obs_batch)`
     call. Small absolute win but unlocks GPU later.

2. **Second (Python‑level vectorization):**
   - Vectorize `bilinear_distribute_value` and the per‑resource loops in
     `observations.perceive_world` / `_build_dense_tensor` /
     `_store_sparse_grid` to operate on `(N_neighbors, …)` tensors instead of
     Python loops. This is a `numpy`/`torch.scatter_add` rewrite, not a port.

3. **Only then native code:**
   - If after (1) and (2) perception is still the bottleneck, the candidate
     for a Rust + PyO3 (or Numba/Cython) port is a **single
     `build_observation_batch(positions, channels_state) -> ndarray`** kernel
     covering the channel handlers in `farm/core/observations.py`. Everything
     else in the loop is either already native (torch) or too small to matter.

A native port of the **whole step loop** — agents, decision, action dispatch,
DB — is **not justified** by this profile. The Python orchestrator costs <5 %
of wall; the gain wouldn't repay the rewrite, and it would force the RL
policies and DB into a foreign‑language interface for no measurable benefit.

---

## Post-change verification (deferred RL scheduler vs immediate training)

After implementing the deferred RL training scheduler (`performance.defer_learning_training=true`,
`performance.max_learning_updates_per_step=4`), we reran the same Stage 0 workload and
compared it against a forced immediate-training run on the same seed and config:

- grid: 30x30
- agents: 30 (10/10/10)
- steps: 100 (+5 warmup)
- seed: `1234567890`
- in-memory DB enabled

| Scenario | wall (s) | steps/sec | ms / step | alive at end |
|---|---:|---:|---:|---:|
| **deferred scheduler** (new default) | **22.085** | **4.53** | **220.85** | 56 |
| **forced immediate training** (`defer_learning_training=false`) | 36.077 | 2.77 | 360.77 | 53 |

Observed improvement on this profile:

- **Wall-time reduction:** `(36.077 - 22.085) / 36.077 = 38.8%`
- `tianshou.py:1104(train_on_batch)` cumulative time:
  - immediate: **21.2985 s**
  - deferred: **7.1386 s**
  - reduction: **66.5%**
- Training-call count in the profile top table:
  - immediate: **690** `train_on_batch` calls
  - deferred: **232** `train_on_batch` calls

Reproduction commands used for this verification:

```bash
source venv/bin/activate
PYTHONHASHSEED=0 python -m scripts.profile_step_loop --steps 100 \
  --out simulations/profile_step_loop_deferred.prof

# forced immediate-training comparison (defer_learning_training=false)
PYTHONHASHSEED=0 python - <<'PY'
from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
import cProfile, pstats, time, os

cfg = SimulationConfig.from_centralized_config(environment="development")
cfg.environment.width = 30
cfg.environment.height = 30
cfg.population.system_agents = 10
cfg.population.independent_agents = 10
cfg.population.control_agents = 10
cfg.population.max_population = 60
cfg.database.use_in_memory_db = True
cfg.database.persist_db_on_completion = False
cfg.database.enable_validation = False
cfg.performance.defer_learning_training = False

run_simulation(num_steps=5, config=cfg, path=None, save_config=False, disable_console_logging=True)

prof = cProfile.Profile()
prof.enable()
t0 = time.time()
run_simulation(num_steps=100, config=cfg, path=None, save_config=False, disable_console_logging=True)
prof.disable()
print("wall", time.time() - t0)
os.makedirs("simulations", exist_ok=True)
prof.dump_stats("simulations/profile_step_loop_immediate.prof")
PY
```

---

## Files / artifacts

- `scripts/profile_step_loop.py` — reproducible Stage 0 profiler.
- `simulations/profile_step_loop.prof` — binary cProfile (with training).
- `simulations/profile_step_loop.txt` — text top‑80 cumulative + tottime.
- `simulations/profile_step_loop_notrain.prof` — binary cProfile (no training).
- `simulations/profile_step_loop_notrain.txt` — text top‑80 cumulative + tottime.
- `simulations/profile_step_loop_deferred.prof` — binary cProfile (deferred RL scheduler).
- `simulations/profile_step_loop_deferred.txt` — text top‑80 cumulative + tottime.
- `simulations/profile_step_loop_immediate.prof` — binary cProfile (forced immediate training).
- `simulations/profile_step_loop_immediate.txt` — text top‑80 cumulative + tottime.

View interactively with `snakeviz simulations/profile_step_loop.prof` (already
in `requirements.txt`).
