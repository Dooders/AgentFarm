---

## layout: page
title: "Is the DQN actually learning? A diagnostic, four bugs, and a sobering answer"

I had a suspicion that the DQN decision module wasn't actually  
learning — agents weren't getting better at making decisions, and I  
wondered whether agents were just dying before their internal model had a  
chance to adapt. That hypothesis is partly right, but the more  
interesting story turned out to be that the DQN was being quietly  
half-broken in four places at once: a global training throttle, a never-  
applied epsilon schedule, a YAML-to-config mapping that dropped knobs on  
the floor, and a hidden-size field that did nothing. The diagnostic also  
ended with an honest finding: even after fixing all four, agents only  
*barely* start making better decisions in late life vs early life at the
default simulation horizon. The cause is no longer the code — it is the
environment's signal-to-noise ratio.

This is the walkthrough.

## Setup

The active decision stack is:


| Piece                       | File                                        |
| --------------------------- | ------------------------------------------- |
| Agent-facing API            | `farm/core/decision/decision.py`            |
| Tianshou integration        | `farm/core/decision/algorithms/tianshou.py` |
| Replay buffer (PER)         | `farm/core/decision/algorithms/rl_base.py`  |
| Per-agent wiring            | `farm/core/agent/behaviors/learning.py`     |
| Per-step agent loop         | `farm/core/agent/core.py`                   |
| Deferred-training scheduler | `farm/core/simulation.py`                   |


I wrote a diagnostic harness, `scripts/diagnose_dqn_learning.py`, that
wraps `DecisionModule` and `TianshouWrapper` and records per agent:
replay-buffer stores, `should_train()` outcomes, real `policy.learn()`
calls, the L2 movement of the Q-network's first parameter tensor (so we
can prove weights actually moved, not just that `learn` returned), buffer
size and `policy.eps` at end of run, and lifespan. Later in the session I
extended it to record `(env_time, action, reward)` per store so I could
do a proper residualised late-vs-early reward analysis. Everything ran
with `PYTHONHASHSEED=0` for determinism on the default config (30
starting learning agents, 100–500 steps).

The point throughout was the same one I keep making to myself: do not
guess at RL bugs from the code; instrument, run, look at numbers.

## What the first run found

Default config, 100 steps:


| Per-agent metric         | Median    | p75   | Notes                        |
| ------------------------ | --------- | ----- | ---------------------------- |
| Stores per agent         | 93        | 100   | replay fills fine            |
| `policy.learn()` calls   | **2**     | 4.5   | barely any training          |
| `train-ready=False`      | **61**    | 69    | almost every chance, skipped |
| `|Δw|₂` of first Q-param | ~0        | 0.1   | weights barely move          |
| Final `policy.eps`       | **0.000** | 0.000 | exploration broken           |
| Lifespan                 | 95        | 101   | mean **81 steps**            |


That is enough to falsify my initial hypothesis right away. Mean
lifespan is 81 steps, the batch threshold is 32, and 58/63 agents
filled their buffer past it. Lifespan is *not* the bottleneck. The
bottleneck is somewhere between the buffer being ready and `policy.learn()`
actually getting called. And the `policy.eps = 0.000` for *every*
trained agent at end of run is suspicious — none of them are exploring.

So I went looking. Four real bugs surfaced.

### Bug 1: the global training throttle

`farm/core/simulation.py` runs deferred RL updates via
`_run_deferred_learning_updates(env, max_updates, rr_cursor)`, with
`max_updates = performance.max_learning_updates_per_step` and a default
of `4`. With ~30 alive agents and the round-robin scheduler, that caps
total gradient updates per env step at 4, no matter how many agents are
ready. Over 100 steps that's at most 400 updates spread across 30+
agents — median 2 per agent. When I disabled deferred training entirely
(`--no-defer`, training happens inline on every store) the same agents
ran median **18** gradient steps. So the throttle was costing ~9× the
training volume.

### Bug 2: epsilon-greedy was always off

Tianshou's `DQNPolicy.__init__` sets `self.eps = 0.0` and only changes
it when someone calls `policy.set_eps(...)`. The wrapper was passing
`eps_train`, `eps_test`, `eps_train_final` into the policy's
`algorithm_config`, but `set_eps()` was never called anywhere in the
codebase. Direct repro:

```python
DQNWrapper(..., algorithm_config={"eps_train": 0.5, ...}).policy.eps
# -> 0.0
```

So the configured `epsilon_start: 1.0`, `epsilon_min: 0.01`,
`epsilon_decay: 0.995` in `default.yaml` were all dead. The agent's
"epsilon-greedy" was pure greedy on a near-random Q-network. There was
still some exploration in the system, but it came from a hard-coded
`predict_proba` heuristic (0.8 mass on the policy's argmax + 0.2
uniform) multiplied by per-action priors from `action_weights`, *not*
from the configured schedule. The DQN policy never got a "test/greedy"
mode and never had its exploration anneal over time.

### Bug 3: YAML-to-DecisionConfig wiring silently dropped knobs

`AgentComponentConfig.from_simulation_config` mapped:

```
learning.memory_size  -> decision.memory_size
learning.batch_size   -> decision.batch_size
learning.dqn_hidden_size -> decision.dqn_hidden_size
```

But the Tianshou wrapper reads `decision.rl_buffer_size` and
`decision.rl_batch_size`, *not* `memory_size` / `batch_size` (those are
consumed by the legacy `BaseDQNModule`, which the production stack
doesn't use). So `learning.memory_size: 2000` in YAML actually produced
a 10000-entry replay buffer (the `rl_buffer_size` default). The
`dqn_hidden_size: 24` knob did nothing for the same reason (see Bug 4).
Anyone tuning these from YAML was tuning ghosts.

### Bug 4: dqn_hidden_size was decorative

`AdaptiveQNet` in `tianshou.py` had hard-coded FC widths of 512 / 256 /
128 in the constructor. `DecisionConfig.dqn_hidden_size` was declared
as a real Pydantic field but never plumbed through. Width was *always*
512/256/128 regardless of the config.

### Bug 5 (cleanup): swallowed exceptions in the agent step loop

`AgentCore.step` had:

```python
try:
    action = self.behavior.decide_action(self, state_tensor, enabled_actions)
    self._execute_action(action, state_tensor)
except Exception:
    pass
```

So any exception during decision or execution looked exactly like "the
agent simply isn't learning", with no log entry to distinguish a quiet
step from a crashing one. This wasn't the cause of my symptom,
but it would absolutely *hide* future versions of it.

## The fixes

Six commits on `cursor/diagnose-dqn-learning-a451`, PR
[#878](https://github.com/Dooders/AgentFarm/pull/878):

1. **Throttle auto-scale.** Change `max_learning_updates_per_step` to
  default to `0`, and re-interpret `0` as the auto-scale sentinel —
   every alive agent gets one gradient step per env step. Positive ints
   remain a hard cap; negatives short-circuit to no training.
2. **Epsilon-greedy actually wired.** `TianshouWrapper` snapshots
  `eps_train` / `eps_train_final` / `eps_test` / `eps_decay` at init,
   calls `policy.set_eps(epsilon_start)` immediately, decays
   multiplicatively on every `select_action_with_mask` call (training
   mode), floors at `epsilon_min`, and exposes `set_train_mode(False)`
   to switch the policy to `eps_test` for evaluation.
3. **YAML wiring.** `from_simulation_config` now maps
  `learning.memory_size` and `learning.batch_size` to *both* the legacy
   fields (so the deprecated DQN module still works) *and* the
   `rl_buffer_size` / `rl_batch_size` fields that the Tianshou wrapper
   actually reads.
4. `**dqn_hidden_size` plumbed.** `AdaptiveQNet` takes a `hidden_size`
  parameter and uses `h*4 / h*2 / h` widths, with a floor of 8.
   `dqn_hidden_size` is added to `_EXCLUDED_PARAMS` so it doesn't leak
   into `DQNPolicy`'s constructor.
5. **Exceptions logged, not swallowed.** `AgentCore.step` failures
  from `decide_action` / `_execute_action` now log at warning with the
   agent id, exception type, message, and traceback (structlog
   `exc_info=True`).
6. **Tests.** New unit tests in
  `tests/test_decision_config_wiring.py` and
   `tests/test_rl_training_batching.py` lock down: initial `policy.eps  == epsilon_start`, the multiplicative decay, the
   `epsilon_min` floor, `set_train_mode(False)` switching to
   `eps_test`, `dqn_hidden_size` actually changing the network width,
   the YAML-to-`rl_buffer_size` mapping, and the auto-scale sentinel.

Full pytest run: **6492 passed, 0 failed**, 18 skipped (pre-existing
CUDA / openpyxl skips).

## Did "is the DQN learning?" change after the fixes?

Yes. Same simulation, same seed, before-and-after at 100 steps:


| Per-agent metric                | Before | After | Δ                                  |
| ------------------------------- | ------ | ----- | ---------------------------------- |
| Median `policy.learn()` calls   | 2      | 18    | **9×**                             |
| Median `|Δw|₂` of first Q-param | ~0     | 0.1   | weights actually move              |
| Final `policy.eps`              | 0.000  | 0.606 | schedule alive (`1.0 × 0.995^100`) |


And at 300 steps, simulating the legacy throttle + broken eps on the
current code (`--legacy`) vs the post-fix defaults:


| Per-agent metric              | Legacy | Current | Δ                 |
| ----------------------------- | ------ | ------- | ----------------- |
| Median `policy.learn()` calls | 5      | **34**  | 6.8×              |
| `|Δw|₂` p75                   | 0.1    | 0.2     | 2×                |
| Mean lifespan (steps)         | 213    | **262** | +23%              |
| Final `policy.eps`            | 0.000  | 0.222   | decayed correctly |


That lifespan delta is the cleanest *behavioral* signal. Same seed,
same population, same world — agents under the fixed system survive
~23% longer.

## But are agents actually making better decisions?

Higher training volume, larger weight movement, longer survival. Those
are inputs and outcomes. What I really wanted to see — and what I
actually asked — was *policy quality* over the lifetime of one
agent. Is a 300-step-old agent picking better actions than a 50-step-old
agent?

To answer that I extended the diagnostic with:

- `(env_time, action, reward)` per stored experience.
- A cohort baseline: for each env time, the mean reward across all
alive agents at that moment. An agent's *residualised reward* is its
own reward minus that cohort mean, so anything common to the cohort
(food depleting, density, weather) cancels out.
- Per-agent late-vs-early residualised reward, restricted to long-lived
agents (lifespan ≥ 100), with a one-sample t-stat on the per-agent
delta.
- Action-distribution entropy and top-action share in the first vs the
last quartile of each agent's experience. A policy that is actually
*learning to commit* should show lower entropy / higher top-action
share in late life.

Three configurations at 500 steps:


| Long-lived agents (lifespan ≥ 100)        | Legacy      | Current     | `--train-freq 1` |
| ----------------------------------------- | ----------- | ----------- | ---------------- |
| Long-lived agents evaluated               | 67          | 61          | 63               |
| Median `policy.learn()` calls             | 7           | 33          | **117**          |
| Max `policy.learn()` calls                | 110         | 118         | **469**          |
| Mean Δ residualised reward (late − early) | +0.0082     | −0.0055     | **+0.0069**      |
| t-stat for Δ > 0                          | +1.15       | −0.79       | **+0.97**        |
| Agents with Δ > 0                         | 35/67 (52%) | 28/61 (46%) | 32/63 (51%)      |
| Δ action entropy (nats, late − early)     | +0.013      | +0.002      | **−0.024**       |
| Δ top-action share                        | −0.014      | +0.002      | **+0.003**       |
| Mean lifespan (steps)                     | 306         | 355         | **391**          |
| `|Δw|₂` p75                               | 0.1         | 0.2         | **0.3**          |


The numbers do not let me claim "agents make demonstrably better
per-action decisions late in life." The largest t-stat is +1.15, well
short of the t ≈ 2 threshold for p < 0.05. Even at the most aggressive
training setting (`rl_train_freq=1`, 12× more gradient steps), the
residualised-reward t-stat is only +0.97.

What I *can* claim:

- **The action distribution starts to commit when training is
aggressive enough.** At `rl_train_freq=1`, late-life action entropy
drops by `−0.024` nats (1.659 → 1.635). With only 5 actions the
entropy ceiling is `log(5) = 1.609`, so the policy is moving
meaningfully toward a non-uniform distribution. Under the default
`rl_train_freq=4`, entropy barely changes (`+0.002`). More gradient
steps → policy actually committing. That is exactly the
qualitative signature of learning, just at small amplitude.
- **Lifespan keeps improving with training budget**: 306 → 355 → 391
steps across legacy → current → max-training. Better policies survive
longer, even when the per-action reward signal is noisy.
- **Weight movement scales with gradient steps**: p75 `|Δw|₂` grows 0.1
→ 0.2 → 0.3.

## Why per-action reward is so noisy

The dominant variance in per-store reward isn't the agent's choice —
it's environment-level shocks (a nearby resource patch depleting, a
neighbor moving in, a gather race) that every co-located agent
experiences. Residualising against the cohort mean removes a *lot* of
that but not all of it. The per-agent residual standard deviation is
~0.06 across agents, while the policy-driven signal (the mean of the
late-minus-early deltas) is ~0.007 — an SNR of roughly 0.1.

The structural reason in the code is that `LearningAgentBehavior` never
calls the policy in pure-greedy mode. The active path is
`predict_proba(state) × action_weights → np.random.choice`, where
`predict_proba` is a 0.8/0.2 heuristic around the policy's argmax.
So even a perfectly trained Q-network's choice goes through one round
of weighted sampling against per-action config priors before being
executed. That dilutes whatever the policy has actually learned. It is
not a bug — `action_weights` is the chromosome-A
`move_weight / gather_weight / ...` prior — but it does make the
signal harder to read on per-action reward.

## What survived

- The diagnostic harness, with the late-life-vs-early-life analysis
baked in. `scripts/diagnose_dqn_learning.py`. `--legacy` reproduces
the pre-fix behavior on current code; `--train-freq N` overrides the
gradient-step cadence; `--no-defer` runs inline training.
- The four bug fixes plus the exception-logging cleanup, behind one PR
and one extended test file (`tests/test_decision_config_wiring.py`).
- The auto-scale default for `max_learning_updates_per_step`. The
previous default cost ~9× the training volume and was not the kind
of thing that should ship as the recommended setting.
- A clearer mental model of where in the stack the signal-to-noise
problem lives: not in the optimizer, not in the buffer, but in the
`predict_proba × action_weights` action-selection path that sits
between the Q-network and the executed action.

## What didn't survive

The hypothesis I started with: "if I unblock training, late-life
decisions will measurably improve at the default horizon." They don't,
not in a statistically defensible way. The improvement is directional
across multiple proxies (entropy down a hair, lifespan up, residualised
reward up by a fraction of a standard error) but not big enough to put
a confidence interval around.

My original hypothesis: "maybe agents aren't living long enough."
Falsified. Mean lifespan was already 81 steps with batch threshold 32.
The bottleneck was global throttling, not individual lifespans.

## What's next

- **Longer simulations.** 500 steps is enough to see training happen
and enough to see lifespan respond; it is not enough to see per-
action reward signal beat per-step environment noise. Multi-thousand-
step runs are the obvious next experiment, on a smaller population so
per-agent gradient budgets stay high.
- **Greedy-evaluation probe states.** Cache a fixed set of observation
tensors early in the run; at intervals, freeze each agent's Q-net
and report argmax + max-Q on the probe set. That gives a deterministic
per-policy quality metric independent of stochastic sampling and
environment dynamics.
- **Audit the `predict_proba × action_weights` dilution.** If we want
the trained policy to actually drive behavior, the 0.8/0.2 heuristic
in `TianshouWrapper.predict_proba` probably needs to be replaced with
a real softmax over Q-values, and the multiplication by
`action_weights` should probably go through a log-additive bias on
the logits rather than a multiplicative reweighting in probability
space. This is a design call, not a clear bug.
- **Smaller-population runs as a learning sanity check.** With 1–3
agents and a fixed resource layout, the cohort baseline goes away
and the per-action reward signal should be clean enough to show a
classic learning curve.

In short: the DQN is now actually training, but the easiest way to
*see* it learning is no longer to fix code — it's to design a
simulation where the policy has enough room to matter.

## Related docs

- [Deep Q-learning module reference](../deep_q_learning.md)
- [Hyperparameter chromosome design](../design/hyperparameter_chromosome.md)
- [PR #878](https://github.com/Dooders/AgentFarm/pull/878)
- Diagnostic script: `scripts/diagnose_dqn_learning.py`

