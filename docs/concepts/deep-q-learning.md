# Deep Q-Learning in AgentFarm

## Overview

AgentFarm currently supports DQN through two paths:

1. **Active decision path (recommended):** `DecisionModule` + Tianshou `DQNWrapper`.
2. **Standalone legacy module:** `BaseDQNModule` in `farm/core/decision/base_dqn.py`.

The active path is what gets used when `DecisionConfig.algorithm_type="dqn"` and Tianshou is available.

## Current DQN Stack (Active Path)

### `DecisionConfig` controls algorithm and replay settings

`DecisionConfig` includes DQN parameters (`learning_rate`, `gamma`, `target_update_freq`, epsilon fields), plus replay controls:

- `replay_strategy`: `"uniform"` or `"prioritized"`
- `per_alpha`, `per_beta_start`, `per_beta_end`, `per_beta_steps`, `per_epsilon`

Example:

```python
from farm.core.decision.config import DecisionConfig

config = DecisionConfig(
    algorithm_type="dqn",
    replay_strategy="prioritized",
    learning_rate=1e-3,
    gamma=0.99,
    target_update_freq=500,
    epsilon_start=1.0,
    epsilon_min=0.01,
)
```

### `DecisionModule` initializes `DQNWrapper`

`DecisionModule` builds a per-agent algorithm instance and forwards replay/PER settings into the Tianshou wrapper.

For DQN, the wrapper creates an adaptive Q-network:

- **3D observations** (for example `(C, H, W)`): CNN backbone + MLP head.
- **1D/2D observations**: fully connected MLP.

### Target network update behavior

In the active Tianshou DQN path:

- `target_update_freq` is used for **hard target sync cadence**.
- The wrapper currently enforces **1-step replay targets** (`n_step` overridden to `1` in the custom replay integration path).

## Replay Buffer and PER

The active wrappers use `PrioritizedReplayBuffer` (`farm/core/decision/algorithms/rl_base.py`), which supports:

- **Uniform replay** and **prioritized replay** behind one API.
- New transitions inserted with current max priority (or `1.0` initially).
- Sample output that always includes:
  - `indices` (for priority updates)
  - `is_weights` (importance-sampling weights; all ones under uniform replay)
- Beta annealing via `update_beta()` during prioritized training.

Training metrics expose replay diagnostics, including:

- `replay_beta`
- `replay_priority_min`, `replay_priority_max`, `replay_priority_mean`
- `replay_is_weight_mean`

## Training Flow (Active Path)

1. Agent interaction stores transitions through `DecisionModule.update(...)`.
2. Wrapper appends to replay and increments step count.
3. `should_train()` gates updates using:
   - replay length >= `batch_size`
   - `step_count % train_freq == 0`
4. `train_on_batch()` samples replay, trains policy, then updates PER priorities (if enabled).

## Action Selection

`DecisionModule.decide_action(...)` supports:

- Action masks for curriculum constraints.
- Optional per-action weight biasing (`action_weights`).
- Algorithm probability paths when available (`predict_proba`), with fallback weighted-random behavior.

## Standalone Legacy DQN Module (`BaseDQNModule`)

`BaseDQNModule` remains available and implements a conventional DQN loop:

- `BaseQNetwork` with LayerNorm + Dropout.
- `deque` replay memory.
- Double-DQN target construction in `train(...)`.
- Soft target updates with `tau`.
- Optional gradient clipping.
- Epsilon-greedy action selection with a small state-action cache.

This module is still useful for direct module-level experiments, but the main agent decision pipeline uses `DecisionModule` + wrappers.

## Notes on Accuracy

- The codebase does **not** currently define `SharedEncoder`, `MoveQNetwork`, or `AttackQNetwork` in the DQN path.
- Agent orchestration has shifted from a monolithic `train_all_modules()` pattern to per-algorithm readiness checks (`train_if_ready` / `should_train`) in the decision layer.
