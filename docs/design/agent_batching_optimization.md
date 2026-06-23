# Proposal: ML compute optimizations for the simulation loop

> Status: proposed / not yet implemented
>
> This document is the design write-up behind the performance follow-up referenced
> in the observation-pipeline optimization PR. It is intended to seed GitHub issues
> and guide implementation.
>
> The lead proposal is **batching per-agent observation construction and policy
> inference** (below). A broader set of ML recommendations — split by whether they
> preserve numerical results — follows in
> [Additional ML recommendations](#additional-ml-recommendations).

## Lead proposal: batch per-agent observation construction and policy inference

## Summary
Profiling of the development workload (`run_simulation.py --environment development --steps 2000 --seed 1234567890`) shows that roughly **50% of simulation runtime is torch ML compute** — DQN policy inference plus deferred training (`torch.conv2d` ~18s, `run_backward` ~16.7s, `linear` ~5.2s, Adam ~13s in a 600-step cProfile run). This work currently runs **once per agent per step** on tiny per-agent observation tensors (≈13×13 multi-channel), which is dominated by Python/dispatch overhead and is highly inefficient on small tensors.

This proposes **batching observation construction and policy inference (and ideally the deferred training step) across all agents that share a policy**, so each simulation step performs one vectorized forward/backward pass over a batched tensor instead of N small ones. This should meaningfully cut the dominant ML cost **without changing learning numerics** if done carefully.

## Background / current state
- Each agent builds its own observation via `PerceptionComponent.get_observation_tensor` → `AgentObservation.perceive_world` (`farm/core/observations.py`), then runs an individual policy forward in `farm/core/decision/`.
- The step loop in `farm/core/simulation.py` already processes agents in batches (`agent_processing_batch_size`, default 32) but calls `agent.act()` per agent, so inference/observation are not vectorized.
- Recent behavior-preserving optimizations reduced the *observation pipeline* overhead (~12% end-to-end), but the ML compute itself was intentionally left untouched because changing it risks changing results:
  - memoize VISIBILITY disk mask
  - disable benchmark-only observation metrics on the live path
  - NumPy-buffer resource accumulation + redundant spatial-update removal

## Proposed approach
1. **Batch observation tensors:** stack the per-agent observation tensors for all alive agents (sharing the same shape/device/dtype) into a single `(N, C, S, S)` batch.
2. **Single inference pass:** run one policy forward over the batch to get per-agent action distributions, then dispatch actions back to agents.
3. **Batched deferred training:** where agents share a policy/optimizer, accumulate transitions and run one (or few) batched training update(s) per step instead of per-agent updates.
4. **Handle heterogeneity:** group agents by policy/network identity (e.g., system vs. independent vs. control agents, or per-shared-module) and batch within each group; fall back to the per-agent path for singletons.

## Hard constraints (must hold)
- **Numerical results must be identical** (or the change must be explicitly gated behind a flag and validated). The repo is fully deterministic with a fixed seed + `PYTHONHASHSEED=0`; a batched forward must produce the same action selections and the same training gradients/updates as the sequential path.
- **Determinism oracle:** validate against a full-sim state signature (agent ids/alive/positions/health/resources + resources + time + cached totals) — it must remain bitwise-identical, or any divergence must be understood and justified.
- `pytest -q` must stay green (modulo the known pre-existing `tests/analysis/*` matplotlib/genetics failures).

## Risks / open questions
- **Batched vs. sequential float math:** batched matmul/conv may produce slightly different floating-point results than N separate calls (reduction order), which would change action selection and break determinism. Need to measure; may require keeping per-agent math where exactness is required, or accepting a flag-gated "fast/approximate" mode.
- **Training order semantics:** sequential deferred updates apply gradients one agent at a time (each update sees the previous one's weights); a single batched update changes that semantics. Decide whether to (a) preserve exact sequential semantics (limits batching to inference only) or (b) define batched training as the new, intended semantics behind a config flag.
- **Variable agent counts / deaths within a step**, mixed devices, and per-agent enabled-action masks (curriculum) need handling in the batch.

## Suggested scope / phasing
- **Phase A (lower risk):** batch *observation construction + inference only* (action selection), keep training per-agent. Validate determinism.
- **Phase B (higher risk):** batch deferred training for shared-policy groups behind a config flag; document the semantic change and validate learning-curve equivalence.

## Acceptance criteria
- Measurable reduction in step time on the development workload (target: a substantial cut in the ~50% ML share).
- Determinism signature unchanged for Phase A (or flag-gated + documented for Phase B).
- `pytest -q` green; new tests covering batched-vs-sequential equivalence.

## Pointers
- `farm/core/simulation.py` — step loop, `_run_deferred_learning_updates`.
- `farm/core/agent/core.py` — `act()` / `step()` / `_execute_action`.
- `farm/core/decision/` — policy/decision modules, tianshou wrappers.
- `farm/core/agent/components/perception.py`, `farm/core/observations.py` — observation construction.

---

# Additional ML recommendations

These extend the lead batching proposal. They are split by whether they **preserve
numerical results** (validate with the bitwise determinism state-signature oracle) or
**change results** (must be opt-in / flag-gated and validated by learning-curve or
outcome equivalence, *not* the bitwise oracle).

## Context from profiling

The policy is a comparatively heavy CNN for its input: 3 conv layers (32/64/64
channels) over the ~13×13 multi-channel observation, followed by
`Linear(flattened ≈ 10,816 → 512) → Linear(512 → 256) → action head` (plus a value
head for the actor-critic path). This network runs **per agent, per step** for both
inference and deferred training. In a 600-step cProfile run the torch ops dominate:
`conv2d` ~18s, `run_backward` ~16.7s, Adam ~13s, `linear` ~5.2s, `sqrt` ~4.1s — roughly
**50% of total runtime**. The first `Linear(10,816 → 512)` alone is ~5.5M parameters and
accounts for much of the `linear`/Adam/backprop cost and memory.

## Tier 1 — numerics-preserving (safe; same results)

1. **Batch observation build + inference across same-policy agents** — the lead proposal
   above; highest, safest payoff.
2. **Use `torch.inference_mode()` on the predict path** (not just `no_grad`). Inference
   already uses `no_grad` contexts; `inference_mode` additionally skips version-counter /
   view bookkeeping. Verify `predict_proba` / `_policy_q_values` in
   `farm/core/decision/algorithms/tianshou.py`.
3. **Cut tensor-creation churn.** `torch.tensor` is called ~342k times (~1.6s) — many
   per-step scalar→tensor conversions and host↔device round-trips. Preallocate/reuse
   buffers and prefer `torch.as_tensor` / `torch.from_numpy`. Bit-identical.
4. **Eliminate duplicate forward passes.** Action selection does a forward and the
   training step does another; where the same state is evaluated twice within a step,
   cache and reuse the Q-values.
5. **Thread tuning (after batching).** On few cores, batch-1 inference can be faster with
   fewer threads while batched training benefits from more. Gate via config rather than
   hardcoding, since the optimum is hardware-dependent.

> Note: `torch.compile` can help but typically perturbs floating-point results, which
> breaks the bitwise-determinism guarantee — only adopt it if exact reproducibility is
> relaxed.

## Tier 2 — architectural / behavior-changing (opt-in; change results)

1. **Right-size the network.** Replace the large `Linear(10,816 → 512)` with a
   global-average-pool after the conv stack and/or smaller hidden dims (e.g. 128/64).
   Large cut to conv/linear/Adam/backprop cost and memory.
2. **Shared conv encoder + lightweight per-action heads.** The move/attack/gather/share
   modules appear to use separate networks; a shared feature extractor feeding small heads
   removes redundant feature extraction.
3. **Share policy weights across same-type agents** (instead of per-agent networks).
   Enables true batching and cuts memory ~N×. Only valid if "agents share a policy"
   matches the research intent.
4. **Mixed precision (bf16/fp16)** for conv/matmul — real speedups, changed numerics.
5. **Tune training cadence / replay batch size** (`max_learning_updates_per_step`, replay
   batch) — already configurable; trades learning quality for speed.

## Suggested sequencing
Implement Tier 1 #1 (batching) first and re-measure. Only then consider Tier 2. Every
Tier 2 item changes results, so it must be flag-gated and validated against
learning-curve / outcome equivalence rather than the bitwise determinism oracle.

## Caveat
The per-agent-vs-shared-network and `inference_mode` specifics above are based on a light
audit of `farm/core/decision/` and should be confirmed during implementation.
