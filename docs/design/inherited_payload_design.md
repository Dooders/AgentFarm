# Inherited Payload Design (Issue #848)

This design note answers a single question from first principles: **at the
moment of reproduction, what should an offspring inherit from its parent's
*learned* decision module — beyond the hyperparameter chromosome it already
gets?**

Issue: [#848](https://github.com/Dooders/AgentFarm/issues/848) ·
Implementation/experiment tracker:
[#904](https://github.com/Dooders/AgentFarm/issues/904) ·
Prereqs: [#901](https://github.com/Dooders/AgentFarm/issues/901),
[#902](https://github.com/Dooders/AgentFarm/issues/902),
[#903](https://github.com/Dooders/AgentFarm/issues/903)

It supersedes the earlier "V1 priors / V2 module-state / V3 gated" sketch
(proposed in PR #880 before the inheritance experiments ran). That sketch
ordered payloads by *richness* and assumed richer → faster adaptation. The
experiments we have since run invert that assumption, so the design below is
re-derived from what the system actually does.

## What we already know (and why the old plan is stale)

Three results post-date the original sketch and constrain the design:

1. **Within-life learning barely moves the policy at the default horizon.**
  The DQN diagnostic ([devlog 2026-05-16](../devlog/2026-05-16-is-the-dqn-actually-learning.md),
   PR #878) showed that after fixing four training bugs, late-vs-early
   per-action decision quality still does not improve in a statistically
   defensible way at 500 steps (best t ≈ 1.15). Weight movement is modest
   (`|Δw|₂` p75 ≈ 0.2) and the per-action reward SNR is ~0.1.
   *Consequence:* the **upper bound** on any inheritance benefit is small in
   the default regime, because there is little learned signal to transmit.
2. **The maximal payload — a full policy-weight copy — is already a null.**
  Lamarckian warm-start ([#849](https://github.com/Dooders/AgentFarm/issues/849),
   `farm/core/policy_inheritance.py`) copies the parent's entire
   `policy_state_dict` into the offspring. The 36-run A/B
   ([devlog 2026-05-21](../devlog/2026-05-21-baldwinian-vs-lamarckian-ab-harness.md))
   and the newborn-level follow-up
   ([devlog 2026-06-04](../devlog/2026-06-04-are-we-measuring-at-the-wrong-level.md))
   found **no fitness gain**: net early RL reward is, if anything, slightly
   *lower* under warm-start; the only robust effect is a ~1pp drop in
   negative-reward actions. Baldwinian stays the default.
   *Consequence:* "copy more of the network" is not the lever. The strongest
   version of the old V2 has already been tested and lost.
3. **Action priors are already inherited via the chromosome.** The executed
  action goes through `predict_proba(Q) × action_weights → sample`, where
   `action_weights` is the Chromosome-A prior (`move_weight`, `gather_weight`,
   …). So the old "V1: lightweight policy-prior summary (action-preference
   logits)" is largely **redundant** with Baldwinian inheritance — those
   priors already cross the generational boundary as genes.

The honest reading: the parent's policy *answer* (its weights) transfers fine
but does not help, while the parent's policy *priors* already transfer as
genes. So the interesting design space is neither "more weights" nor "more
priors."

## First-principles decomposition

A learned decision module is not one object; it is a trained network plus the
machinery that produced and sustains it. Each component is separately
transferable, and — this is the crux — **most components only pay off in the
presence of the others.**


| Component                                         | Carried today?                             | What it encodes                                        | Pays off only with…                   |
| ------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------ | ------------------------------------- |
| Action priors                                     | ✅ chromosome A                             | Coarse action biases                                   | — (already a gene)                    |
| Q-network weights                                 | ✅ Lamarckian                               | The learned value estimates                            | optimizer + experience to *keep* them |
| Optimizer state (Adam moments)                    | ❌                                          | The *trajectory* of learning (per-param step scale)    | inherited weights to continue from    |
| Replay buffer contents                            | ❌ (only the count is in `get_model_state`) | The *experience* that justifies the weights            | continued training to consolidate     |
| Plasticity state (ε, learning rate, `step_count`) | ⚠️ partial                                 | How aggressively the child overwrites what it inherits | inherited weights worth protecting    |


This table explains result #2 mechanically. Lamarckian copies weights into a
**fresh optimizer + empty replay buffer + reset exploration**. The child has
the parent's answer but none of the parent's momentum or supporting data, and
it explores/learns like a blank newborn — so its first noisy updates pull the
inherited weights toward whatever its first few transitions say. Weights
without their continuation machinery get washed out before they can pay off.
That is the hypothesis the new variant ladder is built to test.

## Re-derived variant ladder

Variants are now ordered by **"weights plus the minimal machinery that lets
weights survive and matter,"** not by raw payload size. Each step is a
falsifiable test of *why* the previous step was a null.

- **P0 — Baseline (chromosome only).** Current Baldwinian default. Control.
- **P1 — Weights only.** Current Lamarckian warm-start. Already ≈ null;
retained as the control that establishes "weights alone don't help."
- **P2 — Weights + plasticity damping.** Copy weights *and* lower the child's
initial learning rate and/or exploration (and carry `step_count`) so the
inheritance is not immediately overwritten. Tests the "washed-out"
explanation for P1's null without any new payload bytes.
- **P3 — Weights + continuation machinery.** Copy weights + optimizer state +
a bounded slice of the parent's replay buffer, so the child *continues* the
parent's learning trajectory instead of restarting it. This is the
correctly-specified "module-state micro-transfer."
- **P4 — Gated / blended transfer.** Any of the above, but (i) applied only
when the parent clears a fitness/confidence gate, and (ii) blended
`θ_child = α·θ_parent + (1−α)·θ_init` (soft warm-start) to bound
local-niche lock-in.

The earlier "V1 priors" rung is dropped: it duplicates chromosome-A
inheritance (result #3).

## Precondition gate: is there any signal to inherit?

Because of result #1, none of P1–P4 can beat baseline in a regime where the
parent's trained policy barely outperforms a fresh one. So #848 has a hard
**precondition** before any payload work:

> **Transferable-signal budget.** Measure how much an end-of-life policy
> outperforms a freshly-initialized policy *on the same observations*. Use the
> greedy-evaluation probe-state idea from the 05-16 devlog: cache a fixed set
> of observation tensors, freeze each agent's Q-net at intervals, and report
> argmax/max-Q drift over a lifetime. If end-of-life policies do not measurably
> beat init, **stop** — richer inheritance cannot help here, and the result is
> a clean negative for #848 in that regime.

The natural place to find a non-trivial budget is exactly the regime the 05-16
devlog flagged: **longer simulations on smaller populations**, where per-agent
gradient budgets stay high and the cohort-noise floor drops. #848 experiments
should run there, not on the default 1000-step / large-population config where
learning is known to be ~null.

## Comparative experiment design

Reuse the existing inheritance-A/B machinery
(`scripts/run_inheritance_mode_ab.py`, `scripts/compare_inheritance_arms.py`,
protocol in [inheritance_mode_ab.md](../experiments/intrinsic_evolution/inheritance_mode_ab.md))
and add the `inheritance_mode` literals for P2–P4.

1. **Run only in a learning-positive regime** (long horizon, small population),
  confirmed by the precondition gate above.
2. **Matched conditions:** identical seed cohorts, initial-conditions profiles,
  and policy knobs across variants; vary only the payload.
3. **Cross-ecology transfer test (the overfit discriminator).** Evaluate
  descendants in *shifted* ecologies (resource density, regeneration regime,
   crowding) to separate genuine adaptation from lock-in to the parent's local
   niche. This is the one piece of the old plan that survives intact — it is
   the only way to operationalize "helps vs overfits to local ecology."
4. **Replication:** every condition is a multi-seed distribution (the 6-seed ×
  3-profile cohort pattern), never a single trajectory.

## Trade-off metrics and decision rule

Primary readouts (early-life, where any inheritance effect must appear first):

- **Adaptation speed:** steps-to-threshold for reproduction/resource stability
after spawn; net early RL reward at ages N ∈ {10, 25, 50} (the
fitness-relevant channel from the 06-04 analysis).
- **Stability:** startup death rate, population/resource oscillation amplitude,
lineage churn.
- **Generalization:** performance retention when descendants are moved from the
source ecology to a held-out ecology.

Use the project's standard robustness gate everywhere: **paired (per
profile×seed) delta with 95% CI excluding zero AND within-profile sign
agreement ≥ 75%.**

**Decision rule.** Escalate P1 → P2 → P3 → P4 *cheap-first*, and keep a richer
payload only if it produces a robust early-life **net-RL-reward** gain that
does **not** degrade cross-ecology generalization or stability. Given P1 is
already a null, the burden of proof is on each richer rung to show the missing
continuation machinery (P2/P3) or gating (P4) is what converts inherited
weights into fitness — not just into a behavioral nudge.

## Implementation prerequisites

These are gaps that must close before P2–P4 are runnable:

- **Expose and load the missing module state**
([#901](https://github.com/Dooders/AgentFarm/issues/901)). `get_model_state()`
/ `load_model_state()` in `farm/core/decision/algorithms/tianshou.py`
currently round-trip only `policy_state_dict` and `step_count`. P2 needs
learning rate / ε control on the child; P3 needs optimizer state and a
bounded replay-buffer slice.
- **Bounded, deterministic transfer** ✅ **COMPLETE** 
([#902](https://github.com/Dooders/AgentFarm/issues/902)). Replay-buffer
slice transfer is now available via `PrioritizedReplayBuffer.get_transfer_slice()`
and `load_transfer_slice()`. The transfer is size-capped (configurable via
`max_size` parameter), seed-deterministic (reproducible under `PYTHONHASHSEED=0`),
and integrated with `TianshouWrapper.get_model_state(include_replay_buffer=True,
replay_buffer_limit=N)`. See `tests/decision/test_replay_buffer_transfer.py`
for usage examples and determinism validation.
- **New `inheritance_mode` values**
([#903](https://github.com/Dooders/AgentFarm/issues/903)). Extend the
`InheritanceMode = Literal["baldwinian", "lamarckian"]` union in
`farm/runners/intrinsic_evolution_experiment.py` for P2–P4, keeping
`baldwinian` the default.

## Summary

The first-principles shift from the old plan: stop ranking payloads by size,
and start asking *what makes an inherited policy keep mattering*. The evidence
says copying weights alone is a null and copying priors is redundant with the
chromosome, so the live hypotheses are (a) inherited weights are washed out for
lack of continuation machinery (P2/P3), and (b) any benefit is fragile to
local-niche overfitting unless gated/blended (P4) — all of it gated behind a
precondition that there is measurable learned signal to inherit in the first
place.

## Related docs

- [Hyperparameter chromosome design](hyperparameter_chromosome.md)
- [Inheritance A/B experiment protocol](../experiments/intrinsic_evolution/inheritance_mode_ab.md)
- [Devlog: Baldwinian vs Lamarckian A/B](../devlog/2026-05-21-baldwinian-vs-lamarckian-ab-harness.md)
- [Devlog: Are we measuring at the wrong level?](../devlog/2026-06-04-are-we-measuring-at-the-wrong-level.md)
- [Devlog: Is the DQN actually learning?](../devlog/2026-05-16-is-the-dqn-actually-learning.md)
