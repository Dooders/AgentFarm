# Crossover + Fine-tune Search Space

*Related issues: [#8](https://github.com/Dooders/AgentFarm/issues/8) (Distillation, Quantization, and Crossover pipeline).*

*See also: [`crossover_strategies.md`](crossover_strategies.md) for strategy semantics.*

---

## 1. Overview

This document specifies the search space used by `scripts/run_crossover_search.py` (backed by `farm.core.decision.training.crossover_search`) to find the **Pareto-optimal crossover + fine-tune strategy** for Q-network recombination.

The search is a **Cartesian product** of crossover recipes × fine-tune regimes.  Each combination yields one child network that is evaluated with a fixed harness and scored on a primary metric.

---

## 2. Primary Metric

```
primary_metric = min(child_vs_parent_a_agreement, child_vs_parent_b_agreement)
```

*Higher is better*: a child that achieves high agreement with **both** parents simultaneously is a well-blended offspring.  Taking the minimum penalises children that collapse to one parent.

Additional metrics (KL divergence, MSE, MAE, cosine similarity, oracle agreement) are stored in the leaderboard for secondary analysis.

A child is flagged **degenerate** when `primary_metric < degenerate_threshold` (default 0.0, disabled; set to e.g. 0.3 to flag poor blends).

---

## 3. Crossover Knobs

| Knob | Values (default search) | Notes |
|------|------------------------|-------|
| `mode` | `random`, `layer`, `weighted` | All three strategies explored |
| `alpha` (random) | 0.5 | Probability of selecting from parent A per tensor |
| `seed` (random) | 0, 1, 2 | Controls random tensor selection; three seeds give diversity estimate |
| `alpha` (weighted) | 0.3, 0.5, 0.7 | Linear blend weight; 0.5 = midpoint, 0.3/0.7 = parent-biased |
| `alpha` (layer) | — | Ignored; layer mode is fully deterministic |

### Strategy semantics (summary)

| Strategy | Diversity | Coherence | Key property |
|----------|-----------|-----------|-------------|
| `random` | High | Low | Per-tensor coin flip; high variance across seeds |
| `layer` | Low | High | Structural blocks preserved; only two possible children per pair |
| `weighted` | Medium | Medium | Smooth interpolation; predictable, no randomness |

For full semantics see [`crossover_strategies.md §2`](crossover_strategies.md#2-strategies).

---

## 4. Fine-tune Regimes

Each regime specifies a named hyperparameter set passed to `FineTuner` (using parent A as the reference teacher with a soft KL-divergence objective).

| Regime name | Epochs | LR | Batch size | Notes |
|-------------|--------|----|------------|-------|
| `short` | 5 | 1e-3 | 32 | Quick recovery; good baseline |
| `medium` | 10 | 5e-4 | 32 | Balanced; default for `minimal` grid |
| `long` | 20 | 1e-4 | 32 | Deeper adaptation; captures slower convergence |
| `lr_high` | 5 | 5e-3 | 32 | High-LR exploration; may overshoot on easy tasks |
| `short_qat` | 5 | 1e-4 | 16 | Same epoch budget as `short` but `quantization_applied="ptq_dynamic"` (fake-quant fine-tune) |

**Reference teacher**: parent A (frozen, eval mode).

**Loss function**: KL divergence with temperature=3.0 (soft distillation, pure soft loss α=1.0).

**QAT fine-tuning**: set `quantization_applied` to `"ptq_dynamic"` / `"ptq_static"` / `"qat_float"` in a custom regime to enable QAT-aware fine-tuning (see `crossover_strategies.md §8`).

---

## 5. Pre-defined Search Spaces

### 5.1 `default` (14 children = 7 recipes × 2 regimes)

```python
SearchConfig.default()
```

| # | Crossover recipe | Fine-tune regime |
|---|-----------------|-----------------|
| 1–2 | random, α=0.5, seed=0 | short, long |
| 3–4 | random, α=0.5, seed=1 | short, long |
| 5–6 | random, α=0.5, seed=2 | short, long |
| 7–8 | layer | short, long |
| 9–10 | weighted, α=0.3 | short, long |
| 11–12 | weighted, α=0.5 | short, long |
| 13–14 | weighted, α=0.7 | short, long |

### 5.2 `minimal` (9 children = 3 recipes × 3 regimes)

```python
SearchConfig.minimal()
```

One recipe per crossover mode; three fine-tune regimes.  Good for a fast first leaderboard.

| # | Crossover recipe | Fine-tune regime |
|---|-----------------|-----------------|
| 1–3 | random, α=0.5, seed=0 | short, medium, long |
| 4–6 | layer | short, medium, long |
| 7–9 | weighted, α=0.5 | short, medium, long |

### 5.3 `default-qat` (21 children) and `minimal-qat` (9 children)

Python:

- `SearchConfig.default_with_qat()` — same seven crossover recipes as **default**, but three fine-tune columns: **`short`** (float), **`long`** (float), **`short_qat`** (`quantization_applied="ptq_dynamic"`, weight-only fake quant during fine-tune per `crossover_strategies.md` §8).
- `SearchConfig.minimal_with_qat()` — three recipes × (`short`, `short_qat`, `long`).

CLI: `--search-space default-qat` or `minimal-qat`.

Custom mode can add the **`short_qat`** preset via `--finetune-regimes short_qat`.

### 5.4 Parallel execution (`--workers N`)

When `N > 1`, `run_crossover_search` uses `ProcessPoolExecutor`: each child runs in a **separate process**. Parent **state dicts** and the **states array** are written under `<run-dir>/.crossover_parallel_cache/` and reloaded per worker. **Requirements:** both parents must be **`BaseQNetwork`** with identical architecture. **Quantized** parent modules are not supported on this path (use `--workers 1`). **CPU** is used inside workers for broad compatibility.

Convenience: `make crossover-search-smoke` runs a two-child minimal search with synthetic states.

### 5.5 Custom

```bash
python scripts/run_crossover_search.py \
    --search-space custom \
    --crossover-modes random weighted \
    --alpha-values 0.3 0.5 0.7 \
    --crossover-seeds 0 1 2 \
    --finetune-regimes short long \
    --max-runs 12
```

---

## 6. Evaluation Harness

Every child is evaluated with the **same fixed harness**:

| Parameter | Value |
|-----------|-------|
| Evaluator | `RecombinationEvaluator` (`recombination_eval.py`) |
| State buffer | Fixed NumPy array (`--states-file` or synthetic; shared across all children) |
| Metrics | top-1 action agreement, top-k (k=1,2,3), KL divergence, MSE, MAE, cosine similarity, oracle agreement |
| Latency warmup | 3 forward passes |
| Latency repeats | 20 timed passes (median) |
| Baseline | Parent A vs Parent B comparison (informational, not threshold-checked) |

---

## 7. Reproducibility

Each child run writes a `run_config.json` capturing:

```json
{
  "child_id": "000_random_a0p50_s0_short",
  "crossover": { "mode": "random", "alpha": 0.5, "seed": 0 },
  "finetune": {
    "regime": "short", "epochs": 5, "lr": 0.001,
    "batch_size": 32, "val_fraction": 0.1,
    "loss_fn": "kl", "seed": 42,
    "quantization_applied": "none"
  },
  "finetune_metrics": { "..." : "..." },
  "torch_version": "2.x.y"
}
```

Re-running with the same `run_config.json` parameters, the same state buffer, and the same parent checkpoints produces identical rankings (within floating-point tolerance for deterministic crossover modes).  The `random` crossover mode is fully reproducible given its `seed`.

---

## 8. Budget Guidance

| Grid size | Children | Approx time (CPU, 1000 states) |
|-----------|----------|-------------------------------|
| Smoke (3 pairs) | 3 | < 2 min |
| Minimal (3×3) | 9 | 5–15 min |
| Minimal-qat (3×3) | 9 | longer (includes QAT fine-tune) |
| Default (7×2) | 14 | 10–25 min |
| Default-qat (7×3) | 21 | longer (includes QAT column) |
| Extended (7×4) | 28 | 20–50 min |

*Times are indicative for synthetic states on CPU.  Real state buffers (larger N) and longer fine-tune regimes scale proportionally.*

Use `--max-runs N` to cap the total number of children for CI or smoke tests.

---

## 9. Interpreting the Leaderboard

```
rank | child_id                | primary | agree_a | agree_b | degenerate
-----|-------------------------|---------|---------|---------|----------
   1 | 011_weighted_a0p50_...  | 0.8120  | 0.8500  | 0.8120  | False
   2 | 010_weighted_a0p30_...  | 0.7943  | 0.7943  | 0.8210  | False
   ...
 n+1 | parent_a (baseline)    | 0.6500  | 1.0000  | 0.6500  | —
 n+1 | parent_b (baseline)    | 0.6500  | 0.6500  | 1.0000  | —
```

- **`primary_metric`**: `min(agree_a, agree_b)` — the leaderboard sort key.
- **Parent baselines** (agreement = parent A vs parent B): provide context for the inter-parent diversity.  A child that exceeds the baseline on both sides is genuinely blending information from both parents.
- **Degenerate flag**: set when `primary_metric < degenerate_threshold`.  Degenerate children should be inspected (possible weight collapse or training instability).

---

## 10. Conclusions and Recommended Defaults

Run the search and read `recommendation.txt` for a data-driven recommendation.  Expected findings based on the strategy tradeoffs described in `crossover_strategies.md §6`:

| Scenario | Recommended default |
|----------|-------------------|
| Maximum diversity (population search) | `random`, α=0.5, 3+ seeds + `long` fine-tune |
| Structural coherence (stable convergence) | `layer` + `long` fine-tune |
| Smooth interpolation (ensembling / warm-start) | `weighted`, α=0.5 + `short` fine-tune |
| Unknown parents (default recommendation) | Run `minimal` grid first; pick top-ranked strategy |

These recommendations are updated by the search output; see `runs/crossover_search/recommendation.txt` after running the experiment.

---

## 11. References

- Implementation: `farm/core/decision/training/crossover_search.py`
- CLI runner: `scripts/run_crossover_search.py`
- Crossover operators: `farm/core/decision/training/crossover.py`
- Fine-tuning pipeline: `farm/core/decision/training/finetune.py`
- Evaluation harness: `farm/core/decision/training/recombination_eval.py`
- Strategy semantics: [`crossover_strategies.md`](crossover_strategies.md)
- Parent epic: [Dooders/AgentFarm#8](https://github.com/Dooders/AgentFarm/issues/8)
