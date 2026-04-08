# Crossover Strategies: Design Note

*Related issues: [#611](https://github.com/Dooders/AgentFarm/issues/611) (crossover operators), [#612](https://github.com/Dooders/AgentFarm/issues/612) (child initialization), [#613](https://github.com/Dooders/AgentFarm/issues/613) (this note).*

---

## 1. Overview

The `farm.core.decision.training.crossover` module provides three strategies for combining two Q-network state dicts (float or quantized) into a child network.  Crossover enables evolutionary or multi-parent search after PTQ / QAT rather than only mixing float weights during training.

All three strategies accept paired state dicts with **identical keys and tensor shapes**, dequantize any `torch.qint8` tensors to `float32` before operating, and return a `float32` offspring state dict compatible with `nn.Module.load_state_dict`.

---

## 2. Strategies

### 2.1 `random` — per-tensor selection

For each parameter tensor independently, flip a (possibly biased) coin and take the tensor from parent A (probability `alpha`) or parent B (probability `1 - alpha`).

```python
child_sd = crossover_quantized_state_dict(
    state_dict_a, state_dict_b,
    mode="random",
    seed=42,          # for reproducibility
    alpha=0.5,        # default: 50/50
)
```

**Determinism**: fully reproducible given `seed` or a seeded `np.random.Generator`.

**Edge cases**:
- `alpha=1.0` → exact copy of parent A.
- `alpha=0.0` → exact copy of parent B.

### 2.2 `layer` — layer-group alternation

Parameters are first grouped by the first two name segments (for example, `network.0`, `network.1`, `network.4`, `network.5`, …) and then merged into logical blocks before alternating parents. In the current network layout, this means `network.0` + `network.1` form block 0, `network.4` + `network.5` form block 1, `network.8` forms block 2, and so on. Even-numbered blocks come from parent A, odd-numbered blocks come from parent B. This keeps each `Linear` layer's weight and bias aligned with the following LayerNorm parameters from the same parent, avoiding inconsistent feature scaling.

```python
child_sd = crossover_quantized_state_dict(
    state_dict_a, state_dict_b,
    mode="layer",
)
```

**Determinism**: fully deterministic (no RNG).

### 2.3 `weighted` — parameter-wise averaging

For every aligned tensor compute `child = alpha * a + (1 - alpha) * b` in `float32`.

```python
child_sd = crossover_quantized_state_dict(
    state_dict_a, state_dict_b,
    mode="weighted",
    alpha=0.5,  # midpoint blend
)
```

**Determinism**: fully deterministic (arithmetic only).

**Edge cases**:
- `alpha=1.0` → exact copy of parent A.
- `alpha=0.0` → exact copy of parent B.

---

## 3. High-level API

`initialize_child_from_crossover` is the single entry point that resolves parents (live models, checkpoint paths, or state dicts), infers architecture, instantiates a fresh child, runs crossover, loads the state dict, and returns the child in `eval()` mode:

```python
from farm.core.decision.training.crossover import initialize_child_from_crossover

child = initialize_child_from_crossover(
    parent_a,          # nn.Module, path, or state dict
    parent_b,
    strategy="weighted",
    alpha=0.7,
)
out = child(state_batch)
```

---

## 4. Experimental Setup

The numbers in Section 5 were produced by:

| Parameter      | Value |
|----------------|-------|
| `input_dim`    | 8     |
| `hidden_size`  | 64    |
| `output_dim`   | 4     |
| `seed_a`       | 0     |
| `seed_b`       | 1     |
| `state_seed`   | 42    |
| `n_states`     | 256   |
| `n_repeats`    | 20    |
| `alpha`        | 0.5   |
| Hardware       | CPU   |

**Quality reference**: parent A's Q-values (float32) on the fixed 256-state batch.

**Metrics**:

| Metric | Definition |
|--------|-----------|
| `mean_q_error` | Mean absolute difference between child and reference Q-values, averaged across states and actions |
| `max_q_error`  | Maximum absolute difference across all (state, action) pairs |
| `action_agreement` | Fraction of states where `argmax` of child Q-values matches `argmax` of reference Q-values (higher = more similar to parent A) |
| `mean_time_ms` | Mean wall-clock milliseconds for `crossover_quantized_state_dict` + `load_state_dict`, averaged over `n_repeats` |

**To regenerate**:

```bash
# From the repository root
source venv/bin/activate
python scripts/benchmark_crossover.py --n-repeats 20 --output-csv reports/crossover_bench.csv

# Or via pytest (slow marker required)
pytest tests/decision/test_crossover_performance.py -m slow -v -s
```

---

## 5. Results

> **Note**: The values below are reference numbers produced on a standard development CPU.
> Re-run `scripts/benchmark_crossover.py` to get numbers for your hardware.

| Strategy   | Alpha | Time (ms) | Mean Q Err | Max Q Err | Act. Agree |
|------------|-------|----------:|----------:|----------:|----------:|
| `random`   | N/A   | 0.385     | 0.8350     | 3.2098    | 0.383      |
| `layer`    | N/A   | 0.326     | 0.0000     | 0.0000    | 1.000      |
| `weighted` | 0.5   | 0.501     | 0.6045     | 2.7700    | 0.461      |

> Numbers produced by `python scripts/benchmark_crossover.py` (20 repeats, CPU).
> Re-run to get hardware-specific values; results may vary.

**Note on `layer` metrics**: The zero Q-error and perfect action agreement for the `layer` strategy in this benchmark is an artifact of the untrained model setup used in the experiment.  `BaseQNetwork._initialize_weights` only re-initialises `nn.Linear` parameters via Xavier init; `nn.LayerNorm` layers keep their PyTorch defaults (weight=1, bias=0) regardless of the random seed.  In the benchmark, logical block 0 (`network.0` + `network.1`, the first Linear + its LayerNorm) and block 2 (`network.8`, the output Linear) are both even-indexed and therefore come from parent A, while block 1 (`network.4` + `network.5`, the second Linear + its LayerNorm) is odd-indexed and comes from parent B.  Because LayerNorm parameters are identical across parents in an untrained model, and parent A's Linear layers dominate the Q-value output in this synthetic setup, the child happens to be functionally equivalent to parent A here.  On trained models where all parameters—including LayerNorm weight and bias—are distinct across parents, the Q-error and action-agreement metrics will reflect genuine blending from both parents.

---

## 6. Tradeoff Interpretation

**`random`** — maximises offspring *diversity*: each parameter tensor is independently drawn from either parent, so children can explore a wide range of policy combinations.  The downside is high *variance*: the child's quality relative to both parents is unpredictable.  Good for population-based search where diversity is the objective.

**`layer`** — preserves *structural coherence*: all parameters within a layer block (weight, bias, LayerNorm scale/shift) always come from the same parent.  This avoids the representational inconsistency of mixing, say, a weight from parent A's distribution with a LayerNorm learned for parent B's activations.  Tradeoff: only two possible children per parent pair (up to group-order symmetry), so it produces less diversity than `random`.

**`weighted`** — provides *smooth interpolation*: at `alpha=0.5` the child sits at the arithmetic midpoint of the two parents in weight space.  This can smooth sharp features and reduce maximum Q-error relative to both parents, but may also blur distinctive policy structure from either parent.  Good for model ensembling or as a warm-start for further training.  The child's quality degrades gracefully as `alpha` moves away from 0 or 1.

---

## 7. Test Coverage

| Test file | Markers | Content |
|-----------|---------|---------|
| `tests/decision/test_crossover.py` | (default) | Correctness / regression: all three modes on synthetic fixtures, edge cases (`alpha=0/1`), quantized inputs, round-trip forward pass, `crossover_checkpoints`, `initialize_child_from_crossover` |
| `tests/decision/test_crossover_performance.py` | (default + `slow`) | Smoke checks (default run), wall-clock + quality benchmarks (`--m slow`), diversity check, strategy comparison summary |

---

## 9. Post-Crossover Fine-tuning and QAT

After crossover produces a float32 child, the child is often fine-tuned against a frozen reference model (one of the parents, or the distilled teacher) to recover performance.  See `farm/core/decision/training/finetune.py` and `scripts/finetune_child.py`.

### When to use float fine-tuning vs QAT fine-tuning

| Scenario | Recommended mode |
|----------|-----------------|
| Child will be deployed in **float32** | `quantization_applied="none"` (default) |
| Child will be converted to **int8** via `quantize_dynamic` | `quantization_applied="ptq_dynamic"` |
| Parents were statically quantized before crossover | `quantization_applied="ptq_static"` |
| Parents were QAT-trained float checkpoints (pre-convert) | `quantization_applied="qat_float"` |

**Why this matters**: Crossover produces float32 children even when the parents were int8.  If the child is fine-tuned in full float32 but then converted to int8 for deployment, the optimiser adapted to float noise instead of quantization noise.  QAT-aware fine-tuning inserts `WeightOnlyFakeQuantLinear` (STE, weight-only, same scope as `QATTrainer`) so the loss is minimised under the same approximation error as int8 inference.

### Recipe for QAT-aware fine-tuning

```python
from farm.core.decision.training.finetune import FineTuningConfig, FineTuner
from farm.core.decision.training.crossover import crossover_quantized_state_dict

# 1. Crossover (parents may have been quantized)
child_sd = crossover_quantized_state_dict(sd_a, sd_b, mode="weighted")
child.load_state_dict(child_sd)

# 2. QAT fine-tune (fake-quant weights during training)
cfg = FineTuningConfig(
    quantization_applied="ptq_dynamic",  # or "ptq_static" / "qat_float"
    epochs=10,
    learning_rate=1e-4,
)
tuner = FineTuner(reference=parent_a, child=child, config=cfg)
metrics = tuner.finetune(states, checkpoint_path="child_qat.pt")

# 3. Convert and save as int8 (PTQ-compatible format)
q_model = tuner.convert()
tuner.save_quantized(q_model, "child_qat_int8.pt")
```

From the CLI:

```bash
python scripts/finetune_child.py \
    --parent-a-ckpt checkpoints/parent_a.pt \
    --parent-b-ckpt checkpoints/parent_b.pt \
    --quantization-applied ptq_dynamic \
    --epochs 10 --lr 1e-4
```

The script automatically calls `convert()` + `save_quantized()` and writes
`child_finetuned_qat_int8.pt` (and companion JSON) alongside the float QAT
checkpoint.

### Validating the int8 output

Use the existing `compare_outputs` helper or `scripts/validate_quantized.py`
to check fidelity between the float QAT child and the converted int8 model:

```python
from farm.core.decision.training.quantize_ptq import compare_outputs
result = compare_outputs(tuner._active_child, q_model, states)
```

See also `farm/core/decision/training/quantize_ptq.py` (`QuantizedValidator`) for
full JSON reports with fidelity / latency / size sections.

### Related issues

- Parent epic: [Dooders/AgentFarm#8](https://github.com/Dooders/AgentFarm/issues/8) – Distillation, Quantization, and Crossover pipeline.
- Implementation: `farm/core/decision/training/finetune.py` (`FineTuner`, `FineTuningConfig`, `QUANTIZATION_APPLIED_MODES`).
- QAT building blocks: `farm/core/decision/training/quantize_qat.py` (`WeightOnlyFakeQuantLinear`, `QATTrainer`).

---

## 8. References

- Implementation: `farm/core/decision/training/crossover.py`
- Training package exports: `farm/core/decision/training/__init__.py`
- Benchmark script: `scripts/benchmark_crossover.py`
- Related validation patterns: `scripts/validate_quantized.py`, `farm/core/decision/training/quantize_ptq.py`
