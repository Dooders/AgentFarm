# Neural Recombination Runbook

*Step-by-step guide for repeating the full distillation → quantization → crossover → fine-tune → validation pipeline from [AgentFarm#8](https://github.com/Dooders/AgentFarm/issues/8).*

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Reference: Pipeline Order](#2-quick-reference-pipeline-order)
3. [Stage 1 — Distillation](#3-stage-1--distillation)
4. [Stage 2 — Post-Training Quantization (PTQ)](#4-stage-2--post-training-quantization-ptq)
5. [Stage 3 — Quantization-Aware Training (QAT) — Optional](#5-stage-3--quantization-aware-training-qat--optional)
6. [Stage 4 — Crossover + Fine-tuning](#6-stage-4--crossover--fine-tuning)
7. [Stage 5 — Validation](#7-stage-5--validation)
8. [Optional: Compare Distillation Modes](#8-optional-compare-distillation-modes)
9. [Parameter Reference](#9-parameter-reference)
10. [Tuning Guide](#10-tuning-guide)
11. [Copy-Paste Recipes](#11-copy-paste-recipes)
12. [Generalization: Holdout & Domain-Shift Evaluation](#12-generalization-holdout--domain-shift-evaluation)
13. [Publication Ablations](#13-publication-ablations)

---

## 1. Prerequisites

### Environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Python 3.8+ required; 3.9+ recommended. All scripts below assume the repo root is your working directory and that the venv is active.

### Parent checkpoints

Each pipeline run requires **two pre-trained parent `BaseQNetwork` state-dicts** — `parent_A.pt` and `parent_B.pt`.  These are the teachers for distillation.  Architecture dimensions (`input_dim`, `output_dim`, `hidden_size`) **must stay consistent across every stage.**

The default architecture shipped with `farm/config/default.yaml` is:

| Dimension | Default |
|-----------|---------|
| `input_dim` | 8 |
| `output_dim` | 4 |
| `hidden_size` / `parent_hidden` | 64 |

If your parents use different dimensions, pass the corresponding flags to **every** script.

### State buffer

All scripts share a common evaluation dataset: an `(N, input_dim)` float32 NumPy array.

- **Synthetic** (quickest): generated at runtime via `--n-states` + `--seed`; always reproducible but not representative of real agent behaviour.
- **Real replay buffer**: pass `--states-file path/to/states.npy`. The `.npy` file must have shape `(N, input_dim)` in float32. Using the same distribution as training/deployment gives the most realistic validation metrics.

> **Reproducibility rule:** use the same `--states-file` (or the same `--n-states`/`--seed` pair) across all stages so that metrics are comparable.

---

## 2. Quick Reference: Pipeline Order

```
parent_A.pt ─┐
             ├─► run_distillation.py ─► student_A.pt ─┐
parent_B.pt ─┘                          student_B.pt ─┘
                                               │
                              (PTQ) quantize_distilled.py ─► student_A_int8.pt
                         (QAT) qat_distilled.py    ─► student_A_qat_int8.pt
                                               │
                         finetune_child.py (crossover inside) ─► child_finetuned.pt
                                               │
                      validate_distillation.py · validate_quantized.py
                      validate_recombination.py
```

Each stage writes checkpoints and companion JSON metadata (`*.pt.json`) that become inputs for the next stage.

---

## 3. Stage 1 — Distillation

**Script:** `scripts/run_distillation.py`

**Goal:** train a smaller `StudentQNetwork` to reproduce the Q-value distribution of each frozen parent (`BaseQNetwork`).

### Minimal synthetic run

```bash
python scripts/run_distillation.py \
  --parent-a-ckpt checkpoints/parent_A.pt \
  --parent-b-ckpt checkpoints/parent_B.pt \
  --n-states 2000 \
  --seed 42 \
  --output-dir checkpoints/distillation
```

### With a real replay buffer

```bash
python scripts/run_distillation.py \
  --parent-a-ckpt checkpoints/parent_A.pt \
  --parent-b-ckpt checkpoints/parent_B.pt \
  --states-file data/replay_states.npy \
  --seed 42 \
  --output-dir checkpoints/distillation
```

Expected shape of `replay_states.npy`: `(N, 8)` float32 (or whatever your `input_dim` is).

### Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--pair` | `both` | `A`, `B`, or `both` |
| `--input-dim` | `8` | State feature dimension |
| `--output-dim` | `4` | Number of actions |
| `--parent-hidden` | `64` | Teacher hidden width |
| `--parent-a-ckpt` | _(required)_ | Path to `parent_A.pt` |
| `--parent-b-ckpt` | _(required)_ | Path to `parent_B.pt` |
| `--n-states` | `1000` | Synthetic state count (ignored if `--states-file`) |
| `--states-file` | — | `.npy` replay buffer path |
| `--temperature` | `3.0` | Softmax temperature for soft labels |
| `--alpha` | `1.0` | Soft/hard blend: `1.0` = pure soft KL, `0.0` = pure hard CE |
| `--epochs` | `10` | Training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch-size` | `32` | Mini-batch size |
| `--max-grad-norm` | `1.0` | Gradient clipping norm |
| `--val-fraction` | `0.1` | Held-out validation split |
| `--loss-fn` | `kl` | Soft loss: `kl` (recommended) or `mse` |
| `--seed` | `None` | RNG seed |
| `--output-dir` | `checkpoints/distillation` | Output directory |

### Outputs

```
checkpoints/distillation/
  student_A.pt          # StudentQNetwork state dict
  student_A.pt.json     # config + per-epoch metrics
  student_B.pt
  student_B.pt.json
```

---

## 4. Stage 2 — Post-Training Quantization (PTQ)

**Script:** `scripts/quantize_distilled.py`

**Goal:** compress `student_*.pt` to int8 with no re-training.  Start here before trying QAT.

### Minimal dynamic PTQ

```bash
python scripts/quantize_distilled.py \
  --checkpoint-dir checkpoints/distillation \
  --input-dim 8 --output-dim 4 --parent-hidden 64 \
  --mode dynamic \
  --output-dir checkpoints/quantized
```

### Static PTQ (with calibration data)

```bash
python scripts/quantize_distilled.py \
  --checkpoint-dir checkpoints/distillation \
  --states-file data/replay_states.npy \
  --mode static \
  --calibration-batches 10 \
  --calibration-batch-size 64 \
  --output-dir checkpoints/quantized
```

### Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--pair` | `both` | `A`, `B`, or `both` |
| `--checkpoint-dir` | — | Dir containing `student_A.pt` / `student_B.pt` |
| `--student-a-ckpt` / `--student-b-ckpt` | — | Explicit paths (override `--checkpoint-dir`) |
| `--input-dim` | `8` | Must match distillation |
| `--output-dim` | `4` | Must match distillation |
| `--parent-hidden` | `64` | Must match distillation |
| `--mode` | `dynamic` | `dynamic` (weight-only, no calibration) or `static` |
| `--dtype` | `qint8` | Quantization dtype |
| `--backend` | `auto` | `auto`, `x86`, `fbgemm`, `qnnpack` |
| `--calibration-batches` | `10` | Static mode: number of calibration batches |
| `--calibration-batch-size` | `64` | Static mode: batch size for calibration |
| `--states-file` | — | Calibration states (static mode); also used for output comparison |
| `--n-states` | `1000` | Synthetic calibration states if no file |
| `--seed` | `42` | RNG seed |
| `--output-dir` | `checkpoints/quantized` | Output directory |

### Outputs

```
checkpoints/quantized/
  student_A_int8.pt        # Quantized model (CPU int8 pickle)
  student_A_int8.pt.json   # QuantizationConfig + timing
  student_B_int8.pt
  student_B_int8.pt.json
```

---

## 5. Stage 3 — Quantization-Aware Training (QAT) — Optional

**Script:** `scripts/qat_distilled.py`

**When to use QAT instead of PTQ:**

> Use QAT when PTQ action agreement (reported by `validate_quantized.py`) falls below your target threshold (e.g. < 90%).  QAT adds a short training pass with fake quantization, recovering accuracy at the cost of a few extra minutes.

### Minimal QAT run

```bash
python scripts/qat_distilled.py \
  --checkpoint-dir checkpoints/distillation \
  --input-dim 8 --output-dim 4 --parent-hidden 64 \
  --epochs 5 \
  --n-states 2000 \
  --seed 42 \
  --output-dir checkpoints/qat
```

Omit `--no-convert` (default) to also produce the converted int8 checkpoint.

### Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--pair` | `both` | `A`, `B`, or `both` |
| `--checkpoint-dir` | — | Dir with `parent_<pair>.pt` + `student_<pair>.pt` |
| `--teacher-a-ckpt` / `--student-a-ckpt` | — | Explicit path overrides |
| `--input-dim` | `8` | Must match distillation |
| `--output-dim` | `4` | Must match distillation |
| `--parent-hidden` | `64` | Must match distillation |
| `--epochs` | `5` | QAT fine-tuning epochs |
| `--learning-rate` | `1e-4` | Adam LR (lower than distillation; model is already trained) |
| `--batch-size` | `32` | Mini-batch size |
| `--max-grad-norm` | `1.0` | Gradient clipping norm |
| `--val-fraction` | `0.1` | Validation split |
| `--loss-fn` | `mse` | `mse` (default for QAT) or `kl` |
| `--temperature` | `3.0` | Temperature for `kl` mode |
| `--alpha` | `1.0` | Soft/hard blend for `kl` mode |
| `--no-convert` | `False` | Skip int8 conversion; save only float QAT checkpoint |
| `--states-file` / `--n-states` / `--seed` | — | Same semantics as other stages |
| `--output-dir` | `checkpoints/qat` | Output directory |

### Outputs

```
checkpoints/qat/
  student_A_qat.pt           # Float QAT checkpoint
  student_A_qat.pt.json
  student_A_qat_int8.pt      # Converted int8 (same format as PTQ output)
  student_A_qat_int8.pt.json
```

---

## 6. Stage 4 — Crossover + Fine-tuning

**Script:** `scripts/finetune_child.py`

**Goal:** blend two parent state dicts into a child via a crossover strategy, then fine-tune the child against a frozen reference (parent A) using a distillation-style loss.

### Minimal synthetic run

```bash
python scripts/finetune_child.py \
  --parent-a-ckpt checkpoints/parent_A.pt \
  --parent-b-ckpt checkpoints/parent_B.pt \
  --crossover-mode weighted \
  --crossover-alpha 0.5 \
  --crossover-seed 42 \
  --n-states 2000 \
  --seed 42 \
  --output-dir checkpoints/finetune
```

### With a replay buffer and YAML overrides

```bash
python scripts/finetune_child.py \
  --parent-a-ckpt checkpoints/parent_A.pt \
  --parent-b-ckpt checkpoints/parent_B.pt \
  --crossover-mode random \
  --crossover-alpha 0.5 \
  --crossover-seed 0 \
  --states-file data/replay_states.npy \
  --config-yaml farm/config/default.yaml \
  --epochs 10 \
  --lr 5e-4 \
  --output-dir checkpoints/finetune
```

### YAML defaults (`farm/config/default.yaml`)

The `crossover_child_finetune` section provides all fine-tuning defaults:

```yaml
crossover_child_finetune:
  learning_rate: 0.001
  epochs: 5
  batch_size: 32
  max_grad_norm: 1.0
  val_fraction: 0.1
  seed: null
  loss_fn: kl
  temperature: 3.0
  temp_decay: 1.0          # per-epoch temperature multiplier; 1.0 = no decay
  alpha: 1.0               # soft/hard blend (1.0 = pure KL, 0.0 = pure CE)
  lr_schedule_patience: 0  # ReduceLROnPlateau patience; 0 = disabled
  lr_schedule_factor: 0.5  # LR reduction factor when plateau detected
  quantization_applied: none   # none | ptq_dynamic | ptq_static | qat_float
  optimizer: adam
  optimizer_kwargs: {}
  early_stopping_patience: 0   # 0 = disabled
```

CLI flags such as `--lr`, `--epochs`, `--alpha` override the YAML values when specified.

### Key parameters

| Flag | Default (YAML) | Description |
|------|----------------|-------------|
| `--input-dim` | `8` | Must match parent architecture |
| `--output-dim` | `4` | Must match parent architecture |
| `--hidden-size` | `64` | Must match parent architecture |
| `--parent-a-ckpt` | _(required)_ | Parent A checkpoint (also the fine-tune teacher) |
| `--parent-b-ckpt` | _(required)_ | Parent B checkpoint |
| `--crossover-mode` | _(required)_ | `random`, `layer`, or `weighted` |
| `--crossover-alpha` | — | Blend/selection coefficient (see below) |
| `--crossover-seed` | — | RNG seed for `random` mode |
| `--n-states` / `--states-file` | — | State buffer (same as other stages) |
| `--config-yaml` | `farm/config/default.yaml` | YAML with `crossover_child_finetune` section |
| `--lr` | `1e-3` | Adam learning rate |
| `--epochs` | `5` | Fine-tuning epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--max-grad-norm` | `1.0` | Gradient clipping norm |
| `--val-fraction` | `0.1` | Validation split |
| `--loss-fn` | `kl` | Distillation loss (`kl` or `mse`) |
| `--temperature` | `3.0` | Softmax temperature |
| `--alpha` | `1.0` | Soft/hard blend |
| `--lr-patience` | `0` | ReduceLROnPlateau patience (0 = off) |
| `--lr-factor` | `0.5` | LR reduction factor |
| `--seed` | `null` | Fine-tuning RNG seed (separate from crossover seed) |
| `--quantization-applied` | `none` | `none`, `ptq_dynamic`, `ptq_static`, or `qat_float` |
| `--optimizer` | `adam` | `adam`, `adamw`, `sgd`, or `rmsprop` |
| `--early-stopping-patience` | `0` | Validation-loss patience (0 = off) |
| `--output-dir` | _(required)_ | Output directory |

### Crossover modes

| Mode | `alpha` meaning | Deterministic? | Notes |
|------|-----------------|----------------|-------|
| `random` | Probability of selecting from parent A per tensor | No (needs `--crossover-seed`) | High diversity; use multiple seeds to estimate variance |
| `layer` | Ignored | Yes | Even blocks from A, odd blocks from B; structurally coherent |
| `weighted` | Linear blend weight: `child = alpha*A + (1-alpha)*B` | Yes | Smooth interpolation; `0.5` = midpoint |

### Outputs

```
checkpoints/finetune/
  child.pt                  # Raw crossover child (pre fine-tune)
  child.pt.json
  child_finetuned.pt        # Fine-tuned child
  child_finetuned.pt.json
```

---

## 7. Stage 5 — Validation

### 7.1 Distillation quality

**Script:** `scripts/validate_distillation.py`

Checks KL divergence, MSE, MAE, cosine similarity, top-k agreement, latency ratio, and robustness slices between parent and student.

```bash
python scripts/validate_distillation.py \
  --checkpoint-dir checkpoints/distillation \
  --parent-a-ckpt checkpoints/parent_A.pt \
  --parent-b-ckpt checkpoints/parent_B.pt \
  --n-states 2000 --seed 42 \
  --report-dir reports/distillation
```

Key threshold flags (all have sensible defaults; override to tighten):

| Flag | Default | Description |
|------|---------|-------------|
| `--min-action-agreement` | `0.8` | Minimum top-1 action agreement |
| `--max-kl-divergence` | `0.1` | Maximum KL divergence |
| `--max-mse` | `0.01` | Maximum mean-squared error |
| `--min-cosine-similarity` | `0.9` | Minimum cosine similarity |
| `--max-latency-ratio` | `2.0` | Maximum student/parent latency ratio |
| `--min-robustness-action-agreement` | `0.7` | Agreement on noisy/out-of-distribution slices |

### 7.2 Quantization fidelity

**Script:** `scripts/validate_quantized.py`

Compares float student vs int8 student on agreement, Q-error, latency, and memory.

```bash
python scripts/validate_quantized.py \
  --float-dir checkpoints/distillation \
  --quant-dir checkpoints/quantized \
  --n-states 2000 --seed 42 \
  --report-dir reports/quantized
```

Key threshold flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--min-action-agreement` | `0.9` | Quantized vs float top-1 agreement |
| `--max-mean-q-error` | `0.05` | Mean absolute Q-value error |
| `--min-cosine-similarity` | `0.95` | Cosine similarity |
| `--max-latency-ratio` | `1.5` | Int8 / float latency ratio |

### 7.3 Recombination quality

**Script:** `scripts/validate_recombination.py`

Evaluates the fine-tuned child against both parents.

```bash
python scripts/validate_recombination.py \
  --checkpoint-dir checkpoints/finetune \
  --parent-a-ckpt checkpoints/parent_A.pt \
  --parent-b-ckpt checkpoints/parent_B.pt \
  --child-ckpt checkpoints/finetune/child_finetuned.pt \
  --n-states 2000 --seed 42 \
  --include-parent-baseline \
  --report-dir reports/recombination
```

For quantized children add `--child-quantized` (and `--parent-a-quantized` / `--parent-b-quantized` if parents are also int8).

Key threshold flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--min-action-agreement` | `0.7` | Child vs parent action agreement |
| `--max-kl-divergence` | `0.3` | Child vs parent KL |
| `--max-mse` | `0.05` | Child vs parent MSE |
| `--min-cosine-similarity` | `0.8` | Child vs parent cosine similarity |

> **"Good enough" heuristic:** aim for child-vs-A and child-vs-B top-1 agreement both ≥ 0.7, with neither collapsing to one parent.  The `primary_metric = min(agreement_A, agreement_B)` used by `run_crossover_search.py` captures this directly.

### Validation report layout

Each script writes a JSON report alongside a human-readable markdown summary.  JSON keys of interest:

- `action_agreement` — top-1 agreement between models
- `oracle_action_agreement` — agreement when both models are uncertain
- `kl_divergence`, `mse`, `mae`, `cosine_similarity` — Q-value closeness metrics
- `latency_ms_median` / `latency_ratio` — speed comparison

---

## 8. Optional: Compare Distillation Modes

**Script:** `scripts/compare_distillation_modes.py`

Runs hard-only (`alpha=0`), soft-only (`alpha=1`), and blended distillation back-to-back with a shared frozen teacher and state buffer so results are directly comparable.

```bash
python scripts/compare_distillation_modes.py \
  --seed 42 --epochs 10 --n-states 2000 \
  --json-out reports/distillation_mode_comparison.json
```

See [`docs/distillation_soft_label_comparison.md`](../distillation_soft_label_comparison.md) for recorded results and discussion.

---

## 9. Parameter Reference

### Architecture consistency

The following dimensions **must match** across all stages:

| Parameter | CLI flag (all scripts) | YAML key | Notes |
|-----------|----------------------|----------|-------|
| State feature dim | `--input-dim` | — | Parent network input size |
| Action count | `--output-dim` | — | Parent network output size |
| Teacher hidden width | `--parent-hidden` / `--hidden-size` | — | Used to reconstruct `BaseQNetwork` |

### Seed inventory

| Stage | Flag | Purpose |
|-------|------|---------|
| Distillation | `--seed` | State generation + training RNG |
| Quantization (PTQ) | `--seed` | Synthetic calibration state generation |
| QAT | `--seed` | Synthetic state generation + training RNG |
| Crossover | `--crossover-seed` | Per-tensor selection RNG (`random` mode) |
| Fine-tuning | `--seed` | Training batch shuffle + dropout |

### `crossover_child_finetune` YAML keys

| Key | Type | Safe range | Description |
|-----|------|-----------|-------------|
| `learning_rate` | float | `5e-5` – `5e-3` | Adam LR |
| `epochs` | int | `3` – `20` | Training epochs |
| `batch_size` | int | `16` – `128` | Mini-batch size |
| `max_grad_norm` | float | `0.5` – `5.0`; `0` = off | Gradient clipping |
| `val_fraction` | float | `0.05` – `0.2` | Held-out validation fraction |
| `seed` | int or null | any | `null` = non-deterministic |
| `loss_fn` | `kl` / `mse` | — | `kl` preferred for soft distillation |
| `temperature` | float | `1.0` – `10.0` | Softmax temperature for `kl` loss |
| `temp_decay` | float | `0.9` – `1.0` | Per-epoch temperature multiplier |
| `alpha` | float | `0.0` – `1.0` | Soft/hard blend; `1.0` = pure soft |
| `lr_schedule_patience` | int | `0` – `5` | ReduceLROnPlateau; `0` = off |
| `lr_schedule_factor` | float | `0.1` – `0.9` | LR multiplier on plateau |
| `quantization_applied` | string | see below | Triggers QAT-aware fine-tune path |
| `optimizer` | string | `adam`, `adamw`, `sgd`, `rmsprop` | Optimizer choice |
| `early_stopping_patience` | int | `0` – `10` | Val-loss patience; `0` = off |

`quantization_applied` values: `none` (default, float path), `ptq_dynamic`, `ptq_static`, `qat_float` — when not `none`, `FineTuner` replaces `Linear` layers with `WeightOnlyFakeQuantLinear`; call `convert()` + `save_quantized()` after `finetune()` for int8 output.

---

## 10. Tuning Guide

### State data choice

- **Use a real replay buffer** whenever possible.  Synthetic standard-normal states cover the full input range but may not reflect the distribution your agents encounter at inference time.  Validation metrics on synthetic states can be optimistic.
- The `.npy` file should be float32, shape `(N, input_dim)`.  Prefer N ≥ 2 000 for stable statistics; 10 000+ for final validation.
- **Keep the same states file across all stages** (or at least the same `--seed`) so that reported metrics are on a common distribution.

### Distillation

- **Temperature (`--temperature`, default `3.0`):** higher values flatten the teacher's distribution, making inter-action confidence ordering more visible.  Range `2.0`–`6.0` is typical; reduce toward `1.0` if the student converges too slowly.
- **Alpha (`--alpha`, default `1.0`):** `1.0` = pure KL soft loss; `0.0` = pure cross-entropy on the argmax.  Start with `1.0`; blend toward `0.5` if action agreement is high but hard-label accuracy is poor.
- **Learning rate instability:** if training loss oscillates or diverges, lower `--lr` (try `5e-4`) and ensure `--max-grad-norm 1.0` is in effect.
- **Low agreement after training:** increase `--epochs` (try `20`), or provide a larger / more representative state buffer.
- **`--loss-fn mse`:** simpler objective; useful for debugging, but KL is generally better for Q-value distributions.

### Post-Training Quantization

- **Try dynamic PTQ first** — zero training cost, typically ≥ 90 % action agreement.  Only invest in static PTQ or QAT if `validate_quantized.py` reports agreement below your target.
- **Static PTQ:** requires calibration data that reflects real input distribution; use `--states-file`.  Use `--calibration-batches 10`–`50` and `--calibration-batch-size 64`–`256`.
- **Backend choice:** on Intel CPUs `fbgemm` is often fastest; on ARM/mobile use `qnnpack`; `auto` selects automatically.  Benchmark with `--throughput-batch-size` in `validate_quantized.py`.
- **QAT vs PTQ decision rule:** if PTQ `action_agreement < 0.90`, try QAT with `--epochs 5 --learning-rate 1e-4`.  Use `--loss-fn mse` for QAT (default) unless the teacher distribution is very soft.

### Crossover

- **`weighted` mode** is the smoothest starting point: `--crossover-alpha 0.5` gives a midpoint blend.  Move `alpha` toward `0.3` or `0.7` to bias toward one parent.
- **`random` mode** produces the most diverse children but highest variance across seeds.  Run three seeds (`--crossover-seed 0,1,2` via separate invocations) to gauge variance before committing to one.
- **`layer` mode** preserves structural coherence (each Linear + LayerNorm pair from the same parent) at the cost of diversity — only two possible children per pair.  Useful when the network is sensitive to feature-scaling mismatches.
- **If child collapses to one parent:** `primary_metric = min(agreement_A, agreement_B)` will be low.  Try `weighted` at `0.5`, or `random` with a different seed.

### Fine-tuning

- **Reference teacher:** `finetune_child.py` always uses parent A as the fine-tune teacher.  This biases the child toward A's behaviour; if you want a more balanced child, check parent B agreement explicitly with `validate_recombination.py --include-parent-baseline`.
- **Loss function:** `kl` with `temperature 3.0` and `alpha 1.0` mirrors the distillation loss and tends to produce soft, calibrated Q-values.  Use `mse` for a simpler target.
- **LR schedule:** enable `--lr-patience 3 --lr-factor 0.5` if validation loss plateaus early.  This is disabled by default to keep runs short.
- **Early stopping:** `--early-stopping-patience 5` prevents overfitting on small state buffers; safe to enable for production runs.
- **`quantization_applied`:** set to `ptq_dynamic`, `ptq_static`, or `qat_float` only when you intend to produce an int8 fine-tuned child.  Requires calling `FineTuner.convert()` + `save_quantized()` programmatically after the script; for a pure float32 child leave as `none`.

### Validation

- **Dimension/architecture mismatches** are the most common failure mode.  Always pass the same `--input-dim`, `--output-dim`, and `--parent-hidden` (or `--hidden-size`) that were used at distillation time.  The checkpoint JSON metadata (`*.pt.json`) records these for reference.
- **Agreement vs oracle agreement:** `action_agreement` counts top-1 matches; `oracle_action_agreement` counts matches only when both models are confident.  Low oracle agreement with high action agreement usually indicates one model is uncertain overall.
- **Threshold calibration:** default thresholds in the validation scripts are conservative.  After a successful baseline run you can tighten `--min-action-agreement` toward `0.85`–`0.95` for production gates.
- **Using the same state buffer as training:** evaluation metrics will be optimistically inflated if the same states are used for both training and validation.  Prefer a held-out test split by passing a separate `.npy` file to the validation scripts.

---

## 11. Copy-Paste Recipes

### Recipe A — Minimal synthetic run (no parent checkpoints required for architecture test)

```bash
# 1. Distil (synthetic states, no real parents — useful for smoke test)
python scripts/run_distillation.py \
  --n-states 2000 --seed 42 \
  --epochs 5 \
  --output-dir checkpoints/distillation

# 2. PTQ
python scripts/quantize_distilled.py \
  --checkpoint-dir checkpoints/distillation \
  --n-states 2000 --seed 42 \
  --mode dynamic \
  --output-dir checkpoints/quantized

# 3. Validate distillation
python scripts/validate_distillation.py \
  --checkpoint-dir checkpoints/distillation \
  --n-states 2000 --seed 42 \
  --report-dir reports/distillation

# 4. Validate quantization
python scripts/validate_quantized.py \
  --float-dir checkpoints/distillation \
  --quant-dir checkpoints/quantized \
  --n-states 2000 --seed 42 \
  --report-dir reports/quantized

# 5. Crossover + fine-tune
python scripts/finetune_child.py \
  --parent-a-ckpt checkpoints/distillation/student_A.pt \
  --parent-b-ckpt checkpoints/distillation/student_B.pt \
  --crossover-mode weighted --crossover-alpha 0.5 \
  --n-states 2000 --seed 42 \
  --output-dir checkpoints/finetune

# 6. Validate recombination
python scripts/validate_recombination.py \
  --parent-a-ckpt checkpoints/distillation/student_A.pt \
  --parent-b-ckpt checkpoints/distillation/student_B.pt \
  --child-ckpt checkpoints/finetune/child_finetuned.pt \
  --n-states 2000 --seed 42 \
  --report-dir reports/recombination
```

### Recipe B — Full run with real parent checkpoints and replay buffer

```bash
PARENTS=checkpoints/parents
STATES=data/replay_states.npy   # shape (N, 8) float32
OUT=checkpoints/run1

# 1. Distil
python scripts/run_distillation.py \
  --parent-a-ckpt $PARENTS/parent_A.pt \
  --parent-b-ckpt $PARENTS/parent_B.pt \
  --states-file $STATES --seed 42 \
  --temperature 3.0 --alpha 1.0 \
  --epochs 20 --lr 1e-3 \
  --output-dir $OUT/distillation

# 2. Validate distillation
python scripts/validate_distillation.py \
  --checkpoint-dir $OUT/distillation \
  --parent-a-ckpt $PARENTS/parent_A.pt \
  --parent-b-ckpt $PARENTS/parent_B.pt \
  --states-file $STATES --seed 42 \
  --report-dir $OUT/reports/distillation

# 3. PTQ (try dynamic first)
python scripts/quantize_distilled.py \
  --checkpoint-dir $OUT/distillation \
  --states-file $STATES --seed 42 \
  --mode dynamic \
  --output-dir $OUT/quantized

# 4. Validate quantization
python scripts/validate_quantized.py \
  --float-dir $OUT/distillation \
  --quant-dir $OUT/quantized \
  --states-file $STATES --seed 42 \
  --report-dir $OUT/reports/quantized

# (Optional) If PTQ agreement < 0.90, run QAT instead:
# python scripts/qat_distilled.py \
#   --checkpoint-dir $OUT/distillation \
#   --states-file $STATES --seed 42 \
#   --epochs 5 --learning-rate 1e-4 \
#   --output-dir $OUT/qat

# 5. Crossover + fine-tune (using float parents)
python scripts/finetune_child.py \
  --parent-a-ckpt $PARENTS/parent_A.pt \
  --parent-b-ckpt $PARENTS/parent_B.pt \
  --crossover-mode weighted --crossover-alpha 0.5 \
  --crossover-seed 42 \
  --states-file $STATES --seed 42 \
  --epochs 10 --lr 5e-4 \
  --lr-patience 3 --early-stopping-patience 5 \
  --output-dir $OUT/finetune

# 6. Validate recombination
python scripts/validate_recombination.py \
  --parent-a-ckpt $PARENTS/parent_A.pt \
  --parent-b-ckpt $PARENTS/parent_B.pt \
  --child-ckpt $OUT/finetune/child_finetuned.pt \
  --states-file $STATES --seed 42 \
  --include-parent-baseline \
  --report-dir $OUT/reports/recombination
```

---

## 12. Generalization: Holdout & Domain-Shift Evaluation

**Script:** `scripts/eval_generalization.py`

Standard validation metrics (Sections 7.1–7.3) are measured on a single state buffer that may overlap with calibration or training data.  For publication-grade **generalization** claims, you need:

1. A held-out test split that was never used for training or calibration.
2. An optional **domain-shift** evaluation on states from a shifted distribution (sensor noise, input scaling, or a second `.npy` profile).

`eval_generalization.py` automates both steps by:
- Splitting the state buffer into an **in-distribution (ID)** and **holdout** subset.
- Optionally perturbing the holdout set with Gaussian noise or input scaling.
- Running `RecombinationEvaluator` on each subset and writing a per-set JSON report plus a combined `generalization_summary.json`.

### Library helpers

The split and perturbation logic is available as standalone functions in
`farm.core.decision.training.holdout_utils`:

| Function | Purpose |
|----------|---------|
| `split_replay_buffer(states, holdout_fraction, seed)` | Random train/holdout split |
| `apply_gaussian_noise(states, std, seed)` | Add i.i.d. Gaussian noise |
| `apply_input_scaling(states, scale_factor)` | Multiply all features by a scalar |
| `make_shifted_states(states, shift_type, **kwargs)` | Factory dispatcher for the above |

All helpers are also re-exported from `farm.core.decision.training`.

### Minimal synthetic run (no checkpoints needed for a smoke test)

> This example assumes you have trained parent A, parent B, and child checkpoints
> from the Recipe A workflow in Section 11.  Replace paths as needed.

```bash
python scripts/eval_generalization.py \
  --parent-a-ckpt checkpoints/distillation/student_A.pt \
  --parent-b-ckpt checkpoints/distillation/student_B.pt \
  --child-ckpt    checkpoints/finetune/child_finetuned.pt \
  --n-states 2000 --seed 42 \
  --holdout-fraction 0.2 \
  --report-dir reports/generalization
```

Output:
```
reports/generalization/
  id_report.json           # in-distribution split report
  holdout_report.json      # held-out split report
  generalization_summary.json
```

### With a real replay buffer

```bash
python scripts/eval_generalization.py \
  --parent-a-ckpt checkpoints/parents/parent_A.pt \
  --parent-b-ckpt checkpoints/parents/parent_B.pt \
  --child-ckpt    checkpoints/finetune/child_finetuned.pt \
  --states-file   data/replay_states.npy \
  --holdout-fraction 0.2 \
  --report-dir    reports/generalization
```

### With Gaussian-noise domain shift

```bash
python scripts/eval_generalization.py \
  --parent-a-ckpt checkpoints/parents/parent_A.pt \
  --parent-b-ckpt checkpoints/parents/parent_B.pt \
  --child-ckpt    checkpoints/finetune/child_finetuned.pt \
  --states-file   data/replay_states.npy \
  --holdout-fraction 0.2 \
  --shift-type    gaussian_noise \
  --shift-std     0.1 \
  --shift-seed    0 \
  --report-dir    reports/generalization
```

Output adds `reports/generalization/shifted_report.json`.

### With input-scaling domain shift

```bash
python scripts/eval_generalization.py \
  --parent-a-ckpt checkpoints/parents/parent_A.pt \
  --parent-b-ckpt checkpoints/parents/parent_B.pt \
  --child-ckpt    checkpoints/finetune/child_finetuned.pt \
  --states-file   data/replay_states.npy \
  --shift-type    input_scaling \
  --shift-scale-factor 2.0 \
  --report-dir    reports/generalization
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--holdout-fraction` | `0.2` | Fraction of states reserved for holdout |
| `--no-shuffle` | off | Skip shuffle before split (for pre-randomised buffers) |
| `--shift-type` | — | `gaussian_noise` or `input_scaling`; omit to skip shifted eval |
| `--shift-std` | `0.1` | Gaussian noise standard deviation |
| `--shift-scale-factor` | `2.0` | Input scaling multiplier |
| `--shift-seed` | `0` | Noise RNG seed |
| `--report-only` | off | Write reports without applying pass/fail thresholds |

### Reading the `generalization_summary.json`

```json
{
  "overall_passed": true,
  "report_only": false,
  "holdout_fraction": 0.2,
  "shift_type": "gaussian_noise",
  "sets": {
    "in_distribution": {
      "child_agrees_with_parent_a": 0.82,
      "child_agrees_with_parent_b": 0.79,
      "oracle_agreement": 0.91,
      "n_states": 1600,
      "passed": true
    },
    "holdout": {
      "child_agrees_with_parent_a": 0.80,
      "child_agrees_with_parent_b": 0.77,
      "oracle_agreement": 0.89,
      "n_states": 400,
      "passed": true
    },
    "shifted": {
      "child_agrees_with_parent_a": 0.73,
      "child_agrees_with_parent_b": 0.70,
      "oracle_agreement": 0.84,
      "n_states": 400,
      "passed": true,
      "shift_type": "gaussian_noise"
    }
  }
}
```

A **meaningful generalization drop** is when holdout or shifted agreement falls
more than ~5 pp below the ID score.  If this happens, consider:

- Training on a larger or more diverse replay buffer.
- Increasing the holdout fraction to detect over-fitting earlier in development.
- Tuning the crossover alpha or fine-tuning LR to reduce ID–holdout gap.

---

## 13. Publication Ablations

**Script:** `scripts/run_recombination_ablation.py`

For reproducible paper tables and CI-style regression of the full pipeline,
use the unified ablation runner.  A single invocation sweeps multiple
**conditions** (e.g. distill-only, distill+quantize, or the full pipeline)
across a list of **seeds**, shares the same state buffer across all runs, and
writes every result into a structured `results/` tree together with a
consolidated CSV and Markdown summary table.

### Quick start (no config file needed)

```bash
# Dry-run: validate plan, write stub summary, no training
python scripts/run_recombination_ablation.py --smoke-test --dry-run

# Smoke-test: tiny synthetic run (2 seeds × 3 conditions, 50 states, 2 epochs)
python scripts/run_recombination_ablation.py --smoke-test --results-dir /tmp/ablation_smoke
```

### Full run from a config file

```bash
python scripts/run_recombination_ablation.py --config ablation.yaml
```

The config file is YAML (recommended) or JSON.  A minimal example:

```yaml
seeds: [0, 1, 2]
n_states: 2000
states_file: ""           # leave empty to synthesise from seed
input_dim: 8
output_dim: 4
hidden_size: 64
results_dir: results/ablation

conditions:
  - name: distill_only
    stages: [distill]
  - name: distill_quantize
    stages: [distill, quantize]
  - name: full_pipeline
    stages: [distill, quantize, crossover, compare]

distillation:
  epochs: 20
  temperature: 3.0
  alpha: 1.0
  lr: 0.001
  batch_size: 32

quantization:
  mode: dynamic

crossover:
  mode: weighted
  alpha: 0.5

comparison:
  report_only: true
```

### Output layout

```
results/ablation/
  distill_only/
    seed_0/student_A.pt  student_B.pt
    seed_1/...
    seed_2/...
  distill_quantize/
    seed_0/student_A.pt  student_B.pt  student_A_int8.pt  student_B_int8.pt
    ...
  full_pipeline/
    seed_0/student_A.pt  student_B.pt  child_finetuned.pt
             compare_child_vs_students.json
    ...
  ablation_summary.csv          ← consolidated table (paste into spreadsheet)
  ablation_summary.md           ← Markdown version (paste into GitHub issues)
```

### Per-condition stage overrides

Each condition can override any global distillation / quantization /
crossover / comparison setting:

```yaml
conditions:
  - name: high_temp_distill
    stages: [distill, crossover, compare]
    distillation:
      temperature: 6.0   # overrides global temperature: 3.0
      epochs: 30
```

### Valid stages

| Stage | What it runs |
|-------|-------------|
| `distill` | `DistillationTrainer` for both A and B pairs; writes `student_A.pt`, `student_B.pt` |
| `quantize` | `PostTrainingQuantizer` on both students; writes `student_A_int8.pt`, `student_B_int8.pt` |
| `crossover` | `crossover_quantized_state_dict` + `FineTuner`; writes `child_finetuned.pt` |
| `compare` | `RecombinationEvaluator` (child vs students); writes `compare_child_vs_students.json` |

Stages are always applied in the order listed above regardless of declaration
order in the config.  A `quantize` stage without a preceding `distill` stage
will log a warning and skip.

### Dry-run mode

```bash
python scripts/run_recombination_ablation.py --config ablation.yaml --dry-run
```

Prints the full execution plan (conditions × seeds × stages × directories)
and writes a stub `ablation_summary.md` / `ablation_summary.csv` without
running any training.  Use this to verify the config before a long run.

### Using a shared real replay buffer

Set `states_file` in the config to a `.npy` file of shape `(N, input_dim)`
float32.  All seeds and conditions will use **the same** state file, ensuring
metrics are comparable across the ablation.

```yaml
states_file: data/replay_states.npy
```

### Reading the summary table

The Markdown summary table (`ablation_summary.md`) contains one row per
(condition, seed) pair.  Key columns:

| Column | Meaning |
|--------|---------|
| `child_vs_ref_a_agreement` | Top-1 action agreement of child vs student A |
| `child_vs_ref_b_agreement` | Top-1 action agreement of child vs student B |
| `oracle_agreement` | Fraction where child matches *at least one* reference |
| `elapsed_s` | Wall-clock seconds for the (condition, seed) run |

`child_vs_ref_*_agreement` columns are populated only when the `compare`
stage is included.  Conditions without a `compare` stage show `n/a`.

---

## Related Documentation

| Document | Contents |
|----------|---------|
| [`docs/design/distill_quantize_crossover_finetune.md`](../design/distill_quantize_crossover_finetune.md) | Architecture overview, Mermaid pipeline diagram, module map, and recorded experimental results |
| [`docs/design/crossover_strategies.md`](../design/crossover_strategies.md) | Detailed semantics of `random`, `layer`, and `weighted` crossover strategies with code examples |
| [`docs/design/crossover_search_space.md`](../design/crossover_search_space.md) | Grid definitions, pre-defined search presets, and leaderboard format for `run_crossover_search.py` |
| [`docs/distillation_soft_label_comparison.md`](../distillation_soft_label_comparison.md) | Hard vs blended vs soft distillation objective comparison with reproducible results |
| [`farm/config/default.yaml`](../../farm/config/default.yaml) | All YAML defaults including the `crossover_child_finetune` section |
