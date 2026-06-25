# Neural recombination (distillation, PTQ, QAT)

After distilling student Q-networks you can apply **8-bit post-training quantization (PTQ)** and, if accuracy is not good enough, **quantization-aware training (QAT)**. Implementation details and PyTorch version notes live in [`farm/core/decision/training/quantize_ptq.py`](../../farm/core/decision/training/quantize_ptq.py) and [`farm/core/decision/training/quantize_qat.py`](../../farm/core/decision/training/quantize_qat.py).

For the full step-by-step pipeline, see the [Neural Recombination Runbook](neural-recombination-runbook.md).

## Typical flow

1. **Distill** float students (`student_A.pt` / `student_B.pt`):

   ```bash
   python scripts/run_distillation.py --help
   ```

2. **PTQ** (default: dynamic weight-only `qint8`; static mode needs calibration states):

   ```bash
   python scripts/quantize_distilled.py \
       --checkpoint-dir checkpoints/distillation \
       --output-dir checkpoints/quantized
   ```

   For static PTQ, use `--states-file` or synthetic `--n-states` / `--seed`; calibration volume uses `--calibration-batches` and `--calibration-batch-size` (defaults 10 / 64). Match distillation architecture with `--input-dim`, `--output-dim`, `--parent-hidden` (defaults **8**, **4**, **64**).

3. **Validate** float students (optional): `python scripts/validate_distillation.py --help`

4. **Validate quantized vs float** (CPU): `python scripts/validate_quantized.py --help`. The validator loads quantized checkpoints as full-model pickles, so pass `--allow-unsafe-unpickle` only for trusted artifacts. The JSON report includes median/mean/p95 single-sample latency, optional **throughput** (`--throughput-batch-size`), **memory** RSS snapshots, float–quant **MSE/KL/top-k** agreement, and optional **teacher** metrics if `parent_*.pt` is found under `--float-dir` / `--teacher-dir` or via `--teacher-*-ckpt`.

5. **Evaluate a crossover child vs both parents** (offline Q metrics, versioned JSON): `python scripts/validate_recombination.py --help`. Baselines: **child vs parent A**, **child vs parent B**, optional **parent A vs parent B** (`--include-parent-baseline`), plus **oracle** agreement in the report summary. Use the same `--states-file` / `--seed` / `--n-states` pattern as other validation scripts. For **quantized** full-model checkpoints (PTQ or post-QAT `torch.save` exports), add `--parent-a-quantized`, `--parent-b-quantized`, and/or `--child-quantized` together with `--allow-unsafe-unpickle`; those roles are loaded with `load_quantized_checkpoint` and run on **CPU**.

6. **Search many crossover + fine-tune combinations** (leaderboard + manifest): `python scripts/run_crossover_search.py --help`. Presets include `minimal` / `default`, plus **`minimal-qat`** / **`default-qat`** (adds a `short_qat` / `ptq_dynamic` regime). Use **`--workers N`** for process-parallel children (float `BaseQNetwork` parents only). Quick check: `make crossover-search-smoke`. Design notes: [crossover search space](../design/crossover_search_space.md), strategy semantics: [crossover strategies](../design/crossover_strategies.md).

## Crossover from PTQ parent paths (Python)

[`initialize_child_from_crossover`](../../farm/core/decision/training/crossover.py) can auto-detect a **dynamic** PTQ sidecar next to a `.pt` file (same JSON shape as `PostTrainingQuantizer.save_checkpoint`) and load via `load_quantized_checkpoint`. That path uses full-model unpickling (`weights_only=False`); pass **`allow_unsafe_unpickle=True` only for trusted checkpoints**. Static PTQ sidecars are not auto-loaded here—use float state dicts or in-memory modules. Details: [crossover strategies](../design/crossover_strategies.md).

## Optional QAT

After PTQ, if action agreement or Q-error is unacceptable: weight-only fake quant on linear layers, same int8 export format as PTQ after convert.

```bash
python scripts/qat_distilled.py \
    --checkpoint-dir checkpoints/distillation \
    --output-dir checkpoints/qat
```

Use `--teacher-a-ckpt` / `--student-a-ckpt` (and `*-b-*` for pair B) when paths are not under a single `--checkpoint-dir`; see `python scripts/qat_distilled.py --help` for epochs, learning rate, and `--no-convert` (float QAT checkpoint only). Quantized QAT checkpoints work with `scripts/validate_quantized.py` like PTQ outputs.

## Tests

```bash
pytest tests/decision/test_ptq.py tests/decision/test_validate_quantized.py tests/decision/test_qat.py tests/decision/test_crossover_search.py
```
