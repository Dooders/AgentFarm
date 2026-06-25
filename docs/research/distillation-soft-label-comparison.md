# Soft-label distillation: hard vs soft vs blended (issue #596)

This note documents a **reproducible** three-way comparison of knowledge-distillation objectives implemented in `farm.core.decision.training.trainer_distill` and runnable via CLI (`scripts/run_distillation.py`). It addresses the “results documented” acceptance item from [issue #596](https://github.com/Dooders/AgentFarm/issues/596).

## Method

- **Script:** `scripts/compare_distillation_modes.py`
- **Controlled variables:** One frozen `BaseQNetwork` teacher (fixed seed), one shared synthetic state buffer `(N, input_dim)` from NumPy, one fixed `StudentQNetwork` initialization (reloaded for each run).
- **What changes:** Only `DistillationConfig.alpha`:
  - **hard_only:** `alpha = 0` → cross-entropy on teacher argmax only.
  - **blended:** `alpha = 0.7` → `0.7 * L_soft + 0.3 * L_hard` (KL soft term, temperature 3).
  - **soft_only:** `alpha = 1` → temperature-scaled KL distillation only.
- **Metrics (validation split, last epoch):**
  - **Action agreement:** fraction of states where student argmax equals teacher argmax.
  - **Mean probability similarity:** `1 - mean(|p_teacher - p_student|)` over actions and samples (temperature-1 softmax in the evaluator; see `DistillationTrainer._evaluate`).

**Hyperparameters (default run):** `seed_base=42`, `input_dim=8`, `output_dim=4`, `parent_hidden=64`, `n_states=5000`, `temperature=3`, `epochs=25`, `batch_size=64`, `lr=1e-3`, `val_fraction=0.1`, `loss_fn=kl`.

## Results (2026-04-08, local run)

| Mode       | α   | Final action agreement | Final mean prob. similarity | Best val loss\* |
|------------|-----|------------------------|----------------------------|-----------------|
| hard_only  | 0.0 | 93.4%                  | 0.814                      | 0.145           |
| blended    | 0.7 | 93.2%                  | 0.981                      | 0.162           |
| soft_only  | 1.0 | 93.2%                  | 0.989                      | 0.015           |

\***Best val loss** is the trainer’s validation objective at the best epoch. It is **not** comparable across rows: hard-only optimizes CE (scale ~0.14 here), while soft-only optimizes scaled KL (~0.015 here). Use agreement and probability similarity for cross-mode fidelity, not raw val loss.

### Interpretation

- **Distribution match:** Soft-only and blended training produce **much** closer full-action distributions to the teacher than hard-only (probability similarity ~0.98–0.99 vs ~0.81), which is exactly what soft labels are meant to preserve when Q-values are close across actions.
- **Top-1 agreement:** On this synthetic setup, hard-only is **marginally** higher (~0.2 percentage points) than soft-only/blended at the last epoch. That is plausible: argmax CE directly targets the teacher’s top action, while KL spreads gradient across the distribution. On real replay data and longer training, rankings can differ; re-run the script with `--states-file` and your checkpoints for a task-specific read.

## Reproduce

```bash
source venv/bin/activate
python scripts/compare_distillation_modes.py --json-out reports/distillation_mode_comparison.json
```

Machine-readable summary: `reports/distillation_mode_comparison.json` (regenerate locally; paths may be gitignored depending on your `reports/` layout).

To mirror production-style usage (pairs A/B, optional parent checkpoints), use:

```bash
python scripts/run_distillation.py --alpha 0.0   # hard-focused
python scripts/run_distillation.py --alpha 1.0   # soft-only (default)
python scripts/run_distillation.py --alpha 0.7   # blended
```

with shared `--seed`, `--states-file`, and parent checkpoints for a fair comparison on your data.
