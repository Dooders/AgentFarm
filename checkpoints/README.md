# Checkpoints (generated)

Files under this directory are **not committed**: PyTorch weights (`.pt`), NumPy buffers (`.npy`), and run metadata (`.pt.json`) are listed in `.gitignore`.

To produce distillation checkpoints locally:

```bash
source venv/bin/activate
python scripts/run_distillation.py --output-dir checkpoints/distillation
```

Optional validation:

```bash
python scripts/validate_distillation.py --checkpoint-dir checkpoints/distillation_demo
```

For a documented hard-only vs soft-only vs blended comparison (synthetic controlled setup), see [docs/distillation_soft_label_comparison.md](../docs/distillation_soft_label_comparison.md).

Sample paths in `reports/**` JSON files describe layouts from example runs; regenerate assets before relying on them.
