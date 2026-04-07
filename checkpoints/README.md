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

Sample paths in `reports/**` JSON files describe layouts from example runs; regenerate assets before relying on them.
