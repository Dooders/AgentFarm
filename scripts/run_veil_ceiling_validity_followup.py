#!/usr/bin/env python3
"""Re-analyse an existing Veil Ceiling record to ask why observed behaviour stays predictive.

Runs the feature ablation, the honest-calibrated (C1/C4-fitted) evaluator and the
feature-profile analysis on the raw outputs under ``--output-dir`` and writes
``analysis/validity_followup/*.csv``, two figures and ``VALIDITY_FOLLOWUP.md``.

    python scripts/run_veil_ceiling_validity_followup.py
    python scripts/run_veil_ceiling_validity_followup.py --output-dir /tmp/veil_pilot --bootstrap-reps 200
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from farm.experiments.veil_ceiling.config import THRESHOLDS
from farm.experiments.veil_ceiling.experiment import load_outputs
from farm.experiments.veil_ceiling.validity_followup import run_followup, write_followup

DEFAULT_OUTPUT_DIR = "experiments/veil_ceiling/results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--bootstrap-reps", type=int, default=THRESHOLDS.bootstrap_reps)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {message}", flush=True)

    thresholds = replace(THRESHOLDS, bootstrap_reps=args.bootstrap_reps)
    outputs = load_outputs(args.output_dir)
    log(f"loaded {len(outputs.runs)} runs from {args.output_dir}")
    result = run_followup(outputs, thresholds)
    report = write_followup(result, outputs, args.output_dir, thresholds)
    log(f"report: {report}")
    c2 = result.ablation[result.ablation["condition"] == "C2"]
    for mode, group in c2.groupby("inheritance_mode"):
        summary = ", ".join(f"{r.feature_set}={r.auc:.3f}" for r in group.itertuples())
        log(f"{mode} C2 ablation: {summary}")
    for r in result.cross_condition[result.cross_condition["condition"] == "C2"].itertuples():
        log(f"{r.inheritance_mode} C2 evaluated with {r.calibration_condition}-calibrated model: AUC {r.auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
