#!/usr/bin/env python3
"""Exploratory follow-up to manipulation check M1 of the cooperation experiment.

M1 expected an unenforced exit cost (recognition 0) to leave bond life
unchanged; in the registered run it shortened bond life slightly. The
candidate mechanism is that agents' settings fit is computed from *nominal*
settings, so a stated-but-unenforced exit cost still changes who accepts a
match and how members value staying. This re-runs the six recognition-0 cells
at exit cost 0 and 0.6 with the fit term switched off (``fit_scale = 0``) on
the registered seeds. If the mechanism is right, the M1 contrast vanishes.

    python scripts/run_cooperation_m1_followup.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from farm.experiments.cooperation.analysis import _grid_mean, _pivot, paired_test
from farm.experiments.cooperation.config import PRIMARY_SCOPE, DecisionConfig, grid_cell_name
from farm.experiments.cooperation.experiment import CONFIRMATORY_SEEDS, MatrixConfig, run_matrix

DEFAULT_OUTPUT_DIR = "experiments/cooperation/results/followup_m1"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    t0 = time.time()
    names = tuple(grid_cell_name(s, e, 0.0) for e in (0.0, 0.6) for s in PRIMARY_SCOPE)
    matrix = MatrixConfig(
        seeds=CONFIRMATORY_SEEDS,
        condition_names=names,
        decisions=replace(DecisionConfig(), fit_scale=0.0),
        workers=args.workers,
    )
    outputs = run_matrix(matrix, args.output_dir, progress=lambda m: print(f"[{time.time() - t0:6.1f}s] {m}"))
    rmst = _pivot(outputs.runs, "bond_rmst")
    result = paired_test(
        "M1-nofit",
        "Bond life at exit cost 0.6 vs 0 with no recognition, settings fit switched off (two-sided).",
        "bond_rmst",
        _grid_mean(rmst, 0.6, 0.0) - _grid_mean(rmst, 0.0, 0.0),
        alternative="two-sided",
    )
    out = Path(args.output_dir) / "m1_followup.json"
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
