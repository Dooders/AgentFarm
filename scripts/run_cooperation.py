#!/usr/bin/env python3
"""Run the registered cooperation matrix, analyse it and write REPORT.md.

Examples
--------
Full registered matrix (27 grid cells + 7 secondary cells, 20 seeds, plus solo twins)::

    PYTHONHASHSEED=0 python scripts/run_cooperation.py

Quick pilot on two calibration seeds::

    python scripts/run_cooperation.py --seeds 1 2 --output-dir /tmp/coop_pilot

Include the exploratory all-one-motive populations::

    python scripts/run_cooperation.py --exploratory

Re-analyse existing raw outputs without simulating::

    python scripts/run_cooperation.py --analyze-only
"""

from __future__ import annotations

import argparse
import time

from farm.experiments.cooperation.analysis import analyze, write_analysis
from farm.experiments.cooperation.config import (
    EXPLORATORY_CONDITION_ORDER,
    PRIMARY_CONDITION_ORDER,
    SECONDARY_CONDITION_ORDER,
)
from farm.experiments.cooperation.experiment import CONFIRMATORY_SEEDS, MatrixConfig, load_outputs, run_matrix
from farm.experiments.cooperation.report import write_report

DEFAULT_OUTPUT_DIR = "experiments/cooperation/results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=list(CONFIRMATORY_SEEDS))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--conditions", nargs="+", default=None, help="condition names (default: primary + secondary)")
    p.add_argument("--exploratory", action="store_true", help="also run the all-one-motive populations")
    p.add_argument("--analyze-only", action="store_true", help="skip simulation; analyse existing raw outputs")
    p.add_argument("--no-resume", action="store_true", help="re-simulate cells that already have outputs")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {message}", flush=True)

    names = tuple(args.conditions) if args.conditions else PRIMARY_CONDITION_ORDER + SECONDARY_CONDITION_ORDER
    if args.exploratory:
        names = names + tuple(n for n in EXPLORATORY_CONDITION_ORDER if n not in names)
    if args.analyze_only:
        outputs = load_outputs(args.output_dir)
    else:
        matrix = MatrixConfig(seeds=tuple(args.seeds), condition_names=names, workers=args.workers)
        outputs = run_matrix(matrix, args.output_dir, progress=log, resume=not args.no_resume)
    log("analysing")
    result = analyze(outputs)
    write_analysis(result, args.output_dir)
    path = write_report(outputs, result, args.output_dir)
    log(f"wrote {path}")
    for key, verdict in result.verdicts.items():
        log(f"{key}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
