#!/usr/bin/env python3
"""Run the pre-registered Veil Ceiling matrix, analyse it and write REPORT.md.

Examples
--------
Full pre-registered matrix (30 seeds × 2 inheritance modes × 13 conditions)::

    PYTHONHASHSEED=0 python scripts/run_veil_ceiling.py

Quick pilot without the penalty-robustness cells::

    python scripts/run_veil_ceiling.py --seeds 3 --no-robustness --output-dir /tmp/veil_pilot

Re-analyse existing raw outputs without simulating::

    python scripts/run_veil_ceiling.py --analyze-only
"""

from __future__ import annotations

import argparse
import time

from farm.experiments.veil_ceiling.analysis import analyze, write_analysis
from farm.experiments.veil_ceiling.config import PRIMARY_CONDITION_ORDER, SEEDS_PER_CELL
from farm.experiments.veil_ceiling.experiment import MatrixConfig, load_outputs, run_matrix
from farm.experiments.veil_ceiling.report import write_report

DEFAULT_OUTPUT_DIR = "experiments/veil_ceiling/results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seeds", type=int, default=SEEDS_PER_CELL, help="seeds per cell (1..N)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-ticks", type=int, default=1500)
    p.add_argument("--eval-ticks", type=int, default=300)
    p.add_argument(
        "--conditions",
        nargs="+",
        default=list(PRIMARY_CONDITION_ORDER),
        help="primary condition names to run (default: all pre-registered cells)",
    )
    p.add_argument("--no-robustness", action="store_true", help="skip the penalty-robustness cells (C1/C2 at p=3, 9)")
    p.add_argument("--analyze-only", action="store_true", help="skip simulation; analyse existing raw outputs")
    p.add_argument("--no-resume", action="store_true", help="re-simulate cells that already have outputs")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {message}", flush=True)

    if args.analyze_only:
        outputs = load_outputs(args.output_dir)
    else:
        matrix = MatrixConfig(
            seeds=tuple(range(1, args.seeds + 1)),
            condition_names=tuple(args.conditions),
            include_robustness=not args.no_robustness,
            train_ticks=args.train_ticks,
            eval_ticks=args.eval_ticks,
            workers=args.workers,
        )
        log(f"simulating {len(matrix.run_configs())} runs into {args.output_dir}")
        outputs = run_matrix(matrix, args.output_dir, progress=log, resume=not args.no_resume)

    log(f"analysing {len(outputs.runs)} runs")
    result = analyze(outputs)
    analysis_dir = write_analysis(result, args.output_dir)
    report_path = write_report(result, outputs, args.output_dir)
    log(f"analysis tables: {analysis_dir}")
    log(f"report: {report_path}")
    for mode, verdicts in result.hypotheses.items():
        if mode == "H5":
            log(f"H5 ({verdicts['verdict']})")
        else:
            log(f"{mode}: " + ", ".join(f"{h}={v['verdict']}" for h, v in verdicts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
