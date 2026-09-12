#!/usr/bin/env python3
"""Run the adaptive-monitor follow-up to the Veil Ceiling experiment.

Keeps coverage and penalty fixed and reallocates the true monitor map each
epoch from last epoch's observed residual. Seed-matched against the static
cells of the same fidelity (C2 / C3_f0.9 / C3_f0.7).

    PYTHONHASHSEED=0 python scripts/run_veil_ceiling_adaptive.py
    python scripts/run_veil_ceiling_adaptive.py --seeds 3 --output-dir /tmp/veil_adaptive
    python scripts/run_veil_ceiling_adaptive.py --analyze-only
"""

from __future__ import annotations

import argparse
import time

from farm.experiments.veil_ceiling.adaptive import analyze_adaptive, write_adaptive
from farm.experiments.veil_ceiling.config import ADAPTIVE_MATRIX_CONDITION_ORDER, SEEDS_PER_CELL
from farm.experiments.veil_ceiling.experiment import MatrixConfig, load_outputs, run_matrix

DEFAULT_OUTPUT_DIR = "experiments/veil_ceiling/adaptive_results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seeds", type=int, default=SEEDS_PER_CELL)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-ticks", type=int, default=1500)
    p.add_argument("--eval-ticks", type=int, default=300)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--no-resume", action="store_true")
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
            condition_names=tuple(ADAPTIVE_MATRIX_CONDITION_ORDER),
            include_robustness=False,
            train_ticks=args.train_ticks,
            eval_ticks=args.eval_ticks,
            workers=args.workers,
        )
        log(f"simulating {len(matrix.run_configs())} runs into {args.output_dir}")
        outputs = run_matrix(matrix, args.output_dir, progress=log, resume=not args.no_resume)

    log(f"analysing {len(outputs.runs)} runs")
    result = analyze_adaptive(outputs)
    report = write_adaptive(result, outputs, args.output_dir)
    log(f"report: {report}")
    for mode, verdicts in result.hypotheses.items():
        if not isinstance(verdicts, dict):
            continue
        log(f"{mode}: " + ", ".join(f"{h}={v['verdict']}" for h, v in verdicts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
