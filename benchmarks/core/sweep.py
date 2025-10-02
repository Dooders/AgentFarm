from __future__ import annotations

"""
SweepRunner: expands parameter sweeps and runs multiple experiments.
"""

import itertools
import os
from typing import Any, Dict, Iterable, List, Tuple

from benchmarks.core.registry import REGISTRY
from benchmarks.core.runner import Runner
from benchmarks.core.results import RunResult
from benchmarks.core.reporting.markdown import write_sweep_report


def _expand_cartesian(base_params: Dict[str, Any], sweep: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not sweep:
        return [base_params]
    keys = sorted(sweep.keys())
    values = [sweep[k] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        params = dict(base_params)
        for k, v in zip(keys, prod):
            params[k] = v
        combos.append(params)
    return combos


class SweepRunner:
    def __init__(self, *, experiment_slug: str, base_params: Dict[str, Any], output_dir: str,
                 iterations_warmup: int, iterations_measured: int, seed: int | None,
                 tags: List[str], notes: str, instruments: List[str]) -> None:
        self.experiment_slug = experiment_slug
        self.base_params = base_params
        self.output_dir = output_dir
        self.iterations_warmup = iterations_warmup
        self.iterations_measured = iterations_measured
        self.seed = seed
        self.tags = tags
        self.notes = notes
        self.instruments = instruments

    def run_cartesian(self, sweep: Dict[str, List[Any]]) -> List[RunResult]:
        REGISTRY.discover_package("benchmarks.implementations")
        params_list = _expand_cartesian(self.base_params, sweep)
        results: List[RunResult] = []
        for params in params_list:
            experiment = REGISTRY.create(self.experiment_slug, params)
            runner = Runner(
                name=self.experiment_slug,
                experiment=experiment,
                output_dir=self.output_dir,
                iterations_warmup=self.iterations_warmup,
                iterations_measured=self.iterations_measured,
                seed=self.seed,
                tags=self.tags,
                notes=self.notes,
                instruments=self.instruments,
            )
            result = runner.run()
            results.append(result)
        # Write sweep report to top-level output dir
        try:
            write_sweep_report(results, self.output_dir, title=f"{self.experiment_slug} Sweep Summary")
        except Exception:
            pass
        return results

