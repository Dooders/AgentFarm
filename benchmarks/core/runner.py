from __future__ import annotations

"""
Runner implementation orchestrating experiment execution.
"""

import os
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.results import RunResult
from benchmarks.core.instrumentation.timing import time_block


def _random_run_id(n: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


class Runner:
    def __init__(self, *, name: str, experiment: Experiment, output_dir: str,
                 iterations_warmup: int = 0, iterations_measured: int = 1,
                 seed: Optional[int] = None, tags: Optional[List[str]] = None,
                 notes: str = "", instruments: Optional[List[str]] = None) -> None:
        self.name = name
        self.experiment = experiment
        self.output_dir = output_dir
        self.iterations_warmup = max(0, int(iterations_warmup))
        self.iterations_measured = max(1, int(iterations_measured))
        self.seed = seed
        self.tags = tags or []
        self.notes = notes
        self.instruments = instruments or ["timing"]

        self.run_id = _random_run_id()
        self.run_dir = os.path.join(self.output_dir, f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

    def _seed_all(self) -> None:
        if self.seed is None:
            return
        random.seed(self.seed)
        try:
            import numpy as np  # type: ignore
            np.random.seed(self.seed)  # type: ignore
        except Exception:
            pass

    def run(self) -> RunResult:
        self._seed_all()

        context = ExperimentContext(
            run_id=self.run_id,
            output_dir=self.output_dir,
            run_dir=self.run_dir,
            iteration_index=None,
            seed=self.seed,
            instruments=[],
            extras={},
        )

        # Setup
        self.experiment.setup(context)

        result = RunResult(
            name=self.name,
            run_id=self.run_id,
            parameters=self.experiment.params,
            iterations={"warmup": self.iterations_warmup, "measured": self.iterations_measured},
            tags=self.tags,
            notes=self.notes,
        )

        # Warmups (not recorded)
        for _ in range(self.iterations_warmup):
            context.iteration_index = None
            _ = self.experiment.execute_once(context)

        # Measured iterations
        for i in range(self.iterations_measured):
            context.iteration_index = i
            metrics: Dict[str, Any] = {}
            with time_block(metrics, key="duration_s"):
                iter_metrics = self.experiment.execute_once(context)
            # Merge inner metrics into metrics namespace
            for k, v in (iter_metrics or {}).items():
                metrics[k] = v
            result.add_iteration(index=i, duration_s=float(metrics.get("duration_s", 0.0)), metrics=metrics)

        # Aggregate simple summary metrics if present
        if result.iteration_metrics:
            durations = [it.duration_s for it in result.iteration_metrics]
            mean = sum(durations) / len(durations)
            durations_sorted = sorted(durations)
            p50 = durations_sorted[int(0.5 * (len(durations_sorted) - 1))]
            p95 = durations_sorted[int(0.95 * (len(durations_sorted) - 1))]
            result.metrics["duration_s"] = {"mean": mean, "p50": p50, "p95": p95}

        # Teardown
        try:
            self.experiment.teardown(context)
        finally:
            path = result.save(self.run_dir)
            # Expose saved path in notes for convenience
            result.notes = (result.notes or "") + f"\nSaved to: {path}"
        return result

