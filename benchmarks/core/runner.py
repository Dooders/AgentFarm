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
from contextlib import ExitStack
from benchmarks.core.instrumentation.cprofile import cprofile_capture
from benchmarks.core.instrumentation.psutil_monitor import psutil_sampling
from benchmarks.core.reporting.markdown import write_run_report
import shutil


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
            with ExitStack() as stack:
                # Timing always on if instrumented
                stack.enter_context(time_block(metrics, key="duration_s"))
                # Optional instruments
                for inst in self.instruments:
                    # Support simple names or config dicts {name: ..., ...}
                    if isinstance(inst, str):
                        name = inst
                        cfg: Dict[str, Any] = {}
                    elif isinstance(inst, dict):
                        name = str(inst.get("name"))
                        cfg = {k: v for k, v in inst.items() if k != "name"}
                    else:
                        continue
                    if name == "cprofile":
                        top_n = int(cfg.get("top_n", 30))
                        stack.enter_context(cprofile_capture(self.run_dir, self.name, i, metrics, top_n=top_n))
                    elif name == "psutil":
                        interval_ms = int(cfg.get("interval_ms", 200))
                        max_samples = int(cfg.get("max_samples", 1000))
                        stack.enter_context(psutil_sampling(self.run_dir, self.name, i, metrics, interval_ms=interval_ms, max_samples=max_samples))
                    elif name == "timing":
                        # already added above
                        pass
                    else:
                        raise ValueError(f"Unknown instrument: {name}")
                iter_metrics = self.experiment.execute_once(context)
            # Merge inner metrics into metrics namespace
            for k, v in (iter_metrics or {}).items():
                metrics[k] = v
            # Register artifacts from metrics, if present
            if "cprofile_artifact" in metrics:
                result.add_artifact(name=f"cprofile_iter_{i}", type="profile", path=str(metrics["cprofile_artifact"]))
            if "cprofile_summary_path" in metrics:
                result.add_artifact(name=f"cprofile_summary_iter_{i}", type="json", path=str(metrics["cprofile_summary_path"]))
            if "psutil_artifact" in metrics:
                result.add_artifact(name=f"psutil_iter_{i}", type="jsonl", path=str(metrics["psutil_artifact"]))
            result.add_iteration(index=i, duration_s=float(metrics.get("duration_s", 0.0)), metrics=metrics)

        # Aggregate simple summary metrics if present
        if result.iteration_metrics:
            durations = [it.duration_s for it in result.iteration_metrics]
            mean = sum(durations) / len(durations)
            durations_sorted = sorted(durations)
            def _pct(p: float) -> float:
                if not durations_sorted:
                    return 0.0
                # nearest-rank method
                idx = max(0, min(len(durations_sorted) - 1, int(round(p * (len(durations_sorted) - 1)))))
                return float(durations_sorted[idx])
            p50 = _pct(0.50)
            p95 = _pct(0.95)
            p99 = _pct(0.99) if len(durations_sorted) >= 3 else p95
            result.metrics["duration_s"] = {"mean": mean, "p50": p50, "p95": p95, "p99": p99}

        # Teardown
        try:
            self.experiment.teardown(context)
        finally:
            path = result.save(self.run_dir)
            # Expose saved path in notes for convenience
            result.notes = (result.notes or "") + f"\nSaved to: {path}"
            # Save spec if provided via context.extras
            try:
                spec_path = context.extras.get("spec_path") if isinstance(context.extras, dict) else None
                if spec_path and os.path.exists(spec_path):
                    shutil.copy2(spec_path, os.path.join(self.run_dir, os.path.basename(spec_path)))
            except Exception:
                pass
            try:
                write_run_report(result, self.run_dir)
            except Exception:
                pass
        return result

