"""
Runner implementation orchestrating experiment execution.

This module provides the Runner class which handles the complete lifecycle of
benchmark execution including setup, warmup iterations, measured iterations
with instrumentation, and result aggregation.
"""

import logging
import os
import random
import shutil
import string
from contextlib import ExitStack
from datetime import datetime
from typing import Any, Dict, List, Optional

from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.instrumentation.cprofile import cprofile_capture
from benchmarks.core.instrumentation.psutil_monitor import psutil_sampling
from benchmarks.core.instrumentation.timing import time_block
from benchmarks.core.reporting.markdown import write_run_report
from benchmarks.core.results import RunResult


def _random_run_id(n: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


class Runner:
    """
    Orchestrates experiment execution with instrumentation and result collection.

    The Runner handles the complete lifecycle of benchmark execution including:
    - Setting up reproducible execution environments with seeding
    - Running warmup iterations to stabilize performance
    - Executing measured iterations with timing and instrumentation
    - Collecting and aggregating metrics across iterations
    - Generating reports and saving results

    Attributes
    ----------
    name : str
        Human-readable name for this benchmark run
    experiment : Experiment
        The experiment instance to execute
    output_dir : str
        Base directory where run artifacts will be saved
    iterations_warmup : int
        Number of warmup iterations (not measured)
    iterations_measured : int
        Number of measured iterations for metrics collection
    seed : Optional[int]
        Random seed for reproducible execution
    tags : List[str]
        Tags for categorizing this run
    notes : str
        Additional notes about this run
    instruments : List[str]
        List of instrumentation tools to use (timing, cprofile, psutil)
    """

    def __init__(
        self,
        *,
        name: str,
        experiment: Experiment,
        output_dir: str,
        iterations_warmup: int = 0,
        iterations_measured: int = 1,
        seed: Optional[int] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        instruments: Optional[List[str]] = None,
    ) -> None:
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
        self.run_dir = os.path.join(
            self.output_dir,
            f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.run_id}",
        )
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
        """
        Execute the experiment and return aggregated results.

        This method orchestrates the complete benchmark execution:
        1. Seeds random number generators for reproducibility
        2. Calls experiment.setup() for resource allocation
        3. Runs warmup iterations (not measured)
        4. Executes measured iterations with instrumentation
        5. Aggregates metrics across all measured iterations
        6. Calls experiment.teardown() for cleanup
        7. Saves results and generates reports

        Returns
        -------
        RunResult
            Aggregated results including timing statistics, metrics, and artifacts
        """
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
            iterations={
                "warmup": self.iterations_warmup,
                "measured": self.iterations_measured,
            },
            tags=self.tags,
            notes=self.notes,
        )

        # Run warmup iterations to stabilize performance (not measured)
        for _ in range(self.iterations_warmup):
            context.iteration_index = None  # Mark as warmup iteration
            _ = self.experiment.execute_once(context)

        # Execute measured iterations with instrumentation
        for i in range(self.iterations_measured):
            context.iteration_index = i  # Mark as measured iteration
            metrics: Dict[str, Any] = {}

            # Use ExitStack to manage multiple context managers for instrumentation
            with ExitStack() as stack:
                # Timing instrumentation is always enabled if any instruments are configured
                stack.enter_context(time_block(metrics, key="duration_s"))

                # Configure additional instrumentation tools
                for inst in self.instruments:
                    # Support both simple string names and configuration dictionaries
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
                        stack.enter_context(
                            cprofile_capture(
                                self.run_dir, self.name, i, metrics, top_n=top_n
                            )
                        )
                    elif name == "psutil":
                        interval_ms = int(cfg.get("interval_ms", 200))
                        max_samples = int(cfg.get("max_samples", 1000))
                        stack.enter_context(
                            psutil_sampling(
                                self.run_dir,
                                self.name,
                                i,
                                metrics,
                                interval_ms=interval_ms,
                                max_samples=max_samples,
                            )
                        )
                    elif name == "timing":
                        # already added above
                        pass
                    else:
                        raise ValueError(f"Unknown instrument: {name}")
                # Execute the actual experiment workload
                iter_metrics = self.experiment.execute_once(context)

            # Merge experiment metrics with instrumentation metrics
            for k, v in (iter_metrics or {}).items():
                metrics[k] = v

            # Register instrumentation artifacts for this iteration
            if "cprofile_artifact" in metrics:
                result.add_artifact(
                    name=f"cprofile_iter_{i}",
                    type="profile",
                    path=str(metrics["cprofile_artifact"]),
                )
            if "cprofile_summary_path" in metrics:
                result.add_artifact(
                    name=f"cprofile_summary_iter_{i}",
                    type="json",
                    path=str(metrics["cprofile_summary_path"]),
                )
            if "psutil_artifact" in metrics:
                result.add_artifact(
                    name=f"psutil_iter_{i}",
                    type="jsonl",
                    path=str(metrics["psutil_artifact"]),
                )

            # Record this iteration's results
            result.add_iteration(
                index=i,
                duration_s=float(metrics.get("duration_s", 0.0)),
                metrics=metrics,
            )

        # Calculate aggregate timing statistics across all measured iterations
        if result.iteration_metrics:
            durations = [it.duration_s for it in result.iteration_metrics]
            mean = sum(durations) / len(durations)
            durations_sorted = sorted(durations)

            def _pct(p: float) -> float:
                """Calculate percentile using nearest-rank method."""
                if not durations_sorted:
                    return 0.0
                # nearest-rank method: find index closest to percentile
                idx = max(
                    0,
                    min(
                        len(durations_sorted) - 1,
                        int(round(p * (len(durations_sorted) - 1))),
                    ),
                )
                return float(durations_sorted[idx])

            # Calculate key performance percentiles
            p50 = _pct(0.50)  # Median
            p95 = _pct(0.95)  # 95th percentile (typical performance target)
            p99 = _pct(0.99) if len(durations_sorted) >= 3 else p95  # 99th percentile

            # Store aggregated timing metrics
            result.metrics["duration_s"] = {
                "mean": mean,
                "p50": p50,
                "p95": p95,
                "p99": p99,
            }

        # Cleanup and finalization
        try:
            self.experiment.teardown(context)
        finally:
            # Save results to disk
            path = result.save(self.run_dir)
            # Add save location to notes for user convenience
            result.notes = (result.notes or "") + f"\nSaved to: {path}"

            # Copy spec file to run directory for reproducibility
            try:
                spec_path = (
                    context.extras.get("spec_path")
                    if isinstance(context.extras, dict)
                    else None
                )
                if spec_path and os.path.exists(spec_path):
                    shutil.copy2(
                        spec_path,
                        os.path.join(self.run_dir, os.path.basename(spec_path)),
                    )
            except (OSError, IOError, PermissionError) as e:
                logging.warning(f"Failed to copy spec file: {e}")
            except Exception as e:
                logging.error(f"Unexpected error copying spec file: {e}")

            # Generate human-readable markdown report
            try:
                write_run_report(result, self.run_dir)
            except (OSError, IOError) as e:
                logging.warning(f"Failed to write run report: {e}")
            except Exception as e:
                logging.error(f"Unexpected error writing run report: {e}")

        return result
