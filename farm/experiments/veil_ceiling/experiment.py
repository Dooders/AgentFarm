"""Matrix orchestration: every condition x inheritance mode x seed, in parallel.

Raw outputs are written per cell so the analysis can be re-run from disk
without re-simulating (Appendix B: every figure reproducible from the seeds).
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from farm.experiments.veil_ceiling.config import (
    CONDITIONS,
    INHERITANCE_MODES,
    PRIMARY_CONDITION_ORDER,
    SEEDS_PER_CELL,
    THRESHOLDS,
    Condition,
    LearnerConfig,
    RunConfig,
    WorldConfig,
    robustness_conditions,
)
from farm.experiments.veil_ceiling.simulation import RunResult, run_simulation

RUNS_FILENAME = "runs.csv"
WINDOWS_FILENAME = "windows.csv"
CELLS_DIRNAME = "cells"
AGENTS_FILENAME = "agents.csv.gz"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class MatrixConfig:
    """Which cells to run and with what shared world/learner parameters."""

    seeds: tuple[int, ...] = tuple(range(1, SEEDS_PER_CELL + 1))
    inheritance_modes: tuple[str, ...] = INHERITANCE_MODES
    condition_names: tuple[str, ...] = PRIMARY_CONDITION_ORDER
    include_robustness: bool = True
    train_ticks: int = 1500
    eval_ticks: int = 300
    window_ticks: int = 50
    world: WorldConfig = field(default_factory=WorldConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    workers: int = 4

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        unknown = [m for m in self.inheritance_modes if m not in INHERITANCE_MODES]
        if unknown:
            raise ValueError(f"unknown inheritance modes: {unknown}")
        unknown_c = [c for c in self.condition_names if c not in CONDITIONS]
        if unknown_c:
            raise ValueError(f"unknown conditions: {unknown_c}")
        if self.workers < 1:
            raise ValueError("workers must be positive")

    def conditions(self) -> dict[str, Condition]:
        out = {name: CONDITIONS[name] for name in self.condition_names}
        if self.include_robustness:
            out.update(robustness_conditions())
        return out

    def run_configs(self) -> list[RunConfig]:
        configs: list[RunConfig] = []
        for condition in self.conditions().values():
            for mode in self.inheritance_modes:
                for seed in self.seeds:
                    configs.append(
                        RunConfig(
                            condition=condition,
                            seed=seed,
                            inheritance_mode=mode,
                            train_ticks=self.train_ticks,
                            eval_ticks=self.eval_ticks,
                            window_ticks=self.window_ticks,
                            world=self.world,
                            learner=self.learner,
                        )
                    )
        return configs


@dataclass
class MatrixOutputs:
    """Everything the analysis needs, either fresh from a run or loaded from disk."""

    runs: pd.DataFrame
    windows: pd.DataFrame
    agents_by_cell: dict[str, pd.DataFrame]
    train_ticks: int

    def cells(self) -> list[str]:
        return sorted(self.agents_by_cell)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def matrix_manifest(matrix: MatrixConfig) -> dict:
    """Pre-registered parameters actually used, plus provenance."""
    conditions = {
        name: {
            "family": c.family,
            "coverage": c.monitoring.coverage,
            "fidelity": c.monitoring.fidelity,
            "penalty": c.monitoring.penalty,
            "decorrelated": c.monitoring.decorrelated,
            "epoch_ticks": c.monitoring.epoch_ticks,
            "expected_penalty_per_defection": c.monitoring.expected_penalty_per_defection,
            "description": c.description,
        }
        for name, c in matrix.conditions().items()
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seeds": list(matrix.seeds),
        "seeds_per_cell": len(matrix.seeds),
        "inheritance_modes": list(matrix.inheritance_modes),
        "train_ticks": matrix.train_ticks,
        "eval_ticks": matrix.eval_ticks,
        "window_ticks": matrix.window_ticks,
        "world": asdict(matrix.world),
        "learner": asdict(matrix.learner),
        "conditions": conditions,
        "analysis_thresholds": asdict(THRESHOLDS),
        "n_runs": len(matrix.run_configs()),
    }


def _run_one(cfg: RunConfig) -> RunResult:
    return run_simulation(cfg)


def _attach_run_columns(df: pd.DataFrame, result: RunResult) -> pd.DataFrame:
    df = df.copy()
    cfg = result.config
    df.insert(0, "run_id", cfg.run_id)
    df.insert(1, "cell_id", cfg.cell_id)
    df.insert(2, "condition", cfg.condition.name)
    df.insert(3, "family", cfg.condition.family)
    df.insert(4, "inheritance_mode", cfg.inheritance_mode)
    df.insert(5, "seed", cfg.seed)
    return df


def run_matrix(
    matrix: MatrixConfig,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
    resume: bool = True,
) -> MatrixOutputs:
    """Simulate every cell and persist raw outputs under ``output_dir``."""
    out = Path(output_dir)
    cells_dir = out / CELLS_DIRNAME
    cells_dir.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST_FILENAME).write_text(json.dumps(matrix_manifest(matrix), indent=2), encoding="utf-8")

    by_cell: dict[str, list[RunConfig]] = {}
    for cfg in matrix.run_configs():
        by_cell.setdefault(cfg.cell_id, []).append(cfg)

    todo = [
        cfg
        for cell, cfgs in by_cell.items()
        for cfg in cfgs
        if not (resume and (cells_dir / cell / AGENTS_FILENAME).exists())
    ]
    n_total = sum(len(cfgs) for cfgs in by_cell.values())
    if progress:
        progress(f"{len(todo)} runs to simulate across {len(by_cell)} cells ({n_total - len(todo)} resumed)")

    results_by_cell: dict[str, list[RunResult]] = {}
    pending = dict(Counter(cfg.cell_id for cfg in todo))
    done = 0
    with ProcessPoolExecutor(max_workers=matrix.workers) as pool:
        futures = {pool.submit(_run_one, cfg): cfg for cfg in todo}
        for future in as_completed(futures):
            result = future.result()
            cell = result.config.cell_id
            results_by_cell.setdefault(cell, []).append(result)
            pending[cell] -= 1
            done += 1
            if pending[cell] == 0:
                _write_cell(cells_dir / cell, results_by_cell.pop(cell))
            if progress and (done % 10 == 0 or done == len(todo)):
                progress(f"{done}/{len(todo)} runs complete")
    return load_outputs(out, train_ticks=matrix.train_ticks)


def _write_cell(cell_dir: Path, results: list[RunResult]) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda r: r.config.seed)
    agents = pd.concat([_attach_run_columns(r.agents, r) for r in results], ignore_index=True)
    windows = pd.concat([_attach_run_columns(r.windows, r) for r in results], ignore_index=True)
    runs = pd.DataFrame([r.summary for r in results])
    agents.to_csv(cell_dir / AGENTS_FILENAME, index=False, compression="gzip")
    windows.to_csv(cell_dir / WINDOWS_FILENAME, index=False)
    runs.to_csv(cell_dir / RUNS_FILENAME, index=False)


def load_outputs(output_dir: str | Path, train_ticks: int | None = None) -> MatrixOutputs:
    """Reload raw per-cell outputs written by :func:`run_matrix`."""
    out = Path(output_dir)
    cells_dir = out / CELLS_DIRNAME
    agents_by_cell: dict[str, pd.DataFrame] = {}
    windows_frames: list[pd.DataFrame] = []
    runs_frames: list[pd.DataFrame] = []
    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
        agents_path = cell_dir / AGENTS_FILENAME
        if not agents_path.exists():
            continue
        agents_by_cell[cell_dir.name] = pd.read_csv(agents_path)
        windows_frames.append(pd.read_csv(cell_dir / WINDOWS_FILENAME))
        runs_frames.append(pd.read_csv(cell_dir / RUNS_FILENAME))
    if not runs_frames:
        raise FileNotFoundError(f"no completed cells under {cells_dir}")
    runs = pd.concat(runs_frames, ignore_index=True)
    windows = pd.concat(windows_frames, ignore_index=True)
    if train_ticks is None:
        manifest = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        train_ticks = int(manifest["train_ticks"])
    runs.to_csv(out / RUNS_FILENAME, index=False)
    return MatrixOutputs(runs=runs, windows=windows, agents_by_cell=agents_by_cell, train_ticks=train_ticks)
