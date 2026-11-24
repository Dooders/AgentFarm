"""Matrix runner: conditions x seeds, each paired with its solo twin.

The solo twin depends only on the seed (cooperation off, same world), so it
is simulated once per seed and shared by every condition. Outputs per cell:

* ``runs.csv`` -- one row per run (:func:`~farm.experiments.cooperation.metrics.summarize_run`
  plus the founder welfare summary).
* ``founders.csv.gz`` -- one row per founder per run, paired with its twin.
* ``units.csv.gz`` -- one row per unit with its diagnosis.
* ``timeseries.csv.gz`` -- per-tick population and coverage (every ``ts_stride`` ticks).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from farm.core.cooperation import UnitEconomics, UnitGoals
from farm.experiments.cooperation.config import (
    CONDITIONS,
    PRIMARY_CONDITION_ORDER,
    SECONDARY_CONDITION_ORDER,
    THRESHOLDS,
    DecisionConfig,
    RunConfig,
    WorldConfig,
)
from farm.experiments.cooperation.metrics import RMST_TAU, founder_pairs, summarize_run, welfare_summary
from farm.experiments.cooperation.simulation import RunResult, run_simulation

#: Seeds used for the registered run. Seeds 1-2 were used for calibration
#: pilots and are deliberately excluded.
CONFIRMATORY_SEEDS: tuple[int, ...] = tuple(range(101, 121))

MANIFEST_FILENAME = "manifest.json"
RUNS_FILENAME = "runs.csv"
FOUNDERS_FILENAME = "founders.csv.gz"
UNITS_FILENAME = "units.csv.gz"
TIMESERIES_FILENAME = "timeseries.csv.gz"
CELLS_DIRNAME = "cells"
CELL_META_FILENAME = "cell.json"
SOLO_DIRNAME = "solo"


@dataclass(frozen=True)
class MatrixConfig:
    """Which cells to run and with which shared parameters."""

    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS
    condition_names: tuple[str, ...] = PRIMARY_CONDITION_ORDER + SECONDARY_CONDITION_ORDER
    world: WorldConfig = field(default_factory=WorldConfig)
    decisions: DecisionConfig = field(default_factory=DecisionConfig)
    economics: UnitEconomics = field(default_factory=UnitEconomics)
    unit_goals: UnitGoals = field(default_factory=UnitGoals)
    workers: int = 2
    ts_stride: int = 5

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        unknown = [c for c in self.condition_names if c not in CONDITIONS]
        if unknown:
            raise ValueError(f"unknown conditions: {unknown}")
        if self.workers < 1 or self.ts_stride < 1:
            raise ValueError("workers and ts_stride must be positive")

    def run_config(self, condition_name: str, seed: int, cooperative: bool = True) -> RunConfig:
        return RunConfig(
            condition=CONDITIONS[condition_name],
            seed=seed,
            world=self.world,
            decisions=self.decisions,
            economics=self.economics,
            unit_goals=self.unit_goals,
            cooperative=cooperative,
        )

    def solo_configs(self) -> list[RunConfig]:
        anchor = self.condition_names[0]
        return [self.run_config(anchor, seed, cooperative=False) for seed in self.seeds]

    def cooperative_configs(self) -> list[RunConfig]:
        return [self.run_config(name, seed) for name in self.condition_names for seed in self.seeds]


@dataclass
class MatrixOutputs:
    """Everything the analysis needs, fresh from a run or loaded from disk."""

    runs: pd.DataFrame
    founders: pd.DataFrame
    units: pd.DataFrame
    timeseries: pd.DataFrame
    solo_runs: pd.DataFrame
    manifest: dict


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def matrix_manifest(matrix: MatrixConfig) -> dict:
    """Registered parameters actually used, plus provenance."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seeds": list(matrix.seeds),
        "conditions": {name: CONDITIONS[name].to_dict() for name in matrix.condition_names},
        "world": asdict(matrix.world),
        "decisions": asdict(matrix.decisions),
        "economics": asdict(matrix.economics),
        "unit_goals": asdict(matrix.unit_goals),
        "thresholds": asdict(THRESHOLDS),
        "rmst_tau": RMST_TAU,
        "n_runs": len(matrix.cooperative_configs()) + len(matrix.seeds),
    }


def _signature(manifest: dict) -> dict:
    keys = ("seeds", "world", "decisions", "economics", "unit_goals", "thresholds", "rmst_tau")
    return {k: manifest.get(k) for k in keys}


def _run_one(cfg: RunConfig) -> RunResult:
    return run_simulation(cfg)


def _attach(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "seed", cfg.seed)
    df.insert(0, "condition", cfg.condition.name if cfg.cooperative else "solo")
    df.insert(0, "run_id", cfg.run_id)
    return df


def _write_frames(directory: Path, runs: pd.DataFrame, founders: pd.DataFrame | None, units, timeseries) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    runs.to_csv(directory / RUNS_FILENAME, index=False)
    if founders is not None:
        founders.to_csv(directory / FOUNDERS_FILENAME, index=False, compression="gzip")
    if units is not None:
        units.to_csv(directory / UNITS_FILENAME, index=False, compression="gzip")
    timeseries.to_csv(directory / TIMESERIES_FILENAME, index=False, compression="gzip")


def _cell_meta(manifest: dict, condition: str) -> dict:
    """What a cell's outputs were produced under: its definition and the run signature."""
    return {"condition": manifest["conditions"][condition], "signature": _signature(manifest)}


def _cell_complete(directory: Path, seeds: Sequence[int], meta: dict | None = None) -> bool:
    """Outputs exist for exactly ``seeds`` and (when ``meta`` is given) were produced under ``meta``."""
    if meta is not None:
        try:
            recorded = json.loads((directory / CELL_META_FILENAME).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return False
        if recorded != meta:
            return False
    try:
        runs = pd.read_csv(directory / RUNS_FILENAME, usecols=["seed"])
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        return False
    return sorted(runs["seed"].tolist()) == sorted(seeds) and all(
        (directory / name).exists() for name in (FOUNDERS_FILENAME, UNITS_FILENAME, TIMESERIES_FILENAME)
    )


def run_matrix(
    matrix: MatrixConfig,
    output_dir: str | Path,
    progress: Callable[[str], None] | None = None,
    resume: bool = True,
) -> MatrixOutputs:
    """Simulate solo twins and every cooperative cell; persist raw outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Round-trip through JSON so a fresh manifest compares equal to one read back from disk.
    manifest = json.loads(json.dumps(matrix_manifest(matrix), default=str))
    manifest_path = out / MANIFEST_FILENAME
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _signature(existing) != _signature(manifest):
            raise ValueError(f"incompatible output directory for resume: {out}")
        previous = existing.get("conditions", {})
        changed = sorted(n for n, spec in manifest["conditions"].items() if n in previous and previous[n] != spec)
        if changed:
            raise ValueError(f"condition definitions changed since the outputs in {out} were made: {changed}")
        manifest["conditions"] = {**previous, **manifest["conditions"]}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    log = progress or (lambda _msg: None)
    solo_dir = out / SOLO_DIRNAME
    solo_results: dict[int, RunResult] = {}
    with ProcessPoolExecutor(max_workers=matrix.workers) as pool:
        log(f"simulating {len(matrix.seeds)} solo twins")
        futures = [pool.submit(_run_one, cfg) for cfg in matrix.solo_configs()]
        for future in as_completed(futures):
            result = future.result()
            solo_results[result.config.seed] = result
        solo_runs = pd.DataFrame([summarize_run(solo_results[s]) for s in sorted(solo_results)])
        solo_ts = pd.concat(
            [
                _attach(solo_results[s].timeseries.iloc[:: matrix.ts_stride], solo_results[s].config)
                for s in sorted(solo_results)
            ],
            ignore_index=True,
        )
        _write_frames(solo_dir, solo_runs, None, None, solo_ts)

        cells_dir = out / CELLS_DIRNAME
        todo: dict[str, list[RunConfig]] = {}
        for cfg in matrix.cooperative_configs():
            cell = cfg.condition.name
            if resume and _cell_complete(cells_dir / cell, matrix.seeds, _cell_meta(manifest, cell)):
                continue
            todo.setdefault(cell, []).append(cfg)
        n_todo = sum(len(v) for v in todo.values())
        log(
            f"{n_todo} cooperative runs across {len(todo)} cells "
            f"({len(matrix.condition_names) - len(todo)} cells resumed)"
        )
        pending = {cell: len(cfgs) for cell, cfgs in todo.items()}
        collected: dict[str, list[tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = {}
        futures = {pool.submit(_run_one, cfg): cfg for cfgs in todo.values() for cfg in cfgs}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            cfg = result.config
            solo = solo_results[cfg.seed]
            pairs = founder_pairs(result, solo)
            row = summarize_run(result)
            row.update(welfare_summary(pairs))
            collected.setdefault(cfg.condition.name, []).append(
                (
                    row,
                    pairs,
                    _attach(result.units, cfg),
                    _attach(result.timeseries.iloc[:: matrix.ts_stride], cfg),
                )
            )
            pending[cfg.condition.name] -= 1
            if pending[cfg.condition.name] == 0:
                parts = sorted(collected.pop(cfg.condition.name), key=lambda p: p[0]["seed"])
                cell_dir = cells_dir / cfg.condition.name
                _write_frames(
                    cell_dir,
                    pd.DataFrame([p[0] for p in parts]),
                    pd.concat([p[1] for p in parts], ignore_index=True),
                    pd.concat([p[2] for p in parts], ignore_index=True),
                    pd.concat([p[3] for p in parts], ignore_index=True),
                )
                (cell_dir / CELL_META_FILENAME).write_text(
                    json.dumps(_cell_meta(manifest, cfg.condition.name), indent=2, default=str), encoding="utf-8"
                )
            if done % 20 == 0 or done == n_todo:
                log(f"{done}/{n_todo} cooperative runs complete")
    return load_outputs(out, condition_names=matrix.condition_names)


def load_outputs(output_dir: str | Path, condition_names: Sequence[str] | None = None) -> MatrixOutputs:
    """Reload raw outputs written by :func:`run_matrix`.

    Without ``condition_names``, loads the cells the manifest lists, so
    leftover cell directories from an earlier, different run are ignored.
    Directories without a manifest fall back to every cell present.
    """
    out = Path(output_dir)
    cells_dir = out / CELLS_DIRNAME
    manifest_path = out / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if condition_names is not None:
        names = list(condition_names)
    elif manifest.get("conditions"):
        names = sorted(manifest["conditions"])
    else:
        names = sorted(p.name for p in cells_dir.iterdir() if p.is_dir())
    frames: dict[str, list[pd.DataFrame]] = {"runs": [], "founders": [], "units": [], "timeseries": []}
    for name in names:
        cell = cells_dir / name
        if not (cell / RUNS_FILENAME).exists():
            continue
        frames["runs"].append(pd.read_csv(cell / RUNS_FILENAME))
        frames["founders"].append(pd.read_csv(cell / FOUNDERS_FILENAME))
        frames["units"].append(pd.read_csv(cell / UNITS_FILENAME))
        frames["timeseries"].append(pd.read_csv(cell / TIMESERIES_FILENAME))
    if not frames["runs"]:
        raise FileNotFoundError(f"no completed cells under {cells_dir}")
    solo_dir = out / SOLO_DIRNAME
    solo_runs = pd.read_csv(solo_dir / RUNS_FILENAME) if (solo_dir / RUNS_FILENAME).exists() else pd.DataFrame()
    solo_ts = (
        pd.read_csv(solo_dir / TIMESERIES_FILENAME) if (solo_dir / TIMESERIES_FILENAME).exists() else pd.DataFrame()
    )
    runs = pd.concat(frames["runs"], ignore_index=True)
    runs.to_csv(out / RUNS_FILENAME, index=False)
    timeseries = pd.concat(frames["timeseries"] + ([solo_ts] if len(solo_ts) else []), ignore_index=True)
    return MatrixOutputs(
        runs=runs,
        founders=pd.concat(frames["founders"], ignore_index=True),
        units=pd.concat(frames["units"], ignore_index=True),
        timeseries=timeseries,
        solo_runs=solo_runs,
        manifest=manifest,
    )
