from __future__ import annotations

"""
RunSpec loader and validator for benchmark/profiling runs.

Supports YAML or JSON files. Provides a simple validated dict structure
consumable by the Runner and SweepRunner.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


SPEC_DEFAULTS = {
    "iterations": {"warmup": 0, "measured": 1},
    "instrumentation": ["timing"],
    "output_dir": "benchmarks/results",
}


@dataclass
class RunSpec:
    experiment: str
    params: Dict[str, Any]
    iterations: Dict[str, int]
    instrumentation: List[str]
    output_dir: str
    tags: List[str]
    notes: str
    seed: Optional[int]
    sweep: Optional[Dict[str, List[Any]]]
    parallelism: int
    strategy: str
    samples: Optional[int]


def _load_raw(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    if path.endswith(".yaml") or path.endswith(".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is not installed; cannot load YAML spec")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    # Try JSON as a fallback
    with open(path, "r") as f:
        return json.load(f)


def load_spec(path: str) -> RunSpec:
    data = _load_raw(path) or {}

    # Minimal validation
    if "experiment" not in data:
        raise ValueError("Spec missing required 'experiment' field")

    iterations = data.get("iterations", SPEC_DEFAULTS["iterations"]) or {"warmup": 0, "measured": 1}
    if iterations.get("measured", 0) <= 0:
        raise ValueError("'iterations.measured' must be >= 1")

    instrumentation = data.get("instrumentation", SPEC_DEFAULTS["instrumentation"]) or ["timing"]
    output_dir = data.get("output_dir", SPEC_DEFAULTS["output_dir"]) or "benchmarks/results"

    return RunSpec(
        experiment=str(data["experiment"]),
        params=dict(data.get("params", {})),
        iterations={"warmup": int(iterations.get("warmup", 0)), "measured": int(iterations.get("measured", 1))},
        instrumentation=list(instrumentation),
        output_dir=str(output_dir),
        tags=list(data.get("tags", []) or []),
        notes=str(data.get("notes", "")),
        seed=(int(data["seed"]) if "seed" in data and data["seed"] is not None else None),
        sweep=(data.get("sweep") if data.get("sweep") else None),
        parallelism=int(data.get("parallelism", 1)),
        strategy=str(data.get("strategy", "cartesian")),
        samples=(int(data.get("samples")) if data.get("samples") is not None else None),
    )

