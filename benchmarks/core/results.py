from __future__ import annotations

"""
Results model with environment and VCS capture, plus metrics helpers.
"""

import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _safe_git(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return None


def capture_environment() -> Dict[str, Any]:
    env = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }
    # GPU info (best-effort)
    try:
        import subprocess

        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpus.append({"name": parts[0], "memory_total": parts[1]})
        if gpus:
            env["gpus"] = gpus
    except Exception:
        pass
    return env


def capture_vcs() -> Dict[str, Any]:
    commit = _safe_git(["git", "rev-parse", "HEAD"]) or ""
    branch = _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or ""
    status = _safe_git(["git", "status", "--porcelain"]) or ""
    return {"commit": commit, "branch": branch, "dirty": bool(status.strip())}


@dataclass
class Artifact:
    name: str
    type: str
    path: str


@dataclass
class IterationResult:
    index: int
    duration_s: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    name: str
    run_id: str
    parameters: Dict[str, Any]
    iterations: Dict[str, int]
    metrics: Dict[str, Any] = field(default_factory=dict)
    iteration_metrics: List[IterationResult] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=capture_environment)
    vcs: Dict[str, Any] = field(default_factory=capture_vcs)
    artifacts: List[Artifact] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    status: str = "success"

    def add_iteration(
        self, index: int, duration_s: float, metrics: Dict[str, Any]
    ) -> None:
        self.iteration_metrics.append(
            IterationResult(index=index, duration_s=duration_s, metrics=metrics)
        )

    def add_artifact(self, name: str, type: str, path: str) -> None:
        self.artifacts.append(Artifact(name=name, type=type, path=path))

    def get_mean_duration(self) -> float:
        """Get the mean duration of all iterations."""
        if not self.iteration_metrics:
            return 0.0
        durations = [it.duration_s for it in self.iteration_metrics]
        return sum(durations) / len(durations)

    def get_median_duration(self) -> float:
        """Get the median duration of all iterations."""
        if not self.iteration_metrics:
            return 0.0
        durations = sorted([it.duration_s for it in self.iteration_metrics])
        n = len(durations)
        if n % 2 == 0:
            return (durations[n // 2 - 1] + durations[n // 2]) / 2
        else:
            return durations[n // 2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "parameters": self.parameters,
            "iterations": self.iterations,
            "metrics": self.metrics,
            "iteration_metrics": [
                {"index": it.index, "duration_s": it.duration_s, "metrics": it.metrics}
                for it in self.iteration_metrics
            ],
            "environment": self.environment,
            "vcs": self.vcs,
            "artifacts": [
                {"name": a.name, "type": a.type, "path": a.path} for a in self.artifacts
            ],
            "tags": self.tags,
            "notes": self.notes,
            "status": self.status,
        }

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}_{self.run_id}.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
