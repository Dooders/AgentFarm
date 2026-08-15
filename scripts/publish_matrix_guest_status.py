#!/usr/bin/env python3
"""Publish matrix status from disk to GCE guest attributes.

Runs independently of the matrix orchestrator (e.g. via cron) so operators can
still see progress when the parent process is wedged or starved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from farm.runners.matrix_live_status import (  # noqa: E402
    LIVE_STATUS_FILENAME,
    publish_live_status,
)

_DONE_RE = re.compile(
    r"\[(?P<done>\d+)/(?P<total>\d+)\]\s+(?P<status>ok|error)\s+"
    r"rc=(?P<rc>-?\d+)\s+(?P<elapsed>[\d.]+)s\s+(?P<cell>\S+)\s+seed=(?P<seed>\d+)"
)
_RETRYABLE_RCS = frozenset({-9, -15})


def _parse_latest_launch(master_log: Path) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    """Return ``(n_ok, n_fail, n_killed, total_jobs, recent)`` for the latest launch."""
    if not master_log.is_file():
        return 0, 0, 0, 0, []
    try:
        text = master_log.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return 0, 0, 0, 0, []

    launches = list(re.finditer(r"Launching\s+(\d+)\s+runs\s+across\s+(\d+)\s+worker", text))
    if not launches:
        return 0, 0, 0, 0, []
    start = launches[-1].start()
    total = int(launches[-1].group(1))
    n_ok = 0
    n_fail = 0
    n_killed = 0
    recent: list[dict[str, Any]] = []
    for match in _DONE_RE.finditer(text[start:]):
        rc = int(match.group("rc"))
        status = match.group("status")
        cell = match.group("cell")
        # cell path ends with pop-...; derive axes when possible
        cell_name = Path(cell).name
        entry: dict[str, Any] = {
            "status": status,
            "returncode": rc,
            "elapsed_seconds": float(match.group("elapsed")),
            "seed": int(match.group("seed")),
            "cell_dir": cell,
        }
        parts = cell_name.split("__")
        for part in parts:
            if part.startswith("pop-"):
                entry["population"] = part[4:]
            elif part.startswith("pressure-"):
                entry["pressure"] = part[len("pressure-") :]
            elif part.startswith("geneflow-"):
                entry["gene_flow"] = part[len("geneflow-") :]
        if status == "ok":
            n_ok += 1
        elif rc in _RETRYABLE_RCS:
            n_killed += 1
            entry["status"] = "killed"
            entry["log_tail"] = f"rc={rc} SIGKILL/SIGTERM (not a sim error)"
            n_fail += 1  # still unfinished for this launch's counter
        else:
            n_fail += 1
        recent.append(entry)
        total = max(total, int(match.group("total")))
    return n_ok, n_fail, n_killed, total, recent[-6:]


def _workers_from_live_or_default(output_dir: Path, default: int) -> int:
    path = output_dir / LIVE_STATUS_FILENAME
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return int(payload.get("workers") or default)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "AgentFarm" / "experiments" / "intrinsic_matrix",
    )
    parser.add_argument("--total-jobs", type=int, default=96)
    parser.add_argument("--workers", type=int, default=0, help="0 = detect from live status / log")
    parser.add_argument("--no-guest-attributes", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser()
    master = output_dir / "matrix_master.log"
    n_ok, n_fail, n_killed, total_from_log, recent = _parse_latest_launch(master)
    total_jobs = total_from_log or args.total_jobs
    workers = args.workers or _workers_from_live_or_default(output_dir, default=3)
    note = f"watchdog pid={os.getpid()}"
    if n_killed:
        note += f"; {n_killed} killed(rc=-9/-15) will retry on relaunch"

    publish_live_status(
        output_dir,
        total_jobs=total_jobs,
        n_ok=n_ok,
        n_fail=n_fail,
        workers=workers,
        recent=recent,
        note=note,
        guest_attributes=not args.no_guest_attributes,
    )
    print(
        f"published n_ok={n_ok} n_fail={n_fail} n_killed={n_killed} "
        f"total={total_jobs} workers={workers}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
