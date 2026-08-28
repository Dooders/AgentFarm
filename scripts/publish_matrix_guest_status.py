#!/usr/bin/env python3
"""Publish matrix status from disk to GCE guest attributes.

Runs independently of the matrix orchestrator (e.g. via cron) so operators can
still see progress when the parent process is wedged or starved.

Progress counters prefer on-disk completed metadata (same rule as ``--resume``)
so a relaunch that skips already-finished seeds does not look like 0/N done.
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
    count_completed_seed_runs,
    publish_live_status,
)

# Legacy: "Launching 96 runs across 1 worker(s)"
_LAUNCH_LEGACY_RE = re.compile(
    r"Launching\s+(?P<total>\d+)\s+runs\s+across\s+(?P<workers>\d+)\s+worker"
)
# Resume-aware: "Launching 63 remaining of 96 runs (33 already complete) across 1 worker(s)"
_LAUNCH_RESUME_RE = re.compile(
    r"Launching\s+(?P<remaining>\d+)\s+remaining\s+of\s+(?P<total>\d+)\s+runs\s+"
    r"\((?P<skipped>\d+)\s+already complete\)\s+across\s+(?P<workers>\d+)\s+worker"
)
_DONE_RE = re.compile(
    r"\[(?P<done>\d+)/(?P<total>\d+)\]\s+(?P<status>ok|error)\s+"
    r"rc=(?P<rc>-?\d+)\s+(?P<elapsed>[\d.]+)s\s+(?P<cell>\S+)\s+seed=(?P<seed>\d+)"
)
_RETRYABLE_RCS = frozenset({-9, -15})


def _parse_latest_launch(
    master_log: Path,
) -> tuple[int, int, int, int, int, int, list[dict[str, Any]]]:
    """Return launch accounting parsed from the latest master-log start line.

    Returns
    -------
    n_ok, n_fail, n_killed, total_jobs, workers, skipped_at_start, recent
        ``n_ok`` here is completions logged *after* the launch line only (does
        not include the resume skip count). ``skipped_at_start`` comes from the
        resume-aware launch line when present.
    """
    if not master_log.is_file():
        return 0, 0, 0, 0, 0, 0, []
    try:
        text = master_log.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return 0, 0, 0, 0, 0, 0, []

    launch_matches: list[tuple[int, re.Match[str], str]] = []
    for match in _LAUNCH_RESUME_RE.finditer(text):
        launch_matches.append((match.start(), match, "resume"))
    for match in _LAUNCH_LEGACY_RE.finditer(text):
        # Prefer the resume-aware line when both could match nearby; skip legacy
        # hits that are the same span as a resume line (resume contains "runs across").
        if any(abs(match.start() - start) < 8 for start, _, kind in launch_matches if kind == "resume"):
            continue
        launch_matches.append((match.start(), match, "legacy"))
    if not launch_matches:
        return 0, 0, 0, 0, 0, 0, []

    launch_matches.sort(key=lambda item: item[0])
    start, match, kind = launch_matches[-1]
    total = int(match.group("total"))
    workers = int(match.group("workers"))
    skipped_at_start = int(match.group("skipped")) if kind == "resume" else 0

    n_ok = 0
    n_fail = 0
    n_killed = 0
    recent: list[dict[str, Any]] = []
    for done_match in _DONE_RE.finditer(text[start:]):
        rc = int(done_match.group("rc"))
        status = done_match.group("status")
        cell = done_match.group("cell")
        cell_name = Path(cell).name
        entry: dict[str, Any] = {
            "status": status,
            "returncode": rc,
            "elapsed_seconds": float(done_match.group("elapsed")),
            "seed": int(done_match.group("seed")),
            "cell_dir": cell,
        }
        for part in cell_name.split("__"):
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
            n_fail += 1
        else:
            n_fail += 1
        recent.append(entry)
        total = max(total, int(done_match.group("total")))
    return n_ok, n_fail, n_killed, total, workers, skipped_at_start, recent[-6:]


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
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=0, help="0 = detect from live status / log")
    parser.add_argument("--no-guest-attributes", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser()
    master = output_dir / "matrix_master.log"
    (
        log_ok,
        n_fail,
        n_killed,
        total_from_log,
        workers_from_log,
        skipped_at_start,
        recent,
    ) = _parse_latest_launch(master)

    disk_ok = count_completed_seed_runs(output_dir, args.num_steps)
    # Disk metadata is authoritative for finished seeds under --resume. Log ok
    # counts only cover jobs finished *this* launch after the start line.
    n_ok = max(disk_ok, skipped_at_start + log_ok)
    total_jobs = total_from_log or args.total_jobs
    workers = (
        args.workers
        or workers_from_log
        or _workers_from_live_or_default(output_dir, default=1)
    )
    note = f"watchdog pid={os.getpid()}; disk_ok={disk_ok}"
    if skipped_at_start:
        note += f"; skipped_at_start={skipped_at_start}"
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
        f"total={total_jobs} workers={workers} disk_ok={disk_ok}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
