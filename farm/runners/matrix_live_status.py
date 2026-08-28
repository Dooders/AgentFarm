"""Live status snapshots for long matrix runs (SSH-independent monitoring).

Writes an atomic JSON status file under the matrix output directory and, when
running on GCE with guest attributes enabled, mirrors a compact copy to the
instance guest-attribute namespace so operators can poll status via
``gcloud compute instances get-guest-attributes`` without SSH.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from farm.utils.logging import get_logger

logger = get_logger(__name__)

LIVE_STATUS_FILENAME = "matrix_live_status.json"
GUEST_ATTR_NAMESPACE = "status"
# Guest-attribute keys may only use letters, numbers, underscores, hyphens
# (a trailing ".json" is rejected by the metadata server with HTTP 400).
GUEST_ATTR_KEY = "matrix"
_PROGRESS_RE = re.compile(
    r"Simulation progress:\s*(?P<pct>\d+)%\|.*?\|?\s*(?P<cur>\d+)/(?P<total>\d+)",
    re.MULTILINE,
)
_GCE_METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1/"


@dataclass
class ActiveRunProgress:
    """Best-effort progress parsed from a seed log's tqdm line."""

    cell: str
    seed: int
    log_path: str
    percent: Optional[int] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None


@dataclass
class MatrixLiveStatus:
    """Operator-facing snapshot of matrix progress."""

    updated_at: str
    output_dir: str
    total_jobs: int
    n_ok: int
    n_fail: int
    n_done: int
    n_pending: int
    workers: int
    running: list[ActiveRunProgress] = field(default_factory=list)
    recent: list[dict[str, Any]] = field(default_factory=list)
    hostname: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_log_tail(log_path: Path, *, max_bytes: int = 16384) -> str:
    """Read the end of a log file (tqdm rewrites; full reads can stall status)."""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
        return data.decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return ""


def parse_seed_log_progress(log_path: Path) -> Optional[tuple[int, int, int]]:
    """Return ``(percent, current, total)`` from the latest tqdm line, if any."""
    if not log_path.is_file():
        return None
    text = _read_log_tail(log_path)
    if not text:
        return None
    matches = list(_PROGRESS_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return int(match.group("pct")), int(match.group("cur")), int(match.group("total"))


def count_completed_seed_runs(output_dir: Path, num_steps: int) -> int:
    """Count finished ``stable_balanced/seed_*`` runs under a matrix output directory.

    The orchestrator launches the balanced profile. Extra ``stable_*`` trees
    (conservative, buffered) must not inflate ``n_ok`` past the launch total.

    A run counts as complete when ``intrinsic_evolution_metadata.json`` exists
    and ``num_steps_completed >= num_steps``. This is the durable source of truth
    for ``--resume`` progress (independent of master-log parsing).
    """
    if not output_dir.is_dir():
        return 0
    completed = 0
    for meta_path in output_dir.glob("*/stable_balanced/seed_*/intrinsic_evolution_metadata.json"):
        try:
            with meta_path.open(encoding="utf-8") as handle:
                meta = json.load(handle)
            steps = meta.get("num_steps_completed")
            if steps is not None and int(steps) >= int(num_steps):
                completed += 1
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return completed


def scan_active_runs(
    output_dir: Path,
    *,
    limit: int = 16,
    max_age_seconds: float = 600.0,
) -> list[ActiveRunProgress]:
    """Scan recent seed logs under ``output_dir`` for tqdm progress.

    Only logs touched within ``max_age_seconds`` are treated as running, so
    leftover tqdm lines from earlier interrupted attempts do not clutter status.
    """
    found: list[ActiveRunProgress] = []
    if not output_dir.is_dir():
        return found
    log_paths = sorted(
        output_dir.glob("pop-*/seed_*.log"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for log_path in log_paths:
        if len(found) >= limit:
            break
        # seed_42.log → 42
        try:
            seed = int(log_path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        mtime_age = time.time() - log_path.stat().st_mtime
        if mtime_age > max_age_seconds:
            continue
        parsed = parse_seed_log_progress(log_path)
        # Skip stale finished logs with 100% unless mtime is very recent.
        if parsed is not None and parsed[0] >= 100 and mtime_age > 120:
            continue
        if parsed is None and mtime_age > 300:
            continue
        item = ActiveRunProgress(
            cell=log_path.parent.name,
            seed=seed,
            log_path=str(log_path),
        )
        if parsed is not None:
            item.percent, item.current_step, item.total_steps = parsed
        found.append(item)
    return found


def build_live_status(
    *,
    output_dir: Path,
    total_jobs: int,
    n_ok: int,
    n_fail: int,
    workers: int,
    recent: Iterable[dict[str, Any]] | None = None,
    note: str = "",
) -> MatrixLiveStatus:
    """Assemble a status snapshot from counters + on-disk seed logs."""
    n_done = int(n_ok) + int(n_fail)
    return MatrixLiveStatus(
        updated_at=_utc_now_iso(),
        output_dir=str(output_dir),
        total_jobs=int(total_jobs),
        n_ok=int(n_ok),
        n_fail=int(n_fail),
        n_done=n_done,
        n_pending=max(0, int(total_jobs) - n_done),
        workers=int(workers),
        running=scan_active_runs(output_dir),
        recent=list(recent or [])[-12:],
        hostname=os.uname().nodename if hasattr(os, "uname") else "",
        note=note,
    )


def write_live_status(output_dir: Path, status: MatrixLiveStatus) -> Path:
    """Atomically write ``matrix_live_status.json`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / LIVE_STATUS_FILENAME
    payload = (json.dumps(status.to_dict(), indent=2) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".live_status_", dir=str(output_dir))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def _on_gce() -> bool:
    try:
        request = urllib.request.Request(
            _GCE_METADATA_ROOT + "instance/id",
            headers={"Metadata-Flavor": "Google"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=1.5) as response:
            return response.status == 200
    except Exception:
        return False


def publish_guest_attribute(status: MatrixLiveStatus) -> bool:
    """Publish compact status JSON to GCE guest attributes when available.

    Requires the instance metadata flag ``enable-guest-attributes=TRUE``.
    Returns True on success, False when not on GCE or publish fails.
    """
    if not _on_gce():
        return False
    # Keep the guest-attribute payload small (attribute value limits apply).
    # Include recent completions so operators can see failures without SSH.
    recent_compact = []
    for item in list(status.recent or [])[-6:]:
        entry = {
            "status": item.get("status"),
            "rc": item.get("returncode"),
            "seed": item.get("seed"),
            "population": item.get("population"),
            "pressure": item.get("pressure"),
            "gene_flow": item.get("gene_flow"),
            "elapsed": item.get("elapsed_seconds"),
        }
        tail = item.get("log_tail")
        if tail and item.get("status") != "ok":
            # Keep guest-attribute payloads small; last line is usually enough.
            entry["err"] = str(tail).replace("\n", " | ")[-220:]
        recent_compact.append(entry)
    compact = {
        "updated_at": status.updated_at,
        "n_ok": status.n_ok,
        "n_fail": status.n_fail,
        "n_done": status.n_done,
        "n_pending": status.n_pending,
        "total_jobs": status.total_jobs,
        "workers": status.workers,
        "note": status.note,
        "running": [
            {
                "cell": item.cell,
                "seed": item.seed,
                "percent": item.percent,
                "current_step": item.current_step,
                "total_steps": item.total_steps,
            }
            for item in status.running[:8]
        ],
        "recent": recent_compact,
        "hostname": status.hostname,
    }
    body = json.dumps(compact, separators=(",", ":")).encode("utf-8")
    url = (
        f"{_GCE_METADATA_ROOT}instance/guest-attributes/"
        f"{GUEST_ATTR_NAMESPACE}/{GUEST_ATTR_KEY}"
    )
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Metadata-Flavor": "Google",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            ok = 200 <= response.status < 300
            if ok:
                logger.debug("matrix_status_guest_attribute_published", bytes=len(body))
            return ok
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("matrix_status_guest_attribute_publish_failed", error=str(exc))
        return False


def publish_live_status(
    output_dir: Path,
    *,
    total_jobs: int,
    n_ok: int,
    n_fail: int,
    workers: int,
    recent: Iterable[dict[str, Any]] | None = None,
    note: str = "",
    guest_attributes: bool = True,
) -> MatrixLiveStatus:
    """Write the live status file and optionally mirror to guest attributes."""
    status = build_live_status(
        output_dir=output_dir,
        total_jobs=total_jobs,
        n_ok=n_ok,
        n_fail=n_fail,
        workers=workers,
        recent=recent,
        note=note,
    )
    write_live_status(output_dir, status)
    if guest_attributes:
        publish_guest_attribute(status)
    return status
