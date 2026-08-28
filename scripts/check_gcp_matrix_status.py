#!/usr/bin/env python3
"""Check intrinsic-evolution matrix progress without relying on SSH.

Primary path: GCE guest attributes published by the matrix orchestrator
(``status/matrix``). This works even when the guest is CPU-saturated and
``sshd`` no longer answers.

Fallback: SSH + read ``matrix_live_status.json`` on the VM (may time out under
load — that is why guest attributes exist).

Examples
--------
::

    python scripts/check_gcp_matrix_status.py
    python scripts/check_gcp_matrix_status.py --instance agentfarm-sweep --zone us-central1-a
    python scripts/check_gcp_matrix_status.py --watch 60
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any


def _run_gcloud(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gcloud", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def fetch_guest_attribute_status(
    *,
    instance: str,
    zone: str,
    project: str | None,
    timeout: float,
) -> dict[str, Any] | None:
    """Return parsed guest-attribute JSON, or None if unavailable."""
    cmd = [
        "compute",
        "instances",
        "get-guest-attributes",
        instance,
        f"--zone={zone}",
        "--query-path=status/matrix",
        "--format=get(value)",
    ]
    if project:
        cmd.append(f"--project={project}")
    try:
        completed = _run_gcloud(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("error: gcloud guest-attributes timed out", file=sys.stderr)
        return None
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        print(f"guest-attributes unavailable: {err}", file=sys.stderr)
        return None
    raw = (completed.stdout or "").strip()
    if not raw:
        print("guest-attributes empty (matrix not started, or attrs disabled)", file=sys.stderr)
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: guest-attributes value is not JSON: {exc}", file=sys.stderr)
        return None


def fetch_status_via_ssh(
    *,
    instance: str,
    zone: str,
    project: str | None,
    remote_path: str,
    timeout: float,
) -> dict[str, Any] | None:
    """Best-effort SSH fallback; often fails under worker load."""
    remote_cmd = f"cat {remote_path}"
    cmd = [
        "compute",
        "ssh",
        instance,
        f"--zone={zone}",
        "--tunnel-through-iap",
        "--ssh-flag=-o ConnectTimeout=15",
        "--ssh-flag=-o ServerAliveInterval=5",
        "--ssh-flag=-o ServerAliveCountMax=2",
        f"--command={remote_cmd}",
    ]
    if project:
        cmd.append(f"--project={project}")
    try:
        completed = _run_gcloud(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("error: SSH status fallback timed out (expected under load)", file=sys.stderr)
        return None
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        print(f"SSH status fallback failed: {err}", file=sys.stderr)
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(f"error: remote status file is not JSON: {exc}", file=sys.stderr)
        return None


def format_status(status: dict[str, Any], *, source: str) -> str:
    """Human-readable one-screen summary."""
    lines = [
        f"source          : {source}",
        f"updated_at      : {status.get('updated_at', '?')}",
        f"hostname        : {status.get('hostname', '?')}",
        (
            f"progress        : {status.get('n_done', '?')}/"
            f"{status.get('total_jobs', '?')} done "
            f"({status.get('n_ok', '?')} ok, {status.get('n_fail', '?')} fail, "
            f"{status.get('n_pending', '?')} pending)"
        ),
        f"workers         : {status.get('workers', '?')}",
    ]
    if status.get("note"):
        lines.append(f"note            : {status['note']}")
    running = status.get("running") or []
    if running:
        lines.append("running:")
        for item in running:
            pct = item.get("percent")
            cur = item.get("current_step")
            total = item.get("total_steps")
            if pct is not None and cur is not None and total is not None:
                prog = f"{pct}% ({cur}/{total})"
            elif pct is not None:
                prog = f"{pct}%"
            else:
                prog = "progress unknown"
            lines.append(f"  - {item.get('cell')} seed={item.get('seed')}: {prog}")
    else:
        lines.append("running         : (none detected / logs not yet writing tqdm)")
    recent = status.get("recent") or []
    if recent:
        lines.append("recent:")
        for item in recent[-6:]:
            cell = (
                f"pop-{item.get('population')}__pressure-{item.get('pressure')}"
                f"__geneflow-{item.get('gene_flow')}"
            )
            lines.append(
                f"  - {item.get('status')} rc={item.get('rc', item.get('returncode'))} "
                f"{cell} seed={item.get('seed')} "
                f"({item.get('elapsed', item.get('elapsed_seconds'))}s)"
            )
            err = item.get("err") or item.get("log_tail")
            if err and item.get("status") != "ok":
                lines.append(f"      err: {err}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poll matrix live status via GCE guest attributes (no SSH required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--instance", default="agentfarm-sweep")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--project", default=None, help="GCP project (default: gcloud config).")
    parser.add_argument(
        "--remote-status-path",
        default="~/AgentFarm/experiments/intrinsic_matrix/matrix_live_status.json",
        help="Path used only for the SSH fallback.",
    )
    parser.add_argument(
        "--allow-ssh-fallback",
        action="store_true",
        help="If guest attributes are empty/unavailable, try SSH (may hang under load).",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="gcloud command timeout (s).")
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Re-poll every SECONDS until interrupted (0 = once).",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON only.")
    return parser


def _once(args: argparse.Namespace) -> int:
    status = fetch_guest_attribute_status(
        instance=args.instance,
        zone=args.zone,
        project=args.project,
        timeout=args.timeout,
    )
    source = "guest-attributes:status/matrix"
    if status is None and args.allow_ssh_fallback:
        status = fetch_status_via_ssh(
            instance=args.instance,
            zone=args.zone,
            project=args.project,
            remote_path=args.remote_status_path,
            timeout=args.timeout,
        )
        source = f"ssh:{args.remote_status_path}"
    if status is None:
        return 1
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(format_status(status, source=source))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.watch and args.watch > 0:
        while True:
            code = _once(args)
            if code != 0:
                print("(retrying…)", file=sys.stderr)
            try:
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print(file=sys.stderr)
                return 0
            print(file=sys.stderr)
    return _once(args)


if __name__ == "__main__":
    raise SystemExit(main())
