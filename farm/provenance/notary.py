"""AgentFarm adapter for FarmNotary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

RUN_CONFIG_NAME = "run_config.json"

# Explicit FarmNotary allowlist. Voter-level files are also denied by name
# fragment (ballot, vote, voter, individual_choice, private).
OFFICIAL_PUBLISH_PATTERNS: tuple[str, ...] = (
    "trials.csv",
    "summary.csv",
    "allocation_means.csv",
    "contrasts.csv",
    "run_config.json",
    "REPORT.md",
    "figures/*.png",
)


def _require_farm_notary():
    try:
        import farm_notary
    except ImportError as exc:
        raise ImportError(
            "Run notarization needs the optional farm-notary package: "
            "pip install farm-notary (add farm-notary[ots] to anchor via "
            "OpenTimestamps)"
        ) from exc
    return farm_notary


def farm_notary_available() -> bool:
    try:
        _require_farm_notary()
    except ImportError:
        return False
    return True


def _read_run_config(run_dir: Path) -> dict:
    """The experiment's own run_config.json, when present."""
    path = Path(run_dir) / RUN_CONFIG_NAME
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def _git_sha() -> str | None:
    """Return the HEAD SHA of the AgentFarm checkout, or None."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / ".git").exists():
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def notarize(
    run_dir: Path,
    *,
    config: Mapping[str, Any] | None = None,
    command: str | None = None,
    runner: str = "agentfarm",
    official_record: Mapping[str, Any] | None = None,
    publish_patterns: Sequence[str] | None = None,
    backend: str = "dry-run",
    calendars: Sequence[str] | None = None,
    pin: bool = False,
    ipfs_api: str | None = None,
    lockfile: Path | None = None,
) -> tuple[Any, Any]:
    """Notarize a finished run directory; returns (manifest, receipt)."""
    fn = _require_farm_notary()
    from farm_notary.anchor import get_backend

    run_config = _read_run_config(run_dir)
    return fn.notarize_run(
        Path(run_dir),
        publish_patterns=list(publish_patterns or OFFICIAL_PUBLISH_PATTERNS),
        config=dict(config) if config is not None else run_config.get("config", {}),
        command=command or run_config.get("command"),
        runner=runner,
        lockfile=lockfile,
        official_record=official_record,
        backend=get_backend(backend, calendars=calendars),
        pin=pin,
        ipfs_api=ipfs_api,
    )


def verify(run_dir: Path) -> list:
    """All integrity problems for a notarized run directory ([] means OK)."""
    fn = _require_farm_notary()
    from farm_notary.verify import verify_anchor, verify_receipt, verify_run_dir

    run_dir = Path(run_dir)
    manifest = fn.load_manifest(run_dir)
    return (
        verify_run_dir(manifest, run_dir)
        + verify_anchor(manifest, run_dir)
        + verify_receipt(manifest, run_dir)
    )


def reproduce(run_dir: Path, *, ignore: Sequence[str] = (), anchor: bool = False):
    """Re-run the notarized command and byte-compare; returns the result."""
    fn = _require_farm_notary()
    from farm_notary.reproduce import (
        RECEIPT_PROOF_NAME,
        build_receipt,
        receipt_hash,
        reproduce_run,
        write_receipt,
    )

    run_dir = Path(run_dir)
    manifest = fn.load_manifest(run_dir)
    result = reproduce_run(manifest, ignore=ignore)
    receipt = build_receipt(manifest, result)
    write_receipt(receipt, run_dir)
    if anchor:
        from farm_notary.ots import stamp_digest

        proof, _ = stamp_digest(bytes.fromhex(receipt_hash(receipt)))
        (run_dir / RECEIPT_PROOF_NAME).write_bytes(proof)
    return result


def notarize_run_dir(
    run_dir: str | Path,
    *,
    runner: str | None = None,
    config: Mapping[str, Any] | None = None,
    official_record: Mapping[str, Any] | None = None,
    git_sha: str | None = None,
    publish_patterns: Sequence[str] | None = None,
    anchor: bool = False,
) -> dict[str, Any] | None:
    """Write manifest.json if FarmNotary is installed.

    Returns a small receipt dict, or None if the extra is missing.
    """
    if not farm_notary_available():
        return None

    from farm_notary.anchor import anchor_run
    from farm_notary.manifest import build_manifest, write_manifest

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    manifest = build_manifest(
        run_dir,
        publish_patterns=list(publish_patterns or OFFICIAL_PUBLISH_PATTERNS),
        config=config,
        git_sha=git_sha or _git_sha(),
        runner=runner,
        official_record=official_record,
    )
    path = write_manifest(manifest, run_dir)
    receipt: dict[str, Any] = {
        "manifest_path": str(path),
        "content_hash": manifest.content_hash(),
        "anchored": False,
    }
    if anchor:
        anchored = anchor_run(manifest)
        receipt["anchored"] = not anchored.dry_run
        receipt["backend"] = anchored.backend
        receipt["tx_hash"] = anchored.tx_hash
        write_manifest(manifest, run_dir)
    return receipt
