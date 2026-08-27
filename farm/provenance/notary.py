"""Thin optional wrapper around Dooders/FarmNotary."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional


def farm_notary_available() -> bool:
    try:
        import farm_notary  # noqa: F401
    except ImportError:
        return False
    return True


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def notarize_run_dir(
    run_dir: str | Path,
    *,
    runner: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    official_record: Optional[Mapping[str, Any]] = None,
    git_sha: Optional[str] = None,
    anchor: bool = False,
) -> Optional[dict[str, Any]]:
    """Write manifest.json if FarmNotary is installed.

    Returns a small receipt dict, or None if the extra is missing.
    """
    if not farm_notary_available():
        return None

    from farm_notary.anchor import anchor_run
    from farm_notary.manifest import build_manifest, write_manifest

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        run_dir,
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
