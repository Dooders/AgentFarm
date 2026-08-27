"""AgentFarm adapter for FarmNotary.

FarmNotary is an optional dependency; every entry point here raises a helpful
error when it is missing. The adapter is a thin veneer: manifests record the
command (with a ``{run_dir}`` placeholder), config, git identity, and
environment; anchoring goes through OpenTimestamps (free public calendars) or
stays a dry run; reproduction re-executes the recorded command and
byte-compares the artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

RUN_CONFIG_NAME = "run_config.json"


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


def _read_run_config(run_dir: Path) -> dict:
    """The experiment's own run_config.json, when present."""
    path = Path(run_dir) / RUN_CONFIG_NAME
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def notarize(
    run_dir: Path,
    *,
    config: Optional[Mapping[str, Any]] = None,
    command: Optional[str] = None,
    runner: str = "agentfarm",
    official_record: Optional[Mapping[str, Any]] = None,
    backend: str = "dry-run",
    calendars: Optional[Sequence[str]] = None,
    pin: bool = False,
    ipfs_api: Optional[str] = None,
    lockfile: Optional[Path] = None,
) -> Tuple[Any, Any]:
    """Notarize a finished run directory; returns (manifest, receipt).

    When the run directory contains run_config.json (written by
    run_experiment.py), its command and config are used automatically, so the
    manifest is reproduce-ready without extra arguments. backend is "dry-run"
    (default, no network) or "ots" (OpenTimestamps public calendars, free).
    """
    fn = _require_farm_notary()
    from farm_notary.anchor import get_backend

    run_config = _read_run_config(run_dir)
    return fn.notarize_run(
        Path(run_dir),
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
    """Re-run the notarized command and byte-compare; returns the result.

    Writes a reproduction.json receipt into run_dir; with anchor=True the
    receipt hash is timestamped via OpenTimestamps (reproduction.ots).
    """
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
