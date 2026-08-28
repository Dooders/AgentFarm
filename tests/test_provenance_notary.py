"""Tests for the FarmNotary adapter (skipped when farm-notary is not installed)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("farm_notary")

from farm.provenance import notarize, reproduce, verify

pytestmark = pytest.mark.unit


def make_run(tmp_path: Path) -> Path:
    """A tiny fake run whose recorded command regenerates it deterministically.

    Like run_experiment.py, the generator writes run_config.json itself (with
    the {run_dir} placeholder), so a re-run reproduces every artifact.
    """
    script = tmp_path / "gen.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'summary.csv').write_text('a,b\\n1,2\\n')\n"
        "command = f'{sys.executable} {sys.argv[0]} {{run_dir}}'\n"
        "(out / 'run_config.json').write_text("
        "json.dumps({'command': command, 'config': {'seed': 3}}))\n"
    )
    run_dir = tmp_path / "run"
    command = f"{sys.executable} {script} {{run_dir}}"
    import subprocess

    subprocess.run(
        command.replace("{run_dir}", str(run_dir)), shell=True, check=True
    )
    return run_dir


def test_notarize_picks_up_run_config(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    manifest, receipt = notarize(run_dir)
    assert receipt.backend == "dry-run" and receipt.dry_run
    assert manifest.command.endswith("{run_dir}")
    assert manifest.config == {"seed": 3}
    assert "summary.csv" in manifest.artifact_hashes
    assert verify(run_dir) == []


def test_reproduce_byte_compares_and_writes_receipt(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    notarize(run_dir)
    result = reproduce(run_dir)
    assert result.ok
    receipt = json.loads((run_dir / "reproduction.json").read_text())
    assert receipt["ok"] is True
    assert verify(run_dir) == []

    # Tampering with an artifact after notarization is caught.
    (run_dir / "summary.csv").write_text("a,b\n9,9\n")
    assert any("summary.csv" in p for p in verify(run_dir))
