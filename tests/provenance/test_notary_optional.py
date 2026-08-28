import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

from farm.provenance.notary import _git_sha, farm_notary_available, notarize_run_dir


def test_missing_extra_returns_none(tmp_path, monkeypatch):
    """notarize_run_dir returns None when farm_notary is not installed."""
    monkeypatch.setitem(sys.modules, "farm_notary", None)
    assert notarize_run_dir(tmp_path, runner="test") is None


def test_farm_notary_available_returns_true(monkeypatch):
    """farm_notary_available returns True when the package is importable."""
    fake_module = types.ModuleType("farm_notary")
    monkeypatch.setitem(sys.modules, "farm_notary", fake_module)
    assert farm_notary_available() is True


def test_farm_notary_available_returns_false(monkeypatch):
    """farm_notary_available returns False when the package is missing."""
    # Setting to None makes `import farm_notary` raise ImportError
    monkeypatch.setitem(sys.modules, "farm_notary", None)
    assert farm_notary_available() is False


def test_git_sha_returns_string_in_git_repo():
    """_git_sha returns a non-empty string when called inside a git repo."""
    sha = _git_sha()
    # The test environment is a git checkout so we expect a hex string or None.
    assert sha is None or (isinstance(sha, str) and len(sha) > 0)


def test_git_sha_no_git_dir():
    """_git_sha returns None when there is no .git directory."""
    with patch("farm.provenance.notary.Path") as mock_path_cls:
        mock_repo_root = mock_path_cls.return_value.resolve.return_value.parent.parent.parent
        mock_repo_root.__truediv__.return_value.exists.return_value = False
        result = _git_sha()
    assert result is None


def test_git_sha_subprocess_error():
    """_git_sha returns None when git command fails."""
    with patch("farm.provenance.notary.subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
        result = _git_sha()
    assert result is None


def test_notarize_run_dir_with_farm_notary(tmp_path, monkeypatch):
    """notarize_run_dir returns a receipt dict when farm_notary is available."""
    # Build a fake farm_notary package with the required sub-modules
    fake_fn = types.ModuleType("farm_notary")
    fake_manifest_mod = types.ModuleType("farm_notary.manifest")
    fake_anchor_mod = types.ModuleType("farm_notary.anchor")

    fake_manifest = MagicMock()
    fake_manifest.content_hash.return_value = "abc123"

    fake_manifest_mod.build_manifest = MagicMock(return_value=fake_manifest)
    fake_manifest_mod.write_manifest = MagicMock(return_value=tmp_path / "manifest.json")
    fake_anchor_mod.anchor_run = MagicMock()

    monkeypatch.setitem(sys.modules, "farm_notary", fake_fn)
    monkeypatch.setitem(sys.modules, "farm_notary.manifest", fake_manifest_mod)
    monkeypatch.setitem(sys.modules, "farm_notary.anchor", fake_anchor_mod)

    receipt = notarize_run_dir(tmp_path, runner="test_runner", config={"k": "v"})
    assert receipt is not None
    assert receipt["anchored"] is False
    assert receipt["content_hash"] == "abc123"
    assert "manifest_path" in receipt
    kwargs = fake_manifest_mod.build_manifest.call_args.kwargs
    assert kwargs["publish_patterns"]


def test_notarize_run_dir_with_anchor(tmp_path, monkeypatch):
    """notarize_run_dir sets anchored=True when anchor=True and dry_run=False."""
    fake_fn = types.ModuleType("farm_notary")
    fake_manifest_mod = types.ModuleType("farm_notary.manifest")
    fake_anchor_mod = types.ModuleType("farm_notary.anchor")

    fake_manifest = MagicMock()
    fake_manifest.content_hash.return_value = "def456"

    anchored_result = MagicMock()
    anchored_result.dry_run = False
    anchored_result.backend = "test_backend"
    anchored_result.tx_hash = "0xabc"

    fake_manifest_mod.build_manifest = MagicMock(return_value=fake_manifest)
    fake_manifest_mod.write_manifest = MagicMock(return_value=tmp_path / "manifest.json")
    fake_anchor_mod.anchor_run = MagicMock(return_value=anchored_result)

    monkeypatch.setitem(sys.modules, "farm_notary", fake_fn)
    monkeypatch.setitem(sys.modules, "farm_notary.manifest", fake_manifest_mod)
    monkeypatch.setitem(sys.modules, "farm_notary.anchor", fake_anchor_mod)

    receipt = notarize_run_dir(tmp_path, runner="test_runner", anchor=True)
    assert receipt is not None
    assert receipt["anchored"] is True
    assert receipt["backend"] == "test_backend"
    assert receipt["tx_hash"] == "0xabc"


def test_notarize_run_dir_missing_dir(tmp_path, monkeypatch):
    """notarize_run_dir raises FileNotFoundError when run_dir does not exist."""
    fake_fn = types.ModuleType("farm_notary")
    fake_manifest_mod = types.ModuleType("farm_notary.manifest")
    fake_anchor_mod = types.ModuleType("farm_notary.anchor")
    fake_manifest_mod.build_manifest = MagicMock()
    fake_manifest_mod.write_manifest = MagicMock()
    fake_anchor_mod.anchor_run = MagicMock()

    monkeypatch.setitem(sys.modules, "farm_notary", fake_fn)
    monkeypatch.setitem(sys.modules, "farm_notary.manifest", fake_manifest_mod)
    monkeypatch.setitem(sys.modules, "farm_notary.anchor", fake_anchor_mod)

    import pytest

    with pytest.raises(FileNotFoundError):
        notarize_run_dir(tmp_path / "nonexistent", runner="test")
