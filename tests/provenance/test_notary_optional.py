import sys
import unittest.mock

from farm.provenance.notary import notarize_run_dir


def test_missing_extra_returns_none(tmp_path, monkeypatch):
    """notarize_run_dir returns None when farm_notary is not installed."""
    monkeypatch.setitem(sys.modules, "farm_notary", None)
    assert notarize_run_dir(tmp_path, runner="test") is None
