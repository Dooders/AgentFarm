from farm.provenance.notary import farm_notary_available, notarize_run_dir


def test_missing_extra_returns_none(tmp_path, monkeypatch):
    if farm_notary_available():
        receipt = notarize_run_dir(tmp_path, runner="test")
        assert receipt is not None
        assert (tmp_path / "manifest.json").exists()
        return
    assert notarize_run_dir(tmp_path, runner="test") is None
