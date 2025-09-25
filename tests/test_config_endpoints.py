from fastapi.testclient import TestClient

from server.backend.app.main import app


client = TestClient(app)


def test_validate_endpoint_accepts_default_dict_and_normalizes():
    # Minimal valid config relies on dataclass defaults; empty dict should still construct
    resp = client.post("/config/validate", json={"config": {}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert isinstance(data.get("config"), dict)
    # Spot check a few normalized fields
    assert "width" in data["config"]
    assert "visualization" in data["config"]


def test_save_then_load_roundtrip(tmp_path):
    # Save a simple config override
    override = {"width": 123, "height": 77}
    target = tmp_path / "roundtrip.yaml"
    resp = client.post("/config/save", json={"config": override, "path": str(target)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data.get("path").endswith("roundtrip.yaml")
    # Then load it back
    resp2 = client.post("/config/load", json={"path": str(target)})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["success"] is True
    cfg = data2.get("config")
    assert cfg["width"] == 123
    assert cfg["height"] == 77


def test_load_missing_file_returns_uniform_error(tmp_path):
    missing = tmp_path / "nope.yaml"
    resp = client.post("/config/load", json={"path": str(missing)})
    assert resp.status_code == 200  # uniform response with success flag
    data = resp.json()
    assert data["success"] is False
    assert "Missing" in data.get("errors", [""])[0]

