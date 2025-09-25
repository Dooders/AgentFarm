from fastapi.testclient import TestClient

from server.backend.app.main import app


client = TestClient(app)


def test_config_schema_endpoint_basic_structure():
    resp = client.get("/config/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert "sections" in data
    sections = data["sections"]
    for key in ["simulation", "visualization", "redis", "observation"]:
        assert key in sections
        assert "properties" in sections[key]
        assert isinstance(sections[key]["properties"], dict)

    # Spot-check a few expected fields
    sim_props = sections["simulation"]["properties"]
    assert "width" in sim_props
    assert sim_props["width"]["type"] in ["integer", "number"]
    obs_props = sections["observation"]["properties"]
    assert "R" in obs_props
    assert obs_props["R"]["type"] in ["integer", "number"]


def test_config_schema_has_titles_and_nonempty_sections():
    resp = client.get("/config/schema")
    assert resp.status_code == 200
    data = resp.json()
    sections = data["sections"]
    # Titles present
    assert sections["simulation"].get("title", "")
    assert sections["visualization"].get("title", "")
    assert sections["redis"].get("title", "")
    assert sections["observation"].get("title", "")
    # Each section has at least one property
    for key, meta in sections.items():
        assert isinstance(meta.get("properties", {}), dict)
        assert len(meta["properties"]) > 0


def test_config_schema_sections_keys_align_with_titles():
    resp = client.get("/config/schema")
    assert resp.status_code == 200
    data = resp.json()
    sections = data["sections"]
    # Ensure expected keys exist and the title is a string
    for key in ("simulation", "visualization", "redis", "observation"):
        assert key in sections
        assert isinstance(sections[key].get("title", ""), str)

