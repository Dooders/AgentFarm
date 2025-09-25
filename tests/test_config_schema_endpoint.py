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

