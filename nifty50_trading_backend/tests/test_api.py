import os

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_and_info_endpoints():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("message") == "Healthy"
    r2 = client.get("/info")
    assert r2.status_code == 200
    info = r2.json()
    # directories should be present
    for k in ("model_dir", "data_dir", "features_dir", "rl_dir"):
        assert os.path.isdir(info[k])
