import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(not (Path(__file__).parent.parent / "models/model.joblib").exists(),
                                reason="run train.py first")


def test_predict_endpoint():
    from fastapi.testclient import TestClient
    from app.api import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json=dict(step=100, type="TRANSFER", amount=250000, nameOrig="C1",
                                         oldbalanceOrg=250000, nameDest="C90000001", oldbalanceDest=0))
        j = r.json()
        assert r.status_code == 200 and 0 <= j["fraud_probability"] <= 1
        assert j["risk_tier"] in {"LOW", "MEDIUM", "HIGH"} and j["explanation"]
