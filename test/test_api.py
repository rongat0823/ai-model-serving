from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_success():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0]}
    )

    assert response.status_code == 422
