def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Stock Analysis Tool API"
    assert data["status"] == "running"
    assert "version" in data


def test_health_endpoint(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Stock Analysis Tool"
    assert "version" in data