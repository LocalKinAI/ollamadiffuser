"""Tests for API server"""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


@pytest.fixture
def client():
    """Create a test client"""
    from ollamadiffuser.api.server import create_app
    from fastapi.testclient import TestClient

    with patch("ollamadiffuser.api.server.model_manager") as mock_mm:
        mock_mm.is_model_loaded.return_value = False
        mock_mm.get_current_model.return_value = None
        mock_mm.list_available_models.return_value = ["flux.1-dev"]
        mock_mm.list_installed_models.return_value = []
        app = create_app()
        yield TestClient(app), mock_mm


class TestHealthEndpoint:
    def test_health(self, client):
        tc, _ = client
        resp = tc.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    def test_root(self, client):
        tc, _ = client
        resp = tc.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "OllamaDiffuser API"
        assert data["version"] == "2.0.0"


class TestModelsEndpoint:
    def test_list_models(self, client):
        tc, _ = client
        resp = tc.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "installed" in data

    def test_running_model_none(self, client):
        tc, _ = client
        resp = tc.get("/api/models/running")
        assert resp.status_code == 200
        assert resp.json()["loaded"] is False


class TestGenerateEndpoint:
    def test_generate_no_model(self, client):
        tc, _ = client
        resp = tc.post("/api/generate", json={"prompt": "test"})
        assert resp.status_code == 400
        assert "No model loaded" in resp.json()["detail"]
