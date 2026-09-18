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


class TestVideoEndpoint:
    def test_text_to_video_needs_no_files(self, client, tmp_path):
        """The ordinary call — a prompt and nothing else — must reach the engine.

        It did not: the temp-file helper checked for a missing upload, but the
        suffix was read off `upload.filename` at the call site, before the
        check. So every text-to-video request died with an AttributeError
        about NoneType and came back as a 500 with no detail. Found on the
        first real generation, against a loaded model.
        """
        tc, mm = client
        clip = tmp_path / "out.mp4"
        clip.write_bytes(b"\x00" * 64)
        engine = MagicMock()
        engine.generate_video.return_value = str(clip)
        mm.is_model_loaded.return_value = True
        mm.loaded_model = engine

        resp = tc.post("/api/generate/video", data={"prompt": "a cat", "seconds": "2"})
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "video/mp4"
        assert resp.headers["X-Output-Path"] == str(clip)
        kwargs = engine.generate_video.call_args.kwargs
        assert kwargs["prompt"] == "a cat" and kwargs["seconds"] == 2.0
        # Nothing was uploaded, so nothing about files is forwarded.
        assert "image" not in kwargs and "audio" not in kwargs

    def test_video_refusals_are_400(self, client):
        tc, mm = client
        engine = MagicMock()
        engine.generate_video.side_effect = RuntimeError("This is a video model")
        mm.is_model_loaded.return_value = True
        mm.loaded_model = engine
        resp = tc.post("/api/generate/video", data={"prompt": "a cat"})
        assert resp.status_code == 400
        assert "video model" in resp.json()["detail"]
