"""Tests for b64_json response_format on /api/generate."""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture
def client_with_model():
    """Create a test client with a mock loaded model."""
    from fastapi.testclient import TestClient

    from ollamadiffuser.api.server import create_app

    with patch("ollamadiffuser.api.server.model_manager") as mock_mm:
        mock_mm.is_model_loaded.return_value = True
        mock_mm.get_current_model.return_value = "test-model"

        mock_engine = MagicMock()
        mock_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        mock_engine.generate_image.return_value = mock_image
        mock_mm.loaded_model = mock_engine

        app = create_app()
        yield TestClient(app)


class TestB64JsonResponseFormat:
    def test_generate_default_returns_png(self, client_with_model):
        resp = client_with_model.post("/api/generate", json={"prompt": "test"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_generate_b64_json_returns_json(self, client_with_model):
        resp = client_with_model.post(
            "/api/generate",
            json={"prompt": "test", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image" in data
        assert data["format"] == "png"
        assert data["width"] == 512
        assert data["height"] == 512

        # Verify base64 decodes to a valid PNG
        decoded = base64.b64decode(data["image"])
        img = Image.open(BytesIO(decoded))
        assert img.format == "PNG"
        assert img.size == (512, 512)

    def test_generate_null_format_returns_png(self, client_with_model):
        resp = client_with_model.post(
            "/api/generate",
            json={"prompt": "test", "response_format": None},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_generate_no_model_returns_400(self):
        from fastapi.testclient import TestClient

        from ollamadiffuser.api.server import create_app

        with patch("ollamadiffuser.api.server.model_manager") as mock_mm:
            mock_mm.is_model_loaded.return_value = False
            app = create_app()
            tc = TestClient(app)
            resp = tc.post(
                "/api/generate",
                json={"prompt": "test", "response_format": "b64_json"},
            )
            assert resp.status_code == 400
