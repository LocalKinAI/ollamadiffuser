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


class TestNoiseGuard:
    """A model can return undenoised latents: a valid PNG of confetti.

    Every cheap check passes it — it decodes, it is the right size, it is
    megabytes — so without a look at the pixels the failure reaches the user
    as art. Measured on FLUX.1-Kontext int8: the same weights, prompt and
    input image draw a portrait at seed 1234 and static at seed 473366517.
    """

    def test_static_is_caught_and_a_picture_is_not(self):
        import numpy as np
        from PIL import Image
        from ollamadiffuser.core.inference.base import InferenceStrategy

        rng = np.random.default_rng(0)
        static = Image.fromarray((rng.random((256, 256, 3)) * 255).astype("uint8"))
        gradient = Image.fromarray(
            np.tile(np.linspace(0, 255, 256, dtype="uint8"), (256, 1)))
        assert InferenceStrategy.looks_like_noise(static) is True
        assert InferenceStrategy.looks_like_noise(gradient) is False

    def test_a_busy_photograph_is_not_noise(self):
        """The check has to survive detail, not just flat images.

        One metric could not: adjacent-pixel difference reads 14.4 for a
        photograph of a dog in grass and 14.7 for a failed generation. This
        builds a stand-in for the busy photograph — fine texture on top of a
        composition — and it must pass while static does not.
        """
        import numpy as np
        from PIL import Image
        from ollamadiffuser.core.inference.base import InferenceStrategy

        rng = np.random.default_rng(1)
        y, x = np.mgrid[0:256, 0:256]
        composition = (128 + 100 * np.sin(x / 40.0) * np.cos(y / 55.0))
        texture = rng.normal(0, 28, (256, 256))
        busy = np.clip(composition + texture, 0, 255).astype("uint8")
        assert InferenceStrategy.looks_like_noise(Image.fromarray(busy)) is False

    def test_a_detector_that_cannot_run_passes_the_image(self):
        from ollamadiffuser.core.inference.base import InferenceStrategy

        class NotAnImage:
            def convert(self, mode):
                raise ValueError("nope")

        assert InferenceStrategy.looks_like_noise(NotAnImage()) is False
