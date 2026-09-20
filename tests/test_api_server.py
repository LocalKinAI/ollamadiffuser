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


class TestImg2ImgEndpoint:
    @staticmethod
    def _png(colour):
        import io
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), colour).save(buf, format="PNG")
        return buf.getvalue()

    def _engine(self, mm):
        engine = MagicMock()
        engine.generate_image.return_value = Image.new("RGB", (64, 48), "white")
        mm.is_model_loaded.return_value = True
        mm.loaded_model = engine
        return engine

    def test_one_picture_forwards_no_list(self, client):
        """The ordinary edit. A model that takes one picture must not be
        handed an `images` keyword it has never heard of."""
        tc, mm = client
        engine = self._engine(mm)
        resp = tc.post("/api/generate/img2img", data={"prompt": "make it night"},
                       files={"image": ("a.png", self._png("red"), "image/png")})
        assert resp.status_code == 200, resp.text
        kwargs = engine.generate_image.call_args.kwargs
        assert "images" not in kwargs
        assert kwargs["image"].size == (64, 48)
        assert (kwargs["width"], kwargs["height"]) == (64, 48)

    def test_extra_references_arrive_in_order_after_the_first(self, client):
        """Who from one picture, where from another: the first upload is
        image 1 and sets the size, the extras follow in upload order."""
        tc, mm = client
        engine = self._engine(mm)
        resp = tc.post(
            "/api/generate/img2img", data={"prompt": "the woman from image 1 in the place from image 2"},
            files=[("image", ("who.png", self._png("red"), "image/png")),
                   ("images", ("where.png", self._png("green"), "image/png")),
                   ("images", ("also.png", self._png("blue"), "image/png"))])
        assert resp.status_code == 200, resp.text
        kwargs = engine.generate_image.call_args.kwargs
        colours = [ref.getpixel((0, 0)) for ref in kwargs["images"]]
        assert colours == [(255, 0, 0), (0, 128, 0), (0, 0, 255)]
        assert kwargs["image"].getpixel((0, 0)) == (255, 0, 0)


class TestFaceCompareEndpoint:
    @staticmethod
    def _png(colour):
        import io
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), colour).save(buf, format="PNG")
        return buf.getvalue()

    def test_needs_no_model_and_answers_in_upload_order(self, client):
        """Frames in, numbers out — from a server with no diffusion model
        loaded, because this is not a diffusion model's question."""
        tc, mm = client
        mm.is_model_loaded.return_value = False
        matcher = MagicMock()
        matcher.compare.return_value = {"model": "sface", "results": [{"found": True, "similarity": 0.7}, {"found": False}]}
        tc.app.state.face_matcher = matcher
        resp = tc.post("/api/face/compare",
                       files=[("anchor", ("her.png", self._png("red"), "image/png")),
                              ("images", ("one.png", self._png("green"), "image/png")),
                              ("images", ("two.png", self._png("blue"), "image/png"))])
        assert resp.status_code == 200, resp.text
        assert resp.json()["results"] == [{"found": True, "similarity": 0.7}, {"found": False}]
        who, pictures = matcher.compare.call_args.args
        assert who.getpixel((0, 0)) == (255, 0, 0)
        assert [p.getpixel((0, 0)) for p in pictures] == [(0, 128, 0), (0, 0, 255)]

    def test_missing_models_say_how_to_get_them(self, client, tmp_path):
        from ollamadiffuser.core.utils.face_match import FaceMatcher
        tc, _ = client
        tc.app.state.face_matcher = FaceMatcher(directory=tmp_path)      # an empty folder
        resp = tc.post("/api/face/compare",
                       files=[("anchor", ("her.png", self._png("red"), "image/png")),
                              ("images", ("one.png", self._png("green"), "image/png"))])
        assert resp.status_code == 503
        assert "--download" in resp.json()["detail"] and "face_recognition_sface" in resp.json()["detail"]


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
