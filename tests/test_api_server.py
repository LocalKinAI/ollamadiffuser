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


class TestScheduler:
    """Viggle's 6-step LoRA for Qwen-Image-2.1 needs mflux's viggle_turbo schedule."""

    def _engine(self, mm):
        engine = MagicMock()
        engine.generate_image.return_value = Image.new("RGB", (8, 8))
        mm.is_model_loaded.return_value = True
        mm.loaded_model = engine
        return engine

    def test_generate_passes_a_scheduler_through(self, client):
        tc, mm = client
        engine = self._engine(mm)
        resp = tc.post("/api/generate", json={"prompt": "a cabin", "steps": 6, "scheduler": "viggle_turbo"})
        assert resp.status_code == 200
        kwargs = engine.generate_image.call_args.kwargs
        assert kwargs["scheduler"] == "viggle_turbo" and kwargs["num_inference_steps"] == 6

    def test_no_scheduler_sends_none(self, client):
        tc, mm = client
        engine = self._engine(mm)
        tc.post("/api/generate", json={"prompt": "a cabin"})
        assert "scheduler" not in engine.generate_image.call_args.kwargs


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

    def test_a_mask_reaches_the_model_only_when_sent(self, client):
        """Qwen-Image-2.1's edit takes a mask; `ollamadiffuser edit --mask`
        sends it here when a server already holds the model."""
        tc, mm = client
        engine = self._engine(mm)
        tc.post("/api/generate/img2img", data={"prompt": "p"},
                files={"image": ("a.png", self._png("red"), "image/png")})
        assert "mask_image" not in engine.generate_image.call_args.kwargs
        tc.post("/api/generate/img2img", data={"prompt": "p"},
                files=[("image", ("a.png", self._png("red"), "image/png")),
                       ("mask", ("m.png", self._png("white"), "image/png"))])
        assert engine.generate_image.call_args.kwargs["mask_image"].getpixel((0, 0)) == (255, 255, 255)

    def test_an_input_the_model_needs_is_the_callers_to_fix(self, client):
        tc, mm = client
        engine = self._engine(mm)
        engine.generate_image.side_effect = ValueError("qwen21-controlnet needs control_image")
        resp = tc.post("/api/generate/img2img", data={"prompt": "p"},
                       files={"image": ("a.png", self._png("red"), "image/png")})
        assert resp.status_code == 400 and "control_image" in resp.json()["detail"]


class TestLoadEndpoint:
    def test_a_refused_load_says_why(self, client):
        tc, mm = client
        mm.load_model.return_value = False
        mm.last_error = "qwen-image-2.1-mlx needs about 46 GB of memory and 12.0 GB is free."
        resp = tc.post("/api/models/load", json={"model_name": "qwen-image-2.1-mlx"})
        assert resp.status_code == 400 and "needs about 46 GB" in resp.json()["detail"]


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


def _serving(manager, **kwargs):
    from ollamadiffuser.api.server import create_app
    from fastapi.testclient import TestClient
    with patch("ollamadiffuser.api.server.model_manager", manager):
        app = create_app(**kwargs)
        yield TestClient(app), app


@pytest.fixture
def on_demand(box):
    """`serve` with "a" loaded, one model in memory at a time."""
    assert box.manager.load_model("a")
    for client, app in _serving(box.manager):
        yield client, box, app


@pytest.fixture
def for_a(box):
    assert box.manager.load_model("a")
    for client, app in _serving(box.manager, pinned="a"):
        yield client, box, app


class TestModelsOnDemand:
    """`serve`: one server for every installed model, one in memory at a time."""

    def test_a_request_naming_another_model_swaps_it_in(self, on_demand):
        tc, box, _ = on_demand
        resp = tc.post("/api/generate", json={"prompt": "p", "model": "b"})
        assert resp.status_code == 200, resp.text
        assert box.manager.resident_models() == ["b"] and box.unloads() == ["a"]
        assert box.live["b"].generate_image.called

    def test_no_model_named_uses_the_one_loaded(self, on_demand):
        tc, box, _ = on_demand
        assert tc.post("/api/generate", json={"prompt": "p"}).status_code == 200
        assert box.loads() == ["a"] and box.live["a"].generate_image.called

    def test_forms_name_the_model_too(self, on_demand):
        tc, box, _ = on_demand
        resp = tc.post("/api/generate/img2img", data={"prompt": "p", "model": "b"},
                       files={"image": ("a.png", TestImg2ImgEndpoint._png("red"), "image/png")})
        assert resp.status_code == 200, resp.text
        assert box.manager.current_model_name == "b"

    def test_a_model_not_installed_is_404(self, on_demand):
        tc, box, _ = on_demand
        resp = tc.post("/api/generate", json={"prompt": "p", "model": "nope"})
        assert resp.status_code == 404 and "ollamadiffuser pull nope" in resp.json()["detail"]
        assert box.manager.current_model_name == "a"

    def test_a_model_that_cannot_fit_is_refused_and_the_one_loaded_stays(self, on_demand):
        """Not even with "a" unloaded: refused before anything is unloaded."""
        tc, box, _ = on_demand
        box.other_use = 30.0                       # 56 GB free, 66 with "a" gone; "big" needs 70
        resp = tc.post("/api/generate", json={"prompt": "p", "model": "big"})
        assert resp.status_code == 503 and "needs about 70 GB" in resp.json()["detail"]
        assert "66.0 GB with the other models loaded here unloaded" in resp.json()["detail"]
        assert box.manager.current_model_name == "a" and box.unloads() == []

    def test_a_swap_that_wont_fit_puts_the_previous_model_back(self, box):
        """What "a" holds is not known here, so it goes before the check can tell."""
        box.models["a"]["counted"] = False
        box.other_use = 30.0
        assert box.manager.load_model("a")
        for tc, _ in _serving(box.manager):
            resp = tc.post("/api/generate", json={"prompt": "p", "model": "big"})
            assert resp.status_code == 503 and "needs about 70 GB" in resp.json()["detail"]
            assert box.events == [("load", "a"), ("unload", "a"), ("load", "a")]
            assert box.manager.current_model_name == "a"

    def test_nothing_loaded_and_nothing_named_says_to_name_one(self, box):
        for tc, _ in _serving(box.manager):
            resp = tc.post("/api/generate", json={"prompt": "p"})
            assert resp.status_code == 400 and "No model loaded" in resp.json()["detail"]

    def test_running_says_how_the_server_takes_models(self, box):
        assert box.manager.load_model("a")
        for tc, _ in _serving(box.manager, keep_alive=600.0):
            data = tc.get("/api/models/running").json()
            assert data["on_demand"] is True and data["pinned"] is None and data["max_loaded"] == 1
            assert data["keep_alive_s"] == 600.0 and 590 < data["unloads_in_s"] <= 600
            (only,) = data["models"]
            assert only["model"] == "a" and only["memory_gb"] == 10.0 and only["type"] == "mlx"
        for tc, _ in _serving(box.manager, pinned="a"):
            data = tc.get("/api/models/running").json()
            assert data["on_demand"] is False and data["pinned"] == "a" and data["keep_alive_s"] is None

    def test_explicit_load_still_answers_400_when_refused(self, on_demand):
        tc, box, _ = on_demand
        box.other_use = 30.0
        resp = tc.post("/api/models/load", json={"model_name": "big"})
        assert resp.status_code == 400 and "needs about 70 GB" in resp.json()["detail"]
        assert box.manager.current_model_name == "a"


@pytest.fixture
def two_kept(box):
    """`serve --max-loaded 2` with "a" loaded."""
    for client, app in _serving(box.manager, max_loaded=2):
        assert box.manager.load_model("a")
        yield client, box, app


class TestSeveralModelsInMemory:
    """`serve --max-loaded N`: the models stay in memory until one needs the
    place or the memory; jobs still run one at a time."""

    def test_another_model_loads_beside_and_the_first_needs_no_load_again(self, two_kept):
        tc, box, _ = two_kept
        assert tc.post("/api/generate", json={"prompt": "p", "model": "b"}).status_code == 200
        assert tc.post("/api/generate", json={"prompt": "p", "model": "a"}).status_code == 200
        assert box.loads() == ["a", "b"] and box.unloads() == []
        assert box.manager.resident_models() == ["a", "b"]
        assert box.live["a"].generate_image.call_count == 1 and box.live["b"].generate_image.call_count == 1

    def test_past_the_count_the_least_recently_used_goes(self, two_kept):
        tc, box, _ = two_kept
        for model in ("b", "a", "c"):   # "b" was used before "a"
            assert tc.post("/api/generate", json={"prompt": "p", "model": model}).status_code == 200
        assert box.unloads() == ["b"] and box.manager.resident_models() == ["c", "a"]

    def test_short_of_memory_the_least_recently_used_goes_before_the_count_says(self, box):
        for tc, _ in _serving(box.manager, max_loaded=3):
            for model in ("a", "b"):
                assert tc.post("/api/generate", json={"prompt": "p", "model": model}).status_code == 200
            box.other_use = 66.0                   # 10 GB free; "c" needs 25 — "a" going makes 20, "b" too 30
            assert tc.post("/api/generate", json={"prompt": "p", "model": "c"}).status_code == 200
            assert box.unloads() == ["a", "b"] and box.manager.resident_models() == ["c"]

    def test_one_that_cannot_fit_however_many_go_unloads_none(self, two_kept):
        tc, box, _ = two_kept
        assert tc.post("/api/generate", json={"prompt": "p", "model": "b"}).status_code == 200
        box.models["big"]["needs"] = 80
        box.other_use = 20.0                       # 56 GB free, 76 with both gone
        resp = tc.post("/api/generate", json={"prompt": "p", "model": "big"})
        assert resp.status_code == 503 and box.unloads() == []
        assert box.manager.resident_models() == ["b", "a"]

    def test_a_load_that_fails_puts_back_what_went_for_it(self, two_kept):
        tc, box, _ = two_kept
        assert tc.post("/api/generate", json={"prompt": "p", "model": "b"}).status_code == 200
        box.models["c"]["fails"] = True
        resp = tc.post("/api/generate", json={"prompt": "p", "model": "c"})
        assert resp.status_code == 503 and "failed to load" in resp.json()["detail"]
        assert box.unloads() == ["a"] and box.loads() == ["a", "b", "a"]
        assert box.manager.resident_models() == ["b", "a"]   # as before, "b" current

    def test_a_job_that_would_not_fit_beside_the_others_sends_them_away(self, two_kept):
        """Loaded with room to spare, then the machine filled up: the job's
        own working memory comes before keeping the other model."""
        tc, box, _ = two_kept
        assert tc.post("/api/generate", json={"prompt": "p", "model": "b"}).status_code == 200
        box.other_use = 74.0                       # 2 GB free; a job on "b" needs 3 more than it holds
        assert tc.post("/api/generate", json={"prompt": "p", "model": "b"}).status_code == 200
        assert box.unloads() == ["a"] and box.manager.resident_models() == ["b"]

    def test_running_lists_every_model_the_current_one_first(self, two_kept):
        tc, box, _ = two_kept
        tc.post("/api/generate", json={"prompt": "p", "model": "b"})
        data = tc.get("/api/models/running").json()
        assert data["model"] == "b" and data["max_loaded"] == 2
        assert [m["model"] for m in data["models"]] == ["b", "a"]
        assert [m["memory_gb"] for m in data["models"]] == [10.0, 10.0]
        assert tc.get("/api/health").json()["models"] == data["models"]

    def test_unload_takes_one_model_or_all(self, two_kept):
        tc, box, _ = two_kept
        tc.post("/api/generate", json={"prompt": "p", "model": "b"})
        assert tc.post("/api/models/unload", json={"model_name": "b"}).status_code == 200
        assert box.manager.resident_models() == ["a"]
        assert tc.post("/api/models/unload").status_code == 200
        assert box.manager.resident_models() == []

    def test_each_model_has_its_own_keep_alive(self, box):
        import asyncio
        import time
        for tc, app in _serving(box.manager, max_loaded=2, keep_alive=60.0):
            for model in ("a", "b"):
                assert tc.post("/api/generate", json={"prompt": "p", "model": model}).status_code == 200
            app.state.last_used["a"] = time.monotonic() - 61
            assert asyncio.run(app.state.expire()) is True
            assert box.unloads() == ["a"] and box.manager.resident_models() == ["b"]
            assert asyncio.run(app.state.expire()) is False

    def test_a_server_for_one_model_keeps_one(self, box):
        for _ in _serving(box.manager, pinned="a", max_loaded=3):
            assert box.manager.max_loaded == 1


class TestServerForOneModel:
    """`run MODEL`: that model only, as before."""

    def test_another_model_is_refused_and_says_where_to_go(self, for_a):
        tc, box, _ = for_a
        resp = tc.post("/api/generate", json={"prompt": "p", "model": "b"})
        assert resp.status_code == 409 and "ollamadiffuser serve" in resp.json()["detail"]
        assert box.manager.current_model_name == "a" and box.loads() == ["a"]
        assert tc.post("/api/generate", json={"prompt": "p", "model": "a"}).status_code == 200

    def test_the_refusal_survives_the_video_endpoints_own_error_handling(self, for_a):
        tc, _, _ = for_a
        resp = tc.post("/api/generate/video", data={"prompt": "p", "model": "b"})
        assert resp.status_code == 409

    def test_its_model_comes_back_after_a_keep_alive_let_it_go(self, box):
        for tc, _ in _serving(box.manager, pinned="a", keep_alive=60.0):
            assert tc.post("/api/generate", json={"prompt": "p"}).status_code == 200
            assert box.loads() == ["a"] and box.manager.current_model_name == "a"


class TestKeepAlive:
    def test_an_idle_model_is_unloaded_and_a_busy_one_kept(self, box):
        import asyncio
        import time
        assert box.manager.load_model("a")
        for _, app in _serving(box.manager, keep_alive=60.0):
            app.state.last_used["a"] = time.monotonic() - 30
            assert asyncio.run(app.state.expire()) is False and box.unloads() == []
            app.state.last_used["a"] = time.monotonic() - 61
            assert asyncio.run(app.state.expire()) is True and box.unloads() == ["a"]
            assert box.manager.current_model_name is None

    def test_a_model_loaded_before_the_server_counts_from_its_start(self, box):
        import asyncio
        assert box.manager.load_model("a")
        for _, app in _serving(box.manager, keep_alive=60.0):
            app.state.started -= 61
            assert asyncio.run(app.state.expire()) is True and box.unloads() == ["a"]

    def test_without_keep_alive_nothing_is_unloaded(self, box):
        import asyncio
        import time
        assert box.manager.load_model("a")
        for _, app in _serving(box.manager):
            app.state.last_used["a"] = time.monotonic() - 86400
            assert asyncio.run(app.state.expire()) is False and box.unloads() == []

    def test_durations(self):
        from ollamadiffuser.api.server import parse_keep_alive
        assert parse_keep_alive("10m") == 600 and parse_keep_alive("1h") == 3600
        assert parse_keep_alive("90s") == 90 and parse_keep_alive("90") == 90 and parse_keep_alive("0") == 0
        assert parse_keep_alive(None) is None and parse_keep_alive("forever") is None
        assert parse_keep_alive("-1") is None
        with pytest.raises(ValueError, match="not a duration"):
            parse_keep_alive("soon")


class TestServeAndRunOptions:
    def test_serve_preloads_and_passes_keep_alive(self, monkeypatch):
        from click.testing import CliRunner
        from ollamadiffuser.cli import model_commands
        started, loaded = {}, []
        monkeypatch.setattr(model_commands, "run_server", lambda **kw: started.update(kw))
        monkeypatch.setattr(model_commands.model_manager, "is_model_installed", lambda name: True)
        monkeypatch.setattr(model_commands.model_manager, "load_model",
                            lambda name, force=False, persist=True: loaded.append((name, force)) or True)
        result = CliRunner().invoke(model_commands.serve, ["--keep-alive", "10m", "--preload", "a", "-p", "8010"])
        assert result.exit_code == 0, result.output
        assert loaded == [("a", False)]
        assert started == {"host": None, "port": 8010, "keep_alive": 600.0, "max_loaded": 1}

    def test_serve_keeps_several_and_preloads_them(self, monkeypatch):
        from click.testing import CliRunner
        from ollamadiffuser.cli import model_commands
        started, loaded = {}, []
        monkeypatch.setattr(model_commands, "run_server", lambda **kw: started.update(kw))
        monkeypatch.setattr(model_commands.model_manager, "is_model_installed", lambda name: True)
        monkeypatch.setattr(model_commands.model_manager, "load_model",
                            lambda name, force=False, persist=True: loaded.append(
                                (name, model_commands.model_manager.max_loaded)) or True)
        monkeypatch.setattr(model_commands.model_manager, "max_loaded", 1)
        result = CliRunner().invoke(model_commands.serve, ["--max-loaded", "2", "--preload", "a", "--preload", "b"])
        assert result.exit_code == 0, result.output
        assert loaded == [("a", 2), ("b", 2)] and started["max_loaded"] == 2
        result = CliRunner().invoke(model_commands.serve, ["--preload", "a", "--preload", "b"])
        assert result.exit_code != 0 and "--max-loaded is 1" in result.output
        result = CliRunner().invoke(model_commands.serve, [], env={"OLLAMADIFFUSER_MAX_LOADED_MODELS": "3"})
        assert result.exit_code == 0, result.output
        assert started["max_loaded"] == 3

    def test_run_pins_its_model(self, monkeypatch):
        from click.testing import CliRunner
        from ollamadiffuser.cli import model_commands
        started = {}
        monkeypatch.setattr(model_commands, "run_server", lambda **kw: started.update(kw))
        monkeypatch.setattr(model_commands.model_manager, "is_model_installed", lambda name: True)
        monkeypatch.setattr(model_commands.model_manager, "load_model", lambda name, force=False, persist=True: True)
        result = CliRunner().invoke(model_commands.run, ["a", "--keep-alive", "0"])
        assert result.exit_code == 0, result.output
        assert started["pinned"] == "a" and started["keep_alive"] == 0.0
