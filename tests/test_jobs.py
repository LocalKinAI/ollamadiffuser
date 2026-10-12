"""One job, wherever the model already is (core/jobs.py)."""
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from ollamadiffuser.core import jobs
from ollamadiffuser.core.jobs import Job, JobError


def _png(size=(8, 8), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


def _server(model, port=8010, on_demand=False):
    return {"port": port, "host": "127.0.0.1", "url": f"http://127.0.0.1:{port}", "pid": None,
            "answering": True, "loaded": model is not None, "model": model, "type": "mlx", "device": "mps",
            "on_demand": on_demand, "keep_alive_s": None}


@pytest.fixture
def manager():
    """A manager with nothing loaded and z-image-turbo-mlx installed."""
    mm = MagicMock()
    mm.is_model_loaded.return_value = False
    mm.current_model_name = None
    mm.last_error = None
    mm.is_model_installed.side_effect = lambda name: name == "z-image-turbo-mlx"
    engine = MagicMock()
    engine.generate_image.return_value = Image.new("RGBA", (16, 16))

    def load(name, force=False, persist=True):
        mm.is_model_loaded.return_value = True
        mm.current_model_name = name
        mm.loaded_model = engine
        return True

    mm.load_model.side_effect = load
    mm.engine = engine
    return mm


@pytest.fixture
def running(monkeypatch):
    found = []
    monkeypatch.setattr(jobs, "servers", lambda: found)
    return found


@pytest.fixture
def posted(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "_over_http", lambda url, job, seed: calls.append((url, job, seed)) or _png())
    return calls


class TestWhereAJobRuns:
    def test_a_server_holding_the_model_does_it(self, manager, running, posted, tmp_path):
        running.append(_server("z-image-turbo-mlx"))
        result = jobs.run(Job("image", "a fox", model="z-image-turbo-mlx"), output_dir=str(tmp_path), manager=manager)
        manager.load_model.assert_not_called()
        assert posted[0][0] == "http://127.0.0.1:8010"
        assert result["server"] == "http://127.0.0.1:8010" and result["model"] == "z-image-turbo-mlx"
        assert Path(result["path"]).read_bytes() == _png()

    def test_no_model_named_uses_the_one_server_running(self, manager, running, posted, tmp_path):
        running.append(_server("qwen-turbo-one"))
        result = jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager)
        assert result["model"] == "qwen-turbo-one"

    def test_no_model_and_nothing_running_says_so(self, manager, running, tmp_path):
        with pytest.raises(JobError, match="No model loaded or named"):
            jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager)

    def test_no_model_and_several_servers_asks_which(self, manager, running, tmp_path):
        running.extend([_server("a", 8010), _server("b", 8011)])
        with pytest.raises(JobError, match="a on port 8010, b on port 8011"):
            jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager)

    def test_otherwise_the_model_loads_here_and_goes_after(self, manager, running, tmp_path):
        result = jobs.run(Job("image", "a fox", model="z-image-turbo-mlx", seed=7),
                          output_dir=str(tmp_path), manager=manager)
        manager.load_model.assert_called_once_with("z-image-turbo-mlx", force=False, persist=False)
        manager.unload_model.assert_called_once_with(persist=False)
        kwargs = manager.engine.generate_image.call_args.kwargs
        assert kwargs["seed"] == 7 and (kwargs["width"], kwargs["height"]) == (1024, 1024)
        assert result["server"] is None and result["seed"] == 7
        assert Image.open(result["path"]).size == (16, 16)

    def test_keep_loaded_keeps_it(self, manager, running, tmp_path):
        jobs.run(Job("image", "p", model="z-image-turbo-mlx"), output_dir=str(tmp_path),
                 keep_loaded=True, manager=manager)
        manager.load_model.assert_called_once_with("z-image-turbo-mlx", force=False, persist=True)
        manager.unload_model.assert_not_called()

    def test_a_refused_load_says_why(self, manager, running, tmp_path):
        manager.load_model.side_effect = None
        manager.load_model.return_value = False
        manager.last_error = "z-image-turbo-mlx needs about 6 GB of memory and 2.0 GB is free."
        with pytest.raises(JobError, match="needs about 6 GB"):
            jobs.run(Job("image", "p", model="z-image-turbo-mlx"), output_dir=str(tmp_path), manager=manager)

    def test_force_reaches_the_load(self, manager, running, tmp_path):
        jobs.run(Job("image", "p", model="z-image-turbo-mlx"), output_dir=str(tmp_path),
                 force=True, manager=manager)
        assert manager.load_model.call_args.kwargs["force"] is True

    def test_a_model_not_installed_says_how_to_get_it(self, manager, running, tmp_path):
        with pytest.raises(JobError, match="ollamadiffuser pull flux.1-dev-mlx"):
            jobs.run(Job("image", "p", model="flux.1-dev-mlx"), output_dir=str(tmp_path), manager=manager)

    def test_a_model_failure_is_a_job_error_and_the_model_still_goes(self, manager, running, tmp_path):
        manager.engine.generate_image.side_effect = RuntimeError("Metal ran out")
        with pytest.raises(JobError, match="Metal ran out"):
            jobs.run(Job("image", "p", model="z-image-turbo-mlx"), output_dir=str(tmp_path), manager=manager)
        manager.unload_model.assert_called_once_with(persist=False)

    def test_a_seed_is_picked_and_reported(self, manager, running, posted, tmp_path):
        running.append(_server("m"))
        result = jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager)
        assert isinstance(result["seed"], int) and posted[0][2] == result["seed"]


class TestServersThatLoadOnDemand:
    """A `serve` server takes a job for any model: the request names it and it swaps it in."""

    def test_a_model_no_server_holds_goes_to_one_that_loads_on_demand(self, manager, running, posted, tmp_path):
        running.extend([_server("x", 8010), _server("y", 8011, on_demand=True)])
        result = jobs.run(Job("image", "p", model="z-image-turbo-mlx"), output_dir=str(tmp_path), manager=manager)
        url, job, _ = posted[0]
        assert url == "http://127.0.0.1:8011" and job.model == "z-image-turbo-mlx"
        assert result["model"] == "z-image-turbo-mlx" and result["server"] == url
        manager.load_model.assert_not_called()

    def test_an_idle_one_is_preferred_so_nobodys_model_is_swapped_out(self, manager, running, posted, tmp_path):
        running.extend([_server("y", 8011, on_demand=True), _server(None, 8012, on_demand=True)])
        jobs.run(Job("image", "p", model="z"), output_dir=str(tmp_path), manager=manager)
        assert posted[0][0] == "http://127.0.0.1:8012"

    def test_a_server_keeping_the_model_beside_another_does_it(self, manager, running, posted, tmp_path):
        keeper = _server("x", 8011, on_demand=True)
        keeper.update(models=["x", "z"], max_loaded=2)
        running.extend([_server(None, 8010, on_demand=True), keeper])
        jobs.run(Job("image", "p", model="z"), output_dir=str(tmp_path), manager=manager)
        assert posted[0][0] == "http://127.0.0.1:8011" and posted[0][1].model == "z"

    def test_one_with_room_is_preferred_over_one_that_would_unload(self, manager, running, posted, tmp_path):
        roomy = _server("y", 8012, on_demand=True)
        roomy.update(models=["y"], max_loaded=2)
        running.extend([_server("x", 8011, on_demand=True), roomy])
        jobs.run(Job("image", "p", model="z"), output_dir=str(tmp_path), manager=manager)
        assert posted[0][0] == "http://127.0.0.1:8012"

    def test_the_model_goes_with_the_request_even_when_the_job_named_none(self, manager, running, posted, tmp_path):
        running.append(_server("y", 8011))
        jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager)
        assert posted[0][1].model == "y"

    def test_a_remote_one_takes_a_model_it_does_not_hold(self, manager, posted, monkeypatch, tmp_path):
        class Running:
            status_code = 200

            @staticmethod
            def json():
                return {"loaded": True, "model": "y", "on_demand": True}

        monkeypatch.setattr("requests.get", lambda url, timeout=None: Running())
        result = jobs.run(Job("image", "p", model="z"), output_dir=str(tmp_path), manager=manager,
                          server="192.168.0.21:8000")
        assert posted[0][1].model == "z" and result["model"] == "z"

    def test_a_remote_one_holding_nothing_needs_a_model_name(self, manager, monkeypatch, tmp_path):
        class Running:
            status_code = 200

            @staticmethod
            def json():
                return {"loaded": False, "on_demand": True}

        monkeypatch.setattr("requests.get", lambda url, timeout=None: Running())
        with pytest.raises(JobError, match="loads models on demand"):
            jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager, server="h:8000")


class TestLoRA:
    def test_not_on_a_shared_server(self, manager, running, tmp_path):
        running.append(_server("qwen-image-2.1-mlx"))
        with pytest.raises(JobError, match="/api/lora/load"):
            jobs.run(Job("image", "p", model="qwen-image-2.1-mlx", lora="Viggle/turbo"),
                     output_dir=str(tmp_path), manager=manager)

    def test_a_video_lora_is_part_of_the_request(self, manager, running, posted, tmp_path):
        running.append(_server("ltx-2.3-mlx-q8"))
        jobs.run(Job("video", "p", model="ltx-2.3-mlx-q8", lora="Lightricks/pose"),
                 output_dir=str(tmp_path), manager=manager)
        assert posted[0][1].lora == "Lightricks/pose"

    def test_loaded_here_for_the_job(self, manager, running, tmp_path):
        manager.engine.load_lora_runtime.return_value = True
        jobs.run(Job("image", "p", model="z-image-turbo-mlx", lora="a.safetensors", lora_scale=0.8),
                 output_dir=str(tmp_path), manager=manager)
        manager.engine.load_lora_runtime.assert_called_once_with("a.safetensors", None, 0.8)


class TestFiles:
    def test_edit_needs_a_picture_that_exists(self, manager, running, tmp_path):
        with pytest.raises(JobError, match="needs the picture"):
            jobs.run(Job("edit", "p", model="z-image-turbo-mlx"), output_dir=str(tmp_path), manager=manager)
        with pytest.raises(JobError, match="No such file: /nope.png"):
            jobs.run(Job("edit", "p", model="z-image-turbo-mlx", image="/nope.png"),
                     output_dir=str(tmp_path), manager=manager)

    def test_a_jpeg_output_is_a_jpeg_even_from_an_rgba_model(self, manager, running, tmp_path):
        out = tmp_path / "out.jpg"
        jobs.run(Job("image", "p", model="z-image-turbo-mlx"), output=str(out), manager=manager)
        assert Image.open(out).format == "JPEG"

    def test_an_output_without_a_suffix_gets_one(self, manager, running, posted, tmp_path):
        running.append(_server("m"))
        result = jobs.run(Job("video", "p"), output=str(tmp_path / "clip"), manager=manager)
        assert result["path"].endswith("clip.mp4")

    def test_named_files_do_not_overwrite_each_other(self, manager, running, posted, tmp_path):
        running.append(_server("m"))
        first = jobs.run(Job("image", "p", seed=1), output_dir=str(tmp_path), manager=manager)
        second = jobs.run(Job("image", "p", seed=1), output_dir=str(tmp_path), manager=manager)
        assert first["path"] != second["path"] and Path(second["path"]).exists()

    def test_a_video_made_elsewhere_is_moved_to_the_output(self, manager, running, tmp_path):
        made = tmp_path / "tmp-ltx.mp4"

        def video(prompt, output=None, seconds=None, **kwargs):
            made.write_bytes(b"mp4")
            return str(made)

        manager.is_model_installed.side_effect = None
        manager.is_model_installed.return_value = True
        manager.engine.generate_video.side_effect = video
        out = tmp_path / "clip.mp4"
        jobs.run(Job("video", "p", model="ltx-2.3-mlx-q8", seconds=2), output=str(out), manager=manager)
        assert out.read_bytes() == b"mp4" and not made.exists()


class TestOverHttp:
    @pytest.fixture
    def requests_post(self, monkeypatch):
        calls = []

        class Answer:
            status_code = 200
            content = b"made"

        def post(url, **kwargs):
            files = kwargs.get("files") or []
            calls.append({"url": url, **kwargs, "files": [(name, upload[0]) for name, upload in files]})
            return Answer()

        monkeypatch.setattr("requests.post", post)
        return calls

    def test_image(self, requests_post):
        jobs._over_http("http://h:1", Job("image", "p", steps=8, guidance=1.0, scheduler="s"), 42)
        call = requests_post[0]
        assert call["url"] == "http://h:1/api/generate"
        assert call["json"] == {"prompt": "p", "width": 1024, "height": 1024, "steps": 8,
                                "cfg_scale": 1.0, "seed": 42, "scheduler": "s"}

    def test_edit_sends_the_pictures_and_the_mask(self, requests_post, tmp_path):
        a, b, m = (tmp_path / n for n in ("a.png", "b.png", "m.png"))
        for f in (a, b, m):
            f.write_bytes(_png())
        jobs._over_http("http://h:1", Job("edit", "p", image=str(a), references=[str(b)], mask=str(m),
                                          strength=0.6), 7)
        call = requests_post[0]
        assert call["url"] == "http://h:1/api/generate/img2img"
        assert call["files"] == [("image", "a.png"), ("images", "b.png"), ("mask", "m.png")]
        assert call["data"] == {"prompt": "p", "seed": "7", "strength": "0.6"}

    def test_video(self, requests_post, tmp_path):
        first = tmp_path / "first.png"
        first.write_bytes(_png())
        jobs._over_http("http://h:1", Job("video", "p", seconds=4, image=str(first)), 3)
        call = requests_post[0]
        assert call["url"] == "http://h:1/api/generate/video"
        assert call["files"] == [("image", "first.png")] and call["data"]["seconds"] == "4"

    def test_a_refusal_carries_the_servers_reason(self, monkeypatch):
        class Refused:
            status_code = 400

            @staticmethod
            def json():
                return {"detail": "z-image-turbo-mlx is not a video model"}

        monkeypatch.setattr("requests.post", lambda url, **kwargs: Refused())
        with pytest.raises(JobError, match="not a video model"):
            jobs._over_http("http://h:1", Job("video", "p"), 1)


class TestRemoteServer:
    """`--server` / OLLAMADIFFUSER_SERVER: the job runs on another machine, the file lands here."""

    @pytest.fixture
    def remote(self, monkeypatch):
        state = {"answer": {"loaded": True, "model": "boogu-image-turbo-mlx",
                            "info": {"type": "mlx", "device": "mps"}},
                 "loads": [], "load_status": 200, "load_detail": None}

        class Answer:
            def __init__(self, status, body):
                self.status_code, self._body = status, body

            def json(self):
                return self._body

        def get(url, timeout=None):
            if state["answer"] is None:
                import requests
                raise requests.ConnectionError("connection refused")
            return Answer(200, state["answer"])

        def post(url, json=None, timeout=None):
            state["loads"].append((url, json))
            return Answer(state["load_status"], {"detail": state["load_detail"]})

        monkeypatch.setattr("requests.get", get)
        monkeypatch.setattr("requests.post", post)
        monkeypatch.setattr(jobs, "servers", lambda host=None: pytest.fail("no local discovery"))
        return state

    def test_addresses(self):
        assert jobs.server_url("192.168.0.21:8010") == "http://192.168.0.21:8010"
        assert jobs.server_url("box.local") == "http://box.local:8000"
        assert jobs.server_url("http://192.168.0.21:8010/") == "http://192.168.0.21:8010"
        with pytest.raises(JobError, match="Not a server address"):
            jobs.server_url("")

    def test_the_server_holding_the_model_does_it_and_the_file_lands_here(self, manager, remote, posted, tmp_path):
        result = jobs.run(Job("image", "a fox"), output_dir=str(tmp_path), manager=manager,
                          server="192.168.0.21:8010")
        assert posted[0][0] == "http://192.168.0.21:8010"
        assert result["server"] == "http://192.168.0.21:8010" and result["model"] == "boogu-image-turbo-mlx"
        assert Path(result["path"]).parent == tmp_path
        manager.load_model.assert_not_called()

    def test_another_model_there_is_left_alone(self, manager, remote, tmp_path):
        with pytest.raises(JobError, match="runs boogu-image-turbo-mlx, not qwen-image-2.1-mlx"):
            jobs.run(Job("image", "p", model="qwen-image-2.1-mlx"), output_dir=str(tmp_path),
                     manager=manager, server="192.168.0.21:8010")
        assert remote["loads"] == []

    def test_an_idle_server_loads_the_model_named(self, manager, remote, posted, tmp_path):
        remote["answer"] = {"loaded": False}
        result = jobs.run(Job("image", "p", model="z-image-turbo-mlx"), output_dir=str(tmp_path),
                          manager=manager, server="192.168.0.21:8010")
        assert remote["loads"] == [("http://192.168.0.21:8010/api/models/load", {"model_name": "z-image-turbo-mlx"})]
        assert result["model"] == "z-image-turbo-mlx" and posted

    def test_an_idle_server_refusing_the_load_says_why(self, manager, remote, tmp_path):
        remote.update(answer={"loaded": False}, load_status=400, load_detail="needs about 46 GB")
        with pytest.raises(JobError, match="could not load qwen-image-2.1-mlx: needs about 46 GB"):
            jobs.run(Job("image", "p", model="qwen-image-2.1-mlx"), output_dir=str(tmp_path),
                     manager=manager, server="192.168.0.21:8010")

    def test_an_idle_server_needs_a_model_name(self, manager, remote, tmp_path):
        remote["answer"] = {"loaded": False}
        with pytest.raises(JobError, match="no model loaded"):
            jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager, server="h:1")

    def test_nobody_answering_says_how_to_start_one(self, manager, remote, tmp_path):
        remote["answer"] = None
        with pytest.raises(JobError, match="--host 0.0.0.0"):
            jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager, server="192.168.0.21:8019")

    def test_something_else_on_the_port_is_named_as_such(self, manager, monkeypatch, tmp_path):
        class NotOurs:
            status_code = 200

            @staticmethod
            def json():
                raise ValueError("Extra data: line 1 column 5")

        monkeypatch.setattr("requests.get", lambda url, timeout=None: NotOurs())
        with pytest.raises(JobError, match="not an OllamaDiffuser server"):
            jobs.run(Job("image", "p"), output_dir=str(tmp_path), manager=manager, server="127.0.0.1:8019")

    def test_a_lora_is_not_forced_on_it(self, manager, remote, tmp_path):
        with pytest.raises(JobError, match="/api/lora/load"):
            jobs.run(Job("image", "p", lora="Viggle/turbo"), output_dir=str(tmp_path),
                     manager=manager, server="192.168.0.21:8010")
