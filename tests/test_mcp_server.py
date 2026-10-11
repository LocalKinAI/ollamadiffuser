"""Tests for MCP server tools — on mcp 1.x (FastMCP) and 2.x (MCPServer) alike."""

import asyncio
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage


def _mcp_available():
    try:
        import mcp  # noqa: F401

        return True
    except ImportError:
        return False


MCP_SKIP = pytest.mark.skipif(not _mcp_available(), reason="mcp package not installed")


def _run(coro):
    return asyncio.run(coro)


def _content(result):
    """call_tool's content blocks: 1.x returns (content, structured), 2.x a result object."""
    if isinstance(result, tuple):
        return list(result[0])
    if hasattr(result, "content"):
        return list(result.content)
    return list(result)


def _error_text(server, name, args):
    """The message a client sees when the tool fails, whichever way the SDK reports it."""
    from ollamadiffuser.mcp.server import _mcp_api
    _, _, ToolError = _mcp_api()
    try:
        result = _run(server.call_tool(name, args))
    except ToolError as e:
        return str(e)
    assert getattr(result, "is_error", False), "the tool was expected to fail"
    return _content(result)[0].text


def _png(color=(255, 0, 0)):
    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.is_model_loaded.return_value = False
    mm.current_model_name = None
    mm.get_current_model.return_value = None
    mm.list_available_models.return_value = [
        "flux.1-schnell",
        "dreamshaper",
        "stable-diffusion-1.5",
    ]
    mm.list_installed_models.return_value = ["dreamshaper"]
    mm.is_model_installed.return_value = False
    mm.last_error = None
    return mm


@pytest.fixture
def running_servers():
    """What jobs see running on this machine; nothing unless a test says so."""
    return []


@pytest.fixture
def server(mock_model_manager, running_servers, monkeypatch, tmp_path):
    from ollamadiffuser.core import jobs
    from ollamadiffuser.core.config.settings import settings
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    monkeypatch.delenv(jobs.SERVER_ENV, raising=False)
    with patch("ollamadiffuser.mcp.server.model_manager", mock_model_manager), \
         patch("ollamadiffuser.core.jobs.servers", return_value=running_servers):
        from ollamadiffuser.mcp.server import create_mcp_server
        yield create_mcp_server()


def _loaded(mm, name="z-image-turbo-mlx"):
    """The mock manager with ``name`` loaded in this (the MCP) process."""
    mm.is_model_loaded.return_value = True
    mm.current_model_name = name
    mm.get_current_model.return_value = name
    engine = MagicMock()
    engine.generate_image.return_value = PILImage.new("RGB", (16, 16), (255, 0, 0))
    mm.loaded_model = engine
    return engine


def _a_server(model, port=8011):
    return {"port": port, "host": "127.0.0.1", "url": f"http://127.0.0.1:{port}", "pid": 4242,
            "answering": True, "loaded": True, "model": model, "type": "mlx", "device": "mps"}


@MCP_SKIP
class TestMCPServerCreation:
    def test_create_server(self, server):
        assert server is not None
        assert server.name == "OllamaDiffuser"

    def test_server_has_its_tools(self, server):
        names = {t.name for t in _run(server.list_tools())}
        assert names == {"generate_image", "edit_image", "generate_video", "list_servers",
                         "list_models", "load_model", "get_status"}

    def test_the_installed_sdk_is_used(self):
        """mcp 2.x renamed FastMCP to MCPServer; importing only the old name
        left `ollamadiffuser mcp` unable to start on a fresh install."""
        import importlib.metadata as md
        from ollamadiffuser.mcp.server import _mcp_api
        Server, _, _ = _mcp_api()
        expected = "MCPServer" if int(md.version("mcp").split(".")[0]) >= 2 else "FastMCP"
        assert Server.__name__ == expected


@MCP_SKIP
class TestListModelsTool:
    def test_list_models_output(self, server):
        text = _content(_run(server.call_tool("list_models", {})))[0].text
        assert "dreamshaper" in text
        assert "installed" in text
        assert "flux.1-schnell" in text
        assert "Installed: 1/3" in text


@MCP_SKIP
class TestGetStatusTool:
    def test_status_no_model(self, server):
        text = _content(_run(server.call_tool("get_status", {})))[0].text
        assert "Model loaded: no" in text
        assert "Installed models: 1" in text

    def test_status_with_model(self, server, mock_model_manager):
        mock_model_manager.is_model_loaded.return_value = True
        mock_model_manager.get_current_model.return_value = "dreamshaper"
        mock_engine = MagicMock()
        mock_engine.get_model_info.return_value = {"device": "mps", "type": "sd15"}
        mock_model_manager.loaded_model = mock_engine
        text = _content(_run(server.call_tool("get_status", {})))[0].text
        assert "Model loaded: yes" in text
        assert "Current model: dreamshaper" in text
        assert "Device: mps" in text


@MCP_SKIP
class TestLoadModelTool:
    def test_load_not_installed(self, server):
        text = _content(_run(server.call_tool("load_model", {"model_name": "nonexistent"})))[0].text
        assert "not installed" in text

    def test_load_success(self, server, mock_model_manager):
        mock_model_manager.is_model_installed.return_value = True
        mock_model_manager.load_model.return_value = True
        text = _content(_run(server.call_tool("load_model", {"model_name": "dreamshaper"})))[0].text
        assert "loaded successfully" in text

    def test_a_refused_load_says_why(self, server, mock_model_manager):
        mock_model_manager.is_model_installed.return_value = True
        mock_model_manager.load_model.return_value = False
        mock_model_manager.last_error = "qwen-image-2.1-mlx needs about 46 GB of memory and 12.0 GB is free."
        text = _content(_run(server.call_tool("load_model", {"model_name": "qwen-image-2.1-mlx"})))[0].text
        assert "needs about 46 GB" in text


@MCP_SKIP
class TestGenerateImageTool:
    def test_generate_no_model_says_so(self, server):
        assert "No model loaded" in _error_text(server, "generate_image", {"prompt": "test"})

    def test_generate_returns_the_saved_path_as_text_then_the_image(self, server, mock_model_manager, tmp_path):
        _loaded(mock_model_manager)
        blocks = _content(_run(server.call_tool("generate_image", {"prompt": "a red square", "seed": 3})))
        assert blocks[0].type == "text" and "Saved to" in blocks[0].text
        saved = blocks[0].text.split("Saved to ", 1)[1].split(" (", 1)[0]
        assert saved.startswith(str(tmp_path / "outputs")) and saved.endswith("-3.png")
        assert PILImage.open(saved).size == (16, 16)
        assert blocks[1].type == "image"
        # The model stays loaded here for the next call.
        mock_model_manager.unload_model.assert_not_called()

    def test_scheduler_reaches_the_model(self, server, mock_model_manager):
        engine = _loaded(mock_model_manager, "qwen-image-2.1-mlx")
        _run(server.call_tool("generate_image", {"prompt": "p", "scheduler": "viggle_turbo", "steps": 6}))
        kwargs = engine.generate_image.call_args.kwargs
        assert kwargs["scheduler"] == "viggle_turbo" and kwargs["num_inference_steps"] == 6

    def test_a_lora_is_loaded_once_and_dropped_when_not_asked_for(self, server, mock_model_manager):
        engine = _loaded(mock_model_manager, "qwen-image-2.1-mlx")
        engine.load_lora_runtime.return_value = True
        args = {"prompt": "p", "lora": "Viggle/turbo", "lora_weight": "v.safetensors"}
        _run(server.call_tool("generate_image", args))
        _run(server.call_tool("generate_image", args))
        engine.load_lora_runtime.assert_called_once_with("Viggle/turbo", "v.safetensors", 1.0)
        _run(server.call_tool("generate_image", {"prompt": "p"}))
        engine.unload_lora.assert_called_once()

    def test_a_running_server_with_the_model_does_the_work(self, server, mock_model_manager, running_servers):
        """Loading the model in the MCP process as well would hold it twice."""
        running_servers.append(_a_server("qwen-image-2.1-mlx"))
        posted = []
        with patch("ollamadiffuser.core.jobs._over_http",
                   side_effect=lambda url, job, seed: posted.append((url, job, seed)) or _png()):
            blocks = _content(_run(server.call_tool(
                "generate_image", {"prompt": "p", "model": "qwen-image-2.1-mlx", "steps": 6})))
        mock_model_manager.load_model.assert_not_called()
        url, job, _ = posted[0]
        assert url == "http://127.0.0.1:8011" and job.steps == 6 and job.kind == "image"
        assert "qwen-image-2.1-mlx" in blocks[0].text and "via http://127.0.0.1:8011" in blocks[0].text

    def test_a_lora_is_not_forced_on_a_shared_server(self, server, running_servers):
        running_servers.append(_a_server("qwen-image-2.1-mlx"))
        text = _error_text(server, "generate_image",
                           {"prompt": "p", "model": "qwen-image-2.1-mlx", "lora": "Viggle/turbo"})
        assert "/api/lora/load" in text


@MCP_SKIP
class TestEditImageTool:
    def test_edit_passes_the_picture_references_and_mask(self, server, mock_model_manager, tmp_path):
        engine = _loaded(mock_model_manager, "qwen-image-2.1-edit-mlx")
        a, b, m = (tmp_path / n for n in ("a.png", "b.png", "m.png"))
        for f in (a, b, m):
            PILImage.new("RGB", (32, 24)).save(f)
        blocks = _content(_run(server.call_tool("edit_image", {
            "prompt": "make it night", "image": str(a), "references": [str(b)],
            "mask": str(m), "strength": 0.6})))
        kwargs = engine.generate_image.call_args.kwargs
        assert kwargs["image"].size == (32, 24) and (kwargs["width"], kwargs["height"]) == (32, 24)
        assert len(kwargs["images"]) == 2 and kwargs["mask_image"] is not None
        assert kwargs["strength"] == 0.6
        assert blocks[1].type == "image"

    def test_a_missing_picture_says_which(self, server, mock_model_manager):
        _loaded(mock_model_manager)
        assert "No such file" in _error_text(server, "edit_image", {"prompt": "p", "image": "/nope.png"})


@MCP_SKIP
class TestGenerateVideoTool:
    def test_video_is_saved_and_its_path_returned(self, server, mock_model_manager):
        engine = _loaded(mock_model_manager, "ltx-2.3-mlx-q8")

        def make(prompt, output=None, seconds=None, **kwargs):
            Path(output).write_bytes(b"mp4")
            return output

        engine.generate_video.side_effect = make
        text = _content(_run(server.call_tool("generate_video", {"prompt": "waves", "seconds": 2})))[0].text
        assert "Saved to" in text and text.split("Saved to ", 1)[1].split(" (", 1)[0].endswith(".mp4")
        assert engine.generate_video.call_args.kwargs["seconds"] == 2


@MCP_SKIP
class TestListServersTool:
    def test_lists_each_server_and_its_model(self, server, running_servers):
        running_servers.extend([_a_server("qwen-image-2.1-turbo-mlx", 8010), _a_server("ltx-2.3-mlx-q8", 8012)])
        text = _content(_run(server.call_tool("list_servers", {})))[0].text
        assert "port 8010: qwen-image-2.1-turbo-mlx" in text and "port 8012: ltx-2.3-mlx-q8" in text

    def test_nothing_running(self, server):
        text = _content(_run(server.call_tool("list_servers", {})))[0].text
        assert "No OllamaDiffuser server is running" in text


@MCP_SKIP
class TestServerOnAnotherMachine:
    """OLLAMADIFFUSER_SERVER in the client's config: every job goes to that server."""

    def test_jobs_go_there(self, server, mock_model_manager, monkeypatch):
        from ollamadiffuser.core import jobs

        class Running:
            status_code = 200

            @staticmethod
            def json():
                return {"loaded": True, "model": "boogu-image-turbo-mlx", "info": {}}

        monkeypatch.setenv(jobs.SERVER_ENV, "192.168.0.21:8010")
        monkeypatch.setattr("requests.get", lambda url, timeout=None: Running())
        with patch("ollamadiffuser.core.jobs._over_http", side_effect=lambda url, job, seed: _png()):
            blocks = _content(_run(server.call_tool("generate_image", {"prompt": "a fox"})))
        assert "via http://192.168.0.21:8010" in blocks[0].text and blocks[1].type == "image"
        mock_model_manager.load_model.assert_not_called()

    def test_list_servers_says_where_jobs_go(self, server, running_servers, monkeypatch):
        from ollamadiffuser.core import jobs
        monkeypatch.setenv(jobs.SERVER_ENV, "192.168.0.21:8010")
        running_servers.append(_a_server("boogu-image-turbo-mlx", 8010))
        text = _content(_run(server.call_tool("list_servers", {})))[0].text
        assert "Jobs go to http://192.168.0.21:8010" in text and "port 8010: boogu-image-turbo-mlx" in text
