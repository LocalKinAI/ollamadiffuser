"""Tests for MCP server tools — on mcp 1.x (FastMCP) and 2.x (MCPServer) alike."""

import asyncio
import io
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


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.is_model_loaded.return_value = False
    mm.get_current_model.return_value = None
    mm.list_available_models.return_value = [
        "flux.1-schnell",
        "dreamshaper",
        "stable-diffusion-1.5",
    ]
    mm.list_installed_models.return_value = ["dreamshaper"]
    mm.is_model_installed.return_value = False
    return mm


@pytest.fixture
def server(mock_model_manager, monkeypatch, tmp_path):
    from ollamadiffuser.core.config.settings import settings
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    with patch("ollamadiffuser.mcp.server.model_manager", mock_model_manager), \
         patch("ollamadiffuser.mcp.server._serving", return_value=None):
        from ollamadiffuser.mcp.server import create_mcp_server
        yield create_mcp_server()


@MCP_SKIP
class TestMCPServerCreation:
    def test_create_server(self, server):
        assert server is not None
        assert server.name == "OllamaDiffuser"

    def test_server_has_four_tools(self, server):
        names = {t.name for t in _run(server.list_tools())}
        assert names == {"generate_image", "list_models", "load_model", "get_status"}

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


@MCP_SKIP
class TestGenerateImageTool:
    def test_generate_no_model_says_so(self, server):
        assert "No model loaded" in _error_text(server, "generate_image", {"prompt": "test"})

    def test_generate_returns_the_saved_path_as_text_then_the_image(self, server, mock_model_manager, tmp_path):
        mock_model_manager.is_model_loaded.return_value = True
        mock_model_manager.get_current_model.return_value = "z-image-turbo-mlx"
        engine = MagicMock()
        engine.generate_image.return_value = PILImage.new("RGB", (16, 16), (255, 0, 0))
        mock_model_manager.loaded_model = engine
        blocks = _content(_run(server.call_tool("generate_image", {"prompt": "a red square", "seed": 3})))
        assert blocks[0].type == "text" and "Saved to" in blocks[0].text
        saved = blocks[0].text.split("Saved to ", 1)[1].split(" (", 1)[0]
        assert saved.startswith(str(tmp_path / "outputs")) and saved.endswith("-3.png")
        assert PILImage.open(saved).size == (16, 16)
        assert blocks[1].type == "image"

    def test_a_running_server_with_the_model_does_the_work(self, mock_model_manager, monkeypatch, tmp_path):
        """Loading the model in the MCP process as well would hold it twice."""
        from ollamadiffuser.core.config.settings import settings
        monkeypatch.setattr(settings, "config_dir", tmp_path)
        buf = io.BytesIO()
        PILImage.new("RGB", (8, 8)).save(buf, format="PNG")
        posted = []
        with patch("ollamadiffuser.mcp.server.model_manager", mock_model_manager), \
             patch("ollamadiffuser.mcp.server._serving", return_value="http://127.0.0.1:8011") as serving, \
             patch("ollamadiffuser.mcp.server._generate_over_http",
                   side_effect=lambda base, body: posted.append((base, body)) or buf.getvalue()):
            from ollamadiffuser.mcp.server import create_mcp_server
            server = create_mcp_server()
            blocks = _content(_run(server.call_tool(
                "generate_image", {"prompt": "p", "model": "qwen-image-2.1-mlx", "steps": 6})))
        serving.assert_called_once_with("qwen-image-2.1-mlx")
        mock_model_manager.load_model.assert_not_called()
        assert posted[0][0] == "http://127.0.0.1:8011" and posted[0][1]["steps"] == 6
        assert "qwen-image-2.1-mlx" in blocks[0].text
