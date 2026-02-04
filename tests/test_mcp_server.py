"""Tests for MCP server tools."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _mcp_available():
    try:
        import mcp  # noqa: F401

        return True
    except ImportError:
        return False


MCP_SKIP = pytest.mark.skipif(not _mcp_available(), reason="mcp package not installed")


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


@MCP_SKIP
class TestMCPServerCreation:
    def test_create_server(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            assert server is not None
            assert server.name == "OllamaDiffuser"

    def test_server_has_four_tools(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            tools = server._tool_manager._tools
            assert "generate_image" in tools
            assert "list_models" in tools
            assert "load_model" in tools
            assert "get_status" in tools
            assert len(tools) == 4


@MCP_SKIP
class TestListModelsTool:
    def test_list_models_output(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.get_event_loop().run_until_complete(
                server.call_tool("list_models", {})
            )
            text = content[0].text
            assert "dreamshaper" in text
            assert "installed" in text
            assert "flux.1-schnell" in text
            assert "Installed: 1/3" in text


@MCP_SKIP
class TestGetStatusTool:
    def test_status_no_model(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.get_event_loop().run_until_complete(
                server.call_tool("get_status", {})
            )
            text = content[0].text
            assert "Model loaded: no" in text
            assert "Installed models: 1" in text

    def test_status_with_model(self, mock_model_manager):
        mock_model_manager.is_model_loaded.return_value = True
        mock_model_manager.get_current_model.return_value = "dreamshaper"
        mock_engine = MagicMock()
        mock_engine.get_model_info.return_value = {
            "device": "mps",
            "type": "sd15",
        }
        mock_model_manager.loaded_model = mock_engine

        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.get_event_loop().run_until_complete(
                server.call_tool("get_status", {})
            )
            text = content[0].text
            assert "Model loaded: yes" in text
            assert "Current model: dreamshaper" in text
            assert "Device: mps" in text


@MCP_SKIP
class TestLoadModelTool:
    def test_load_not_installed(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.get_event_loop().run_until_complete(
                server.call_tool("load_model", {"model_name": "nonexistent"})
            )
            text = content[0].text
            assert "not installed" in text

    def test_load_success(self, mock_model_manager):
        mock_model_manager.is_model_installed.return_value = True
        mock_model_manager.load_model.return_value = True

        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.get_event_loop().run_until_complete(
                server.call_tool("load_model", {"model_name": "dreamshaper"})
            )
            text = content[0].text
            assert "loaded successfully" in text


@MCP_SKIP
class TestGenerateImageTool:
    def test_generate_no_model_raises(self, mock_model_manager):
        from mcp.server.fastmcp.exceptions import ToolError

        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            with pytest.raises(ToolError, match="No model loaded"):
                asyncio.get_event_loop().run_until_complete(
                    server.call_tool("generate_image", {"prompt": "test"})
                )
