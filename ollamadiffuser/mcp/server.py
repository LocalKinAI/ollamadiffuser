"""OllamaDiffuser MCP Server - Model Context Protocol integration."""

import asyncio
import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from ..core.models.manager import model_manager

logger = logging.getLogger(__name__)


def _ensure_mcp():
    """Check that the mcp package is available."""
    try:
        import mcp  # noqa: F401

        return True
    except ImportError:
        logger.error(
            "MCP package not installed. Install with: pip install 'ollamadiffuser[mcp]'"
        )
        return False


def _mcp_api():
    """The server class, Image and ToolError of whichever mcp is installed.

    mcp 2.x renamed FastMCP to MCPServer (mcp.server.mcpserver); 1.x has it
    as mcp.server.fastmcp. `ollamadiffuser enable mcp` installs the newest,
    so importing only the 1.x name left the server unable to start.
    """
    try:
        from mcp.server.mcpserver import Image, MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        return MCPServer, Image, ToolError
    except ImportError:
        from mcp.server.fastmcp import FastMCP, Image
        from mcp.server.fastmcp.exceptions import ToolError
        return FastMCP, Image, ToolError


def _serving(model: Optional[str]) -> Optional[str]:
    """Base URL of a running server that already holds ``model`` (any model
    when None) — generating there spares a second copy in this process."""
    from ..core.utils import server_registry
    for record in server_registry.running():
        answer = server_registry.ask(record.get("host"), record.get("port"), timeout=1.0)
        if answer and answer.get("loaded") and (model is None or answer.get("model") == model):
            host = record.get("host")
            host = "127.0.0.1" if host in ("0.0.0.0", "", None) else host
            return f"http://{host}:{record.get('port')}"
    return None


def _generate_over_http(base: str, body: dict) -> bytes:
    import requests
    response = requests.post(f"{base}/api/generate", json={k: v for k, v in body.items() if v is not None},
                             timeout=3600)
    response.raise_for_status()
    return response.content


def _save(png: bytes, seed: Optional[int]) -> Path:
    from ..core.config.settings import settings
    folder = Path(settings.config_dir) / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"mcp-{time.strftime('%Y%m%d-%H%M%S')}-{seed if seed is not None else 'r'}.png"
    path.write_bytes(png)
    return path


def create_mcp_server():
    """Create and configure the MCP server with all tools."""
    Server, Image, ToolError = _mcp_api()

    mcp_server = Server(
        "OllamaDiffuser",
        instructions=(
            "Local AI image generation via Stable Diffusion, FLUX, and 30+ models"
        ),
    )

    @mcp_server.tool()
    async def generate_image(
        prompt: str,
        model: Optional[str] = None,
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> list:
        """Generate an image from a text prompt using a local diffusion model.

        Returns the path of the saved PNG as text, then the image itself — a
        client that reads only text still gets something it can open.

        Args:
            prompt: Text description of the desired image
            model: Model to use (auto-loads if needed). Leave empty to use current model.
            negative_prompt: What to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of inference steps (model-specific default if omitted)
            guidance_scale: Guidance scale (model-specific default if omitted)
            seed: Random seed for reproducibility
        """
        # A server already holding the model does the work: loading it here
        # as well would put a second copy — tens of GB — in memory.
        # Named model: a server holding it. No name and nothing loaded here:
        # any server holding a model. Otherwise this process's own model.
        if model:
            base = await asyncio.to_thread(_serving, model)
        elif not model_manager.is_model_loaded():
            base = await asyncio.to_thread(_serving, None)
        else:
            base = None
        if base:
            used = model or "the running server's model"
            try:
                png = await asyncio.to_thread(_generate_over_http, base, {
                    "prompt": prompt, "negative_prompt": negative_prompt, "width": width,
                    "height": height, "steps": steps, "cfg_scale": guidance_scale, "seed": seed,
                })
            except Exception as e:
                raise ToolError(f"The OllamaDiffuser server at {base} failed: {e}")
        else:
            if model and model_manager.get_current_model() != model:
                if not model_manager.is_model_installed(model):
                    raise ToolError(
                        f"Model '{model}' is not installed. "
                        f"Install it first: ollamadiffuser pull {model}"
                    )
                logger.info(f"Loading model: {model}")
                success = await asyncio.to_thread(model_manager.load_model, model)
                if not success:
                    raise ToolError(f"Failed to load model '{model}'")

            if not model_manager.is_model_loaded():
                raise ToolError(
                    "No model loaded. Load one with: load_model('model-name') "
                    "or pass model= parameter"
                )

            used = model_manager.get_current_model()
            engine = model_manager.loaded_model
            result = await asyncio.to_thread(
                engine.generate_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
            )
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            png = buf.getvalue()

        path = await asyncio.to_thread(_save, png, seed)
        return [f"Saved to {path} ({used})", Image(data=png, format="png")]

    @mcp_server.tool()
    async def list_models() -> str:
        """List all available and installed image generation models.

        Returns a formatted list showing which models are available to download
        and which are already installed locally.
        """
        available = model_manager.list_available_models()
        installed = model_manager.list_installed_models()
        current = model_manager.get_current_model()

        lines = ["Available models:"]
        for name in sorted(available):
            status_parts = []
            if name in installed:
                status_parts.append("installed")
            if name == current:
                status_parts.append("loaded")
            suffix = f" ({', '.join(status_parts)})" if status_parts else ""
            lines.append(f"  - {name}{suffix}")

        lines.append(f"\nInstalled: {len(installed)}/{len(available)}")
        if current:
            lines.append(f"Currently loaded: {current}")

        return "\n".join(lines)

    @mcp_server.tool()
    async def load_model(model_name: str) -> str:
        """Load a specific image generation model into memory.

        Args:
            model_name: Name of the model to load (must be installed first)
        """
        if not model_manager.is_model_installed(model_name):
            installed = model_manager.list_installed_models()
            return (
                f"Model '{model_name}' is not installed. "
                f"Installed models: {', '.join(installed) if installed else 'none'}. "
                f"Use 'ollamadiffuser pull {model_name}' to install it first."
            )

        success = await asyncio.to_thread(model_manager.load_model, model_name)
        if success:
            return f"Model '{model_name}' loaded successfully"
        return f"Failed to load model '{model_name}'"

    @mcp_server.tool()
    async def get_status() -> str:
        """Get the current status of OllamaDiffuser.

        Returns device info, loaded model, and installed model count.
        """
        is_loaded = model_manager.is_model_loaded()
        current = model_manager.get_current_model()
        installed = model_manager.list_installed_models()

        lines = ["OllamaDiffuser Status:"]
        lines.append(f"  Model loaded: {'yes' if is_loaded else 'no'}")
        if current:
            lines.append(f"  Current model: {current}")
        lines.append(f"  Installed models: {len(installed)}")

        if is_loaded and model_manager.loaded_model:
            engine = model_manager.loaded_model
            info = engine.get_model_info()
            if info:
                lines.append(f"  Device: {info.get('device', 'unknown')}")
                lines.append(f"  Model type: {info.get('type', 'unknown')}")

        return "\n".join(lines)

    return mcp_server


def main(
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 9000,
):
    """Entry point for the MCP server.

    Args:
        transport: Transport type - "stdio", "sse", or "streamable-http".
        host: Bind address for network transports.
        port: Port for network transports.
    """
    if not _ensure_mcp():
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    server = create_mcp_server()

    if transport == "stdio":
        server.run(transport="stdio")
    elif transport in ("sse", "streamable-http"):
        import uvicorn

        app = server.sse_app() if transport == "sse" else server.streamable_http_app()
        uvicorn.run(app, host=host, port=port)
    else:
        logger.error(f"Unknown transport: {transport}. Use stdio, sse, or streamable-http.")
        sys.exit(1)


if __name__ == "__main__":
    main()
