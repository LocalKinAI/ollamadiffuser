"""OllamaDiffuser MCP Server - Model Context Protocol integration."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from ..core import jobs
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


def _outputs() -> Path:
    from ..core.config.settings import settings
    return Path(settings.config_dir) / "outputs"


def create_mcp_server():
    """Create and configure the MCP server with all tools."""
    Server, Image, ToolError = _mcp_api()

    mcp_server = Server(
        "OllamaDiffuser",
        instructions=(
            "Local image and video generation on this machine: text-to-image, editing a "
            "picture by instruction, and short videos, with FLUX, Qwen-Image, Z-Image, LTX-2 "
            "and more. Files are paths on this machine. A job goes to an OllamaDiffuser server "
            "already running the model when there is one, and otherwise loads the model here — "
            "or, when OLLAMADIFFUSER_SERVER names one on another machine, to that server only. "
            "list_servers shows what is running, list_models what is installed."
        ),
    )

    async def _run(job: jobs.Job) -> dict:
        # OLLAMADIFFUSER_SERVER, set in the client's config, sends every job to
        # that server — another machine's GPU — and the files come back here.
        try:
            return await asyncio.to_thread(
                jobs.run, job, output_dir=str(_outputs()), keep_loaded=True, manager=model_manager,
                server=os.environ.get(jobs.SERVER_ENV) or None,
            )
        except jobs.JobError as e:
            raise ToolError(str(e))

    def _said(result: dict) -> str:
        via = f" via {result['server']}" if result["server"] else ""
        return (f"Saved to {result['path']} ({result['model']}, seed {result['seed']}, "
                f"{result['elapsed_s']} s{via})")

    @mcp_server.tool()
    async def generate_image(
        prompt: str,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
        lora: Optional[str] = None,
        lora_weight: Optional[str] = None,
        lora_scale: float = 1.0,
    ) -> list:
        """Generate an image from a text prompt using a local diffusion model.

        Returns the path of the saved PNG as text, then the image itself — a
        client that reads only text still gets something it can open.

        Args:
            prompt: Text description of the desired image
            model: Model to use, e.g. z-image-turbo-mlx (fast) or qwen-image-2.1-turbo-mlx
                (text in pictures). Leave empty for the model already running.
            negative_prompt: What to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of inference steps (model-specific default if omitted)
            guidance_scale: Guidance scale (model-specific default if omitted)
            seed: Random seed for reproducibility (one is picked and reported if omitted)
            scheduler: A sampling schedule the model takes, e.g. viggle_turbo
            lora: A LoRA to apply: a .safetensors path or a Hugging Face repo
            lora_weight: The file inside that repo, when it holds several
            lora_scale: How strongly to apply the LoRA
        """
        result = await _run(jobs.Job(
            kind="image", prompt=prompt, model=model, negative_prompt=negative_prompt,
            width=width, height=height, steps=steps, guidance=guidance_scale, seed=seed,
            scheduler=scheduler, lora=lora, lora_weight=lora_weight, lora_scale=lora_scale,
        ))
        return [_said(result), Image(data=Path(result["path"]).read_bytes(), format="png")]

    @mcp_server.tool()
    async def edit_image(
        prompt: str,
        image: str,
        references: Optional[List[str]] = None,
        mask: Optional[str] = None,
        strength: Optional[float] = None,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
        lora: Optional[str] = None,
        lora_weight: Optional[str] = None,
        lora_scale: float = 1.0,
    ) -> list:
        """Edit a picture by instruction ("make it a snowy night", "put the person from
        image 2 into image 1"), and return the saved path and the result.

        Args:
            prompt: The instruction
            image: Path of the picture to edit, on this machine; it sets the output size
            references: Paths of further pictures for models that take several
                (image 2, image 3, ...)
            mask: Path of a mask (white where to change), for models that take one
            strength: How far to move from the picture (model default if omitted)
            model: An editing model, e.g. qwen-image-2.1-edit-mlx or flux.2-klein-4b-edit-mlx.
                Leave empty for the model already running.
            negative_prompt: What to avoid
            steps: Number of inference steps (model-specific default if omitted)
            guidance_scale: Guidance scale (model-specific default if omitted)
            seed: Random seed (one is picked and reported if omitted)
            scheduler: A sampling schedule the model takes
            lora: A LoRA to apply: a .safetensors path or a Hugging Face repo
            lora_weight: The file inside that repo, when it holds several
            lora_scale: How strongly to apply the LoRA
        """
        result = await _run(jobs.Job(
            kind="edit", prompt=prompt, model=model, image=image, references=list(references or []),
            mask=mask, strength=strength, negative_prompt=negative_prompt, steps=steps,
            guidance=guidance_scale, seed=seed, scheduler=scheduler, lora=lora,
            lora_weight=lora_weight, lora_scale=lora_scale,
        ))
        return [_said(result), Image(data=Path(result["path"]).read_bytes(), format="png")]

    @mcp_server.tool()
    async def generate_video(
        prompt: str,
        model: Optional[str] = None,
        seconds: Optional[float] = None,
        frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
    ) -> str:
        """Make a short video (an mp4) and return where it was saved.

        Args:
            prompt: What happens in the video
            model: A video model, e.g. ltx-2.3-mlx-q8 on a Mac. Leave empty for the model
                already running.
            seconds: Length (model default if omitted)
            frames: Length in frames; wins over seconds
            width: Width in pixels (model default if omitted)
            height: Height in pixels (model default if omitted)
            seed: Random seed (one is picked and reported if omitted)
            steps: Number of inference steps (model default if omitted)
            image: Path of a picture to start from (image-to-video)
            audio: Path of an audio track to drive it (e.g. a voice for a talking face)
        """
        result = await _run(jobs.Job(
            kind="video", prompt=prompt, model=model, seconds=seconds, frames=frames,
            width=width, height=height, seed=seed, steps=steps, image=image, audio=audio,
        ))
        return _said(result)

    @mcp_server.tool()
    async def list_servers() -> str:
        """List the OllamaDiffuser servers running on this machine and the model each
        holds. A job for one of those models goes to its server instead of loading the
        model a second time, and one for a model no server holds goes to a server that
        loads models on demand. With OLLAMADIFFUSER_SERVER set, the servers on that machine."""
        lines = []
        remote = os.environ.get(jobs.SERVER_ENV)
        if remote:
            from urllib.parse import urlsplit
            try:
                url = jobs.server_url(remote)
            except jobs.JobError as e:
                raise ToolError(str(e))
            lines.append(f"Jobs go to {url} ({jobs.SERVER_ENV} is set).")
            found = await asyncio.to_thread(jobs.servers, urlsplit(url).hostname)
        else:
            found = await asyncio.to_thread(jobs.servers)
        if not found:
            return "\n".join(lines + ["No OllamaDiffuser server is running."])
        for server in found:
            if server["loaded"]:
                what = f"{server['model']} ({server['type']}, {server['device']})"
            else:
                what = "no model loaded" if server["answering"] else "not answering"
            pid = f", pid {server['pid']}" if server["pid"] else ""
            mode = "; loads models on demand" if server.get("on_demand") else ""
            lines.append(f"port {server['port']}: {what}{pid}{mode}")
        return "\n".join(lines)

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
        reason = getattr(model_manager, "last_error", None)
        return reason if isinstance(reason, str) and reason else f"Failed to load model '{model_name}'"

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
