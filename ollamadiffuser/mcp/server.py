"""OllamaDiffuser MCP Server - Model Context Protocol integration."""

import asyncio
import io
import logging
import sys
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


def create_mcp_server():
    """Create and configure the MCP server with all tools."""
    from mcp.server.fastmcp import FastMCP, Image

    mcp_server = FastMCP(
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
    ) -> Image:
        """Generate an image from a text prompt using a local diffusion model.

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
        if model and model_manager.get_current_model() != model:
            if not model_manager.is_model_installed(model):
                raise ValueError(
                    f"Model '{model}' is not installed. "
                    f"Install it first: ollamadiffuser pull {model}"
                )
            logger.info(f"Loading model: {model}")
            success = await asyncio.to_thread(model_manager.load_model, model)
            if not success:
                raise RuntimeError(f"Failed to load model '{model}'")

        if not model_manager.is_model_loaded():
            raise RuntimeError(
                "No model loaded. Load one with: load_model('model-name') "
                "or pass model= parameter"
            )

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
        return Image(data=buf.getvalue(), format="png")

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


def main():
    """Entry point for the MCP server (stdio transport)."""
    if not _ensure_mcp():
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
