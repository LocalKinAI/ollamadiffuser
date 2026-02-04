"""
OllamaDiffuser - Local AI Image Generation with Ollama-style CLI

A tool for managing and running Stable Diffusion, FLUX.1, and other AI image generation models locally.
"""

__version__ = "2.0.0"
__author__ = "OllamaDiffuser Team"
__email__ = "ollamadiffuser@gmail.com"
__description__ = "🎨 Local AI Image Generation with Ollama-style CLI for Stable Diffusion, FLUX.1, and LoRA support"
__url__ = "https://www.ollamadiffuser.com/"
__repository__ = "https://github.com/ollamadiffuser/ollamadiffuser"

def get_version_info():
    """Get formatted version information"""
    return {
        "version": __version__,
        "description": __description__,
        "url": __url__,
        "repository": __repository__
    }

def print_version():
    """Print formatted version information"""
    from rich import print as rprint
    rprint(f"[bold cyan]OllamaDiffuser v{__version__}[/bold cyan]")
    rprint(__description__)
    rprint(f"🔗 {__url__}")

# For backward compatibility
__all__ = ["__version__", "__author__", "__email__", "__description__", "__url__", "__repository__", "get_version_info", "print_version"]
