"""OllamaDiffuser CLI - Main entry point"""

import sys
import logging

import click
from rich import print as rprint

from .. import __version__, print_version
from ..core.config.settings import settings
from ..api.server import run_server


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--version", "-V", is_flag=True, help="Show version and exit")
@click.option(
    "--mode",
    type=click.Choice(["cli", "api", "ui"]),
    help="Running mode: cli, api (server), ui (web interface)",
)
@click.option("--host", default=None, help="Server host address (for api/ui modes)")
@click.option("--port", type=int, default=None, help="Server port (for api/ui modes)")
@click.pass_context
def cli(ctx, verbose, version, mode, host, port):
    """OllamaDiffuser - Image generation model management tool"""
    if version:
        print_version()
        sys.exit(0)

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if mode:
        if mode == "api":
            rprint("[blue]Starting OllamaDiffuser API server...[/blue]")
            run_server(host=host, port=port)
            sys.exit(0)
        elif mode == "ui":
            rprint("[blue]Starting OllamaDiffuser Web UI...[/blue]")
            import uvicorn
            from ..ui.web import create_ui_app

            app = create_ui_app()
            ui_host = host or settings.server.host
            ui_port = port or (settings.server.port + 1)
            uvicorn.run(app, host=ui_host, port=ui_port)
            sys.exit(0)

    if ctx.invoked_subcommand is None and not version and not mode:
        rprint(ctx.get_help())
        sys.exit(0)


# --- Register model commands ---
from .model_commands import (
    pull,
    run,
    list,
    show,
    check,
    rm,
    ps,
    serve,
    load,
    unload,
    stop,
)

cli.add_command(pull)
cli.add_command(run)
cli.add_command(list)
cli.add_command(show)
cli.add_command(check)
cli.add_command(rm)
cli.add_command(ps)
cli.add_command(serve)
cli.add_command(load)
cli.add_command(unload)
cli.add_command(stop)

# --- Register LoRA commands ---
from .lora_commands import lora

cli.add_command(lora)

# --- Register registry commands ---
from .registry_commands import registry

cli.add_command(registry)

# --- Register recommend command ---
from .recommend_command import recommend

cli.add_command(recommend)

# --- Utility commands ---


@cli.command()
def version():
    """Show version information"""
    print_version()
    rprint("\n[bold]Supported Models:[/bold]")
    rprint("  FLUX.1-schnell, FLUX.1-dev, SD 3.5 Medium, SDXL, SD 1.5")
    rprint("  ControlNet (SD15 + SDXL), AnimateDiff, HiDream, GGUF variants")
    rprint("\n[bold]Features:[/bold]")
    rprint("  img2img, inpainting, LoRA, ControlNet, async API, Web UI")
    rprint("\n[dim]For help: ollamadiffuser --help[/dim]")


@cli.command(name="verify-deps")
def verify_deps_cmd():
    """Verify and install missing dependencies"""
    from .commands import verify_deps

    ctx = click.Context(verify_deps)
    ctx.invoke(verify_deps)


@cli.command()
def doctor():
    """Run comprehensive system diagnostics"""
    from .commands import doctor

    ctx = click.Context(doctor)
    ctx.invoke(doctor)


@cli.command(name="mcp")
def mcp_cmd():
    """Start the MCP (Model Context Protocol) server for AI assistant integration."""
    try:
        from ..mcp.server import main as mcp_main

        mcp_main()
    except ImportError:
        rprint("[red]MCP package not installed. Install with:[/red]")
        rprint("[yellow]  pip install 'ollamadiffuser[mcp]'[/yellow]")
        sys.exit(1)


@cli.command(name="create-samples")
@click.option("--force", is_flag=True, help="Force recreation of samples")
def create_samples_cmd(force):
    """Create ControlNet sample images for the Web UI"""
    from .commands import create_samples

    ctx = click.Context(create_samples)
    ctx.invoke(create_samples, force=force)


if __name__ == "__main__":
    cli()
