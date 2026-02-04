"""Configuration management CLI commands"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from ..core.config.settings import settings

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """Show or modify OllamaDiffuser configuration"""
    if ctx.invoked_subcommand is None:
        _show_config()


def _show_config():
    """Display all current configuration values."""
    # Paths section
    rprint("[bold cyan]Paths[/bold cyan]")
    table = Table(show_header=True)
    table.add_column("Key", style="yellow")
    table.add_column("Value", style="white")

    default_models_dir = settings.config_dir / "models"
    default_cache_dir = settings.config_dir / "cache"

    table.add_row("config_dir", str(settings.config_dir))
    table.add_row("config_file", str(settings.config_file))

    models_label = str(settings.models_dir)
    if settings.models_dir != default_models_dir:
        models_label += "  [dim](custom)[/dim]"
    table.add_row("models_dir", models_label)

    cache_label = str(settings.cache_dir)
    if settings.cache_dir != default_cache_dir:
        cache_label += "  [dim](custom)[/dim]"
    table.add_row("cache_dir", cache_label)

    console.print(table)

    # Server section
    rprint("\n[bold cyan]Server[/bold cyan]")
    server_table = Table(show_header=True)
    server_table.add_column("Key", style="yellow")
    server_table.add_column("Value", style="white")
    server_table.add_row("server.host", settings.server.host)
    server_table.add_row("server.port", str(settings.server.port))
    server_table.add_row("server.max_queue_size", str(settings.server.max_queue_size))
    server_table.add_row("server.timeout", str(settings.server.timeout))
    server_table.add_row("server.enable_cors", str(settings.server.enable_cors))
    console.print(server_table)


# Map of settable keys to (type_hint, setter_function)
SETTABLE_KEYS = {
    "models_dir": ("path", lambda v: _set_path("models_dir", v)),
    "cache_dir": ("path", lambda v: _set_path("cache_dir", v)),
    "server.host": ("str", lambda v: setattr(settings.server, "host", v)),
    "server.port": ("int", lambda v: setattr(settings.server, "port", int(v))),
    "server.max_queue_size": ("int", lambda v: setattr(settings.server, "max_queue_size", int(v))),
    "server.timeout": ("int", lambda v: setattr(settings.server, "timeout", int(v))),
    "server.enable_cors": ("bool", lambda v: setattr(settings.server, "enable_cors", v.lower() in ("true", "1", "yes"))),
}


def _set_path(attr: str, value: str):
    """Set a path attribute on settings, creating the directory."""
    p = Path(value).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    setattr(settings, attr, p)


@config.command("set")
@click.argument("key")
@click.argument("value")
def set_value(key: str, value: str):
    """Set a configuration value.

    \b
    Settable keys:
      models_dir              Custom model storage directory
      cache_dir               Custom cache directory
      server.host             Server bind address
      server.port             Server port number
      server.max_queue_size   Maximum request queue size
      server.timeout          Request timeout in seconds
      server.enable_cors      Enable CORS (true/false)

    \b
    Examples:
      ollamadiffuser config set models_dir /mnt/ssd/models
      ollamadiffuser config set server.port 9000
    """
    if key not in SETTABLE_KEYS:
        rprint(f"[red]Unknown key: {key}[/red]")
        rprint(f"[dim]Settable keys: {', '.join(sorted(SETTABLE_KEYS))}[/dim]")
        sys.exit(1)

    type_hint, setter = SETTABLE_KEYS[key]

    try:
        if type_hint == "int":
            int(value)  # pre-validate
        setter(value)
    except (ValueError, OSError) as e:
        rprint(f"[red]Invalid value for {key}: {e}[/red]")
        sys.exit(1)

    settings.save_config()
    rprint(f"[green]Set {key} = {value}[/green]")
