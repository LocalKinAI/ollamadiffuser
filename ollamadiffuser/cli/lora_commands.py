"""LoRA management CLI commands"""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from ..core.models.manager import model_manager
from ..core.config.settings import settings

console = Console()


@click.group()
def lora():
    """LoRA (Low-Rank Adaptation) management commands"""
    pass


@lora.command()
@click.argument("repo_id")
@click.option("--weight-name", "-w", help="Specific weight file name")
@click.option("--alias", "-a", help="Local alias name for the LoRA")
def pull(repo_id: str, weight_name: Optional[str], alias: Optional[str]):
    """Download LoRA weights from Hugging Face Hub"""
    from ..core.utils.lora_manager import lora_manager

    rprint(f"[blue]Downloading LoRA: {repo_id}[/blue]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Downloading LoRA...", total=None)

        def cb(msg):
            progress.update(task, description=msg)

        if lora_manager.pull_lora(repo_id, weight_name=weight_name, alias=alias, progress_callback=cb):
            rprint(f"[green]LoRA {repo_id} downloaded successfully![/green]")
        else:
            rprint(f"[red]LoRA {repo_id} download failed![/red]")
            sys.exit(1)


@lora.command()
@click.argument("lora_name")
@click.option("--scale", "-s", default=1.0, type=float, help="LoRA scale (default: 1.0)")
def load(lora_name: str, scale: float):
    """Load LoRA weights into the current model"""
    from ..core.utils.lora_manager import lora_manager

    rprint(f"[blue]Loading LoRA: {lora_name} (scale: {scale})[/blue]")
    if lora_manager.load_lora(lora_name, scale=scale):
        rprint(f"[green]LoRA {lora_name} loaded![/green]")
    else:
        rprint(f"[red]Failed to load LoRA {lora_name}![/red]")
        sys.exit(1)


@lora.command()
def unload():
    """Unload current LoRA weights"""
    from ..core.utils.lora_manager import lora_manager

    if lora_manager.unload_lora():
        rprint("[green]LoRA unloaded![/green]")
    else:
        rprint("[red]Failed to unload LoRA![/red]")
        sys.exit(1)


@lora.command()
@click.argument("lora_name")
@click.confirmation_option(prompt="Are you sure you want to delete this LoRA?")
def rm(lora_name: str):
    """Remove LoRA weights"""
    from ..core.utils.lora_manager import lora_manager

    if lora_manager.remove_lora(lora_name):
        rprint(f"[green]LoRA {lora_name} removed![/green]")
    else:
        rprint(f"[red]Failed to remove LoRA {lora_name}![/red]")
        sys.exit(1)


@lora.command()
def ps():
    """Show currently loaded LoRA status"""
    from ..core.utils.lora_manager import lora_manager

    server_running = lora_manager._is_server_running()
    current_lora = lora_manager.get_current_lora()

    if server_running:
        rprint(f"[green]Server: Running on {settings.server.host}:{settings.server.port}[/green]")
        try:
            import requests
            resp = requests.get(f"http://{settings.server.host}:{settings.server.port}/api/lora/status", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("loaded"):
                    info = data.get("info", {})
                    rprint(f"\n[bold green]LoRA: LOADED[/bold green]")
                    rprint(f"  Repo: {info.get('repo_id', 'Unknown')}")
                    rprint(f"  Scale: {info.get('scale', 'Unknown')}")
                    return
                else:
                    rprint("[dim]No LoRA loaded on server[/dim]")
                    return
        except Exception:
            pass
    elif not model_manager.is_model_loaded():
        rprint("[yellow]No model loaded[/yellow]")
        return

    if current_lora:
        info = lora_manager.get_lora_info(current_lora)
        if info:
            rprint(f"\n[bold green]LoRA: {current_lora}[/bold green]")
            rprint(f"  Repo: {info.get('repo_id', 'Unknown')}")
            rprint(f"  Size: {info.get('size', 'Unknown')}")
    else:
        rprint("[dim]No LoRA loaded[/dim]")


@lora.command()
def list():
    """List installed LoRA weights"""
    from ..core.utils.lora_manager import lora_manager

    installed = lora_manager.list_installed_loras()
    current = lora_manager.get_current_lora()

    if not installed:
        rprint("[yellow]No LoRA weights installed.[/yellow]")
        rprint("[dim]Use 'ollamadiffuser lora pull <repo_id>' to download[/dim]")
        return

    table = Table(title="Installed LoRA Weights")
    table.add_column("Name", style="cyan")
    table.add_column("Repository", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Size", style="yellow")

    for name, info in installed.items():
        status = "Loaded" if name == current else "Available"
        table.add_row(name, info.get("repo_id", "?"), status, info.get("size", "?"))

    console.print(table)


@lora.command()
@click.argument("lora_name")
def show(lora_name: str):
    """Show detailed LoRA information"""
    from ..core.utils.lora_manager import lora_manager

    info = lora_manager.get_lora_info(lora_name)
    if not info:
        rprint(f"[red]LoRA {lora_name} not found.[/red]")
        sys.exit(1)

    rprint(f"[bold cyan]LoRA: {lora_name}[/bold cyan]")
    rprint(f"Repository: {info.get('repo_id', 'Unknown')}")
    rprint(f"Weight File: {info.get('weight_name', 'Unknown')}")
    rprint(f"Path: {info.get('path', 'Unknown')}")
    rprint(f"Size: {info.get('size', 'Unknown')}")
    rprint(f"Downloaded: {info.get('downloaded_at', 'Unknown')}")
