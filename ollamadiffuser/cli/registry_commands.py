"""Model registry CLI commands"""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from ..core.models.manager import model_manager
from ..core.config.settings import settings
from ..core.config.model_registry import model_registry

console = Console()


@click.group(hidden=True)
def registry():
    """Manage model registry"""
    pass


@registry.command()
@click.option("--format", "-f", type=click.Choice(["table", "json", "yaml"]), default="table")
@click.option("--installed-only", is_flag=True)
@click.option("--available-only", is_flag=True)
@click.option("--external-only", is_flag=True)
def list(format: str, installed_only: bool, available_only: bool, external_only: bool):
    """List models in the registry"""
    if installed_only:
        models = model_registry.get_installed_models()
        title = "Installed Models"
    elif available_only:
        models = model_registry.get_available_models()
        title = "Available Models"
    elif external_only:
        models = model_registry.get_external_api_models_only()
        title = "External API Models"
    else:
        models = model_registry.get_all_models()
        title = "All Models"

    installed_names = set(model_registry.get_installed_models().keys())
    current = model_manager.get_current_model()

    if not models:
        rprint(f"[yellow]No models found: {title}[/yellow]")
        return

    if format == "table":
        table = Table(title=title)
        table.add_column("Model", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Repository", style="blue")
        table.add_column("Status", style="green")

        for name, info in models.items():
            status = "Installed" if name in installed_names else "Available"
            if name == current:
                status += " (current)"
            table.add_row(name, info.get("model_type", "?"), info.get("repo_id", "?"), status)

        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} | Installed: {len(installed_names)}[/dim]")

    elif format == "json":
        import json
        print(json.dumps(models, indent=2, ensure_ascii=False))
    elif format == "yaml":
        import yaml
        print(yaml.dump(models, default_flow_style=False, allow_unicode=True))


@registry.command()
@click.argument("model_name")
@click.argument("repo_id")
@click.argument("model_type")
@click.option("--variant", help="Model variant")
@click.option("--save", is_flag=True, help="Save to config file")
def add(model_name: str, repo_id: str, model_type: str, variant: Optional[str], save: bool):
    """Add a model to the registry"""
    config = {"repo_id": repo_id, "model_type": model_type}
    if variant:
        config["variant"] = variant

    if model_registry.add_model(model_name, config):
        rprint(f"[green]Model '{model_name}' added![/green]")
        if save:
            import json
            config_path = settings.config_dir / "models.json"
            user_models = {}
            if config_path.exists():
                with open(config_path) as f:
                    data = json.load(f)
                    user_models = data.get("models", {})
            user_models[model_name] = config
            model_registry.save_user_config(user_models, config_path)
            rprint(f"[green]Saved to {config_path}[/green]")
    else:
        rprint(f"[red]Failed to add '{model_name}'![/red]")
        sys.exit(1)


@registry.command()
@click.argument("model_name")
def remove(model_name: str):
    """Remove a model from the registry"""
    if model_registry.remove_model(model_name):
        rprint(f"[green]Model '{model_name}' removed![/green]")
    else:
        rprint(f"[red]Model '{model_name}' not found![/red]")
        sys.exit(1)


@registry.command()
def reload():
    """Reload the registry from config files"""
    model_registry.reload()
    models = model_registry.get_all_models()
    rprint(f"[green]Registry reloaded: {len(models)} models[/green]")


@registry.command("import-config")
@click.argument("config_file", type=click.Path(exists=True))
def import_config(config_file: str):
    """Import models from a config file"""
    from pathlib import Path
    import json
    import yaml

    path = Path(config_file)
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".json":
            data = json.load(f)
        elif path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            rprint(f"[red]Unsupported format: {path.suffix}[/red]")
            sys.exit(1)

    if "models" not in data:
        rprint("[red]Config must contain 'models' section[/red]")
        sys.exit(1)

    count = 0
    for name, cfg in data["models"].items():
        if model_registry.add_model(name, cfg):
            count += 1
    rprint(f"[green]Imported {count} models[/green]")


@registry.command()
@click.option("--output", "-o", help="Output file path")
@click.option("--format", "-f", type=click.Choice(["json", "yaml"]), default="json")
def export(output: Optional[str], format: str):
    """Export registry to a file"""
    from pathlib import Path
    import json
    import yaml

    models = model_registry.get_all_models()
    data = {"models": models}

    out = Path(output) if output else Path(f"models.{format}")
    with open(out, "w", encoding="utf-8") as f:
        if format == "json":
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    rprint(f"[green]Exported {len(models)} models to {out}[/green]")


@registry.command("check-gguf")
def check_gguf():
    """Check GGUF support status"""
    from ..core.models.gguf_loader import GGUF_AVAILABLE

    if GGUF_AVAILABLE:
        rprint("[green]GGUF Support Available[/green]")
        models = model_registry.get_all_models()
        gguf_models = {n: i for n, i in models.items() if model_manager.is_gguf_model(n)}
        if gguf_models:
            table = Table()
            table.add_column("Model", style="cyan")
            table.add_column("Variant", style="yellow")
            table.add_column("VRAM", style="green")
            table.add_column("Installed")
            for name, info in gguf_models.items():
                hw = info.get("hardware_requirements", {})
                installed = "Yes" if model_manager.is_model_installed(name) else "No"
                table.add_row(name, info.get("variant", "?"), f"{hw.get('min_vram_gb', '?')}GB", installed)
            console.print(table)
    else:
        rprint("[red]GGUF Support Not Available[/red]")
        rprint("Install with: pip install 'ollamadiffuser[gguf]'")
