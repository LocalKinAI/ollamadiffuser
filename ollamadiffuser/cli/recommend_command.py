"""Hardware-aware model recommendation command."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..core.config.model_registry import model_registry

console = Console()


def _detect_hardware():
    """Detect hardware capabilities."""
    hw = {
        "device": "cpu",
        "device_name": "CPU",
        "total_ram_gb": 0,
        "available_ram_gb": 0,
        "vram_gb": 0,
    }

    # RAM detection
    try:
        import psutil

        mem = psutil.virtual_memory()
        hw["total_ram_gb"] = round(mem.total / (1024**3), 1)
        hw["available_ram_gb"] = round(mem.available / (1024**3), 1)
    except ImportError:
        pass

    # GPU detection
    try:
        import torch

        if torch.cuda.is_available():
            hw["device"] = "cuda"
            hw["device_name"] = torch.cuda.get_device_name(0)
            hw["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw["device"] = "mps"
            hw["device_name"] = "Apple Silicon (MPS)"
            # On MPS, VRAM is unified memory = total RAM
            hw["vram_gb"] = hw["total_ram_gb"]
    except ImportError:
        pass

    return hw


def _classify_model(model_name, model_info, hw):
    """Classify a model as 'recommended', 'possible', 'too_large', or 'incompatible'.

    Returns (tier, reason).
    """
    hr = model_info.get("hardware_requirements", {})
    supported = [d.upper() for d in hr.get("supported_devices", [])]
    device_upper = hw["device"].upper()

    if device_upper not in supported:
        return "incompatible", f"Not supported on {device_upper}"

    # Determine effective memory budget
    if hw["device"] == "mps":
        # Unified memory: reserve ~4GB for OS/system
        effective_mem = hw["total_ram_gb"] - 4
    elif hw["device"] == "cuda":
        effective_mem = hw["vram_gb"]
    else:
        effective_mem = hw["total_ram_gb"]

    min_vram = hr.get("min_vram_gb", 0)
    rec_vram = hr.get("recommended_vram_gb", min_vram)

    if min_vram > effective_mem:
        return "too_large", f"Needs {min_vram}GB, have ~{effective_mem:.0f}GB"

    if rec_vram <= effective_mem:
        return "recommended", f"Fits well ({min_vram}-{rec_vram}GB needed)"
    else:
        return "possible", f"Tight fit ({min_vram}GB min, {rec_vram}GB rec)"


def _make_table(title, style, entries):
    """Build a rich Table for a tier of models."""
    table = Table(title=title)
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("VRAM", style=style)
    table.add_column("License", style="blue")
    table.add_column("Notes", style="dim")
    for name, info, reason in entries:
        hr = info.get("hardware_requirements", {})
        license_type = info.get("license_info", {}).get("type", "Unknown")
        table.add_row(
            name,
            info.get("model_type", "?"),
            f"{hr.get('min_vram_gb', '?')}-{hr.get('recommended_vram_gb', '?')}GB",
            license_type,
            reason,
        )
    return table


@click.command()
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Target device (default: auto-detect)",
)
@click.option(
    "--all", "show_all", is_flag=True, help="Show all models including incompatible"
)
@click.option(
    "--commercial-only", is_flag=True, help="Only show commercially licensed models"
)
def recommend(device, show_all, commercial_only):
    """Recommend models for your hardware.

    Detects your GPU, memory, and suggests models that will fit.
    """
    hw = _detect_hardware()
    if device != "auto":
        hw["device"] = device

    # Hardware summary
    lines = [
        f"[bold blue]Hardware Detection[/bold blue]",
        f"Device: {hw['device_name']}",
        f"Total RAM: {hw['total_ram_gb']} GB",
    ]
    if hw["device"] == "cuda":
        lines.append(f"VRAM: {hw['vram_gb']} GB")
    elif hw["device"] == "mps":
        lines.append(f"Unified Memory: {hw['total_ram_gb']} GB (MPS uses shared RAM)")
    console.print(Panel.fit("\n".join(lines)))

    # Classify all models
    all_models = model_registry.get_all_models()
    recommended = []
    possible = []
    too_large = []

    for name, info in sorted(all_models.items()):
        if commercial_only:
            license_info = info.get("license_info", {})
            if not license_info.get("commercial_use", False):
                continue

        tier, reason = _classify_model(name, info, hw)
        entry = (name, info, reason)
        if tier == "recommended":
            recommended.append(entry)
        elif tier == "possible":
            possible.append(entry)
        elif tier == "too_large":
            too_large.append(entry)

    # Display tables
    if recommended:
        console.print(_make_table("[green]Recommended Models[/green]", "green", recommended))

    if possible:
        console.print(_make_table("[yellow]Possible (tight fit)[/yellow]", "yellow", possible))

    if show_all and too_large:
        table = Table(title="[red]Too Large for Current Hardware[/red]")
        table.add_column("Model", style="cyan")
        table.add_column("Min VRAM", style="red")
        table.add_column("Reason", style="dim")
        for name, info, reason in too_large:
            hr = info.get("hardware_requirements", {})
            table.add_row(name, f"{hr.get('min_vram_gb', '?')} GB", reason)
        console.print(table)

    # Summary
    total = len(recommended) + len(possible)
    if total == 0:
        console.print("[yellow]No compatible models found for your hardware.[/yellow]")
        console.print("[dim]Try --device cpu to see CPU-compatible models.[/dim]")
    else:
        console.print(
            f"\n[green]Found {len(recommended)} recommended + "
            f"{len(possible)} possible models[/green]"
        )
        # Pick a good quick-start model: prefer standalone gen models over controlnet
        quick_start_priority = [
            "pixart-sigma", "sana-1.5", "flux.1-dev-gguf-q4ks",
            "dreamshaper", "stable-diffusion-1.5",
            "flux.1-schnell", "realvisxl-v4", "sdxl-turbo",
        ]
        top = None
        rec_names = {name for name, _, _ in recommended}
        for candidate in quick_start_priority:
            if candidate in rec_names:
                top = candidate
                break
        if top is None and recommended:
            top = recommended[0][0]
        if top:
            console.print(f"\n[bold]Quick start:[/bold] ollamadiffuser pull {top}")
