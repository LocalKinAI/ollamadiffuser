"""Model-related CLI commands for OllamaDiffuser."""

import sys
import subprocess
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from ..core.models.manager import model_manager
from ..core.config.settings import settings
from ..core.config.model_registry import model_registry
from ..api.server import run_server

console = Console()


class OllamaStyleProgress:
    """Enhanced progress tracker that mimics Ollama's progress display"""

    def __init__(self, console: Console):
        self.console = console
        self.last_message = ""

    def update(self, message: str):
        """Update progress with a message"""
        # Skip duplicate messages
        if message == self.last_message:
            return

        self.last_message = message

        # Handle different types of messages
        if message.startswith("pulling ") and ":" in message and "%" in message:
            # This is a file progress message from download_utils
            # Format: "pulling e6a7edc1a4d7: 12% ▕██                ▏ 617 MB/5200 MB 44 MB/s 1m44s"
            self.console.print(message)
        elif message.startswith("pulling manifest"):
            self.console.print(message)
        elif message.startswith("📦 Repository:"):
            # Repository info
            self.console.print(f"[dim]{message}[/dim]")
        elif message.startswith("📁 Found"):
            # Existing files info
            self.console.print(f"[dim]{message}[/dim]")
        elif message.startswith("✅") and "download completed" in message:
            self.console.print(f"[green]{message}[/green]")
        elif message.startswith("❌"):
            self.console.print(f"[red]{message}[/red]")
        elif message.startswith("⚠️"):
            self.console.print(f"[yellow]{message}[/yellow]")
        else:
            # For other messages, print with dimmed style
            self.console.print(f"[dim]{message}[/dim]")


@click.command()
@click.argument('model_name')
@click.option('--force', '-f', is_flag=True, help='Force re-download')
def pull(model_name: str, force: bool):
    """Download model"""
    rprint(f"[blue]Downloading model: {model_name}[/blue]")

    # Use the new Ollama-style progress tracker
    progress_tracker = OllamaStyleProgress(console)

    def progress_callback(message: str):
        """Enhanced progress callback with Ollama-style display"""
        progress_tracker.update(message)

    try:
        if model_manager.pull_model(model_name, force=force, progress_callback=progress_callback):
            progress_tracker.update("✅ download completed")
            rprint(f"[green]Model {model_name} downloaded successfully![/green]")
        else:
            rprint(f"[red]Model {model_name} download failed![/red]")
            sys.exit(1)
    except KeyboardInterrupt:
        rprint("\n[yellow]Download cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        rprint(f"[red]Download failed: {str(e)}[/red]")
        sys.exit(1)


@click.command()
@click.argument('model_name')
@click.option('--host', '-h', default=None, help='Server host address')
@click.option('--port', '-p', default=None, type=int, help='Server port')
def run(model_name: str, host: Optional[str], port: Optional[int]):
    """Run model service"""
    rprint(f"[blue]Starting model service: {model_name}[/blue]")

    # Check if model is installed
    if not model_manager.is_model_installed(model_name):
        rprint(f"[red]Model {model_name} is not installed. Please run first: ollamadiffuser pull {model_name}[/red]")
        sys.exit(1)

    # Load model
    rprint("[yellow]Loading model...[/yellow]")
    if not model_manager.load_model(model_name):
        rprint(f"[red]Failed to load model {model_name}![/red]")
        sys.exit(1)

    rprint(f"[green]Model {model_name} loaded successfully![/green]")

    # Start server
    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        rprint("\n[yellow]Server stopped[/yellow]")
        model_manager.unload_model()
        # Clear the current model from settings when server stops
        settings.current_model = None
        settings.save_config()


@click.command()
@click.option('--hardware', '-hw', is_flag=True, help='Show hardware requirements')
def list(hardware: bool):
    """List installed models only"""
    installed_models = model_manager.list_installed_models()
    current_model = model_manager.get_current_model()

    if not installed_models:
        rprint("[yellow]No models installed[/yellow]")
        rprint("\n[dim]💡 Download models with: ollamadiffuser pull <model-name>[/dim]")
        rprint("[dim]💡 See all available models: ollamadiffuser registry list[/dim]")
        rprint("[dim]💡 See only available models: ollamadiffuser registry list --available-only[/dim]")
        return

    if hardware:
        # Show detailed hardware requirements
        for model_name in installed_models:
            info = model_manager.get_model_info(model_name)
            if not info:
                continue

            # Check installation status
            status = "✅ Installed"
            if model_name == current_model:
                status += " (current)"
            size = info.get('size', 'Unknown')

            # Create individual table for each model
            table = Table(title=f"[bold cyan]{model_name}[/bold cyan] - {status}")
            table.add_column("Property", style="yellow", no_wrap=True)
            table.add_column("Value", style="white")

            # Basic info
            table.add_row("Type", info.get('model_type', 'Unknown'))
            table.add_row("Size", size)

            # Hardware requirements
            hw_req = info.get('hardware_requirements', {})
            if hw_req:
                table.add_row("Min VRAM", f"{hw_req.get('min_vram_gb', 'Unknown')} GB")
                table.add_row("Recommended VRAM", f"{hw_req.get('recommended_vram_gb', 'Unknown')} GB")
                table.add_row("Min RAM", f"{hw_req.get('min_ram_gb', 'Unknown')} GB")
                table.add_row("Recommended RAM", f"{hw_req.get('recommended_ram_gb', 'Unknown')} GB")
                table.add_row("Disk Space", f"{hw_req.get('disk_space_gb', 'Unknown')} GB")
                table.add_row("Supported Devices", ", ".join(hw_req.get('supported_devices', [])))
                table.add_row("Performance Notes", hw_req.get('performance_notes', 'N/A'))

            console.print(table)
            console.print()  # Add spacing between models
    else:
        # Show compact table
        table = Table(title="Installed Models")
        table.add_column("Model Name", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Size", style="blue")
        table.add_column("Type", style="magenta")
        table.add_column("Min VRAM", style="yellow")

        for model_name in installed_models:
            # Check installation status
            status = "✅ Installed"
            if model_name == current_model:
                status += " (current)"

            # Get model information
            info = model_manager.get_model_info(model_name)
            size = info.get('size', 'Unknown') if info else 'Unknown'
            model_type = info.get('model_type', 'Unknown') if info else 'Unknown'

            # Get hardware requirements
            hw_req = info.get('hardware_requirements', {}) if info else {}
            min_vram = f"{hw_req.get('min_vram_gb', '?')} GB" if hw_req else "Unknown"

            table.add_row(model_name, status, size, model_type, min_vram)

        console.print(table)

        # Get counts for summary
        available_models = model_registry.get_available_models()
        external_models = model_registry.get_external_api_models_only()

        console.print(f"\n[dim]💡 Installed: {len(installed_models)} models[/dim]")
        console.print(f"[dim]💡 Available for download: {len(available_models)} models[/dim]")
        if external_models:
            console.print(f"[dim]💡 External API models: {len(external_models)} models[/dim]")
        console.print("\n[dim]💡 Use --hardware flag to see detailed hardware requirements[/dim]")
        console.print("[dim]💡 See all models: ollamadiffuser registry list[/dim]")
        console.print("[dim]💡 See available models: ollamadiffuser registry list --available-only[/dim]")


@click.command()
@click.argument('model_name')
def show(model_name: str):
    """Show model detailed information"""
    info = model_manager.get_model_info(model_name)

    if info is None:
        rprint(f"[red]Model {model_name} does not exist[/red]")
        sys.exit(1)

    rprint(f"[bold cyan]Model Information: {model_name}[/bold cyan]")
    rprint(f"Type: {info.get('model_type', 'Unknown')}")
    rprint(f"Variant: {info.get('variant', 'Unknown')}")
    rprint(f"Installed: {'Yes' if info.get('installed', False) else 'No'}")

    if info.get('installed', False):
        rprint(f"Local Path: {info.get('local_path', 'Unknown')}")
        rprint(f"Size: {info.get('size', 'Unknown')}")

    # Hardware requirements
    if 'hardware_requirements' in info and info['hardware_requirements']:
        hw_req = info['hardware_requirements']
        rprint("\n[bold]Hardware Requirements:[/bold]")
        rprint(f"  Min VRAM: {hw_req.get('min_vram_gb', 'Unknown')} GB")
        rprint(f"  Recommended VRAM: {hw_req.get('recommended_vram_gb', 'Unknown')} GB")
        rprint(f"  Min RAM: {hw_req.get('min_ram_gb', 'Unknown')} GB")
        rprint(f"  Recommended RAM: {hw_req.get('recommended_ram_gb', 'Unknown')} GB")
        rprint(f"  Disk Space: {hw_req.get('disk_space_gb', 'Unknown')} GB")
        rprint(f"  Supported Devices: {', '.join(hw_req.get('supported_devices', []))}")
        if hw_req.get('performance_notes'):
            rprint(f"  Performance Notes: {hw_req.get('performance_notes')}")

    if 'parameters' in info and info['parameters']:
        rprint("\n[bold]Default Parameters:[/bold]")
        for key, value in info['parameters'].items():
            rprint(f"  {key}: {value}")

    if 'components' in info and info['components']:
        rprint("\n[bold]Components:[/bold]")
        for key, value in info['components'].items():
            rprint(f"  {key}: {value}")


@click.command()
@click.argument('model_name', required=False)
@click.option('--list', '-l', is_flag=True, help='List all available models')
def check(model_name: str, list: bool):
    """Check model download status and integrity"""
    if list:
        rprint("[bold blue]📋 Available Models:[/bold blue]")
        available_models = model_manager.list_available_models()
        for model in available_models:
            model_info = model_manager.get_model_info(model)
            status = "✅ Installed" if model_manager.is_model_installed(model) else "⬇️ Available"
            license_type = model_info.get("license_info", {}).get("type", "Unknown")
            rprint(f"   {model:<30} {status:<15} ({license_type})")
        return

    if not model_name:
        rprint("[bold red]❌ Please specify a model name or use --list[/bold red]")
        rprint("[dim]Usage: ollamadiffuser check MODEL_NAME[/dim]")
        rprint("[dim]       ollamadiffuser check --list[/dim]")
        return

    # Check model download status directly
    status = _check_download_status(model_name)

    rprint("\n" + "="*60)

    if status is True:
        rprint(f"[green]🎉 {model_name} is ready to use![/green]")
        rprint(f"\n[blue]💡 You can now run:[/blue]")
        rprint(f"   [cyan]ollamadiffuser run {model_name}[/cyan]")
    elif status == "needs_config":
        rprint(f"[yellow]⚠️ {model_name} files are complete but model needs configuration[/yellow]")
        rprint(f"\n[blue]💡 Try reinstalling:[/blue]")
        rprint(f"   [cyan]ollamadiffuser pull {model_name} --force[/cyan]")
    elif status == "downloading":
        rprint(f"[yellow]🔄 {model_name} is currently downloading[/yellow]")
        rprint(f"\n[blue]💡 Wait for download to complete or check progress[/blue]")
    elif status == "incomplete":
        rprint(f"[yellow]⚠️ Download is incomplete[/yellow]")
        rprint(f"\n[blue]💡 Resume download with:[/blue]")
        rprint(f"   [cyan]ollamadiffuser pull {model_name}[/cyan]")
        rprint(f"\n[blue]💡 Or force fresh download with:[/blue]")
        rprint(f"   [cyan]ollamadiffuser pull {model_name} --force[/cyan]")
    else:
        rprint(f"[red]❌ {model_name} is not downloaded[/red]")
        rprint(f"\n[blue]💡 Download with:[/blue]")
        rprint(f"   [cyan]ollamadiffuser pull {model_name}[/cyan]")

    _show_model_specific_help(model_name)

    rprint(f"\n[dim]📚 For more help: ollamadiffuser --help[/dim]")


def _check_download_status(model_name: str):
    """Check the current download status of any model"""
    from ..core.utils.download_utils import check_download_integrity, get_repo_file_list, format_size
    import subprocess

    rprint(f"[blue]🔍 Checking {model_name} download status...[/blue]\n")

    # Check if model is in registry
    if model_name not in model_manager.model_registry:
        rprint(f"[red]❌ {model_name} not found in model registry[/red]")
        available_models = model_manager.list_available_models()
        rprint(f"[blue]📋 Available models: {', '.join(available_models)}[/blue]")
        return False

    model_info = model_manager.model_registry[model_name]
    repo_id = model_info["repo_id"]
    model_path = settings.get_model_path(model_name)

    rprint(f"[cyan]📦 Model: {model_name}[/cyan]")
    rprint(f"[cyan]🔗 Repository: {repo_id}[/cyan]")
    rprint(f"[cyan]📁 Local path: {model_path}[/cyan]")

    # Show model-specific info
    license_info = model_info.get("license_info", {})
    if license_info:
        rprint(f"[yellow]📄 License: {license_info.get('type', 'Unknown')}[/yellow]")
        rprint(f"[yellow]🔑 HF Token Required: {'Yes' if license_info.get('requires_agreement', False) else 'No'}[/yellow]")
        rprint(f"[yellow]💼 Commercial Use: {'Allowed' if license_info.get('commercial_use', False) else 'Not Allowed'}[/yellow]")

    # Show optimal parameters
    params = model_info.get("parameters", {})
    if params:
        rprint(f"[green]⚡ Optimal Settings:[/green]")
        rprint(f"   Steps: {params.get('num_inference_steps', 'N/A')}")
        rprint(f"   Guidance: {params.get('guidance_scale', 'N/A')}")
        if 'max_sequence_length' in params:
            rprint(f"   Max Seq Length: {params['max_sequence_length']}")

    rprint()

    # Check if directory exists
    if not model_path.exists():
        rprint("[yellow]📂 Status: Not downloaded[/yellow]")
        return False

    # Get repository file list
    rprint("[blue]🌐 Getting repository information...[/blue]")
    try:
        file_sizes = get_repo_file_list(repo_id)
        total_expected_size = sum(file_sizes.values())
        total_files_expected = len(file_sizes)

        rprint(f"[blue]📊 Expected: {total_files_expected} files, {format_size(total_expected_size)} total[/blue]")
    except Exception as e:
        rprint(f"[yellow]⚠️ Could not get repository info: {e}[/yellow]")
        file_sizes = {}
        total_expected_size = 0
        total_files_expected = 0

    # Check local files
    local_files = []
    local_size = 0

    for file_path in model_path.rglob('*'):
        if file_path.is_file():
            rel_path = file_path.relative_to(model_path)
            file_size = file_path.stat().st_size
            local_files.append((str(rel_path), file_size))
            local_size += file_size

    rprint(f"[blue]💾 Downloaded: {len(local_files)} files, {format_size(local_size)} total[/blue]")

    if total_expected_size > 0:
        progress_percent = (local_size / total_expected_size) * 100
        rprint(f"[blue]📈 Progress: {progress_percent:.1f}%[/blue]")

    rprint()

    # Check for missing files
    if file_sizes:
        # Check if we have size information from the API
        has_size_info = any(size > 0 for size in file_sizes.values())

        if has_size_info:
            # Normal case: we have size information, do detailed comparison
            missing_files = []
            incomplete_files = []

            for expected_file, expected_size in file_sizes.items():
                local_file_path = model_path / expected_file
                if not local_file_path.exists():
                    missing_files.append(expected_file)
                elif expected_size > 0 and local_file_path.stat().st_size != expected_size:
                    local_size_actual = local_file_path.stat().st_size
                    incomplete_files.append((expected_file, local_size_actual, expected_size))

            if missing_files:
                rprint(f"[red]❌ Missing files ({len(missing_files)}):[/red]")
                for missing_file in missing_files[:10]:  # Show first 10
                    rprint(f"   - {missing_file}")
                if len(missing_files) > 10:
                    rprint(f"   ... and {len(missing_files) - 10} more")
                rprint()

            if incomplete_files:
                rprint(f"[yellow]⚠️ Incomplete files ({len(incomplete_files)}):[/yellow]")
                for incomplete_file, actual_size, expected_size in incomplete_files[:5]:
                    rprint(f"   - {incomplete_file}: {format_size(actual_size)}/{format_size(expected_size)}")
                if len(incomplete_files) > 5:
                    rprint(f"   ... and {len(incomplete_files) - 5} more")
                rprint()

            if not missing_files and not incomplete_files:
                rprint("[green]✅ All files present and complete![/green]")

                # Check integrity
                rprint("[blue]🔍 Checking download integrity...[/blue]")
                if check_download_integrity(str(model_path), repo_id):
                    rprint("[green]✅ Download integrity verified![/green]")

                    # Check if model is in configuration
                    if model_manager.is_model_installed(model_name):
                        rprint("[green]✅ Model is properly configured[/green]")
                        return True
                    else:
                        rprint("[yellow]⚠️ Model files complete but not in configuration[/yellow]")
                        return "needs_config"
                else:
                    rprint("[red]❌ Download integrity check failed[/red]")
                    return False
            else:
                rprint("[yellow]⚠️ Download is incomplete[/yellow]")
                return "incomplete"
        else:
            # No size information available from API (common with gated repos)
            rprint("[blue]ℹ️ Repository API doesn't provide file sizes (common with gated models)[/blue]")
            rprint("[blue]🔍 Checking essential model files instead...[/blue]")

            # Check for essential model files
            # Determine model type based on repo_id
            is_controlnet = 'controlnet' in repo_id.lower()

            if is_controlnet:
                # ControlNet models have different essential files
                essential_files = ['config.json']
                essential_dirs = []  # ControlNet models don't have complex directory structure
            else:
                # Regular diffusion models
                essential_files = ['model_index.json']
                essential_dirs = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'scheduler']

            missing_essential = []
            for essential_file in essential_files:
                if not (model_path / essential_file).exists():
                    missing_essential.append(essential_file)

            existing_dirs = []
            for essential_dir in essential_dirs:
                if (model_path / essential_dir).exists():
                    existing_dirs.append(essential_dir)

            if missing_essential:
                rprint(f"[red]❌ Missing essential files: {', '.join(missing_essential)}[/red]")
                return "incomplete"

            if existing_dirs:
                rprint(f"[green]✅ Found model components: {', '.join(existing_dirs)}[/green]")

            # Check integrity
            rprint("[blue]🔍 Checking download integrity...[/blue]")
            if check_download_integrity(str(model_path), repo_id):
                rprint("[green]✅ Download integrity verified![/green]")

                # Check if model is in configuration
                if model_manager.is_model_installed(model_name):
                    rprint("[green]✅ Model is properly configured and functional[/green]")
                    return True
                else:
                    rprint("[yellow]⚠️ Model files complete but not in configuration[/yellow]")
                    return "needs_config"
            else:
                rprint("[red]❌ Download integrity check failed[/red]")
                return False

    # Check if download process is running
    rprint("[blue]🔍 Checking for active download processes...[/blue]")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if f'ollamadiffuser pull {model_name}' in result.stdout:
            rprint("[yellow]🔄 Download process is currently running[/yellow]")
            return "downloading"
        else:
            rprint("[blue]💤 No active download process found[/blue]")
    except Exception as e:
        rprint(f"[yellow]⚠️ Could not check processes: {e}[/yellow]")

    return "incomplete"


def _show_model_specific_help(model_name: str):
    """Show model-specific help and recommendations"""
    model_info = model_manager.get_model_info(model_name)
    if not model_info:
        return

    rprint(f"\n[bold blue]💡 {model_name} Specific Tips:[/bold blue]")

    # License-specific help
    license_info = model_info.get("license_info", {})
    if license_info.get("requires_agreement", False):
        rprint(f"   [yellow]🔑 Requires HuggingFace token and license agreement[/yellow]")
        rprint(f"   [blue]📝 Visit: https://huggingface.co/{model_info['repo_id']}[/blue]")
        rprint(f"   [cyan]🔧 Set token: export HF_TOKEN=your_token_here[/cyan]")
    else:
        rprint(f"   [green]✅ No HuggingFace token required![/green]")

    # Model-specific optimizations
    if "schnell" in model_name.lower():
        rprint(f"   [green]⚡ FLUX.1-schnell is 12x faster than FLUX.1-dev[/green]")
        rprint(f"   [green]🎯 Optimized for 4-step generation[/green]")
        rprint(f"   [green]💼 Commercial use allowed (Apache 2.0)[/green]")
    elif "flux.1-dev" in model_name.lower():
        rprint(f"   [blue]🎨 Best quality FLUX model[/blue]")
        rprint(f"   [blue]🔬 Requires 50 steps for optimal results[/blue]")
        rprint(f"   [yellow]⚠️ Non-commercial license only[/yellow]")
    elif "stable-diffusion-1.5" in model_name.lower():
        rprint(f"   [green]🚀 Great for learning and quick tests[/green]")
        rprint(f"   [green]💾 Smallest model, runs on most hardware[/green]")
    elif "stable-diffusion-3.5" in model_name.lower():
        rprint(f"   [green]🏆 Excellent quality-to-speed ratio[/green]")
        rprint(f"   [green]🔄 Great LoRA ecosystem[/green]")

    # Hardware recommendations
    hw_req = model_info.get("hardware_requirements", {})
    if hw_req:
        min_vram = hw_req.get("min_vram_gb", 0)
        if min_vram >= 12:
            rprint(f"   [yellow]🖥️ Requires high-end GPU (RTX 4070+ or M2 Pro+)[/yellow]")
        elif min_vram >= 8:
            rprint(f"   [blue]🖥️ Requires mid-range GPU (RTX 3080+ or M1 Pro+)[/blue]")
        else:
            rprint(f"   [green]🖥️ Runs on most modern GPUs[/green]")


@click.command()
@click.argument('model_name')
@click.confirmation_option(prompt='Are you sure you want to delete this model?')
def rm(model_name: str):
    """Remove model"""
    if model_manager.remove_model(model_name):
        rprint(f"[green]Model {model_name} removed successfully![/green]")
    else:
        rprint(f"[red]Failed to remove model {model_name}![/red]")
        sys.exit(1)


@click.command()
def ps():
    """Show currently running model"""
    current_model = model_manager.get_current_model()
    server_running = model_manager.is_server_running()

    if current_model:
        rprint(f"[green]Current model: {current_model}[/green]")

        # Check server status
        if server_running:
            rprint(f"[green]Server status: Running on {settings.server.host}:{settings.server.port}[/green]")

            # Try to get model info from the running server
            try:
                import requests
                response = requests.get(f"http://{settings.server.host}:{settings.server.port}/api/models/running", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('loaded'):
                        info = data.get('info', {})
                        rprint(f"Device: {info.get('device', 'Unknown')}")
                        rprint(f"Type: {info.get('type', 'Unknown')}")
                        rprint(f"Variant: {info.get('variant', 'Unknown')}")
                    else:
                        rprint("[yellow]Model loaded but not active in server[/yellow]")
            except:
                pass
        else:
            rprint("[yellow]Server status: Not running[/yellow]")
            rprint("[dim]Model is set as current but server is not active[/dim]")

        # Show model info from local config
        model_info = model_manager.get_model_info(current_model)
        if model_info:
            rprint(f"Model type: {model_info.get('model_type', 'Unknown')}")
            if model_info.get('installed'):
                rprint(f"Size: {model_info.get('size', 'Unknown')}")
    else:
        if server_running:
            rprint("[yellow]Server is running but no model is loaded[/yellow]")
            rprint(f"[green]Server status: Running on {settings.server.host}:{settings.server.port}[/green]")
        else:
            rprint("[yellow]No model is currently running[/yellow]")
            rprint("[dim]Use 'ollamadiffuser run <model>' to start a model[/dim]")


@click.command()
@click.argument('model_name')
def load(model_name: str):
    """Load model into memory"""
    rprint(f"[blue]Loading model: {model_name}[/blue]")

    if model_manager.load_model(model_name):
        rprint(f"[green]Model {model_name} loaded successfully![/green]")
    else:
        rprint(f"[red]Failed to load model {model_name}![/red]")
        sys.exit(1)


@click.command()
def unload():
    """Unload current model"""
    if model_manager.is_model_loaded():
        current_model = model_manager.get_current_model()
        model_manager.unload_model()
        rprint(f"[green]Model {current_model} unloaded[/green]")
    else:
        rprint("[yellow]No model to unload[/yellow]")


@click.command()
@click.option('--host', '-h', default=None, help='Server host address')
@click.option('--port', '-p', default=None, type=int, help='Server port')
def serve(host: Optional[str], port: Optional[int]):
    """Start the API server without loading a model"""
    rprint("[blue]Starting OllamaDiffuser API server...[/blue]")
    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        rprint("\n[yellow]Server stopped[/yellow]")


@click.command()
def stop():
    """Stop the running server"""
    import requests

    server_host = settings.server.host
    server_port = settings.server.port

    try:
        response = requests.post(
            f"http://{server_host}:{server_port}/api/shutdown", timeout=5
        )
        if response.status_code == 200:
            rprint("[green]Server shutdown initiated[/green]")
        else:
            rprint(f"[red]Failed to stop server: {response.status_code}[/red]")
    except requests.ConnectionError:
        rprint("[yellow]No server running or already stopped[/yellow]")
    except Exception as e:
        rprint(f"[red]Error stopping server: {e}[/red]")
