import click
import subprocess
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from ..core.models.manager import model_manager
from ..core.config.settings import settings

console = Console()


def _has_nvidia_gpu() -> bool:
    """True when an NVIDIA driver is present. nvidia-smi rather than torch:
    a CPU-only torch wheel says no CUDA on a machine that has a GPU."""
    import shutil

    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        return subprocess.run([smi, "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False



# Optional backends that are NOT installed by default, keyed by the name
# users pass to `ollamadiffuser enable <name>`. The default install is
# intentionally compile-free; these are opt-in because they either need
# extra wheels (mlx, mcp) or compile a native extension (gguf).
def _build_enable_command(backend: str):
    """Build the pip invocation for an optional backend.

    Returns ``(argv, env_overrides_or_None, human_note)``.
    Raises ``ValueError`` for an unknown backend or an unsupported platform.
    """
    import os
    import platform

    if backend == "mlx":
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            raise ValueError(
                "MLX backend requires Apple Silicon (macOS arm64). "
                f"Detected {platform.system()}/{platform.machine()}. "
                "Use the default PyTorch path, or 'enable gguf' for low-VRAM."
            )
        return (
            [sys.executable, "-m", "pip", "install", "mflux>=0.19.0"],
            None,
            "Apple Silicon native inference — typically 2-3x faster than PyTorch+MPS.",
        )

    if backend == "gguf":
        # stable-diffusion-cpp builds CPU-only unless told otherwise, and says
        # nothing about it — so pick the GPU backend here. A CMAKE_ARGS the
        # user already exported wins: they may want ROCm, Vulkan or SYCL.
        env_overrides = None
        note = "Low-VRAM quantized models. Compiles a native extension (~1-3 min)."
        if os.environ.get("CMAKE_ARGS"):
            note += " Using your CMAKE_ARGS."
        elif platform.system() == "Darwin":
            env_overrides = {"CMAKE_ARGS": "-DSD_METAL=ON"}
        elif _has_nvidia_gpu():
            env_overrides = {"CMAKE_ARGS": "-DSD_CUDA=ON"}
            note += " NVIDIA GPU found: building with CUDA (needs the CUDA toolkit)."
        else:
            note += (" No GPU backend detected: this will be a CPU build. For another"
                     " backend, export CMAKE_ARGS first (e.g. -DSD_VULKAN=ON).")
        return (
            [
                sys.executable, "-m", "pip", "install",
                "stable-diffusion-cpp-python>=0.1.0", "gguf>=0.1.0",
            ],
            env_overrides,
            note,
        )

    if backend == "mcp":
        return (
            [sys.executable, "-m", "pip", "install", "mcp[cli]>=1.0.0"],
            None,
            "Model Context Protocol server for OpenClaw / Claude Code / Cursor.",
        )

    raise ValueError(f"Unknown backend {backend!r}. Choose one of: mlx, gguf, mcp.")


def enable_backend(backend: str) -> int:
    """Install an optional backend's dependencies. Returns a process exit code."""
    import os

    try:
        argv, env_overrides, note = _build_enable_command(backend)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    console.print(f"[bold cyan]Enabling '{backend}' backend[/bold cyan] [dim]— {note}[/dim]")
    run_env = None
    if env_overrides:
        run_env = {**os.environ, **env_overrides}
        shown = " ".join(f"{k}={v}" for k, v in env_overrides.items())
        console.print(f"[dim]  build flags: {shown}[/dim]")

    result = subprocess.run(argv, env=run_env)
    if result.returncode == 0:
        console.print(f"[green]✓ '{backend}' backend enabled.[/green]")
    else:
        console.print(
            f"[red]✗ Failed to enable '{backend}' (pip exited {result.returncode}). "
            "See output above.[/red]"
        )
    return result.returncode


@click.command()
def verify_deps():
    """Verify and install missing dependencies"""
    console.print("\n🔍 [bold blue]Checking OllamaDiffuser Dependencies[/bold blue]")
    
    # Check critical dependencies
    deps_status = {}
    
    # OpenCV check
    try:
        import cv2
        deps_status['opencv-python'] = f"✅ Installed (v{cv2.__version__})"
    except ImportError:
        deps_status['opencv-python'] = "❌ Missing"
    
    # ControlNet Aux check
    try:
        import controlnet_aux
        deps_status['controlnet-aux'] = "✅ Installed"
    except ImportError:
        deps_status['controlnet-aux'] = "❌ Missing"
    
    # MediaPipe check (optional but recommended for full ControlNet functionality)
    try:
        import mediapipe
        deps_status['mediapipe'] = f"✅ Installed (v{mediapipe.__version__})"
    except ImportError:
        deps_status['mediapipe'] = "⚠️ Optional (recommended for face/pose ControlNet)"
    
    # Torch check
    try:
        import torch
        deps_status['torch'] = f"✅ Installed (v{torch.__version__})"
    except ImportError:
        deps_status['torch'] = "❌ Missing"
    
    # Diffusers check
    try:
        import diffusers
        deps_status['diffusers'] = f"✅ Installed (v{diffusers.__version__})"
    except ImportError:
        deps_status['diffusers'] = "❌ Missing"
    
    # Create status table
    table = Table(title="Dependency Status")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="white")
    
    missing_deps = []
    optional_deps = []
    for dep, status in deps_status.items():
        table.add_row(dep, status)
        if "❌ Missing" in status:
            missing_deps.append(dep)
        elif "⚠️ Optional" in status:
            optional_deps.append(dep)
    
    console.print(table)
    
    if missing_deps:
        console.print(f"\n⚠️  [bold yellow]{len(missing_deps)} required dependencies are missing[/bold yellow]")
        
        if click.confirm("\nWould you like to install missing dependencies?"):
            for dep in missing_deps:
                console.print(f"\n📦 Installing {dep}...")
                
                # Determine package name
                if dep == 'opencv-python':
                    package = 'opencv-python>=4.8.0'
                elif dep == 'controlnet-aux':
                    package = 'controlnet-aux>=0.0.7'
                else:
                    package = dep
                
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ])
                    console.print(f"✅ {dep} installed successfully")
                except subprocess.CalledProcessError as e:
                    console.print(f"❌ Failed to install {dep}: {e}")
        
        console.print("\n🔄 Re-run 'ollamadiffuser verify-deps' to check status")
    
    if optional_deps:
        console.print(f"\n💡 [bold blue]{len(optional_deps)} optional dependencies available for enhanced functionality[/bold blue]")
        
        if click.confirm("\nWould you like to install optional dependencies for full ControlNet support?"):
            for dep in optional_deps:
                console.print(f"\n📦 Installing {dep}...")
                
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", dep
                    ])
                    console.print(f"✅ {dep} installed successfully")
                except subprocess.CalledProcessError as e:
                    console.print(f"❌ Failed to install {dep}: {e}")
    
    if not missing_deps and not optional_deps:
        console.print("\n🎉 [bold green]All dependencies are installed![/bold green]")
    
    # Check ControlNet preprocessors
    console.print("\n🔧 [bold blue]Testing ControlNet Preprocessors[/bold blue]")
    try:
        from ..core.utils.controlnet_preprocessors import controlnet_preprocessor
        if controlnet_preprocessor.is_available():
            available_types = controlnet_preprocessor.get_available_types()
            console.print(f"✅ Available types: {', '.join(available_types)}")
        else:
            console.print("⚠️  ControlNet preprocessors not fully available")
    except Exception as e:
        console.print(f"❌ Error testing preprocessors: {e}")
    
    # Show warning suppression tip
    console.print("\n💡 [bold blue]Tip:[/bold blue] To suppress harmless import warnings, run:")
    console.print("   [cyan]export PYTHONWARNINGS=\"ignore::UserWarning,ignore::FutureWarning\"[/cyan]")

@click.command()
def doctor():
    """Run comprehensive system diagnostics"""
    console.print(Panel.fit("🩺 [bold blue]OllamaDiffuser Doctor[/bold blue]"))
    
    # System info
    import platform
    console.print(f"\n💻 System: {platform.system()} {platform.release()}")
    console.print(f"🐍 Python: {sys.version.split()[0]}")
    
    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            console.print(f"🎮 CUDA: Available ({torch.cuda.get_device_name()})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            console.print("🍎 Apple Metal: Available")
        else:
            console.print("⚙️ GPU: CPU only")
    except ImportError:
        console.print("❌ PyTorch not installed")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        console.print(f"🧠 RAM: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
    except ImportError:
        console.print("⚠️ Cannot check memory (psutil missing)")
    
    # Run dependency check
    console.print("\n" + "="*50)
    ctx = click.Context(verify_deps)
    ctx.invoke(verify_deps)

@click.command()
@click.option('--force', is_flag=True, help='Force recreation of all samples even if they exist')
def create_samples(force):
    """Create ControlNet sample images for the Web UI"""
    console.print("\n🎨 [bold blue]Creating ControlNet Sample Images[/bold blue]")
    
    try:
        from pathlib import Path
        from ..ui.web import ensure_samples_exist
        
        # Get samples directory path
        samples_dir = Path(__file__).parent.parent / "ui" / "samples"
        
        if force:
            console.print("🗑️ Removing existing samples (force mode)")
            import shutil
            if samples_dir.exists():
                shutil.rmtree(samples_dir)
        
        console.print(f"📁 Samples directory: {samples_dir}")
        
        # Create samples
        ensure_samples_exist(samples_dir)
        
        if samples_dir.exists():
            # Count created samples
            sample_count = 0
            for sample_type in ['canny', 'depth', 'openpose', 'scribble']:
                type_dir = samples_dir / sample_type
                if type_dir.exists():
                    sample_count += len(list(type_dir.glob('*.png')))
            
            console.print(f"\n✅ [bold green]Successfully created {sample_count} sample images![/bold green]")
            console.print(f"📂 Samples saved to: {samples_dir}")
            
            # Show sample types
            table = Table(title="Created Sample Types")
            table.add_column("Type", style="cyan")
            table.add_column("Count", style="white")
            table.add_column("Description", style="green")
            
            descriptions = {
                'canny': 'Edge detection control',
                'depth': 'Depth map control',
                'openpose': 'Pose estimation control',
                'scribble': 'Sketch/scribble control'
            }
            
            for sample_type in ['canny', 'depth', 'openpose', 'scribble']:
                type_dir = samples_dir / sample_type
                if type_dir.exists():
                    count = len(list(type_dir.glob('*.png')))
                    table.add_row(sample_type.title(), str(count), descriptions.get(sample_type, ''))
            
            console.print(table)
            console.print("\n💡 These samples will appear in the Web UI for easy ControlNet testing!")
        else:
            console.print("❌ [bold red]Failed to create samples directory[/bold red]")
            
    except Exception as e:
        console.print(f"❌ [bold red]Error creating samples: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]") 