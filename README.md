🚀 Part of the LocalKinAI Ecosystem

ollamadiffuser is now proudly part of LocalKinAI, an open-source ecosystem for building your own local AI companion. From the creator of ollamadiffuser, discover the future of private AI. Learn more about [LocalKinAI](https://github.com/LocalKinAI)

# OllamaDiffuser 🎨

[![PyPI version](https://badge.fury.io/py/ollamadiffuser.svg)](https://badge.fury.io/py/ollamadiffuser)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


## Local AI Image Generation with OllamaDiffuser

**OllamaDiffuser** simplifies local deployment of **Stable Diffusion**, **FLUX**, **CogView4**, **Kolors**, **SANA**, **PixArt-Sigma**, and 40+ other AI image generation models. An intuitive **local SD** tool inspired by **Ollama's** simplicity - perfect for **local diffuser** workflows with CLI, web UI, and LoRA support.

🌐 **Website**: [ollamadiffuser.com](https://www.ollamadiffuser.com/) | 📦 **PyPI**: [pypi.org/project/ollamadiffuser](https://pypi.org/project/ollamadiffuser/)

> **Upgrading from v1.x?** v2.0 is a major rewrite requiring **Python 3.10+**. Run `pip install --upgrade "ollamadiffuser[full]"` and see the [Migration Guide](#-migration-guide) below.

---

## 🚀 Quick Start (v2.0)

**For Mac/PC Users:**
```bash
pip install "ollamadiffuser[full]"
ollamadiffuser recommend  # Find which models fit your GPU
```

**For OpenClaw/Agent Users:**
```bash
pip install "ollamadiffuser[mcp]"
ollamadiffuser mcp        # Starts the MCP server
```

**For Low-VRAM / Budget GPU Users:**
```bash
pip install "ollamadiffuser[gguf]"
ollamadiffuser pull flux.1-dev-gguf-q4ks  # Only 6GB VRAM needed
ollamadiffuser run flux.1-dev-gguf-q4ks
```

Most models work **without any token** -- just install and go. See [Hugging Face Authentication](#-hugging-face-authentication) when you want gated models like FLUX.1-dev or SD 3.5.

---

## ✨ Features

- **🏗️ Strategy Architecture**: Clean per-model strategy pattern (SD1.5, SDXL, FLUX, SD3, ControlNet, Video, HiDream, GGUF, Generic)
- **🌐 40+ Models**: FLUX.2, SD 3.5, SDXL Lightning, CogView4, Kolors, SANA, PixArt-Sigma, and more
- **🔌 Generic Pipeline**: Add new diffusers models via registry config alone -- no code changes needed
- **🖼️ img2img & Inpainting**: Image-to-image and inpainting support across SD1.5, SDXL, and the API/Web UI
- **⚡ Async API**: Non-blocking FastAPI server using `asyncio.to_thread` for GPU operations
- **🎲 Random Seeds**: Reproducible generation with explicit seeds, random by default
- **🎛️ ControlNet Support**: Precise image generation control with 10+ control types
- **🔄 LoRA Integration**: Dynamic LoRA loading and management
- **🔌 MCP & OpenClaw**: Model Context Protocol server for AI assistant integration (OpenClaw, Claude Code, Cursor)
- **🍎 Apple Silicon**: MPS dtype safety, GGUF Metal acceleration, `ollamadiffuser recommend` for hardware-aware model suggestions
- **📦 GGUF Support**: Memory-efficient quantized models (3GB VRAM minimum!) with CUDA and Metal acceleration
- **🌐 Multiple Interfaces**: CLI, Python API, Web UI, and REST API
- **📦 Model Management**: Easy installation and switching between models
- **⚡ Performance Optimized**: Memory-efficient with GPU acceleration
- **🧪 Test Suite**: 82 tests across settings, registry, engine, API, MPS, and MCP

### Option 1: Install from PyPI (Recommended)
```bash
# Install from PyPI
pip install ollamadiffuser

# Pull and run a model
ollamadiffuser pull flux.1-schnell
ollamadiffuser run flux.1-schnell

# Generate via API (seed is optional for reproducibility)
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset", "seed": 12345}' \
  --output image.png
```

### 🔄 Update to Latest Version

**Always use the latest version** for the newest features and bug fixes:

```bash
# Update to latest version
pip uninstall ollamadiffuser
pip install --no-cache-dir ollamadiffuser
```

This ensures you get:
- 🐛 **Latest bug fixes**
- ✨ **New features and improvements**  
- 🚀 **Performance optimizations**
- 🔒 **Security updates**

### GGUF Quick Start (Low VRAM)
```bash
# For systems with limited VRAM (3GB+)
pip install "ollamadiffuser[gguf]"

# Download memory-efficient GGUF model
ollamadiffuser pull flux.1-dev-gguf-q4ks

# Generate with reduced memory usage
ollamadiffuser run flux.1-dev-gguf-q4ks
```

### Apple Silicon Quick Start (Mac Mini / MacBook)
```bash
# See which models fit your Mac
ollamadiffuser recommend

# Best lightweight model (0.6B, <6GB)
ollamadiffuser pull pixart-sigma
ollamadiffuser run pixart-sigma

# GGUF with Metal acceleration (6GB, great quality)
pip install "ollamadiffuser[gguf]"
CMAKE_ARGS="-DSD_METAL=ON" pip install stable-diffusion-cpp-python
ollamadiffuser pull flux.1-dev-gguf-q4ks
ollamadiffuser run flux.1-dev-gguf-q4ks
```

### Option 2: Development Installation
```bash
# Clone the repository
git clone https://github.com/ollamadiffuser/ollamadiffuser.git
cd ollamadiffuser

# Install dependencies
pip install -e .
```

### Basic Usage
```bash
# Check version
ollamadiffuser -V

# Install a model
ollamadiffuser pull stable-diffusion-1.5

# Run the model (loads and starts API server)
ollamadiffuser run stable-diffusion-1.5

# Generate an image via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains"}' \
  --output image.png

# Start web interface
ollamadiffuser --mode ui

open http://localhost:8001
```

### ControlNet Quick Start
```bash
# Install ControlNet model
ollamadiffuser pull controlnet-canny-sd15

# Run ControlNet model (loads and starts API server)
ollamadiffuser run controlnet-canny-sd15

# Generate with control image
curl -X POST http://localhost:8000/api/generate/controlnet \
  -F "prompt=a beautiful landscape" \
  -F "control_image=@your_image.jpg"
```

---

## 🔑 Hugging Face Authentication

**Do you need a Hugging Face token?** It depends on which models you want to use!

**Models that DON'T require a token** -- ready to use right away:
- FLUX.1-schnell, Stable Diffusion 1.5, DreamShaper, PixArt-Sigma, SANA 1.5, most ControlNet models

**Models that DO require a token:**
- FLUX.1-dev, Stable Diffusion 3.5, some premium LoRAs

**Setup** (only needed for gated models):
```bash
# 1. Create account at https://huggingface.co and generate an access token
# 2. Accept license on the model page (e.g. FLUX.1-dev, SD 3.5)
# 3. Set your token
export HF_TOKEN=your_token_here

# 4. Now you can access gated models
ollamadiffuser pull flux.1-dev
ollamadiffuser pull stable-diffusion-3.5-medium
```

> **Tips:** Use "read" permissions for the token. Your token stays local -- never shared with OllamaDiffuser servers. Add `export HF_TOKEN=...` to `~/.bashrc` or `~/.zshrc` to make it permanent.

---

## 🎯 Supported Models

Choose from 40+ models spanning every major architecture:

### Core Models

| Model | Type | Steps | VRAM | Commercial | License |
|-------|------|-------|------|------------|---------|
| `flux.1-schnell` | flux | 4 | 16GB+ | ✅ | Apache 2.0 |
| `flux.1-dev` | flux | 20 | 20GB+ | ❌ | Non-commercial |
| `stable-diffusion-3.5-medium` | sd3 | 28 | 8GB+ | ⚠️ | Stability AI |
| `stable-diffusion-3.5-large` | sd3 | 28 | 12GB+ | ⚠️ | Stability AI |
| `stable-diffusion-3.5-large-turbo` | sd3 | 4 | 12GB+ | ⚠️ | Stability AI |
| `stable-diffusion-xl-base` | sdxl | 50 | 6GB+ | ⚠️ | CreativeML |
| `stable-diffusion-1.5` | sd15 | 50 | 4GB+ | ⚠️ | CreativeML |

### Next-Generation Models

| Model | Origin | Params | Steps | VRAM | Commercial | License |
|-------|--------|--------|-------|------|------------|---------|
| `flux.2-dev` | Black Forest Labs | 32B | 28 | 14GB+ | ❌ | Non-commercial |
| `flux.2-klein-4b` | Black Forest Labs | 4B | 28 | 10GB+ | ✅ | Apache 2.0 |
| `z-image-turbo` | Alibaba (Tongyi) | 6B | 8 | 10GB+ | ✅ | Apache 2.0 |
| `sana-1.5` | NVIDIA | 1.6B | 20 | 8GB+ | ✅ | Apache 2.0 |
| `cogview4` | Zhipu AI | 6B | 50 | 12GB+ | ✅ | Apache 2.0 |
| `kolors` | Kuaishou | 8.6B | 50 | 8GB+ | ✅ | Kolors License |
| `hunyuan-dit` | Tencent | 1.5B | 50 | 6GB+ | ✅ | Tencent Community |
| `lumina-2` | Alpha-VLLM | 2B | 30 | 8GB+ | ✅ | Apache 2.0 |
| `pixart-sigma` | PixArt | 0.6B | 20 | 6GB+ | ✅ | Open |
| `auraflow` | Fal | 6.8B | 50 | 12GB+ | ✅ | Apache 2.0 |
| `omnigen` | BAAI | 3.8B | 50 | 12GB+ | ✅ | MIT |

### Fast / Turbo Models

| Model | Steps | VRAM | Notes |
|-------|-------|------|-------|
| `sdxl-turbo` | 1 | 6GB+ | Single-step distilled SDXL |
| `sdxl-lightning-4step` | 4 | 6GB+ | ByteDance, custom scheduler |
| `stable-diffusion-3.5-large-turbo` | 4 | 12GB+ | Distilled SD 3.5 Large |
| `z-image-turbo` | 8 | 10GB+ | Alibaba 6B turbo |

### Community Fine-Tunes

| Model | Base | Notes |
|-------|------|-------|
| `realvisxl-v4` | SDXL | Photorealistic, very popular |
| `dreamshaper` | SD 1.5 | Versatile artistic model |
| `realistic-vision-v6` | SD 1.5 | Portrait specialist |

### FLUX Pipeline Variants

| Model | Pipeline | Use Case |
|-------|----------|----------|
| `flux.1-fill-dev` | FluxFillPipeline | Inpainting / outpainting |
| `flux.1-canny-dev` | FluxControlPipeline | Canny edge control |
| `flux.1-depth-dev` | FluxControlPipeline | Depth map control |

### 💾 GGUF Models - Reduced Memory Requirements

GGUF quantized models enable running FLUX.1-dev on budget hardware:

| GGUF Variant | VRAM | Quality | Best For |
|--------------|------|---------|----------|
| `flux.1-dev-gguf-q4ks` | 6GB | ⭐⭐⭐⭐ | **Recommended** - RTX 3060/4060 |
| `flux.1-dev-gguf-q3ks` | 4GB | ⭐⭐⭐ | Mobile GPUs, GTX 1660 Ti |
| `flux.1-dev-gguf-q2k` | 3GB | ⭐⭐ | Entry-level hardware |
| `flux.1-dev-gguf-q6k` | 10GB | ⭐⭐⭐⭐⭐ | RTX 3080/4070+ |

📖 **[Complete GGUF Guide](GGUF_GUIDE.md)** - Hardware recommendations, installation, and optimization tips

---

## 🎛️ ControlNet Features

### ⚡ Lazy Loading Architecture
**New in v1.1.0**: ControlNet preprocessors use intelligent lazy loading:

- **Instant Startup**: `ollamadiffuser --help` runs immediately without downloading models
- **On-Demand Loading**: Preprocessors initialize only when actually needed
- **Automatic Initialization**: Seamless loading when uploading control images
- **User Control**: Manual initialization available for pre-loading

### Available Control Types
- **Canny Edge Detection**: Structural control with edge maps
- **Depth Estimation**: 3D structure control with depth maps
- **OpenPose**: Human pose and body position control
- **Scribble/Sketch**: Artistic control with hand-drawn inputs
- **Advanced Types**: HED, MLSD, Normal, Lineart, Anime Lineart, Content Shuffle

### ControlNet Models
```bash
# SD 1.5 ControlNet Models
ollamadiffuser pull controlnet-canny-sd15
ollamadiffuser pull controlnet-depth-sd15
ollamadiffuser pull controlnet-openpose-sd15
ollamadiffuser pull controlnet-scribble-sd15

# SDXL ControlNet Models
ollamadiffuser pull controlnet-canny-sdxl
ollamadiffuser pull controlnet-depth-sdxl
```

## 🔄 LoRA Support

### Dynamic LoRA Management
```bash
# Download LoRA from Hugging Face
ollamadiffuser lora pull "openfree/flux-chatgpt-ghibli-lora"

# Load LoRA with custom strength
ollamadiffuser lora load ghibli --scale 1.2

# Unload LoRA
ollamadiffuser lora unload
```

### Web UI LoRA Integration
- **Easy Download**: Enter Hugging Face repository ID
- **Strength Control**: Adjust LoRA influence with sliders
- **Real-time Loading**: Load/unload LoRAs without restarting
- **Alias Support**: Create custom names for your LoRAs

## 🌐 Multiple Interfaces

### Command Line Interface
```bash
# Pull and run a model
ollamadiffuser pull stable-diffusion-1.5
ollamadiffuser run stable-diffusion-1.5

# Model registry management
ollamadiffuser registry list
ollamadiffuser registry list --installed-only
ollamadiffuser registry check-gguf

# Configuration management
ollamadiffuser config                                    # show all config
ollamadiffuser config set models_dir /mnt/ssd/models     # custom model path
ollamadiffuser config set server.port 9000               # change server port

# In another terminal, generate images via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic cityscape",
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024
  }' \
  --output image.png
```

### Web UI
```bash
# Start web interface
ollamadiffuser --mode ui
Open http://localhost:8001
```

Features:
- **Responsive Design**: Works on desktop and mobile
- **Real-time Status**: Model and LoRA loading indicators
- **ControlNet Integration**: File upload with preprocessing
- **Parameter Controls**: Intuitive sliders and inputs

### REST API
```bash
# Start API server
ollamadiffuser --mode api
ollamadiffuser load stable-diffusion-1.5

# Text-to-image
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape", "width": 1024, "height": 1024, "seed": 42}'

# Image-to-image
curl -X POST http://localhost:8000/api/generate/img2img \
  -F "prompt=oil painting style" \
  -F "strength=0.75" \
  -F "image=@input.png" \
  --output result.png

# Inpainting
curl -X POST http://localhost:8000/api/generate/inpaint \
  -F "prompt=a red car" \
  -F "image=@photo.png" \
  -F "mask=@mask.png" \
  --output inpainted.png

# API docs: http://localhost:8000/docs
```

### MCP Server (AI Assistant Integration)

OllamaDiffuser includes a [Model Context Protocol](https://modelcontextprotocol.io/) server for integration with AI assistants like OpenClaw, Claude Code, and Cursor.

```bash
# Install MCP support
pip install "ollamadiffuser[mcp]"

# Start MCP server (stdio transport)
ollamadiffuser mcp
```

**MCP client configuration** (e.g. `claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "ollamadiffuser": {
      "command": "ollamadiffuser-mcp"
    }
  }
}
```

**Available MCP tools:**
- `generate_image` -- Generate images from text prompts (auto-loads model)
- `list_models` -- List available and installed models
- `load_model` -- Load a model into memory
- `get_status` -- Check device, loaded model, and system status

### OpenClaw AgentSkill

An [OpenClaw](https://github.com/openclaw/openclaw) skill is included at `integrations/openclaw/SKILL.md`. It uses the REST API with `response_format=b64_json` for agent-friendly base64 image responses. Copy the skill directory to your OpenClaw skills folder or publish to ClawHub.

### Base64 JSON API Response

For AI agents and messaging platforms, use `response_format=b64_json` to get images as JSON:

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a sunset over mountains", "response_format": "b64_json"}'
```

Response: `{"image": "<base64 PNG>", "format": "png", "width": 1024, "height": 1024}`

### Python API
```python
from ollamadiffuser.core.models.manager import model_manager

# Load model
success = model_manager.load_model("stable-diffusion-1.5")
if success:
    engine = model_manager.loaded_model

    # Text-to-image (seed is optional; omit for random)
    image = engine.generate_image(
        prompt="a beautiful sunset",
        width=1024,
        height=1024,
        seed=42,
    )
    image.save("output.jpg")

    # Image-to-image
    from PIL import Image
    input_img = Image.open("photo.jpg")
    result = engine.generate_image(
        prompt="watercolor painting",
        image=input_img,
        strength=0.7,
    )
    result.save("img2img_output.jpg")
else:
    print("Failed to load model")
```

## 📦 Model Ecosystem

### Base Models
- **Stable Diffusion 1.5**: Classic, reliable, fast (img2img + inpainting)
- **Stable Diffusion XL**: High-resolution, detailed (img2img + inpainting, scheduler overrides)
- **Stable Diffusion 3.5**: Medium, Large, and Large Turbo variants
- **FLUX.1**: schnell, dev, Fill, Canny, Depth pipeline variants
- **HiDream**: Multi-prompt generation with bfloat16
- **AnimateDiff**: Video/animation generation

### Next-Generation Models
- **FLUX.2**: 32B dev and 4B Klein variants from Black Forest Labs
- **Chinese Models**: CogView4 (Zhipu), Kolors (Kuaishou), Hunyuan-DiT (Tencent), Z-Image (Alibaba)
- **Efficient Models**: SANA 1.5 (1.6B), PixArt-Sigma (0.6B) -- high quality at low VRAM
- **Open Models**: AuraFlow (6.8B, Apache 2.0), OmniGen (3.8B, MIT), Lumina 2.0 (2B, Apache 2.0)

### Fast / Turbo Models
- **SDXL Turbo**: Single-step inference from Stability AI
- **SDXL Lightning**: 4-step with custom scheduler from ByteDance
- **Z-Image Turbo**: 8-step turbo from Alibaba

### Community Fine-Tunes
- **RealVisXL V4**: Photorealistic SDXL, very popular
- **DreamShaper**: Versatile artistic SD 1.5 model
- **Realistic Vision V6**: Portrait specialist

### GGUF Quantized Models
- **FLUX.1-dev GGUF**: 7 quantization levels (3GB-16GB VRAM)
- **Memory Efficient**: Run high-quality models on budget hardware
- **Optional Install**: `pip install "ollamadiffuser[gguf]"`

### ControlNet Models
- **SD 1.5 ControlNet**: 4 control types (canny, depth, openpose, scribble)
- **SDXL ControlNet**: 2 control types (canny, depth)

### LoRA Support
- **Hugging Face Integration**: Direct download from HF Hub
- **Local LoRA Files**: Support for local .safetensors files
- **Dynamic Loading**: Load/unload without model restart
- **Strength Control**: Adjustable influence (0.1-2.0)

## ⚙️ Architecture

### Strategy Pattern Engine
Each model type has a dedicated strategy class handling loading and generation:

```
InferenceEngine (facade)
  -> SD15Strategy            (512x512, float32 on MPS, img2img, inpainting)
  -> SDXLStrategy            (1024x1024, img2img, inpainting, scheduler overrides)
  -> FluxStrategy            (schnell/dev/Fill/Canny/Depth, dynamic pipeline class)
  -> SD3Strategy             (1024x1024, 28 steps, guidance=3.5)
  -> ControlNetStrategy      (SD15 + SDXL base models)
  -> VideoStrategy           (AnimateDiff, 16 frames)
  -> HiDreamStrategy         (bfloat16, multi-prompt)
  -> GGUFStrategy            (quantized via stable-diffusion-cpp)
  -> GenericPipelineStrategy (any diffusers pipeline via config)
```

The `GenericPipelineStrategy` dynamically loads any `diffusers` pipeline class specified in the model registry, so new models can be added with zero code changes.

### Configuration
Models are automatically configured with optimal settings:
- **Memory Optimization**: Attention slicing, CPU offloading
- **Device Detection**: Automatic CUDA/MPS/CPU selection
- **Precision Handling**: FP16/BF16 per model type
- **Safety Disabled**: Unified `SAFETY_DISABLED_KWARGS` (no monkey-patching)

## 🔧 Advanced Usage

### ControlNet Parameters
```python
# Fine-tune ControlNet behavior
image = engine.generate_image(
    prompt="architectural masterpiece",
    control_image=control_img,
    controlnet_conditioning_scale=1.2,  # Strength (0.0-2.0)
    control_guidance_start=0.0,         # When to start (0.0-1.0)
    control_guidance_end=1.0            # When to end (0.0-1.0)
)
```

### GGUF Model Usage
```bash
# Check GGUF support
ollamadiffuser registry check-gguf

# Download GGUF model for your hardware
ollamadiffuser pull flux.1-dev-gguf-q4ks  # 6GB VRAM
ollamadiffuser pull flux.1-dev-gguf-q3ks  # 4GB VRAM

# Use with optimized settings
ollamadiffuser run flux.1-dev-gguf-q4ks
```

### Batch Processing
```python
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor

# Pre-initialize for faster processing
controlnet_preprocessor.initialize()

# Process multiple images
prompt = "beautiful landscape"  # Define the prompt
for i, image_path in enumerate(image_list):
    control_img = controlnet_preprocessor.preprocess(image_path, "canny")
    result = engine.generate_image(prompt, control_image=control_img)
    result.save(f"output_{i}.jpg")
```

### API Integration
```python
import requests

# Initialize ControlNet preprocessors
response = requests.post("http://localhost:8000/api/controlnet/initialize")

# Check available preprocessors
response = requests.get("http://localhost:8000/api/controlnet/preprocessors")
print(response.json()["available_types"])

# Generate with file upload
with open("control.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/generate/controlnet",
        data={"prompt": "beautiful landscape"},
        files={"control_image": f}
    )
```

## 📚 Documentation & Guides

- **[GGUF Models Guide](GGUF_GUIDE.md)**: Complete guide to memory-efficient GGUF models
- **[ControlNet Guide](CONTROLNET_GUIDE.md)**: Comprehensive ControlNet usage and examples
- **[Website Documentation](https://www.ollamadiffuser.com/)**: Complete tutorials and guides

## 🚀 Performance & Hardware

### Minimum Requirements
- **RAM**: 8GB system RAM
- **Storage**: 10GB free space
- **Python**: 3.10+

### Recommended Hardware

#### For Regular Models
- **GPU**: 8GB+ VRAM (NVIDIA/AMD)
- **RAM**: 16GB+ system RAM
- **Storage**: SSD with 50GB+ free space

#### For Apple Silicon (Mac Mini / MacBook)
- **16GB unified memory**: PixArt-Sigma, SANA 1.5, DreamShaper, SD 1.5/XL, GGUF q2k-q5ks
- **24GB+ unified memory**: CogView4, Kolors, Lumina 2.0, GGUF q6k-q8
- **GGUF with Metal**: Install with `CMAKE_ARGS="-DSD_METAL=ON"` for GPU acceleration
- Run `ollamadiffuser recommend` to see what fits your hardware

#### For GGUF Models (Memory Efficient)
- **GPU**: 3GB+ VRAM (or CPU only)
- **RAM**: 8GB+ system RAM (16GB+ for CPU inference)
- **Storage**: SSD with 20GB+ free space

### Supported Platforms
- **CUDA**: NVIDIA GPUs (recommended)
- **MPS**: Apple Silicon (M1/M2/M3/M4) -- native support for 30+ models including GGUF
- **CPU**: All platforms (slower but functional)

## 🔧 Troubleshooting

### Installation Issues

#### Missing Dependencies (cv2/OpenCV Error)
If you encounter `ModuleNotFoundError: No module named 'cv2'`, run:

```bash
# Quick fix
pip install opencv-python>=4.8.0

# Or use the built-in verification tool
ollamadiffuser verify-deps

# Or install with all optional dependencies
# For bash/sh:
pip install ollamadiffuser[full]

# For zsh (macOS default):
pip install "ollamadiffuser[full]"

# For fish shell:
pip install 'ollamadiffuser[full]'
```

#### GGUF Support Issues
```bash
# Install GGUF dependencies
pip install "ollamadiffuser[gguf]"

# Check GGUF support
ollamadiffuser registry check-gguf

# See full GGUF troubleshooting guide
# Read GGUF_GUIDE.md for detailed troubleshooting
```

#### Complete Dependency Check
```bash
# Run comprehensive system diagnostics
ollamadiffuser doctor

# Verify and install missing dependencies interactively
ollamadiffuser verify-deps
```

#### Clean Installation
If you're having persistent issues:

```bash
# Uninstall and reinstall
pip uninstall ollamadiffuser

# Reinstall with all dependencies (shell-specific syntax):
# For bash/sh:
pip install --no-cache-dir ollamadiffuser[full]

# For zsh (macOS default):
pip install --no-cache-dir "ollamadiffuser[full]"

# For fish shell:
pip install --no-cache-dir 'ollamadiffuser[full]'

# Verify installation
ollamadiffuser verify-deps
```

### Common Issues

#### Slow Startup
If you experience slow startup, ensure you're using the latest version with lazy loading:
```bash
git pull origin main
pip install -e .
```

#### ControlNet Not Working
```bash
# Check preprocessor status
python -c "
from ollamadiffuser.core.utils.controlnet_preprocessors import controlnet_preprocessor
print('Available:', controlnet_preprocessor.is_available())
print('Initialized:', controlnet_preprocessor.is_initialized())
"

# Manual initialization
curl -X POST http://localhost:8000/api/controlnet/initialize
```

#### Memory Issues
```bash
# Use GGUF models for lower memory usage
ollamadiffuser pull flux.1-dev-gguf-q4ks  # 6GB VRAM
ollamadiffuser pull flux.1-dev-gguf-q3ks  # 4GB VRAM

# Use smaller image sizes via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "width": 512, "height": 512}' \
  --output test.png

# CPU offloading is automatic
# Close other applications to free memory
# Use basic preprocessors instead of advanced ones
```

### Platform-Specific Issues

#### macOS Apple Silicon
```bash
# If you encounter OpenCV issues on Apple Silicon
pip uninstall opencv-python
pip install opencv-python-headless>=4.8.0

# For GGUF Metal acceleration
CMAKE_ARGS="-DSD_METAL=ON" pip install stable-diffusion-cpp-python
```

#### Windows
```bash
# If you encounter build errors
pip install --only-binary=all opencv-python>=4.8.0

# For GGUF CUDA acceleration
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
```

#### Linux
```bash
# If you need system dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0
pip install opencv-python>=4.8.0
```

### Debug Mode
```bash
# Enable verbose logging
ollamadiffuser --verbose run model-name
```

## 🤝 Contributing

We welcome contributions! Please check the GitHub repository for contribution guidelines.

## 🤝 Community & Support

### Quick Actions

- **🐛 [Report a Bug](https://github.com/ollamadiffuser/ollamadiffuser/issues)** - Found an issue? Let us know
- **💡 [Feature Request](https://github.com/ollamadiffuser/ollamadiffuser/issues)** - Have an idea? Share it with us  
- **💬 [Join Discussions](https://github.com/ollamadiffuser/ollamadiffuser/discussions)** - Community discussion
- **⭐ [Star on GitHub](https://github.com/ollamadiffuser/ollamadiffuser)** - Show your support

### Community Driven

OllamaDiffuser is an open-source project that thrives on community feedback. Every suggestion, bug report, and contribution helps make it better for everyone.

**Open Source** • **Community Driven** • **Actively Maintained**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI**: For Stable Diffusion models
- **Black Forest Labs**: For FLUX.1 and FLUX.2 models
- **Alibaba (Tongyi-MAI)**: For Z-Image Turbo
- **NVIDIA (Efficient-Large-Model)**: For SANA 1.5
- **Zhipu AI (THUDM)**: For CogView4
- **Kuaishou (Kwai-Kolors)**: For Kolors
- **Tencent (Hunyuan)**: For Hunyuan-DiT
- **Alpha-VLLM**: For Lumina 2.0
- **PixArt-alpha**: For PixArt-Sigma
- **Fal**: For AuraFlow
- **BAAI (Shitao)**: For OmniGen
- **ByteDance**: For SDXL Lightning
- **city96**: For FLUX.1-dev GGUF quantizations
- **Hugging Face**: For model hosting and diffusers library
- **Anthropic**: For Model Context Protocol (MCP)
- **OpenClaw**: For AI agent ecosystem integration
- **ControlNet Team**: For ControlNet architecture
- **Community**: For feedback and contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ollamadiffuser/ollamadiffuser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ollamadiffuser/ollamadiffuser/discussions)

---

**Ready to get started?** Install from PyPI: `pip install ollamadiffuser` or visit [ollamadiffuser.com](https://www.ollamadiffuser.com/) 🎨✨ 
