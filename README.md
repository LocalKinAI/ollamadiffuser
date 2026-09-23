### Project Status: Active Development

**Thank you for the incredible support and over 30,000 downloads!**

`ollamadiffuser` is in **active development**. v2.0 brought a major architecture overhaul (strategy pattern, MCP/OpenClaw integration, Apple Silicon support, GGUF). The May 2026 line (v2.0.13 → v2.0.20) added an **MLX backend for Apple Silicon**, 7 new diffusers-pipeline models, a **compile-free default install** with a one-line `curl | sh` installer, and a **data-driven model registry** (`models.yaml`); v2.0.21 brings the MLX registry up to date with mflux — **22 MLX entries** — see [What's New](#-whats-new) below. Part of the **[LocalKinAI](https://github.com/LocalKinAI)** ecosystem.

## 🆕 What's New

### Unreleased — Qwen-Image-2.1 on Apple Silicon

`qwen-image-2.1-mlx` runs Qwen's new model through mflux 0.20.0 (now the minimum mflux). On an M3 Ultra: **108 s per 1024² image at 25 steps, 45.8 GB peak** — faster than the same prompt in ComfyUI (126–134 s), with Chinese text rendered correctly. It needs a 64 GB+ Mac, and its licence is **research / non-commercial only**.

Eight more models, all registry entries with no new code. **Through mflux:** the undistilled bases of models already here — `z-image-mlx`, `flux.2-klein-base-4b-mlx`, `ernie-image-mlx`, `krea-2-raw-mlx` — for varied output, working negative prompts and LoRA training; `flux.1-krea-dev-mlx`; and `flux.2-klein-9b-kv-edit-mlx`, a multi-reference editor that caches its references. **Through diffusers:** `qwen-image-edit-2511`, the current Qwen edit model (it can't go through mflux, which doesn't read 2511's `zero_cond_t` flag), and `realvisxl-v5`. `pull` also stops fetching weights mflux never loads: `krea-2-turbo-mlx` drops from 62 to 36 GB, `ernie-image-turbo-mlx` from 32 to 24 GB.

**LoRAs now load on Apple Silicon.** `ollamadiffuser lora load` used to refuse every MLX model; it now rebuilds the model with the LoRA baked in, for every mflux family that takes one — FLUX.1, Kontext, FLUX.2 klein, Z-Image, Qwen-Image, Krea 2, ERNIE. LoRAs stack, so a speed LoRA and a style LoRA go on together. Measured on an M3 Ultra with a klein-4B pixel-art LoRA: the edit changes visibly, and after `lora unload` the output is identical to the run before it.

**The same models on NVIDIA.** Recent additions had been Mac-only where diffusers could run them too. They now have diffusers entries: `z-image`, `flux.2-klein-base-4b`, `flux.2-klein-9b`, `ernie-image`, `ernie-image-turbo`, `krea-2-turbo`, `krea-2-raw`, `qwen-image` (2512) and `flux.1-krea-dev`. Also new: `z-anime` / `z-anime-mlx` (a full anime fine-tune of Z-Image), `seedvr2-7b-mlx`, `z-image-turbo-controlnet-mlx` (one ControlNet for canny, depth, pose, HED and MLSD), and style or motion LoRAs on LTX video.

**Licence correction:** `flux.2-klein-9b-mlx` was listed as Apache 2.0. FLUX.2 klein 9B is under the FLUX Non-Commercial License; only the 4B is Apache 2.0.

### v2.0.26 — the registry pulls what the loader loads

`qwen-image-mlx` / `qwen-image-edit-mlx` were pulling 58 GB of a checkpoint mflux never loads (now `Qwen-Image-2512` / `Qwen-Image-Edit-2509`), and `seedvr2-3b-mlx` fetched a 3.4 GB fp8 file it never opened. `ollamadiffuser enable gguf` now builds with CUDA on NVIDIA machines instead of silently producing a CPU build. New: [CONTRIBUTING.md](CONTRIBUTING.md) — adding a model is a data-only PR.

### v2.0.20 — One-line `curl | sh` installer (no system Python needed)

`curl -fsSL .../install.sh | sh` now installs a fully **isolated** OllamaDiffuser via [uv](https://github.com/astral-sh/uv): a standalone Python + the compile-free core under `~/.ollamadiffuser`, with a launcher on your PATH. It doesn't touch your system Python, needs no compiler, and auto-enables MLX on Apple Silicon — the Ollama-style zero-friction install.

### v2.0.19 — Model registry is now data (`models.yaml`)

The 59 built-in models moved from a 1500-line Python dict into a bundled [`models.yaml`](ollamadiffuser/core/config/models.yaml). Adding a model is now a **pure-data PR** — no code. Migration is zero-loss (snapshot-pinned in tests).

### v2.0.18 — Compile-free default install + `enable` command

The default `pip install ollamadiffuser` is now **compile-free** (prebuilt wheels only — no CMake, no CUDA toolchain). Optional backends are opt-in by name via a new command: `ollamadiffuser enable mlx | gguf | mcp` — no shell-quoted `[extras]` to get wrong, and `enable gguf` sets the right `CMAKE_ARGS` (Metal on Mac) for you. The `curl | sh` installer now defaults to the lean core and auto-enables MLX on Apple Silicon.

### v2.0.22 — video, natively: LTX-2 on Apple Silicon

Six new entries (`ltx-2.3-mlx-{q4,q8,bf16}`, `ltx-2.5-mlx-{q4,q8,bf16}`) drive
[ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) for text-to-video with audio,
image-to-video and audio-to-video — plus a `POST /api/generate/video` endpoint and
`engine.generate_video()`. See [the video section](#-video--ltx-2-on-apple-silicon).

### v2.0.21 — eight more MLX families: Krea 2, Boogu, ERNIE-Image, Lens, Ideogram 4, FIBO, FIBO-Edit, SeedVR2

mflux grew a lot of model families after our May line, and the registry had not
caught up. Eight new entries — **22 MLX entries total** — covering photographic
turbo models (Boogu 4-step with bilingual EN/ZH text, ERNIE-Image, Krea 2), a
4-step Microsoft model with a 20B text encoder (Lens), typography (Ideogram 4),
JSON-prompted generation and editing (FIBO, FIBO-Edit) and the best open
upscaler (SeedVR2). Each is a class and an alias: mflux resolves the config from
the alias itself, so adding a family is data plus one line of routing.

### v2.0.17 — MLX Phase 2.5: FLUX.1 family completion

`MLXStrategy` now covers the full FLUX.1 family on Apple Silicon: `flux1-fill` (inpaint/outpaint), `flux1-redux` (image variation), `flux1-depth` (depth-conditioned), `flux1-controlnet` (canny + upscaler). Five new registry entries — **14 MLX entries total** (see [the MLX section](#-mlx-models-apple-silicon-native) below).

### v2.0.15–v2.0.16 — MLX Backend (Phases 1 + 2)

New `MLXStrategy` routes FLUX.1 / FLUX.2 / Z-Image / Qwen-Image / Kontext through [mflux](https://github.com/filipstrand/mflux) for native Apple Silicon inference. **Typically 2-3× faster than the PyTorch + MPS path.** Install with `pip install 'ollamadiffuser[mlx]'`. Tracks [#7](https://github.com/LocalKinAI/ollamadiffuser/issues/7).

### v2.0.14 — Diffusers Pipeline Additions

- **`flux.1-kontext-dev`** (PyTorch) — 12B instruction-based image editing. Pass an input image + edit prompt; the model rewrites the image.
- **`chroma1-hd`** — 8.9B Apache-2.0 base T2I (FLUX-schnell derivative). Rare commercial-friendly license at this quality tier.

### v2.0.13 — Bug Fixes + Discussions

- Fixed `ollamadiffuser recommend` crash on CUDA hosts (PyTorch attribute typo).
- **GitHub Discussions** enabled: https://github.com/LocalKinAI/ollamadiffuser/discussions
- 18 broken org-name URLs fixed across PyPI metadata, README, and guides.

See [CHANGELOG.md](CHANGELOG.md) for the full history (back to v1.0.0, May 2025).

# OllamaDiffuser 🎨

[![PyPI version](https://badge.fury.io/py/ollamadiffuser.svg)](https://badge.fury.io/py/ollamadiffuser)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


## Local AI Image Generation with OllamaDiffuser

**OllamaDiffuser** simplifies local deployment of **Stable Diffusion**, **FLUX**, **CogView4**, **Kolors**, **SANA**, **PixArt-Sigma**, and 40+ other AI image generation models. An intuitive **local SD** tool inspired by **Ollama's** simplicity - perfect for **local diffuser** workflows with CLI, web UI, and LoRA support.

🌐 **Website**: [ollamadiffuser.com](https://www.ollamadiffuser.com/) | 📦 **PyPI**: [pypi.org/project/ollamadiffuser](https://pypi.org/project/ollamadiffuser/)

> **Upgrading from v1.x?** v2.0 is a major rewrite requiring **Python 3.10+**. Run `pip install --upgrade ollamadiffuser` (then `ollamadiffuser enable gguf`/`mlx` if you need those backends) and see the [Migration Guide](#-migration-guide) below.

---

## 🚀 Quick Start

### Zero-dependency install (recommended — no Python setup needed)

```bash
curl -fsSL https://raw.githubusercontent.com/LocalKinAI/ollamadiffuser/main/install.sh | sh
```

This installs an **isolated** Python + OllamaDiffuser under `~/.ollamadiffuser` (via [uv](https://github.com/astral-sh/uv) — a single static binary) and drops an `ollamadiffuser` launcher on your PATH. It **never touches your system Python**, needs no `pip`/`venv`/compiler, and on Apple Silicon auto-enables the MLX backend. Uninstall is `rm -rf ~/.ollamadiffuser`.

### Already have a Python env?

The default install is **compile-free** — prebuilt wheels only, no CMake and no CUDA toolchain. Optional backends are opt-in **by name**, so you never fight bracket-quoting in your shell.

```bash
pip install ollamadiffuser
ollamadiffuser recommend   # Find which models fit your hardware
```

**Mac / Apple Silicon (recommended — typically 2-3× faster):**
```bash
pip install ollamadiffuser
ollamadiffuser enable mlx  # native MLX backend, still compile-free
```

**OpenClaw / Agent users:**
```bash
pip install ollamadiffuser
ollamadiffuser enable mcp  # Model Context Protocol server deps
ollamadiffuser mcp         # start the server
```

**Low-VRAM / GGUF (advanced — compiles a native extension):**
```bash
pip install ollamadiffuser
ollamadiffuser enable gguf                 # sets Metal/CUDA build flags for you
ollamadiffuser pull flux.1-dev-gguf-q4ks   # only 6GB VRAM needed
ollamadiffuser run  flux.1-dev-gguf-q4ks
```

> **Why `enable` instead of `pip install "ollamadiffuser[gguf]"`?** The GGUF backend compiles a native library (`stable-diffusion-cpp-python`). `ollamadiffuser enable gguf` runs that build with the right `CMAKE_ARGS` for your platform (Metal on Mac) so you never have to remember them — and there are no shell-quoted `[extras]` to get wrong. The bracketed extras (`[gguf]`, `[mlx]`, `[mcp]`, `[full]`) still work if you prefer them.

Most models work **without any token** -- just install and go. See [Hugging Face Authentication](#-hugging-face-authentication) when you want gated models like FLUX.1-dev or SD 3.5.

---

## ✨ Features

- **🏗️ Strategy Architecture**: Clean per-model strategy pattern (SD1.5, SDXL, FLUX, SD3, ControlNet, Video, LTX-2 video (MLX), HiDream, GGUF, MLX, Generic)
- **🌐 60+ Models**: FLUX.1/2, SD 3.5, SDXL Lightning, CogView4, Kolors, SANA, PixArt-Sigma, Z-Image, Qwen-Image, Chroma1, and more
- **🔌 Generic Pipeline**: Add new diffusers models via registry config alone -- no code changes needed. Built-in models live in [`models.yaml`](ollamadiffuser/core/config/models.yaml) (data, not code) — contributing a model is a pure-data PR against that file.
- **🖼️ img2img & Inpainting**: Image-to-image and inpainting support across SD1.5, SDXL, and the API/Web UI
- **⚡ Async API**: Non-blocking FastAPI server using `asyncio.to_thread` for GPU operations
- **🎲 Random Seeds**: Reproducible generation with explicit seeds, random by default
- **🎛️ ControlNet Support**: Precise image generation control with 10+ control types (PyTorch + MLX)
- **🔄 LoRA Integration**: Dynamic LoRA loading and management
- **🔌 MCP & OpenClaw**: Model Context Protocol server for AI assistant integration (OpenClaw, Claude Code, Cursor)
- **🍎 Apple Silicon, two paths**:
  - **MLX backend** via [mflux](https://github.com/filipstrand/mflux) — 33 native MLX entries (FLUX.1 family, FLUX.1 Krea, FLUX.2 Klein, Z-Image, Qwen-Image, Kontext, Fill, Redux, Depth, ControlNet, and since v2.0.21 Krea 2, Boogu, ERNIE-Image, Lens, Ideogram 4, FIBO, FIBO-Edit, SeedVR2). Typically **2-3× faster** than the PyTorch + MPS path on M-series.
  - **PyTorch + MPS** — full diffusers pipeline support with per-model dtype handling (float16/bfloat16, NaN sanitization), GGUF Metal acceleration, and `ollamadiffuser recommend` for hardware-aware model suggestions.
- **📦 Smart Downloads**: `ollamadiffuser pull` downloads only diffusers pipeline files — skips root-level checkpoints, ONNX/Flax exports, and safety_checker. Saves 10–200 GB per model.
- **♻️ Reuses what's already downloaded**: if a model is already in the Hugging Face cache (from ComfyUI, mflux, `hf download`…), `ollamadiffuser pull` hard-links it instead of downloading — seconds instead of tens of GB, and no second copy on disk.
- **📦 GGUF Support**: Memory-efficient quantized models (3GB VRAM minimum!) with CUDA and Metal acceleration
- **🌐 Multiple Interfaces**: CLI, Python API, Web UI, and REST API
- **📦 Model Management**: Easy installation and switching between models
- **⚡ Performance Optimized**: Memory-efficient with GPU acceleration
- **🧪 Test Suite**: 243 tests across settings, registry, engine, API, MPS, MLX, LTX-2 video, and MCP

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
# Enable the GGUF backend (compiles once; sets the right build flags for you)
ollamadiffuser enable gguf

# Download a memory-efficient GGUF model (3GB+ VRAM)
ollamadiffuser pull flux.1-dev-gguf-q4ks

# Generate with reduced memory usage
ollamadiffuser run flux.1-dev-gguf-q4ks
```

### Apple Silicon Quick Start (Mac Mini / MacBook)
```bash
# See which models fit your Mac
ollamadiffuser recommend

# Fast single-step model (<6GB)
ollamadiffuser pull sdxl-turbo
ollamadiffuser run sdxl-turbo

# Native MLX path — typically 2-3x faster than PyTorch+MPS (auto-enabled
# by the curl|sh installer; run this if you installed via pip)
ollamadiffuser enable mlx
ollamadiffuser pull flux.1-schnell-mlx
ollamadiffuser run flux.1-schnell-mlx

# GGUF with Metal acceleration (6GB, great quality)
ollamadiffuser enable gguf   # sets CMAKE_ARGS=-DSD_METAL=ON under the hood
ollamadiffuser pull flux.1-dev-gguf-q4ks
ollamadiffuser run flux.1-dev-gguf-q4ks
```

### Option 2: Development Installation
```bash
# Clone the repository
git clone https://github.com/LocalKinAI/ollamadiffuser.git
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
| `z-image` | Alibaba (Tongyi) | 6B | 50 | 16GB+ | ✅ | Apache 2.0 |
| `z-anime` | SeeSee21 (Z-Image fine-tune) | 6B | 40 | 16GB+ | ✅ | Apache 2.0 |
| `flux.2-klein-base-4b` | Black Forest Labs | 4B | 50 | 10GB+ | ✅ | Apache 2.0 |
| `flux.2-klein-9b` | Black Forest Labs | 9B | 4 | 20GB+ | ❌ | Non-commercial |
| `flux.1-krea-dev` | Black Forest Labs × Krea | 12B | 28 | 20GB+ | ❌ | Non-commercial |
| `krea-2-turbo` / `krea-2-raw` | Krea | 12B | 8 / 28 | 24GB+ | ❌ | Krea 2 Community |
| `ernie-image-turbo` / `ernie-image` | Baidu | 8B | 8 / 50 | 20GB+ | ✅ | Apache 2.0 |
| `qwen-image` | Alibaba (Qwen), 2512 | 20B | 50 | 24GB+ (offload) | ✅ | Apache 2.0 |
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
| `sdxl-lightning-4step` | 4 | 6GB+ | ByteDance, single-file checkpoint, custom scheduler |
| `stable-diffusion-3.5-large-turbo` | 4 | 12GB+ | Distilled SD 3.5 Large |
| `z-image-turbo` | 8 | 10GB+ | Alibaba 6B turbo |

### Community Fine-Tunes

| Model | Base | Notes |
|-------|------|-------|
| `realvisxl-v5` | SDXL | Photorealistic, current release (pulls only the 7 GB fp16 set) |
| `realvisxl-v4` | SDXL | Photorealistic, very popular |
| `dreamshaper` | SD 1.5 | Versatile artistic model |
| `realistic-vision-v6` | SD 1.5 | Portrait specialist |

### FLUX Pipeline Variants

| Model | Pipeline | Use Case |
|-------|----------|----------|
| `flux.1-kontext-dev` | FluxKontextPipeline | **Instruction-based image editing** — pass an input image + edit prompt (added v2.0.14) |
| `flux.1-fill-dev` | FluxFillPipeline | Inpainting / outpainting |
| `flux.1-canny-dev` | FluxControlPipeline | Canny edge control |
| `flux.1-depth-dev` | FluxControlPipeline | Depth map control |
| `qwen-image-edit-2511` | QwenImageEditPlusPipeline | **Qwen's current edit model** — one to three reference images, better character and group-photo consistency than 2509. bf16, 58 GB: a 64 GB+ Mac or CPU offload on CUDA. Apache 2.0 |

### Apache-2.0 Commercial-Friendly

| Model | Origin | Params | Notes |
|-------|--------|--------|-------|
| `chroma1-hd` | lodestones | 8.9B | FLUX-schnell derivative with custom MMDiT masking + 250M timestep FFN (added v2.0.14) |
| `flux.1-schnell` | Black Forest Labs | 12B | 4-step distilled |
| `flux.2-klein-4b` | Black Forest Labs | 4B | FLUX.2 family, MPS-friendly |
| `z-image-turbo` | Alibaba (Tongyi) | 6B | 8-step DMD |
| `sana-1.5` | NVIDIA | 1.6B | Fastest >1024² generation |
| `cogview4` | Zhipu AI | 6B | Multilingual including CJK |
| `pixart-sigma` | PixArt | 0.6B | Fits 6GB GPUs |
| `lumina-2` | Alpha-VLLM | 2B | Open multimodal foundation |
| `auraflow` | Fal | 6.8B | Latest open MMDiT |
| `omnigen` | BAAI | 3.8B | Unified gen + edit |

### 💾 GGUF Models - Reduced Memory Requirements

GGUF quantized models enable running FLUX.1-dev on budget hardware:

| GGUF Variant | VRAM | Quality | Best For |
|--------------|------|---------|----------|
| `flux.1-dev-gguf-q4ks` | 6GB | ⭐⭐⭐⭐ | **Recommended** - RTX 3060/4060 |
| `flux.1-dev-gguf-q3ks` | 4GB | ⭐⭐⭐ | Mobile GPUs, GTX 1660 Ti |
| `flux.1-dev-gguf-q2k` | 3GB | ⭐⭐ | Entry-level hardware |
| `flux.1-dev-gguf-q6k` | 10GB | ⭐⭐⭐⭐⭐ | RTX 3080/4070+ |

📖 **[Complete GGUF Guide](GGUF_GUIDE.md)** - Hardware recommendations, installation, and optimization tips

### 🍎 MLX Models — Apple Silicon native

MLX entries run through [mflux](https://github.com/filipstrand/mflux) on Apple Silicon (M1/M2/M3/M4). On M-series hardware they are typically **2-3× faster** than the same model on the PyTorch + MPS path. Install with `pip install 'ollamadiffuser[mlx]'`.

**Text-to-image:**

| Entry | Family | Quant | Disk | Recommended VRAM | License |
|---|---|---|---|---|---|
| `flux.1-schnell-mlx` | FLUX.1 | Q8 | 14 GB | 16 GB (M1 32GB) | Apache 2.0 |
| `flux.1-schnell-mlx-q4` | FLUX.1 | Q4 | 8 GB | 12 GB (**M4 16GB**) | Apache 2.0 |
| `flux.1-dev-mlx` | FLUX.1 | Q8 | 14 GB | 16 GB | Non-Commercial |
| `flux.2-klein-4b-mlx` | FLUX.2 Klein | Q8 | 7 GB | 12 GB (**M4 16GB**) | Apache 2.0 |
| `flux.2-klein-9b-mlx` | FLUX.2 Klein | Q8 | 13 GB | 20 GB | Non-Commercial (gated) |
| `flux.2-klein-base-4b-mlx` | FLUX.2 Klein base (undistilled, 50-step CFG) | Q8 | 7 GB | 12 GB | Apache 2.0 |
| `flux.1-krea-dev-mlx` | FLUX.1 Krea (12B) | Q8 | 14 GB | 16 GB | Non-Commercial (gated) |
| `z-image-turbo-mlx` | Z-Image (6B, 8-step DMD) | Q8 | 8 GB | 12 GB (**M4 16GB**) | Apache 2.0 |
| `z-image-mlx` | Z-Image base (6B, 50-step CFG) | Q8 | 8 GB | 12 GB | Apache 2.0 |
| `z-anime-mlx` | Z-Anime (anime fine-tune of Z-Image base) | Q8 | 8 GB | 12 GB | Apache 2.0 |
| `qwen-image-mlx` | Qwen-Image (20B) | Q8 | 22 GB | 24 GB | Apache 2.0 |
| `qwen-image-2.1-mlx` | Qwen-Image-2.1 (mflux 0.20+) | Q8 | 34 GB | 64 GB (measured 45.8 GB peak) | Qwen Research (non-commercial) |
| `boogu-image-turbo-mlx` | Boogu Image (10B, 4-step DMD) | Q8 | 12 GB | 16 GB | Apache 2.0 |
| `ernie-image-turbo-mlx` | ERNIE-Image (8B, 8-step) | Q8 | 10 GB | 14 GB | Apache 2.0 |
| `ernie-image-mlx` | ERNIE-Image SFT (8B, 50-step CFG) | Q8 | 10 GB | 14 GB | Apache 2.0 |
| `krea-2-turbo-mlx` | Krea 2 (12B, 8-step) | Q8 | 14 GB | 20 GB | Krea 2 Community (gated) |
| `krea-2-raw-mlx` | Krea 2 Raw (12B base, for fine-tuning) | Q8 | 14 GB | 20 GB | Krea 2 Community (gated) |
| `lens-turbo-mlx` | Lens (3.8B + 20B text encoder, 4-step) | Q8 | 16 GB | 20 GB | Other |
| `ideogram-4-mlx` | Ideogram 4 (9B, preset schedules) | Q8 | 12 GB | 18 GB | Other |
| `fibo-mlx` | FIBO (8B, JSON prompts) | Q8 | 11 GB | 16 GB | Bria (gated) |

**Image editing / control:**

| Entry | Required inputs | License |
|---|---|---|
| `flux.1-kontext-dev-mlx` | `image=` | Non-Commercial |
| `flux.1-fill-dev-mlx` | `image=`, `mask_image=` | Non-Commercial |
| `flux.1-redux-dev-mlx` | `redux_images=[...]` | Non-Commercial |
| `flux.1-depth-dev-mlx` | `image=` | Non-Commercial |
| `flux.1-controlnet-canny-mlx` | `control_image=` (canny edges) | Non-Commercial |
| `flux.1-controlnet-upscaler-mlx` | `control_image=` (low-res source) | Non-Commercial |
| `qwen-image-edit-mlx` | `image=` | Apache 2.0 |
| `flux.2-klein-9b-kv-edit-mlx` | `image=` (one or more references) | Non-Commercial (gated) |
| `fibo-edit-mlx` | `image=` | Bria (gated) |
| `seedvr2-3b-mlx` | `image=` (upscales it) | Apache 2.0 |
| `seedvr2-7b-mlx` | `image=` (upscales it) | Apache 2.0 |
| `z-image-turbo-controlnet-mlx` | `control_image=`, `control_type=` (canny / depth / pose / hed / mlsd) | Apache 2.0 |

### 🎬 Video — LTX-2 on Apple Silicon

Video entries (`model_type: ltx-video-mlx`) run [LTX-2](https://github.com/Lightricks/LTX-2) through
[ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), a pure-MLX port: text-to-video **with 48 kHz stereo
audio**, image-to-video, audio-to-video, and LTX-2.5's predicted durations. This is the one strategy
that shells out rather than importing — ltx-2-mlx is a three-package monorepo installed with `uv sync`
that manages its own weight packs, and its CLI is the surface its author supports.

> **Experimental.** Both the runtime and the weight packs (`dgrauet/ltx-2.*-mlx`) are one
> author's community port, not Lightricks' upstream release, and they are young — expect the
> CLI and pack layout to move. If a pack is renamed or withdrawn these entries stop pulling;
> please [open an issue](https://github.com/LocalKinAI/ollamadiffuser/issues) if that happens.

| Entry | Pack | Disk | RAM | Default mode |
|---|---|---|---|---|
| `ltx-2.3-mlx-q4` | int4 | 12 GB | 16 GB+ | distilled, low-ram |
| `ltx-2.3-mlx-q8` | int8 | 21 GB | 32 GB+ | two-stage |
| `ltx-2.3-mlx-bf16` | bf16 | 42 GB | 64 GB+ | two-stages-hq, 720p |
| `ltx-2.5-mlx-q4` / `-q8` / `-bf16` | 2.5 packs (gated) | 13 / 22 / 44 GB | 16 / 32 / 64 GB+ | predicted duration |

```bash
# One-time: the runtime it drives
git clone https://github.com/dgrauet/ltx-2-mlx && cd ltx-2-mlx && uv sync --all-extras
# then put .venv/bin on PATH, or export LTX2MLX_BIN=/path/to/ltx-2-mlx

ollamadiffuser pull ltx-2.3-mlx-q8
ollamadiffuser run ltx-2.3-mlx-q8

# Text to video (4s at 24fps). Frame counts are 8k+1; `seconds` rounds for you.
curl -X POST http://localhost:8000/api/generate/video \
  -F prompt="a courtyard in the rain, slow pan" -F seconds=4 -o clip.mp4

# Image to video — animate a still
curl -X POST http://localhost:8000/api/generate/video \
  -F prompt="she turns and smiles" -F image=@portrait.png -F seconds=4 -o clip.mp4

# Audio to video — a voice drives the face
curl -X POST http://localhost:8000/api/generate/video \
  -F prompt="a woman speaking to camera" -F audio=@line.wav -o clip.mp4
```

Video models refuse the image endpoints rather than returning one frame of the clip, and
`/api/generate/video` answers 400 (not 500) for the things a caller can fix: no model loaded, an image
model loaded, ltx-2-mlx not installed, a frame count off the grid.

**Hardware fit at a glance:**
- **Mac Mini M4 16 GB** can run anything marked ✅ above (Q4 FLUX.1-schnell, FLUX.2 Klein 4B, Z-Image-Turbo).
- **Mac Pro M1 32 GB / M2 Pro 32 GB+** can run all entries except the 20B Qwen-Image at the larger resolutions.

Quick start:
```bash
pip install 'ollamadiffuser[mlx]'
ollamadiffuser pull z-image-turbo-mlx    # smallest Apache-2.0 option
ollamadiffuser run z-image-turbo-mlx
```

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

### LoRAs on Apple Silicon (MLX models)

The same commands work on `-mlx` models. mflux takes a LoRA when the model is built, so loading
or unloading one rebuilds the model with it baked in, and costs nothing per step afterwards.
Measured on an M3 Ultra with FLUX.2 klein 4B: 45 s to load the LoRA the first time, most of it
the 325 MB download; 0.3 s to unload it. LoRAs stack: load a speed LoRA and a style LoRA and
both apply.

```bash
ollamadiffuser lora pull Limbicnation/pixel-art-lora -w pytorch_lora_weights.safetensors -a pixel
ollamadiffuser lora load pixel --scale 1.0
```

Families that take LoRAs: FLUX.1 (and Kontext), FLUX.2 klein (and its edit variant), Z-Image,
Qwen-Image, Krea 2, ERNIE-Image. Qwen-Image-2.1, Boogu, Lens, Ideogram 4, FIBO and SeedVR2 don't
in mflux yet, and say so.

On LTX video, pass `lora=` (a local `.safetensors` or a hub repo id) and `lora_strength=` to
`/api/generate/video` for a style or motion LoRA; with a `control` video, `lora` is the IC-LoRA.

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
- **SDXL Lightning**: 4-step single-file checkpoint from ByteDance (6.5 GB download)
- **Z-Image Turbo**: 8-step turbo from Alibaba

### Community Fine-Tunes
- **RealVisXL V4**: Photorealistic SDXL, very popular
- **DreamShaper**: Versatile artistic SD 1.5 model
- **Realistic Vision V6**: Portrait specialist

### GGUF Quantized Models
- **FLUX.1-dev GGUF**: 7 quantization levels (3GB-16GB VRAM)
- **Memory Efficient**: Run high-quality models on budget hardware
- **Optional Install**: `ollamadiffuser enable gguf`

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
  -> SD15Strategy            (512x512, float16 on MPS, img2img, inpainting)
  -> SDXLStrategy            (1024x1024, float16 on MPS, diffusers force_upcast, img2img, inpainting, scheduler overrides, single-file)
  -> FluxStrategy            (schnell/dev/Fill/Canny/Depth, bfloat16 on MPS, dynamic pipeline class)
  -> SD3Strategy             (1024x1024, float16 on MPS, 28 steps, guidance=3.5)
  -> ControlNetStrategy      (SD15 + SDXL, float16 on MPS, SDXL uses diffusers force_upcast)
  -> VideoStrategy           (AnimateDiff, float16 on MPS, 16 frames)
  -> HiDreamStrategy         (bfloat16 on MPS, multi-prompt)
  -> GGUFStrategy            (quantized via stable-diffusion-cpp)
  -> GenericPipelineStrategy (any diffusers pipeline via config, per-model dtype on MPS, opt-in VAE upcast)
```

The `GenericPipelineStrategy` dynamically loads any `diffusers` pipeline class specified in the model registry, so new models can be added with zero code changes.

### Configuration
Models are automatically configured with optimal settings:
- **Memory Optimization**: Attention slicing, CPU offloading
- **Device Detection**: Automatic CUDA/MPS/CPU selection
- **Precision Handling**: FP16/BF16 per model type
- **Safety Disabled**: Unified `SAFETY_DISABLED_KWARGS` (no monkey-patching)
- **Smart Downloads**: Pipeline-only filtering by model type — skips ONNX, Flax, root checkpoints, and safety_checker

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
- **16GB unified memory**: SANA 1.5, Lumina 2.0, DreamShaper, SD 1.5, SDXL/SDXL Turbo, GGUF q2k-q5ks
- **24GB+ unified memory**: CogView4, Hunyuan-DiT, FLUX.1-schnell, GGUF q6k-q8
- **32GB unified memory**: Kolors, SD 3.5 Large, all MPS-supported models
- **GGUF with Metal**: `ollamadiffuser enable gguf` (sets `CMAKE_ARGS=-DSD_METAL=ON` for GPU acceleration automatically)
- **Note**: CPU offload does not help on Apple Silicon (unified memory) -- the full model must fit in RAM
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
The default install already includes OpenCV (it ships as a prebuilt wheel), so this is rare. If you somehow hit `ModuleNotFoundError: No module named 'cv2'`:

```bash
# Verify and repair the install
ollamadiffuser verify-deps

# Or install OpenCV directly
pip install "opencv-python>=4.8.0"
```

#### GGUF Support Issues
```bash
# Enable the GGUF backend (compiles once; sets build flags for you)
ollamadiffuser enable gguf

# Check GGUF support
ollamadiffuser registry check-gguf

# See GGUF_GUIDE.md for detailed troubleshooting
```

#### Complete Dependency Check
```bash
# Run comprehensive system diagnostics
ollamadiffuser doctor

# Verify and install missing dependencies interactively
ollamadiffuser verify-deps
```

#### Clean Installation
If you're having persistent issues, the isolated `curl | sh` installer sidesteps
any conflicts in your existing Python environment:

```bash
# Fully isolated reinstall (never touches your system Python)
rm -rf ~/.ollamadiffuser
curl -fsSL https://raw.githubusercontent.com/LocalKinAI/ollamadiffuser/main/install.sh | sh
```

Or, staying in your own environment:

```bash
pip uninstall ollamadiffuser
pip install --no-cache-dir ollamadiffuser   # compile-free core
ollamadiffuser enable gguf   # add optional backends only if you need them
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
pip install "opencv-python-headless>=4.8.0"

# GGUF with Metal acceleration — `enable gguf` sets CMAKE_ARGS=-DSD_METAL=ON for you
ollamadiffuser enable gguf
```

#### Windows
```bash
# If you encounter build errors
pip install --only-binary=all "opencv-python>=4.8.0"

# GGUF on CPU: `ollamadiffuser enable gguf`
# GGUF with CUDA acceleration (advanced — set the flag yourself):
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

- **🐛 [Report a Bug](https://github.com/LocalKinAI/ollamadiffuser/issues)** - Found an issue? Let us know
- **💡 [Feature Request](https://github.com/LocalKinAI/ollamadiffuser/issues)** - Have an idea? Share it with us  
- **💬 [Join Discussions](https://github.com/LocalKinAI/ollamadiffuser/discussions)** - Community discussion
- **⭐ [Star on GitHub](https://github.com/LocalKinAI/ollamadiffuser)** - Show your support

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

- **Issues**: [GitHub Issues](https://github.com/LocalKinAI/ollamadiffuser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LocalKinAI/ollamadiffuser/discussions)

---

**Ready to get started?** Install from PyPI: `pip install ollamadiffuser` or visit [ollamadiffuser.com](https://www.ollamadiffuser.com/) 🎨✨ 
