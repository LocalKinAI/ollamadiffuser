# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-02-04

### Fixed
- **Registry command visibility**: Made `ollamadiffuser registry` command visible in CLI help output (was previously hidden)

### Changed
- **README**: Added registry commands to CLI documentation

---

## [2.0.0] - 2026-02-03

### 🏗️ Architecture Overhaul

#### Strategy Pattern Engine
- **Refactored InferenceEngine**: Replaced 1400+ line monolithic class with a ~220 line facade delegating to per-model strategy classes
- **8 Strategy Classes**: `SD15Strategy`, `SDXLStrategy`, `FluxStrategy`, `SD3Strategy`, `ControlNetStrategy`, `VideoStrategy`, `HiDreamStrategy`, `GGUFStrategy`
- **Abstract Base Class**: `InferenceStrategy` in `core/inference/base.py` with shared loading, LoRA, seed, and error handling logic
- **Unified Safety Checker**: Replaced 5+ monkey-patch approaches with a single `SAFETY_DISABLED_KWARGS` dict passed to `from_pretrained`

#### CLI Modularization
- **Split CLI**: Broke 1300+ line `cli/main.py` into 4 focused modules (~139 line router)
- **model_commands.py**: pull, run, list, show, check, rm, ps, load, unload, serve, stop
- **lora_commands.py**: LoRA pull, load, unload, rm, ps, list, show
- **registry_commands.py**: Registry list, add, remove, reload, import-config, export, check-gguf

#### Duplicate Code Removal
- **Removed `core/models/registry.py`**: Eliminated duplicate model registry (kept `core/config/model_registry.py`)
- **Removed dead files**: `constants.py`, `helpers.py` from old engine architecture

### 🚀 New Features

#### Image-to-Image Generation
- **img2img API endpoint**: `POST /api/generate/img2img` with strength control
- **Inpainting API endpoint**: `POST /api/generate/inpaint` with mask support
- **SD1.5 img2img/inpainting**: Via `StableDiffusionImg2ImgPipeline` and `StableDiffusionInpaintPipeline`
- **SDXL img2img/inpainting**: Via `StableDiffusionXLImg2ImgPipeline` and `StableDiffusionXLInpaintPipeline`
- **Web UI img2img**: New collapsible img2img section with input image preview

#### Seed Support
- **Random seeds by default**: Replaced hardcoded `seed=42` with `random.randint(0, 2**32 - 1)`
- **Reproducible generation**: Pass explicit `seed` parameter to any generation endpoint
- **Web UI seed field**: Optional seed input with "Random" placeholder
- **API seed parameter**: `seed` field in `GenerateRequest` model

#### Async API
- **Non-blocking generation**: All GPU-bound operations wrapped in `asyncio.to_thread()`
- **Async model loading**: `/api/models/load` and `/api/models/pull` are non-blocking
- **Web UI async**: Generate and load operations in thread pool

### 🌐 21 New Models

#### GenericPipelineStrategy
- **New strategy class**: Dynamically loads any `diffusers` pipeline class via `model_config.parameters["pipeline_class"]`, enabling new model types without writing new strategy code

#### Tier 1: Classic Models (existing strategies)
- **Stable Diffusion 3.5 Large**: `stabilityai/stable-diffusion-3.5-large` (sd3, 28 steps)
- **Stable Diffusion 3.5 Large Turbo**: `stabilityai/stable-diffusion-3.5-large-turbo` (sd3, 4 steps, guidance=0.0)
- **RealVisXL V4**: `SG161222/RealVisXL_V4.0` (sdxl, photorealistic finetune)
- **DreamShaper**: `Lykon/DreamShaper` (sd15, popular community model)
- **Realistic Vision V6**: `SG161222/Realistic_Vision_V6.0_B1_noVAE` (sd15, portrait specialist)
- **SDXL Turbo**: `stabilityai/sdxl-turbo` (sdxl, single-step inference)

#### Tier 2: Scheduler Override
- **SDXL Lightning 4-Step**: `ByteDance/SDXL-Lightning` (sdxl, EulerDiscreteScheduler with trailing timestep spacing)
- **SDXLStrategy scheduler override**: Reads `scheduler_class` and `scheduler_kwargs` from model parameters to configure custom schedulers

#### Tier 3: FLUX Pipeline Variants
- **FLUX.1 Fill**: `black-forest-labs/FLUX.1-Fill-dev` (FluxFillPipeline, inpainting/outpainting)
- **FLUX.1 Canny**: `black-forest-labs/FLUX.1-Canny-dev` (FluxControlPipeline, edge control)
- **FLUX.1 Depth**: `black-forest-labs/FLUX.1-Depth-dev` (FluxControlPipeline, depth control)
- **FluxStrategy dynamic pipeline**: Reads `pipeline_class` from parameters, defaults to `FluxPipeline` for backward compatibility

#### Tier 4: Next-Generation Models (GenericPipelineStrategy)
- **FLUX.2 Dev**: `black-forest-labs/FLUX.2-dev` (32B params, Flux2Pipeline)
- **FLUX.2 Klein 4B**: `black-forest-labs/FLUX.2-klein-4B` (4B params, Apache 2.0, Flux2KleinPipeline)
- **Z-Image Turbo**: `Tongyi-MAI/Z-Image-Turbo` (Alibaba 6B, 8-step turbo, bilingual CN/EN)
- **SANA 1.5**: `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` (NVIDIA 1.6B, very efficient)
- **CogView4**: `THUDM/CogView4-6B` (Zhipu AI 6B, GLM-4 encoder, bilingual CN/EN)
- **Kolors**: `Kwai-Kolors/Kolors-diffusers` (Kuaishou 8.6B, ChatGLM3, bilingual CN/EN)
- **Hunyuan-DiT**: `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` (Tencent 1.5B, dual text encoders)
- **Lumina 2.0**: `Alpha-VLLM/Lumina-Image-2.0` (2B, unified text+image tokens)
- **PixArt-Sigma**: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (0.6B, <8GB VRAM, 4K capable)
- **AuraFlow**: `fal/AuraFlow-v0.3` (6.8B, largest Apache 2.0 text-to-image model)
- **OmniGen**: `Shitao/OmniGen-v1-diffusers` (BAAI 3.8B, unified generation, no negative prompt)

### 🔌 OpenClaw & MCP Integration
- **MCP Server**: New `ollamadiffuser mcp` command (or `ollamadiffuser-mcp` entry point) starts a Model Context Protocol server with 4 tools: `generate_image`, `list_models`, `load_model`, `get_status` -- works with OpenClaw, Claude Code, Cursor, and any MCP client
- **OpenClaw AgentSkill**: `integrations/openclaw/SKILL.md` provides a ready-to-use skill for OpenClaw with full REST API instructions for model management and image generation
- **Base64 JSON response**: New `response_format=b64_json` option on `POST /api/generate` returns `{"image": "<base64>", "format": "png", "width": w, "height": h}` -- essential for AI agent and messaging platform integration
- **Optional dependency**: `pip install "ollamadiffuser[mcp]"` or `pip install "ollamadiffuser[openclaw]"` adds MCP support; included in `[full]`

### 🍎 Apple Silicon / Mac Mini Support
- **`ollamadiffuser recommend` command**: New CLI command that detects hardware (CUDA/MPS/CPU, RAM/VRAM) and recommends models that fit, with `--commercial-only` and `--device` flags
- **MPS dtype safety**: GenericPipelineStrategy now falls back from bfloat16 to float16 on MPS devices, preventing Metal compatibility issues
- **GGUF Metal acceleration**: All 11 GGUF models now list MPS in supported_devices (Metal acceleration via `CMAKE_ARGS="-DSD_METAL=ON"`)
- **Registry MPS updates**: Added MPS support to `cogview4` and `lumina-2` model entries
- **Smart quick-start**: `recommend` command suggests the best standalone model for your hardware (e.g. `pixart-sigma` for 16GB Mac)

### 🔒 Security Fixes
- **Path traversal prevention**: Removed `control_image_path` from `/api/generate` endpoint (use dedicated `/api/generate/controlnet` with file upload instead)
- **CORS hardening**: Dropped `allow_credentials=True` from CORS middleware
- **Error detail leakage**: Replaced raw exception messages in HTTP responses with generic error messages, log full details server-side with `exc_info=True`

### 🐛 Bug Fixes (v2.0.0)
- **HuggingFace Hub path check**: Fixed `Path.exists()` returning False for Hub IDs like `"black-forest-labs/FLUX.1-schnell"` by only checking local paths
- **Device placement conflict**: Fixed `_move_to_device()` and `enable_model_cpu_offload()` contradicting each other in FluxStrategy, HiDreamStrategy, and VideoStrategy (now mutually exclusive)
- **Falsy zero evaluation**: Fixed `num_inference_steps=0` being treated as falsy across all 8 strategies (now uses `is not None` check)
- **SD1.5 silent guidance clamping**: Removed silent `guidance > 7.0` clamp; kept MPS clamp with logging
- **SD1.5 silent dimension override**: Replaced silent 1024->512 dimension override with logged warning for dimensions > 768
- **VideoStrategy hardcoded dimensions**: Fixed hardcoded `width=512, height=512` to use user-specified values
- **Error image truncation**: Added `prompt[:50] + "..."` truncation in `_create_error_image` for long prompts
- **GGUF duplicate seed logic**: Moved `import random` to module level
- **Web UI LoRA download**: Fixed `final_name` reference error in `pull_lora_ui` by initializing before try block
- **Hardcoded seed=42**: Now random by default across all strategies
- **Safety checker monkey-patches**: Unified into clean `SAFETY_DISABLED_KWARGS`
- **Missing CLI commands**: Added `serve` and `stop` to `model_commands.py`
- **Sync API blocking**: All endpoints now properly async

### 🧪 Test Suite
- **82 tests across 7 modules**: All passing
- **test_settings.py**: ModelConfig, ServerConfig, Settings save/load
- **test_model_registry.py**: Default models, get/add/remove, model names, 21 new model validation (types, pipeline classes, scheduler config, hardware requirements, MPS size constraints)
- **test_engine.py**: Strategy factory (all 9 types including generic), device detection, engine lifecycle, GenericPipelineStrategy (dynamic pipeline loading, missing/invalid pipeline class, CPU offload), base strategy methods, safety kwargs
- **test_api_server.py**: Health, root, models, generate endpoints via TestClient
- **test_mps_support.py**: GenericPipelineStrategy MPS dtype fallback (6 tests), registry MPS device assignments (2 tests), recommend command classification and CLI (7 tests)
- **test_api_base64.py**: Base64 JSON response format (default PNG, b64_json, null format, no-model error)
- **test_mcp_server.py**: MCP server creation, tool registration, list_models, get_status, load_model, generate_image error handling (8 tests, skip gracefully without mcp package)

### 📦 Dependencies Updated
- **torch**: `>=2.4.0` (was `>=2.0.0`)
- **diffusers**: `>=0.34.0` (was `>=0.26.0`)
- **accelerate**: `>=1.0.0` (was `>=0.20.0`)
- **transformers**: `>=4.44.0` (was `>=4.30.0`)
- **numpy**: `>=1.26.0` (was `>=1.24.0`)
- **Pillow**: `>=10.0.0` (was `>=9.5.0`)
- **GGUF optional**: Moved `stable-diffusion-cpp-python` and `gguf` to `[gguf]` extra
- **Dev dependencies**: Added `pytest`, `pytest-cov`, `pytest-asyncio`, `httpx`, `mypy`

### 🌐 Web UI Updates
- **Version display**: Header shows v2.0.0
- **Seed field**: Optional seed input for reproducible results
- **img2img tab**: Collapsible section with strength, steps, guidance, and seed controls
- **img2img results**: Side-by-side input/output image display
- **Async operations**: Model loading and generation run in thread pool

### 🐛 Bug Fixes
- **Hardcoded seed=42**: Now random by default across all strategies
- **Safety checker monkey-patches**: Unified into clean `SAFETY_DISABLED_KWARGS`
- **Missing CLI commands**: Added `serve` and `stop` to `model_commands.py`
- **Sync API blocking**: All endpoints now properly async

### ⚠️ Breaking Changes
- **Minimum Python**: 3.10+ (was 3.8+)
- **Dependency versions**: Major bumps to torch, diffusers, accelerate
- **GGUF install**: Now requires `pip install "ollamadiffuser[gguf]"` instead of separate packages
- **Internal API**: `InferenceEngine` internals restructured (public API unchanged)

### 🔄 Migration Guide
For users upgrading from v1.x:

1. **Update dependencies**: `pip install --upgrade ollamadiffuser`
2. **GGUF users**: `pip install "ollamadiffuser[gguf]"` (separate install no longer works)
3. **API consumers**: No changes needed -- all endpoints remain compatible
4. **Python API users**: `engine.generate_image()` API unchanged, now supports `seed`, `image`, `mask_image`, and `strength` parameters

---

## [1.2.0] - 2025-06-02

### 🚀 Major Features Added

#### ⚡ GGUF Model Support
- **Quantized Models**: Full support for GGUF (GPT-Generated Unified Format) quantized models
- **Massive VRAM Reduction**: Run FLUX.1-dev with 3GB VRAM instead of 20GB+ 
- **7 Quantization Levels**: From q2k (3GB) to f16 (16GB) for different hardware capabilities
- **Hardware Optimization**: Native CUDA and Metal acceleration support
- **CPU Fallback**: Automatic CPU inference when VRAM is insufficient

#### 🎛️ GGUF Model Variants
- **flux.1-dev-gguf-q2k**: Ultra-low VRAM (3GB) for testing and low-end hardware
- **flux.1-dev-gguf-q3ks**: Balanced option (4GB) for mobile GPUs
- **flux.1-dev-gguf-q4ks**: **Recommended** (6GB) - best quality/performance balance
- **flux.1-dev-gguf-q5ks**: High quality (8GB) for mid-range GPUs
- **flux.1-dev-gguf-q6k**: Near-original quality (10GB) 
- **flux.1-dev-gguf-q8**: Minimal quality loss (12GB)
- **flux.1-dev-gguf-f16**: Full precision (16GB)

### 🛠️ Technical Implementation

#### GGUF Engine Integration
- **Backend**: stable-diffusion.cpp with Python bindings integration
- **Automatic Detection**: Seamless GGUF model recognition and loading
- **Memory Management**: Intelligent VRAM usage and CPU offloading
- **Hardware Acceleration**: CMAKE-based CUDA and Metal compilation support

#### CLI Enhancements
- **GGUF Check**: `ollamadiffuser registry check-gguf` command for compatibility verification
- **Model Pull**: Seamless GGUF model downloading with progress tracking
- **Status Monitoring**: Real-time GGUF support and model status checking

### 🎯 Performance Optimizations

#### Generation Parameters
- **Optimized Settings**: 4-step generation (FLUX-optimized)
- **CFG Scale**: guidance_scale=1.0 for best FLUX results
- **Euler Sampler**: Recommended sampler for GGUF models
- **Hardware Adaptation**: Automatic parameter adjustment based on available VRAM

#### Memory Efficiency
- **Smart Loading**: Load only required model components
- **Progressive Quantization**: Automatic fallback to lower quantization when needed
- **Resource Management**: Intelligent GPU memory allocation and cleanup

### 📚 Documentation & Guides

#### Comprehensive GGUF Guide
- **GGUF_GUIDE.md**: Complete 160+ line guide with installation, usage, and troubleshooting
- **Hardware Recommendations**: Specific guidance for different GPU tiers
- **Performance Comparisons**: Quality vs speed vs VRAM usage tables
- **Troubleshooting**: Common issues and solutions for GGUF models

#### Usage Examples
- **CLI Workflows**: Step-by-step GGUF model usage examples
- **Python API**: Code examples for programmatic GGUF model usage
- **Web UI Integration**: Browser-based GGUF model selection and generation

### 🔧 Dependencies & Requirements

#### New Dependencies
- **stable-diffusion-cpp-python**: Core GGUF inference engine
- **gguf**: Model format handling and validation
- **Enhanced OpenCV**: Updated to >=4.8.0 for improved compatibility

#### Hardware Support
- **NVIDIA CUDA**: CMAKE_ARGS="-DSD_CUDA=ON" installation
- **Apple Metal**: CMAKE_ARGS="-DSD_METAL=ON" for M1/M2 Macs
- **CPU Inference**: Full CPU fallback support for any modern processor

### 🎨 User Experience Improvements

#### Accessibility
- **Low-End Hardware**: Enable FLUX.1-dev on 3GB GPUs (previously impossible)
- **Faster Downloads**: Reduced model sizes from ~24GB to 3-16GB
- **Quick Testing**: Instant model switching between quantization levels

#### Web UI Enhancements
- **GGUF Model Selection**: Dropdown menu with GGUF model variants
- **VRAM Monitoring**: Real-time memory usage display
- **Quality Preview**: Visual quality indicators for each quantization level

### 🐛 Bug Fixes & Improvements
- **Memory Leaks**: Improved GGUF model cleanup and resource management
- **Error Handling**: Better error messages for GGUF-specific issues
- **Compatibility**: Enhanced hardware detection and fallback mechanisms

### ⚠️ Breaking Changes
- **Dependency Requirements**: New GGUF dependencies required for full functionality
- **Model Loading**: GGUF models use different loading mechanisms than regular models

### 🔄 Migration Guide
For users upgrading to v1.2.0:

1. **Install GGUF Dependencies**: `pip install stable-diffusion-cpp-python gguf`
2. **Check Compatibility**: `ollamadiffuser registry check-gguf`
3. **Download GGUF Model**: `ollamadiffuser pull flux.1-dev-gguf-q4ks`
4. **Update Hardware Acceleration**: Reinstall with CUDA/Metal support if needed

### 📊 Performance Metrics
- **VRAM Reduction**: Up to 85% reduction (20GB → 3GB)
- **File Size**: Up to 87% smaller downloads (24GB → 3GB)
- **Generation Speed**: Comparable or faster due to optimized quantization
- **Quality Retention**: 90%+ quality retention with q4ks quantization

## [1.1.6] - 2025-5-30

### 🎨 New Features

#### ControlNet Sample Images
- **New CLI Command**: `ollamadiffuser create-samples` for creating ControlNet demonstration images
- **Built-in Samples**: Pre-made control images for Canny, Depth, OpenPose, and Scribble controls
- **Web UI Integration**: Sample images automatically available in the web interface for easy testing
- **Force Recreation**: `--force` flag to recreate all samples even if they exist

#### Installation Helper
- **New Script**: `install_helper.py` for platform-specific installation guidance
- **Shell Detection**: Automatically detects user's shell (bash, zsh, fish) and provides correct install syntax
- **Multiple Installation Options**: Basic, Full, and Development installation commands
- **Interactive Installation**: Option to install directly from the helper script

### 🛠️ Improvements

#### CLI Enhancements
- **Progress Tracking**: Enhanced download progress display with Ollama-style formatting
- **Better Error Handling**: Improved error messages and graceful failure modes
- **Warning Fixes**: Resolved various CLI warnings and edge cases

#### Web UI Enhancements
- **Sample Image Gallery**: Built-in ControlNet samples with 3 images per control type
- **Automatic Sample Creation**: Samples generated automatically when needed
- **Better UX**: Visual samples make ControlNet testing more intuitive

### 🐛 Bug Fixes
- **Version Inconsistencies**: Fixed version numbering across different components
- **Installation Issues**: Resolved shell-specific installation syntax problems
- **CLI Warnings**: Fixed various warning messages and edge cases
- **Sample Generation**: Improved reliability of sample image creation

### 📦 Technical Changes
- **MANIFEST.in**: Updated to include sample images and static files
- **Dependencies**: Refined dependency management for better compatibility
- **Shell Compatibility**: Better support for zsh, fish, and bash shells

### 🎯 Sample Images Created
- **Canny Control**: Geometric shapes, house outline, portrait silhouette (3 samples)
- **Depth Control**: Depth map variations for different scene types (3 samples)
- **OpenPose Control**: Human pose variations for different positions (3 samples)
- **Scribble Control**: Hand-drawn style sketches and outlines (3 samples)

## [1.1.0] - 2025-5-29

### 🚀 Major Features Added

#### ⚡ Lazy Loading Architecture
- **Instant Startup**: Application now starts immediately without downloading ControlNet models
- **On-Demand Loading**: ControlNet preprocessors initialize only when actually needed
- **Performance Boost**: `ollamadiffuser --help` runs in milliseconds instead of 30+ seconds
- **Memory Efficient**: No unnecessary model downloads for users who don't use ControlNet

#### 🎛️ Complete ControlNet Integration
- **6 ControlNet Models**: SD 1.5 and SDXL variants (canny, depth, openpose, scribble)
- **10 Control Types**: canny, depth, openpose, hed, mlsd, normal, lineart, lineart_anime, shuffle, scribble
- **Advanced Preprocessors**: Full controlnet-aux integration with graceful fallbacks
- **Web UI Integration**: File upload, preprocessing, and side-by-side result display
- **REST API Support**: Complete API endpoints for ControlNet generation and preprocessing

#### 🔄 Enhanced LoRA Management
- **Web UI Integration**: Download LoRAs directly from Hugging Face in the browser
- **Alias Support**: Create custom names for your LoRAs
- **Strength Control**: Adjust LoRA influence with intuitive sliders
- **Real-time Loading**: Load/unload LoRAs without restarting the application

### 🛠️ Technical Improvements

#### ControlNet Preprocessor Manager
- **Lazy Initialization**: `ControlNetPreprocessorManager` with `is_initialized()`, `is_available()`, `initialize()` methods
- **Automatic Fallback**: Basic OpenCV processors when advanced ones fail
- **Error Handling**: Robust validation and graceful degradation
- **Status Tracking**: Real-time initialization and availability status

#### Web UI Enhancements
- **ControlNet Section**: Dedicated controls with status indicators
- **Initialization Button**: Manual preprocessor initialization for faster processing
- **File Upload**: Drag-and-drop control image upload with validation
- **Responsive Design**: Mobile-friendly interface with adaptive layouts
- **Real-time Status**: Live model, LoRA, and ControlNet status indicators

#### API Improvements
- **New Endpoints**: `/api/controlnet/initialize`, `/api/controlnet/preprocessors`, `/api/controlnet/preprocess`
- **File Upload Support**: Multipart form data handling for control images
- **Status Endpoints**: Check ControlNet availability and initialization status
- **Error Handling**: Comprehensive error responses with helpful messages

### 📦 Dependencies Updated
- **controlnet-aux**: Added `>=0.0.7` for advanced preprocessing capabilities
- **opencv-python**: Added `>=4.8.0` for basic image processing fallbacks
- **diffusers**: Updated to `>=0.26.0` for ControlNet compatibility

### 🎨 User Experience Improvements

#### Startup Performance
- **Before**: 30+ seconds startup time, 1GB+ automatic downloads
- **After**: Instant startup, downloads only when needed
- **User Control**: Choose when to initialize ControlNet preprocessors

#### Web UI Experience
- **Status Indicators**: Clear visual feedback for all system states
- **Progressive Loading**: Initialize components as needed
- **Error Messages**: Helpful guidance for common issues
- **Mobile Support**: Responsive design works on all devices

#### CLI Experience
- **Fast Commands**: All CLI commands run instantly
- **Lazy Loading**: ControlNet models load only when generating
- **Status Commands**: Check system state without triggering downloads

### 🔧 Configuration Changes
- **setup.py**: Added ControlNet dependencies
- **pyproject.toml**: Updated dependency specifications
- **Model Registry**: Enhanced with ControlNet model definitions

### 📚 Documentation Updates
- **CONTROLNET_GUIDE.md**: Comprehensive 400+ line guide with examples
- **README.md**: Updated with lazy loading features and ControlNet quick start
- **API Documentation**: Complete endpoint reference with examples

### 🐛 Bug Fixes
- **Startup Crashes**: Fixed 404 errors from non-existent model repositories
- **Memory Leaks**: Improved cleanup of ControlNet preprocessors
- **Device Compatibility**: Better handling of CPU/GPU device switching
- **Error Handling**: More graceful failure modes with helpful messages

### ⚠️ Breaking Changes
- **Import Behavior**: `controlnet_preprocessors` module no longer auto-initializes
- **API Changes**: Some ControlNet endpoints require explicit initialization

### 🔄 Migration Guide
For users upgrading from v1.0.x:

1. **No Action Required**: Lazy loading is automatic and transparent
2. **Web UI**: ControlNet preprocessors initialize automatically when uploading images
3. **API Users**: Call `/api/controlnet/initialize` for faster subsequent processing
4. **Python API**: Use `controlnet_preprocessor.initialize()` for batch processing

### 🎯 Performance Metrics
- **Startup Time**: Reduced from 30+ seconds to <1 second
- **Memory Usage**: Reduced baseline memory footprint by ~2GB
- **First Generation**: Slightly slower due to lazy loading, then normal speed
- **Subsequent Generations**: Same performance as before

## [1.0.0] - 2025-5-28

### Added
- Initial release with core functionality
- Support for Stable Diffusion 1.5, SDXL, SD3, and FLUX models
- Basic LoRA support
- CLI interface
- REST API server
- Web UI interface
- Model management system

### Features
- Model downloading and management
- Image generation with various parameters
- Multiple interface options (CLI, API, Web UI)
- Hardware optimization (CUDA, MPS, CPU)
- Safety checker bypass for creative freedom

---

## Development Notes

### Version Numbering
- **Major** (X.0.0): Breaking changes, major feature additions
- **Minor** (1.X.0): New features, significant improvements
- **Patch** (1.1.X): Bug fixes, minor improvements

### Release Process
1. Update version in `__init__.py`
2. Update CHANGELOG.md with new features
3. Update documentation
4. Create release tag
5. Deploy to package repositories 