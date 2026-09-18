# GGUF Models Guide

**GGUF (GPT-Generated Unified Format)** quantized models enable running FLUX.1-dev with significantly reduced VRAM requirements - from 20GB+ down to just 3GB!

## Quick Start

### 1. Install GGUF Dependencies
```bash
pip install stable-diffusion-cpp-python gguf
```

### 2. Check GGUF Support
```bash
ollamadiffuser registry check-gguf
```

### 3. Download and Use a GGUF Model
```bash
# Recommended for most users (6GB VRAM)
ollamadiffuser pull flux.1-dev-gguf-q4ks

# Load and generate
ollamadiffuser run flux.1-dev-gguf-q4ks
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains"}' \
  --output image.png
```

## Available GGUF Models

| Model | Size | VRAM | Quality | Speed | Best For |
|-------|------|------|---------|-------|----------|
| `flux.1-dev-gguf-q2k` | ~3GB | 3GB | ⭐⭐ | ⭐⭐⭐⭐⭐ | Testing, low-end hardware |
| `flux.1-dev-gguf-q3ks` | ~4GB | 4GB | ⭐⭐⭐ | ⭐⭐⭐⭐ | Good balance for mobile GPUs |
| `flux.1-dev-gguf-q4ks` | ~6GB | 6GB | ⭐⭐⭐⭐ | ⭐⭐⭐ | **Recommended** - best balance |
| `flux.1-dev-gguf-q5ks` | ~8GB | 8GB | ⭐⭐⭐⭐ | ⭐⭐ | High quality on mid-range GPUs |
| `flux.1-dev-gguf-q6k` | ~10GB | 10GB | ⭐⭐⭐⭐⭐ | ⭐⭐ | Near-original quality |
| `flux.1-dev-gguf-q8` | ~12GB | 12GB | ⭐⭐⭐⭐⭐ | ⭐ | Minimal quality loss |
| `flux.1-dev-gguf-f16` | ~16GB | 16GB | ⭐⭐⭐⭐⭐ | ⭐ | Full precision |

## What are GGUF Models?

GGUF models are quantized versions of the original FLUX.1-dev model that offer:

- **Reduced VRAM Usage**: Run on lower-end GPUs (3GB minimum vs 20GB+)
- **Smaller File Sizes**: Faster downloads and less disk space
- **CPU Fallback**: Can run on CPU when VRAM is insufficient
- **Multiple Quantization Levels**: Choose quality vs speed tradeoff
- **Commercial Use**: FLUX.1-dev license applies (non-commercial only)

## Hardware Recommendations

### Entry Level (4-6GB VRAM)
- **RTX 3060, RTX 4060, GTX 1660 Ti**
- Recommended: `flux.1-dev-gguf-q3ks` or `flux.1-dev-gguf-q4ks`

### Mid Range (8-12GB VRAM)
- **RTX 3070, RTX 4070, RTX 3080**
- Recommended: `flux.1-dev-gguf-q4ks` or `flux.1-dev-gguf-q5ks`

### High End (16GB+ VRAM)
- **RTX 3090, RTX 4080, RTX 4090, RTX A6000**
- Recommended: `flux.1-dev-gguf-q6k` or `flux.1-dev-gguf-f16`

### CPU Only
- Any modern CPU with 16GB+ RAM
- Recommended: `flux.1-dev-gguf-q2k` or `flux.1-dev-gguf-q3ks`
- Note: Much slower than GPU inference

## Installation & Setup

### Basic Installation
```bash
# Install OllamaDiffuser with GGUF support
pip install ollamadiffuser

# Install GGUF dependencies
pip install stable-diffusion-cpp-python gguf
```

### Hardware-Optimized Installation

#### NVIDIA GPU (CUDA)
```bash
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
```

#### Apple Silicon (Metal)
```bash
CMAKE_ARGS="-DSD_METAL=ON" pip install stable-diffusion-cpp-python
```

#### CPU Only
```bash
# Standard installation works for CPU inference
pip install stable-diffusion-cpp-python
```

## Usage Examples

### Command Line Interface
```bash
# List available GGUF models
ollamadiffuser registry check-gguf

# Download a model
ollamadiffuser pull flux.1-dev-gguf-q4ks

# Run the model
ollamadiffuser run flux.1-dev-gguf-q4ks

# Generate image via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic cityscape at sunset",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 4,
    "guidance_scale": 1.0
  }' \
  --output cityscape.png
```

### Python API
```python
from ollamadiffuser.core.models.manager import model_manager

# Load GGUF model
success = model_manager.load_model("flux.1-dev-gguf-q4ks")
if success:
    engine = model_manager.loaded_model
    
    # Generate image
    image = engine.generate_image(
        prompt="a beautiful landscape with mountains",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=1.0
    )
    image.save("landscape.jpg")
```

### Web Interface
```bash
# Start web UI with GGUF model
ollamadiffuser --mode ui
# Navigate to http://localhost:8001
# Select GGUF model from dropdown
```

## Performance Optimization

### Generation Parameters
FLUX GGUF models work best with these optimized settings:

```bash
# Optimal parameters for FLUX GGUF
{
  "num_inference_steps": 4,     # FLUX works best with 4 steps
  "guidance_scale": 1.0,        # FLUX optimized for CFG=1.0
  "sampler": "euler"            # Best sampler for FLUX
}
```

### Speed Tips
1. **Use 4 steps** - FLUX.1-schnell is optimized for 4-step generation
2. **Keep CFG at 1.0** - FLUX works best with guidance_scale=1.0
3. **Choose right quantization** - Balance quality vs speed for your hardware
4. **Enable GPU acceleration** - Install with CUDA/Metal support

### Memory Tips
1. **Start with lower quantization** - Try q3ks or q4ks first
2. **Reduce image resolution** - Use 512x512 for testing
3. **Close other GPU applications** - Free up VRAM
4. **Enable CPU offloading** - Automatic when VRAM is low

## Advanced Configuration

### Model File Structure
GGUF models require these files:
```
models/flux-dev-gguf/
├── flux1-dev-Q4_K_S.gguf     # Main model (choose quantization)
├── ae.safetensors            # VAE encoder/decoder
├── clip_l.safetensors        # CLIP text encoder
└── t5xxl_fp16.safetensors    # T5 text encoder
```

### Manual Model Setup
```bash
# Create model directory
mkdir -p models/flux-dev-gguf

# Download GGUF model (choose quantization)
hf download city96/FLUX.1-dev-gguf \
  flux1-dev-Q4_K_S.gguf --local-dir models/flux-dev-gguf

# Download required components
hf download black-forest-labs/FLUX.1-dev \
  ae.safetensors --local-dir models/flux-dev-gguf

hf download comfyanonymous/flux_text_encoders \
  clip_l.safetensors t5xxl_fp16.safetensors \
  --local-dir models/flux-dev-gguf
```

### Custom Model Registry
```python
# Add to model registry
"custom-flux-gguf": ModelConfig(
    name="custom-flux-gguf",
    path="models/flux-dev-gguf",
    model_type="flux",
    variant="gguf-q4ks",  # Triggers GGUF detection
    parameters={
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "max_sequence_length": 512
    }
)
```

## Troubleshooting

### Installation Issues

#### Missing Dependencies
```bash
# If cv2/OpenCV errors
pip install opencv-python>=4.8.0

# If GGUF support unavailable
pip install stable-diffusion-cpp-python gguf

# Comprehensive dependency check
ollamadiffuser verify-deps
```

#### Build Errors
```bash
# Clean reinstall
pip uninstall stable-diffusion-cpp-python
pip install --no-cache-dir stable-diffusion-cpp-python

# With CUDA support
CMAKE_ARGS="-DSD_CUDA=ON" pip install --no-cache-dir stable-diffusion-cpp-python
```

### Runtime Issues

#### Model Loading Errors
```bash
# Check GGUF support
ollamadiffuser registry check-gguf

# Verify model files
ls -la models/flux-dev-gguf/

# Check available models
ollamadiffuser list
```

#### Out of Memory
1. Try lower quantization: `q4ks` → `q3ks` → `q2k`
2. Reduce image size: `--width 512 --height 512`
3. Close other GPU applications
4. Use CPU inference for testing

#### Slow Performance
1. Install GPU acceleration (CUDA/Metal)
2. Use appropriate quantization for your hardware
3. Ensure you're using 4 steps (not 20-50)
4. Check `nvidia-smi` to verify GPU usage

### Debug Mode
```bash
# Enable verbose logging
ollamadiffuser --verbose run flux.1-dev-gguf-q4ks

# Check model status
python -c "
from ollamadiffuser.core.models.manager import model_manager
print('GGUF Support:', model_manager.is_gguf_available())
"
```

## Comparison with Regular Models

| Aspect | Regular FLUX.1-dev | GGUF q4ks | GGUF q6k |
|--------|---------------------|-----------|----------|
| **File Size** | ~24GB | ~6GB | ~10GB |
| **VRAM Usage** | 20GB+ | 6GB | 10GB |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Hardware** | High-end | Mid-range | Mid-high |
| **Download Time** | Hours | Minutes | Minutes |

## License & Legal

GGUF models inherit the original model license:
- **FLUX.1-dev**: Non-commercial use only
- **Requires HuggingFace token** for access
- **Commercial alternatives**: Use FLUX.1-schnell (Apache 2.0)

## Contributing

To add more GGUF model variants:

1. Update `ollamadiffuser/core/config/model_registry.py`:
```python
"new-gguf-model": ModelConfig(
    name="new-gguf-model",
    path="path/to/model",
    model_type="flux",
    variant="gguf-q4ks",
    huggingface_repo="user/repo-name"
)
```

2. Test with new model
3. Submit pull request with documentation

## Technical Implementation

### Architecture
- **Backend**: stable-diffusion.cpp with Python bindings
- **Integration**: Seamless with existing OllamaDiffuser engine
- **Detection**: Automatic GGUF model recognition
- **Optimization**: Hardware-specific acceleration

### Engine Integration
The GGUF support is fully integrated into:
- ✅ Model loading and management
- ✅ Image generation pipeline  
- ✅ CLI commands and web interface
- ✅ REST API endpoints
- ✅ Memory management and cleanup

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/LocalKinAI/ollamadiffuser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LocalKinAI/ollamadiffuser/discussions)
- **Documentation**: [ollamadiffuser.com](https://www.ollamadiffuser.com/)

---

**Ready to try GGUF models?** Start with: `pip install ollamadiffuser && ollamadiffuser pull flux.1-dev-gguf-q4ks` 🚀 