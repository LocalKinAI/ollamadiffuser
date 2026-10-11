---
name: ollamadiffuser
display_name: OllamaDiffuser - Local AI Image Generation
description: Generate and edit images, and make short videos, with locally-running AI models (FLUX, Qwen-Image, Z-Image, LTX-2, Stable Diffusion and 100+ more) via OllamaDiffuser
version: 1.1.0
author: LocalKinAI Team
homepage: https://github.com/LocalKinAI/ollamadiffuser
tags:
  - image-generation
  - stable-diffusion
  - flux
  - local-ai
  - creative
metadata:
  openclaw:
    requires:
      binaries: ["ollamadiffuser"]
---

# OllamaDiffuser - Local AI Image Generation

You can generate images using locally-installed AI diffusion models through the OllamaDiffuser API. All processing happens on the user's machine -- no cloud API keys required.

## The Quickest Way: One Command

Since OllamaDiffuser 2.3.0, `ollamadiffuser generate`, `edit` and `video` make a picture, an edit or a video in one command, with no server to start or stop (on an older install, upgrade with `pip install -U ollamadiffuser`, or use the REST API below):

```bash
ollamadiffuser generate "a serene mountain lake at sunset" -m z-image-turbo-mlx -o lake.png --json
ollamadiffuser edit "make it a snowy night" -i lake.png -m flux.2-klein-4b-edit-mlx -o snow.png --json
ollamadiffuser video "waves on a beach at dusk" -m ltx-2.3-mlx-q8 -s 4 -o waves.mp4 --json
```

With `--json`, stdout is one object: `{"path", "kind", "model", "seed", "elapsed_s", "server"}`, or `{"error": "..."}` with exit code 1. When a server already holds the model, the job goes to it; otherwise the model loads for the job. `ollamadiffuser ps --json` lists the servers running, `ollamadiffuser list --json` the installed models. On Apple Silicon a model that does not fit in free memory is refused with the numbers; don't pass `--force` without asking the user.

The REST API below is the way for many images in a row with one model loaded.

## Setup Check

Before generating images, verify OllamaDiffuser is installed and the API server is running.

1. **Check installation:**
   ```bash
   ollamadiffuser version
   ```
   If not installed, the user needs to run: `pip install ollamadiffuser`

2. **Start the API server** (if not already running):
   ```bash
   nohup ollamadiffuser --mode api > /dev/null 2>&1 &
   ```
   The server runs at `http://localhost:8000` by default.

3. **Verify the server is healthy:**
   ```bash
   curl -s http://localhost:8000/api/health
   ```
   Expected response: `{"status": "healthy", "model_loaded": true/false, "current_model": "..."}`

## Choosing a Model

Use the `recommend` command to find the best model for the user's hardware:

```bash
ollamadiffuser recommend
```

Quick reference by hardware:
- **Low VRAM (4-8 GB):** `stable-diffusion-1.5`, `dreamshaper`, `pixart-sigma`
- **Medium (8-16 GB):** `realvisxl-v4`, `sana-1.5`, `flux.1-dev-gguf-q4ks`
- **High VRAM (16+ GB):** `flux.1-schnell`, `stable-diffusion-3.5-large`
- **Apple Silicon (native MLX):** `z-image-turbo-mlx` (fast, 6 GB), `flux.2-klein-4b-mlx`, `qwen-image-2.1-turbo-mlx` (text in pictures, 46 GB); edits: `flux.2-klein-4b-edit-mlx`, `qwen-image-2.1-edit-mlx`; video: `ltx-2.3-mlx-q8`

## Model Management

### List available models
```bash
curl -s http://localhost:8000/api/models | python3 -m json.tool
```

### Download (pull) a model
```bash
curl -s -X POST "http://localhost:8000/api/models/pull?model_name=dreamshaper"
```
This downloads model weights from HuggingFace. It may take several minutes depending on model size.

### Load a model into memory
```bash
curl -s -X POST http://localhost:8000/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "dreamshaper"}'
```

### Check what model is currently loaded
```bash
curl -s http://localhost:8000/api/models/running
```

## Generating Images

Use `POST /api/generate` with `response_format` set to `b64_json` to receive the image as base64 JSON:

```bash
curl -s -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain landscape at sunset, photorealistic",
    "negative_prompt": "low quality, blurry, distorted",
    "width": 1024,
    "height": 1024,
    "response_format": "b64_json"
  }'
```

Response:
```json
{
  "image": "<base64-encoded PNG data>",
  "format": "png",
  "width": 1024,
  "height": 1024
}
```

To save the image to a file, extract the `image` field and decode it:
```bash
curl -s -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cute cat", "response_format": "b64_json"}' \
  | python3 -c "import sys,json,base64; d=json.load(sys.stdin); open('output.png','wb').write(base64.b64decode(d['image']))"
```

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the desired image |
| `negative_prompt` | string | "low quality..." | What to avoid in the image |
| `width` | int | 1024 | Image width in pixels |
| `height` | int | 1024 | Image height in pixels |
| `steps` | int | model default | Number of denoising steps (more = higher quality, slower) |
| `guidance_scale` | float | model default | How closely to follow the prompt (higher = more literal) |
| `seed` | int | random | Random seed for reproducible results |
| `response_format` | string | null | Set to `"b64_json"` for base64 JSON response |

## Complete Workflow

To generate an image from scratch on a new system:

```bash
# 1. Find the best model for this hardware
ollamadiffuser recommend

# 2. Start the API server
nohup ollamadiffuser --mode api > /dev/null 2>&1 &
sleep 3

# 3. Pull a model (dreamshaper is a good lightweight default)
curl -s -X POST "http://localhost:8000/api/models/pull?model_name=dreamshaper"

# 4. Load it into memory
curl -s -X POST http://localhost:8000/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "dreamshaper"}'

# 5. Generate an image
curl -s -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a magical forest with glowing mushrooms, fantasy art",
    "response_format": "b64_json"
  }'
```

## Tips

- Always use `response_format: "b64_json"` so you can process the image data programmatically
- Check `/api/health` before generating -- `model_loaded` must be `true`
- The first generation after loading a model is slower due to pipeline warmup
- Use `seed` for reproducible results when iterating on prompts
- Smaller models (SD 1.5 variants) generate faster; FLUX models produce higher quality
- If the server is not responding, check if it's running: `curl -s http://localhost:8000/api/health`
