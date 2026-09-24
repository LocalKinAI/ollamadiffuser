# OllamaDiffuser for ComfyUI

Three nodes that let a ComfyUI workflow draw, edit and film with the models
OllamaDiffuser already has — the MLX builds of Qwen Image 2.1, FLUX.2 klein,
FLUX.1 Kontext, Boogu, LTX 2.3/2.5 — without ComfyUI loading them.

ComfyUI runs PyTorch, and OllamaDiffuser's Apple Silicon models are MLX
conversions ComfyUI cannot read; converting them back would only put a
second copy of each on the disk. So the nodes call OllamaDiffuser's HTTP API
and hand the result on as an ordinary `IMAGE` or `VIDEO`, to be upscaled,
masked, composited or saved by whatever else is in the graph.

| Node | Calls | Inputs |
|---|---|---|
| OllamaDiffuser · Text to Image | `POST /api/generate` | prompt, size, steps, cfg, seed |
| OllamaDiffuser · Edit Image | `POST /api/generate/img2img` | an image and up to three more references |
| OllamaDiffuser · Video | `POST /api/generate/video` | first frame, audio, control video, LoRA |

## Install

```bash
ln -s /path/to/ollamadiffuser/integrations/comfyui ~/ComfyUI/custom_nodes/comfyui-ollamadiffuser
```

Restart ComfyUI. The nodes are under **OllamaDiffuser** in the node menu.

## Servers

A node uses a server already running its model on any port from 8000 to
8019, and leaves it as it was. When none is running, it starts
`ollamadiffuser run <model>` on a free port from 8010, after asking ComfyUI
to unload its own models first. Unless `keep_loaded` is on, it shuts that
server down again when it is done.

Logs go to `~/Library/Logs/ollamadiffuser-comfy/`. The environment variables
`OLLAMADIFFUSER_BIN`, `OLLAMADIFFUSER_HOST` and `OLLAMADIFFUSER_HOME` override
where it looks.
