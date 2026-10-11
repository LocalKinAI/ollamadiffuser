---
name: ollamadiffuser
description: Make images, edit pictures and make short videos on this machine with OllamaDiffuser (FLUX, Qwen-Image, Z-Image, LTX-2 and more, on the local GPU). Use when the user wants a picture, an edit of a picture, or a video made locally rather than by a cloud service, or asks for OllamaDiffuser.
---

# OllamaDiffuser: local images, edits and video

Everything runs on this machine. A job makes a file; the command prints where it went.
These commands need OllamaDiffuser 2.3.0 or later, and `--server` / `ps --host` need 2.4.0
(`ollamadiffuser --version`; upgrade with `pip install -U ollamadiffuser`).

## One command per job

```bash
ollamadiffuser generate "PROMPT" -m MODEL -o out.png --json
ollamadiffuser edit "INSTRUCTION" -i picture.png -m MODEL -o out.png --json
ollamadiffuser video "PROMPT" -m MODEL -s 4 -o clip.mp4 --json
```

With `--json`, stdout is one JSON object:
`{"path", "kind", "model", "seed", "elapsed_s", "server"}`, or `{"error": "..."}` with exit code 1.
Read the error: it says what to do (install a model, free memory, name a model).

- `edit`: repeat `-i` for more reference pictures (models that take several call them image 1,
  image 2, ... in the prompt; the first sets the size). `--mask m.png` (white = change) for models
  that take one, `--strength` to stay closer to the picture.
- `video`: `-i first.png` starts from a picture, `--audio voice.wav` drives it, `-s` sets seconds.
- Common: `--seed`, `--steps`, `-g/--guidance`, `-W/-H` for size. Leave them out for the model's defaults.
- Leave `-m` out only when exactly one server is running; the job then uses that server's model.

## Before a job

1. `ollamadiffuser ps --json` lists the servers already running and the model each holds. A job
   for one of those models goes to that server: no load, no second copy in memory. Do not stop
   servers you did not start.
2. `ollamadiffuser list --json` lists installed models with `type`, `min_memory_gb` and `notes`.
   `ollamadiffuser registry list --format json` lists everything that can be installed.
3. A model that is not installed: `ollamadiffuser pull MODEL`. Downloads run to tens of GB, so
   ask the user first.

## Choosing a model

| Want | Apple Silicon | NVIDIA |
|---|---|---|
| A picture, fast | `z-image-turbo-mlx` (6 GB) | `z-image-turbo` |
| Text in the picture: signs, posters, Chinese | `qwen-image-2.1-turbo-mlx` (46 GB) | `qwen-image-2.1-turbo` |
| An edit by instruction | `flux.2-klein-4b-edit-mlx` (12 GB), `qwen-image-2.1-edit-mlx` (46 GB) | `qwen-image-2.1` (pass the picture) |
| A video | `ltx-2.3-mlx-q8` (24 GB), `ltx-2.3-mlx-q4` (12 GB) | `ltx-2.3-distilled`, `wan2.2-ti2v-5b` |

Use what is installed when it fits the job; `notes` in `list --json` says what each one is good at.

## Memory

A model needs about `min_memory_gb` free. On a Mac a load is refused when that much is not free,
with the numbers in the error. Then prefer a smaller model, or stop a server you started yourself
(`ollamadiffuser stop --port N`). Pass `--force` only if the user agrees: on a Mac, running out of
memory can stall the whole machine.

## The GPU on another machine

When the models live on another machine (a GPU box on the LAN), send the jobs there:

```bash
ollamadiffuser ps --host 192.168.0.21 --json     # what runs on that machine
ollamadiffuser generate "PROMPT" --server 192.168.0.21:8010 -o out.png --json
export OLLAMADIFFUSER_SERVER=192.168.0.21:8010   # or once, for every job (and for `ollamadiffuser mcp`)
```

The pictures to edit go up from this machine, the result is saved on this machine, and nothing
loads here. A server holding another model is left alone, and the error says which model it
holds; leave `-m` out to use it. A server must be started on that machine with
`--host 0.0.0.0`, because the default host takes no calls from other machines.

## Many jobs with one model

Each job loads the model when no server holds it, which takes seconds to a minute. For a batch,
start a server once and send the jobs to it:

```bash
ollamadiffuser run MODEL --port 8010 > /tmp/ollamadiffuser-8010.log 2>&1 &
# wait until `ollamadiffuser ps --json` lists port 8010 with the model loaded, then run the jobs
ollamadiffuser stop --port 8010   # when done
```

## Other ways in

- MCP: `ollamadiffuser enable mcp`, then `claude mcp add ollamadiffuser -- ollamadiffuser mcp`.
  Tools: `generate_image`, `edit_image`, `generate_video`, `list_servers`, `list_models`,
  `load_model`, `get_status`. They take file paths on this machine.
- HTTP, on a server started with `ollamadiffuser run MODEL`: `POST /api/generate` (JSON;
  `"response_format": "b64_json"` returns base64), `POST /api/generate/img2img` (form: `image`,
  `images`, `mask`), and `POST /api/generate/video` (form: `image`, `audio`). The mp4 is the body,
  and `X-Output-Path` says where it was saved.
