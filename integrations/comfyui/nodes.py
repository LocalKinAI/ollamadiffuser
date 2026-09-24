"""ComfyUI nodes that draw, edit and film with OllamaDiffuser's models.

ComfyUI runs PyTorch; OllamaDiffuser's Apple Silicon models are MLX
conversions (quantized, laid out for MLX), which ComfyUI's loaders cannot
read — and converting them back would only duplicate tens of gigabytes that
are already on the disk. So these nodes do not load anything. They ask an
OllamaDiffuser server for the picture or the clip over its HTTP API and
hand the result to the rest of the graph as an ordinary IMAGE or VIDEO.

A server already running the model (on any port from 8000 to 8019) is used
as it is — including ones something else started, which then stay as they
were. When none is, one is started on a free port from 8010 and, unless
"keep_loaded" is on, shut down again when the node is done, so the memory
goes back to ComfyUI.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management as mm

HOST = os.environ.get("OLLAMADIFFUSER_HOST", "127.0.0.1")
PORTS = range(8000, 8020)
FIRST_OWN_PORT = 8010
MODELS_DIR = Path(os.environ.get("OLLAMADIFFUSER_HOME", Path.home() / ".ollamadiffuser")) / "models"
LOGS = Path.home() / "Library" / "Logs" / "ollamadiffuser-comfy"
NOT_MODELS = {"face", "ltx-loras"}


# --- Finding and starting servers -------------------------------------------

def _get(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _running() -> dict[str, str]:
    """model name → base URL, for every OllamaDiffuser server answering."""
    found = {}
    for port in PORTS:
        base = f"http://{HOST}:{port}"
        try:
            info = _get(base + "/api/models/running", timeout=0.4)
        except Exception:
            continue
        model = info.get("model") if isinstance(info, dict) else None
        if model and model not in found:
            found[model] = base
    return found


_catalog: tuple[float, dict[str, str]] | None = None
_refreshing = False


def _installed() -> dict[str, str]:
    """Installed model → "video" or "image". ComfyUI asks for this every time
    it builds its node list, and its page waits for that list — once, with the
    box busy, for 26 seconds. So the last answer is given at once and a fresh
    one is fetched in the background; only the very first call waits."""
    global _catalog, _refreshing
    if _catalog is None:
        _catalog = (time.time(), _fetch_installed())
    elif time.time() - _catalog[0] > 300 and not _refreshing:
        _refreshing = True
        def refresh():
            global _catalog, _refreshing
            try:
                _catalog = (time.time(), _fetch_installed())
            finally:
                _refreshing = False
        threading.Thread(target=refresh, daemon=True).start()
    return _catalog[1]


def _fetch_installed() -> dict[str, str]:
    kinds: dict[str, str] = {}
    servers = _running()
    if servers:
        base = next(iter(servers.values()))
        try:
            for name in _get(base + "/api/models", timeout=2).get("installed", []):
                try:
                    info = _get(f"{base}/api/models/{name}", timeout=1)
                    kinds[name] = "video" if "video" in str(info.get("model_type", "")) else "image"
                except Exception:
                    kinds[name] = "video" if "ltx" in name or "video" in name else "image"
        except Exception:
            pass
    if not kinds and MODELS_DIR.is_dir():
        for entry in sorted(MODELS_DIR.iterdir()):
            if entry.is_dir() and entry.name not in NOT_MODELS:
                kinds[entry.name] = "video" if "ltx" in entry.name or "video" in entry.name else "image"
    return kinds


def _models(kind: str) -> list[str]:
    names = [name for name, k in _installed().items() if k == kind]
    return names or [f"(no {kind} model installed in OllamaDiffuser)"]


def _binary() -> str:
    for candidate in (os.environ.get("OLLAMADIFFUSER_BIN"),
                      str(Path.home() / ".ollamadiffuser" / "venv" / "bin" / "ollamadiffuser"),
                      shutil.which("ollamadiffuser")):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("ollamadiffuser not found: set OLLAMADIFFUSER_BIN to its path")


def _free_port() -> int:
    for port in range(FIRST_OWN_PORT, PORTS.stop):
        with socket.socket() as s:
            if s.connect_ex((HOST, port)) != 0:
                return port
    raise RuntimeError(f"no free port between {FIRST_OWN_PORT} and {PORTS.stop - 1}")


def _server(model: str) -> tuple[str, bool]:
    """A server running `model`, and whether this call started it."""
    if model.startswith("("):
        raise RuntimeError("Install a model with `ollamadiffuser pull` first")
    base = _running().get(model)
    if base:
        return base, False
    # The model is about to take its memory; give it ComfyUI's first.
    mm.unload_all_models()
    mm.soft_empty_cache()
    port = _free_port()
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{model}.log", "ab")
    env = dict(os.environ)
    # ffmpeg and friends live in Homebrew, which a service's PATH often lacks.
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
    subprocess.Popen([_binary(), "run", model, "--host", HOST, "--port", str(port)],
                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = f"http://{HOST}:{port}"
    deadline = time.time() + 600
    while time.time() < deadline:
        mm.throw_exception_if_processing_interrupted()
        try:
            if _get(base + "/api/health", timeout=1) is not None:
                return base, True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"{model} did not come up in 10 minutes; see {LOGS / (model + '.log')}")


def _shutdown(base: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(base + "/api/shutdown", data=b"", method="POST"), timeout=5)
    except Exception:
        pass  # it closes the connection on its way out


# --- Talking to one ----------------------------------------------------------

def _post(url: str, *, json_body: dict | None = None, fields: dict | None = None,
          files: list[tuple[str, str, bytes, str]] | None = None, timeout: float = 3600) -> tuple[bytes, dict]:
    """POST JSON or a multipart form; the body and headers back. Waits in a
    thread so the Cancel button still works (the server finishes its current
    job either way)."""
    if json_body is not None:
        data = json.dumps({k: v for k, v in json_body.items() if v is not None}).encode()
        headers = {"Content-Type": "application/json"}
    else:
        boundary = uuid.uuid4().hex
        parts = []
        for key, value in (fields or {}).items():
            if value is None:
                continue
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        for key, filename, content, mime in files or []:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                         f'Content-Type: {mime}\r\n\r\n'.encode() + content + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(parts)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    result: dict = {}

    def call():
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method="POST"),
                                        timeout=timeout) as response:
                result["body"], result["headers"] = response.read(), dict(response.headers)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except Exception:
                pass
            result["error"] = RuntimeError(f"OllamaDiffuser: {detail}")
        except Exception as e:
            result["error"] = e

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(0.5)
        mm.throw_exception_if_processing_interrupted()
    if "error" in result:
        raise result["error"]
    return result["body"], result["headers"]


def _to_tensor(picture: Image.Image) -> torch.Tensor:
    array = np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


def _to_png(image: torch.Tensor, index: int = 0) -> bytes:
    array = (image[index].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _to_wav(audio: dict) -> bytes:
    waveform = audio["waveform"][0].clamp(-1, 1).cpu().numpy()  # channels × samples
    pcm = (waveform.T * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(pcm.shape[1])
        out.setsampwidth(2)
        out.setframerate(int(audio["sample_rate"]))
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _optional(value, unset):
    return None if value == unset else value


SEED = ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True})
KEEP = ("BOOLEAN", {"default": False, "tooltip": "Leave a server this node started running afterwards "
                                                 "(faster next time; holds its memory). Servers something "
                                                 "else started are never stopped."})


# --- Nodes ---------------------------------------------------------------------

class OllamaDiffuserTextToImage:
    DESCRIPTION = "Draw with an OllamaDiffuser model (MLX on Apple Silicon)."
    CATEGORY = "OllamaDiffuser"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_models("image"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "0 = the model's own"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "0 = the model's own"}),
                "seed": SEED,
                "keep_loaded": KEEP,
            },
            "optional": {"negative_prompt": ("STRING", {"multiline": True, "default": ""})},
        }

    def run(self, model, prompt, width, height, steps, cfg, seed, keep_loaded, negative_prompt=""):
        base, started = _server(model)
        try:
            body, _ = _post(base + "/api/generate", json_body={
                "prompt": prompt, "negative_prompt": negative_prompt or None, "width": width, "height": height,
                "steps": _optional(steps, 0), "cfg_scale": _optional(cfg, 0.0), "seed": seed,
            })
        finally:
            if started and not keep_loaded:
                _shutdown(base)
        return (_to_tensor(Image.open(io.BytesIO(body))),)


class OllamaDiffuserEditImage:
    DESCRIPTION = ("Edit a picture with an OllamaDiffuser model — up to four references for editors "
                   "that take several (the first sets the size and is image 1 in the prompt).")
    CATEGORY = "OllamaDiffuser"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_models("image"),),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "0 = the model's own"}),
                "strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "0 = the model's own. On a 4-step model 0.75 leaves one step"}),
                "seed": SEED,
                "keep_loaded": KEEP,
            },
            "optional": {
                "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def run(self, model, image, prompt, steps, strength, seed, keep_loaded,
            image_2=None, image_3=None, image_4=None, negative_prompt=""):
        files = [("image", "image.png", _to_png(image), "image/png")]
        extras = [_to_png(i) for i in (image_2, image_3, image_4) if i is not None]
        files += [("images", f"image-{n + 2}.png", png, "image/png") for n, png in enumerate(extras)]
        base, started = _server(model)
        try:
            body, _ = _post(base + "/api/generate/img2img", fields={
                "prompt": prompt, "negative_prompt": negative_prompt or None,
                "num_inference_steps": _optional(steps, 0), "strength": _optional(strength, 0.0), "seed": seed,
            }, files=files)
        finally:
            if started and not keep_loaded:
                _shutdown(base)
        return (_to_tensor(Image.open(io.BytesIO(body))),)


class OllamaDiffuserVideo:
    DESCRIPTION = ("Film with an OllamaDiffuser video model (LTX on MLX): text to video, a first frame, "
                   "a voice that drives it, or a control video it follows frame for frame.")
    CATEGORY = "OllamaDiffuser"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_models("video"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seconds": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.5, "tooltip": "0 = the model decides"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32, "tooltip": "0 = the first frame's shape, or the model's own"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32, "tooltip": "0 = the first frame's shape, or the model's own"}),
                "seed": SEED,
                "keep_loaded": KEEP,
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "audio": ("AUDIO",),
                "control": ("VIDEO", {"tooltip": "A pose / depth / edge video to follow (IC-LoRA)"}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora": ("STRING", {"default": "", "tooltip": "A path or Hugging Face repo id; empty = none (or union control with a control video)"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "0 = the model's own"}),
            },
        }

    def run(self, model, prompt, seconds, width, height, seed, keep_loaded, first_frame=None, audio=None,
            control=None, control_strength=1.0, lora="", lora_strength=1.0, steps=0):
        files = []
        if first_frame is not None:
            files.append(("image", "first.png", _to_png(first_frame), "image/png"))
            if width == 0 and height == 0:
                # Left to itself the model films at its own default (landscape
                # 704×480 for LTX) and crops a portrait first frame to fit.
                # The frame's own shape, in multiples of 32, long side ≤ 1024.
                h, w = first_frame.shape[1], first_frame.shape[2]
                scale = min(1.0, 1024 / max(w, h))
                width, height = (max(32, int(round(w * scale / 32)) * 32), max(32, int(round(h * scale / 32)) * 32))
        if audio is not None:
            files.append(("audio", "audio.wav", _to_wav(audio), "audio/wav"))
        if control is not None:
            spill = Path(folder_paths.get_temp_directory()) / f"od-control-{uuid.uuid4().hex}.mp4"
            spill.parent.mkdir(parents=True, exist_ok=True)
            control.save_to(str(spill))
            files.append(("control", "control.mp4", spill.read_bytes(), "video/mp4"))
            spill.unlink(missing_ok=True)
        base, started = _server(model)
        try:
            body, _ = _post(base + "/api/generate/video", fields={
                "prompt": prompt, "seconds": _optional(seconds, 0.0), "width": _optional(width, 0),
                "height": _optional(height, 0), "seed": seed, "steps": _optional(steps, 0),
                "control_strength": control_strength if control is not None else None,
                "lora": lora or None, "lora_strength": lora_strength if lora else None,
            }, files=files)
        finally:
            if started and not keep_loaded:
                _shutdown(base)
        out = Path(folder_paths.get_temp_directory()) / f"ollamadiffuser-{uuid.uuid4().hex[:12]}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)
        try:
            from comfy_api.latest import VideoFromFile
        except ImportError:  # ComfyUI before the versioned API
            from comfy_api.input_impl import VideoFromFile
        video = VideoFromFile(str(out))
        return (video, str(out))


NODE_CLASS_MAPPINGS = {
    "OllamaDiffuserTextToImage": OllamaDiffuserTextToImage,
    "OllamaDiffuserEditImage": OllamaDiffuserEditImage,
    "OllamaDiffuserVideo": OllamaDiffuserVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaDiffuserTextToImage": "OllamaDiffuser · Text to Image",
    "OllamaDiffuserEditImage": "OllamaDiffuser · Edit Image",
    "OllamaDiffuserVideo": "OllamaDiffuser · Video",
}
