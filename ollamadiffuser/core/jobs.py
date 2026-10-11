"""One generation, wherever the model already is.

An agent that wants one picture should not have to start a server, wait for
it, post to it and stop it again. A job goes to an OllamaDiffuser server
already running with the model — loading a second copy of a 40 GB model next
to it is the one thing never to do — and otherwise loads the model in this
process, runs, and lets it go. ``ollamadiffuser generate``, ``edit`` and
``video`` are this module, and so are the MCP tools.
"""
from __future__ import annotations

import contextlib
import io
import random
import shutil
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

KINDS = ("image", "edit", "video")
# Long enough for a video on a slow machine; past it the server is stuck.
HTTP_TIMEOUT = 3600
_here = threading.Lock()


class JobError(RuntimeError):
    """A job that cannot run; the message says why and what to do about it."""


@dataclass
class Job:
    """What to make. Paths are files on this machine."""

    kind: str                                   # "image", "edit" or "video"
    prompt: str
    model: Optional[str] = None                 # None: the one model running
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None                  # None: one is picked and reported
    scheduler: Optional[str] = None
    image: Optional[str] = None                 # edit: the picture; video: its first frame
    references: List[str] = field(default_factory=list)  # edit: further pictures
    mask: Optional[str] = None                  # edit: where to change it
    strength: Optional[float] = None            # edit
    seconds: Optional[float] = None             # video
    frames: Optional[int] = None                # video
    audio: Optional[str] = None                 # video: a track to drive it
    control: Optional[str] = None               # video: a reference video to follow
    lora: Optional[str] = None                  # a LoRA file or Hugging Face repo
    lora_weight: Optional[str] = None           # the file inside that repo
    lora_scale: float = 1.0


# A server on another machine — the GPU box down the hall — that jobs go to
# instead of anything on this one: `--server`, or this, set once.
SERVER_ENV = "OLLAMADIFFUSER_SERVER"


def servers(host: Optional[str] = None) -> List[Dict]:
    """The OllamaDiffuser servers running on this machine, or on ``host``."""
    from .utils import server_registry
    return server_registry.discover(host=host)


def server_url(value: str) -> str:
    """``192.168.0.21:8010`` → ``http://192.168.0.21:8010``; no port means 8000."""
    from urllib.parse import urlsplit
    value = value.strip().rstrip("/")
    if "://" not in value:
        value = f"http://{value}"
    parts = urlsplit(value)
    if not parts.hostname:
        raise JobError(f"Not a server address: {value}")
    if parts.port is None:
        value = f"{parts.scheme}://{parts.netloc}:8000{parts.path}"
    return value


def run(job: Job, *, output: Optional[str] = None, output_dir: Optional[str] = None,
        keep_loaded: bool = False, force: bool = False, manager=None,
        server: Optional[str] = None) -> Dict:
    """Run ``job``; return the file it made, the model, the seed, the time it
    took, and the server that did it (None when it ran in this process).

    ``server`` sends the job to that server and nowhere else — another
    machine's, typically; the files go up and the result comes back here.
    ``keep_loaded`` keeps a model this process loaded for the next job (the
    MCP server); otherwise it is let go when the job is done. ``force``
    loads a model even when free memory looks too low.
    """
    if manager is None:
        from .models.manager import model_manager as manager
    _check(job)
    seed = job.seed if job.seed is not None else random.randint(0, 2**31 - 1)
    target = _target(job.kind, seed, output, output_dir)
    started = time.time()

    if server:
        server = _remote(server_url(server), job.model)
    else:
        here = manager.current_model_name if manager.is_model_loaded() else None
        if job.model is None and here is None:
            server = _the_running_server()
        elif job.model is not None and job.model != here:
            server = next((s for s in servers() if s["loaded"] and s["model"] == job.model), None)

    if server:
        if job.lora and job.kind != "video":
            raise JobError(
                f"{server['model']} is running at {server['url']} for everyone who calls it, "
                f"and a LoRA there would change it for all of them. Load the LoRA on that server "
                f"yourself (POST {server['url']}/api/lora/load) and call again without one, or stop "
                f"that server first."
            )
        _write(_over_http(server["url"], job, seed), target, job.kind)
        model = server["model"]
    else:
        model = job.model or here
        # The MCP server runs tools in threads; one model loads and runs at a time.
        with _here:
            _in_process(manager, model, job, seed, target, keep_loaded, force)
    return {
        "path": str(target),
        "kind": job.kind,
        "model": model,
        "seed": seed,
        "elapsed_s": round(time.time() - started, 1),
        "server": server["url"] if server else None,
    }


def _remote(url: str, model: Optional[str]) -> Dict:
    """The server at ``url``, holding ``model``.

    One with no model loaded (`ollamadiffuser serve`) is waiting for one and
    loads the model the job names. One holding another model is left alone:
    others may be using it, and swapping it would cost them a reload.
    """
    import requests

    try:
        answer = requests.get(f"{url}/api/models/running", timeout=10).json()
        if not isinstance(answer, dict) or "loaded" not in answer:
            raise ValueError("not an OllamaDiffuser server")
    except Exception as e:
        raise JobError(
            f"No OllamaDiffuser server answers at {url} ({e}). Start one on that machine with "
            f"`ollamadiffuser run MODEL --host 0.0.0.0 --port PORT`; the default host, localhost, "
            f"takes no calls from other machines."
        ) from e
    held = answer.get("model") if answer.get("loaded") else None
    if model and held and held != model:
        raise JobError(f"The server at {url} runs {held}, not {model}. Leave the model out to use "
                       f"{held}, or point at a server that runs {model}.")
    if not held:
        if not model:
            raise JobError(f"The server at {url} has no model loaded; name the one to load there.")
        try:
            response = requests.post(f"{url}/api/models/load", json={"model_name": model}, timeout=HTTP_TIMEOUT)
        except requests.RequestException as e:
            raise JobError(f"The server at {url} did not answer: {e}") from e
        if response.status_code != 200:
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text[:300]
            raise JobError(f"The server at {url} could not load {model}: {detail or response.status_code}")
        held = model
    return {"url": url, "model": held, "loaded": True}


def _the_running_server() -> Dict:
    running = [s for s in servers() if s["loaded"]]
    if len(running) == 1:
        return running[0]
    if not running:
        raise JobError("No model loaded or named, and no OllamaDiffuser server is running one. "
                       "Name a model (`ollamadiffuser list` shows the installed ones).")
    names = ", ".join(f"{s['model']} on port {s['port']}" for s in running)
    raise JobError(f"Several servers are running ({names}); name the model to use.")


def _check(job: Job) -> None:
    if job.kind not in KINDS:
        raise JobError(f"Unknown job kind {job.kind!r}; it is one of {', '.join(KINDS)}.")
    if not (job.prompt or "").strip():
        raise JobError("The prompt is empty.")
    if job.kind == "edit" and not job.image:
        raise JobError("An edit needs the picture to edit.")
    for name in [job.image, *job.references, job.mask, job.audio, job.control]:
        if name and not Path(name).expanduser().is_file():
            raise JobError(f"No such file: {name}")


def _target(kind: str, seed: int, output: Optional[str], output_dir: Optional[str]) -> Path:
    suffix = ".mp4" if kind == "video" else ".png"
    if output:
        path = Path(output).expanduser()
        if not path.suffix:
            path = path.with_suffix(suffix)
    else:
        folder = Path(output_dir).expanduser() if output_dir else Path.cwd()
        stem = f"ollamadiffuser-{time.strftime('%Y%m%d-%H%M%S')}-{seed}"
        path, n = folder / f"{stem}{suffix}", 1
        while path.exists():
            path, n = folder / f"{stem}-{n}{suffix}", n + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _present(values: Dict) -> Dict:
    return {k: v for k, v in values.items() if v is not None}


def _over_http(url: str, job: Job, seed: int) -> bytes:
    """The job's file, made by the server at ``url``."""
    import requests

    try:
        if job.kind == "image":
            body = {"prompt": job.prompt, "negative_prompt": job.negative_prompt,
                    "width": job.width or 1024, "height": job.height or 1024,
                    "steps": job.steps, "cfg_scale": job.guidance, "seed": seed,
                    "scheduler": job.scheduler}
            response = requests.post(f"{url}/api/generate", json=_present(body), timeout=HTTP_TIMEOUT)
        else:
            if job.kind == "edit":
                endpoint = "/api/generate/img2img"
                fields = {"prompt": job.prompt, "negative_prompt": job.negative_prompt,
                          "num_inference_steps": job.steps, "guidance_scale": job.guidance,
                          "seed": seed, "strength": job.strength, "scheduler": job.scheduler}
                uploads = [("image", job.image)] + [("images", r) for r in job.references]
                uploads += [("mask", job.mask)] if job.mask else []
            else:
                endpoint = "/api/generate/video"
                fields = {"prompt": job.prompt, "seconds": job.seconds, "frames": job.frames,
                          "width": job.width, "height": job.height, "seed": seed,
                          "steps": job.steps, "cfg_scale": job.guidance, "lora": job.lora,
                          "lora_strength": job.lora_scale if job.lora else None}
                uploads = [(name, path) for name, path in
                           (("image", job.image), ("audio", job.audio), ("control", job.control)) if path]
            with contextlib.ExitStack() as stack:
                files = [(name, (Path(path).name, stack.enter_context(open(Path(path).expanduser(), "rb"))))
                         for name, path in uploads]
                response = requests.post(f"{url}{endpoint}", timeout=HTTP_TIMEOUT, files=files or None,
                                         data={k: str(v) for k, v in _present(fields).items()})
    except requests.RequestException as e:
        raise JobError(f"The server at {url} did not answer: {e}") from e
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text[:300]
        raise JobError(f"The server at {url} could not do it: {detail or response.status_code}")
    return response.content


def _in_process(manager, model: str, job: Job, seed: int, target: Path,
                keep_loaded: bool, force: bool) -> None:
    if not (manager.is_model_loaded() and manager.current_model_name == model):
        if not manager.is_model_installed(model):
            raise JobError(f"{model} is not installed. Install it first: ollamadiffuser pull {model}")
        if not manager.load_model(model, force=force, persist=keep_loaded):
            raise JobError(manager.last_error or f"{model} failed to load")
    engine = manager.loaded_model
    try:
        if job.kind == "video":
            made = engine.generate_video(job.prompt, output=str(target), seconds=job.seconds, **_present({
                "frames": job.frames, "width": job.width, "height": job.height, "seed": seed,
                "steps": job.steps, "cfg_scale": job.guidance, "image": job.image,
                "audio": job.audio, "control": job.control, "lora": job.lora,
                "lora_strength": job.lora_scale if job.lora else None,
            }))
            if made and Path(made).resolve() != target:
                shutil.move(str(made), target)
            return
        _apply_lora(engine, job)
        kwargs = _present({"negative_prompt": job.negative_prompt, "num_inference_steps": job.steps,
                           "guidance_scale": job.guidance, "seed": seed, "scheduler": job.scheduler})
        if job.kind == "edit":
            pictures = [_open(path) for path in [job.image, *job.references]]
            kwargs.update(image=pictures[0], width=pictures[0].width, height=pictures[0].height)
            if len(pictures) > 1:
                kwargs["images"] = pictures
            if job.mask:
                kwargs["mask_image"] = _open(job.mask)
            if job.strength is not None:
                kwargs["strength"] = job.strength
        else:
            kwargs.update(width=job.width or 1024, height=job.height or 1024)
        _save_image(engine.generate_image(prompt=job.prompt, **kwargs), target)
    except JobError:
        raise
    except Exception as e:
        raise JobError(f"{model} could not do it: {e}") from e
    finally:
        if not keep_loaded:
            manager.unload_model(persist=False)


# The LoRA jobs left on a model this process keeps loaded (the MCP server):
# the next job changes it only when it asks for another, since on MLX every
# change rebuilds the model.
_lora_on: Dict[str, object] = {"engine": None, "lora": None}


def _apply_lora(engine, job: Job) -> None:
    want = (job.lora, job.lora_weight, job.lora_scale) if job.lora else None
    ref = _lora_on["engine"]
    have = _lora_on["lora"] if ref is not None and ref() is engine else None
    if want == have:
        return
    if have is not None:
        engine.unload_lora()
    _lora_on.update(engine=weakref.ref(engine), lora=None)
    if want is not None:
        if not engine.load_lora_runtime(job.lora, job.lora_weight, job.lora_scale):
            raise JobError(f"Could not load the LoRA {job.lora} on this model.")
        _lora_on["lora"] = want


def _open(path: str):
    from PIL import Image
    return Image.open(Path(path).expanduser()).convert("RGB")


def _save_image(image, target: Path) -> None:
    # JPEG has no alpha, and Qwen-Image-2.1's edits come back RGBA.
    if target.suffix.lower() in (".jpg", ".jpeg") and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(target)


def _write(data: bytes, target: Path, kind: str) -> None:
    if kind != "video" and target.suffix.lower() != ".png":
        from PIL import Image
        _save_image(Image.open(io.BytesIO(data)), target)
    else:
        target.write_bytes(data)
