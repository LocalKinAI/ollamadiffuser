"""OllamaDiffuser REST API Server"""

import asyncio
import base64
import io
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from ..core.config.settings import settings
from ..core.models.manager import model_manager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Request models ---


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution"
    num_inference_steps: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    cfg_scale: Optional[float] = None
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    response_format: Optional[str] = None  # "b64_json" for JSON with base64, None for raw PNG
    # An mflux sampling schedule, for the models that take one — e.g.
    # "viggle_turbo" with Viggle's 6-step LoRA on Qwen-Image-2.1.
    scheduler: Optional[str] = None
    # The model to use. A server started with `serve` loads it first if it
    # holds another; one started for a model with `run` takes only that one.
    model: Optional[str] = None


class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    strength: float = 0.75


class LoadModelRequest(BaseModel):
    model_name: str


class LoadLoRARequest(BaseModel):
    lora_name: str
    repo_id: str
    weight_name: Optional[str] = None
    scale: float = 1.0


# --- Helpers ---


def _image_to_response(image: Image.Image) -> Response:
    """Convert PIL Image to PNG Response."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _image_to_json_response(image: Image.Image) -> Dict[str, Any]:
    """Convert PIL Image to JSON response with base64-encoded PNG."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "image": b64_data,
        "format": "png",
        "width": image.width,
        "height": image.height,
    }


def _get_engine():
    """Get the currently loaded inference engine or raise 400."""
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=400, detail="No model loaded")
    return model_manager.loaded_model


def parse_keep_alive(value: Optional[str]) -> Optional[float]:
    """``10m`` / ``1h`` / ``90s`` / ``90`` → seconds; ``forever`` (or nothing,
    or a negative number) → None, never unload; ``0`` → right after each job."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "forever", "never", "inf", "infinite"):
        return None
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([smh]?)", text)
    if not match:
        raise ValueError(f"not a duration: {value!r} (use 90s, 10m, 1h, 0 or forever)")
    seconds = float(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]
    return None if seconds < 0 else seconds


# --- App factory ---


def create_app(pinned: Optional[str] = None, keep_alive: Optional[float] = None) -> FastAPI:
    """Create FastAPI application.

    ``pinned``: the model a server started with `run MODEL` is for. It takes
    only that model, and reloads it after a keep-alive let it go. With none
    (`serve`), the server loads whichever model a request names — Ollama's
    way: one server for every model, one model in memory at a time.

    ``keep_alive``: seconds a model may sit idle before it is unloaded to give
    its memory back; None keeps it until the server stops.
    """

    def _loaded() -> Optional[str]:
        return model_manager.current_model_name if model_manager.is_model_loaded() else None

    def _touch() -> None:
        app.state.last_used = time.monotonic()

    async def _switch_to(name: str) -> None:
        """Load ``name`` in place of what is loaded. A load the memory check
        refuses puts the previous model back rather than leave the server
        empty. Call with the lock held."""
        if not model_manager.is_model_installed(name):
            raise HTTPException(status_code=404, detail=(
                f"Model {name} is not installed. Install it first: ollamadiffuser pull {name}"))
        previous = _loaded()
        logger.info(f"Loading {name}" + (f" in place of {previous}" if previous else ""))
        if await asyncio.to_thread(model_manager.load_model, name):
            _touch()
            return
        reason = getattr(model_manager, "last_error", None) or f"Failed to load model {name}"
        if previous and previous != name:
            await asyncio.to_thread(model_manager.load_model, previous, force=True)
        raise HTTPException(status_code=503, detail=reason)

    async def _engine_for(model: Optional[str]):
        """The engine for a request naming ``model`` (or none). Call with the
        lock held: a load here may unload what another request was using."""
        loaded = _loaded()
        wanted = model or loaded or app.state.pinned
        if wanted is None:
            raise HTTPException(status_code=400, detail=(
                "No model loaded. Name one in `model`; this server loads it."))
        if wanted == loaded:
            return model_manager.loaded_model
        pinned_model = app.state.pinned
        if pinned_model and wanted != pinned_model:
            raise HTTPException(status_code=409, detail=(
                f"This server runs {pinned_model} and takes only that model (it was started with "
                f"`ollamadiffuser run {pinned_model}`). Leave `model` out, or send it to a server "
                f"started with `ollamadiffuser serve`, which loads models on demand."))
        await _switch_to(wanted)
        return model_manager.loaded_model

    async def _expire() -> bool:
        """Unload the model if it has sat idle past the keep-alive; True if it did."""
        limit = app.state.keep_alive
        if limit is None or not model_manager.is_model_loaded():
            return False
        if time.monotonic() - app.state.last_used < limit:
            return False
        async with app.state.lock:  # never in the middle of a job
            if not model_manager.is_model_loaded() or time.monotonic() - app.state.last_used < limit:
                return False
            name = _loaded()
            await asyncio.to_thread(model_manager.unload_model)
            logger.info(f"Unloaded {name}: idle for {limit:g} s (keep-alive)")
            return True

    async def _reap() -> None:
        interval = min(30.0, max(1.0, keep_alive / 4))
        while True:
            await asyncio.sleep(interval)
            try:
                await _expire()
            except Exception as e:
                logger.warning(f"Keep-alive check failed: {e}")

    @asynccontextmanager
    async def lifespan(app):
        reaper = asyncio.create_task(_reap()) if keep_alive is not None else None
        try:
            yield
        finally:
            if reaper:
                reaper.cancel()

    def _serving() -> Dict[str, Any]:
        """How this server takes models, for /api/models/running and health."""
        limit = app.state.keep_alive
        left = None
        if limit is not None and model_manager.is_model_loaded():
            left = round(max(0.0, limit - (time.monotonic() - app.state.last_used)), 1)
        return {"on_demand": app.state.pinned is None, "pinned": app.state.pinned,
                "keep_alive_s": limit, "unloads_in_s": left}

    app = FastAPI(
        title="OllamaDiffuser API",
        description="Image generation model management and inference API",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.pinned = pinned
    app.state.keep_alive = keep_alive
    app.state.lock = asyncio.Lock()  # one job, load or unload at a time
    app.state.last_used = time.monotonic()
    app.state.expire = _expire

    if settings.server.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # --- Root ---

    @app.get("/")
    async def root():
        return {
            "name": "OllamaDiffuser API",
            "version": "2.0.0",
            "status": "running",
            "endpoints": {
                "docs": "/docs",
                "health": "/api/health",
                "models": "/api/models",
                "generate": "/api/generate",
                "img2img": "/api/generate/img2img",
                "inpaint": "/api/generate/inpaint",
            },
        }

    # --- Health ---

    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy",
            "model_loaded": model_manager.is_model_loaded(),
            "current_model": model_manager.get_current_model(),
            **_serving(),
        }

    # --- Model management ---

    @app.get("/api/models")
    async def list_models():
        return {
            "available": model_manager.list_available_models(),
            "installed": model_manager.list_installed_models(),
            "current": model_manager.get_current_model(),
        }

    @app.get("/api/models/running")
    async def get_running_model():
        if model_manager.is_model_loaded():
            engine = model_manager.loaded_model
            return {
                "model": model_manager.get_current_model(),
                "info": engine.get_model_info(),
                "loaded": True,
                **_serving(),
            }
        return {"loaded": False, **_serving()}

    @app.get("/api/models/{model_name}")
    async def get_model_info(model_name: str):
        info = model_manager.get_model_info(model_name)
        if info is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return info

    @app.post("/api/models/pull")
    async def pull_model(model_name: str):
        success = await asyncio.to_thread(model_manager.pull_model, model_name)
        if success:
            return {"message": f"Model {model_name} downloaded successfully"}
        raise HTTPException(
            status_code=400, detail=f"Failed to download model {model_name}"
        )

    @app.post("/api/models/load")
    async def load_model(request: LoadModelRequest):
        async with app.state.lock:
            if request.model_name != _loaded():
                try:
                    await _switch_to(request.model_name)
                except HTTPException as e:
                    raise HTTPException(status_code=400 if e.status_code == 503 else e.status_code,
                                        detail=e.detail)
            if app.state.pinned:
                app.state.pinned = request.model_name  # whoever asked chose it
        return {"message": f"Model {request.model_name} loaded successfully"}

    @app.post("/api/models/unload")
    async def unload_model():
        async with app.state.lock:
            model_manager.unload_model()
        return {"message": "Model unloaded"}

    @app.delete("/api/models/{model_name}")
    async def remove_model(model_name: str):
        if model_manager.remove_model(model_name):
            return {"message": f"Model {model_name} removed"}
        raise HTTPException(
            status_code=400, detail=f"Failed to remove model {model_name}"
        )

    # --- Image generation (async via thread pool) ---

    @app.post("/api/generate")
    async def generate_image(request: GenerateRequest):
        """Generate image from text prompt"""
        steps = request.steps if request.steps is not None else request.num_inference_steps
        guidance = (
            request.cfg_scale
            if request.cfg_scale is not None
            else request.guidance_scale
        )

        async with app.state.lock:
            engine = await _engine_for(request.model)
            try:
                image = await asyncio.to_thread(
                    engine.generate_image,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    **({"scheduler": request.scheduler} if request.scheduler else {}),
                )
            except ValueError as e:
                # An input the model needs and did not get: the caller's to fix.
                raise HTTPException(status_code=400, detail=str(e))
            except RuntimeError as e:
                # The model refused or returned something that is not a picture;
                # its own sentence is the useful one.
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.error(f"Generation failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Image generation failed")
            finally:
                _touch()
        if request.response_format == "b64_json":
            return _image_to_json_response(image)
        return _image_to_response(image)

    @app.post("/api/generate/img2img")
    async def generate_img2img(
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality"),
        num_inference_steps: Optional[int] = Form(None),
        guidance_scale: Optional[float] = Form(None),
        seed: Optional[int] = Form(None),
        strength: Optional[float] = Form(None),
        scheduler: Optional[str] = Form(None),
        image: UploadFile = File(...),
        images: Optional[List[UploadFile]] = File(None),
        mask: Optional[UploadFile] = File(None),
        model: Optional[str] = Form(None),
    ):
        """Image-to-image generation.

        ``model``, as in /api/generate: loaded first by a `serve` server.

        ``mask`` says where to change the picture, for an editor that takes
        one (Qwen-Image-2.1's edit, FLUX.1 Fill); it is forwarded only when
        given.

        ``images`` are further reference pictures, for an editor that takes
        several (FLUX.2 klein: up to four). ``image`` stays the first — it
        sets the output size and is "image 1" in the prompt; the extras are
        "image 2", "image 3"… in upload order. It is what lets one picture
        carry *who* and another *where*: a face from a portrait, a place and
        an outfit from an earlier frame. Nothing is forwarded when nothing
        extra is uploaded, so a model that takes one picture never sees a
        keyword it does not know.

        ``strength`` defaults to nothing rather than to 0.75. A default here
        is not neutral: it is a fraction of the step count, and on a
        four-step distilled model 0.75 leaves **one** denoising step, which
        returns confetti. Measured on FLUX.2 klein 4B — 4 steps with no
        strength draws the picture, the same call at 0.75 does not. Models
        that want a default have their own.
        """
        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        references = [input_image]
        for extra in images or []:
            data = await extra.read()
            if data:  # an empty part is a form field left blank, not a picture
                references.append(Image.open(io.BytesIO(data)).convert("RGB"))
        mask_image = None
        if mask is not None:
            data = await mask.read()
            if data:
                mask_image = Image.open(io.BytesIO(data)).convert("RGB")

        async with app.state.lock:
            engine = await _engine_for(model)
            try:
                result = await asyncio.to_thread(
                    engine.generate_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=input_image.width,
                    height=input_image.height,
                    seed=seed,
                    image=input_image,
                    **({"images": references} if len(references) > 1 else {}),
                    **({"mask_image": mask_image} if mask_image is not None else {}),
                    **({"strength": strength} if strength is not None else {}),
                    **({"scheduler": scheduler} if scheduler else {}),
                )
            except ValueError as e:
                # An input the model needs and did not get: the caller's to fix.
                raise HTTPException(status_code=400, detail=str(e))
            except RuntimeError as e:
                # The model refused or returned something that is not a picture;
                # its own sentence is the useful one.
                logger.error(f"img2img failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.error(f"img2img failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Image-to-image generation failed")
            finally:
                _touch()
        return _image_to_response(result)

    # --- Video generation (LTX-2 on Apple Silicon) ---

    @app.post("/api/face/compare")
    async def face_compare(
        anchor: UploadFile = File(...),
        images: List[UploadFile] = File(...),
    ):
        """Is it still the same person? The largest face in each of ``images``
        against the largest face in ``anchor``, as cosine similarity.

        Needs no model loaded — it is not a diffusion model's job — so it
        answers from whichever server is up. Frames in, numbers out:
        ``{"results": [{"found": true, "width": 138, "similarity": 0.61}, …]}``,
        in upload order; a frame with no face in it says so instead of
        scoring. 503 with the way to fix it when the two model files have not
        been fetched; see ``core/utils/face_match.py`` for which and why.
        """
        from ..core.utils.face_match import FaceMatcher, FaceModelsMissing

        matcher = getattr(app.state, "face_matcher", None)
        if matcher is None:
            matcher = app.state.face_matcher = FaceMatcher()
        try:
            who = Image.open(io.BytesIO(await anchor.read())).convert("RGB")
            pictures = []
            for upload in images:
                data = await upload.read()
                if data:
                    pictures.append(Image.open(io.BytesIO(data)).convert("RGB"))
            return await asyncio.to_thread(matcher.compare, who, pictures)
        except FaceModelsMissing as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/generate/video")
    async def generate_video(
        prompt: str = Form(...),
        seconds: Optional[float] = Form(None),
        frames: Optional[int] = Form(None),
        width: Optional[int] = Form(None),
        height: Optional[int] = Form(None),
        seed: Optional[int] = Form(None),
        mode: Optional[str] = Form(None),
        steps: Optional[int] = Form(None),
        cfg_scale: Optional[float] = Form(None),
        low_ram: Optional[bool] = Form(None),
        enhance_prompt: Optional[bool] = Form(None),
        image: Optional[UploadFile] = File(None),
        audio: Optional[UploadFile] = File(None),
        control: Optional[UploadFile] = File(None),
        control_strength: Optional[float] = Form(None),
        lora: Optional[str] = Form(None),
        lora_strength: Optional[float] = Form(None),
        model: Optional[str] = Form(None),
    ):
        """Generate a video, and return the mp4.

        ``model``, as in /api/generate: loaded first by a `serve` server.

        Form rather than JSON because the interesting inputs are files: a
        reference image (image-to-video) and an audio track (audio-to-video,
        which is how a voice drives a face). Both are optional; with neither
        this is text-to-video.

        ``control`` is a third: a reference video the generation follows frame
        for frame (IC-LoRA — a pose skeleton, a depth pass, edges). With
        ``image`` as the first frame it is motion transfer: the still says who
        and where, the control video says how she moves. ``lora`` names the
        control LoRA (a path or a Hugging Face repo id); left out, it is
        Lightricks' union control.

        The mp4 comes back as the body, and ``X-Output-Path`` says where it
        also stays on disk — a four-second clip is a few MB, but a caller that
        wants to hand the file to something else should not have to write it
        out again.
        """
        # The uploads have to exist as files: the CLI takes paths.
        temp_paths: list[Path] = []

        async def _spill(upload: Optional[UploadFile], fallback: str) -> Optional[str]:
            # The suffix is worked out in here, not at the call site: an
            # omitted file is None, and `upload.filename` at the call site is
            # read before this function gets to check that.
            if upload is None:
                return None
            suffix = Path(upload.filename or "").suffix or fallback
            data = await upload.read()
            if not data:
                return None
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            handle.write(data)
            handle.close()
            temp_paths.append(Path(handle.name))
            return handle.name

        try:
            image_path = await _spill(image, ".png")
            audio_path = await _spill(audio, ".wav")
            control_path = await _spill(control, ".mp4")

            options = {
                "frames": frames, "width": width, "height": height, "seed": seed,
                "mode": mode, "steps": steps, "cfg_scale": cfg_scale,
                "low_ram": low_ram, "enhance_prompt": enhance_prompt,
                "image": image_path, "audio": audio_path,
                "control": control_path, "control_strength": control_strength, "lora": lora,
                "lora_strength": lora_strength,
            }
            options = {k: v for k, v in options.items() if v is not None}

            async with app.state.lock:
                engine = await _engine_for(model)
                try:
                    output = await asyncio.to_thread(
                        engine.generate_video, prompt=prompt, seconds=seconds, **options
                    )
                finally:
                    _touch()
            path = Path(output)
            return Response(
                content=path.read_bytes(),
                media_type="video/mp4",
                headers={"X-Output-Path": str(path)},
            )
        except HTTPException:
            raise
        except RuntimeError as e:
            # "not a video model", "not loaded", "ltx-2-mlx not found" — the
            # caller can act on every one of these, so they are not 500s.
            logger.error(f"Video generation refused: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Video generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Video generation failed")
        finally:
            for temp in temp_paths:
                temp.unlink(missing_ok=True)

    @app.post("/api/generate/inpaint")
    async def generate_inpaint(
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality"),
        num_inference_steps: Optional[int] = Form(None),
        guidance_scale: Optional[float] = Form(None),
        seed: Optional[int] = Form(None),
        strength: float = Form(0.75),
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        model: Optional[str] = Form(None),
    ):
        """Inpainting generation"""
        image_data = await image.read()
        mask_data = await mask.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        mask_image = Image.open(io.BytesIO(mask_data)).convert("RGB")

        async with app.state.lock:
            engine = await _engine_for(model)
            try:
                result = await asyncio.to_thread(
                    engine.generate_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=input_image.width,
                    height=input_image.height,
                    seed=seed,
                    image=input_image,
                    mask_image=mask_image,
                    strength=strength,
                )
            except Exception as e:
                logger.error(f"Inpainting failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Inpainting generation failed")
            finally:
                _touch()
        return _image_to_response(result)

    @app.post("/api/generate/controlnet")
    async def generate_controlnet(
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality"),
        num_inference_steps: Optional[int] = Form(None),
        guidance_scale: Optional[float] = Form(None),
        width: int = Form(1024),
        height: int = Form(1024),
        seed: Optional[int] = Form(None),
        controlnet_conditioning_scale: float = Form(1.0),
        control_guidance_start: float = Form(0.0),
        control_guidance_end: float = Form(1.0),
        control_image: Optional[UploadFile] = File(None),
        control_type: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
    ):
        """Generate image with ControlNet.

        ``control_type`` says what the control image is — canny, depth, pose,
        hed, mlsd — for union ControlNets that take several kinds
        (z-image-turbo-controlnet-mlx). Single-kind ControlNets ignore it.
        """
        control_pil = None
        if control_image:
            data = await control_image.read()
            control_pil = Image.open(io.BytesIO(data)).convert("RGB")

        async with app.state.lock:
            engine = await _engine_for(model)
            if not engine.is_controlnet_pipeline:
                raise HTTPException(
                    status_code=400, detail="Current model is not a ControlNet model"
                )
            try:
                result = await asyncio.to_thread(
                    engine.generate_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    seed=seed,
                    control_image=control_pil,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    **({"control_type": control_type} if control_type else {}),
                )
            except ValueError as e:
                # A missing input or an unknown control_type: the caller's to fix.
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"ControlNet generation failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="ControlNet generation failed")
            finally:
                _touch()
        return _image_to_response(result)

    # --- LoRA management ---

    @app.post("/api/lora/load")
    async def load_lora(request: LoadLoRARequest):
        async with app.state.lock:
            engine = _get_engine()
            try:
                success = engine.load_lora_runtime(
                    repo_id=request.repo_id,
                    weight_name=request.weight_name,
                    scale=request.scale,
                )
                if success:
                    return {"message": f"LoRA {request.lora_name} loaded (scale={request.scale})"}
                raise HTTPException(status_code=400, detail="Failed to load LoRA")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"LoRA load failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to load LoRA")

    @app.post("/api/lora/unload")
    async def unload_lora():
        async with app.state.lock:
            engine = _get_engine()
            if engine.unload_lora():
                return {"message": "LoRA unloaded"}
        raise HTTPException(status_code=400, detail="Failed to unload LoRA")

    @app.get("/api/lora/status")
    async def get_lora_status():
        if not model_manager.is_model_loaded():
            return {"loaded": False, "message": "No model loaded"}
        engine = model_manager.loaded_model
        if hasattr(engine, "current_lora") and engine.current_lora:
            return {"loaded": True, "info": engine.current_lora}
        return {"loaded": False, "info": None}

    # --- ControlNet preprocessors ---

    @app.post("/api/controlnet/initialize")
    async def initialize_controlnet():
        from ..core.utils.controlnet_preprocessors import controlnet_preprocessor

        success = controlnet_preprocessor.initialize(force=True)
        return {
            "success": success,
            "initialized": controlnet_preprocessor.is_initialized(),
            "available_types": controlnet_preprocessor.get_available_types(),
        }

    @app.get("/api/controlnet/preprocessors")
    async def get_controlnet_preprocessors():
        from ..core.utils.controlnet_preprocessors import controlnet_preprocessor

        return {
            "available_types": controlnet_preprocessor.get_available_types(),
            "available": controlnet_preprocessor.is_available(),
            "initialized": controlnet_preprocessor.is_initialized(),
        }

    @app.post("/api/controlnet/preprocess")
    async def preprocess_control_image(
        control_type: str = Form(...),
        image: UploadFile = File(...),
    ):
        from ..core.utils.controlnet_preprocessors import controlnet_preprocessor

        if not controlnet_preprocessor.is_initialized():
            if not controlnet_preprocessor.initialize():
                raise HTTPException(status_code=500, detail="Failed to init preprocessors")

        data = await image.read()
        input_image = Image.open(io.BytesIO(data)).convert("RGB")
        processed = controlnet_preprocessor.preprocess(input_image, control_type)
        return _image_to_response(processed)

    # --- Server management ---

    @app.post("/api/shutdown")
    async def shutdown_server():
        import os
        import signal

        model_manager.unload_model()

        def shutdown():
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.get_event_loop().call_later(0.5, shutdown)
        return {"message": "Server shutting down..."}

    return app


def run_server(host: str = None, port: int = None, pinned: Optional[str] = None,
               keep_alive: Optional[float] = None):
    """Start the API server. ``pinned`` and ``keep_alive``: see create_app."""
    host = host or settings.server.host
    port = port or settings.server.port

    app = create_app(pinned=pinned, keep_alive=keep_alive)
    how = f"for {pinned}" if pinned else "loading models on demand"
    if keep_alive is not None:
        how += f", unloading after {keep_alive:g} s idle"
    logger.info(f"Starting server: http://{host}:{port} ({how})")
    # So `ollamadiffuser ps` can find this server among others running.
    from ..core.utils import server_registry
    server_registry.register(host, port, model_manager.get_current_model()
                             if model_manager.is_model_loaded() else None)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        server_registry.unregister(port)


if __name__ == "__main__":
    run_server()
