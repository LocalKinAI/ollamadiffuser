"""OllamaDiffuser REST API Server"""

import asyncio
import base64
import io
import logging
from typing import Any, Dict, Optional

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


# --- App factory ---


def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="OllamaDiffuser API",
        description="Image generation model management and inference API",
        version="2.0.0",
    )

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
            }
        return {"loaded": False}

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
        success = await asyncio.to_thread(model_manager.load_model, request.model_name)
        if success:
            return {"message": f"Model {request.model_name} loaded successfully"}
        raise HTTPException(
            status_code=400, detail=f"Failed to load model {request.model_name}"
        )

    @app.post("/api/models/unload")
    async def unload_model():
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
        engine = _get_engine()

        steps = request.steps if request.steps is not None else request.num_inference_steps
        guidance = (
            request.cfg_scale
            if request.cfg_scale is not None
            else request.guidance_scale
        )

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
            )
            if request.response_format == "b64_json":
                return _image_to_json_response(image)
            return _image_to_response(image)
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Image generation failed")

    @app.post("/api/generate/img2img")
    async def generate_img2img(
        prompt: str = Form(...),
        negative_prompt: str = Form("low quality, bad anatomy, worst quality"),
        num_inference_steps: Optional[int] = Form(None),
        guidance_scale: Optional[float] = Form(None),
        seed: Optional[int] = Form(None),
        strength: float = Form(0.75),
        image: UploadFile = File(...),
    ):
        """Image-to-image generation"""
        engine = _get_engine()

        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

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
                strength=strength,
            )
            return _image_to_response(result)
        except Exception as e:
            logger.error(f"img2img failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Image-to-image generation failed")

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
    ):
        """Inpainting generation"""
        engine = _get_engine()

        image_data = await image.read()
        mask_data = await mask.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        mask_image = Image.open(io.BytesIO(mask_data)).convert("RGB")

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
            return _image_to_response(result)
        except Exception as e:
            logger.error(f"Inpainting failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Inpainting generation failed")

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
    ):
        """Generate image with ControlNet"""
        engine = _get_engine()

        if not engine.is_controlnet_pipeline:
            raise HTTPException(
                status_code=400, detail="Current model is not a ControlNet model"
            )

        control_pil = None
        if control_image:
            data = await control_image.read()
            control_pil = Image.open(io.BytesIO(data)).convert("RGB")

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
            )
            return _image_to_response(result)
        except Exception as e:
            logger.error(f"ControlNet generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="ControlNet generation failed")

    # --- LoRA management ---

    @app.post("/api/lora/load")
    async def load_lora(request: LoadLoRARequest):
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
        except Exception as e:
            logger.error(f"LoRA load failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load LoRA")

    @app.post("/api/lora/unload")
    async def unload_lora():
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


def run_server(host: str = None, port: int = None):
    """Start the API server"""
    host = host or settings.server.host
    port = port or settings.server.port

    app = create_app()
    logger.info(f"Starting server: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
