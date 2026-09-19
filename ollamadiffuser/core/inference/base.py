"""Base inference strategy for OllamaDiffuser"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from ..config.settings import ModelConfig

logger = logging.getLogger(__name__)


# Unified safety checker kwargs - use these in all from_pretrained calls
SAFETY_DISABLED_KWARGS = {
    "safety_checker": None,
    "requires_safety_checker": False,
    "feature_extractor": None,
}


class InferenceStrategy(ABC):
    """Abstract base class for all model inference strategies"""

    def __init__(self):
        self.pipeline = None
        self.model_config: Optional[ModelConfig] = None
        self.device: Optional[str] = None
        self.current_lora = None

    @abstractmethod
    def load(self, model_config: ModelConfig, device: str) -> bool:
        """Load the model pipeline"""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Generate an image"""
        pass

    def unload(self) -> None:
        """Unload model and free memory"""
        if self.pipeline:
            self.pipeline = self.pipeline.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del self.pipeline
            self.pipeline = None
        self.model_config = None
        self.current_lora = None
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def get_info(self) -> Dict[str, Any]:
        """Return model information"""
        if not self.model_config:
            return {}
        return {
            "name": self.model_config.name,
            "type": self.model_config.model_type,
            "device": self.device,
            "variant": self.model_config.variant,
            "parameters": self.model_config.parameters,
        }

    def _make_generator(self, seed: Optional[int], device: str) -> tuple:
        """Create a torch Generator with the given or random seed. Returns (generator, seed)."""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        if device == "cpu":
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        logger.info(f"Using seed: {seed}")
        return generator, seed

    def _get_dtype(self, device: str, prefer_bf16: bool = False) -> torch.dtype:
        """Get appropriate dtype for the given device"""
        if device == "cpu":
            return torch.float32
        if prefer_bf16:
            return torch.bfloat16
        return torch.float16

    def _move_to_device(self, device: str) -> bool:
        """Move pipeline to device with fallback to CPU"""
        try:
            self.pipeline = self.pipeline.to(device)
            logger.info(f"Pipeline moved to {device}")
            return True
        except Exception as e:
            logger.warning(f"Failed to move pipeline to {device}: {e}")
            if device != "cpu":
                logger.info("Falling back to CPU")
                self.device = "cpu"
                self.pipeline = self.pipeline.to("cpu")
                return True
            raise

    def _apply_memory_optimizations(self):
        """Apply common memory optimizations"""
        # Attention slicing splits attention into chunks to save VRAM.
        # Skip on MPS: unified memory makes it unnecessary, and it causes
        # NaN in float16 UNet output for some SDXL models (e.g. RealVisXL).
        if self.device != "mps" and hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
            logger.info("Enabled attention slicing")
        if hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
            logger.info("Enabled VAE tiling")
        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
            logger.info("Enabled VAE slicing")

    def load_lora_runtime(
        self, repo_id: str, weight_name: str = None, scale: float = 1.0
    ) -> bool:
        """Load LoRA weights at runtime"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        try:
            if weight_name:
                self.pipeline.load_lora_weights(repo_id, weight_name=weight_name)
            else:
                self.pipeline.load_lora_weights(repo_id)
            if hasattr(self.pipeline, "set_adapters") and scale != 1.0:
                self.pipeline.set_adapters(["default"], adapter_weights=[scale])
            self.current_lora = {
                "repo_id": repo_id,
                "weight_name": weight_name,
                "scale": scale,
                "loaded": True,
            }
            logger.info(f"LoRA loaded from {repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            return False

    def unload_lora(self) -> bool:
        """Unload LoRA weights"""
        if not self.pipeline:
            return False
        try:
            if hasattr(self.pipeline, "unload_lora_weights"):
                self.pipeline.unload_lora_weights()
                self.current_lora = None
                logger.info("LoRA unloaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")
            return False

    @staticmethod
    def looks_like_noise(image: Image.Image, threshold: float = 1.0) -> bool:
        """True when an image is garbage rather than a picture.

        A model can fail by returning latents it never denoised: a full-size,
        perfectly valid PNG of confetti. Every cheap check passes it — it
        decodes, it is the right size, it is megabytes — so the failure
        reaches the user as art.

        One number is not enough. Adjacent-pixel difference alone (measured
        on a 256px thumbnail) reads 36 for static and 4 for a portrait, but a
        busy photograph of a dog in grass reads 14.4 and a failure from
        FLUX.2 klein — driven at 28 steps when it wants 4 — reads 14.7. They
        overlap.

        What separates them is what survives being shrunk to 16×16. A
        photograph still has a composition down there; garbage turns to mush.
        So this compares high-frequency energy against low-frequency
        structure, and the ratio is unambiguous — measured over 45 real
        photographs and two failures:

            45 photographs (busiest: a shiba in grass)   0.06 – 0.38
            FLUX.2 klein at the wrong step count         2.86
            FLUX.1-Kontext on a seed that breaks it      6.60

        The threshold sits at 1.0, a factor of 2.6 clear of both sides.
        """
        try:
            grey = image.convert("L")
            high = grey.copy()
            high.thumbnail((256, 256))
            arr = np.asarray(high, dtype=np.float32)
            if arr.size < 64:
                return False
            dx = np.abs(np.diff(arr, axis=1)).mean()
            dy = np.abs(np.diff(arr, axis=0)).mean()
            detail = float((dx + dy) / 2)
            structure = float(np.asarray(
                grey.resize((16, 16), Image.LANCZOS), dtype=np.float32).std())
            return detail / max(structure, 0.01) > threshold
        except Exception:
            # A detector that cannot run must not fail a generation.
            return False

    @staticmethod
    def _sanitize_image(image: Image.Image) -> Image.Image:
        """Clamp NaN/Inf pixels to avoid 'invalid value encountered in cast' on MPS."""
        arr = np.array(image, dtype=np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            logger.warning("NaN/Inf detected in generated image — clamping to valid range")
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        return image

    def _create_error_image(self, error_msg: str, prompt: str) -> Image.Image:
        """Create an error placeholder image"""
        from PIL import ImageDraw, ImageFont

        img = Image.new("RGB", (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((10, 10), f"Error: {error_msg}", fill=(255, 0, 0), font=font)
        prompt_display = prompt[:50] + "..." if len(prompt) > 50 else prompt
        draw.text(
            (10, 30), f"Prompt: {prompt_display}", fill=(0, 0, 0), font=font
        )
        return img
