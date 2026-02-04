"""
Inference Engine - Facade that delegates to model-specific strategies.

This replaces the former 1400+ line god class with a clean strategy pattern.
Each model type (SD1.5, SDXL, FLUX, SD3, ControlNet, Video, HiDream, GGUF)
has its own strategy class that handles loading and generation.
"""

import logging
from typing import Any, Dict, Optional, Union

from PIL import Image

from ..config.settings import ModelConfig
from .base import InferenceStrategy

logger = logging.getLogger(__name__)


def _get_strategy(model_type: str) -> InferenceStrategy:
    """Create the appropriate strategy for a model type."""
    if model_type == "sd15":
        from .strategies.sd15_strategy import SD15Strategy
        return SD15Strategy()
    elif model_type == "sdxl":
        from .strategies.sdxl_strategy import SDXLStrategy
        return SDXLStrategy()
    elif model_type == "flux":
        from .strategies.flux_strategy import FluxStrategy
        return FluxStrategy()
    elif model_type == "sd3":
        from .strategies.sd3_strategy import SD3Strategy
        return SD3Strategy()
    elif model_type in ("controlnet_sd15", "controlnet_sdxl"):
        from .strategies.controlnet_strategy import ControlNetStrategy
        return ControlNetStrategy()
    elif model_type == "video":
        from .strategies.video_strategy import VideoStrategy
        return VideoStrategy()
    elif model_type == "hidream":
        from .strategies.hidream_strategy import HiDreamStrategy
        return HiDreamStrategy()
    elif model_type == "gguf":
        from .strategies.gguf_strategy import GGUFStrategy
        return GGUFStrategy()
    elif model_type == "generic":
        from .strategies.generic_strategy import GenericPipelineStrategy
        return GenericPipelineStrategy()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _detect_device() -> str:
    """Automatically detect the best available device."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        logger.debug(f"CUDA device count: {torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    if device == "cpu":
        logger.warning("Using CPU - inference will be slower")
    return device


class InferenceEngine:
    """
    Facade engine that delegates to model-type-specific strategies.

    This class maintains backward compatibility with the previous monolithic
    InferenceEngine API while cleanly separating concerns into per-model strategies.
    """

    def __init__(self):
        self._strategy: Optional[InferenceStrategy] = None
        self.model_config: Optional[ModelConfig] = None
        self.device: Optional[str] = None

    # -- Backward-compatible properties --

    @property
    def pipeline(self):
        """Access the underlying pipeline (backward compat)."""
        return self._strategy.pipeline if self._strategy else None

    @property
    def current_lora(self):
        return self._strategy.current_lora if self._strategy else None

    @property
    def is_controlnet_pipeline(self) -> bool:
        return getattr(self._strategy, "is_controlnet_pipeline", False)

    # -- Core API --

    def load_model(self, model_config: ModelConfig) -> bool:
        """Load a model using the appropriate strategy."""
        try:
            if not model_config or not model_config.path:
                logger.error("Invalid model configuration")
                return False

            # Only check existence for local paths, not HuggingFace Hub IDs
            # Hub IDs look like "org/model-name", local paths start with / . or ~
            from pathlib import Path
            model_path = model_config.path
            is_local_path = model_path.startswith(("/", ".", "~")) or (len(model_path) > 1 and model_path[1] == ":")
            if is_local_path and not Path(model_path).expanduser().exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False

            # Detect GGUF models by variant
            model_type = model_config.model_type
            if model_config.variant and "gguf" in model_config.variant.lower():
                model_type = "gguf"

            self.device = _detect_device()
            self._strategy = _get_strategy(model_type)

            if self._strategy.load(model_config, self.device):
                self.model_config = model_config
                # Update device in case strategy fell back to CPU
                self.device = self._strategy.device
                logger.info(f"Model {model_config.name} loaded successfully via {type(self._strategy).__name__}")
                return True

            self._strategy = None
            return False

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._strategy = None
            return False

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        control_image: Optional[Union[Image.Image, str]] = None,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
        strength: float = 0.75,
        **kwargs,
    ) -> Image.Image:
        """Generate an image using the current strategy."""
        if not self._strategy:
            raise RuntimeError("No model loaded")

        # Build kwargs to pass through
        gen_kwargs = dict(kwargs)

        # Pass img2img / inpainting params
        if image is not None:
            gen_kwargs["image"] = image
        if mask_image is not None:
            gen_kwargs["mask_image"] = mask_image
        if image is not None or mask_image is not None:
            gen_kwargs["strength"] = strength

        # Pass ControlNet params
        if control_image is not None:
            gen_kwargs["control_image"] = control_image
            gen_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            gen_kwargs["control_guidance_start"] = control_guidance_start
            gen_kwargs["control_guidance_end"] = control_guidance_end

        return self._strategy.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
            **gen_kwargs,
        )

    def unload(self):
        """Unload the current model and free resources."""
        if self._strategy:
            self._strategy.unload()
            self._strategy = None
        self.model_config = None
        self.device = None
        logger.info("Engine unloaded")

    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._strategy is not None and self._strategy.is_loaded

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently loaded model."""
        if not self._strategy:
            return None
        info = self._strategy.get_info()
        info["strategy"] = type(self._strategy).__name__
        return info

    # -- LoRA support --

    def load_lora_runtime(
        self, repo_id: str, weight_name: str = None, scale: float = 1.0
    ) -> bool:
        """Load LoRA weights at runtime."""
        if not self._strategy:
            raise RuntimeError("No model loaded")
        return self._strategy.load_lora_runtime(repo_id, weight_name, scale)

    def unload_lora(self) -> bool:
        """Unload current LoRA weights."""
        if not self._strategy:
            return False
        return self._strategy.unload_lora()
