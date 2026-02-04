"""GGUF quantized model inference strategy"""

import logging
import random
from typing import Any, Dict, Optional

from PIL import Image

from ..base import InferenceStrategy
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)

try:
    from ...models.gguf_loader import gguf_loader, GGUF_AVAILABLE
except ImportError:
    GGUF_AVAILABLE = False
    gguf_loader = None


class GGUFStrategy(InferenceStrategy):
    """Strategy for GGUF quantized models"""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        if not GGUF_AVAILABLE:
            logger.error("GGUF support not available. Install with: pip install stable-diffusion-cpp-python gguf")
            return False

        try:
            self.device = device
            self.model_config = model_config

            config_dict = {
                "name": model_config.name,
                "path": model_config.path,
                "variant": model_config.variant,
                "model_type": model_config.model_type,
                "parameters": model_config.parameters,
            }

            if gguf_loader.load_model(config_dict):
                self.pipeline = None  # GGUF uses its own loader
                logger.info(f"GGUF model {model_config.name} loaded")
                return True

            logger.error(f"Failed to load GGUF model: {model_config.name}")
            return False
        except Exception as e:
            logger.error(f"GGUF load error: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return GGUF_AVAILABLE and gguf_loader is not None and gguf_loader.is_loaded()

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
        if not self.is_loaded:
            raise RuntimeError("GGUF model not loaded")

        params = self.model_config.parameters or {}
        steps = num_inference_steps if num_inference_steps is not None else params.get("num_inference_steps", 20)
        guidance = guidance_scale if guidance_scale is not None else params.get("guidance_scale", 7.5)

        # Resolve seed (GGUF uses integer seed, not torch.Generator)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using seed: {seed}")

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "seed": seed,
            **kwargs,
        }

        try:
            logger.info(f"Generating GGUF image: steps={steps}, guidance={guidance}, seed={seed}")
            image = gguf_loader.generate_image(**gen_kwargs)
            if image is None:
                return self._create_error_image("GGUF generation returned None", prompt)
            return image
        except Exception as e:
            logger.error(f"GGUF generation failed: {e}")
            return self._create_error_image(str(e), prompt)

    def unload(self) -> None:
        if GGUF_AVAILABLE and gguf_loader is not None and gguf_loader.is_loaded():
            gguf_loader.unload_model()
        self.model_config = None
        self.current_lora = None
        logger.info("GGUF model unloaded")

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        if GGUF_AVAILABLE and gguf_loader is not None:
            info.update(gguf_loader.get_model_info())
        info["is_gguf"] = True
        info["gguf_available"] = GGUF_AVAILABLE
        return info
