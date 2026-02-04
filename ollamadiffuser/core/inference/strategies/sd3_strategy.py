"""SD 3.x inference strategy"""

import logging
from typing import Optional

import torch
from PIL import Image

from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


class SD3Strategy(InferenceStrategy):
    """Strategy for Stable Diffusion 3.x models"""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            from diffusers import StableDiffusion3Pipeline

            self.device = device
            self.model_config = model_config

            load_kwargs = {**SAFETY_DISABLED_KWARGS}
            if model_config.variant == "fp16" and device not in ("cpu", "mps"):
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["variant"] = "fp16"
            else:
                load_kwargs["torch_dtype"] = self._get_dtype(device)

            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_config.path, **load_kwargs
            )
            self._move_to_device(device)
            self._apply_memory_optimizations()

            logger.info(f"SD3 model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load SD3 model: {e}")
            return False

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
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        params = self.model_config.parameters or {}
        steps = num_inference_steps if num_inference_steps is not None else params.get("num_inference_steps", 28)
        guidance = guidance_scale if guidance_scale is not None else params.get("guidance_scale", 3.5)

        generator, used_seed = self._make_generator(seed, self.device)

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "generator": generator,
        }

        try:
            logger.info(f"Generating SD3 image: steps={steps}, guidance={guidance}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except Exception as e:
            logger.error(f"SD3 generation failed: {e}")
            return self._create_error_image(str(e), prompt)
