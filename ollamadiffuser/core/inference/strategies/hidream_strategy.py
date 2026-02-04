"""HiDream inference strategy"""

import logging
from typing import Optional

import torch
from PIL import Image

from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)

# Check availability
try:
    from diffusers import HiDreamImagePipeline
    HIDREAM_AVAILABLE = True
except ImportError:
    HIDREAM_AVAILABLE = False


class HiDreamStrategy(InferenceStrategy):
    """Strategy for HiDream models"""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        if not HIDREAM_AVAILABLE:
            logger.error("HiDreamImagePipeline not available. Install diffusers from source.")
            return False

        try:
            self.device = device
            self.model_config = model_config

            load_kwargs = {**SAFETY_DISABLED_KWARGS}
            if device == "cpu":
                load_kwargs["torch_dtype"] = torch.float32
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16

            self.pipeline = HiDreamImagePipeline.from_pretrained(
                model_config.path, **load_kwargs
            )

            if device in ("cuda", "mps") and hasattr(self.pipeline, "enable_model_cpu_offload"):
                # CPU offloading manages device placement itself — don't call _move_to_device
                self.pipeline.enable_model_cpu_offload(device=device)
            else:
                self._move_to_device(device)

            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()
            if hasattr(self.pipeline, "enable_vae_tiling"):
                self.pipeline.enable_vae_tiling()

            logger.info(f"HiDream model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load HiDream model: {e}")
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
        guidance = guidance_scale if guidance_scale is not None else params.get("guidance_scale", 5.0)
        max_seq_len = kwargs.get("max_sequence_length", params.get("max_sequence_length", 128))

        generator, used_seed = self._make_generator(seed, self.device)

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "max_sequence_length": max_seq_len,
            "generator": generator,
        }

        # Support multiple text encoder prompts
        for key in ("prompt_2", "prompt_3", "prompt_4"):
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]

        try:
            logger.info(f"Generating HiDream image: steps={steps}, guidance={guidance}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except Exception as e:
            logger.error(f"HiDream generation failed: {e}")
            return self._create_error_image(str(e), prompt)
