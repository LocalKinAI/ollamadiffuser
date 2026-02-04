"""SDXL inference strategy"""

import logging
from typing import Optional

import torch
from PIL import Image

from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


class SDXLStrategy(InferenceStrategy):
    """Strategy for Stable Diffusion XL models"""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            from diffusers import StableDiffusionXLPipeline

            self.device = device
            self.model_config = model_config

            load_kwargs = {**SAFETY_DISABLED_KWARGS}
            if model_config.variant == "fp16" and device not in ("cpu", "mps"):
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["variant"] = "fp16"
            elif device == "mps":
                load_kwargs["torch_dtype"] = torch.float32
            else:
                load_kwargs["torch_dtype"] = self._get_dtype(device)

            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_config.path, **load_kwargs
            )

            # Apply scheduler override if specified in parameters
            params = model_config.parameters or {}
            scheduler_class_name = params.get("scheduler_class")
            if scheduler_class_name:
                import diffusers
                scheduler_cls = getattr(diffusers, scheduler_class_name, None)
                if scheduler_cls is None:
                    logger.warning(f"Scheduler '{scheduler_class_name}' not found in diffusers, using default")
                else:
                    scheduler_kwargs = params.get("scheduler_kwargs", {})
                    self.pipeline.scheduler = scheduler_cls.from_config(
                        self.pipeline.scheduler.config, **scheduler_kwargs
                    )
                    logger.info(f"Applied scheduler override: {scheduler_class_name}")

            self._move_to_device(device)
            self._apply_memory_optimizations()

            logger.info(f"SDXL model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load SDXL model: {e}")
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
        image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
        strength: float = 0.75,
        **kwargs,
    ) -> Image.Image:
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        params = self.model_config.parameters or {}
        steps = num_inference_steps if num_inference_steps is not None else params.get("num_inference_steps", 50)
        guidance = guidance_scale if guidance_scale is not None else params.get("guidance_scale", 7.5)

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
            if image is not None and mask_image is not None:
                return self._inpaint(gen_kwargs, image, mask_image, strength)
            elif image is not None:
                return self._img2img(gen_kwargs, image, strength)

            logger.info(f"Generating SDXL image: steps={steps}, guidance={guidance}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except Exception as e:
            logger.error(f"SDXL generation failed: {e}")
            return self._create_error_image(str(e), prompt)

    def _img2img(self, gen_kwargs: dict, image: Image.Image, strength: float) -> Image.Image:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipe = StableDiffusionXLImg2ImgPipeline(**self.pipeline.components)
        pipe = pipe.to(self.device)
        gen_kwargs.pop("width", None)
        gen_kwargs.pop("height", None)
        gen_kwargs["image"] = image
        gen_kwargs["strength"] = strength
        output = pipe(**gen_kwargs)
        return output.images[0]

    def _inpaint(self, gen_kwargs: dict, image: Image.Image, mask_image: Image.Image, strength: float) -> Image.Image:
        from diffusers import StableDiffusionXLInpaintPipeline

        pipe = StableDiffusionXLInpaintPipeline(**self.pipeline.components)
        pipe = pipe.to(self.device)
        gen_kwargs.pop("width", None)
        gen_kwargs.pop("height", None)
        gen_kwargs["image"] = image
        gen_kwargs["mask_image"] = mask_image
        gen_kwargs["strength"] = strength
        output = pipe(**gen_kwargs)
        return output.images[0]
