"""SD 1.5 inference strategy"""

import logging
from typing import Optional

import torch
from PIL import Image

from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


class SD15Strategy(InferenceStrategy):
    """Strategy for Stable Diffusion 1.5 models"""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            from diffusers import StableDiffusionPipeline

            self.device = device
            self.model_config = model_config

            load_kwargs = {**SAFETY_DISABLED_KWARGS}
            dtype = self._get_dtype(device)

            if model_config.variant == "fp16" and device not in ("cpu", "mps"):
                load_kwargs["torch_dtype"] = dtype
                load_kwargs["variant"] = "fp16"
            else:
                load_kwargs["torch_dtype"] = torch.float32

            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_config.path, **load_kwargs
            )
            self._move_to_device(device)
            self._apply_memory_optimizations()

            logger.info(f"SD 1.5 model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load SD 1.5 model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: int = 512,
        height: int = 512,
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

        # Clamp guidance on MPS (known hardware limitation)
        if self.device == "mps" and guidance > 6.0:
            logger.info(f"Clamping guidance_scale from {guidance} to 6.0 for MPS stability")
            guidance = 6.0

        # Warn about non-native resolutions for SD 1.5 (trained on 512x512)
        if width > 768 or height > 768:
            logger.warning(
                f"SD 1.5 was trained on 512x512. Using {width}x{height} may produce artifacts. "
                "Consider using 512x512 or 768x768 for best results."
            )

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
            # Handle img2img mode
            if image is not None and mask_image is not None:
                return self._inpaint(gen_kwargs, image, mask_image, strength)
            elif image is not None:
                return self._img2img(gen_kwargs, image, strength)

            logger.info(f"Generating SD 1.5 image: steps={steps}, guidance={guidance}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except Exception as e:
            logger.error(f"SD 1.5 generation failed: {e}")
            return self._create_error_image(str(e), prompt)

    def _img2img(self, gen_kwargs: dict, image: Image.Image, strength: float) -> Image.Image:
        """Run img2img generation"""
        from diffusers import StableDiffusionImg2ImgPipeline

        img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipeline.components)
        img2img_pipe = img2img_pipe.to(self.device)

        gen_kwargs.pop("width", None)
        gen_kwargs.pop("height", None)
        gen_kwargs["image"] = image
        gen_kwargs["strength"] = strength

        output = img2img_pipe(**gen_kwargs)
        return output.images[0]

    def _inpaint(self, gen_kwargs: dict, image: Image.Image, mask_image: Image.Image, strength: float) -> Image.Image:
        """Run inpainting generation"""
        from diffusers import StableDiffusionInpaintPipeline

        inpaint_pipe = StableDiffusionInpaintPipeline(**self.pipeline.components)
        inpaint_pipe = inpaint_pipe.to(self.device)

        gen_kwargs.pop("width", None)
        gen_kwargs.pop("height", None)
        gen_kwargs["image"] = image
        gen_kwargs["mask_image"] = mask_image
        gen_kwargs["strength"] = strength

        output = inpaint_pipe(**gen_kwargs)
        return output.images[0]
