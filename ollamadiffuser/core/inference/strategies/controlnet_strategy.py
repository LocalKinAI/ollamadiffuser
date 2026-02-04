"""ControlNet inference strategy"""

import logging
from typing import Optional, Union

import torch
from PIL import Image

from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


class ControlNetStrategy(InferenceStrategy):
    """Strategy for ControlNet models (SD 1.5 and SDXL based)"""

    def __init__(self):
        super().__init__()
        self.controlnet = None
        self.is_controlnet_pipeline = True

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetPipeline,
                StableDiffusionXLControlNetPipeline,
            )

            self.device = device
            self.model_config = model_config

            # Determine if SD15 or SDXL based
            is_sdxl = model_config.model_type == "controlnet_sdxl"
            pipeline_class = StableDiffusionXLControlNetPipeline if is_sdxl else StableDiffusionControlNetPipeline

            load_kwargs = {**SAFETY_DISABLED_KWARGS}
            if device in ("cpu", "mps"):
                load_kwargs["torch_dtype"] = torch.float32
            elif model_config.variant == "fp16":
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["variant"] = "fp16"
            else:
                load_kwargs["torch_dtype"] = self._get_dtype(device)

            # Load ControlNet model
            logger.info(f"Loading ControlNet model from: {model_config.path}")
            self.controlnet = ControlNetModel.from_pretrained(
                model_config.path,
                torch_dtype=load_kwargs.get("torch_dtype", torch.float32),
            )

            # Get base model path
            base_model_name = getattr(model_config, "base_model", None)
            if not base_model_name:
                from ...models.manager import model_manager
                model_info = model_manager.get_model_info(model_config.name)
                if model_info and "base_model" in model_info:
                    base_model_name = model_info["base_model"]
                else:
                    raise ValueError(f"No base model specified for ControlNet: {model_config.name}")

            from ...models.manager import model_manager
            if not model_manager.is_model_installed(base_model_name):
                raise ValueError(f"Base model '{base_model_name}' not installed")

            from ...config.settings import settings
            base_config = settings.models[base_model_name]

            # Load pipeline with controlnet
            self.pipeline = pipeline_class.from_pretrained(
                base_config.path,
                controlnet=self.controlnet,
                **load_kwargs,
            )

            self._move_to_device(device)
            self.controlnet = self.controlnet.to(self.device)
            self._apply_memory_optimizations()

            logger.info(f"ControlNet model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ControlNet model: {e}")
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
        control_image: Optional[Union[Image.Image, str]] = None,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        **kwargs,
    ) -> Image.Image:
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        if control_image is None:
            raise ValueError("ControlNet requires a control_image")

        params = self.model_config.parameters or {}
        steps = num_inference_steps if num_inference_steps is not None else params.get("num_inference_steps", 50)
        guidance = guidance_scale if guidance_scale is not None else params.get("guidance_scale", 7.5)

        # Prepare control image
        control_image = self._prepare_control_image(control_image, width, height)

        generator, used_seed = self._make_generator(seed, self.device)

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "image": control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "generator": generator,
        }

        try:
            logger.info(f"Generating ControlNet image: steps={steps}, guidance={guidance}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except Exception as e:
            logger.error(f"ControlNet generation failed: {e}")
            return self._create_error_image(str(e), prompt)

    def _prepare_control_image(
        self, control_image: Union[Image.Image, str], width: int, height: int
    ) -> Image.Image:
        """Prepare and preprocess control image"""
        from ...utils.controlnet_preprocessors import controlnet_preprocessor

        if isinstance(control_image, str):
            control_image = Image.open(control_image).convert("RGB")
        elif not isinstance(control_image, Image.Image):
            raise ValueError("control_image must be PIL Image or file path")

        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")

        # Get controlnet type from model info
        from ...models.manager import model_manager
        model_info = model_manager.get_model_info(self.model_config.name)
        cn_type = model_info.get("controlnet_type", "canny") if model_info else "canny"

        # Initialize preprocessor if needed
        if not controlnet_preprocessor.is_initialized():
            controlnet_preprocessor.initialize()

        processed = controlnet_preprocessor.preprocess(control_image, cn_type)
        processed = controlnet_preprocessor.resize_for_controlnet(processed, width, height)
        return processed

    def unload(self) -> None:
        if self.controlnet:
            del self.controlnet
            self.controlnet = None
        super().unload()
