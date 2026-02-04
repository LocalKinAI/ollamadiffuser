"""FLUX inference strategy"""

import logging
from typing import Optional

import torch
from PIL import Image

from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


class FluxStrategy(InferenceStrategy):
    """Strategy for FLUX models (FLUX.1-dev, FLUX.1-schnell)"""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            import diffusers

            self.device = device
            self.model_config = model_config

            # Resolve pipeline class from parameters, defaulting to FluxPipeline
            params = model_config.parameters or {}
            pipeline_class_name = params.get("pipeline_class", "FluxPipeline")
            pipeline_cls = getattr(diffusers, pipeline_class_name, None)
            if pipeline_cls is None:
                logger.error(
                    f"Pipeline class '{pipeline_class_name}' not found in diffusers. "
                    "You may need to upgrade: pip install --upgrade diffusers"
                )
                return False

            load_kwargs = {**SAFETY_DISABLED_KWARGS}

            if device == "cpu":
                load_kwargs["torch_dtype"] = torch.float32
                logger.warning("FLUX on CPU will be very slow for this 12B parameter model")
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16
                load_kwargs["use_safetensors"] = True

            self.pipeline = pipeline_cls.from_pretrained(
                model_config.path, **load_kwargs
            )

            if device in ("cuda", "mps") and hasattr(self.pipeline, "enable_model_cpu_offload"):
                # CPU offloading manages device placement itself — don't call _move_to_device
                self.pipeline.enable_model_cpu_offload(device=device)
                logger.info(f"Enabled CPU offloading for FLUX on {device}")
            else:
                self._move_to_device(device)
            self._apply_memory_optimizations()

            logger.info(f"FLUX model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FLUX model: {e}")
            return False

    @property
    def _is_schnell(self) -> bool:
        return self.model_config is not None and "schnell" in self.model_config.name.lower()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
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

        # Schnell-specific defaults
        if self._is_schnell:
            steps = num_inference_steps if num_inference_steps is not None else 4
            if steps > 4:
                logger.info(f"FLUX.1-schnell: reducing steps from {steps} to 4")
                steps = 4
            guidance = 0.0
            logger.info("FLUX.1-schnell: using 0.0 guidance (distilled model)")
        else:
            steps = num_inference_steps if num_inference_steps is not None else params.get("num_inference_steps", 20)
            guidance = guidance_scale if guidance_scale is not None else params.get("guidance_scale", 3.5)

        # CPU-specific caps
        if self.device == "cpu":
            if not self._is_schnell and steps > 20:
                steps = 20
                logger.info(f"Reduced steps to {steps} for CPU performance")
            if not self._is_schnell and guidance > 5.0:
                guidance = 5.0

        max_seq_len = kwargs.get("max_sequence_length", params.get("max_sequence_length", 512))

        generator, used_seed = self._make_generator(seed, self.device)

        gen_kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "max_sequence_length": max_seq_len,
            "generator": generator,
        }

        # Pass through params for Fill/Control pipeline variants
        for key in ("image", "mask_image", "strength", "control_image",
                     "controlnet_conditioning_scale"):
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]

        try:
            logger.info(f"Generating FLUX image: steps={steps}, guidance={guidance}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except RuntimeError as e:
            if "CUDA" in str(e) and self.device == "cpu":
                logger.warning("Device mismatch, retrying without generator")
                gen_kwargs.pop("generator", None)
                output = self.pipeline(**gen_kwargs)
                return output.images[0]
            raise
        except Exception as e:
            logger.error(f"FLUX generation failed: {e}")
            return self._create_error_image(str(e), prompt)
