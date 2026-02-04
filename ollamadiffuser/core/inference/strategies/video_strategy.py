"""Video (AnimateDiff) inference strategy"""

import logging
from typing import Optional

import torch
from PIL import Image

from ..base import InferenceStrategy, SAFETY_DISABLED_KWARGS
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


class VideoStrategy(InferenceStrategy):
    """Strategy for AnimateDiff video models"""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

            self.device = device
            self.model_config = model_config

            load_kwargs = {**SAFETY_DISABLED_KWARGS}
            dtype = torch.float16 if device != "cpu" else torch.float32
            load_kwargs["torch_dtype"] = dtype

            # Load motion adapter
            adapter_path = getattr(model_config, "motion_adapter_path", None)
            if not adapter_path:
                adapter_path = "guoyww/animatediff-motion-adapter-v1-5-2"
            motion_adapter = MotionAdapter.from_pretrained(adapter_path, torch_dtype=dtype)
            load_kwargs["motion_adapter"] = motion_adapter

            self.pipeline = AnimateDiffPipeline.from_pretrained(
                model_config.path, **load_kwargs
            )

            # Configure DDIM scheduler for AnimateDiff
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config,
                clip_sample=False,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
            )

            if device in ("cuda", "mps") and hasattr(self.pipeline, "enable_model_cpu_offload"):
                # CPU offloading manages device placement itself — don't call _move_to_device
                self.pipeline.enable_model_cpu_offload(device=device)
            else:
                self._move_to_device(device)

            if hasattr(self.pipeline, "enable_vae_slicing"):
                self.pipeline.enable_vae_slicing()

            logger.info(f"Video model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
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
        num_frames: int = 16,
        **kwargs,
    ) -> Image.Image:
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        params = self.model_config.parameters or {}
        steps = min(num_inference_steps if num_inference_steps is not None else params.get("num_inference_steps", 25), 25)
        guidance = min(
            guidance_scale if guidance_scale is not None else params.get("guidance_scale", 7.5),
            7.5,
        )

        generator, used_seed = self._make_generator(seed, self.device)

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "generator": generator,
        }

        try:
            logger.info(f"Generating video: {num_frames} frames, steps={steps}, seed={used_seed}")
            output = self.pipeline(**gen_kwargs)

            if hasattr(output, "frames") and len(output.frames) > 0:
                return output.frames[0]
            return output.images[0]
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return self._create_error_image(str(e), prompt)
