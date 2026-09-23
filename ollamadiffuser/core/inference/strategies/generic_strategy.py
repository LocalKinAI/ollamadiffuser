"""Generic pipeline inference strategy for any diffusers-compatible model."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..base import InferenceStrategy
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


def _mps_supports_bfloat16() -> bool:
    """Check if MPS backend supports bfloat16 (requires PyTorch 2.3+)."""
    try:
        t = torch.tensor([1.0], dtype=torch.bfloat16, device="mps")
        _ = t + t
        return True
    except (RuntimeError, TypeError):
        return False


# Dtype name -> torch dtype mapping
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class GenericPipelineStrategy(InferenceStrategy):
    """Strategy that dynamically loads any diffusers pipeline class.

    The pipeline class name is read from model_config.parameters["pipeline_class"].
    This allows adding new model types to the registry without writing new strategy code.
    """

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            import diffusers

            self.device = device
            self.model_config = model_config

            params = model_config.parameters or {}
            pipeline_class_name = params.get("pipeline_class")
            if not pipeline_class_name:
                logger.error("GenericPipelineStrategy requires 'pipeline_class' in model parameters")
                return False

            pipeline_cls = getattr(diffusers, pipeline_class_name, None)
            if pipeline_cls is None:
                logger.error(
                    f"Pipeline class '{pipeline_class_name}' not found in diffusers. "
                    "You may need to upgrade: pip install --upgrade diffusers"
                )
                return False

            # Resolve dtype from parameters or auto-detect
            dtype_name = params.get("torch_dtype")
            if dtype_name and dtype_name in _DTYPE_MAP:
                dtype = _DTYPE_MAP[dtype_name]
            elif device == "cpu":
                dtype = torch.float32
            elif device == "mps":
                # MPS default: float32 for numerical stability when model doesn't specify
                dtype = torch.float32
            else:
                dtype = torch.bfloat16

            # MPS bfloat16 support check: fall back to float16 if unsupported
            if device == "mps" and dtype == torch.bfloat16:
                if not _mps_supports_bfloat16():
                    logger.info("MPS does not support bfloat16, falling back to float16")
                    dtype = torch.float16

            load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
            if model_config.variant:
                load_kwargs["variant"] = model_config.variant

            # Some repos keep the diffusers tree in a subfolder beside their
            # single-file checkpoints (SeeSee21/Z-Anime: diffusers/).
            load_path = model_config.path
            if params.get("model_subdir"):
                load_path = str(Path(model_config.path) / params["model_subdir"])
            logger.info(f"Loading {pipeline_class_name} from {load_path} (dtype={dtype})")
            try:
                self.pipeline = pipeline_cls.from_pretrained(
                    load_path, **load_kwargs
                )
            except (OSError, ValueError):
                if "variant" in load_kwargs:
                    logger.info(f"No {model_config.variant} variant files found, loading without variant")
                    load_kwargs.pop("variant")
                    self.pipeline = pipeline_cls.from_pretrained(
                        load_path, **load_kwargs
                    )
                else:
                    raise

            # Device placement
            enable_offload = params.get("enable_cpu_offload", False)
            if enable_offload and device == "cuda":
                if hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                    self.pipeline.enable_sequential_cpu_offload(device=device)
                    logger.info(f"Enabled sequential CPU offloading on {device}")
                elif hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload(device=device)
                    logger.info(f"Enabled model CPU offloading on {device}")
                else:
                    self._move_to_device(device)
            else:
                # MPS: unified memory means CPU offload adds overhead without saving memory
                self._move_to_device(device)

            # VAE upcast for models that need it (e.g. Kolors on MPS)
            if params.get("vae_upcast_float32") and device == "mps":
                if hasattr(self.pipeline, "vae"):
                    self.pipeline.vae.to(dtype=torch.float32)
                    logger.info("Upcast VAE to float32 for MPS stability")

            self._apply_memory_optimizations()

            logger.info(f"{pipeline_class_name} model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {model_config.name}: {e}")
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
        guidance = guidance_scale if guidance_scale is not None else params.get("guidance_scale", 7.0)

        generator, used_seed = self._make_generator(seed, self.device)

        gen_kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "generator": generator,
        }

        # Include negative_prompt only if the pipeline supports it
        if params.get("supports_negative_prompt", True):
            gen_kwargs["negative_prompt"] = negative_prompt

        # Pass through image/mask/control params from kwargs
        for key in ("image", "mask_image", "strength", "control_image",
                     "controlnet_conditioning_scale", "control_guidance_start",
                     "control_guidance_end"):
            if key in kwargs:
                gen_kwargs[key] = kwargs[key]

        try:
            logger.info(
                f"Generating with {type(self.pipeline).__name__}: "
                f"steps={steps}, guidance={guidance}, seed={used_seed}"
            )
            output = self.pipeline(**gen_kwargs)
            image = output.images[0]
            return self._sanitize_image(image)
        except TypeError as e:
            # Some pipelines don't accept all standard params (e.g., width/height)
            # Retry without optional params
            logger.warning(f"Pipeline call failed: {e}. Retrying with minimal params.")
            for key in ("width", "height", "negative_prompt"):
                gen_kwargs.pop(key, None)
            output = self.pipeline(**gen_kwargs)
            image = output.images[0]
            return self._sanitize_image(image)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._create_error_image(str(e), prompt)

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
