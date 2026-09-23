"""Video through diffusers — Wan and LTX-2 on NVIDIA (and wherever torch runs).

The Apple Silicon video path is ``ltx_video_strategy``: LTX-2 through
ltx-2-mlx, a separate MLX runtime. Everywhere else video goes through the
diffusers pipelines themselves — ``WanPipeline`` / ``WanImageToVideoPipeline``
and ``LTX2Pipeline`` / ``LTX2ImageToVideoPipeline`` — which is what this
strategy loads.

Registry entries opt in via::

    model_type: diffusers-video
    parameters:
      pipeline_class: WanPipeline               # text-to-video
      i2v_pipeline_class: WanImageToVideoPipeline  # optional, same weights
      frame_multiple: 4       # num_frames must be multiple*k + 1 (Wan 4, LTX-2 8)
      frame_rate: 24          # what the model was trained at; the mp4's rate
      num_frames: 121         # default length
      width: 1280
      height: 704
      num_inference_steps: 50
      guidance_scale: 5.0
      torch_dtype: bfloat16
      vae_dtype: float32      # optional: Wan decodes best with an fp32 VAE
      sigmas: ltx2-distilled  # optional: the fixed schedule a distilled LTX-2 needs
      enable_cpu_offload: true

``generate_video`` has the same shape as the LTX-2 MLX strategy's, so
``/api/generate/video`` drives both: it returns the path of the mp4.
"""
from __future__ import annotations

import inspect
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ..base import InferenceStrategy
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# LTX-2 MLX features the API also accepts. Dropping one would hand back a
# video that ignored half the request, so asking for one here is an error.
_MLX_ONLY = ("audio", "control", "lora")
# LTX-2 MLX knobs with no counterpart in a diffusers pipeline. Harmless to
# ignore: they choose how the MLX runtime gets there, not what comes out.
_IGNORED = ("mode", "low_ram", "tile_frames", "tile_spatial", "stage1_steps",
            "stage2_steps", "stg_scale", "quiet", "timeout")


def frames_for_seconds(seconds: float, fps: float, multiple: int) -> int:
    """The legal frame count nearest ``seconds`` at ``fps``: ``multiple*k + 1``."""
    k = max(1, round(seconds * fps / multiple))
    return k * multiple + 1


def check_frames(frames: int, multiple: int) -> int:
    frames = int(frames)
    if frames < multiple + 1 or (frames - 1) % multiple:
        k = max(1, round((frames - 1) / multiple))
        raise ValueError(
            f"frames must be {multiple}k+1 (the VAE compresses time {multiple}×); "
            f"{frames} is not — nearest legal is {k * multiple + 1}"
        )
    return frames


class DiffusersVideoStrategy(InferenceStrategy):
    """Text- and image-to-video through a diffusers video pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.defaults: dict = {}
        self._i2v = None

    # ---------------------------------------------------------------- load

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            import diffusers
        except ImportError as e:
            logger.error(f"diffusers is not installed: {e}")
            return False

        params = dict(model_config.parameters or {})
        name = params.get("pipeline_class")
        pipeline_cls = getattr(diffusers, name, None) if name else None
        if pipeline_cls is None:
            logger.error(
                f"{name!r} is not in diffusers {diffusers.__version__}. "
                "Upgrade it: pip install --upgrade diffusers"
            )
            return False

        if device == "mps" and name.startswith("LTX2"):
            # Measured on an M3 Ultra with diffusers 0.40: LTX-2's text
            # connectors build a float64 tensor, which the Apple GPU cannot
            # hold, and the first generation dies there.
            logger.error(
                f"{name} does not run on the Apple GPU (diffusers' LTX-2 uses float64). "
                "On a Mac, use the ltx-2.3-mlx-* / ltx-2.5-mlx-* entries."
            )
            return False

        dtype = _DTYPES.get(params.get("torch_dtype", "bfloat16"), torch.bfloat16)
        if device == "cpu":
            dtype = torch.float32
        elif device == "mps" and dtype == torch.bfloat16:
            from .generic_strategy import _mps_supports_bfloat16
            if not _mps_supports_bfloat16():
                dtype = torch.float16

        path = model_config.path
        if params.get("model_subdir"):
            path = str(Path(path) / params["model_subdir"])

        try:
            logger.info(f"Loading {name} from {path} (dtype={dtype})")
            pipe = pipeline_cls.from_pretrained(path, torch_dtype=dtype)
            vae_dtype = _DTYPES.get(params.get("vae_dtype", ""))
            if vae_dtype is not None and getattr(pipe, "vae", None) is not None:
                pipe.vae.to(dtype=vae_dtype)
            if device == "cuda" and params.get("enable_cpu_offload"):
                pipe.enable_model_cpu_offload()
                logger.info("Model CPU offload on: each component visits the GPU in turn")
            else:
                pipe.to(device)
            # A video VAE decodes every frame at once; tiling keeps that from
            # being the step that runs out of memory.
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
        except Exception as e:
            logger.error(f"Failed to load {model_config.name}: {e}", exc_info=True)
            return False

        self.pipeline = pipe
        self.defaults = params
        self.model_config = model_config
        self.device = device
        self._i2v = None
        logger.info(f"{name} video model {model_config.name} loaded on {device}")
        return True

    def unload(self) -> None:
        # Not the base class's pipeline.to("cpu"): with CPU offload on, the
        # components are already spread across devices by accelerate hooks.
        import gc
        self._i2v = None
        if self.pipeline is not None:
            if hasattr(self.pipeline, "remove_all_hooks"):
                self.pipeline.remove_all_hooks()
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_config = None
        self.current_lora = None
        logger.info("Video model unloaded")

    def _image_pipeline(self):
        """The image-to-video pipeline, built from the loaded one's components.

        ``from_pipe`` shares the weights, so asking for a first frame costs no
        second copy of the model.
        """
        name = self.defaults.get("i2v_pipeline_class")
        if not name:
            return None
        if self._i2v is None:
            import diffusers
            cls = getattr(diffusers, name, None)
            if cls is None:
                raise RuntimeError(f"{name} is not in diffusers {diffusers.__version__}")
            self._i2v = cls.from_pipe(self.pipeline)
        return self._i2v

    # ------------------------------------------------------------ generate

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        raise RuntimeError(
            "This is a video model — call generate_video() (or POST "
            "/api/generate/video) instead of the image path."
        )

    def generate_video(
        self,
        prompt: str,
        output: Optional[str] = None,
        *,
        seconds: Optional[float] = None,
        **kwargs,
    ) -> Path:
        """Generate one video and return the mp4 it was written to.

        Takes what ``/api/generate/video`` sends: ``frames`` or ``seconds``,
        ``width``, ``height``, ``steps``, ``cfg_scale``, ``seed``, ``image``
        (a path, for the first frame), ``frame_rate``, ``negative_prompt``.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded — call load() first")
        name = getattr(self.model_config, "name", "this model")
        for key in _MLX_ONLY:
            if kwargs.get(key):
                raise ValueError(
                    f"{key} is an LTX-2 MLX feature; {name} runs through diffusers and "
                    f"takes a prompt, an image, a length, a size, steps, cfg_scale and a seed"
                )
        for key in _IGNORED:
            if kwargs.get(key) is not None:
                logger.info(f"{key}={kwargs[key]!r} has no meaning for {name}; ignored")

        d = self.defaults
        fps = float(kwargs.get("frame_rate") or d.get("frame_rate") or 24)
        multiple = int(d.get("frame_multiple", 4))
        frames = kwargs.get("frames")
        if frames is None and seconds:
            frames = frames_for_seconds(seconds, fps, multiple)
        if frames is None:
            frames = d.get("num_frames")
        if frames is not None:
            frames = check_frames(frames, multiple)

        def pick(key, default_key):
            value = kwargs.get(key)
            return value if value is not None else d.get(default_key)

        pipe = self.pipeline
        image = kwargs.get("image")
        if image is not None:
            pipe = self._image_pipeline()
            if pipe is None:
                raise ValueError(f"{name} makes video from text only; drop the image")
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")

        generator, used_seed = self._make_generator(kwargs.get("seed"), "cpu")
        call = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt") or d.get("negative_prompt"),
            "image": image,
            "num_frames": frames,
            "width": pick("width", "width"),
            "height": pick("height", "height"),
            "num_inference_steps": pick("steps", "num_inference_steps"),
            "guidance_scale": pick("cfg_scale", "guidance_scale"),
            "frame_rate": fps,
            "generator": generator,
            "output_type": "np",
        }
        if d.get("sigmas") == "ltx2-distilled":
            # The distilled LTX-2 checkpoint was trained on one fixed schedule
            # and its card says to pass it every time.
            from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
            call["sigmas"] = list(DISTILLED_SIGMA_VALUES)
        if kwargs.get("enhance_prompt"):
            call["enable_prompt_enhancement"] = True

        accepted = set(inspect.signature(pipe.__call__).parameters)
        call = {k: v for k, v in call.items() if v is not None and k in accepted}

        logger.info(
            f"{type(pipe).__name__}: {call.get('num_frames')} frames at "
            f"{call.get('width')}x{call.get('height')}, {call.get('num_inference_steps')} steps, "
            f"guidance {call.get('guidance_scale')}, seed {used_seed}"
        )
        out = pipe(**call)
        video = out.frames[0]
        audio = getattr(out, "audio", None)
        if audio is not None:
            audio = audio[0]

        target = Path(output) if output else Path(
            tempfile.gettempdir()) / f"{name}-{int(time.time())}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        self._write(video, target, fps, audio)
        return target

    def _write(self, video, target: Path, fps: float, audio=None) -> None:
        """H.264 mp4, with LTX-2's soundtrack when it made one."""
        try:
            from diffusers.utils import encode_video
        except ImportError:
            encode_video = None
        if encode_video is not None:
            extra = {}
            if audio is not None:
                audio = audio.float().cpu() if hasattr(audio, "float") else audio
                vocoder = getattr(self.pipeline, "vocoder", None)
                rate = getattr(getattr(vocoder, "config", None), "output_sampling_rate", None)
                extra = {"audio": audio, "audio_sample_rate": rate or 24000}
            try:
                encode_video(video, fps=int(round(fps)), output_path=str(target), **extra)
                return
            except ImportError as e:
                raise RuntimeError(f"Writing the mp4 needs PyAV: pip install av ({e})") from e
        # diffusers before encode_video moved to diffusers.utils: no sound.
        if audio is not None:
            logger.warning("This diffusers cannot mux audio; writing the video without its sound")
        from diffusers.utils import export_to_video
        export_to_video(list(video), str(target), fps=int(round(fps)))

    def get_info(self):
        info = super().get_info()
        info["backend"] = "diffusers-video"
        return info
