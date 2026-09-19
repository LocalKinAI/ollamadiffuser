"""MLX inference strategy for Apple Silicon.

Routes through `mflux <https://github.com/filipstrand/mflux>`_ — a pure
MLX implementation of FLUX-family models (and a few others). On
M-series Macs MLX typically runs **2-3× faster** than the PyTorch+MPS
path that the other strategies in this package use, because MLX targets
the Metal/Neural Engine stack natively without the PyTorch wrapper.

This is an **Apple-Silicon-only strategy.** It refuses to load on
Linux/Windows or Intel Macs.

Registry entries opt in via::

    model_type: "mlx"
    parameters:
      mlx_variant: "flux1"       # currently: "flux1" (txt2img)
      mlx_model_name: "schnell"  # passed to mflux's ModelConfig.from_name():
                                 #   "schnell" | "dev" | "krea-dev"
      quantize: 8                # int | null. 4 or 8 bits via MLX quant.

      # Generation defaults (overridable per-call)
      num_inference_steps: 4
      guidance_scale: 0.0
      max_sequence_length: 256

Tracking issue: https://github.com/LocalKinAI/ollamadiffuser/issues/7
"""
from __future__ import annotations

import concurrent.futures
import logging
import platform
from pathlib import Path
import random
from typing import Optional

from PIL import Image

from ..base import InferenceStrategy
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Public constants
# --------------------------------------------------------------------------

# Variants this strategy can route. Adding a new mflux-supported family
# is a matter of (a) extending this dict, (b) plumbing the variant
# through ``_resolve_model_and_config``, and (c) listing any required
# input kwargs in ``_VARIANT_REQUIRED_INPUTS``.
SUPPORTED_MLX_VARIANTS = frozenset({
    "flux1",             # FLUX.1 schnell / dev / krea-dev (text-to-image)
    "flux1-kontext",     # FLUX.1 Kontext (image editing)
    "flux1-fill",        # FLUX.1 Fill (inpaint / outpaint / CatVTON try-on)
    "flux1-redux",       # FLUX.1 Redux (image-to-image variation)
    "flux1-depth",       # FLUX.1 Depth (depth-conditioned generation)
    "flux1-controlnet",  # FLUX.1 ControlNet (canny / upscaler)
    "flux2",             # FLUX.2 klein 4B/9B (text-to-image)
    "flux2-edit",        # FLUX.2 klein editing — several reference images at once
    "z_image",           # Z-Image / Z-Image-Turbo (text-to-image)
    "qwen-image",        # Qwen-Image / Qwen-Image-Edit
    # Families mflux added after our May 2026 line. Each one is a class and
    # an alias — see _ALIAS_ROUTED — because mflux resolves the config from
    # the alias itself, so there is nothing per-family left to write.
    "krea2",             # Krea 2 Turbo / Raw (text-to-image)
    "boogu",             # Boogu Image Turbo (4-step, photographic, EN/ZH text)
    "ernie-image",       # ERNIE-Image / -Turbo (Baidu)
    "lens",              # Lens Turbo (Microsoft, 4-step)
    "ideogram4",         # Ideogram 4 (JSON captions, typography)
    "fibo",              # FIBO / FIBO-lite (JSON prompts)
    "fibo-edit",         # FIBO-Edit / -rmbg (instruction editing)
    "seedvr2",           # SeedVR2 3B / 7B (upscaling)
})

# Families whose mflux config comes straight from the alias: variant →
# (module, class). ``ModelConfig.from_name`` resolves every alias in mflux's
# own table, so these need no per-family factory — only the class to build.
# All of them take ``(quantize=..., model_config=...)``, which is what
# ``load()`` already passes.
_ALIAS_ROUTED = {
    "krea2":       ("mflux.models.krea2.variants.txt2img.krea2", "Krea2"),
    "boogu":       ("mflux.models.boogu.variants.txt2img.boogu_image", "BooguImage"),
    "ernie-image": ("mflux.models.ernie_image.variants.txt2img.ernie_image", "ErnieImage"),
    "lens":        ("mflux.models.lens.variants.txt2img.lens_image", "LensImage"),
    "ideogram4":   ("mflux.models.ideogram4.variants.txt2img.ideogram4", "Ideogram4"),
    "fibo":        ("mflux.models.fibo.variants.txt2img.fibo", "FIBO"),
    "fibo-edit":   ("mflux.models.fibo.variants.edit.fibo_edit", "FIBOEdit"),
    "seedvr2":     ("mflux.models.seedvr2.variants.upscale.seedvr2", "SeedVR2"),
}

# Per-variant required-input kwargs (passed to ``generate()``). If any
# of these are absent, ``generate()`` raises ValueError early instead of
# letting mflux crash deep inside the pipeline.
_VARIANT_REQUIRED_INPUTS = {
    "flux1-kontext":    ["image"],
    # flux2-edit is deliberately absent: with no reference it is a
    # text-to-image model, which is the point of routing both jobs through
    # one set of weights.
    "flux1-fill":       ["image", "mask_image"],
    "flux1-redux":      ["redux_images"],
    "flux1-depth":      ["image"],
    "flux1-controlnet": ["control_image"],
    "fibo-edit":        ["image"],
    "seedvr2":          ["image"],
}

# mflux quantization values it actually accepts. None means "no quant".
_VALID_QUANTIZE = (None, 4, 8)


# --------------------------------------------------------------------------
# Apple-Silicon detection
# --------------------------------------------------------------------------

def is_apple_silicon() -> bool:
    """Return True iff we're running on macOS arm64."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def looks_like_model_dir(path: Optional[str]) -> bool:
    """True when ``path`` holds a downloaded model rather than nothing.

    Loose on purpose: mflux families differ in layout — a diffusers tree with
    `model_index.json`, a flat directory of safetensors — and the only thing
    every one of them has is weights and some json beside them.
    """
    if not path:
        return False
    folder = Path(path)
    if not folder.is_dir():
        return False
    for marker in ("model_index.json", "config.json", "transformer/config.json"):
        if (folder / marker).is_file():
            return True
    return any(folder.glob("*.safetensors")) or any(folder.glob("*/*.safetensors"))


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------

class MLXStrategy(InferenceStrategy):
    """Apple-Silicon-native inference via mflux.

    Covers every mflux family we register: the FLUX.1 variants, FLUX.2,
    Z-Image, Qwen-Image, and — since mflux grew them — Krea 2, Boogu,
    ERNIE-Image, Lens, Ideogram 4, FIBO, FIBO-Edit and SeedVR2. All of
    them arrive through the same ``mlx_variant`` parameter; see
    ``SUPPORTED_MLX_VARIANTS``.

    Note: mflux uses MLX arrays under the hood, not PyTorch tensors.
    The base class's ``unload()`` calls ``pipeline.to("cpu")`` which
    doesn't apply here, so we override it. ``load_lora_runtime`` from
    the base also assumes diffusers — mflux LoRAs are passed at
    construction time, so we override that too with a clear error.
    """

    def __init__(self) -> None:
        super().__init__()
        # Track the mflux model on a separate attribute so the base
        # class's `pipeline`-based checks still work for "loaded?" queries
        # — we store the same object in both ``self.pipeline`` and
        # ``self._mlx_model`` to keep is_loaded semantics intact while
        # still letting MLX-specific code path use the typed attribute.
        self._mlx_model = None
        # Cached for capability checks in generate() (e.g. Kontext needs image).
        self._variant: Optional[str] = None
        # Every MLX call runs on this one thread; see _mlx_call.
        self._runner: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # MLX's streams are per-thread: an array built on one thread cannot be
    # evaluated on another, and the API generates in a thread pool. Measured on
    # the box — model loaded on the main thread, POST /api/generate answered
    # with "There is no Stream(cpu, 0) in current thread" from mx.eval inside
    # mflux. So building the model and running it both go through one thread of
    # our own. The endpoint stays async; it waits on this thread's result.
    def _mlx_call(self, fn, *args, **kwargs):
        if self._runner is None:
            self._runner = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mlx"
            )
        return self._runner.submit(fn, *args, **kwargs).result()

    # ----- Loading ------------------------------------------------------

    def load(self, model_config: ModelConfig, device: str) -> bool:
        if not is_apple_silicon():
            logger.error(
                "MLXStrategy requires macOS on Apple Silicon (arm64). "
                f"Detected platform: {platform.system()}/{platform.machine()}. "
                "Use the FLUX / GenericPipeline strategies instead on this host."
            )
            return False

        params = model_config.parameters or {}
        variant = params.get("mlx_variant", "flux1")
        mlx_model_name = params.get("mlx_model_name")
        quantize = params.get("quantize")

        if variant not in SUPPORTED_MLX_VARIANTS:
            logger.error(
                f"Unknown mlx_variant={variant!r}. "
                f"Supported: {sorted(SUPPORTED_MLX_VARIANTS)}"
            )
            return False
        if not mlx_model_name:
            logger.error(
                "MLXStrategy requires parameters.mlx_model_name in the "
                "registry entry (e.g. 'schnell' or 'dev')."
            )
            return False
        if quantize not in _VALID_QUANTIZE:
            logger.error(
                f"parameters.quantize must be one of {_VALID_QUANTIZE}, "
                f"got {quantize!r}."
            )
            return False

        try:
            model_cls, mflux_config = self._resolve_model_and_config(
                variant, mlx_model_name
            )
        except ImportError as e:
            logger.error(
                "mflux is not installed. Install with: "
                "pip install 'ollamadiffuser[mlx]'  (or: pip install mflux). "
                f"Original: {e}"
            )
            return False
        except ValueError as e:
            logger.error(str(e))
            return False

        logger.info(
            f"Loading mflux {variant} model '{mlx_model_name}' "
            f"(quantize={quantize})..."
        )
        # Load what `pull` already downloaded. Without this, mflux resolves the
        # model from its own name and fetches the repo a second time into the
        # Hugging Face cache — measured on the box: a 36 GB pull sitting unused
        # while the server downloaded the same weights again on first load.
        local_dir = getattr(model_config, "path", None)
        local_dir = str(local_dir) if looks_like_model_dir(local_dir) else None
        try:
            if local_dir:
                logger.info(f"Using the downloaded weights at {local_dir}")
                self._mlx_model = self._mlx_call(
                    model_cls, quantize=quantize, model_config=mflux_config,
                    model_path=local_dir,
                )
            else:
                self._mlx_model = self._mlx_call(
                    model_cls, quantize=quantize, model_config=mflux_config
                )
        except Exception as e:
            if not local_dir:
                logger.error(f"Failed to load MLX model: {e}", exc_info=True)
                return False
            # A filtered download can be missing a file this family wants. The
            # hub still has it, so say so and go there rather than failing.
            logger.warning(
                f"Could not load from {local_dir} ({e}); falling back to the hub."
            )
            try:
                self._mlx_model = self._mlx_call(
                    model_cls, quantize=quantize, model_config=mflux_config
                )
            except Exception as e2:
                logger.error(f"Failed to load MLX model: {e2}", exc_info=True)
                return False

        # Cache the variant so generate() can branch on capabilities.
        self._variant = variant
        # Mirror onto self.pipeline so base class's is_loaded property works.
        self.pipeline = self._mlx_model
        self.model_config = model_config
        # MLX runs on unified memory; we report "mps" for compatibility with
        # caller code paths that branch on device.
        self.device = "mps"
        logger.info(
            f"MLX model '{model_config.name}' loaded "
            f"({variant} / {mlx_model_name} / quantize={quantize})"
        )
        return True

    @staticmethod
    def _resolve_model_and_config(variant: str, mlx_model_name: str):
        """Return ``(ModelClass, mflux_model_config)`` for a registry entry.

        All four mflux families share the constructor signature
        ``Cls(quantize=..., model_config=...)`` but the ``ModelConfig``
        comes from variant-specific factories, e.g.
        ``ModelConfig.flux2_klein_4b()``.

        ``mlx_model_name`` selects the specific config within a family:
          krea2         → "krea-2" | "krea-2-raw"
          boogu         → "boogu-image-turbo"
          ernie-image   → "ernie-image-turbo" | "ernie-image"
          lens          → "lens-turbo"
          ideogram4     → "ideogram4-fp8"
          fibo          → "fibo" | "fibo-lite"
          fibo-edit     → "fibo-edit" | "fibo-edit-rmbg"
          seedvr2       → "seedvr2-3b" | "seedvr2-7b"
          flux1         → "schnell" | "dev" | "krea-dev"
          flux1-kontext → "dev"  (Kontext is single-config in mflux today)
          flux2         → "klein-4b" | "klein-9b"
                          | "klein-base-4b" | "klein-base-9b"
          z_image       → "z-image" | "z-image-turbo"
          qwen-image    → "qwen-image" | "qwen-image-edit"

        Imports are local so this module is safe to import on non-MLX
        platforms.
        """
        if variant in _ALIAS_ROUTED:
            import importlib
            from mflux.models.common.config.model_config import ModelConfig
            module_path, class_name = _ALIAS_ROUTED[variant]
            module = importlib.import_module(module_path)
            model_cls = getattr(module, class_name)
            # mflux raises its own error listing every alias it knows, which
            # is a better message than one we would write here.
            return model_cls, ModelConfig.from_name(model_name=mlx_model_name, base_model=None)

        if variant == "flux1":
            from mflux.models.flux.variants.txt2img.flux import Flux1
            from mflux.models.common.config.model_config import ModelConfig
            mc = ModelConfig.from_name(model_name=mlx_model_name, base_model=None)
            return Flux1, mc

        if variant == "flux2-edit":
            from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
            from mflux.models.common.config.model_config import ModelConfig
            return Flux2KleinEdit, ModelConfig.from_name(mlx_model_name or "klein-4b")

        if variant == "flux1-kontext":
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
            from mflux.models.common.config.model_config import ModelConfig
            # mflux only ships dev_kontext() today; accept "dev" as the alias.
            if mlx_model_name not in ("dev", "kontext"):
                raise ValueError(
                    f"flux1-kontext: mlx_model_name must be 'dev' or 'kontext', "
                    f"got {mlx_model_name!r}"
                )
            return Flux1Kontext, ModelConfig.dev_kontext()

        if variant == "flux1-fill":
            from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
            from mflux.models.common.config.model_config import ModelConfig
            # Two factories: dev_fill (general) and dev_fill_catvton (try-on).
            factories = {
                "dev": ModelConfig.dev_fill,
                "fill": ModelConfig.dev_fill,
                "catvton": ModelConfig.dev_fill_catvton,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"flux1-fill: mlx_model_name must be one of {sorted(factories)}, "
                    f"got {mlx_model_name!r}"
                )
            return Flux1Fill, factories[mlx_model_name]()

        if variant == "flux1-redux":
            from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
            from mflux.models.common.config.model_config import ModelConfig
            if mlx_model_name not in ("dev", "redux"):
                raise ValueError(
                    f"flux1-redux: mlx_model_name must be 'dev' or 'redux', "
                    f"got {mlx_model_name!r}"
                )
            return Flux1Redux, ModelConfig.dev_redux()

        if variant == "flux1-depth":
            from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
            from mflux.models.common.config.model_config import ModelConfig
            if mlx_model_name not in ("dev", "depth"):
                raise ValueError(
                    f"flux1-depth: mlx_model_name must be 'dev' or 'depth', "
                    f"got {mlx_model_name!r}"
                )
            return Flux1Depth, ModelConfig.dev_depth()

        if variant == "flux1-controlnet":
            from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
            from mflux.models.common.config.model_config import ModelConfig
            factories = {
                "canny":             ModelConfig.dev_controlnet_canny,
                "canny-dev":         ModelConfig.dev_controlnet_canny,
                "canny-schnell":     ModelConfig.schnell_controlnet_canny,
                "upscaler":          ModelConfig.dev_controlnet_upscaler,
                "upscaler-dev":      ModelConfig.dev_controlnet_upscaler,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"flux1-controlnet: mlx_model_name must be one of "
                    f"{sorted(factories)}, got {mlx_model_name!r}"
                )
            return Flux1Controlnet, factories[mlx_model_name]()

        if variant == "flux2":
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
            from mflux.models.common.config.model_config import ModelConfig
            factories = {
                "klein-4b": ModelConfig.flux2_klein_4b,
                "klein-9b": ModelConfig.flux2_klein_9b,
                "klein-base-4b": ModelConfig.flux2_klein_base_4b,
                "klein-base-9b": ModelConfig.flux2_klein_base_9b,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"flux2: mlx_model_name must be one of {sorted(factories)}, "
                    f"got {mlx_model_name!r}"
                )
            return Flux2Klein, factories[mlx_model_name]()

        if variant == "z_image":
            from mflux.models.z_image.variants.z_image import ZImage
            from mflux.models.common.config.model_config import ModelConfig
            factories = {
                "z-image": ModelConfig.z_image,
                "z-image-turbo": ModelConfig.z_image_turbo,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"z_image: mlx_model_name must be one of {sorted(factories)}, "
                    f"got {mlx_model_name!r}"
                )
            return ZImage, factories[mlx_model_name]()

        if variant == "qwen-image":
            from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
            from mflux.models.common.config.model_config import ModelConfig
            factories = {
                "qwen-image": ModelConfig.qwen_image,
                "qwen-image-edit": ModelConfig.qwen_image_edit,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"qwen-image: mlx_model_name must be one of {sorted(factories)}, "
                    f"got {mlx_model_name!r}"
                )
            return QwenImage, factories[mlx_model_name]()

        raise ValueError(f"Unhandled mlx_variant: {variant}")

    # ----- Generation ---------------------------------------------------

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
        if self._mlx_model is None:
            raise RuntimeError("Model not loaded — call load() first")

        params = self.model_config.parameters if self.model_config else {}
        params = params or {}

        # Resolve generation params, allowing per-call override of registry defaults.
        steps = (
            num_inference_steps
            if num_inference_steps is not None
            else int(params.get("num_inference_steps", 4))
        )
        guidance = (
            guidance_scale
            if guidance_scale is not None
            else float(params.get("guidance_scale", 0.0))
        )

        # Seed (mflux requires an explicit int — no None allowed).
        used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        # mflux's variants accept different image-conditioning kwargs.
        # We accept the same convention from our existing strategies:
        #   kwargs["image"]         → image_path           (img2img, kontext, fill, depth)
        #   kwargs["mask_image"]    → masked_image_path    (fill / inpainting)
        #   kwargs["depth_image"]   → depth_image_path     (depth, optional)
        #   kwargs["control_image"] → controlnet_image_path (controlnet)
        #   kwargs["redux_images"]  → redux_image_paths    (redux, list)
        # Each may be a PIL Image, a path str, or (for redux) a list thereof.
        image_path           = self._materialize_image_path(kwargs.get("image"))
        masked_image_path    = self._materialize_image_path(kwargs.get("mask_image"))
        depth_image_path     = self._materialize_image_path(kwargs.get("depth_image"))
        controlnet_image_path = self._materialize_image_path(kwargs.get("control_image"))
        redux_image_paths    = self._materialize_image_path_list(kwargs.get("redux_images"))

        # Per-variant required-input enforcement. Fail loud here rather
        # than letting mflux crash mid-pipeline with a cryptic error.
        required = _VARIANT_REQUIRED_INPUTS.get(self._variant, [])
        provided = {
            "image":         image_path,
            "mask_image":    masked_image_path,
            "depth_image":   depth_image_path,
            "control_image": controlnet_image_path,
            "redux_images":  redux_image_paths,
        }
        missing = [k for k in required if not provided.get(k)]
        if missing:
            raise ValueError(
                f"MLX variant {self._variant!r} requires kwargs {missing}; "
                f"pass them to generate(), e.g. generate(prompt, image=<PIL|path>)."
            )

        # Negative prompt: mflux accepts None or str.
        neg = negative_prompt or None
        # Reference images, for the variants that take a list of them. Extra
        # ones can be passed as `images=[...]`; one as `image=` works too.
        variant_takes_list = self._variant in ("flux2-edit",)
        extra_paths = self._materialize_image_path_list(kwargs.get("images"))
        if variant_takes_list and extra_paths:
            image_path = image_path or extra_paths[0]

        # Build the kwargs we'd LIKE to pass; filter to what this variant's
        # generate_image() actually accepts (Flux2Klein has no negative_prompt
        # parameter, for instance; Flux1Fill has masked_image_path).
        candidate_kwargs = {
            "seed": int(used_seed),
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "height": int(height),
            "width": int(width),
            "guidance": float(guidance),
            "image_path": image_path,
            # flux2-edit takes several references rather than one, and a
            # caller with a single picture should not have to know that.
            "image_paths": (extra_paths or ([image_path] if image_path else None))
                           if variant_takes_list else None,
            "masked_image_path": masked_image_path,
            "depth_image_path": depth_image_path,
            "controlnet_image_path": controlnet_image_path,
            "redux_image_paths": redux_image_paths,
            "image_strength": kwargs.get("strength"),
            "controlnet_strength": kwargs.get("controlnet_strength"),
            "redux_image_strengths": kwargs.get("redux_image_strengths"),
            "negative_prompt": neg,
        }
        try:
            import inspect
            accepted = set(
                inspect.signature(self._mlx_model.generate_image).parameters.keys()
            )
        except (TypeError, ValueError):
            accepted = set(candidate_kwargs.keys())  # fall back to everything

        gen_kwargs = {}
        for key, val in candidate_kwargs.items():
            if key in accepted:
                gen_kwargs[key] = val
            elif val not in (None, ""):
                logger.debug(
                    f"MLX variant {self._variant!r} doesn't accept "
                    f"{key!r}={val!r}; dropping."
                )

        logger.info(
            f"MLX generating ({self._variant}): "
            f"steps={steps}, guidance={guidance}, seed={used_seed}, "
            f"size={width}x{height}"
        )
        try:
            generated = self._mlx_call(self._mlx_model.generate_image, **gen_kwargs)
            # generated is mflux.utils.generated_image.GeneratedImage; .image is PIL.
            image = self._sanitize_image(generated.image)
        except Exception as e:
            logger.error(f"MLX generation failed: {e}", exc_info=True)
            return self._create_error_image(str(e), prompt)

        # Some seeds come back as undenoised latents: a valid, full-size PNG
        # of coloured confetti. Measured on FLUX.1-Kontext int8 with one
        # image and one prompt — seed 1234 draws the portrait, seed
        # 473366517 draws static, and 28 steps or 7 makes no difference. The
        # failure is silent, so it is checked for rather than hoped about.
        if not self.looks_like_noise(image):
            return image
        if kwargs.get("seed") is None:
            retry_seed = int(used_seed) % 90_000 + 1_000
            logger.warning(
                f"MLX ({self._variant}) returned noise at seed {used_seed}; "
                f"one more try at {retry_seed}"
            )
            gen_kwargs["seed"] = retry_seed
            try:
                generated = self._mlx_call(self._mlx_model.generate_image, **gen_kwargs)
                image = self._sanitize_image(generated.image)
            except Exception as e:
                logger.error(f"MLX retry failed: {e}", exc_info=True)
                return self._create_error_image(str(e), prompt)
            if not self.looks_like_noise(image):
                return image
        raise RuntimeError(
            f"{self._variant} returned noise rather than a picture "
            f"(seed {gen_kwargs.get('seed')}). Try another seed, or fewer "
            f"quantisation bits — this is the model failing, not the request."
        )

    @staticmethod
    def _materialize_image_path(image_arg) -> Optional[str]:
        """Normalize an image kwarg to a path string mflux can read."""
        if image_arg is None:
            return None
        if isinstance(image_arg, str):
            return image_arg
        # PIL Image → save to a tempfile.
        if hasattr(image_arg, "save"):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            image_arg.save(tmp.name)
            return tmp.name
        raise TypeError(
            f"Unsupported image type for MLX img2img: {type(image_arg).__name__}"
        )

    @classmethod
    def _materialize_image_path_list(cls, images_arg) -> Optional[list]:
        """Normalize a list-of-images kwarg (PIL or paths) to list-of-paths."""
        if images_arg is None:
            return None
        if not isinstance(images_arg, (list, tuple)):
            # Allow scalar PIL/path too — wrap in a list.
            images_arg = [images_arg]
        return [cls._materialize_image_path(x) for x in images_arg if x is not None]

    # ----- Lifecycle ----------------------------------------------------

    def unload(self) -> None:
        """Release mflux model. Base class's torch-specific cleanup is skipped."""
        if self._mlx_model is not None:
            del self._mlx_model
            self._mlx_model = None
        if self._runner is not None:
            self._runner.shutdown(wait=False)
            self._runner = None
        self.pipeline = None
        self.model_config = None
        self.current_lora = None
        self._variant = None
        logger.info("MLX model unloaded")

    # ----- Unsupported on this backend ----------------------------------

    def load_lora_runtime(self, repo_id, weight_name=None, scale=1.0) -> bool:
        """LoRA at runtime is not yet wired for the MLX backend.

        mflux expects ``lora_paths`` and ``lora_scales`` at construction
        time, not at runtime. Supporting hot-swap would require
        reconstructing the Flux1 instance — deferred until users ask.
        """
        logger.error(
            "Runtime LoRA loading is not supported on the MLX backend yet. "
            "mflux takes lora_paths/lora_scales at construction time. "
            "Track via https://github.com/LocalKinAI/ollamadiffuser/issues/7"
        )
        return False

    def unload_lora(self) -> bool:
        return False

    def get_info(self):
        info = super().get_info()
        info["backend"] = "mlx"
        info["mflux_variant"] = (
            (self.model_config.parameters or {}).get("mlx_variant")
            if self.model_config
            else None
        )
        return info
