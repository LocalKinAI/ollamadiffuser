"""LTX-2 video inference on Apple Silicon, via the ltx-2-mlx CLI.

Why a subprocess when every other strategy imports its pipeline
-----------------------------------------------------------------
`ltx-2-mlx <https://github.com/dgrauet/ltx-2-mlx>`_ is a pure-MLX port of
LTX-2 (2.3 and 2.5) — text-to-video with 48 kHz stereo audio,
image-to-video, audio-to-video, keyframe interpolation, retake/extend, and
block streaming plus modality tiling that put HD generation on a Mac Studio
instead of an A100. It is also a three-package monorepo that is not on PyPI,
installs through `uv sync`, and downloads its own weight packs. Importing it
into our process would mean pinning three packages we do not build, on a
machine that may not have them, to reach an API that is not the one its
author supports. Its CLI is that API. So this strategy builds an argv and
runs it.

What that buys, besides not vendoring somebody's monorepo: the whole thing is
testable without a single weight. ``build_argv`` is a pure function, and the
tests below it cover every mode, every memory flag and the frame-count rule
(``8k + 1``, from the VAE's 8× temporal compression) that otherwise fails a
minute into a generation.

The models this routes are registered as ``model_type: ltx-video-mlx``:

    ltx-2.3-mlx-q8   int8, ~21 GB on disk, 32 GB+ RAM   (the CLI's own default)
    ltx-2.3-mlx-q4   int4, ~12 GB, 16 GB+
    ltx-2.3-mlx      bf16, ~42 GB, 64 GB+
    ltx-2.5-*        the same three for LTX-2.5 (gated on Hugging Face)
"""

from __future__ import annotations

import logging
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from ..base import InferenceStrategy
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


# The CLI's own default pack, so a registry entry that names none still works.
DEFAULT_PACK = "dgrauet/ltx-2.3-mlx-q8"

# Generation modes, as the CLI spells them. "distilled" is the fastest,
# "two-stage" is upstream's recommended production default, "two-stages-hq"
# adds the res_2s sampler, "one-stage" is dev + CFG at full resolution.
MODES = ("distilled", "one-stage", "two-stage", "two-stages-hq")

# LTX-2.3 was trained at 24 fps; the packs' frame counts are quoted against it.
TRAINED_FPS = 24

# What a registry entry (or a caller) may set, named as build_argv takes it.
# Spelled out rather than introspected: a typo in a registry entry should be
# ignored loudly in one place, not silently forwarded into somebody's argv.
ARGV_OPTIONS = frozenset({
    "pack", "mode", "frames", "width", "height", "seed", "steps",
    "cfg_scale", "stg_scale", "stage1_steps", "stage2_steps",
    "image", "audio", "frame_rate", "audio_start", "auto_duration",
    "enhance_prompt", "low_ram", "tile_frames", "tile_spatial", "quiet",
})

# The registry's shared vocabulary, in video terms. An entry may write either.
ARGV_ALIASES = {
    "num_inference_steps": "steps",
    "guidance_scale": "cfg_scale",
}


def argv_options(source: dict) -> dict:
    """The options in ``source`` that build_argv understands, aliases applied."""
    out = {}
    for key, value in (source or {}).items():
        key = ARGV_ALIASES.get(key, key)
        if key in ARGV_OPTIONS and value is not None:
            out[key] = value
    return out


def is_apple_silicon() -> bool:
    """Return True iff we're running on macOS arm64."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def find_ltx_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Locate the ``ltx-2-mlx`` executable.

    In order: what the caller passed, ``$LTX2MLX_BIN``, PATH, then the two
    places a ``uv sync`` checkout puts it — because the documented install is
    a git clone, so the binary usually is not on PATH at all.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    if env := os.environ.get("LTX2MLX_BIN"):
        candidates.append(env)
    if found := shutil.which("ltx-2-mlx"):
        candidates.append(found)
    home = Path.home()
    for root in (home / "ltx-2-mlx", home / ".ltx-2-mlx",
                 home / "Documents/Workspace/ltx-2-mlx"):
        candidates.append(str(root / ".venv/bin/ltx-2-mlx"))
    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK) and Path(candidate).is_file():
            return candidate
    return None


# What a downloaded pack looks like on disk: its own config plus the
# transformer weights. Checked because `ollamadiffuser pull` and ltx-2-mlx
# would otherwise each keep their own 21 GB copy of the same pack.
PACK_MARKERS = ("config.json", "split_model.json")


def looks_like_pack(path: Optional[str]) -> bool:
    """True when ``path`` is a directory holding an LTX-2 MLX pack."""
    if not path:
        return False
    folder = Path(path)
    if not folder.is_dir():
        return False
    has_marker = any((folder / name).is_file() for name in PACK_MARKERS)
    return has_marker and any(folder.glob("transformer*.safetensors"))


def valid_frame_count(frames: int) -> bool:
    """Frames must be ``8k + 1``: the VAE compresses time 8×."""
    return frames >= 9 and (frames - 1) % 8 == 0


def nearest_frame_count(frames: int) -> int:
    """The closest legal frame count, for turning seconds into frames.

    Halves round up — 13 becomes 17, not 9 — so a duration on the boundary
    gives slightly more video rather than slightly less. Spelled with floor
    rather than ``round`` because Python rounds halves to even, which would
    send 13 up and 21 down for no reason a caller could predict.
    """
    if frames <= 9:
        return 9
    return math.floor((frames - 1) / 8 + 0.5) * 8 + 1


def frames_for_seconds(seconds: float, fps: int = TRAINED_FPS) -> int:
    """Frames for a duration, rounded to the ``8k + 1`` grid."""
    return nearest_frame_count(int(round(seconds * fps)))


def build_argv(
    binary: str,
    prompt: str,
    output: str,
    *,
    pack: str = DEFAULT_PACK,
    mode: str = "two-stage",
    frames: Optional[int] = None,
    width: int = 704,
    height: int = 480,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    stg_scale: Optional[float] = None,
    stage1_steps: Optional[int] = None,
    stage2_steps: Optional[int] = None,
    image: Optional[str] = None,
    audio: Optional[str] = None,
    frame_rate: Optional[int] = None,
    audio_start: Optional[float] = None,
    auto_duration: Optional[str] = None,
    enhance_prompt: bool = False,
    low_ram: bool = False,
    tile_frames: Optional[int] = None,
    tile_spatial: Optional[int] = None,
    quiet: bool = True,
) -> list[str]:
    """The command line for one generation.

    ``--frame-rate`` is mandatory on both subcommands — the CLI enforces it at
    argparse, so a call without it dies before a single weight is read. It
    defaults to the pack's trained 24.

    Audio input picks the ``a2v`` subcommand, which is a narrower shape than
    ``generate``: the audio carries the length, and as of ltx-2-mlx 0.15.6 it
    takes no mode flag, no ``--steps`` and no ``--enhance-prompt`` (verified
    against the binary, not the README — argparse answers "unrecognized
    arguments" to all three). Everything else is ``generate``, where the mode
    is a flag and ``-f`` carries the length.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if frames is not None and not valid_frame_count(frames):
        raise ValueError(
            f"frames must be 8k+1 and at least 9 (the VAE compresses time 8×); "
            f"{frames} is not — nearest legal is {nearest_frame_count(frames)}"
        )

    if audio:
        argv = [binary, "a2v", "--audio", audio, "--output", output,
                "--prompt", prompt, "--model", pack]
        # Length comes from the audio; a frame count is still allowed, and
        # anything else about the pipeline is not this subcommand's to choose.
        if frames is not None:
            argv += ["--frames", str(frames)]
        argv += ["--height", str(height), "--width", str(width)]
        if audio_start is not None:
            argv += ["--audio-start", str(audio_start)]
    else:
        argv = [binary, "generate", "--prompt", prompt, "--output", output,
                "--model", pack]
        # 2.3 packs require -f; 2.5 packs predict the duration when it is
        # omitted, which is why frames is optional here rather than defaulted.
        if frames is not None:
            argv += ["--frames", str(frames)]
        elif auto_duration:
            argv += ["--auto-duration", auto_duration]
        argv += ["--height", str(height), "--width", str(width)]
        if mode == "two-stage":
            argv.append("--two-stage")
        elif mode == "two-stages-hq":
            argv.append("--two-stages-hq")
        elif mode == "distilled":
            argv.append("--distilled")
        else:
            argv.append("--one-stage")
        if steps is not None:
            argv += ["--steps", str(steps)]
        if enhance_prompt:
            argv.append("--enhance-prompt")

    # Mandatory, on both.
    argv += ["--frame-rate", str(frame_rate or TRAINED_FPS)]
    if image:
        argv += ["--image", image]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if cfg_scale is not None:
        argv += ["--cfg-scale", str(cfg_scale)]
    if stg_scale is not None:
        argv += ["--stg-scale", str(stg_scale)]
    if stage1_steps is not None:
        argv += ["--stage1-steps", str(stage1_steps)]
    if stage2_steps is not None:
        argv += ["--stage2-steps", str(stage2_steps)]
    if low_ram:
        argv.append("--low-ram")
    if tile_frames is not None:
        argv += ["--tile-frames", str(tile_frames)]
    if tile_spatial is not None:
        argv += ["--tile-spatial", str(tile_spatial)]
    if quiet:
        argv.append("--quiet")
    return argv


class LTXVideoMLXStrategy(InferenceStrategy):
    """LTX-2 video generation through the ltx-2-mlx CLI.

    ``load()`` does not load weights — the CLI owns those, and it downloads
    and caches its own packs. What load checks is everything that would
    otherwise fail minutes into a generation: the platform, the binary, and
    ffmpeg, which the CLI needs to write the mp4.
    """

    def __init__(self):
        super().__init__()
        self.binary: Optional[str] = None
        self.pack: str = DEFAULT_PACK
        self.defaults: dict = {}

    # ---------------------------------------------------------------- load

    def load(self, model_config: ModelConfig, device: str) -> bool:
        if not is_apple_silicon():
            logger.error(
                "LTX-2 MLX runs on Apple Silicon only. On CUDA, use the "
                "diffusers video path instead."
            )
            return False

        params = dict(getattr(model_config, "parameters", None) or {})
        binary = find_ltx_binary(params.get("ltx_binary"))
        if not binary:
            logger.error(
                "ltx-2-mlx was not found. Install it with:\n"
                "  git clone https://github.com/dgrauet/ltx-2-mlx && "
                "cd ltx-2-mlx && uv sync --all-extras\n"
                "then either put its .venv/bin on PATH or set LTX2MLX_BIN."
            )
            return False
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg is required to write the mp4: brew install ffmpeg")
            return False

        self.binary = binary
        # A pack `pull` already downloaded is the one to use: the CLI takes a
        # local path as happily as a hub id, and pointing it at the hub instead
        # would download the same 21 GB a second time into its own cache.
        local = getattr(model_config, "path", None)
        if looks_like_pack(local):
            self.pack = str(local)
            logger.info(f"LTX-2: using the pack at {local}")
        else:
            self.pack = (params.get("ltx_pack")
                         or getattr(model_config, "repo_id", None)
                         or DEFAULT_PACK)
        self.defaults = params
        self.model_config = model_config
        self.device = device
        # No weights in this process; the marker is what is_loaded reads.
        self.pipeline = f"ltx-2-mlx:{self.pack}"
        logger.info(f"LTX-2 MLX ready — {self.pack} via {binary}")
        return True

    def unload(self) -> None:
        # Nothing of ours is on the GPU: the CLI exits with its weights.
        self.pipeline = None
        self.binary = None

    # ------------------------------------------------------------ generate

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        """Not an image model.

        Raising beats returning one frame of the video, which is what the
        AnimateDiff strategy does and what makes a video model look like a
        broken image model.
        """
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
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Path:
        """Generate one video and return the file it landed in.

        ``seconds`` is offered because nobody thinks in frames: it is rounded
        to the ``8k + 1`` grid at the pack's 24 fps. An explicit ``frames``
        wins, as it does in the CLI.
        """
        if not self.binary:
            raise RuntimeError("Model not loaded — call load() first")

        target = Path(output) if output else Path(
            tempfile.gettempdir()) / f"ltx-{int(time.time())}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)

        options = argv_options(self.defaults)
        asked = argv_options(kwargs)
        options.update(asked)
        options.pop("pack", None)          # the pack is the loaded model's
        # Precedence: a frame count the caller gave, then their seconds, then
        # whatever the registry entry defaults to. Without the middle one a
        # registry that sets `frames` silently ignores every `seconds=` — which
        # is exactly what the q4 entry's `frames: 97` did to a two-second ask.
        if "frames" not in asked and seconds:
            # At the requested rate, not always 24: four seconds at 30 fps is
            # 121 frames, and rounding it against 24 would hand back 3.2s.
            options["frames"] = frames_for_seconds(
                seconds, int(options.get("frame_rate") or TRAINED_FPS))

        argv = build_argv(self.binary, prompt, str(target), pack=self.pack, **options)
        logger.info("LTX-2: %s", " ".join(argv[1:]))
        result = self._run(argv, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"ltx-2-mlx failed ({result.returncode}): {_tail(result.stdout)}"
            )
        if not target.exists():
            raise RuntimeError(
                f"ltx-2-mlx reported success but {target} is not there: "
                f"{_tail(result.stdout)}"
            )
        return target

    @staticmethod
    def _run(argv: Sequence[str], timeout: Optional[float] = None):
        """Run the CLI, keeping its output. Generations take minutes."""
        return subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )


def _tail(output: Optional[str], lines: int = 3) -> str:
    """The last few lines worth showing. Progress bars write carriage
    returns, so the raw tail is one very long line."""
    if not output:
        return "(no output)"
    parts = [p.strip() for p in output.replace("\r", "\n").split("\n") if p.strip()]
    return " / ".join(parts[-lines:]) if parts else "(no output)"
