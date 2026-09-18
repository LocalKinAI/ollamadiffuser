# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.23] - 2026-09-18

### 🔧 The LTX argv, checked against the binary instead of the README

`build_argv` was written from ltx-2-mlx's documentation. Running its `--help`
on the machine that will serve video found two things the docs do not say, both
of which lose a generation at argparse — before a weight is read, with a
message about arguments rather than about video:

- **`--frame-rate` is mandatory on `generate`, not just `a2v`.** Nothing about
  a text-to-video call suggests a frame rate is required; it is, so it is now
  always sent, defaulting to the pack's trained 24.
- **`a2v` takes no mode, no `--steps` and no `--enhance-prompt`** (0.15.6).
  The README reads as though the pipeline modes apply everywhere. They do not,
  and the binary answers "unrecognized arguments" to all three, so the audio
  path no longer forwards them — and it does pass `--frames`, `--height` and
  `--width`, which it does accept.

`seconds=` now converts at the requested frame rate rather than always 24: four
seconds at 30 fps is 121 frames, and rounding that against 24 handed back 3.2s
of video.

### Fixed — two found by the first real generation

Verified end to end on a 96 GB Mac Studio: **4 seconds of 704×448 in 83
seconds**, H.264 with a 48 kHz stereo AAC track, from an LTX-2.3 int4 pack.
Getting there cost two bugs no unit test had reason to catch:

- **Every text-to-video request was a 500.** `POST /api/generate/video` read
  `upload.filename` at the call site to work out a temp-file suffix — before
  the helper it passed it to checked the upload for None. So the ordinary
  call, a prompt and no files, died with an `AttributeError` and came back
  with `{"detail": "Video generation failed"}` and nothing else.
- **`seconds=` was ignored whenever the registry entry set `frames`.** A
  default and a caller's request landed in the same dict, and the q4 entry
  defaults to 97 — so a two-second ask spent twice as long making four
  seconds. Precedence is now: the caller's frames, then the caller's seconds,
  then the entry's default.

194 tests pass, 56 of them on the argv.

## [2.0.22] - 2026-09-18

### 🎬 Video on Apple Silicon — LTX-2 through ltx-2-mlx

The `video` strategy was AnimateDiff, 16 frames of 512×512 from the SD1.5 era,
with no registry entry pointing at it — and it returned one frame, so a video
model behaved like a broken image model. Video now means LTX-2, through
[ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)'s pure-MLX port: text-to-video
with 48 kHz stereo audio, image-to-video, audio-to-video, and LTX-2.5's
predicted durations.

Six registry entries — int4 / int8 / bf16 for LTX-2.3 and LTX-2.5 — sized from
upstream's own table: 12 GB on disk and 16 GB of RAM at the low end, 42 GB and
64 GB at the top. The 2.5 packs are marked gated, because they are.

**Why this one shells out** when every other strategy imports its pipeline:
ltx-2-mlx is a three-package monorepo that is not on PyPI, installs with
`uv sync`, and downloads its own weight packs. Importing it would mean pinning
three packages we do not build to reach an API its author does not support. Its
CLI is that API. The cost is a subprocess; the gain is that `build_argv` is a
pure function, so every mode, every memory flag and the `8k + 1` frame rule are
covered by tests that need no weights and no MLX — 50 of them.

New surfaces:

- `engine.generate_video(prompt, output=..., seconds=...)` — separate from
  `generate_image` rather than a flag on it, and a model that cannot make video
  says so before anything downloads.
- `POST /api/generate/video` — multipart, because the interesting inputs are
  files (a reference image, an audio track). Returns the mp4 as the body with
  `X-Output-Path` for the copy on disk, and answers **400** for everything a
  caller can fix (no model, an image model, ltx-2-mlx missing, a frame count
  off the grid) rather than a blanket 500.
- `seconds=` anywhere frames are taken: nobody thinks in frames, and the
  `8k + 1` grid (the VAE compresses time 8×) is not obvious. Halves round up,
  spelled with `floor(x + 0.5)` rather than `round`, which rounds halves to
  even and would send 13 up and 21 down for no reason a caller could predict.
- `LTX2MLX_BIN`, plus discovery in `~/ltx-2-mlx/.venv/bin` and the other places
  a `uv sync` checkout leaves the binary — the documented install is a git
  clone, so it usually is not on PATH.

Refusals are checked at load, not mid-generation: Apple Silicon, the binary,
and ffmpeg (which the CLI needs to write the mp4). 183 tests pass.

## [2.0.21] - 2026-09-18

### 🍎 Eight more MLX families — the registry catches up with mflux

mflux added a lot of model families after our May line, and none of them were
reachable from here. Now they are, as eight registry entries and one routing
table:

| Entry | Family | Why it is here |
|---|---|---|
| `boogu-image-turbo-mlx` | Boogu Image Turbo (10B) | 4-step DMD, photographic, bilingual EN/ZH text rendering. Apache 2.0. |
| `ernie-image-turbo-mlx` | ERNIE-Image Turbo (8B) | Baidu single-stream DiT, 8 steps, vivid output. Apache 2.0. |
| `krea-2-turbo-mlx` | Krea 2 Turbo (12B) | Wide stylistic range, photographic default, nine upstream style LoRAs. Gated. |
| `lens-turbo-mlx` | Lens (3.8B + 20B TE) | Microsoft dual-stream MMDiT — strong prompt adherence in 4 steps. |
| `ideogram-4-mlx` | Ideogram 4 (9B) | JSON-caption-native, typography-focused. |
| `fibo-mlx` | FIBO (8B) | Structured JSON prompts rather than a sentence. Gated. |
| `fibo-edit-mlx` | FIBO-Edit (8B) | Instruction editing, the FIBO family's Kontext. Requires `image=`. Gated. |
| `seedvr2-3b-mlx` | SeedVR2 3B | The best open upscaler. Requires `image=`. Apache 2.0. |

Adding a family is now data plus one line: `ModelConfig.from_name` resolves
every alias in mflux's own table, so `_ALIAS_ROUTED` carries only the module
and class name, and `_resolve_model_and_config` has one branch for all eight.
`fibo-edit` and `seedvr2` are listed in `_VARIANT_REQUIRED_INPUTS`, so calling
them without an input image fails here with a sentence instead of crashing deep
inside mflux.

### 🧪 Tests: two that could not survive a new model, fixed

- **The registry snapshot was an equality check**, so every model added after
  the `models.yaml` migration broke it — and what the snapshot is *for* is the
  guarantee that nothing from before the migration was lost or mangled, not
  that the registry never grows. It now asserts every snapshot entry is still
  present and identical, and that the count never shrinks.
- **The MLX resolution tests import mflux for real** (that is the point of
  them: they check our module paths still match theirs), but they were gated
  only on Apple Silicon — so on a Mac without the optional backend installed
  they failed with `ModuleNotFoundError` and read as a broken strategy. They
  now skip unless mflux is importable.

132 tests pass, 31 skip without the optional backends.

## [2.0.20] - 2026-05-19

### 📥 One-line `curl | sh` installer — no system Python required

New `install.sh` at the repo root is the Ollama-style zero-friction install:

```bash
curl -fsSL https://raw.githubusercontent.com/LocalKinAI/ollamadiffuser/main/install.sh | sh
```

What it does, and deliberately does NOT depend on the user's Python:

1. installs [`uv`](https://github.com/astral-sh/uv) (a single static binary) if absent;
2. fetches a **standalone Python 3.12** via uv (python-build-standalone) — the
   system Python is never touched;
3. creates an **isolated** env at `~/.ollamadiffuser/venv` (with `--seed`, so
   `ollamadiffuser enable ...` keeps working);
4. installs the compile-free core (prebuilt wheels only);
5. on Apple Silicon, also enables the native MLX backend (still compile-free);
6. drops an `ollamadiffuser` launcher on PATH (symlink into `~/.local/bin` if
   it's already on PATH, else an idempotent shell-rc line).

Uninstall is `rm -rf ~/.ollamadiffuser`. Re-running upgrades in place.

This removes the two structural barriers to an Ollama-class install: no
reliance on the user's Python/pip/compiler, and full environment isolation
(no clobbering the user's torch/numpy). The pip paths (`pip install
ollamadiffuser`, extras, `enable`) are unchanged for users who prefer them.

**One installer, not two.** The legacy `install_ollamadiffuser.sh` (a
pip-into-your-own-env wrapper) was removed. Its entire function — `pip
install ollamadiffuser` plus auto-enabling MLX on Apple Silicon — is now
covered by documented one-line commands, so a second script was just
clutter. `install.sh` (isolated, uv-based) is the single canonical
installer; users who want their own environment run `pip install
ollamadiffuser` directly.

**README consistency pass:** every install snippet that still taught a
compile path as the *primary* action was reconciled with the new story —
GGUF and Apple Silicon Quick Starts now lead with `ollamadiffuser enable
gguf` / `enable mlx`; the Troubleshooting section drops the obsolete
`[full]` shell-quoting blocks (the default install already carries OpenCV
etc.) and points persistent-issue users at the isolated `curl | sh`
reinstall; the manual `CMAKE_ARGS=-DSD_METAL=ON` step is reframed as "what
`enable gguf` does under the hood." The manual `CMAKE_ARGS=-DSD_CUDA=ON`
path stays documented as the advanced CUDA-acceleration route (`enable
gguf` currently only auto-sets the Metal flag on macOS).

**Validated:** installer passes `sh -n` + `dash -n` + shellcheck. The uv
mechanics were exercised end-to-end on Apple Silicon (isolated `uv venv
--seed` yields a working pip; the launcher shim forwards args; `uv pip
install --dry-run ollamadiffuser` resolves the full wheel set with no
compile step). The fresh-machine path (no uv/Python at all) should be
smoke-tested on a clean Mac before the release is announced.

## [2.0.19] - 2026-05-19

### 🗂️ Model registry moved to data (`models.yaml`) — contributor flywheel

The 59 built-in models moved out of a ~1500-line Python dict literal in
`model_registry.py` and into a bundled **`models.yaml`** data file.
`README` has long claimed "add a model via config alone, no code changes"
— but the config *was* Python, so adding a model still meant editing code
and reviewing logic. Now it's literally true: **contributing a model is a
pure-data PR** against `ollamadiffuser/core/config/models.yaml`.

- `_load_default_models()` shrank from ~1500 lines to a ~10-line YAML load.
  `model_registry.py` went from 1747 → ~260 lines.
- The YAML is the single source of truth for built-ins; user overrides in
  `~/.ollamadiffuser/models.yaml` (or `OLLAMADIFFUSER_MODEL_CONFIG`) still
  load and override on top, exactly as before.
- Packaged via `MANIFEST.in` + `[tool.setuptools.package-data]`; verified
  the built wheel ships `models.yaml` (a missing data file would silently
  produce an empty registry, so this is tested at build time).

### Zero-loss guarantee

A snapshot of the registry was captured **before** the migration
(`tests/fixtures/registry_snapshot.json`) and the YAML-loaded registry is
asserted **deep-equal** to it. 8 new tests in `tests/test_registry_yaml.py`
cover the snapshot equality, model count, YAML well-formedness, the minimal
per-entry schema, MLX entries surviving, and user-override precedence.
**144 passed, 8 skipped.**

## [2.0.18] - 2026-05-19

### 📦 Compile-free default install + `enable` command (install-friction / Tier 0)

The single biggest onboarding failure mode was the installer defaulting to
the `[full]` extra, which pulls `stable-diffusion-cpp-python` and compiles
a native extension (needs CMake + `CMAKE_ARGS=-DSD_METAL=ON`, and fails on
machines without a toolchain). The core dependency set is already 100%
prebuilt wheels on the supported platforms — nothing in it needs to
compile. This release stops making everyone pay the GGUF compile tax up
front.

#### New: `ollamadiffuser enable <backend>`

```bash
ollamadiffuser enable mlx     # Apple Silicon native (2-3x faster); installs mflux
ollamadiffuser enable gguf    # low-VRAM quantized models; compiles, sets CMAKE_ARGS for you
ollamadiffuser enable mcp     # Model Context Protocol server deps
```

- Ollama-style opt-in **by name** — no shell-quoted `[extras]` to get wrong
  (the reason `install_helper.py` existed; see Removed).
- `enable gguf` injects `CMAKE_ARGS=-DSD_METAL=ON` on macOS so users don't
  have to remember the Metal build flag; other platforms use their default
  toolchain.
- `enable mlx` guards on Apple Silicon (macOS arm64) and refuses elsewhere
  with a clear message instead of a confusing pip error.
- Implemented as `_build_enable_command()` (pure, testable) +
  `enable_backend()` in `cli/commands.py`, wired as `enable` in `cli/main.py`.

#### Installer script rewrite (`install_ollamadiffuser.sh`)

- Default install is now `pip install ollamadiffuser` (compile-free core),
  **not** `[full]`.
- On Apple Silicon it auto-enables the MLX backend (`mflux`, also
  compile-free) — sharpening the Mac-first wedge.
- GGUF is surfaced as an opt-in hint (`ollamadiffuser enable gguf`), not a
  forced compile.

#### Docs

- README Quick Start rewritten to lead with the compile-free path and the
  `enable` command; explains why `enable gguf` beats
  `pip install "ollamadiffuser[gguf]"`. Migration note no longer defaults
  to `[full]`.

### Removed

- **`install_helper.py`** — its entire job was telling users how to quote
  `pip install "ollamadiffuser[full]"` for their shell (zsh vs fish vs
  bash). With the default install carrying no bracketed extras, that
  friction no longer exists. Nothing imported it.

### Tests

- 12 new tests in `tests/test_enable_backend.py`: per-backend command
  construction, MLX platform guard (Linux / Intel-Mac rejection), GGUF
  Metal-flag injection on Mac vs none off-Mac, exit-code plumbing, and
  the platform-guard/unknown-backend paths never invoking pip. `subprocess.run`
  is mocked — no real installs. **136 passed, 8 skipped.**

### Note

The bracketed extras (`[gguf]`, `[mlx]`, `[mcp]`, `[full]`) are unchanged and
still work for users who prefer them or pin them in requirements files.

## [2.0.17] - 2026-05-18

### 🍎 MLX Backend Phase 2.5 — FLUX.1 family completion (#7)

Four new mflux variants wired (`flux1-fill`, `flux1-redux`, `flux1-depth`,
`flux1-controlnet`), bringing the MLX-supported FLUX.1 family to feature
parity with the PyTorch path. Five new registry entries; 14 MLX entries
in the registry now.

#### Strategy changes

`ollamadiffuser/core/inference/strategies/mlx_strategy.py`
- `SUPPORTED_MLX_VARIANTS` now nine entries (was five): adds
  `flux1-fill`, `flux1-redux`, `flux1-depth`, `flux1-controlnet`.
- New `_VARIANT_REQUIRED_INPUTS` map encodes per-variant required
  kwargs. Used by `generate()` for early validation:

  | Variant | Required kwargs |
  |---|---|
  | `flux1-kontext` | `image` |
  | `flux1-fill` | `image`, `mask_image` |
  | `flux1-redux` | `redux_images` |
  | `flux1-depth` | `image` |
  | `flux1-controlnet` | `control_image` |

- `generate()` accepts new kwargs (`mask_image`, `depth_image`,
  `control_image`, `redux_images`, `controlnet_strength`,
  `redux_image_strengths`) and materializes PIL Images / paths
  → mflux's `*_path` parameters. The existing kwarg-filter passes
  only what the variant's `generate_image` accepts.
- New `_materialize_image_path_list` helper accepts a scalar or list of
  images for Redux's `redux_image_paths`.

#### New registry entries (all gated by `[mlx]` extra)

| Name | mflux variant / model | Required kwargs |
|---|---|---|
| `flux.1-fill-dev-mlx` | `flux1-fill` / `dev` | `image`, `mask_image` |
| `flux.1-redux-dev-mlx` | `flux1-redux` / `dev` | `redux_images` |
| `flux.1-depth-dev-mlx` | `flux1-depth` / `dev` | `image` |
| `flux.1-controlnet-canny-mlx` | `flux1-controlnet` / `canny` | `control_image` |
| `flux.1-controlnet-upscaler-mlx` | `flux1-controlnet` / `upscaler` | `control_image` |

All five inherit FLUX.1 Non-Commercial license. Disk ~14-16 GB Q8.

#### Why this batch and not Phase 3 (HiDream-O1)

Investigated `mlx-community/HiDream-O1-Image-Dev-mlx-bf16` and the
upstream repo — the only Python entry point is a CLI script
(`scripts/hidream_o1/generate_hidream_o1_mlx.py`) plus 3 custom
helper modules. No clean Python API, no PyPI package. Clean
integration would require either vendoring those scripts into
this repo (maintenance burden) or shelling out via subprocess
(ugly). Deferred; tracked in #7.

#### Tests

17 new tests in `tests/test_mlx_strategy.py`:
- 11 variant-resolution tests (each new variant + each variant's
  bad-name rejection).
- `test_variant_required_input_enforcement` — parametrized across
  the four new variants + Fill's two-required-kwarg case; asserts
  `ValueError` with "requires kwargs" message.
- `test_redux_accepts_scalar_or_list` — covers the new
  `_materialize_image_path_list` helper's scalar/list/None paths.
- Existing Kontext test updated to match new "requires kwargs"
  error message.

**Total: 124 passed, 8 skipped.**

#### Phase 3 — still open

HiDream-O1 via `mlx-community` still needs an inference-API wrapper.
Tracking [#7](https://github.com/LocalKinAI/ollamadiffuser/issues/7).

## [2.0.16] - 2026-05-18

### 🍎 MLX Backend Phase 2 (#7)

`MLXStrategy` now routes **all five** mflux model families. Phase 1
shipped just FLUX.1 (schnell / dev); this release adds FLUX.1 Kontext,
FLUX.2-klein, Z-Image, and Qwen-Image. Seven new registry entries.

#### Strategy changes

`ollamadiffuser/core/inference/strategies/mlx_strategy.py`
- `SUPPORTED_MLX_VARIANTS` expanded to
  `{flux1, flux1-kontext, flux2, z_image, qwen-image}`.
- Renamed `_resolve_model_class` → `_resolve_model_and_config`. Now
  returns `(ModelClass, mflux_model_config)` so the unified load path
  is `cls(quantize=..., model_config=mc)` across all families.
  (Only `Flux1` has the `from_name()` factory; the other four require
  `ModelConfig.<factory>()`.)
- New `_variant` instance attribute caches which variant was loaded —
  enables per-call capability checks in `generate()`.
- `generate()` now **filters kwargs against the bound `generate_image`
  signature**. This matters because `Flux2Klein.generate_image()` has
  no `negative_prompt` parameter; the strategy detects this via
  `inspect.signature()` and drops unsupported kwargs with a debug log.
- `generate()` enforces `flux1-kontext` requires an `image=` kwarg —
  fails loud with `ValueError` rather than letting mflux crash mid-
  pipeline.

#### New registry entries

| Name | Variant / mflux model | License | Disk (Q8) | Rec. VRAM | Fits |
|---|---|---|---|---|---|
| `flux.1-kontext-dev-mlx` | flux1-kontext / dev | Non-Commercial | ~14 GB | 20 GB | M1 32GB |
| `flux.2-klein-4b-mlx` | flux2 / klein-4b | Apache 2.0 | ~7 GB | 12 GB | M4 16GB ✅ |
| `flux.2-klein-9b-mlx` | flux2 / klein-9b | Apache 2.0 | ~13 GB | 20 GB | M1 32GB |
| `z-image-turbo-mlx` | z_image / z-image-turbo | Apache 2.0 | ~8 GB | 12 GB | M4 16GB ✅ |
| `qwen-image-mlx` | qwen-image / qwen-image | Apache 2.0 | ~22 GB | 24 GB | M1 32GB |
| `qwen-image-edit-mlx` | qwen-image / qwen-image-edit | Apache 2.0 | ~22 GB | 24 GB | M1 32GB |

Plus the three from Phase 1 (v2.0.15): `flux.1-schnell-mlx`,
`flux.1-schnell-mlx-q4`, `flux.1-dev-mlx`.

#### Tests

13 new tests in `tests/test_mlx_strategy.py`:
- Variant resolution: 11 cases covering each family + bad-name rejection
  for each.
- `test_generate_filters_kwargs_for_variants_without_negative_prompt` —
  asserts the inspect-signature-based filter actually drops
  unsupported kwargs.
- `test_kontext_without_image_raises` — asserts the
  image-required guard.
- Existing 16 tests rewritten against the new `_resolve_model_and_config`
  API and a new stub helper (`_stub_resolution`) that uses real
  functions so `inspect.signature()` works.

**Total: 107 passed, 8 skipped.**

#### Phase 3 — still open

[`mlx-community/HiDream-O1-Image-Dev-mlx-bf16`](https://huggingface.co/mlx-community/HiDream-O1-Image-Dev-mlx-bf16)
ships its own loader (not on mflux). Either a separate
`HiDreamO1MLXStrategy` or an `mlx_backend: "mflux" | "mlx-community"`
abstraction on `MLXStrategy` will be needed. ~3-4 days of work.

## [2.0.15] - 2026-05-18

### 🍎 MLX Backend (Phase 1 of #7)

Apple-Silicon-native inference is here. New `MLXStrategy` routes through
[mflux](https://github.com/filipstrand/mflux) — a pure MLX implementation
of FLUX-family models that targets the Metal/Neural Engine stack natively.
On M-series Macs this is typically **2-3× faster** than the PyTorch+MPS
path that the other strategies use.

**This is Phase 1.** It ships FLUX.1 text-to-image (schnell / dev) with
optional MLX 4-bit and 8-bit quantization. Phase 2 (FLUX.2, Z-Image,
Qwen-Image via mflux) and Phase 3 (HiDream-O1 via mlx-community) are
follow-ups — track [#7](https://github.com/LocalKinAI/ollamadiffuser/issues/7).

#### New strategy

`ollamadiffuser/core/inference/strategies/mlx_strategy.py`
  - `MLXStrategy` class — routes via mflux's `Flux1.from_name()` /
    `generate_image()` API. Lazy-imports mflux so the module is safe
    to import on non-Mac hosts.
  - Refuses to load on anything other than macOS arm64 with a clear
    error message.
  - Overrides `unload()` (no torch cleanup) and `load_lora_runtime()`
    (mflux takes LoRAs at construction time, not at runtime — runtime
    LoRA is deferred).
  - `_get_strategy("mlx")` in `engine.py` now dispatches to it.

#### New registry entries

| Name | Quant | Disk | Recommended VRAM | License |
|---|---|---|---|---|
| `flux.1-schnell-mlx` | 8-bit | 14 GB | 16 GB | Apache 2.0 |
| `flux.1-schnell-mlx-q4` | 4-bit | 8 GB | 12 GB (M4 16GB friendly) | Apache 2.0 |
| `flux.1-dev-mlx` | 8-bit | 14 GB | 16 GB | FLUX Non-Commercial |

#### New optional dependency

```bash
pip install 'ollamadiffuser[mlx]'   # installs mflux >= 0.17.0
```

The `mflux` wheel is `darwin/arm64`-only — pip's environment marker
prevents installs on Linux/Windows so the extra is a no-op there.
`[full]` extra now bundles `mflux` alongside GGUF + MCP.

#### Tests

16 new tests in `tests/test_mlx_strategy.py` covering dispatch, platform
refusal, config validation, mocked load/generate (no real download), and
the unsupported-feature contract (runtime LoRA). All 94 tests pass.

## [2.0.14] - 2026-05-17

### 📋 Registry — New Models

- **`flux.1-kontext-dev`** ([black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)) — 12B instruction-based image-editing model. Currently the top-trending image-edit model on HuggingFace. Routed through the existing FluxStrategy with `pipeline_class: "FluxKontextPipeline"` (already exposed by diffusers ≥ 0.34). Pass an input `image` to `generate()` along with the edit instruction. FLUX Non-Commercial License.

- **`chroma1-hd`** ([lodestones/Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD)) — 8.9B FLUX-schnell derivative with custom MMDiT masking and a 250M timestep FFN. Routed through `GenericPipelineStrategy` with `pipeline_class: "ChromaPipeline"`. **Apache 2.0** — rare for a model this capable. On 16GB Macs, enable CPU offload at runtime.

## [2.0.13] - 2026-05-08

### 🐛 Bug Fixes
- **Fix `ollamadiffuser recommend` crash on CUDA hosts**: The hardware-detection routine accessed `torch.cuda.get_device_properties(0).total_mem`, but the PyTorch attribute is `total_memory`. Every CUDA user running `recommend` hit `AttributeError`. Fixes #5.

### 📚 Documentation
- **Fix broken repository URLs across the project**: 18 references pointed to `github.com/ollamadiffuser/ollamadiffuser` (non-existent org). All updated to the correct `github.com/LocalKinAI/ollamadiffuser`:
  - `pyproject.toml` `[project.urls]` (5 keys: Repository, Issues, "Bug Reports", "Feature Requests", "Source Code") — PyPI project page links now resolve.
  - `setup.py` `url=`, `ollamadiffuser/__init__.py` `__repository__`
  - `README.md`, `GGUF_GUIDE.md`, `PUBLISHING.md` (Docker GHCR path), `integrations/openclaw/SKILL.md`
- **Enable GitHub Discussions on the repo** so the "Discussions" links in README/GGUF_GUIDE actually resolve instead of 404'ing. Fixes #6.

## [2.0.12] - 2026-04-26

### 🐛 Bug Fixes
- **Fix Web UI 500 error on Starlette ≥ 0.29.0 (including 1.0+)**: All `templates.TemplateResponse(name, context)` calls in `ollamadiffuser/ui/web.py` were using the legacy two-positional-argument API. Starlette 0.29 made `request` the first positional argument and the old call was being interpreted as `TemplateResponse(name=<dict>, ...)`, causing `TypeError: unhashable type: 'dict'` when Jinja2 tried to look up the dict as a template name. Updated all 8 call sites to the new `TemplateResponse(request, name, context)` signature, which works on Starlette 0.29.0+. Fixes #4.

## [2.0.11] - 2026-03-10

### 🐛 Bug Fixes
- **Fix download failure with huggingface_hub >= 1.4**: Newer huggingface_hub passes a `name` kwarg to the tqdm progress bar class, which tqdm rejects with `TqdmKeyError`. Strip unknown kwargs before calling `super().__init__()`. Fixes #2.
- **Fix HiDream strategy import**: Catch `RuntimeError` (not just `ImportError`) when diffusers fails to import `HiDreamImagePipeline` due to missing `FLAX_WEIGHTS_NAME` in newer transformers.

### ⚡ Improvements
- **Generic strategy VAE upcast**: Add opt-in `vae_upcast_float32` parameter for models like Kolors that need VAE in float32 on MPS for numerical stability.

### 📋 Registry Updates
- **pixart-sigma**: Added MPS to supported devices (0.6B model runs well with float16).
- **flux.2-klein-4b**: Added MPS to supported devices (4B model works with bfloat16 on PyTorch 2.3+).
- **kolors**: Added `vae_upcast_float32` parameter for MPS stability.

## [2.0.10] - 2026-02-08

### 🐛 Bug Fixes
- **Fix MPS SD15 dtype mismatch**: Removed manual VAE upcast to float32 for SD 1.5, ControlNet (SD15), and generic strategy. `StableDiffusionPipeline` has no `force_upcast` mechanism, so upcasting VAE caused `c10::Half` input vs `float` bias error (UNet outputs float16 latents to float32 VAE). Everything now stays in float16 on MPS.

### ⚡ GGUF Loader Fixes
- **Fix `txt_to_img` → `generate_image`**: Updated to match current `stable-diffusion-cpp-python` API (`txt_to_img` was renamed to `generate_image`).
- **Fix sampler name format**: Normalize `dpmpp2m` → `dpm++2m` (and other DPM++ variants) to match library's expected format. Default sampler changed to `euler` for FLUX compatibility.

## [2.0.9] - 2026-02-07

### 📦 Smart Download Filtering
- **Pipeline-only downloads**: For known model types (SD 1.5, SDXL, SD3, FLUX), only download diffusers pipeline directories — skips root-level monolithic checkpoints, ONNX/Flax/OpenVINO exports. Saves 10–200 GB per model (e.g. DreamShaper 212 GB → 4 GB, SDXL-Turbo 52 GB → 7 GB).
- **Default ignore patterns**: All non-GGUF models skip `safety_checker/`, `feature_extractor/`, `*.ckpt`, `*.onnx`, `*.msgpack`, `*.xml`, `comfyui/`, `text_encoders/`, `__pycache__/`.
- **SDXL-Lightning fix**: Downloads only the 4-step checkpoint (6.5 GB) instead of all variants (52 GB). Uses `from_single_file` since the repo has no diffusers pipeline layout.
- **Per-model allow_patterns**: AuraFlow and FLUX.2-dev skip root-level monolithic checkpoints.

### ⚡ Loading Improvements
- **SDXL single-file loading**: SDXLStrategy now supports `from_single_file` for models like SDXL-Lightning that ship a single `.safetensors` checkpoint instead of a diffusers pipeline layout. Configured via `parameters.single_file` in the model registry.
- **Variant fallback in all strategies**: SD15, SDXL, SD3, and ControlNet strategies now catch `OSError`/`ValueError` when fp16 variant files are missing and retry loading without `variant`. Previously only the generic strategy had this fallback.
- **Registry parameter refresh on load**: `load_model()` now merges the latest registry parameters into the saved model config, so new fields (e.g. `single_file`, `allow_patterns`) are picked up without requiring a re-pull.

### 🐛 Bug Fixes
- **Skip attention slicing on MPS**: Attention slicing causes NaN in float16 UNet output for some SDXL models (e.g. RealVisXL) on Apple Silicon. Disabled on MPS since unified memory makes chunking unnecessary.

## [2.0.8] - 2026-02-06

### 🍎 MPS Float16 for SD15, SDXL, ControlNet
- **SDXL float16 on MPS**: Changed from float32 to float16, fixing OOM on 16GB Macs (~7GB instead of ~13GB). No manual VAE upcast — diffusers' built-in `force_upcast`/`upcast_vae()` handles VAE dtype correctly (manual upcast bypassed latent casting → black images).
- **SDXL: removed UNet upcast on MPS**: Upcasting UNet to float32 while text encoders stayed in float16 caused `MPSNDArrayMatrixMultiplication` assertion failure (dtype mismatch in cross-attention).
- **SDXL: force float16 on MPS**: Removed bfloat16 path for SDXL (bfloat16 is only for FLUX/HiDream).
- **SD 1.5 float16 on MPS**: Changed from float32 to float16 for consistency and lower memory usage (~1.7GB instead of ~3.4GB). VAE upcast to float32 (SD 1.5 pipeline has no `force_upcast`).
- **ControlNet float16 on MPS**: Changed from float32 to float16. SD15-based: VAE upcast to float32. SDXL-based: no manual upcast (diffusers handles it).
- **Generic strategy**: Skip manual VAE upcast for pipelines with `upcast_vae` method (SDXL-family); only manually upcast for pipelines without built-in handling.
- **Variant passthrough on MPS**: SDXL, SD 1.5, and ControlNet now pass `variant="fp16"` on MPS when the model has fp16 variant files (previously only passed on CUDA).

### 📦 Download Filtering
- **Non-GGUF models**: Skip `safety_checker/`, `feature_extractor/`, preview images (`.png`/`.jpg`/`.webp`), READMEs, and git metadata during `ollamadiffuser pull`. Saves ~1.2GB for SD models (safety_checker CLIP model is never loaded).

### 📋 Registry Updates
- **pixart-sigma**: Removed MPS from supported_devices (not working on Apple Silicon) *(re-added in v2.0.11)*
- **flux.2-klein-4b**: Removed MPS from supported_devices (not working on Apple Silicon) *(re-added in v2.0.11)*
- **cogview4**: min_ram_gb 16 → 24 (14GB model won't fit on 16GB Mac with OS overhead)
- **kolors**: min_ram_gb 16 → 32 (18GB model requires 32GB Apple Silicon)
- **hunyuan-dit**: min_ram_gb 16 → 24 (12GB model won't fit on 16GB Mac with OS overhead)
- **Updated performance_notes**: All three now explain that MPS requires more unified memory because CPU offload doesn't help on Apple Silicon

### 🐛 Bug Fixes
- **Symlink models_dir crash**: Fixed `FileExistsError` when `~/.ollamadiffuser/models` is a symlink (Python 3.10 `mkdir(exist_ok=True)` fails on symlinks)

## [2.0.7] - 2026-02-05

### 🍎 MPS Dtype Overhaul
- **Per-model dtype on MPS**: Generic strategy now respects each model's configured `torch_dtype` instead of blanket-overriding to float16
- **MPS default float32**: When a model doesn't specify dtype, default to float32 on MPS for numerical stability
- **Runtime bfloat16 check**: Detects MPS bfloat16 support (PyTorch 2.3+) and only falls back to float16 if unsupported
- **VAE upcast on MPS**: Automatically upcasts VAE to float32 when using float16 on MPS to prevent NaN in decode
- **NaN/Inf sanitization**: Clamps invalid pixels in generated images to avoid `invalid value encountered in cast` errors

### 🐛 Bug Fixes
- **CogView4 variant removed**: Removed `variant: "fp16"` from CogView4 registry entry (repo doesn't publish fp16 variant files)
- **Removed forced `use_safetensors`**: No longer forces `use_safetensors=True` for float16/bfloat16 models, letting diffusers auto-detect

### 🧪 Tests
- **VAE upcast test**: Verifies VAE is upcast to float32 when loading float16 models on MPS
- **NaN sanitization test**: Verifies NaN/Inf pixels are clamped to valid range
- **Updated dtype tests**: MPS default is now float32; bfloat16 is respected when specified

## [2.0.6] - 2026-02-04

### 🐛 Bug Fixes
- **Variant fallback**: Catch `ValueError` in addition to `OSError` when variant files are missing, retry without variant

## [2.0.5] - 2026-02-04

### 🐛 Bug Fixes
- **Variant passthrough**: Pass `variant` parameter from model config to `from_pretrained` in generic strategy for models with fp16 variant files

## [2.0.4] - 2026-02-04

### 🐛 Bug Fixes
- **FLUX/HiDream MPS dtype**: Use `bfloat16` on MPS for FLUX and HiDream strategies, matching v1.x behavior that worked on Apple Silicon

## [2.0.3] - 2026-02-04

### 🐛 Bug Fixes
- **MPS CPU offload removed**: Disabled `enable_model_cpu_offload` on MPS -- Apple Silicon unified memory means offloading adds overhead (slowness + disk swap) without saving memory

## [2.0.2] - 2026-02-04

### 🐛 Bug Fixes
- **MPS bfloat16 crash**: Fixed bfloat16 crashes in FLUX and HiDream strategies on MPS devices

## [2.0.1] - 2026-02-04

### 🐛 Bug Fixes
- **peft dependency**: Fixed missing `peft` dependency for LoRA support
- **Project status**: Updated project status and metadata

## [2.0.0] - 2026-02-04

### 🏗️ Architecture Overhaul

#### Strategy Pattern Engine
- **Refactored InferenceEngine**: Replaced 1400+ line monolithic class with a ~220 line facade delegating to per-model strategy classes
- **8 Strategy Classes**: `SD15Strategy`, `SDXLStrategy`, `FluxStrategy`, `SD3Strategy`, `ControlNetStrategy`, `VideoStrategy`, `HiDreamStrategy`, `GGUFStrategy`
- **Abstract Base Class**: `InferenceStrategy` in `core/inference/base.py` with shared loading, LoRA, seed, and error handling logic
- **Unified Safety Checker**: Replaced 5+ monkey-patch approaches with a single `SAFETY_DISABLED_KWARGS` dict passed to `from_pretrained`

#### CLI Modularization
- **Split CLI**: Broke 1300+ line `cli/main.py` into 5 focused modules (~139 line router)
- **model_commands.py**: pull, run, list, show, check, rm, ps, load, unload, serve, stop
- **lora_commands.py**: LoRA pull, load, unload, rm, ps, list, show
- **registry_commands.py**: Registry list, add, remove, reload, import-config, export, check-gguf
- **config_commands.py**: Config show, set (models_dir, cache_dir, server settings)

#### Duplicate Code Removal
- **Removed `core/models/registry.py`**: Eliminated duplicate model registry (kept `core/config/model_registry.py`)
- **Removed dead files**: `constants.py`, `helpers.py` from old engine architecture

### 🚀 New Features

#### Image-to-Image Generation
- **img2img API endpoint**: `POST /api/generate/img2img` with strength control
- **Inpainting API endpoint**: `POST /api/generate/inpaint` with mask support
- **SD1.5 img2img/inpainting**: Via `StableDiffusionImg2ImgPipeline` and `StableDiffusionInpaintPipeline`
- **SDXL img2img/inpainting**: Via `StableDiffusionXLImg2ImgPipeline` and `StableDiffusionXLInpaintPipeline`
- **Web UI img2img**: New collapsible img2img section with input image preview

#### Seed Support
- **Random seeds by default**: Replaced hardcoded `seed=42` with `random.randint(0, 2**32 - 1)`
- **Reproducible generation**: Pass explicit `seed` parameter to any generation endpoint
- **Web UI seed field**: Optional seed input with "Random" placeholder
- **API seed parameter**: `seed` field in `GenerateRequest` model

#### Async API
- **Non-blocking generation**: All GPU-bound operations wrapped in `asyncio.to_thread()`
- **Async model loading**: `/api/models/load` and `/api/models/pull` are non-blocking
- **Web UI async**: Generate and load operations in thread pool

### 🌐 21 New Models

#### GenericPipelineStrategy
- **New strategy class**: Dynamically loads any `diffusers` pipeline class via `model_config.parameters["pipeline_class"]`, enabling new model types without writing new strategy code

#### Tier 1: Classic Models (existing strategies)
- **Stable Diffusion 3.5 Large**: `stabilityai/stable-diffusion-3.5-large` (sd3, 28 steps)
- **Stable Diffusion 3.5 Large Turbo**: `stabilityai/stable-diffusion-3.5-large-turbo` (sd3, 4 steps, guidance=0.0)
- **RealVisXL V4**: `SG161222/RealVisXL_V4.0` (sdxl, photorealistic finetune)
- **DreamShaper**: `Lykon/DreamShaper` (sd15, popular community model)
- **Realistic Vision V6**: `SG161222/Realistic_Vision_V6.0_B1_noVAE` (sd15, portrait specialist)
- **SDXL Turbo**: `stabilityai/sdxl-turbo` (sdxl, single-step inference)

#### Tier 2: Scheduler Override
- **SDXL Lightning 4-Step**: `ByteDance/SDXL-Lightning` (sdxl, EulerDiscreteScheduler with trailing timestep spacing)
- **SDXLStrategy scheduler override**: Reads `scheduler_class` and `scheduler_kwargs` from model parameters to configure custom schedulers

#### Tier 3: FLUX Pipeline Variants
- **FLUX.1 Fill**: `black-forest-labs/FLUX.1-Fill-dev` (FluxFillPipeline, inpainting/outpainting)
- **FLUX.1 Canny**: `black-forest-labs/FLUX.1-Canny-dev` (FluxControlPipeline, edge control)
- **FLUX.1 Depth**: `black-forest-labs/FLUX.1-Depth-dev` (FluxControlPipeline, depth control)
- **FluxStrategy dynamic pipeline**: Reads `pipeline_class` from parameters, defaults to `FluxPipeline` for backward compatibility

#### Tier 4: Next-Generation Models (GenericPipelineStrategy)
- **FLUX.2 Dev**: `black-forest-labs/FLUX.2-dev` (32B params, Flux2Pipeline)
- **FLUX.2 Klein 4B**: `black-forest-labs/FLUX.2-klein-4B` (4B params, Apache 2.0, Flux2KleinPipeline)
- **Z-Image Turbo**: `Tongyi-MAI/Z-Image-Turbo` (Alibaba 6B, 8-step turbo, bilingual CN/EN)
- **SANA 1.5**: `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` (NVIDIA 1.6B, very efficient)
- **CogView4**: `THUDM/CogView4-6B` (Zhipu AI 6B, GLM-4 encoder, bilingual CN/EN)
- **Kolors**: `Kwai-Kolors/Kolors-diffusers` (Kuaishou 8.6B, ChatGLM3, bilingual CN/EN)
- **Hunyuan-DiT**: `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` (Tencent 1.5B, dual text encoders)
- **Lumina 2.0**: `Alpha-VLLM/Lumina-Image-2.0` (2B, unified text+image tokens)
- **PixArt-Sigma**: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (0.6B, <8GB VRAM, 4K capable)
- **AuraFlow**: `fal/AuraFlow-v0.3` (6.8B, largest Apache 2.0 text-to-image model)
- **OmniGen**: `Shitao/OmniGen-v1-diffusers` (BAAI 3.8B, unified generation, no negative prompt)

### 🔌 OpenClaw & MCP Integration
- **MCP Server**: New `ollamadiffuser mcp` command (or `ollamadiffuser-mcp` entry point) starts a Model Context Protocol server with 4 tools: `generate_image`, `list_models`, `load_model`, `get_status` -- works with OpenClaw, Claude Code, Cursor, and any MCP client
- **OpenClaw AgentSkill**: `integrations/openclaw/SKILL.md` provides a ready-to-use skill for OpenClaw with full REST API instructions for model management and image generation
- **Base64 JSON response**: New `response_format=b64_json` option on `POST /api/generate` returns `{"image": "<base64>", "format": "png", "width": w, "height": h}` -- essential for AI agent and messaging platform integration
- **Optional dependency**: `pip install "ollamadiffuser[mcp]"` or `pip install "ollamadiffuser[openclaw]"` adds MCP support; included in `[full]`

### 🍎 Apple Silicon / Mac Mini Support
- **`ollamadiffuser recommend` command**: New CLI command that detects hardware (CUDA/MPS/CPU, RAM/VRAM) and recommends models that fit, with `--commercial-only` and `--device` flags
- **MPS dtype safety**: GenericPipelineStrategy now falls back from bfloat16 to float16 on MPS devices, preventing Metal compatibility issues
- **GGUF Metal acceleration**: All 11 GGUF models now list MPS in supported_devices (Metal acceleration via `CMAKE_ARGS="-DSD_METAL=ON"`)
- **Registry MPS updates**: Added MPS support to `cogview4` and `lumina-2` model entries
- **Smart quick-start**: `recommend` command suggests the best standalone model for your hardware (e.g. `pixart-sigma` for 16GB Mac)

### ⚙️ Configuration Management
- **`ollamadiffuser config` command**: New CLI command that shows all configuration (paths, server settings)
- **`ollamadiffuser config set`**: Set custom model storage directory, cache directory, and server settings
- **Custom paths**: `models_dir` and `cache_dir` can now be overridden and persisted in `config.json`
- **Registry command visibility**: Made `ollamadiffuser registry` command visible in CLI help output

### 🔒 Security Fixes
- **Path traversal prevention**: Removed `control_image_path` from `/api/generate` endpoint (use dedicated `/api/generate/controlnet` with file upload instead)
- **CORS hardening**: Dropped `allow_credentials=True` from CORS middleware
- **Error detail leakage**: Replaced raw exception messages in HTTP responses with generic error messages, log full details server-side with `exc_info=True`

### 🐛 Bug Fixes (v2.0.0)
- **HuggingFace Hub path check**: Fixed `Path.exists()` returning False for Hub IDs like `"black-forest-labs/FLUX.1-schnell"` by only checking local paths
- **Device placement conflict**: Fixed `_move_to_device()` and `enable_model_cpu_offload()` contradicting each other in FluxStrategy, HiDreamStrategy, and VideoStrategy (now mutually exclusive)
- **Falsy zero evaluation**: Fixed `num_inference_steps=0` being treated as falsy across all 8 strategies (now uses `is not None` check)
- **SD1.5 silent guidance clamping**: Removed silent `guidance > 7.0` clamp; kept MPS clamp with logging
- **SD1.5 silent dimension override**: Replaced silent 1024->512 dimension override with logged warning for dimensions > 768
- **VideoStrategy hardcoded dimensions**: Fixed hardcoded `width=512, height=512` to use user-specified values
- **Error image truncation**: Added `prompt[:50] + "..."` truncation in `_create_error_image` for long prompts
- **GGUF duplicate seed logic**: Moved `import random` to module level
- **Web UI LoRA download**: Fixed `final_name` reference error in `pull_lora_ui` by initializing before try block
- **Hardcoded seed=42**: Now random by default across all strategies
- **Safety checker monkey-patches**: Unified into clean `SAFETY_DISABLED_KWARGS`
- **Missing CLI commands**: Added `serve` and `stop` to `model_commands.py`
- **Sync API blocking**: All endpoints now properly async

### 🧪 Test Suite
- **82 tests across 7 modules**: All passing
- **test_settings.py**: ModelConfig, ServerConfig, Settings save/load
- **test_model_registry.py**: Default models, get/add/remove, model names, 21 new model validation (types, pipeline classes, scheduler config, hardware requirements, MPS size constraints)
- **test_engine.py**: Strategy factory (all 9 types including generic), device detection, engine lifecycle, GenericPipelineStrategy (dynamic pipeline loading, missing/invalid pipeline class, CPU offload), base strategy methods, safety kwargs
- **test_api_server.py**: Health, root, models, generate endpoints via TestClient
- **test_mps_support.py**: GenericPipelineStrategy MPS dtype fallback (6 tests), registry MPS device assignments (2 tests), recommend command classification and CLI (7 tests)
- **test_api_base64.py**: Base64 JSON response format (default PNG, b64_json, null format, no-model error)
- **test_mcp_server.py**: MCP server creation, tool registration, list_models, get_status, load_model, generate_image error handling (8 tests, skip gracefully without mcp package)

### 📦 Dependencies Updated
- **torch**: `>=2.4.0` (was `>=2.0.0`)
- **diffusers**: `>=0.34.0` (was `>=0.26.0`)
- **accelerate**: `>=1.0.0` (was `>=0.20.0`)
- **transformers**: `>=4.44.0` (was `>=4.30.0`)
- **numpy**: `>=1.26.0` (was `>=1.24.0`)
- **Pillow**: `>=10.0.0` (was `>=9.5.0`)
- **GGUF optional**: Moved `stable-diffusion-cpp-python` and `gguf` to `[gguf]` extra
- **Dev dependencies**: Added `pytest`, `pytest-cov`, `pytest-asyncio`, `httpx`, `mypy`

### 🌐 Web UI Updates
- **Version display**: Header shows v2.0.0
- **Seed field**: Optional seed input for reproducible results
- **img2img tab**: Collapsible section with strength, steps, guidance, and seed controls
- **img2img results**: Side-by-side input/output image display
- **Async operations**: Model loading and generation run in thread pool

### 🐛 Bug Fixes
- **Hardcoded seed=42**: Now random by default across all strategies
- **Safety checker monkey-patches**: Unified into clean `SAFETY_DISABLED_KWARGS`
- **Missing CLI commands**: Added `serve` and `stop` to `model_commands.py`
- **Sync API blocking**: All endpoints now properly async

### ⚠️ Breaking Changes
- **Minimum Python**: 3.10+ (was 3.8+)
- **Dependency versions**: Major bumps to torch, diffusers, accelerate
- **GGUF install**: Now requires `pip install "ollamadiffuser[gguf]"` instead of separate packages
- **Internal API**: `InferenceEngine` internals restructured (public API unchanged)

### 🔄 Migration Guide
For users upgrading from v1.x:

1. **Update dependencies**: `pip install --upgrade ollamadiffuser`
2. **GGUF users**: `pip install "ollamadiffuser[gguf]"` (separate install no longer works)
3. **API consumers**: No changes needed -- all endpoints remain compatible
4. **Python API users**: `engine.generate_image()` API unchanged, now supports `seed`, `image`, `mask_image`, and `strength` parameters

---

## [1.2.0] - 2025-06-02

### 🚀 Major Features Added

#### ⚡ GGUF Model Support
- **Quantized Models**: Full support for GGUF (GPT-Generated Unified Format) quantized models
- **Massive VRAM Reduction**: Run FLUX.1-dev with 3GB VRAM instead of 20GB+ 
- **7 Quantization Levels**: From q2k (3GB) to f16 (16GB) for different hardware capabilities
- **Hardware Optimization**: Native CUDA and Metal acceleration support
- **CPU Fallback**: Automatic CPU inference when VRAM is insufficient

#### 🎛️ GGUF Model Variants
- **flux.1-dev-gguf-q2k**: Ultra-low VRAM (3GB) for testing and low-end hardware
- **flux.1-dev-gguf-q3ks**: Balanced option (4GB) for mobile GPUs
- **flux.1-dev-gguf-q4ks**: **Recommended** (6GB) - best quality/performance balance
- **flux.1-dev-gguf-q5ks**: High quality (8GB) for mid-range GPUs
- **flux.1-dev-gguf-q6k**: Near-original quality (10GB) 
- **flux.1-dev-gguf-q8**: Minimal quality loss (12GB)
- **flux.1-dev-gguf-f16**: Full precision (16GB)

### 🛠️ Technical Implementation

#### GGUF Engine Integration
- **Backend**: stable-diffusion.cpp with Python bindings integration
- **Automatic Detection**: Seamless GGUF model recognition and loading
- **Memory Management**: Intelligent VRAM usage and CPU offloading
- **Hardware Acceleration**: CMAKE-based CUDA and Metal compilation support

#### CLI Enhancements
- **GGUF Check**: `ollamadiffuser registry check-gguf` command for compatibility verification
- **Model Pull**: Seamless GGUF model downloading with progress tracking
- **Status Monitoring**: Real-time GGUF support and model status checking

### 🎯 Performance Optimizations

#### Generation Parameters
- **Optimized Settings**: 4-step generation (FLUX-optimized)
- **CFG Scale**: guidance_scale=1.0 for best FLUX results
- **Euler Sampler**: Recommended sampler for GGUF models
- **Hardware Adaptation**: Automatic parameter adjustment based on available VRAM

#### Memory Efficiency
- **Smart Loading**: Load only required model components
- **Progressive Quantization**: Automatic fallback to lower quantization when needed
- **Resource Management**: Intelligent GPU memory allocation and cleanup

### 📚 Documentation & Guides

#### Comprehensive GGUF Guide
- **GGUF_GUIDE.md**: Complete 160+ line guide with installation, usage, and troubleshooting
- **Hardware Recommendations**: Specific guidance for different GPU tiers
- **Performance Comparisons**: Quality vs speed vs VRAM usage tables
- **Troubleshooting**: Common issues and solutions for GGUF models

#### Usage Examples
- **CLI Workflows**: Step-by-step GGUF model usage examples
- **Python API**: Code examples for programmatic GGUF model usage
- **Web UI Integration**: Browser-based GGUF model selection and generation

### 🔧 Dependencies & Requirements

#### New Dependencies
- **stable-diffusion-cpp-python**: Core GGUF inference engine
- **gguf**: Model format handling and validation
- **Enhanced OpenCV**: Updated to >=4.8.0 for improved compatibility

#### Hardware Support
- **NVIDIA CUDA**: CMAKE_ARGS="-DSD_CUDA=ON" installation
- **Apple Metal**: CMAKE_ARGS="-DSD_METAL=ON" for M1/M2 Macs
- **CPU Inference**: Full CPU fallback support for any modern processor

### 🎨 User Experience Improvements

#### Accessibility
- **Low-End Hardware**: Enable FLUX.1-dev on 3GB GPUs (previously impossible)
- **Faster Downloads**: Reduced model sizes from ~24GB to 3-16GB
- **Quick Testing**: Instant model switching between quantization levels

#### Web UI Enhancements
- **GGUF Model Selection**: Dropdown menu with GGUF model variants
- **VRAM Monitoring**: Real-time memory usage display
- **Quality Preview**: Visual quality indicators for each quantization level

### 🐛 Bug Fixes & Improvements
- **Memory Leaks**: Improved GGUF model cleanup and resource management
- **Error Handling**: Better error messages for GGUF-specific issues
- **Compatibility**: Enhanced hardware detection and fallback mechanisms

### ⚠️ Breaking Changes
- **Dependency Requirements**: New GGUF dependencies required for full functionality
- **Model Loading**: GGUF models use different loading mechanisms than regular models

### 🔄 Migration Guide
For users upgrading to v1.2.0:

1. **Install GGUF Dependencies**: `pip install stable-diffusion-cpp-python gguf`
2. **Check Compatibility**: `ollamadiffuser registry check-gguf`
3. **Download GGUF Model**: `ollamadiffuser pull flux.1-dev-gguf-q4ks`
4. **Update Hardware Acceleration**: Reinstall with CUDA/Metal support if needed

### 📊 Performance Metrics
- **VRAM Reduction**: Up to 85% reduction (20GB → 3GB)
- **File Size**: Up to 87% smaller downloads (24GB → 3GB)
- **Generation Speed**: Comparable or faster due to optimized quantization
- **Quality Retention**: 90%+ quality retention with q4ks quantization

## [1.1.6] - 2025-5-30

### 🎨 New Features

#### ControlNet Sample Images
- **New CLI Command**: `ollamadiffuser create-samples` for creating ControlNet demonstration images
- **Built-in Samples**: Pre-made control images for Canny, Depth, OpenPose, and Scribble controls
- **Web UI Integration**: Sample images automatically available in the web interface for easy testing
- **Force Recreation**: `--force` flag to recreate all samples even if they exist

#### Installation Helper
- **New Script**: `install_helper.py` for platform-specific installation guidance
- **Shell Detection**: Automatically detects user's shell (bash, zsh, fish) and provides correct install syntax
- **Multiple Installation Options**: Basic, Full, and Development installation commands
- **Interactive Installation**: Option to install directly from the helper script

### 🛠️ Improvements

#### CLI Enhancements
- **Progress Tracking**: Enhanced download progress display with Ollama-style formatting
- **Better Error Handling**: Improved error messages and graceful failure modes
- **Warning Fixes**: Resolved various CLI warnings and edge cases

#### Web UI Enhancements
- **Sample Image Gallery**: Built-in ControlNet samples with 3 images per control type
- **Automatic Sample Creation**: Samples generated automatically when needed
- **Better UX**: Visual samples make ControlNet testing more intuitive

### 🐛 Bug Fixes
- **Version Inconsistencies**: Fixed version numbering across different components
- **Installation Issues**: Resolved shell-specific installation syntax problems
- **CLI Warnings**: Fixed various warning messages and edge cases
- **Sample Generation**: Improved reliability of sample image creation

### 📦 Technical Changes
- **MANIFEST.in**: Updated to include sample images and static files
- **Dependencies**: Refined dependency management for better compatibility
- **Shell Compatibility**: Better support for zsh, fish, and bash shells

### 🎯 Sample Images Created
- **Canny Control**: Geometric shapes, house outline, portrait silhouette (3 samples)
- **Depth Control**: Depth map variations for different scene types (3 samples)
- **OpenPose Control**: Human pose variations for different positions (3 samples)
- **Scribble Control**: Hand-drawn style sketches and outlines (3 samples)

## [1.1.0] - 2025-5-29

### 🚀 Major Features Added

#### ⚡ Lazy Loading Architecture
- **Instant Startup**: Application now starts immediately without downloading ControlNet models
- **On-Demand Loading**: ControlNet preprocessors initialize only when actually needed
- **Performance Boost**: `ollamadiffuser --help` runs in milliseconds instead of 30+ seconds
- **Memory Efficient**: No unnecessary model downloads for users who don't use ControlNet

#### 🎛️ Complete ControlNet Integration
- **6 ControlNet Models**: SD 1.5 and SDXL variants (canny, depth, openpose, scribble)
- **10 Control Types**: canny, depth, openpose, hed, mlsd, normal, lineart, lineart_anime, shuffle, scribble
- **Advanced Preprocessors**: Full controlnet-aux integration with graceful fallbacks
- **Web UI Integration**: File upload, preprocessing, and side-by-side result display
- **REST API Support**: Complete API endpoints for ControlNet generation and preprocessing

#### 🔄 Enhanced LoRA Management
- **Web UI Integration**: Download LoRAs directly from Hugging Face in the browser
- **Alias Support**: Create custom names for your LoRAs
- **Strength Control**: Adjust LoRA influence with intuitive sliders
- **Real-time Loading**: Load/unload LoRAs without restarting the application

### 🛠️ Technical Improvements

#### ControlNet Preprocessor Manager
- **Lazy Initialization**: `ControlNetPreprocessorManager` with `is_initialized()`, `is_available()`, `initialize()` methods
- **Automatic Fallback**: Basic OpenCV processors when advanced ones fail
- **Error Handling**: Robust validation and graceful degradation
- **Status Tracking**: Real-time initialization and availability status

#### Web UI Enhancements
- **ControlNet Section**: Dedicated controls with status indicators
- **Initialization Button**: Manual preprocessor initialization for faster processing
- **File Upload**: Drag-and-drop control image upload with validation
- **Responsive Design**: Mobile-friendly interface with adaptive layouts
- **Real-time Status**: Live model, LoRA, and ControlNet status indicators

#### API Improvements
- **New Endpoints**: `/api/controlnet/initialize`, `/api/controlnet/preprocessors`, `/api/controlnet/preprocess`
- **File Upload Support**: Multipart form data handling for control images
- **Status Endpoints**: Check ControlNet availability and initialization status
- **Error Handling**: Comprehensive error responses with helpful messages

### 📦 Dependencies Updated
- **controlnet-aux**: Added `>=0.0.7` for advanced preprocessing capabilities
- **opencv-python**: Added `>=4.8.0` for basic image processing fallbacks
- **diffusers**: Updated to `>=0.26.0` for ControlNet compatibility

### 🎨 User Experience Improvements

#### Startup Performance
- **Before**: 30+ seconds startup time, 1GB+ automatic downloads
- **After**: Instant startup, downloads only when needed
- **User Control**: Choose when to initialize ControlNet preprocessors

#### Web UI Experience
- **Status Indicators**: Clear visual feedback for all system states
- **Progressive Loading**: Initialize components as needed
- **Error Messages**: Helpful guidance for common issues
- **Mobile Support**: Responsive design works on all devices

#### CLI Experience
- **Fast Commands**: All CLI commands run instantly
- **Lazy Loading**: ControlNet models load only when generating
- **Status Commands**: Check system state without triggering downloads

### 🔧 Configuration Changes
- **setup.py**: Added ControlNet dependencies
- **pyproject.toml**: Updated dependency specifications
- **Model Registry**: Enhanced with ControlNet model definitions

### 📚 Documentation Updates
- **CONTROLNET_GUIDE.md**: Comprehensive 400+ line guide with examples
- **README.md**: Updated with lazy loading features and ControlNet quick start
- **API Documentation**: Complete endpoint reference with examples

### 🐛 Bug Fixes
- **Startup Crashes**: Fixed 404 errors from non-existent model repositories
- **Memory Leaks**: Improved cleanup of ControlNet preprocessors
- **Device Compatibility**: Better handling of CPU/GPU device switching
- **Error Handling**: More graceful failure modes with helpful messages

### ⚠️ Breaking Changes
- **Import Behavior**: `controlnet_preprocessors` module no longer auto-initializes
- **API Changes**: Some ControlNet endpoints require explicit initialization

### 🔄 Migration Guide
For users upgrading from v1.0.x:

1. **No Action Required**: Lazy loading is automatic and transparent
2. **Web UI**: ControlNet preprocessors initialize automatically when uploading images
3. **API Users**: Call `/api/controlnet/initialize` for faster subsequent processing
4. **Python API**: Use `controlnet_preprocessor.initialize()` for batch processing

### 🎯 Performance Metrics
- **Startup Time**: Reduced from 30+ seconds to <1 second
- **Memory Usage**: Reduced baseline memory footprint by ~2GB
- **First Generation**: Slightly slower due to lazy loading, then normal speed
- **Subsequent Generations**: Same performance as before

## [1.0.0] - 2025-5-28

### Added
- Initial release with core functionality
- Support for Stable Diffusion 1.5, SDXL, SD3, and FLUX models
- Basic LoRA support
- CLI interface
- REST API server
- Web UI interface
- Model management system

### Features
- Model downloading and management
- Image generation with various parameters
- Multiple interface options (CLI, API, Web UI)
- Hardware optimization (CUDA, MPS, CPU)
- Safety checker bypass for creative freedom

---

## Development Notes

### Version Numbering
- **Major** (X.0.0): Breaking changes, major feature additions
- **Minor** (1.X.0): New features, significant improvements
- **Patch** (1.1.X): Bug fixes, minor improvements

### Release Process
1. Update version in `__init__.py`
2. Update CHANGELOG.md with new features
3. Update documentation
4. Create release tag
5. Deploy to package repositories 