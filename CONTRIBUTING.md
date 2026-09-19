# Contributing to OllamaDiffuser

The most useful contribution is usually **a model**, and adding one is a data
change: the built-in registry is a YAML file, not code.

## Adding a model

Edit [`ollamadiffuser/core/config/models.yaml`](ollamadiffuser/core/config/models.yaml)
and add an entry under `models:`. The key is the name users type
(`ollamadiffuser pull <name>`).

```yaml
  my-model:
    repo_id: org/repo-on-huggingface      # required
    model_type: generic                   # required — see the table below
    variant: bf16
    # Optional. Without it, pull downloads the whole repo. Repos often carry
    # the same weights several times (fp8 + fp16, a ComfyUI single-file copy).
    allow_patterns:
    - "transformer/*"
    - "vae/*"
    - "*.json"
    parameters:
      pipeline_class: ChromaPipeline      # generic only: the diffusers class
      torch_dtype: bfloat16
      num_inference_steps: 28
      guidance_scale: 4.0
    hardware_requirements:
      min_vram_gb: 12
      recommended_vram_gb: 16
      min_ram_gb: 16
      recommended_ram_gb: 32
      disk_space_gb: 18                   # measured, please — see below
      supported_devices: [CUDA, MPS]
      performance_notes: "One sentence a user can act on."
    license_info:
      type: Apache 2.0
      requires_agreement: false           # true if the repo is gated
      commercial_use: true
```

### Which `model_type`

| `model_type` | Use it for |
|---|---|
| `generic` | Any diffusers pipeline. Set `parameters.pipeline_class`. **Start here** — most new models need nothing else. |
| `mlx` | Apple Silicon, through [mflux](https://github.com/filipstrand/mflux). Set `parameters.mlx_variant` and `parameters.mlx_model_name`. |
| `flux`, `sdxl`, `sd15`, `sd3` | Those families, when the model is a drop-in checkpoint for one. |
| `controlnet_sd15`, `controlnet_sdxl` | ControlNets. Also set `base_model` and `controlnet_type`. |
| `ltx-video-mlx` | LTX-2 video packs on Apple Silicon. |

If none fits, the model needs a strategy class in
`ollamadiffuser/core/inference/strategies/` — open an issue first so we can
agree on the shape.

### For `mlx` entries: check what mflux will actually load

`mlx_model_name` is an mflux alias, and mflux decides which HuggingFace repo
that alias means. `repo_id` must be **that** repo, or `pull` downloads one
checkpoint and mflux is handed another. Ask mflux:

```bash
python -c "from mflux.models.common.config.model_config import ModelConfig; \
           print(ModelConfig.from_name('qwen-image').model_name)"
```

### Measure, don't guess

`disk_space_gb` and the memory figures are what `ollamadiffuser recommend`
shows people before they start a 50 GB download. Please pull the model and
report what it really took (`du -sh` on the model directory). If you could
not run it, say so in the PR — an honest "untested on CUDA" is fine.

### Before opening the PR

```bash
pip install -e ".[dev]"
pytest tests/test_registry_yaml.py tests/test_model_registry.py
ollamadiffuser pull my-model && ollamadiffuser run my-model
```

Say in the PR which hardware you ran it on.

## Trying a model without a PR

Put the same YAML under `models:` in `~/.ollamadiffuser/models.yaml` (or point
`OLLAMADIFFUSER_MODEL_CONFIG` at a file). Entries there override the built-ins.

## Code changes

- Python 3.10+, `pytest` must pass. Tests mock the model libraries, so the
  suite runs in seconds with no GPU and no downloads.
- The default install stays compile-free. A dependency that needs a compiler
  or a platform-specific wheel belongs behind `ollamadiffuser enable <backend>`
  (see `_build_enable_command` in `ollamadiffuser/cli/commands.py`), not in
  `dependencies`.
- Bug reports with the full traceback and your OS, Python and
  `ollamadiffuser --version` get fixed fastest.
