"""Tests for the MLX inference strategy (issue #7, Phases 1 + 2).

These tests run on any platform — when not on Apple Silicon they
exercise the platform-check refusal path; when on Apple Silicon they
mock the mflux model class so no weights need to download.
"""
from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ollamadiffuser.core.config.settings import ModelConfig
from ollamadiffuser.core.inference.engine import _get_strategy
from ollamadiffuser.core.inference.strategies import mlx_strategy
from ollamadiffuser.core.inference.strategies.mlx_strategy import (
    MLXStrategy,
    SUPPORTED_MLX_VARIANTS,
    is_apple_silicon,
)

# The resolution tests import mflux for real — that is the point of them, since
# what they check is that our module paths and alias names still match theirs.
# So they need the optional backend present, not just Apple Silicon: without
# this they fail with ModuleNotFoundError on a machine that never installed it,
# which reads as "the strategy is broken" and is not.
_HAS_MFLUX = importlib.util.find_spec("mflux") is not None
_needs_mflux = pytest.mark.skipif(
    not is_apple_silicon() or not _HAS_MFLUX,
    reason="MLX resolution needs Apple Silicon and mflux installed",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_config(**overrides) -> ModelConfig:
    """Build a ModelConfig that points the strategy at mflux."""
    base = {
        "name": "flux.1-schnell-mlx",
        "path": "/tmp/unused",
        "model_type": "mlx",
        "variant": "mlx-q8",
        "parameters": {
            "mlx_variant": "flux1",
            "mlx_model_name": "schnell",
            "quantize": 8,
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
        },
    }
    base.update(overrides)
    try:
        return ModelConfig(**base)
    except TypeError:
        from types import SimpleNamespace
        return SimpleNamespace(**base)


def _stub_resolution(*, supports_negative=True):
    """Return (cls_mock, instance_mock, mflux_config_mock).

    ``instance_mock.generate_image`` is a plain function so that
    ``inspect.signature()`` returns the exact parameters the strategy
    is allowed to forward — MLXStrategy filters kwargs against that
    signature. Set ``supports_negative=False`` to simulate a class
    like Flux2Klein that has no ``negative_prompt`` parameter.
    Call records are kept on ``instance_mock._call_kwargs`` (a list of
    dicts) — assert against the last entry.
    """
    fake_pil = Image.new("RGB", (64, 64), color=(0, 128, 255))
    generated_obj = MagicMock(name="GeneratedImage", image=fake_pil)

    call_log: list = []

    if supports_negative:
        def fake_generate_image(
            seed, prompt, num_inference_steps=4, height=1024,
            width=1024, guidance=4.0, image_path=None,
            image_strength=None, scheduler="linear",
            negative_prompt=None,
        ):
            call_log.append({
                "seed": seed,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "guidance": guidance,
                "image_path": image_path,
                "image_strength": image_strength,
                "scheduler": scheduler,
                "negative_prompt": negative_prompt,
            })
            return generated_obj
    else:
        def fake_generate_image(
            seed, prompt, num_inference_steps=4, height=1024,
            width=1024, guidance=1.0, image_path=None,
            image_strength=None, scheduler="flow_match_euler_discrete",
        ):
            call_log.append({
                "seed": seed,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "guidance": guidance,
                "image_path": image_path,
                "image_strength": image_strength,
                "scheduler": scheduler,
            })
            return generated_obj

    instance = MagicMock(name="MfluxInstance")
    instance.generate_image = fake_generate_image
    instance._call_kwargs = call_log  # for test assertions

    cls = MagicMock(name="MfluxClass", return_value=instance)
    mflux_config = MagicMock(name="MfluxModelConfig")
    return cls, instance, mflux_config


# --------------------------------------------------------------------------
# Dispatch + module constants
# --------------------------------------------------------------------------

class TestDispatch:
    def test_engine_dispatches_mlx_model_type(self):
        strategy = _get_strategy("mlx")
        assert isinstance(strategy, MLXStrategy)

    def test_supported_variants_constant(self):
        # Phase 1 + 2 + 2.5 (the FLUX/Z-Image/Qwen families), plus the eight
        # families mflux added after them, plus FLUX.2's editor, plus Qwen-Image-2.1,
        # plus Z-Image-Turbo's union ControlNet — twenty.
        assert SUPPORTED_MLX_VARIANTS == frozenset({
            "flux1", "flux1-kontext",
            "flux1-fill", "flux1-redux", "flux1-depth", "flux1-controlnet",
            "flux2", "flux2-edit", "z_image", "z_image-controlnet", "qwen-image",
            "krea2", "boogu", "ernie-image", "lens", "ideogram4",
            "fibo", "fibo-edit", "seedvr2", "qwen21",
        })

    def test_flux2_edit_does_not_demand_an_image_here(self):
        """Not listed as image-required, even though it is an editor.

        Its signature takes `image_paths | None`, which reads like one model
        for both jobs. It is not: called with no reference, mflux 0.19.2
        raises a concatenate type error out of mlx. The check stays out of
        _VARIANT_REQUIRED_INPUTS anyway, because the sentence it would print
        ("pass image=") is about our kwarg names rather than about what
        happened, and a future mflux may well support it.
        """
        assert "flux2-edit" not in mlx_strategy._VARIANT_REQUIRED_INPUTS

    def test_alias_routed_families_are_supported(self):
        # Every alias-routed family must also be in the supported set, or
        # load() refuses a variant the resolver can actually build.
        assert set(mlx_strategy._ALIAS_ROUTED) <= SUPPORTED_MLX_VARIANTS

    def test_image_only_families_require_an_image(self):
        # An editor and an upscaler with no input image is a crash deep in
        # mflux; the strategy has to catch it first.
        for variant in ("fibo-edit", "seedvr2"):
            assert "image" in mlx_strategy._VARIANT_REQUIRED_INPUTS[variant]

    def test_is_apple_silicon_returns_bool(self):
        assert isinstance(is_apple_silicon(), bool)

    def test_looks_like_model_dir(self, tmp_path):
        assert mlx_strategy.looks_like_model_dir(None) is False
        assert mlx_strategy.looks_like_model_dir(str(tmp_path / "nope")) is False
        assert mlx_strategy.looks_like_model_dir(str(tmp_path)) is False
        (tmp_path / "model_index.json").write_text("{}")
        assert mlx_strategy.looks_like_model_dir(str(tmp_path)) is True
        flat = tmp_path / "flat"
        (flat / "sub").mkdir(parents=True)
        (flat / "sub" / "w.safetensors").write_bytes(b"\x00")
        assert mlx_strategy.looks_like_model_dir(str(flat)) is True


# --------------------------------------------------------------------------
# Platform refusal
# --------------------------------------------------------------------------

class TestPlatformGuard:
    def test_refuses_non_apple_silicon(self):
        s = MLXStrategy()
        config = _make_config()
        with patch.object(mlx_strategy, "is_apple_silicon", return_value=False):
            assert s.load(config, device="cuda") is False
        assert s.pipeline is None
        assert s.is_loaded is False


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestConfigValidation:
    def test_rejects_unknown_variant(self):
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "nonsense",
            "mlx_model_name": "schnell",
        })
        assert s.load(config, device="mps") is False
        assert s.is_loaded is False

    def test_rejects_missing_model_name(self):
        s = MLXStrategy()
        config = _make_config(parameters={"mlx_variant": "flux1"})
        assert s.load(config, device="mps") is False

    def test_rejects_invalid_quantize(self):
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "flux1",
            "mlx_model_name": "schnell",
            "quantize": 3,
        })
        assert s.load(config, device="mps") is False


# --------------------------------------------------------------------------
# Resolution per variant (no actual mflux call — verifies dispatch logic)
# --------------------------------------------------------------------------

@_needs_mflux
class TestVariantResolution:
    def test_flux1_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1", "schnell")
        assert cls.__name__ == "Flux1"

    def test_flux1_kontext_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-kontext", "dev")
        assert cls.__name__ == "Flux1Kontext"

    def test_flux1_kontext_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-kontext"):
            MLXStrategy._resolve_model_and_config("flux1-kontext", "bogus")

    def test_flux2_klein_4b_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux2", "klein-4b")
        assert cls.__name__ == "Flux2Klein"

    def test_flux2_klein_9b_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux2", "klein-9b")
        assert cls.__name__ == "Flux2Klein"

    def test_flux2_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux2"):
            MLXStrategy._resolve_model_and_config("flux2", "klein-99b")

    def test_z_image_turbo_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("z_image", "z-image-turbo")
        assert cls.__name__ == "ZImage"

    def test_z_image_rejects_bad_name(self):
        with pytest.raises(ValueError, match="z_image"):
            MLXStrategy._resolve_model_and_config("z_image", "z-image-pro")

    def test_qwen_image_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("qwen-image", "qwen-image")
        assert cls.__name__ == "QwenImage"

    def test_qwen_image_edit_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("qwen-image", "qwen-image-edit")
        assert cls.__name__ == "QwenImage"

    def test_qwen_image_rejects_bad_name(self):
        with pytest.raises(ValueError, match="qwen-image"):
            MLXStrategy._resolve_model_and_config("qwen-image", "qwen-bogus")

    # --- The families mflux added after our May 2026 line ---

    @pytest.mark.parametrize("variant,alias,expected", [
        ("krea2",       "krea-2",             "Krea2"),
        ("boogu",       "boogu-image-turbo",  "BooguImage"),
        ("ernie-image", "ernie-image-turbo",  "ErnieImage"),
        ("lens",        "lens-turbo",         "LensImage"),
        ("ideogram4",   "ideogram4-fp8",      "Ideogram4"),
        ("fibo",        "fibo",               "FIBO"),
        ("fibo-edit",   "fibo-edit",          "FIBOEdit"),
        ("seedvr2",     "seedvr2-3b",         "SeedVR2"),
        ("qwen21",      "qwen-image-2.1",     "QwenImage21"),
    ])
    def test_alias_routed_family_resolves(self, variant, alias, expected):
        cls, config = MLXStrategy._resolve_model_and_config(variant, alias)
        assert cls.__name__ == expected
        assert config is not None

    def test_alias_routed_family_rejects_bad_alias(self):
        # mflux raises on an alias it does not know, and its message lists the
        # ones it does — a better error than one written here.
        with pytest.raises(Exception):
            MLXStrategy._resolve_model_and_config("krea2", "krea-9000")

    # --- Phase 2.5: additional FLUX.1 family variants ---

    def test_flux1_fill_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-fill", "dev")
        assert cls.__name__ == "Flux1Fill"

    def test_flux1_fill_catvton_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-fill", "catvton")
        assert cls.__name__ == "Flux1Fill"

    def test_flux1_fill_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-fill"):
            MLXStrategy._resolve_model_and_config("flux1-fill", "bogus")

    def test_flux1_redux_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-redux", "dev")
        assert cls.__name__ == "Flux1Redux"

    def test_flux1_redux_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-redux"):
            MLXStrategy._resolve_model_and_config("flux1-redux", "bogus")

    def test_flux1_depth_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-depth", "dev")
        assert cls.__name__ == "Flux1Depth"

    def test_flux1_depth_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-depth"):
            MLXStrategy._resolve_model_and_config("flux1-depth", "bogus")

    def test_flux1_controlnet_canny_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-controlnet", "canny")
        assert cls.__name__ == "Flux1Controlnet"

    def test_flux1_controlnet_upscaler_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-controlnet", "upscaler")
        assert cls.__name__ == "Flux1Controlnet"

    def test_flux1_controlnet_canny_schnell_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config(
            "flux1-controlnet", "canny-schnell"
        )
        assert cls.__name__ == "Flux1Controlnet"

    def test_flux1_controlnet_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-controlnet"):
            MLXStrategy._resolve_model_and_config("flux1-controlnet", "pose")


# --------------------------------------------------------------------------
# Loading + generation (mflux mocked out)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestLoadAndGenerate:
    def test_load_uses_the_weights_pull_downloaded(self, tmp_path):
        """No second copy: `pull` already has them."""
        (tmp_path / "model_index.json").write_text("{}")
        (tmp_path / "weights.safetensors").write_bytes(b"\x00")
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_and_config",
                          return_value=(cls_mock, mflux_config_mock)):
            ok = s.load(_make_config(path=str(tmp_path)), device="mps")
        assert ok is True
        cls_mock.assert_called_once_with(
            quantize=8, model_config=mflux_config_mock, model_path=str(tmp_path)
        )

    def test_load_falls_back_to_the_hub_when_the_local_copy_will_not_do(self, tmp_path):
        """A filtered download can be missing a file; the hub still has it."""
        (tmp_path / "model_index.json").write_text("{}")
        cls_mock, instance, mflux_config_mock = _stub_resolution()
        calls = []

        def _maybe_fail(**kwargs):
            calls.append(kwargs)
            if "model_path" in kwargs:
                raise FileNotFoundError("text_encoder/model.safetensors")
            return instance

        cls_mock.side_effect = _maybe_fail
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_and_config",
                          return_value=(cls_mock, mflux_config_mock)):
            ok = s.load(_make_config(path=str(tmp_path)), device="mps")
        assert ok is True
        assert len(calls) == 2 and "model_path" not in calls[1]

    def test_every_mlx_call_happens_on_one_thread(self, tmp_path):
        """MLX streams are per-thread: build and run must share a thread."""
        import threading
        (tmp_path / "model_index.json").write_text("{}")
        threads = []
        cls_mock, instance, mflux_config_mock = _stub_resolution()

        def _record(**kwargs):
            threads.append(threading.current_thread().name)
            return instance

        cls_mock.side_effect = _record
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_and_config",
                          return_value=(cls_mock, mflux_config_mock)):
            assert s.load(_make_config(path=str(tmp_path)), device="mps") is True
        s.generate("a cat", seed=1)
        # The generate call goes through the same runner; the log records the
        # constructor's thread, and it is not the caller's.
        assert threads and threads[0].startswith("mlx")
        assert threads[0] != threading.current_thread().name
        s.unload()

    def test_load_calls_class_constructor_with_quantize_and_config(self):
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            ok = s.load(_make_config(), device="mps")
        assert ok is True
        cls_mock.assert_called_once_with(quantize=8, model_config=mflux_config_mock)
        assert s.is_loaded
        assert s.device == "mps"
        assert s._variant == "flux1"

    def test_load_reports_failure_on_mflux_import_error(self):
        s = MLXStrategy()

        def raise_import(*a, **kw):
            raise ImportError("mflux missing")
        with patch.object(
            MLXStrategy, "_resolve_model_and_config", side_effect=raise_import
        ):
            assert s.load(_make_config(), device="mps") is False
        assert s.is_loaded is False

    def test_load_reports_failure_on_bad_model_name(self):
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "flux2",
            "mlx_model_name": "klein-bogus",
        })
        # Don't mock — let resolution actually raise ValueError.
        assert s.load(config, device="mps") is False

    def test_generate_forwards_params_and_returns_pil(self):
        cls_mock, inst_mock, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")

        out = s.generate(
            prompt="A dog",
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
            seed=42,
        )
        assert isinstance(out, Image.Image)
        assert len(inst_mock._call_kwargs) == 1
        called = inst_mock._call_kwargs[-1]
        assert called["prompt"] == "A dog"
        assert called["seed"] == 42
        assert called["num_inference_steps"] == 4
        assert called["guidance"] == 0.0
        assert called["height"] == 512
        assert called["width"] == 512

    def test_generate_filters_kwargs_for_variants_without_negative_prompt(self):
        """Flux2Klein has no negative_prompt — strategy must not pass it."""
        cls_mock, inst_mock, mflux_config_mock = _stub_resolution(supports_negative=False)
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "flux2",
            "mlx_model_name": "klein-4b",
            "quantize": 8,
        })
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(config, device="mps")

        # If the strategy fails to filter, this raises TypeError because
        # our fake fn has no `negative_prompt` parameter.
        s.generate(prompt="hi", negative_prompt="ugly", seed=1)
        called = inst_mock._call_kwargs[-1]
        # And the call was actually made (no exception).
        assert called["prompt"] == "hi"

    def test_kontext_without_image_raises(self):
        """flux1-kontext is an image-editor — must reject if no image given."""
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "flux1-kontext",
            "mlx_model_name": "dev",
            "quantize": 8,
        })
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(config, device="mps")

        with pytest.raises(ValueError, match="requires kwargs"):
            s.generate(prompt="make it sunset")  # no image=

    @pytest.mark.parametrize("variant,model_name,missing_kwargs,passed_kwargs", [
        ("flux1-fill", "dev", ["image", "mask_image"], {}),
        ("flux1-fill", "dev", ["mask_image"], {"image": "/tmp/x.png"}),
        ("flux1-redux", "dev", ["redux_images"], {}),
        ("flux1-depth", "dev", ["image"], {}),
        ("flux1-controlnet", "canny", ["control_image"], {}),
    ])
    def test_variant_required_input_enforcement(
        self, variant, model_name, missing_kwargs, passed_kwargs
    ):
        """Each variant's required-input contract is enforced before mflux call."""
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": variant,
            "mlx_model_name": model_name,
            "quantize": 8,
        })
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(config, device="mps")

        with pytest.raises(ValueError, match="requires kwargs"):
            s.generate(prompt="x", **passed_kwargs)

    def test_redux_accepts_scalar_or_list(self):
        """redux_images may be a single PIL/path or a list of them."""
        from ollamadiffuser.core.inference.strategies.mlx_strategy import (
            MLXStrategy as _S,
        )
        # Scalar PIL → list of 1
        pil = Image.new("RGB", (8, 8))
        out = _S._materialize_image_path_list(pil)
        assert isinstance(out, list) and len(out) == 1

        # List of PIL → list of paths
        out2 = _S._materialize_image_path_list([pil, pil])
        assert isinstance(out2, list) and len(out2) == 2

        # None → None
        assert _S._materialize_image_path_list(None) is None

    def test_generate_without_seed_gets_random_int(self):
        cls_mock, inst_mock, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")

        s.generate(prompt="hi")
        called = inst_mock._call_kwargs[-1]
        assert isinstance(called["seed"], int)
        assert 0 <= called["seed"] < 2**31

    def test_generate_before_load_raises(self):
        s = MLXStrategy()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            s.generate(prompt="hi")

    def test_unload_clears_state_and_variant(self):
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")
        assert s.is_loaded
        assert s._variant == "flux1"
        s.unload()
        assert s.is_loaded is False
        assert s._mlx_model is None
        assert s.pipeline is None
        assert s._variant is None


# --------------------------------------------------------------------------
# LoRA: mflux bakes LoRAs in when the model is built
# --------------------------------------------------------------------------

class _FakeMflux:
    """Stands in for an mflux class: records how each instance was built."""
    built: list = []
    fail_on: str = ""

    def __init__(self, quantize=None, model_path=None, lora_paths=None,
                 lora_scales=None, model_config=None):
        if self.fail_on and any(self.fail_on in p for p in (lora_paths or [])):
            raise ValueError("unmappable LoRA")
        type(self).built.append({
            "model_path": model_path,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
        })

    def generate_image(self, seed, prompt, num_inference_steps=4, height=1024,
                       width=1024, guidance=1.0):
        return MagicMock(image=Image.new("RGB", (64, 64), color=(0, 128, 255)))


class _FakeNoLora:
    def __init__(self, quantize=None, model_path=None, model_config=None):
        pass


@pytest.fixture
def lora_strategy(tmp_path):
    """A loaded strategy over a real model directory, built by _FakeMflux."""
    model_dir = tmp_path / "model"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "transformer" / "config.json").write_text("{}")
    _FakeMflux.built = []
    _FakeMflux.fail_on = ""
    s = MLXStrategy()
    with patch("ollamadiffuser.core.inference.strategies.mlx_strategy.is_apple_silicon",
               return_value=True), \
         patch.object(MLXStrategy, "_resolve_model_and_config",
                      return_value=(_FakeMflux, MagicMock(name="mflux_config"))):
        assert s.load(_make_config(path=str(model_dir)), device="mps")
    return s, model_dir


def _lora_file(tmp_path, name="style.safetensors"):
    f = tmp_path / name
    f.write_bytes(b"x")
    return f


class TestLoRA:
    def test_loading_a_lora_rebuilds_with_it_baked_in(self, lora_strategy, tmp_path):
        s, model_dir = lora_strategy
        f = _lora_file(tmp_path)
        assert s.load_lora_runtime(str(f), scale=0.8) is True
        last = _FakeMflux.built[-1]
        assert last["lora_paths"] == [str(f)]
        assert last["lora_scales"] == [0.8]
        # Still the downloaded weights, not a trip to the hub.
        assert last["model_path"] == str(model_dir)
        assert s.current_lora["stack"] == [{"path": str(f), "scale": 0.8}]
        assert s.is_loaded

    def test_loras_stack_and_reloading_one_changes_its_scale(self, lora_strategy, tmp_path):
        s, _ = lora_strategy
        speed, style = _lora_file(tmp_path, "speed.safetensors"), _lora_file(tmp_path)
        s.load_lora_runtime(str(speed), scale=1.0)
        s.load_lora_runtime(str(style), scale=0.7)
        assert _FakeMflux.built[-1]["lora_paths"] == [str(speed), str(style)]
        s.load_lora_runtime(str(speed), scale=0.5)
        assert _FakeMflux.built[-1]["lora_paths"] == [str(style), str(speed)]
        assert _FakeMflux.built[-1]["lora_scales"] == [0.7, 0.5]

    def test_unload_builds_without_loras(self, lora_strategy, tmp_path):
        s, _ = lora_strategy
        s.load_lora_runtime(str(_lora_file(tmp_path)))
        assert s.unload_lora() is True
        assert _FakeMflux.built[-1]["lora_paths"] is None
        assert s.current_lora is None
        assert s.unload_lora() is False  # nothing left to drop

    def test_directory_with_a_weight_name(self, lora_strategy, tmp_path):
        s, _ = lora_strategy
        d = tmp_path / "lora-dir"
        d.mkdir()
        (d / "a.safetensors").write_bytes(b"x")
        (d / "b.safetensors").write_bytes(b"x")
        assert s.load_lora_runtime(str(d), weight_name="b.safetensors")
        assert _FakeMflux.built[-1]["lora_paths"] == [str(d / "b.safetensors")]

    def test_ambiguous_directory_is_refused_without_a_rebuild(self, lora_strategy, tmp_path, caplog):
        s, _ = lora_strategy
        d = tmp_path / "lora-dir"
        d.mkdir()
        (d / "a.safetensors").write_bytes(b"x")
        (d / "b.safetensors").write_bytes(b"x")
        builds = len(_FakeMflux.built)
        with caplog.at_level("ERROR"):
            assert s.load_lora_runtime(str(d)) is False
        assert len(_FakeMflux.built) == builds
        assert any("weight_name" in r.message for r in caplog.records)

    def test_hub_repo_is_passed_as_repo_colon_file(self, lora_strategy):
        s, _ = lora_strategy
        assert s.load_lora_runtime("lightx2v/Qwen-Image-Lightning",
                                   weight_name="Qwen-Image-Lightning-8steps-V1.0-bf16.safetensors")
        assert _FakeMflux.built[-1]["lora_paths"] == [
            "lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-8steps-V1.0-bf16.safetensors"
        ]

    def test_failed_lora_restores_the_previous_model_and_never_goes_to_the_hub(
            self, lora_strategy, tmp_path):
        s, model_dir = lora_strategy
        good = _lora_file(tmp_path, "good.safetensors")
        s.load_lora_runtime(str(good))
        _FakeMflux.fail_on = "bad"
        assert s.load_lora_runtime(str(_lora_file(tmp_path, "bad.safetensors"))) is False
        # Every build used the downloaded weights; none fell back to the hub.
        assert all(b["model_path"] == str(model_dir) for b in _FakeMflux.built)
        assert _FakeMflux.built[-1]["lora_paths"] == [str(good)]
        assert s.is_loaded

    def test_family_without_loras_says_so(self, tmp_path, caplog):
        s = MLXStrategy()
        with patch("ollamadiffuser.core.inference.strategies.mlx_strategy.is_apple_silicon",
                   return_value=True), \
             patch.object(MLXStrategy, "_resolve_model_and_config",
                          return_value=(_FakeNoLora, MagicMock())):
            assert s.load(_make_config(), device="mps")
        with caplog.at_level("ERROR"):
            assert s.load_lora_runtime(str(_lora_file(tmp_path))) is False
        assert any("takes no LoRAs" in r.message for r in caplog.records)

    def test_model_subdir_points_the_build_at_the_subfolder(self, tmp_path):
        tree = tmp_path / "z-anime" / "diffusers"
        (tree / "transformer").mkdir(parents=True)
        (tree / "transformer" / "config.json").write_text("{}")
        _FakeMflux.built = []
        _FakeMflux.fail_on = ""
        cfg = _make_config(path=str(tmp_path / "z-anime"))
        cfg.parameters = {**cfg.parameters, "model_subdir": "diffusers"}
        s = MLXStrategy()
        with patch("ollamadiffuser.core.inference.strategies.mlx_strategy.is_apple_silicon",
                   return_value=True), \
             patch.object(MLXStrategy, "_resolve_model_and_config",
                          return_value=(_FakeMflux, MagicMock())):
            assert s.load(cfg, device="mps")
        assert _FakeMflux.built[-1]["model_path"] == str(tree)


class _FakeZImageControlnet:
    """ZImageTurboControlnet's generate_image signature, recording its calls."""
    calls: list = []

    def __init__(self, quantize=None, model_path=None, lora_paths=None,
                 lora_scales=None, model_config=None):
        pass

    def generate_image(self, *, seed, prompt, controls, num_inference_steps=8,
                       height=1024, width=1024, controlnet_strength=0.8, scheduler="linear"):
        type(self).calls.append({"controls": controls, "controlnet_strength": controlnet_strength})
        return MagicMock(image=Image.new("RGB", (64, 64), color=(0, 128, 255)))


class TestControlNet:
    def _load(self, variant, cls=_FakeZImageControlnet):
        cfg = _make_config()
        cfg.parameters = {**cfg.parameters, "mlx_variant": variant,
                          "mlx_model_name": "z-image-turbo-controlnet-union-2.1"}
        s = MLXStrategy()
        with patch("ollamadiffuser.core.inference.strategies.mlx_strategy.is_apple_silicon",
                   return_value=True), \
             patch.object(MLXStrategy, "_resolve_model_and_config",
                          return_value=(cls, MagicMock())):
            assert s.load(cfg, device="mps")
        return s

    def test_controlnet_variants_say_so_to_the_controlnet_endpoint(self):
        assert self._load("z_image-controlnet").is_controlnet_pipeline is True
        assert self._load("flux1-controlnet").is_controlnet_pipeline is True
        assert self._load("z_image", cls=_FakeMflux).is_controlnet_pipeline is False

    def test_z_image_control_becomes_a_typed_control_spec(self, tmp_path):
        _FakeZImageControlnet.calls = []
        s = self._load("z_image-controlnet")
        edges = tmp_path / "edges.png"
        Image.new("RGB", (64, 64)).save(edges)
        s.generate("a lighthouse", control_image=str(edges), control_type="canny",
                   controlnet_conditioning_scale=0.7)
        call = _FakeZImageControlnet.calls[-1]
        (spec,) = call["controls"]
        assert spec.type.value == "canny"
        assert str(spec.image_path) == str(edges)
        # diffusers' name for the strength reaches mflux's parameter.
        assert call["controlnet_strength"] == 0.7

    def test_z_image_control_without_a_type_is_refused(self, tmp_path):
        s = self._load("z_image-controlnet")
        edges = tmp_path / "edges.png"
        Image.new("RGB", (64, 64)).save(edges)
        with pytest.raises(ValueError, match="control_type"):
            s.generate("a lighthouse", control_image=str(edges))

    def test_z_image_control_without_an_image_is_refused(self):
        s = self._load("z_image-controlnet")
        with pytest.raises(ValueError, match="control_image"):
            s.generate("a lighthouse", control_type="depth")


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestInfo:
    def test_get_info_reports_backend_mlx(self):
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")
        info = s.get_info()
        assert info["backend"] == "mlx"
        assert info["mflux_variant"] == "flux1"


# --------------------------------------------------------------------------
# The registry and mflux must agree on which repo an alias means
# --------------------------------------------------------------------------

@_needs_mflux
class TestRegistryPullsWhatMfluxLoads:
    """`pull` downloads `repo_id`; mflux loads whatever its alias resolves to.

    When those differ the user pays for it twice over: qwen-image-mlx pointed
    at Qwen/Qwen-Image while mflux's `qwen-image` alias meant
    Qwen/Qwen-Image-2512 — 58 GB of a checkpoint mflux was never asked for.
    An mflux release that moves an alias to a newer checkpoint will trip this,
    which is the point: the registry entry has to move with it.
    """

    def test_alias_routed_entries_point_at_the_repo_mflux_resolves(self):
        from mflux.models.common.config.model_config import ModelConfig
        from ollamadiffuser.core.config.model_registry import ModelRegistry

        checked = 0
        for name, cfg in ModelRegistry()._registry.items():
            if cfg.get("model_type") != "mlx":
                continue
            params = cfg.get("parameters") or {}
            variant = params.get("mlx_variant")
            # Only families whose config comes from the alias alone. The FLUX
            # variants use short names ("dev", "canny") that mean something
            # else, or nothing, to ModelConfig.from_name.
            if variant not in mlx_strategy._ALIAS_ROUTED and variant != "qwen-image":
                continue
            resolved = ModelConfig.from_name(params["mlx_model_name"]).model_name
            assert cfg["repo_id"] == resolved, (
                f"{name}: pull fetches {cfg['repo_id']!r} but mflux loads "
                f"{resolved!r} for alias {params['mlx_model_name']!r}"
            )
            checked += 1
        assert checked >= 8, "the filter above stopped matching anything"
