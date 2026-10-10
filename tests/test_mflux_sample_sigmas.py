"""A checkpoint's own sigma table (Qwen-Image-2.1-Turbo) drives mflux's sampler."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mflux")

from ollamadiffuser.core.inference.strategies.mflux_sample_sigmas import (
    read_sample_sigmas,
    register,
)

TURBO = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def _model_dir(tmp_path, **index):
    d = tmp_path / "turbo"
    (d / "transformer").mkdir(parents=True)
    (d / "transformer" / "config.json").write_text("{}")
    (d / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline", **index}))
    return d


class TestReading:
    def test_the_turbo_table_is_read(self, tmp_path):
        assert read_sample_sigmas(str(_model_dir(tmp_path, sample_sigmas=TURBO))) == TURBO

    def test_no_table_or_no_index_means_none(self, tmp_path):
        assert read_sample_sigmas(str(_model_dir(tmp_path))) is None
        assert read_sample_sigmas(str(tmp_path / "missing")) is None
        assert read_sample_sigmas(None) is None

    def test_a_table_that_is_not_a_schedule_is_ignored(self, tmp_path):
        assert read_sample_sigmas(str(_model_dir(tmp_path, sample_sigmas=[0.4, 0.9, 1.0]))) is None


class TestScheduler:
    def test_registered_schedule_walks_the_nodes_then_zero(self):
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY
        name = register(TURBO)
        cls = SCHEDULER_REGISTRY[name]
        sched = cls(SimpleNamespace(num_inference_steps=8))
        assert [round(float(x), 6) for x in sched.sigmas.tolist()] == TURBO + [0.0]
        assert sched.timesteps.tolist() == list(range(8))
        latents = mx.ones((1, 4))
        out = sched.step(noise=mx.ones((1, 4)), timestep=7, latents=latents)
        assert abs(float(out[0, 0]) - (1.0 - 0.414568)) < 1e-6   # Euler to sigma 0
        assert register(TURBO) == name                            # registered once

    def test_another_step_count_is_refused(self):
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY
        cls = SCHEDULER_REGISTRY[register(TURBO)]
        with pytest.raises(ValueError, match="8-step schedule"):
            cls(SimpleNamespace(num_inference_steps=25))


class _FakeQwen21:
    calls: list = []

    def __init__(self, quantize=None, model_path=None, model_config=None, lora_paths=None,
                 lora_scales=None, bake_lora=True, compute_precision=None):
        pass

    def generate_image(self, seed, prompt, num_inference_steps=40, height=1024, width=1024,
                       guidance=1.0, image_path=None, image_strength=None, scheduler="linear",
                       negative_prompt=None):
        type(self).calls.append({"steps": num_inference_steps, "scheduler": scheduler})
        return MagicMock(image=Image.new("RGB", (64, 64), color=(0, 128, 255)))


def _load(tmp_path, **params):
    from ollamadiffuser.core.config.settings import ModelConfig
    from ollamadiffuser.core.inference.strategies.mlx_strategy import MLXStrategy
    d = _model_dir(tmp_path, sample_sigmas=TURBO)
    cfg = ModelConfig(name="qwen-image-2.1-turbo-mlx", path=str(d), model_type="mlx",
                      parameters={"mlx_variant": "qwen21", "mlx_model_name": "qwen-image-2.1",
                                  "quantize": 8, "num_inference_steps": 8, **params})
    s = MLXStrategy()
    with patch("ollamadiffuser.core.inference.strategies.mlx_strategy.is_apple_silicon", return_value=True), \
         patch.object(MLXStrategy, "_resolve_model_and_config", return_value=(_FakeQwen21, MagicMock())):
        assert s.load(cfg, "mps")
    return s


class TestStrategyUsesIt:
    def test_the_checkpoints_schedule_and_step_count_are_used(self, tmp_path):
        _FakeQwen21.calls = []
        s = _load(tmp_path)
        s.generate("a cabin", num_inference_steps=25)
        call = _FakeQwen21.calls[-1]
        assert call["steps"] == 8 and call["scheduler"].startswith("ollamadiffuser_sample_sigmas_")

    def test_a_named_scheduler_wins(self, tmp_path):
        _FakeQwen21.calls = []
        s = _load(tmp_path)
        s.generate("a cabin", num_inference_steps=6, scheduler="viggle_turbo")
        assert _FakeQwen21.calls[-1] == {"steps": 6, "scheduler": "viggle_turbo"}

    def test_an_entry_scheduler_turns_it_off(self, tmp_path):
        s = _load(tmp_path, scheduler="linear")
        assert s._fixed_schedule is None
