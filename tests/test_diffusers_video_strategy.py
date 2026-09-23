"""Video through diffusers: Wan and LTX-2 off Apple Silicon.

A fake ``diffusers`` module stands in for the real one — the pipelines are
tens of gigabytes, and what is under test is the plumbing around them: which
pipeline runs, what it is called with, and how the mp4 gets written.
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from ollamadiffuser.core.config.settings import ModelConfig
from ollamadiffuser.core.inference.engine import _get_strategy
from ollamadiffuser.core.inference.strategies.diffusers_video_strategy import (
    DiffusersVideoStrategy,
    check_frames,
    frames_for_seconds,
)


class _FakeVAE:
    def __init__(self):
        self.tiled = False
        self.dtype = None

    def enable_tiling(self):
        self.tiled = True

    def to(self, dtype=None):
        self.dtype = dtype
        return self


class _FakeWan:
    calls: list = []

    def __init__(self):
        self.vae = _FakeVAE()
        self.device = None
        self.offloaded = False
        self.source = None

    @classmethod
    def from_pretrained(cls, path, torch_dtype=None):
        pipe = cls()
        pipe.path = path
        return pipe

    @classmethod
    def from_pipe(cls, pipe):
        new = cls()
        new.source = pipe
        return new

    def to(self, device):
        self.device = device
        return self

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def remove_all_hooks(self):
        pass

    def __call__(self, prompt=None, negative_prompt=None, num_frames=81, width=832,
                 height=480, num_inference_steps=50, guidance_scale=5.0, generator=None,
                 output_type="np"):
        type(self).calls.append({"pipe": type(self).__name__, "prompt": prompt,
                                 "num_frames": num_frames, "width": width, "height": height,
                                 "num_inference_steps": num_inference_steps,
                                 "guidance_scale": guidance_scale, "negative_prompt": negative_prompt})
        return SimpleNamespace(frames=[np.zeros((num_frames, 8, 8, 3))])


class _FakeWanI2V(_FakeWan):
    def __call__(self, image=None, prompt=None, negative_prompt=None, num_frames=81, width=832,
                 height=480, num_inference_steps=50, guidance_scale=5.0, generator=None,
                 output_type="np"):
        type(self).calls.append({"pipe": "i2v", "image": image, "num_frames": num_frames})
        return SimpleNamespace(frames=[np.zeros((num_frames, 8, 8, 3))])


class _FakeLTX2(_FakeWan):
    def __init__(self):
        super().__init__()
        self.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))

    def __call__(self, prompt=None, negative_prompt=None, num_frames=None, width=768, height=512,
                 frame_rate=24.0, num_inference_steps=30, sigmas=None, guidance_scale=3.0,
                 generator=None, enable_prompt_enhancement=False, output_type="pil"):
        type(self).calls.append({"pipe": "ltx2", "sigmas": sigmas, "frame_rate": frame_rate,
                                 "num_frames": num_frames, "guidance_scale": guidance_scale,
                                 "enable_prompt_enhancement": enable_prompt_enhancement})
        return SimpleNamespace(frames=[np.zeros((num_frames, 8, 8, 3))],
                               audio=[np.zeros((2, 100), dtype=np.float32)])


@pytest.fixture
def fake_diffusers(monkeypatch):
    written = []

    def encode_video(video, fps, output_path, audio=None, audio_sample_rate=None):
        written.append({"frames": len(video), "fps": fps, "path": output_path,
                        "audio": audio is not None, "rate": audio_sample_rate})
        open(output_path, "wb").write(b"mp4")

    mod = types.ModuleType("diffusers")
    mod.__version__ = "0.40.0"
    mod.WanPipeline = _FakeWan
    mod.WanImageToVideoPipeline = _FakeWanI2V
    mod.LTX2Pipeline = _FakeLTX2
    utils = types.ModuleType("diffusers.utils")
    utils.encode_video = encode_video
    mod.utils = utils
    ltx2_utils = types.ModuleType("diffusers.pipelines.ltx2.utils")
    ltx2_utils.DISTILLED_SIGMA_VALUES = [1.0, 0.99, 0.5, 0.0]
    monkeypatch.setitem(sys.modules, "diffusers", mod)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines", types.ModuleType("diffusers.pipelines"))
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.ltx2", types.ModuleType("diffusers.pipelines.ltx2"))
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.ltx2.utils", ltx2_utils)
    _FakeWan.calls = []
    _FakeWanI2V.calls = []
    _FakeLTX2.calls = []
    return written


def _wan_config(**params):
    base = {
        "pipeline_class": "WanPipeline",
        "i2v_pipeline_class": "WanImageToVideoPipeline",
        "frame_multiple": 4, "frame_rate": 24, "num_frames": 121,
        "width": 1280, "height": 704, "num_inference_steps": 50, "guidance_scale": 5.0,
        "vae_dtype": "float32", "enable_cpu_offload": True,
    }
    base.update(params)
    return ModelConfig(name="wan2.2-ti2v-5b", path="/models/wan", model_type="diffusers-video",
                       parameters=base)


def _loaded(config, device="cuda"):
    s = DiffusersVideoStrategy()
    assert s.load(config, device) is True
    return s


class TestFrames:
    def test_seconds_round_to_the_models_grid(self):
        assert frames_for_seconds(5, 24, 4) == 121
        assert frames_for_seconds(5, 24, 8) == 121
        assert frames_for_seconds(2, 16, 4) == 33

    def test_illegal_counts_name_the_nearest_legal_one(self):
        assert check_frames(81, 4) == 81
        with pytest.raises(ValueError, match="nearest legal is 81"):
            check_frames(80, 4)


class TestDispatch:
    def test_engine_maps_the_type(self):
        assert isinstance(_get_strategy("diffusers-video"), DiffusersVideoStrategy)

    def test_image_path_is_refused(self, fake_diffusers):
        s = _loaded(_wan_config())
        with pytest.raises(RuntimeError, match="video model"):
            s.generate("a cat")


class TestLoad:
    def test_cuda_offloads_tiles_the_vae_and_keeps_it_fp32(self, fake_diffusers):
        import torch
        s = _loaded(_wan_config())
        assert s.pipeline.offloaded is True
        assert s.pipeline.vae.tiled is True
        assert s.pipeline.vae.dtype == torch.float32

    def test_without_offload_it_moves_to_the_device(self, fake_diffusers):
        s = _loaded(_wan_config(enable_cpu_offload=False))
        assert s.pipeline.device == "cuda"
        assert s.pipeline.offloaded is False

    def test_ltx2_refuses_the_apple_gpu_and_names_the_mlx_entries(self, fake_diffusers, caplog):
        cfg = ModelConfig(name="ltx", path="/m", model_type="diffusers-video",
                          parameters={"pipeline_class": "LTX2Pipeline"})
        with caplog.at_level("ERROR"):
            assert DiffusersVideoStrategy().load(cfg, "mps") is False
        assert any("ltx-2.3-mlx" in r.message for r in caplog.records)

    def test_unknown_pipeline_class_fails_cleanly(self, fake_diffusers):
        assert DiffusersVideoStrategy().load(_wan_config(pipeline_class="WanNextPipeline"), "cuda") is False


class TestGenerateVideo:
    def test_text_to_video_with_the_registry_defaults(self, fake_diffusers, tmp_path):
        s = _loaded(_wan_config())
        out = s.generate_video("a cat surfing", output=str(tmp_path / "cat.mp4"))
        call = _FakeWan.calls[-1]
        assert call["num_frames"] == 121 and (call["width"], call["height"]) == (1280, 704)
        assert call["num_inference_steps"] == 50 and call["guidance_scale"] == 5.0
        assert out.exists()
        assert fake_diffusers[-1] == {"frames": 121, "fps": 24, "path": str(out),
                                      "audio": False, "rate": None}

    def test_caller_values_win_and_seconds_become_frames(self, fake_diffusers, tmp_path):
        s = _loaded(_wan_config())
        s.generate_video("p", output=str(tmp_path / "o.mp4"), seconds=2, width=704,
                         height=1280, steps=30, cfg_scale=4.0)
        call = _FakeWan.calls[-1]
        assert call["num_frames"] == 49
        assert (call["width"], call["height"], call["num_inference_steps"],
                call["guidance_scale"]) == (704, 1280, 30, 4.0)

    def test_an_image_switches_to_the_image_pipeline_sharing_weights(self, fake_diffusers, tmp_path):
        from PIL import Image
        first = tmp_path / "first.png"
        Image.new("RGB", (16, 16)).save(first)
        s = _loaded(_wan_config())
        s.generate_video("she turns", output=str(tmp_path / "o.mp4"), image=str(first))
        assert _FakeWanI2V.calls[-1]["pipe"] == "i2v"
        assert isinstance(_FakeWanI2V.calls[-1]["image"], Image.Image)
        assert s._i2v.source is s.pipeline

    def test_text_only_model_refuses_an_image(self, fake_diffusers, tmp_path):
        s = _loaded(_wan_config(i2v_pipeline_class=None))
        with pytest.raises(ValueError, match="text only"):
            s.generate_video("p", output=str(tmp_path / "o.mp4"), image="x.png")

    @pytest.mark.parametrize("key", ["audio", "control", "lora"])
    def test_mlx_only_inputs_are_refused_not_dropped(self, fake_diffusers, tmp_path, key):
        s = _loaded(_wan_config())
        with pytest.raises(ValueError, match="LTX-2 MLX feature"):
            s.generate_video("p", output=str(tmp_path / "o.mp4"), **{key: "x"})

    def test_mlx_knobs_are_ignored(self, fake_diffusers, tmp_path):
        s = _loaded(_wan_config())
        s.generate_video("p", output=str(tmp_path / "o.mp4"), mode="two-stage", low_ram=True)
        assert _FakeWan.calls[-1]["num_frames"] == 121

    def test_distilled_ltx2_gets_its_sigmas_and_writes_the_soundtrack(self, fake_diffusers, tmp_path):
        cfg = ModelConfig(name="ltx-2.3-distilled", path="/models/ltx", model_type="diffusers-video",
                          parameters={"pipeline_class": "LTX2Pipeline", "frame_multiple": 8,
                                      "frame_rate": 24, "num_frames": 121, "width": 768,
                                      "height": 512, "num_inference_steps": 8,
                                      "guidance_scale": 1.0, "sigmas": "ltx2-distilled"})
        s = _loaded(cfg)
        s.generate_video("a river", output=str(tmp_path / "r.mp4"), enhance_prompt=True)
        call = _FakeLTX2.calls[-1]
        assert call["sigmas"] == [1.0, 0.99, 0.5, 0.0]
        assert call["frame_rate"] == 24.0 and call["guidance_scale"] == 1.0
        assert call["enable_prompt_enhancement"] is True
        assert fake_diffusers[-1]["audio"] is True and fake_diffusers[-1]["rate"] == 24000

    def test_ltx2_frame_grid_is_eight(self, fake_diffusers, tmp_path):
        cfg = ModelConfig(name="ltx", path="/m", model_type="diffusers-video",
                          parameters={"pipeline_class": "LTX2Pipeline", "frame_multiple": 8})
        s = _loaded(cfg)
        with pytest.raises(ValueError, match="8k\\+1"):
            s.generate_video("p", output=str(tmp_path / "o.mp4"), frames=85)

    def test_unload_forgets_both_pipelines(self, fake_diffusers, tmp_path):
        s = _loaded(_wan_config())
        s._image_pipeline()
        s.unload()
        assert s.pipeline is None and s._i2v is None and not s.is_loaded
