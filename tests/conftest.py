"""Shared test fixtures"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import json

from ollamadiffuser.core.config.settings import ModelConfig, Settings, ServerConfig


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory"""
    config_dir = tmp_path / ".ollamadiffuser"
    config_dir.mkdir()
    (config_dir / "models").mkdir()
    (config_dir / "cache").mkdir()
    return config_dir


@pytest.fixture
def sample_model_config(tmp_path):
    """Create a sample model config"""
    model_path = tmp_path / "test-model"
    model_path.mkdir()
    return ModelConfig(
        name="test-model",
        path=str(model_path),
        model_type="sd15",
        variant="fp16",
        parameters={"num_inference_steps": 20, "guidance_scale": 7.5},
    )


@pytest.fixture
def flux_model_config(tmp_path):
    """Create a FLUX model config"""
    model_path = tmp_path / "flux-model"
    model_path.mkdir()
    return ModelConfig(
        name="flux.1-schnell",
        path=str(model_path),
        model_type="flux",
        variant="fp16",
        parameters={"num_inference_steps": 4, "guidance_scale": 0.0, "max_sequence_length": 256},
    )


@pytest.fixture
def mock_pipeline():
    """Create a mock diffusion pipeline"""
    pipeline = MagicMock()
    pipeline.to.return_value = pipeline
    pipeline.enable_attention_slicing = MagicMock()
    pipeline.load_lora_weights = MagicMock()
    pipeline.unload_lora_weights = MagicMock()
    pipeline.set_adapters = MagicMock()

    # Mock image output
    from PIL import Image
    mock_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
    mock_output = MagicMock()
    mock_output.images = [mock_image]
    pipeline.return_value = mock_output

    return pipeline


class Box:
    """A machine for the model manager: installed models, the memory each
    holds once loaded and needs to load, and a total that loads take from.

    The manager is the real one; its engines are stand-ins that take their
    model's memory while loaded, and memory_check reads this machine.
    ``counted=False`` is a model whose engine cannot say what it holds.
    """

    MODELS = {
        "a": {"holds": 10, "needs": 13},
        "b": {"holds": 10, "needs": 13},
        "c": {"holds": 20, "needs": 25},
        "big": {"holds": 60, "needs": 70},
    }

    def __init__(self, monkeypatch, tmp_path, total_gb=96.0, models=None):
        from ollamadiffuser.core.config.settings import ModelConfig, settings
        from ollamadiffuser.core.models.manager import ModelManager
        from ollamadiffuser.core.utils import memory_check

        self.models = {name: dict(spec) for name, spec in (models or self.MODELS).items()}
        self.total, self.other_use = total_gb, 0.0   # other_use: other programs on the machine
        self.live, self.events = {}, []              # model -> its engine while loaded; loads and unloads
        monkeypatch.setattr(settings, "config_dir", tmp_path)
        monkeypatch.setattr(settings, "config_file", tmp_path / "config.json")
        monkeypatch.setattr(settings, "models", {
            name: ModelConfig(name=name, path=str(tmp_path), model_type="mlx", parameters={})
            for name in self.models})
        monkeypatch.setattr(settings, "current_model", None)
        monkeypatch.delenv(memory_check.SKIP_ENV, raising=False)
        monkeypatch.setattr(memory_check, "unified_memory", lambda: True)
        monkeypatch.setattr(memory_check, "free_gb", self.free_gb)
        monkeypatch.setattr(memory_check, "need_for",
                            lambda name, info: (self.models[name]["needs"], "the test's number"))
        monkeypatch.setattr("ollamadiffuser.core.inference.engine.InferenceEngine", self._engine)
        self.manager = ModelManager()

    def free_gb(self) -> float:
        return self.total - self.other_use - sum(self.models[name]["holds"] for name in self.live)

    def loads(self):
        return [name for kind, name in self.events if kind == "load"]

    def unloads(self):
        return [name for kind, name in self.events if kind == "unload"]

    def _engine(self):
        from PIL import Image

        box, engine = self, MagicMock()
        engine.held_bytes = None

        def load(config):
            spec = box.models[config.name]
            if spec.get("fails"):
                return False
            box.events.append(("load", config.name))
            box.live[config.name] = engine
            engine.model = config.name
            engine.held_bytes = int(spec["holds"] * 1e9) if spec.get("counted", True) else None
            return True

        def unload():
            box.events.append(("unload", engine.model))
            box.live.pop(engine.model, None)
            engine.held_bytes = None

        engine.load_model.side_effect = load
        engine.unload.side_effect = unload
        engine.generate_image.return_value = Image.new("RGB", (8, 8))
        engine.get_model_info.return_value = {"type": "mlx", "device": "mps"}
        engine.is_controlnet_pipeline = False
        return engine


@pytest.fixture
def box(monkeypatch, tmp_path):
    return Box(monkeypatch, tmp_path)
