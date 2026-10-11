"""A model that does not fit is not loaded (utils/memory_check.py)."""
from unittest.mock import MagicMock, patch

import pytest

from ollamadiffuser.core.utils import memory_check

QWEN = {"hardware_requirements": {"min_vram_gb": 46}}


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.delenv(memory_check.SKIP_ENV, raising=False)
    monkeypatch.setattr(memory_check, "unified_memory", lambda: True)


def test_refused_when_free_memory_is_below_the_minimum(mac, monkeypatch):
    monkeypatch.setattr(memory_check, "free_gb", lambda: 12.04)
    reason = memory_check.refusal("qwen-image-2.1-mlx", QWEN)
    assert "needs about 46 GB" in reason and "12.0 GB is free" in reason
    assert "--force" in reason and memory_check.SKIP_ENV in reason


def test_loaded_when_it_fits(mac, monkeypatch):
    monkeypatch.setattr(memory_check, "free_gb", lambda: 50.0)
    assert memory_check.refusal("qwen-image-2.1-mlx", QWEN) is None


def test_no_stated_minimum_means_no_check(mac, monkeypatch):
    monkeypatch.setattr(memory_check, "free_gb", lambda: 1.0)
    assert memory_check.refusal("my-own-model", {}) is None
    assert memory_check.refusal("my-own-model", None) is None


def test_skipped_on_request_and_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(memory_check, "free_gb", lambda: 1.0)
    monkeypatch.setattr(memory_check, "unified_memory", lambda: True)
    monkeypatch.setenv(memory_check.SKIP_ENV, "1")
    assert memory_check.refusal("qwen-image-2.1-mlx", QWEN) is None
    monkeypatch.delenv(memory_check.SKIP_ENV)
    monkeypatch.setattr(memory_check, "unified_memory", lambda: False)
    assert memory_check.refusal("qwen-image-2.1-mlx", QWEN) is None


class TestLoadModel:
    @pytest.fixture
    def manager(self, monkeypatch, tmp_path):
        from ollamadiffuser.core.config.settings import ModelConfig, settings
        from ollamadiffuser.core.models.manager import ModelManager
        monkeypatch.setattr(settings, "config_dir", tmp_path)
        monkeypatch.setattr(settings, "config_file", tmp_path / "config.json")
        monkeypatch.setattr(settings, "models", {"qwen-image-2.1-mlx": ModelConfig(
            name="qwen-image-2.1-mlx", path=str(tmp_path), model_type="mlx", parameters={})})
        monkeypatch.setattr(settings, "current_model", "something-else")
        return ModelManager()

    def test_a_refusal_loads_nothing_and_says_why(self, manager, monkeypatch):
        monkeypatch.setattr(memory_check, "refusal", lambda name, info: "needs about 46 GB")
        with patch("ollamadiffuser.core.inference.engine.InferenceEngine") as engine_cls:
            assert manager.load_model("qwen-image-2.1-mlx") is False
        engine_cls.assert_not_called()
        assert manager.last_error == "needs about 46 GB"

    def test_force_loads_anyway_and_persist_false_leaves_the_config_alone(self, manager, monkeypatch):
        from ollamadiffuser.core.config.settings import settings
        monkeypatch.setattr(memory_check, "refusal", lambda name, info: "needs about 46 GB")
        engine = MagicMock()
        engine.load_model.return_value = True
        with patch("ollamadiffuser.core.inference.engine.InferenceEngine", return_value=engine):
            assert manager.load_model("qwen-image-2.1-mlx", force=True, persist=False) is True
        assert manager.current_model_name == "qwen-image-2.1-mlx"
        assert settings.current_model == "something-else"
        manager.unload_model(persist=False)
        assert settings.current_model == "something-else" and manager.loaded_model is None
