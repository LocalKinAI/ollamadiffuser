"""A model that does not fit is not loaded (utils/memory_check.py)."""
from unittest.mock import MagicMock, patch

import pytest

from ollamadiffuser.core.utils import memory_check

QWEN = {"hardware_requirements": {"min_vram_gb": 46}}


@pytest.fixture
def mac(monkeypatch, tmp_path):
    from ollamadiffuser.core.config.settings import settings
    monkeypatch.setattr(settings, "config_dir", tmp_path)  # no measured peaks but the test's own
    monkeypatch.delenv(memory_check.SKIP_ENV, raising=False)
    monkeypatch.setattr(memory_check, "unified_memory", lambda: True)


def test_refused_when_free_memory_is_below_the_minimum(mac, monkeypatch):
    monkeypatch.setattr(memory_check, "free_gb", lambda: 12.04)
    reason = memory_check.refusal("qwen-image-2.1-mlx", QWEN)
    assert "needs about 46 GB" in reason and "12.0 GB is free" in reason
    assert "(the registry's estimate)" in reason
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


class TestMeasuredPeaks:
    """What a model really took, recorded by its generations (memory_record.py)."""

    @pytest.fixture
    def records(self, tmp_path, monkeypatch):
        from ollamadiffuser.core.config.settings import settings
        from ollamadiffuser.core.utils import memory_record
        monkeypatch.setattr(settings, "config_dir", tmp_path)
        return memory_record

    def test_the_highest_peak_is_kept_with_its_size(self, records):
        records.record("qwen-image-2.1-turbo-mlx", 30.0e9, 768, 768)
        records.record("qwen-image-2.1-turbo-mlx", 38.7e9, 1024, 1024)
        records.record("qwen-image-2.1-turbo-mlx", 20.0e9, 512, 512)
        seen = records.measured("qwen-image-2.1-turbo-mlx")
        assert seen["peak_gb"] == 38.7 and (seen["width"], seen["height"]) == (1024, 1024)
        assert seen["runs"] == 3

    def test_a_reading_of_nothing_is_not_a_peak(self, records):
        records.record("ltx-2.3-mlx-q8", 40_000_000)   # a model that ran in another process
        records.record("m", None)
        assert records.measured("ltx-2.3-mlx-q8") is None and records.measured("m") is None

    def test_any_model_name_makes_a_file_name(self, records, tmp_path):
        records.record("org/model v2", 2.5e9)
        assert (tmp_path / "memory" / "org_model_v2.json").exists()
        assert records.measured("org/model v2")["peak_gb"] == 2.5

    def test_the_check_goes_by_the_measured_peak(self, mac, records, monkeypatch):
        records.record("qwen-image-2.1-turbo-mlx", 38.7e9, 1024, 1024)
        monkeypatch.setattr(memory_check, "free_gb", lambda: 40.0)
        reason = memory_check.refusal("qwen-image-2.1-turbo-mlx", QWEN)
        assert "needs about 41.7 GB of memory (its measured peak, 38.7 GB, plus 3)" in reason
        monkeypatch.setattr(memory_check, "free_gb", lambda: 42.0)
        assert memory_check.refusal("qwen-image-2.1-turbo-mlx", QWEN) is None  # the estimate says 46

    def test_a_generation_records_its_peak_and_a_failure_does_not(self, records, monkeypatch):
        from PIL import Image
        from ollamadiffuser.core.config.settings import ModelConfig
        from ollamadiffuser.core.inference.base import InferenceStrategy
        from ollamadiffuser.core.inference.engine import InferenceEngine

        class Meter:
            def __init__(self, backend):
                self.backend, self.peak_bytes = backend, None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.peak_bytes = 12.4e9 if self.backend == "mlx" else None
                return False

        class Draws(InferenceStrategy):
            fail = False

            def load(self, *a, **kw):
                return True

            def generate(self, prompt, **kw):
                return self._create_error_image("boom", prompt) if self.fail else Image.new("RGB", (8, 8))

        monkeypatch.setattr(records, "Meter", Meter)
        engine = InferenceEngine()
        engine._strategy = Draws()
        engine.model_config = ModelConfig(name="boogu-image-turbo-mlx", path="/m", model_type="mlx")
        engine.generate_image(prompt="a fox", width=1024, height=1024)
        assert records.measured("boogu-image-turbo-mlx")["peak_gb"] == 12.4

        engine.model_config = ModelConfig(name="other-mlx", path="/m", model_type="mlx")
        engine._strategy.fail = True
        with pytest.raises(RuntimeError):
            engine.generate_image(prompt="a fox")
        assert records.measured("other-mlx") is None

    def test_a_model_that_renders_in_another_process_is_not_measured(self):
        from ollamadiffuser.core.config.settings import ModelConfig
        from ollamadiffuser.core.inference.engine import InferenceEngine
        engine = InferenceEngine()
        engine.device = "mps"
        engine.model_config = ModelConfig(name="ltx-2.3-mlx-q8", path="/m", model_type="ltx-video-mlx")
        assert engine._memory_backend() is None
        engine.model_config = ModelConfig(name="flux.1-schnell", path="/m", model_type="flux")
        assert engine._memory_backend() == "mps"
