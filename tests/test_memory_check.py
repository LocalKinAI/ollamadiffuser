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
        monkeypatch.setattr(memory_check, "refusal", lambda name, info, freeable_gb=0.0: "needs about 46 GB")
        with patch("ollamadiffuser.core.inference.engine.InferenceEngine") as engine_cls:
            assert manager.load_model("qwen-image-2.1-mlx") is False
        engine_cls.assert_not_called()
        assert manager.last_error == "needs about 46 GB"

    def test_force_loads_anyway_and_persist_false_leaves_the_config_alone(self, manager, monkeypatch):
        from ollamadiffuser.core.config.settings import settings
        monkeypatch.setattr(memory_check, "refusal", lambda name, info, freeable_gb=0.0: "needs about 46 GB")
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
            def __init__(self, backend, held=None, others=0):
                self.backend, self.peak_bytes, self.held_after = backend, None, held

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


def test_a_refusal_counts_what_unloading_the_others_would_free(mac, monkeypatch):
    monkeypatch.setattr(memory_check, "free_gb", lambda: 30.0)
    assert memory_check.refusal("qwen-image-2.1-mlx", QWEN, freeable_gb=20.0) is None
    reason = memory_check.refusal("qwen-image-2.1-mlx", QWEN, freeable_gb=10.0)
    assert "30.0 GB is free, 40.0 GB with the other models loaded here unloaded" in reason


class TestShortOfMemory:
    """What a server unloads other models by (memory_check.short_gb)."""

    def test_on_apple_silicon_it_is_free_memory(self, mac, monkeypatch):
        monkeypatch.setattr(memory_check, "free_gb", lambda: 40.0)
        assert memory_check.short_gb("qwen-image-2.1-mlx", QWEN) == 6.0
        assert memory_check.short_gb("qwen-image-2.1-mlx", QWEN, held_gb=10.0) == 0.0
        monkeypatch.setenv(memory_check.SKIP_ENV, "1")      # a refusal is skipped, room is still made
        assert memory_check.short_gb("qwen-image-2.1-mlx", QWEN) == 6.0

    def test_on_nvidia_it_is_free_gpu_memory_with_what_pytorch_keeps_cached(self, monkeypatch):
        import sys
        import types
        cuda = types.SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (8e9, 24e9),
                                     memory_reserved=lambda: 6e9, memory_allocated=lambda: 4e9)
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
        monkeypatch.setattr(memory_check, "unified_memory", lambda: False)
        monkeypatch.setattr(memory_check, "need_for", lambda name, info: (16.0, "test"))
        assert memory_check.room_gb() == 10.0 and memory_check.short_gb("flux", {}) == 6.0

    def test_no_telling_is_never_short(self, mac, monkeypatch):
        monkeypatch.setattr(memory_check, "free_gb", lambda: None)
        assert memory_check.short_gb("qwen-image-2.1-mlx", QWEN) == 0.0
        monkeypatch.setattr(memory_check, "free_gb", lambda: 1.0)
        assert memory_check.short_gb("my-own-model", {}) == 0.0


class TestOwnShare:
    """A process holding several models counts all of them in its peak; each
    records its own share (memory_record.Meter)."""

    @pytest.fixture
    def backend(self, monkeypatch, tmp_path):
        from ollamadiffuser.core.config.settings import settings
        from ollamadiffuser.core.utils import memory_record
        monkeypatch.setattr(settings, "config_dir", tmp_path)   # the records a job writes
        state = {"in_use": 0, "peak": 0}
        monkeypatch.setattr(memory_record, "in_use", lambda b: state["in_use"] if b else None)
        monkeypatch.setattr(memory_record, "_reset_peak", lambda b: bool(b))
        monkeypatch.setattr(memory_record, "_peak", lambda b: state["peak"] if b else None)
        return state

    def test_the_other_models_weights_are_not_this_ones(self, backend):
        from ollamadiffuser.core.utils import memory_record
        backend["in_use"] = 30e9                            # 20 of them another model's
        with memory_record.Meter("mlx", held=10e9, others=20e9) as meter:
            backend["peak"], backend["in_use"] = 38e9, 30e9
        assert meter.peak_bytes == 18e9 and meter.held_after == 10e9

    def test_weights_the_first_job_brings_in_are_held_after_it(self, backend):
        from ollamadiffuser.core.utils import memory_record
        backend["in_use"] = 20e9                            # the other model; this one loaded lazily
        with memory_record.Meter("mlx", held=0, others=20e9) as meter:
            backend["peak"], backend["in_use"] = 37e9, 32e9
        assert meter.peak_bytes == 17e9 and meter.held_after == 12e9

    def test_the_larger_share_is_kept_when_the_counts_drift(self, backend):
        from ollamadiffuser.core.utils import memory_record
        backend["in_use"] = 25e9                            # 5 more than the counts say
        with memory_record.Meter("mlx", held=10e9, others=10e9) as meter:
            backend["peak"] = 33e9
        assert meter.peak_bytes == 23e9                     # 33 - 10, not 10 + (33 - 25)

    def test_alone_it_is_the_peak_as_before(self, backend):
        from ollamadiffuser.core.utils import memory_record
        backend["in_use"] = 12e9
        with memory_record.Meter("mlx", held=12e9) as meter:
            backend["peak"] = 15e9
        assert meter.peak_bytes == 15e9
        with memory_record.Meter(None, held=None) as meter:
            pass
        assert meter.peak_bytes is None and meter.held_after is None

    def test_an_engine_counts_what_its_load_and_jobs_leave_and_what_the_others_hold(self, backend):
        from PIL import Image
        from ollamadiffuser.core.config.settings import ModelConfig
        from ollamadiffuser.core.inference import engine as engine_module
        from ollamadiffuser.core.inference.base import InferenceStrategy

        class Lazy(InferenceStrategy):
            def load(self, model_config, device):
                backend["in_use"] += 1e9        # what building the model allocates
                self.device = device
                return True

            def generate(self, prompt, **kw):
                backend["in_use"] += 9e9        # the weights, on first use
                backend["peak"] = backend["in_use"] + 3e9
                return Image.new("RGB", (8, 8))

        with patch.object(engine_module, "_detect_device", return_value="mps"), \
                patch.object(engine_module, "_get_strategy", side_effect=lambda kind: Lazy()):
            from ollamadiffuser.core.utils import memory_record
            first, second = engine_module.InferenceEngine(), engine_module.InferenceEngine()
            assert first.load_model(ModelConfig(name="m1", path="org/m1", model_type="mlx"))
            first.generate_image(prompt="p")
            assert first.held_bytes == 10e9
            assert second.load_model(ModelConfig(name="m2", path="org/m2", model_type="mlx"))
            assert second.held_bytes == 1e9
            seen, real = [], memory_record.Meter
            with patch.object(memory_record, "Meter", side_effect=lambda *a, **kw: seen.append(kw) or real(*a, **kw)):
                second.generate_image(prompt="p")
            assert seen[0]["others"] == 10e9 and second.held_bytes == 10e9
            assert memory_record.measured("m2")["peak_gb"] == 13.0   # 23 at the peak, 10 of it m1's
            first.unload()
            assert first.held_bytes is None and first not in engine_module._LOADED
