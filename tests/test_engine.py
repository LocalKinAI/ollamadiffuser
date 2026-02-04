"""Tests for inference engine facade"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from PIL import Image

from ollamadiffuser.core.inference.engine import InferenceEngine, _get_strategy, _detect_device
from ollamadiffuser.core.inference.base import InferenceStrategy, SAFETY_DISABLED_KWARGS
from ollamadiffuser.core.config.settings import ModelConfig


class TestGetStrategy:
    def test_sd15_strategy(self):
        s = _get_strategy("sd15")
        from ollamadiffuser.core.inference.strategies.sd15_strategy import SD15Strategy
        assert isinstance(s, SD15Strategy)

    def test_sdxl_strategy(self):
        s = _get_strategy("sdxl")
        from ollamadiffuser.core.inference.strategies.sdxl_strategy import SDXLStrategy
        assert isinstance(s, SDXLStrategy)

    def test_flux_strategy(self):
        s = _get_strategy("flux")
        from ollamadiffuser.core.inference.strategies.flux_strategy import FluxStrategy
        assert isinstance(s, FluxStrategy)

    def test_sd3_strategy(self):
        s = _get_strategy("sd3")
        from ollamadiffuser.core.inference.strategies.sd3_strategy import SD3Strategy
        assert isinstance(s, SD3Strategy)

    def test_controlnet_strategies(self):
        s = _get_strategy("controlnet_sd15")
        from ollamadiffuser.core.inference.strategies.controlnet_strategy import ControlNetStrategy
        assert isinstance(s, ControlNetStrategy)

    def test_video_strategy(self):
        s = _get_strategy("video")
        from ollamadiffuser.core.inference.strategies.video_strategy import VideoStrategy
        assert isinstance(s, VideoStrategy)

    def test_hidream_strategy(self):
        s = _get_strategy("hidream")
        from ollamadiffuser.core.inference.strategies.hidream_strategy import HiDreamStrategy
        assert isinstance(s, HiDreamStrategy)

    def test_gguf_strategy(self):
        s = _get_strategy("gguf")
        from ollamadiffuser.core.inference.strategies.gguf_strategy import GGUFStrategy
        assert isinstance(s, GGUFStrategy)

    def test_generic_strategy(self):
        s = _get_strategy("generic")
        from ollamadiffuser.core.inference.strategies.generic_strategy import GenericPipelineStrategy
        assert isinstance(s, GenericPipelineStrategy)

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported model type"):
            _get_strategy("unknown_type")


class TestDetectDevice:
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda(self, _):
        assert _detect_device() == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps(self, *_):
        assert _detect_device() == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu(self, *_):
        assert _detect_device() == "cpu"


class TestInferenceEngine:
    def test_initial_state(self):
        engine = InferenceEngine()
        assert engine.pipeline is None
        assert engine.current_lora is None
        assert engine.is_controlnet_pipeline is False
        assert engine.is_loaded() is False
        assert engine.get_model_info() is None

    def test_load_model_invalid_config(self):
        engine = InferenceEngine()
        assert engine.load_model(None) is False

    def test_load_model_missing_path(self):
        config = ModelConfig(name="test", path="", model_type="sd15")
        engine = InferenceEngine()
        assert engine.load_model(config) is False

    def test_load_model_nonexistent_path(self):
        config = ModelConfig(name="test", path="/nonexistent/path", model_type="sd15")
        engine = InferenceEngine()
        assert engine.load_model(config) is False

    def test_generate_without_model(self):
        engine = InferenceEngine()
        with pytest.raises(RuntimeError, match="No model loaded"):
            engine.generate_image("test prompt")

    def test_unload_empty(self):
        engine = InferenceEngine()
        engine.unload()  # Should not raise
        assert engine.is_loaded() is False


class TestSafetyDisabledKwargs:
    def test_kwargs_present(self):
        assert "safety_checker" in SAFETY_DISABLED_KWARGS
        assert SAFETY_DISABLED_KWARGS["safety_checker"] is None
        assert SAFETY_DISABLED_KWARGS["requires_safety_checker"] is False
        assert SAFETY_DISABLED_KWARGS["feature_extractor"] is None


class TestGenericStrategy:
    def test_generic_strategy_type(self):
        s = _get_strategy("generic")
        from ollamadiffuser.core.inference.strategies.generic_strategy import GenericPipelineStrategy
        assert isinstance(s, GenericPipelineStrategy)

    def test_generic_strategy_dynamic_pipeline(self):
        import sys
        from ollamadiffuser.core.inference.strategies.generic_strategy import GenericPipelineStrategy

        mock_pipeline_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.to.return_value = mock_pipe_instance
        mock_pipe_instance.enable_attention_slicing = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe_instance

        mock_diffusers = MagicMock()
        mock_diffusers.SanaPipeline = mock_pipeline_cls

        strategy = GenericPipelineStrategy()
        config = ModelConfig(
            name="sana-test",
            path="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            model_type="generic",
            parameters={"pipeline_class": "SanaPipeline", "torch_dtype": "float16"},
        )

        original = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_diffusers
        try:
            result = strategy.load(config, "cpu")
        finally:
            if original is not None:
                sys.modules["diffusers"] = original
            else:
                sys.modules.pop("diffusers", None)

        assert result is True
        mock_pipeline_cls.from_pretrained.assert_called_once()

    def test_generic_strategy_missing_pipeline_class(self):
        from ollamadiffuser.core.inference.strategies.generic_strategy import GenericPipelineStrategy

        strategy = GenericPipelineStrategy()
        config = ModelConfig(
            name="bad-model",
            path="some/path",
            model_type="generic",
            parameters={},  # No pipeline_class
        )
        result = strategy.load(config, "cpu")
        assert result is False

    def test_generic_strategy_invalid_pipeline_class(self):
        import sys
        from ollamadiffuser.core.inference.strategies.generic_strategy import GenericPipelineStrategy

        mock_diffusers = MagicMock(spec=[])  # Empty spec = no attributes

        strategy = GenericPipelineStrategy()
        config = ModelConfig(
            name="bad-model",
            path="some/path",
            model_type="generic",
            parameters={"pipeline_class": "NonExistentPipeline"},
        )

        original = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_diffusers
        try:
            result = strategy.load(config, "cpu")
        finally:
            if original is not None:
                sys.modules["diffusers"] = original
            else:
                sys.modules.pop("diffusers", None)

        assert result is False

    def _load_generic_with_offload(self, device):
        """Helper to load a GenericPipelineStrategy with CPU offload on the given device."""
        import sys
        from ollamadiffuser.core.inference.strategies.generic_strategy import GenericPipelineStrategy

        mock_pipeline_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe_instance

        mock_diffusers = MagicMock()
        mock_diffusers.TestPipeline = mock_pipeline_cls

        strategy = GenericPipelineStrategy()
        config = ModelConfig(
            name="offload-test",
            path="some/model",
            model_type="generic",
            parameters={
                "pipeline_class": "TestPipeline",
                "enable_cpu_offload": True,
            },
        )

        original = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_diffusers
        try:
            result = strategy.load(config, device)
        finally:
            if original is not None:
                sys.modules["diffusers"] = original
            else:
                sys.modules.pop("diffusers", None)

        return result, mock_pipe_instance

    def test_generic_strategy_cpu_offload_cuda(self):
        result, mock_pipe = self._load_generic_with_offload("cuda")
        assert result is True
        # CUDA prefers sequential offload for lowest VRAM usage
        mock_pipe.enable_sequential_cpu_offload.assert_called_once()
        mock_pipe.to.assert_not_called()

    def test_generic_strategy_cpu_offload_mps(self):
        result, mock_pipe = self._load_generic_with_offload("mps")
        assert result is True
        # MPS prefers model-level offload (more effective on unified memory)
        mock_pipe.enable_model_cpu_offload.assert_called_once()
        mock_pipe.to.assert_not_called()


class TestInferenceStrategyBase:
    def test_make_generator_with_seed(self):
        from ollamadiffuser.core.inference.base import InferenceStrategy
        import torch

        class DummyStrategy(InferenceStrategy):
            def load(self, *a, **kw): pass
            def generate(self, *a, **kw): pass

        s = DummyStrategy()
        gen, seed = s._make_generator(42, "cpu")
        assert seed == 42
        assert isinstance(gen, torch.Generator)

    def test_make_generator_random_seed(self):
        from ollamadiffuser.core.inference.base import InferenceStrategy

        class DummyStrategy(InferenceStrategy):
            def load(self, *a, **kw): pass
            def generate(self, *a, **kw): pass

        s = DummyStrategy()
        _, seed1 = s._make_generator(None, "cpu")
        _, seed2 = s._make_generator(None, "cpu")
        # Very unlikely to be equal
        assert isinstance(seed1, int)
        assert isinstance(seed2, int)

    def test_create_error_image(self):
        from ollamadiffuser.core.inference.base import InferenceStrategy

        class DummyStrategy(InferenceStrategy):
            def load(self, *a, **kw): pass
            def generate(self, *a, **kw): pass

        s = DummyStrategy()
        img = s._create_error_image("test error", "test prompt")
        assert isinstance(img, Image.Image)
        assert img.size == (512, 512)
