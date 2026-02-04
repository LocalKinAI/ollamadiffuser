"""Tests for Apple Silicon (MPS) support improvements."""

import sys
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from ollamadiffuser.core.config.settings import ModelConfig
from ollamadiffuser.core.config.model_registry import ModelRegistry


class TestGenericStrategyMPSDtype:
    """Verify GenericPipelineStrategy uses float16, not bfloat16, on MPS."""

    def _load_generic_on_device(self, device, torch_dtype=None):
        """Helper: load GenericPipelineStrategy and return the dtype passed to from_pretrained."""
        import torch
        from ollamadiffuser.core.inference.strategies.generic_strategy import (
            GenericPipelineStrategy,
        )

        mock_pipeline_cls = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_pipe.enable_attention_slicing = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        mock_diffusers = MagicMock()
        mock_diffusers.TestPipeline = mock_pipeline_cls

        params = {"pipeline_class": "TestPipeline"}
        if torch_dtype:
            params["torch_dtype"] = torch_dtype

        config = ModelConfig(
            name="test-model",
            path="org/test-model",
            model_type="generic",
            parameters=params,
        )

        strategy = GenericPipelineStrategy()
        original = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_diffusers
        try:
            strategy.load(config, device)
        finally:
            if original is not None:
                sys.modules["diffusers"] = original
            else:
                sys.modules.pop("diffusers", None)

        call_kwargs = mock_pipeline_cls.from_pretrained.call_args
        return call_kwargs[1].get("torch_dtype") if call_kwargs else None

    def test_mps_no_dtype_param_uses_float16(self):
        import torch

        dtype = self._load_generic_on_device("mps", torch_dtype=None)
        assert dtype == torch.float16

    def test_mps_bfloat16_param_falls_back_to_float16(self):
        import torch

        dtype = self._load_generic_on_device("mps", torch_dtype="bfloat16")
        assert dtype == torch.float16

    def test_mps_float16_param_stays_float16(self):
        import torch

        dtype = self._load_generic_on_device("mps", torch_dtype="float16")
        assert dtype == torch.float16

    def test_cuda_bfloat16_param_stays_bfloat16(self):
        import torch

        dtype = self._load_generic_on_device("cuda", torch_dtype="bfloat16")
        assert dtype == torch.bfloat16

    def test_cuda_no_dtype_param_uses_bfloat16(self):
        import torch

        dtype = self._load_generic_on_device("cuda", torch_dtype=None)
        assert dtype == torch.bfloat16

    def test_cpu_always_float32(self):
        import torch

        dtype = self._load_generic_on_device("cpu", torch_dtype=None)
        assert dtype == torch.float32


class TestRegistryMPSDevices:
    """Verify model registry MPS-compatible entries."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        with patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_req.get.side_effect = Exception("no network")
            self.reg = ModelRegistry()

    EXPECTED_MPS_MODELS = [
        "flux.1-schnell",
        "stable-diffusion-3.5-medium",
        "stable-diffusion-3.5-large",
        "stable-diffusion-3.5-large-turbo",
        "stable-diffusion-xl-base",
        "stable-diffusion-1.5",
        "realvisxl-v4",
        "dreamshaper",
        "realistic-vision-v6",
        "sdxl-turbo",
        "sdxl-lightning-4step",
        "sana-1.5",
        "pixart-sigma",
        "flux.2-klein-4b",
        "kolors",
        "hunyuan-dit",
        "lumina-2",
        "cogview4",
        # GGUF models support MPS via Metal acceleration in stable-diffusion-cpp
        "flux.1-dev-gguf-q2k",
        "flux.1-dev-gguf-q3ks",
        "flux.1-dev-gguf-q4ks",
        "flux.1-dev-gguf-q4-0",
        "flux.1-dev-gguf-q4-1",
        "flux.1-dev-gguf-q5ks",
        "flux.1-dev-gguf-q5-0",
        "flux.1-dev-gguf-q5-1",
        "flux.1-dev-gguf-q6k",
        "flux.1-dev-gguf-q8",
        "flux.1-dev-gguf-f16",
    ]

    EXPECTED_NO_MPS_MODELS = [
        "flux.1-dev",
        "flux.2-dev",
        "flux.1-fill-dev",
        "flux.1-canny-dev",
        "flux.1-depth-dev",
        "auraflow",
        "omnigen",
        "z-image-turbo",
    ]

    def test_mps_models_have_mps_device(self):
        for name in self.EXPECTED_MPS_MODELS:
            model = self.reg.get_model(name)
            assert model is not None, f"Model '{name}' not in registry"
            devices = model.get("hardware_requirements", {}).get(
                "supported_devices", []
            )
            assert "MPS" in devices, (
                f"Model '{name}' should list MPS but has {devices}"
            )

    def test_non_mps_models_lack_mps_device(self):
        for name in self.EXPECTED_NO_MPS_MODELS:
            model = self.reg.get_model(name)
            if model is None:
                continue
            devices = model.get("hardware_requirements", {}).get(
                "supported_devices", []
            )
            assert "MPS" not in devices, (
                f"Model '{name}' should NOT list MPS but has {devices}"
            )


class TestRecommendCommand:
    """Test the recommend CLI command."""

    def _mock_hw(self, device="mps", total_ram=16):
        return {
            "device": device,
            "device_name": "Apple Silicon (MPS)" if device == "mps" else device.upper(),
            "total_ram_gb": total_ram,
            "available_ram_gb": total_ram - 4,
            "vram_gb": total_ram if device == "mps" else 0,
        }

    def test_recommend_runs(self):
        from ollamadiffuser.cli.recommend_command import recommend

        runner = CliRunner()
        with patch("ollamadiffuser.cli.recommend_command._detect_hardware") as mock_hw, \
             patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_hw.return_value = self._mock_hw()
            mock_req.get.side_effect = Exception("no network")
            result = runner.invoke(recommend, [])
            assert result.exit_code == 0
            assert "Hardware Detection" in result.output

    def test_recommend_shows_models(self):
        from ollamadiffuser.cli.recommend_command import recommend

        runner = CliRunner()
        with patch("ollamadiffuser.cli.recommend_command._detect_hardware") as mock_hw, \
             patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_hw.return_value = self._mock_hw()
            mock_req.get.side_effect = Exception("no network")
            result = runner.invoke(recommend, [])
            # Should find at least some models for 16GB MPS
            assert "recommended" in result.output.lower() or "possible" in result.output.lower()

    def test_recommend_device_override(self):
        from ollamadiffuser.cli.recommend_command import recommend

        runner = CliRunner()
        with patch("ollamadiffuser.cli.recommend_command._detect_hardware") as mock_hw, \
             patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_hw.return_value = self._mock_hw(device="cpu", total_ram=32)
            mock_req.get.side_effect = Exception("no network")
            result = runner.invoke(recommend, ["--device", "mps"])
            assert result.exit_code == 0

    def test_classify_model_recommended(self):
        from ollamadiffuser.cli.recommend_command import _classify_model

        hw = {"device": "mps", "total_ram_gb": 16, "vram_gb": 16}
        model_info = {
            "hardware_requirements": {
                "min_vram_gb": 4,
                "recommended_vram_gb": 6,
                "supported_devices": ["CUDA", "MPS"],
            }
        }
        tier, _ = _classify_model("test", model_info, hw)
        assert tier == "recommended"

    def test_classify_model_possible(self):
        from ollamadiffuser.cli.recommend_command import _classify_model

        hw = {"device": "mps", "total_ram_gb": 16, "vram_gb": 16}
        model_info = {
            "hardware_requirements": {
                "min_vram_gb": 8,
                "recommended_vram_gb": 16,
                "supported_devices": ["CUDA", "MPS"],
            }
        }
        tier, _ = _classify_model("test", model_info, hw)
        assert tier == "possible"

    def test_classify_model_too_large(self):
        from ollamadiffuser.cli.recommend_command import _classify_model

        hw = {"device": "mps", "total_ram_gb": 16, "vram_gb": 16}
        model_info = {
            "hardware_requirements": {
                "min_vram_gb": 20,
                "recommended_vram_gb": 24,
                "supported_devices": ["CUDA", "MPS"],
            }
        }
        tier, _ = _classify_model("test", model_info, hw)
        assert tier == "too_large"

    def test_classify_model_incompatible(self):
        from ollamadiffuser.cli.recommend_command import _classify_model

        hw = {"device": "mps", "total_ram_gb": 16, "vram_gb": 16}
        model_info = {
            "hardware_requirements": {
                "min_vram_gb": 4,
                "supported_devices": ["CUDA"],
            }
        }
        tier, _ = _classify_model("test", model_info, hw)
        assert tier == "incompatible"
