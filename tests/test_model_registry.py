"""Tests for model registry"""

import pytest
from unittest.mock import patch, MagicMock

from ollamadiffuser.core.config.model_registry import ModelRegistry


class TestModelRegistry:
    def test_default_models_loaded(self):
        with patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_req.get.side_effect = Exception("no network")
            reg = ModelRegistry()
            models = reg.get_all_models()
            assert "flux.1-dev" in models
            assert "flux.1-schnell" in models
            assert "stable-diffusion-1.5" in models
            assert "stable-diffusion-xl-base" in models

    def test_get_model(self):
        with patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_req.get.side_effect = Exception("no network")
            reg = ModelRegistry()
            model = reg.get_model("flux.1-schnell")
            assert model is not None
            assert model["model_type"] == "flux"
            assert model["repo_id"] == "black-forest-labs/FLUX.1-schnell"

    def test_get_nonexistent_model(self):
        with patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_req.get.side_effect = Exception("no network")
            reg = ModelRegistry()
            assert reg.get_model("nonexistent-model") is None

    def test_add_and_remove_model(self):
        with patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_req.get.side_effect = Exception("no network")
            reg = ModelRegistry()
            assert reg.add_model("custom", {"repo_id": "user/model", "model_type": "sd15"})
            assert reg.get_model("custom") is not None
            assert reg.remove_model("custom")
            assert reg.get_model("custom") is None

    def test_get_model_names(self):
        with patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_req.get.side_effect = Exception("no network")
            reg = ModelRegistry()
            names = reg.get_model_names()
            assert isinstance(names, list)
            assert len(names) > 0
            assert "flux.1-dev" in names


class TestNewModelsRegistered:
    """Verify all 21 new models are registered with correct configuration."""

    # All 21 new model names and their expected model_type
    NEW_MODELS = {
        # Tier 1: Existing strategy models
        "stable-diffusion-3.5-large": "sd3",
        "stable-diffusion-3.5-large-turbo": "sd3",
        "realvisxl-v4": "sdxl",
        "dreamshaper": "sd15",
        "realistic-vision-v6": "sd15",
        "sdxl-turbo": "sdxl",
        # Tier 2: Scheduler override
        "sdxl-lightning-4step": "sdxl",
        # Tier 3: FLUX pipeline variants
        "flux.1-fill-dev": "flux",
        "flux.1-canny-dev": "flux",
        "flux.1-depth-dev": "flux",
        # Tier 4: Generic pipeline models
        "flux.2-dev": "generic",
        "flux.2-klein-4b": "generic",
        "z-image-turbo": "generic",
        "sana-1.5": "generic",
        "cogview4": "generic",
        "kolors": "generic",
        "hunyuan-dit": "generic",
        "lumina-2": "generic",
        "pixart-sigma": "generic",
        "auraflow": "generic",
        "omnigen": "generic",
    }

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        with patch("ollamadiffuser.core.config.model_registry.requests") as mock_req:
            mock_req.get.side_effect = Exception("no network")
            self.reg = ModelRegistry()

    def test_all_new_models_exist(self):
        models = self.reg.get_all_models()
        for name in self.NEW_MODELS:
            assert name in models, f"Model '{name}' not found in registry"

    def test_all_new_models_have_correct_type(self):
        for name, expected_type in self.NEW_MODELS.items():
            model = self.reg.get_model(name)
            assert model is not None, f"Model '{name}' not found"
            assert model["model_type"] == expected_type, (
                f"Model '{name}' has type '{model['model_type']}', expected '{expected_type}'"
            )

    def test_all_new_models_have_repo_id(self):
        for name in self.NEW_MODELS:
            model = self.reg.get_model(name)
            assert "repo_id" in model, f"Model '{name}' missing repo_id"
            assert model["repo_id"], f"Model '{name}' has empty repo_id"

    def test_generic_models_have_pipeline_class(self):
        generic_models = [n for n, t in self.NEW_MODELS.items() if t == "generic"]
        for name in generic_models:
            model = self.reg.get_model(name)
            params = model.get("parameters", {})
            assert "pipeline_class" in params, (
                f"Generic model '{name}' missing pipeline_class in parameters"
            )
            assert params["pipeline_class"], (
                f"Generic model '{name}' has empty pipeline_class"
            )

    def test_flux_variants_have_pipeline_class(self):
        flux_variants = ["flux.1-fill-dev", "flux.1-canny-dev", "flux.1-depth-dev"]
        expected_classes = {
            "flux.1-fill-dev": "FluxFillPipeline",
            "flux.1-canny-dev": "FluxControlPipeline",
            "flux.1-depth-dev": "FluxControlPipeline",
        }
        for name in flux_variants:
            model = self.reg.get_model(name)
            params = model.get("parameters", {})
            assert params.get("pipeline_class") == expected_classes[name], (
                f"FLUX variant '{name}' has pipeline_class '{params.get('pipeline_class')}', "
                f"expected '{expected_classes[name]}'"
            )

    def test_lightning_has_scheduler_config(self):
        model = self.reg.get_model("sdxl-lightning-4step")
        params = model.get("parameters", {})
        assert params.get("scheduler_class") == "EulerDiscreteScheduler"
        assert "scheduler_kwargs" in params
        assert params["scheduler_kwargs"].get("timestep_spacing") == "trailing"

    def test_turbo_models_have_low_steps(self):
        turbo_models = {
            "sdxl-turbo": 1,
            "stable-diffusion-3.5-large-turbo": 4,
            "sdxl-lightning-4step": 4,
        }
        for name, expected_steps in turbo_models.items():
            model = self.reg.get_model(name)
            params = model.get("parameters", {})
            assert params.get("num_inference_steps") == expected_steps, (
                f"Turbo model '{name}' has {params.get('num_inference_steps')} steps, expected {expected_steps}"
            )

    def test_omnigen_no_negative_prompt(self):
        model = self.reg.get_model("omnigen")
        params = model.get("parameters", {})
        assert params.get("supports_negative_prompt") is False

    def test_all_models_have_hardware_requirements(self):
        for name in self.NEW_MODELS:
            model = self.reg.get_model(name)
            assert "hardware_requirements" in model, (
                f"Model '{name}' missing hardware_requirements"
            )

    def test_mps_models_are_reasonably_sized(self):
        """Models listing MPS should have min_vram_gb <= 16."""
        for name in self.NEW_MODELS:
            model = self.reg.get_model(name)
            hr = model.get("hardware_requirements", {})
            devices = hr.get("supported_devices", [])
            if "MPS" in devices:
                min_vram = hr.get("min_vram_gb", 0)
                assert min_vram <= 16, (
                    f"Model '{name}' lists MPS but needs {min_vram}GB min VRAM"
                )
