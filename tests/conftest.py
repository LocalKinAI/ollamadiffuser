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
