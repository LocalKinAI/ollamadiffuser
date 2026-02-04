"""Tests for settings module"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from ollamadiffuser.core.config.settings import Settings, ModelConfig, ServerConfig


class TestModelConfig:
    def test_create_model_config(self):
        config = ModelConfig(name="test", path="/tmp/test", model_type="sd15")
        assert config.name == "test"
        assert config.model_type == "sd15"
        assert config.variant is None
        assert config.parameters is None

    def test_create_with_all_fields(self):
        config = ModelConfig(
            name="test",
            path="/tmp/test",
            model_type="flux",
            variant="fp16",
            parameters={"steps": 20},
        )
        assert config.variant == "fp16"
        assert config.parameters == {"steps": 20}


class TestServerConfig:
    def test_defaults(self):
        config = ServerConfig()
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.enable_cors is True

    def test_custom_values(self):
        config = ServerConfig(host="0.0.0.0", port=9000)
        assert config.host == "0.0.0.0"
        assert config.port == 9000


class TestSettings:
    def test_settings_creates_directories(self, tmp_path):
        with patch.object(Settings, "__init__", lambda self: None):
            s = Settings()
            s.config_dir = tmp_path / "config"
            s.models_dir = s.config_dir / "models"
            s.cache_dir = s.config_dir / "cache"
            s.config_file = s.config_dir / "config.json"
            s.server = ServerConfig()
            s.models = {}
            s.current_model = None
            s.hf_token = None

            s.config_dir.mkdir(exist_ok=True)
            s.models_dir.mkdir(exist_ok=True)
            s.cache_dir.mkdir(exist_ok=True)

            assert s.config_dir.exists()
            assert s.models_dir.exists()

    def test_add_and_remove_model(self, tmp_path):
        with patch.object(Settings, "__init__", lambda self: None):
            s = Settings()
            s.config_dir = tmp_path
            s.models_dir = tmp_path / "models"
            s.cache_dir = tmp_path / "cache"
            s.config_file = tmp_path / "config.json"
            s.server = ServerConfig()
            s.models = {}
            s.current_model = None
            s.hf_token = None
            s.models_dir.mkdir()
            s.cache_dir.mkdir()

            config = ModelConfig(name="test", path="/tmp/test", model_type="sd15")
            s.add_model(config)
            assert "test" in s.models

            s.remove_model("test")
            assert "test" not in s.models

    def test_save_and_load_config(self, tmp_path):
        with patch.object(Settings, "__init__", lambda self: None):
            s = Settings()
            s.config_dir = tmp_path
            s.models_dir = tmp_path / "models"
            s.cache_dir = tmp_path / "cache"
            s.config_file = tmp_path / "config.json"
            s.server = ServerConfig()
            s.models = {}
            s.current_model = None
            s.hf_token = None
            s.models_dir.mkdir()
            s.cache_dir.mkdir()

            config = ModelConfig(name="m1", path="/tmp/m1", model_type="flux")
            s.add_model(config)
            s.save_config()

            assert s.config_file.exists()
            data = json.loads(s.config_file.read_text())
            assert "m1" in data["models"]
