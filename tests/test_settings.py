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


def _add_models(home, tag, count):
    """In a separate process: add ``count`` models, saving after each, as `pull` does."""
    import os
    os.environ["HOME"] = home
    from ollamadiffuser.core.config.settings import Settings, ModelConfig
    s = Settings()
    for i in range(count):
        s.add_model(ModelConfig(name=f"{tag}-{i}", path=f"/m/{tag}-{i}", model_type="sd15"))


class TestSharedConfigFile:
    """Several processes share ~/.ollamadiffuser/config.json — servers, one-off
    jobs, agents adding an entry — and each holds the copy it read at start."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        return tmp_path

    @staticmethod
    def _file(home):
        return json.loads((home / ".ollamadiffuser" / "config.json").read_text())

    @staticmethod
    def _seed(home, data):
        folder = home / ".ollamadiffuser"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(data))

    def test_another_processs_entry_and_parameter_edit_survive(self, home):
        """What happened on the GPU box: a server started, another session added
        an entry and changed the Turbo's parameters, and the server's later
        saves (current model set at load, cleared at shutdown) would have
        written its old copy over them."""
        self._seed(home, {"models": {"turbo": {"name": "turbo", "path": "/m/turbo", "model_type": "mlx",
                                               "parameters": {"quantize": 8, "num_inference_steps": 8}}},
                          "current_model": None})
        server, agent = Settings(), Settings()
        agent.models["turbo"].parameters = {"quantize": None, "compute_precision": "float16",
                                            "num_inference_steps": 8}
        agent.add_model(ModelConfig(name="turbo-edit", path="/m/turbo", model_type="mlx",
                                    parameters={"mlx_variant": "qwen21-edit"}))
        server.set_current_model("turbo")
        server.current_model = None
        server.save_config()

        data = self._file(home)
        assert data["models"]["turbo"]["parameters"] == {"quantize": None, "compute_precision": "float16",
                                                          "num_inference_steps": 8}
        assert data["models"]["turbo-edit"]["parameters"] == {"mlx_variant": "qwen21-edit"}
        assert data["current_model"] is None
        assert "turbo-edit" in server.models  # and the server now knows of it

    def test_one_process_changing_a_parameter_keeps_anothers_change_to_another(self, home):
        self._seed(home, {"models": {"m": {"name": "m", "path": "/m", "model_type": "mlx",
                                           "parameters": {"steps": 8, "guidance": 1.0}}}})
        a, b = Settings(), Settings()
        a.models["m"].parameters = {"steps": 6, "guidance": 1.0}
        a.save_config()
        b.models["m"].parameters = {"steps": 8, "guidance": 3.5}
        b.save_config()
        assert self._file(home)["models"]["m"]["parameters"] == {"steps": 6, "guidance": 3.5}

    def test_additions_and_removals_from_different_processes_all_land(self, home):
        self._seed(home, {"models": {"old": {"name": "old", "path": "/m/old", "model_type": "sd15"}}})
        a, b = Settings(), Settings()
        a.add_model(ModelConfig(name="a", path="/m/a", model_type="sd15"))
        b.remove_model("old")
        b.add_model(ModelConfig(name="b", path="/m/b", model_type="sd15"))
        a.save_config()
        assert set(self._file(home)["models"]) == {"a", "b"}

    def test_a_value_changed_and_changed_back_is_written(self, home):
        a, b = Settings(), Settings()
        a.add_model(ModelConfig(name="x", path="/m/x", model_type="sd15"))
        b.load_config()
        a.set_current_model("x")
        a.current_model = None
        a.save_config()
        assert self._file(home)["current_model"] is None

    def test_what_this_version_does_not_read_stays_in_the_file(self, home):
        self._seed(home, {"models": {
            "ok": {"name": "ok", "path": "/m/ok", "model_type": "sd15", "note": "added by hand"},
            "odd": {"name": "odd", "model_type": "sd15"},  # no path: not loadable here
        }, "added_by_a_newer_version": {"x": 1}})
        s = Settings()
        assert set(s.models) == {"ok"}
        s.add_model(ModelConfig(name="new", path="/m/new", model_type="sd15"))
        data = self._file(home)
        assert data["models"]["ok"]["note"] == "added by hand"
        assert data["models"]["odd"] == {"name": "odd", "model_type": "sd15"}
        assert data["added_by_a_newer_version"] == {"x": 1}

    def test_a_failed_write_leaves_the_old_file_whole(self, home, monkeypatch):
        s = Settings()
        s.add_model(ModelConfig(name="x", path="/m/x", model_type="sd15"))
        before = (home / ".ollamadiffuser" / "config.json").read_text()

        def broken_dump(data, f, **kwargs):
            f.write('{"half": ')
            raise OSError("disk full")

        monkeypatch.setattr("ollamadiffuser.core.config.settings.json.dump", broken_dump)
        s.add_model(ModelConfig(name="y", path="/m/y", model_type="sd15"))
        assert (home / ".ollamadiffuser" / "config.json").read_text() == before
        assert not list((home / ".ollamadiffuser").glob(".config-*.json"))

    def test_a_broken_file_is_kept_aside_not_lost(self, home):
        self._seed(home, {})
        (home / ".ollamadiffuser" / "config.json").write_text('{"models": {"x": ')
        s = Settings()  # cannot read it; says so and starts empty
        s.add_model(ModelConfig(name="y", path="/m/y", model_type="sd15"))
        assert set(self._file(home)["models"]) == {"y"}
        (kept,) = (home / ".ollamadiffuser").glob("config.json.broken-*")
        assert kept.read_text() == '{"models": {"x": '

    def test_processes_saving_at_once_take_turns(self, home):
        import multiprocessing
        self._seed(home, {})
        ctx = multiprocessing.get_context("spawn")
        workers = [ctx.Process(target=_add_models, args=(str(home), tag, 15)) for tag in ("a", "b", "c")]
        for w in workers:
            w.start()
        for w in workers:
            w.join(120)
            assert w.exitcode == 0
        models = self._file(home)["models"]
        assert len(models) == 45 and {f"{t}-{i}" for t in "abc" for i in range(15)} == set(models)
