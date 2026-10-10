"""`ps` sees every running server, not only the last one started."""
import json
import os

import pytest
from click.testing import CliRunner

from ollamadiffuser.core.utils import server_registry


@pytest.fixture
def home(tmp_path, monkeypatch):
    from ollamadiffuser.core.config.settings import settings
    monkeypatch.setattr(settings, "config_dir", tmp_path)
    return tmp_path


def test_register_lists_and_unregister(home):
    server_registry.register("127.0.0.1", 8000, "z-image-turbo-mlx")
    server_registry.register("127.0.0.1", 8011, None)
    found = server_registry.running()
    assert [r["port"] for r in found] == [8000, 8011]
    assert found[0]["pid"] == os.getpid() and found[0]["model"] == "z-image-turbo-mlx"
    server_registry.unregister(8000)
    assert [r["port"] for r in server_registry.running()] == [8011]


def test_records_of_dead_processes_are_dropped(home):
    folder = server_registry.servers_dir()
    folder.mkdir(parents=True)
    (folder / "8005.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "host": "h", "port": 8005}))
    assert server_registry.running() == []
    assert not (folder / "8005.json").exists()


def test_unregister_leaves_another_process_record(home):
    folder = server_registry.servers_dir()
    folder.mkdir(parents=True)
    other = {"pid": 1, "host": "h", "port": 8002}          # pid 1 is always alive
    (folder / "8002.json").write_text(json.dumps(other))
    server_registry.unregister(8002)
    assert (folder / "8002.json").exists()


def test_ps_shows_each_server_and_what_it_holds(home, monkeypatch):
    from ollamadiffuser.cli import model_commands
    server_registry.register("127.0.0.1", 8000, "qwen-image-2.1-mlx")
    server_registry.register("127.0.0.1", 8012, "ltx-2.3-mlx-q8")
    answers = {
        8000: {"loaded": True, "model": "qwen-image-2.1-mlx", "info": {"type": "mlx", "device": "mps"}},
        8012: {"loaded": True, "model": "ltx-2.3-mlx-q8", "info": {"type": "ltx-video-mlx", "device": "mps"}},
    }
    monkeypatch.setattr(server_registry, "ask", lambda host, port, timeout=2.0: answers.get(port))
    monkeypatch.setattr(model_commands.model_manager, "is_server_running", lambda: True)
    monkeypatch.setattr(model_commands.settings.server, "port", 8000)
    result = CliRunner().invoke(model_commands.ps, [], terminal_width=200)
    assert result.exit_code == 0, result.output
    assert "qwen-image-2.1-mlx" in result.output and "ltx-2.3-mlx-q8" in result.output
    assert "2 running servers" in result.output


def test_ps_with_nothing_running(home, monkeypatch):
    from ollamadiffuser.cli import model_commands
    monkeypatch.setattr(model_commands.model_manager, "is_server_running", lambda: False)
    result = CliRunner().invoke(model_commands.ps, [])
    assert "No model server is running" in result.output
