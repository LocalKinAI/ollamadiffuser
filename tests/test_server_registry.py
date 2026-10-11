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
    monkeypatch.setattr(server_registry, "ask", lambda host, port, timeout=2.0: None)
    result = CliRunner().invoke(model_commands.ps, [])
    assert "No model server is running" in result.output


def _answer(model):
    return {"loaded": True, "model": model, "info": {"type": "mlx", "device": "mps"}}


def test_discover_finds_servers_that_left_no_record(home, monkeypatch):
    """A server started with another config directory — an agent's sandbox —
    or by an older version writes no record here, but still answers."""
    server_registry.register("0.0.0.0", 8000, "z-image-turbo-mlx")
    answers = {
        8000: _answer("z-image-turbo-mlx"),
        8010: _answer("qwen-turbo-one"),       # no record: found by probing
        8015: {"status": "something else"},    # another program on the port
    }
    monkeypatch.setattr(server_registry, "ask", lambda host, port, timeout=2.0: answers.get(port))
    found = {s["port"]: s for s in server_registry.discover()}
    assert sorted(found) == [8000, 8010]
    assert found[8000]["pid"] == os.getpid() and found[8000]["url"] == "http://127.0.0.1:8000"
    assert found[8010]["pid"] is None and found[8010]["model"] == "qwen-turbo-one"
    assert found[8010]["type"] == "mlx" and found[8010]["loaded"]


def test_discover_reports_a_recorded_server_that_stopped_answering(home, monkeypatch):
    server_registry.register("127.0.0.1", 8003, "z-image-turbo-mlx")
    monkeypatch.setattr(server_registry, "ask", lambda host, port, timeout=2.0: None)
    (only,) = server_registry.discover()
    assert only["port"] == 8003 and not only["answering"] and only["model"] is None


def test_ps_json_is_the_discovered_list(home, monkeypatch):
    from ollamadiffuser.cli import model_commands
    monkeypatch.setattr(server_registry, "ask", lambda host, port, timeout=2.0:
                        _answer("ltx-2.3-mlx-q8") if port == 8012 else None)
    result = CliRunner().invoke(model_commands.ps, ["--json"])
    assert result.exit_code == 0, result.output
    (only,) = json.loads(result.output)
    assert only["port"] == 8012 and only["model"] == "ltx-2.3-mlx-q8"


def test_stop_port_and_all(home, monkeypatch):
    from ollamadiffuser.cli import model_commands
    monkeypatch.setattr(server_registry, "ask", lambda host, port, timeout=2.0:
                        _answer("m") if port in (8010, 8012) else None)
    posted = []

    class Ok:
        status_code = 200

    monkeypatch.setattr("requests.post", lambda url, timeout=None: posted.append(url) or Ok())
    result = CliRunner().invoke(model_commands.stop, ["--port", "8012"])
    assert result.exit_code == 0 and posted == ["http://127.0.0.1:8012/api/shutdown"]
    posted.clear()
    CliRunner().invoke(model_commands.stop, ["--all"])
    assert sorted(posted) == ["http://127.0.0.1:8010/api/shutdown", "http://127.0.0.1:8012/api/shutdown"]
