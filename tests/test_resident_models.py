"""Several models in memory at once (models/manager.py, `serve --max-loaded`)."""
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def kept(box):
    box.manager.max_loaded = 3
    return box


def test_the_current_model_is_the_last_used_and_the_rest_wait(kept):
    m = kept.manager
    for name in ("a", "b", "c"):
        assert m.load_model(name)
    assert m.resident_models() == ["c", "b", "a"] and m.current_model_name == "c"
    assert m.load_model("a") and kept.loads() == ["a", "b", "c"]
    assert m.resident_models() == ["a", "c", "b"] and m.engine_of("b") is kept.live["b"]


def test_a_named_unload_hands_current_to_the_last_used_of_the_rest(kept):
    from ollamadiffuser.core.config.settings import settings
    m = kept.manager
    for name in ("a", "b", "c"):
        m.load_model(name)
    m.unload_model(name="b")
    assert m.resident_models() == ["c", "a"] and settings.current_model == "c"
    m.unload_model(name="c")
    assert m.resident_models() == ["a"] and settings.current_model == "a"
    m.unload_model()
    assert m.resident_models() == [] and settings.current_model is None
    assert sorted(kept.unloads()) == ["a", "b", "c"]


def test_put_back_keeps_the_old_order(kept):
    m = kept.manager
    for name in ("a", "b", "c"):
        m.load_model(name)
    m.load_model("a")                                     # a, c, b — "b" the least recently used
    kept.other_use = 30.0                                 # 26 GB free, 66 with all three gone
    kept.models["big"].update(needs=60, fails=True)       # so all three go, and then it fails
    assert m.load_model("big") is False and "failed to load" in m.last_error
    assert kept.unloads() == ["b", "c", "a"]
    assert m.resident_models() == ["a", "c", "b"]


def test_a_job_alone_makes_no_room(kept):
    kept.manager.load_model("a")
    kept.other_use = 95.0
    assert kept.manager.make_room("a") == [] and kept.unloads() == []


def test_removing_a_model_in_memory_unloads_it_first(kept, monkeypatch, tmp_path):
    from ollamadiffuser.core.config.settings import settings
    m = kept.manager
    m.load_model("a")
    m.load_model("b")
    (tmp_path / "a-files").mkdir()
    settings.models["a"].path = str(tmp_path / "a-files")
    monkeypatch.setattr(settings, "remove_model", lambda name: settings.models.pop(name))
    assert m.remove_model("a")
    assert kept.unloads() == ["a"] and m.resident_models() == ["b"]
    assert not (tmp_path / "a-files").exists()


def test_one_process_holds_one_gguf_model(kept, monkeypatch):
    from ollamadiffuser.core.config.settings import ModelConfig, settings
    from ollamadiffuser.core.models import manager as manager_module
    loader = MagicMock()
    loader.load_model.return_value = True
    monkeypatch.setattr(manager_module, "gguf_loader", loader)
    monkeypatch.setattr(manager_module, "GGUF_AVAILABLE", True)
    for name in ("g1-gguf", "g2-gguf"):
        kept.models[name] = {"holds": 5, "needs": 6}
        settings.models[name] = ModelConfig(name=name, path="/g", model_type="flux", variant="gguf")
    m = kept.manager
    monkeypatch.setattr(m, "is_gguf_model", lambda name: name.endswith("-gguf"))
    assert m.load_model("g1-gguf") and m.load_model("a") and m.load_model("g2-gguf")
    assert loader.unload_model.call_count == 1
    assert m.resident_models() == ["g2-gguf", "a"]
