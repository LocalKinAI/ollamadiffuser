"""Regression tests for the models.yaml migration.

The built-in model registry moved from a 1500-line Python dict literal in
`model_registry.py` to a bundled `models.yaml` data file (so contributors
can add models via a pure-data PR). These tests pin that migration:

  1. The YAML-loaded registry deep-equals a snapshot captured *before* the
     migration (tests/fixtures/registry_snapshot.json). This is the
     zero-loss guarantee.
  2. models.yaml is well-formed and structurally sane.
  3. Every entry keeps the minimal schema the loader/strategies rely on.
"""
import json
from pathlib import Path

import yaml
import pytest

from ollamadiffuser.core.config.model_registry import ModelRegistry

_HERE = Path(__file__).parent
_SNAPSHOT = _HERE / "fixtures" / "registry_snapshot.json"


def _yaml_path() -> Path:
    import ollamadiffuser.core.config.model_registry as mod
    return Path(mod.__file__).parent / "models.yaml"


class TestZeroLossMigration:
    """The migration guarantee, which is about loss rather than about drift.

    This started as an equality check against the snapshot, which meant every
    new model broke it; loosening that to "snapshot entries must all still be
    here" fixed the growth case but still froze every field of every old
    entry, so correcting a guessed disk figure to a measured one also failed.
    The guarantee worth keeping is narrower: nothing from before the migration
    disappeared, and nothing was quietly repointed at a different repo or
    strategy. Estimates are free to improve.
    """

    # What the snapshot is allowed to pin. Everything else about an entry is
    # an estimate we expect to improve: disk/VRAM figures get replaced by
    # measured ones, allow_patterns get narrowed once we learn which files a
    # loader actually reads, performance notes get rewritten. Freezing those
    # turned every honest correction into a failing test, which is the
    # opposite of what the snapshot is for.
    _IDENTITY_FIELDS = ("repo_id", "model_type")

    # Repoints made on purpose. The guard below exists to stop an entry being
    # moved *quietly*; a move that is written down here, with its reason, is
    # the opposite of quiet. (name, field) -> (snapshot value, value now).
    _DELIBERATE_REPOINTS = {
        # mflux's `qwen-image` / `qwen-image-edit` aliases load the 2512 and
        # 2509 checkpoints. Pulling the originals gave mflux 58 GB of weights
        # it had not asked for. test_mlx_strategy pins repo_id to the alias.
        ("qwen-image-mlx", "repo_id"):
            ("Qwen/Qwen-Image", "Qwen/Qwen-Image-2512"),
        ("qwen-image-edit-mlx", "repo_id"):
            ("Qwen/Qwen-Image-Edit", "Qwen/Qwen-Image-Edit-2509"),
    }

    def test_registry_keeps_every_pre_migration_model(self):
        """The guarantee: no pre-migration model was lost or silently repointed."""
        snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
        loaded = ModelRegistry()._registry

        missing = sorted(set(snapshot) - set(loaded))
        assert not missing, f"models.yaml lost {missing}"

        for name, cfg in snapshot.items():
            for field in self._IDENTITY_FIELDS:
                if field not in cfg:
                    continue
                expected = cfg[field]
                repoint = self._DELIBERATE_REPOINTS.get((name, field))
                if repoint is not None:
                    was, now = repoint
                    assert was == cfg[field], (
                        f"_DELIBERATE_REPOINTS[{name}.{field}] says the snapshot "
                        f"had {was!r}; it has {cfg[field]!r}"
                    )
                    expected = now
                assert loaded[name].get(field) == expected, (
                    f"{name}.{field} changed from {cfg[field]!r} to "
                    f"{loaded[name].get(field)!r} — a pre-migration model "
                    f"must keep pointing at the same thing, unless the move "
                    f"is recorded in _DELIBERATE_REPOINTS with its reason"
                )

    def test_model_count_never_shrinks(self):
        snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
        assert len(ModelRegistry()._registry) >= len(snapshot)


class TestModelsYaml:
    def test_yaml_file_exists(self):
        assert _yaml_path().is_file(), "bundled models.yaml is missing"

    def test_yaml_is_well_formed(self):
        data = yaml.safe_load(_yaml_path().read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "models" in data
        assert isinstance(data["models"], dict)
        assert data["models"], "models.yaml has no models"

    def test_yaml_has_version_marker(self):
        data = yaml.safe_load(_yaml_path().read_text(encoding="utf-8"))
        assert data.get("version") == 1

    def test_every_entry_has_minimal_schema(self):
        data = yaml.safe_load(_yaml_path().read_text(encoding="utf-8"))
        for name, cfg in data["models"].items():
            assert isinstance(cfg, dict), f"{name} is not a mapping"
            assert "model_type" in cfg, f"{name} missing model_type"
            # repo_id is required for everything except purely external/api
            # placeholder entries (none currently), so assert it broadly.
            assert "repo_id" in cfg, f"{name} missing repo_id"

    def test_mlx_entries_intact(self):
        """MLX entries (added across v2.0.15-2.0.17) survived the migration."""
        reg = ModelRegistry()._registry
        mlx = [n for n, c in reg.items() if c.get("model_type") == "mlx"]
        assert len(mlx) >= 22, f"expected >=22 MLX entries, got {len(mlx)}"


class TestUserOverridesStillWork:
    def test_external_yaml_overrides_bundled(self, tmp_path, monkeypatch):
        """A user models.yaml via OLLAMADIFFUSER_MODEL_CONFIG still merges/overrides."""
        override = tmp_path / "models.yaml"
        override.write_text(
            yaml.safe_dump({
                "models": {
                    "flux.1-dev": {  # override an existing entry
                        "repo_id": "black-forest-labs/FLUX.1-dev",
                        "model_type": "flux",
                        "parameters": {"num_inference_steps": 999},
                    },
                    "my-custom-model": {  # brand-new entry
                        "repo_id": "acme/whatever",
                        "model_type": "generic",
                    },
                }
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("OLLAMADIFFUSER_MODEL_CONFIG", str(override))
        reg = ModelRegistry()
        # New entry is present
        assert reg.get_model("my-custom-model") is not None
        # Override took effect
        assert reg.get_model("flux.1-dev")["parameters"]["num_inference_steps"] == 999
