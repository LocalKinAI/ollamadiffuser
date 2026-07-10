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
    def test_registry_deep_equals_pre_migration_snapshot(self):
        """The single most important test: no model was lost or mangled."""
        snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
        loaded = ModelRegistry()._registry
        assert loaded == snapshot, "models.yaml drifted from the pinned snapshot"

    def test_model_count_matches_snapshot(self):
        snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
        assert len(ModelRegistry()._registry) == len(snapshot)


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
        assert len(mlx) >= 14, f"expected >=14 MLX entries, got {len(mlx)}"


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
