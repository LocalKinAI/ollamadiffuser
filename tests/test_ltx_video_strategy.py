"""Tests for the LTX-2 MLX video strategy.

Everything here runs on any machine with no weights and no ltx-2-mlx
installed, which is the point of building the command line in a pure
function: the mistakes that cost a real generation — a frame count off the
``8k + 1`` grid, the wrong subcommand for audio input, a mode flag that does
not exist — are the ones a unit test can catch in a millisecond.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

from ollamadiffuser.core.config.model_registry import ModelRegistry
from ollamadiffuser.core.inference.engine import _get_strategy
from ollamadiffuser.core.inference.strategies import ltx_video_strategy as ltx
from ollamadiffuser.core.inference.strategies.ltx_video_strategy import (
    DEFAULT_PACK,
    LTXVideoMLXStrategy,
    argv_options,
    build_argv,
    find_ltx_binary,
    frames_for_seconds,
    nearest_frame_count,
    valid_frame_count,
)


def _argv(**kwargs) -> list[str]:
    return build_argv("/bin/ltx", "a cat", "/tmp/out.mp4", **kwargs)


# --------------------------------------------------------------------------
# The frame-count grid
# --------------------------------------------------------------------------

class TestFrameCounts:
    @pytest.mark.parametrize("frames,ok", [
        (9, True), (25, True), (97, True), (121, True),
        (8, False), (10, False), (96, False), (100, False), (0, False),
    ])
    def test_valid_frame_count(self, frames, ok):
        assert valid_frame_count(frames) is ok

    @pytest.mark.parametrize("asked,nearest", [
        (1, 9), (10, 9), (12, 9), (100, 97), (120, 121),
        # Halves round up, predictably: 13 and 21 both go up.
        (13, 17), (21, 25),
    ])
    def test_nearest_is_on_the_grid(self, asked, nearest):
        assert nearest_frame_count(asked) == nearest
        assert valid_frame_count(nearest_frame_count(asked))

    def test_seconds_to_frames_at_24fps(self):
        # 4s is the CLI's own default length.
        assert frames_for_seconds(4) == 97
        assert frames_for_seconds(1) == 25
        assert valid_frame_count(frames_for_seconds(7.5))

    def test_build_argv_rejects_off_grid_frames(self):
        with pytest.raises(ValueError, match="8k\\+1"):
            _argv(frames=100)

    def test_the_error_names_the_nearest_legal_count(self):
        with pytest.raises(ValueError, match="97"):
            _argv(frames=100)


# --------------------------------------------------------------------------
# Text-to-video
# --------------------------------------------------------------------------

class TestTextToVideo:
    def test_defaults(self):
        argv = _argv(frames=97)
        assert argv[:2] == ["/bin/ltx", "generate"]
        assert "--prompt" in argv and "a cat" in argv
        assert argv[argv.index("--output") + 1] == "/tmp/out.mp4"
        assert argv[argv.index("--model") + 1] == DEFAULT_PACK
        assert argv[argv.index("--frames") + 1] == "97"
        assert "--two-stage" in argv        # upstream's production default
        assert "--quiet" in argv

    @pytest.mark.parametrize("mode,flag", [
        ("distilled", "--distilled"),
        ("one-stage", "--one-stage"),
        ("two-stage", "--two-stage"),
        ("two-stages-hq", "--two-stages-hq"),
    ])
    def test_every_mode_has_its_flag(self, mode, flag):
        assert flag in _argv(frames=97, mode=mode)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            _argv(frames=97, mode="turbo-max")

    def test_size_and_seed_and_guidance(self):
        argv = _argv(frames=97, width=1280, height=720, seed=7,
                     cfg_scale=3.5, stg_scale=1.0,
                     stage1_steps=30, stage2_steps=3)
        assert argv[argv.index("--width") + 1] == "1280"
        assert argv[argv.index("--height") + 1] == "720"
        assert argv[argv.index("--seed") + 1] == "7"
        assert argv[argv.index("--cfg-scale") + 1] == "3.5"
        assert argv[argv.index("--stg-scale") + 1] == "1.0"
        assert argv[argv.index("--stage1-steps") + 1] == "30"
        assert argv[argv.index("--stage2-steps") + 1] == "3"

    def test_memory_flags(self):
        argv = _argv(frames=97, low_ram=True, tile_frames=8, tile_spatial=2)
        assert "--low-ram" in argv
        assert argv[argv.index("--tile-frames") + 1] == "8"
        assert argv[argv.index("--tile-spatial") + 1] == "2"

    def test_auto_duration_only_without_frames(self):
        # 2.5 packs predict the length; an explicit count wins, as in the CLI.
        assert "--auto-duration" in _argv(auto_duration="2.0:8.0")
        assert "--frames" not in _argv(auto_duration="2.0:8.0")
        with_both = _argv(frames=97, auto_duration="2.0:8.0")
        assert "--auto-duration" not in with_both
        assert with_both[with_both.index("--frames") + 1] == "97"

    def test_enhance_prompt_is_opt_in(self):
        assert "--enhance-prompt" not in _argv(frames=97)
        assert "--enhance-prompt" in _argv(frames=97, enhance_prompt=True)

    def test_frame_rate_is_always_sent(self):
        """The CLI makes --frame-rate mandatory on generate as well as a2v.

        Nothing about a text-to-video call suggests it, and leaving it out
        fails at argparse rather than anywhere informative.
        """
        argv = _argv(frames=97)
        assert argv[argv.index("--frame-rate") + 1] == "24"
        assert _argv(frames=97, frame_rate=30)[
            _argv(frames=97, frame_rate=30).index("--frame-rate") + 1] == "30"


# --------------------------------------------------------------------------
# Image- and audio-conditioned
# --------------------------------------------------------------------------

class TestConditioned:
    def test_image_to_video_stays_on_generate(self):
        argv = _argv(frames=97, image="/tmp/her.png")
        assert argv[1] == "generate"
        assert argv[argv.index("--image") + 1] == "/tmp/her.png"

    def test_audio_switches_subcommand(self):
        argv = _argv(audio="/tmp/voice.wav")
        assert argv[1] == "a2v"
        assert argv[argv.index("--audio") + 1] == "/tmp/voice.wav"
        # The audio is the length, so no frame count is sent unless asked for.
        assert argv[argv.index("--frame-rate") + 1] == "24"
        assert "--frames" not in argv
        assert "--two-stage" not in argv

    def test_audio_takes_no_pipeline_flags(self):
        """a2v accepts neither a mode, nor --steps, nor --enhance-prompt.

        Checked against ltx-2-mlx 0.15.6 itself, which answers "unrecognized
        arguments" to all three — the README reads as though the modes apply
        everywhere, and sending one loses the generation at argparse.
        """
        argv = _argv(audio="/tmp/voice.wav", mode="two-stages-hq",
                     steps=8, enhance_prompt=True)
        assert argv[1] == "a2v"
        for flag in ("--two-stages-hq", "--two-stage", "--distilled",
                     "--one-stage", "--steps", "--enhance-prompt"):
            assert flag not in argv

    def test_audio_with_reference_image_and_start(self):
        argv = _argv(audio="/tmp/voice.wav", image="/tmp/her.png",
                     frames=97, frame_rate=30, audio_start=1.5)
        assert argv[1] == "a2v"
        assert argv[argv.index("--image") + 1] == "/tmp/her.png"
        assert argv[argv.index("--audio-start") + 1] == "1.5"
        assert argv[argv.index("--frame-rate") + 1] == "30"
        assert argv[argv.index("--frames") + 1] == "97"


# --------------------------------------------------------------------------
# Registry options
# --------------------------------------------------------------------------

class TestArgvOptions:
    def test_keeps_known_and_drops_the_rest(self):
        got = argv_options({"mode": "distilled", "frames": 97, "nonsense": True})
        assert got == {"mode": "distilled", "frames": 97}

    def test_translates_the_shared_vocabulary(self):
        # Registry entries speak num_inference_steps / guidance_scale.
        assert argv_options({"num_inference_steps": 8}) == {"steps": 8}
        assert argv_options({"guidance_scale": 3.0}) == {"cfg_scale": 3.0}

    def test_drops_nones(self):
        assert argv_options({"seed": None, "frames": 25}) == {"frames": 25}


# --------------------------------------------------------------------------
# Finding the CLI
# --------------------------------------------------------------------------

class TestBinaryDiscovery:
    def test_explicit_path_wins(self, tmp_path):
        binary = tmp_path / "ltx-2-mlx"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        assert find_ltx_binary(str(binary)) == str(binary)

    def test_env_var(self, tmp_path, monkeypatch):
        binary = tmp_path / "ltx-2-mlx"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setenv("LTX2MLX_BIN", str(binary))
        assert find_ltx_binary() == str(binary)

    def test_none_when_absent(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LTX2MLX_BIN", raising=False)
        monkeypatch.setattr(ltx.shutil, "which", lambda _: None)
        monkeypatch.setattr(ltx.Path, "home", staticmethod(lambda: tmp_path))
        assert find_ltx_binary() is None

    def test_a_directory_is_not_a_binary(self, tmp_path, monkeypatch):
        (tmp_path / "ltx-2-mlx").mkdir()
        monkeypatch.delenv("LTX2MLX_BIN", raising=False)
        monkeypatch.setattr(ltx.shutil, "which", lambda _: None)
        assert find_ltx_binary(str(tmp_path / "ltx-2-mlx")) is None


# --------------------------------------------------------------------------
# The strategy's refusals — every one a sentence the caller can act on
# --------------------------------------------------------------------------

class TestStrategy:
    def _config(self, **params):
        from types import SimpleNamespace
        return SimpleNamespace(
            name="ltx-2.3-mlx-q8", path="/tmp/unused",
            model_type="ltx-video-mlx", variant="mlx-int8",
            repo_id="dgrauet/ltx-2.3-mlx-q8",
            parameters={"ltx_pack": "dgrauet/ltx-2.3-mlx-q8", **params},
        )

    def test_engine_dispatches_the_model_type(self):
        assert isinstance(_get_strategy("ltx-video-mlx"), LTXVideoMLXStrategy)

    def test_refuses_off_apple_silicon(self, monkeypatch):
        monkeypatch.setattr(ltx, "is_apple_silicon", lambda: False)
        assert LTXVideoMLXStrategy().load(self._config(), "cuda") is False

    def test_refuses_without_the_cli(self, monkeypatch):
        monkeypatch.setattr(ltx, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(ltx, "find_ltx_binary", lambda explicit=None: None)
        s = LTXVideoMLXStrategy()
        assert s.load(self._config(), "mps") is False
        assert s.is_loaded is False

    def test_refuses_without_ffmpeg(self, monkeypatch, tmp_path):
        binary = tmp_path / "ltx-2-mlx"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(ltx, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(ltx, "find_ltx_binary", lambda explicit=None: str(binary))
        monkeypatch.setattr(ltx.shutil, "which", lambda name: None)
        assert LTXVideoMLXStrategy().load(self._config(), "mps") is False

    def test_loads_without_touching_weights(self, monkeypatch, tmp_path):
        binary = tmp_path / "ltx-2-mlx"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(ltx, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(ltx, "find_ltx_binary", lambda explicit=None: str(binary))
        monkeypatch.setattr(ltx.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        s = LTXVideoMLXStrategy()
        assert s.load(self._config(mode="distilled"), "mps") is True
        assert s.is_loaded is True
        assert s.pack == "dgrauet/ltx-2.3-mlx-q8"
        s.unload()
        assert s.is_loaded is False

    def test_a_pulled_pack_is_used_from_disk(self, monkeypatch, tmp_path):
        """No second copy: ollamadiffuser pull already has the 21 GB."""
        from types import SimpleNamespace
        pack = tmp_path / "pack"
        pack.mkdir()
        (pack / "config.json").write_text("{}")
        (pack / "transformer-distilled.safetensors").write_bytes(b"\x00")
        binary = tmp_path / "ltx-2-mlx"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(ltx, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(ltx, "find_ltx_binary", lambda explicit=None: str(binary))
        monkeypatch.setattr(ltx.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        config = SimpleNamespace(
            name="ltx-2.3-mlx-q8", path=str(pack), model_type="ltx-video-mlx",
            variant="mlx-int8", repo_id="dgrauet/ltx-2.3-mlx-q8",
            parameters={"ltx_pack": "dgrauet/ltx-2.3-mlx-q8"},
        )
        s = LTXVideoMLXStrategy()
        assert s.load(config, "mps") is True
        assert s.pack == str(pack)

    def test_an_empty_folder_is_not_a_pack(self, tmp_path):
        assert ltx.looks_like_pack(str(tmp_path)) is False
        assert ltx.looks_like_pack(None) is False
        (tmp_path / "config.json").write_text("{}")
        assert ltx.looks_like_pack(str(tmp_path)) is False   # no weights yet

    def test_generate_image_says_it_is_a_video_model(self):
        with pytest.raises(RuntimeError, match="video model"):
            LTXVideoMLXStrategy().generate("a cat")

    def test_generate_video_before_load(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            LTXVideoMLXStrategy().generate_video("a cat")

    def test_registry_defaults_reach_the_argv(self, monkeypatch, tmp_path):
        """A registry entry's mode and size are what a bare call uses."""
        binary = tmp_path / "ltx-2-mlx"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(ltx, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(ltx, "find_ltx_binary", lambda explicit=None: str(binary))
        monkeypatch.setattr(ltx.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        s = LTXVideoMLXStrategy()
        s.load(self._config(mode="distilled", width=1280, height=720, frames=97), "mps")

        seen = {}
        out = tmp_path / "clip.mp4"

        class _Result:
            returncode = 0
            stdout = "done"

        def _fake_run(argv, timeout=None):
            seen["argv"] = list(argv)
            out.write_bytes(b"mp4")
            return _Result()

        monkeypatch.setattr(LTXVideoMLXStrategy, "_run", staticmethod(_fake_run))
        path = s.generate_video("a cat", output=str(out))
        assert Path(path) == out
        argv = seen["argv"]
        assert "--distilled" in argv
        assert argv[argv.index("--width") + 1] == "1280"
        assert argv[argv.index("--frames") + 1] == "97"
        assert argv[argv.index("--model") + 1] == "dgrauet/ltx-2.3-mlx-q8"

    def test_seconds_beats_the_registrys_frame_count(self, monkeypatch, tmp_path):
        """A caller who asks for two seconds gets two seconds.

        The q4 entry defaults to `frames: 97`, and a registry default used to
        land in the same dict the caller's options did — so `seconds=2` was
        dropped and the box spent twice as long making four seconds of video.
        An explicit `frames=` still wins, as it does in the CLI.
        """
        binary = tmp_path / "ltx-2-mlx"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(ltx, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(ltx, "find_ltx_binary", lambda explicit=None: str(binary))
        monkeypatch.setattr(ltx.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        s = LTXVideoMLXStrategy()
        s.load(self._config(mode="distilled", frames=97), "mps")

        seen = {}
        out = tmp_path / "clip.mp4"

        class _Result:
            returncode = 0
            stdout = "done"

        def _fake_run(argv, timeout=None):
            seen["argv"] = list(argv)
            out.write_bytes(b"mp4")
            return _Result()

        monkeypatch.setattr(LTXVideoMLXStrategy, "_run", staticmethod(_fake_run))

        s.generate_video("a cat", output=str(out), seconds=2)
        assert seen["argv"][seen["argv"].index("--frames") + 1] == "49"

        s.generate_video("a cat", output=str(out), seconds=2, frames=97)
        assert seen["argv"][seen["argv"].index("--frames") + 1] == "97"

    def test_failure_carries_the_cli_output(self, monkeypatch, tmp_path):
        binary = tmp_path / "ltx-2-mlx"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setattr(ltx, "is_apple_silicon", lambda: True)
        monkeypatch.setattr(ltx, "find_ltx_binary", lambda explicit=None: str(binary))
        monkeypatch.setattr(ltx.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        s = LTXVideoMLXStrategy()
        s.load(self._config(), "mps")

        class _Result:
            returncode = 1
            stdout = "loading pack\rout of memory"

        monkeypatch.setattr(LTXVideoMLXStrategy, "_run",
                            staticmethod(lambda argv, timeout=None: _Result()))
        with pytest.raises(RuntimeError, match="out of memory"):
            s.generate_video("a cat", output=str(tmp_path / "x.mp4"), frames=97)


# --------------------------------------------------------------------------
# The packs, as registered
# --------------------------------------------------------------------------

class TestRegistryEntries:
    def test_six_packs_are_registered(self):
        reg = ModelRegistry()._registry
        video = {n: c for n, c in reg.items() if c.get("model_type") == "ltx-video-mlx"}
        assert len(video) == 6, f"expected the six LTX-2 packs, got {sorted(video)}"

    def test_every_pack_names_its_weights_and_a_mode(self):
        reg = ModelRegistry()._registry
        for name, cfg in reg.items():
            if cfg.get("model_type") != "ltx-video-mlx":
                continue
            params = cfg.get("parameters") or {}
            assert params.get("ltx_pack"), f"{name} has no ltx_pack"
            assert params.get("mode") in ltx.MODES, f"{name} has a bad mode"

    def test_the_2_5_packs_are_marked_gated(self):
        reg = ModelRegistry()._registry
        for name, cfg in reg.items():
            if cfg.get("model_type") != "ltx-video-mlx":
                continue
            gated = cfg["license_info"]["requires_agreement"]
            assert gated is ("2.5" in name), f"{name}: gated={gated}"

# --------------------------------------------------------------------------
# What a pack actually contains, and what each mode loads out of it
# --------------------------------------------------------------------------

# File names and sizes read off the hub on 2026-09-18, in GiB. Kept here
# because the rules below are about the intersection of an entry's patterns
# with real filenames: a test that invents file names proves nothing, and
# these entries were first written from upstream's README, which is how they
# came to fetch weights their own mode could not use.
_SHARED_23 = {
    "connector.safetensors": 5.91, "vae_decoder.safetensors": 0.76,
    "vae_encoder.safetensors": 0.59, "audio_vae.safetensors": 0.10,
    "vocoder.safetensors": 0.24, "spatial_upscaler_x1_5_v1_0.safetensors": 1.02,
    "spatial_upscaler_x2_v1_1.safetensors": 0.93,
    "temporal_upscaler_x2_v1_0.safetensors": 0.24,
    "split_model.json": 0.0, "embedded_config.json": 0.0, "LICENSE": 0.0,
}
_SHARED_25 = {
    "connector.safetensors": 3.76, "vae_decoder_av.safetensors": 0.78,
    "vae_decoder_conv.safetensors": 0.76, "vae_encoder_conv.safetensors": 0.59,
    "vae_encoder_av.safetensors": 0.59, "audio_vae.safetensors": 0.10,
    "vocoder.safetensors": 0.24, "spatial_upscaler_x2_v1_0.safetensors": 0.93,
    "temporal_upscaler_x2_v1_0.safetensors": 0.24,
    "duration_head.safetensors": 0.01, "text_encoder_config.json": 0.0,
    "tokenizer.json": 0.03, "split_model.json": 0.0, "LICENSE": 0.0,
}

def _pack(transformer_gb, *, family, text_encoder_gb=None, lora=None, lora_gb=0.0):
    files = dict(_SHARED_23 if family == "2.3" else _SHARED_25)
    if family == "2.3":
        for stem in ("transformer-dev", "transformer-distilled",
                     "transformer-distilled-1.1"):
            files[f"{stem}.safetensors"] = transformer_gb
    else:
        for stem in ("transformer-dev", "transformer-distilled"):
            files[f"{stem}.safetensors"] = transformer_gb
        files["text_encoder.safetensors"] = text_encoder_gb
    if lora:
        for name in lora:
            files[name] = lora_gb
    return files

_LORA_23 = ("ltx-2.3-22b-distilled-lora-384.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
_LORA_25 = ("ltx-2.5-22b-distilled-lora-450-bf16.safetensors",)

PACKS = {
    "ltx-2.3-mlx-q4":   _pack(10.54, family="2.3", lora=_LORA_23, lora_gb=7.08),
    "ltx-2.3-mlx-q8":   _pack(19.18, family="2.3", lora=_LORA_23, lora_gb=7.08),
    "ltx-2.3-mlx-bf16": _pack(35.38, family="2.3", lora=_LORA_23, lora_gb=7.08),
    "ltx-2.5-mlx-q4":   _pack(10.54, family="2.5", text_encoder_gb=9.84,
                              lora=_LORA_25, lora_gb=8.29),
    "ltx-2.5-mlx-q8":   _pack(19.18, family="2.5", text_encoder_gb=14.91,
                              lora=_LORA_25, lora_gb=8.29),
    "ltx-2.5-mlx-bf16": _pack(35.38, family="2.5", text_encoder_gb=24.43,
                              lora=_LORA_25, lora_gb=8.29),
}


def _fetched(patterns, files):
    """The files an entry's allow_patterns would pull out of a pack."""
    return {name: size for name, size in files.items()
            if any(fnmatch.fnmatch(name, p) for p in patterns)}


class TestPatternsMatchTheMode:
    """An entry must fetch the weights its own mode loads — no more, no less.

    Read out of ltx-2-mlx 0.15.6's pipelines: stage 1 of two-stage takes
    ``transformer-dev``; stage 2 either streams the pre-fused
    ``transformer-distilled*`` (under ``--low-ram``) or fuses the distilled
    LoRA; ``--distilled`` takes the distilled transformer, preferring the
    versioned file; and a 2.5 pack carries its own ``text_encoder`` plus the
    ``duration_head`` that predicts a clip's length, where a 2.3 pack has
    neither and the CLI downloads Gemma instead.
    """

    def entries(self):
        reg = ModelRegistry()._registry
        out = [(n, c) for n, c in reg.items() if c.get("model_type") == "ltx-video-mlx"]
        assert out, "no LTX entries in the registry"
        return out

    def test_every_pack_is_covered_by_this_test(self):
        assert {n for n, _ in self.entries()} == set(PACKS)

    def test_stage_one_weights(self):
        for name, cfg in self.entries():
            mode = cfg["parameters"]["mode"]
            got = _fetched(cfg["allow_patterns"], PACKS[name])
            if mode == "distilled":
                assert any(f.startswith("transformer-distilled") for f in got), name
            else:
                assert "transformer-dev.safetensors" in got, f"{name} ({mode}) needs the dev transformer"

    def test_stage_two_weights(self):
        for name, cfg in self.entries():
            if cfg["parameters"]["mode"] not in ("two-stage", "two-stages-hq"):
                continue
            got = _fetched(cfg["allow_patterns"], PACKS[name])
            assert any(f.startswith("transformer-distilled") for f in got), name
            assert any("distilled-lora" in f for f in got), \
                f"{name}: stage 2 fuses the distilled LoRA when --low-ram is off"

    def test_the_2_5_packs_bring_their_own_text_encoder(self):
        for name, cfg in self.entries():
            got = _fetched(cfg["allow_patterns"], PACKS[name])
            if "2.5" in name:
                assert "text_encoder.safetensors" in got, f"{name} would fall back to Gemma"
                assert "duration_head.safetensors" in got, f"{name} could not predict a duration"
            else:
                assert "text_encoder.safetensors" not in got, \
                    f"{name}: 2.3 packs have none — the CLI downloads Gemma"

    def test_the_parts_every_pipeline_loads(self):
        for name, cfg in self.entries():
            got = _fetched(cfg["allow_patterns"], PACKS[name])
            assert "connector.safetensors" in got, name
            assert "vocoder.safetensors" in got, name
            assert any("vae" in f for f in got), name
            assert any("upscaler" in f for f in got), name

    def test_disk_space_is_what_the_patterns_would_download(self):
        """The registry's figure is measured, not quoted from a README."""
        for name, cfg in self.entries():
            total = sum(_fetched(cfg["allow_patterns"], PACKS[name]).values())
            stated = cfg["hardware_requirements"]["disk_space_gb"]
            assert total <= stated <= total + 1.5, \
                f"{name}: patterns pull {total:.1f} GB, entry says {stated}"

    def test_nothing_unused_is_fetched(self):
        """A pack holds three transformers; an entry should not take all of them."""
        for name, cfg in self.entries():
            got = _fetched(cfg["allow_patterns"], PACKS[name])
            transformers = [f for f in got if f.startswith("transformer")]
            expected = 1 if cfg["parameters"]["mode"] == "distilled" else 2
            assert len(transformers) == expected, \
                f"{name}: fetches {sorted(transformers)}"
